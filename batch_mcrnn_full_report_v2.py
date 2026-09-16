#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TURNLAB · Batch-MCRNN 完整分析报告（v4 · 拟合度优先 / 早停 / 9 θ 候选图例）
======================================================================
相对 v3 的核心改动：
  1. 弃用验证集：原 train + val 合并为新 train，test 不变 → 比例 5.5y : 1y
  2. 选参从"验证集总收益" → "训练集 loss 拟合度"（早停 patience=10, min_epochs=20）
  3. K 线图例：9 个 θ 候选（0.1~0.9，步长 0.1）同色系渐变（浅红→深红），单选模式
  4. 删去所有交易模块（backtest_long / RSI+CR(DC) 双重过滤 / 网格搜索选参 等）
  5. 保留 v3 全部：3 处残差 / unweighted BCE / 单主路损失

用法:
  python batch_mcrnn_full_report_v2.py <xlsx_dir> [out.html]
"""
import os, sys, json, glob, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYMBOL_FILES = {
    "SR.CZC": ["SR.CZC (1).xlsx", "SR.CZC.xlsx"],
    "SB.NYB": ["SB.NYB (1).xlsx", "SB.NYB.xlsx"],
    "Y.DCE":  ["Y.DCE (1).xlsx", "Y.DCE.xlsx"],
    "A.DCE":  ["A.DCE (1).xlsx", "A.DCE.xlsx"],
    "B.DCE":  ["B.DCE (1).xlsx", "B.DCE.xlsx"],
    "OI.CZC": ["OI.CZC (1).xlsx", "OI.CZC.xlsx"],
    "BO.CBT": ["BO.CBT (1).xlsx", "BO.CBT.xlsx"],
    "P.DCE":  ["P.DCE (1).xlsx", "P.DCE.xlsx"],
    "M.DCE":  ["M.DCE (1).xlsx", "M.DCE.xlsx"],
    "RM.CZC": ["RM.CZC (1).xlsx", "RM.CZC.xlsx"],
    "SM.CBT": ["SM.CBT (1).xlsx", "SM.CBT.xlsx"],
    "S.CBT":  ["S.CBT (1).xlsx", "S.CBT.xlsx"],
}
SYMBOL_INFO = {
    "SR.CZC": ("郑商所白糖",   "CZCE"),
    "SB.NYB": ("ICE 11号糖",   "ICE"),
    "Y.DCE":  ("大商所豆油",   "DCE"),
    "A.DCE":  ("大商所豆一",   "DCE"),
    "B.DCE":  ("大商所豆二",   "DCE"),
    "OI.CZC": ("郑商所菜油",   "CZCE"),
    "BO.CBT": ("CBOT 豆油",    "CBOT"),
    "P.DCE":  ("大商所棕榈油", "DCE"),
    "M.DCE":  ("大商所豆粕",   "DCE"),
    "RM.CZC": ("郑商所菜粕",   "CZCE"),
    "SM.CBT": ("CBOT 豆粕",    "CBOT"),
    "S.CBT":  ("CBOT 大豆",    "CBOT"),
}
SYMBOLS = list(SYMBOL_FILES.keys())
N_SYM = len(SYMBOLS)

FEATURE_COLS = ["open", "high", "low", "close", "volume", "oi",
                "ma5", "ma10", "ma20", "rsi", "roc5", "roc10",
                "mom5", "amp", "vol_chg", "itk"]
N_FEAT = len(FEATURE_COLS)

SEQ = 24                 # 序列长度 L
CNN_CHANNELS = 6         # 每尺度通道数 M
LSTM_HIDDEN = 48
LSTM_LAYERS = 2
ATTN_DIM = 48
BATCH_ATTN_HEADS = 4
BATCH_ATTN_LAYERS = 1
DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 256
# v3 改动 1: 去掉加权 BCE（标准 nn.BCEWithLogitsLoss() 不传 pos_weight）→ v4 保留
POS_WEIGHT = 1.0

# ★ 训练控制（v4 · 拟合度优先 / 早停）
# 弃用验证集；改为在 train 集上监控 loss 拟合度，patience 后未突破历史最低即停。
# 最大 epoch 设为 200（用户指定），早停触发后会大幅少于 200。
EPOCHS = 300                 # v4: 上限 300（用户指定）
EARLY_STOP_PATIENCE = 15     # v4: 连续 15 个 epoch train loss 未刷新历史最低即停
EARLY_STOP_MIN_EPOCHS = 20   # v4: 至少训 20 个 epoch 才允许触发早停（避免噪声早停）

# ★ K 线图例的 9 个 θ 候选（v4 新增）
#   K 线图上每个候选都作为可选 series，默认全不显示，用户可单选其中一个
#   颜色：同色系渐变，θ=0.1 浅红 → θ=0.9 深红
#   同一个 θ 内同时画顶（▲）和底（▼）—— 合并为 "预测顶/底（θ=0.X）" 一个图例名
THETA_CANDIDATES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]  # 9 个

# ★ 顶/底判定（v4 简化：仅用 6 日窗内极值，不再做 RSI+CR 双重过滤）
PEAK_WINDOW = 6

# ------------------------------------------------------------
# 数据
# ------------------------------------------------------------
def find_file(symbol, directory):
    for name in SYMBOL_FILES[symbol]:
        p = os.path.join(directory, name)
        if os.path.exists(p):
            return p
    matches = sorted(glob.glob(os.path.join(directory, f"{symbol}*.xlsx")))
    if matches:
        return _pick_most_complete(matches)
    matches = sorted(glob.glob(os.path.join(directory, f"{symbol}*.csv")))
    return matches[0] if matches else None


def _pick_most_complete(paths):
    """从多个同品种 xlsx/csv 中挑"数据最完整"的那份。

    评判标准：取每份的最近 30 个交易日，统计 OHLC 关系"合理"的行数
    （H >= max(O,C) 且 L <= min(O,C) 且 H > L 视为"合理"），取合理行数最多的。
    行数相同则取文件最大的。
    """
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    candidates = []
    for p in paths:
        try:
            if p.lower().endswith(".csv"):
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)
            df.columns = [str(c).strip() for c in df.columns]
            # 找日期/open/high/low/close 列
            col_map = {
                "日期": "date", "Date": "date",
                "开盘价(元)": "open", "Open": "open",
                "最高价(元)": "high", "High": "high",
                "最低价(元)": "low", "Low": "low",
                "收盘价(元)": "close", "Close": "close",
            }
            df = df.rename(columns=col_map)
            for c in ("open", "high", "low", "close", "date"):
                if c not in df.columns:
                    df = None
                    break
            if df is None:
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for c in ("open", "high", "low", "close"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
            if len(df) == 0:
                continue
            # 取最近 30 个交易日
            sub = df.tail(30)
            ok = ((sub["high"] >= sub[["open", "close"]].max(axis=1) - 1e-6) &
                  (sub["low"]  <= sub[["open", "close"]].min(axis=1) + 1e-6) &
                  (sub["high"] >  sub["low"])).sum()
            # v4 增量: 排序键改为 mtime 最新优先（防御性，避免选旧快照）
            #   (mtime_desc, 行数 desc, 文件大小 desc) —— mtime 是关键
            candidates.append((p, int(ok), len(sub), os.path.getsize(p), os.path.getmtime(p)))
        except Exception:
            continue
    if not candidates:
        return paths[0]
    # v4 增量: 按 (mtime desc, 合理行数 desc, 行数 desc, 大小 desc) 排序取第一
    candidates.sort(key=lambda x: (x[4], x[1], x[2], x[3]), reverse=True)
    return candidates[0][0]


def load_futures(path, symbol):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {
        "代码": "code", "名称": "name", "日期": "date",
        "开盘价(元)": "open", "最高价(元)": "high", "最低价(元)": "low",
        "收盘价(元)": "close", "结算价": "settle", "涨跌幅": "pct_chg",
        "成交额(百万)": "amount", "成交量": "volume", "持仓量": "oi",
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume", "OI": "oi",
    }
    df = df.rename(columns=col_map)
    df = df.dropna(subset=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 1.0
    if "oi" not in df.columns:
        df["oi"] = 1.0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1.0)
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(1.0)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0].reset_index(drop=True)
    df["symbol"] = symbol
    return df


def add_tech(df):
    df = df.copy()
    c = df["close"]
    df["ma5"] = c.rolling(5).mean()
    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()
    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / (l + 1e-12))
    df["roc5"] = c.pct_change(5)
    df["roc10"] = c.pct_change(10)
    df["mom5"] = c / c.shift(5) - 1
    df["amp"] = (df["high"] - df["low"]) / (df["low"] + 1e-8)
    df["vol_chg"] = df["volume"].pct_change()
    df["itk"] = np.where(df["close"] > df["open"], 1.0, -1.0)
    df = df.replace([np.inf, -np.inf], np.nan).bfill().fillna(0)
    return df


# ------------------------------------------------------------
# RA-PLR
# ------------------------------------------------------------
def top_down_plr(prices, max_error):
    n = len(prices)
    if n < 3:
        return [0, n - 1]
    def max_dev(s, e):
        if e - s < 2:
            return 0.0, s
        x = np.arange(e - s + 1, dtype=float)
        y = prices[s:e + 1]
        slope = (y[-1] - y[0]) / (x[-1] - x[0] + 1e-12)
        approx = y[0] + slope * x
        dev = np.abs(y - approx)
        i = int(np.argmax(dev))
        return float(dev[i]), s + i
    segs = [(0, n - 1)]
    while True:
        md, bs, bp = 0.0, None, None
        for i, (s, e) in enumerate(segs):
            d, p = max_dev(s, e)
            if d > md:
                md, bs, bp = d, i, p
        if md <= max_error or bp is None:
            break
        s, e = segs[bs]
        segs = segs[:bs] + [(s, bp), (bp, e)] + segs[bs + 1:]
        if len(segs) > n // 2:
            break
    return sorted(set([s for s, e in segs] + [e for s, e in segs]))


def ra_plr(prices, alpha=0.015, beta=12):
    pr = float(prices.max() - prices.min())
    cands = [r * pr for r in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08)]
    best, bp, bd = -np.inf, [0, len(prices) - 1], cands[0]
    for delta in cands:
        pts = top_down_plr(prices, delta)
        if len(pts) < 3:
            continue
        ret = sum(abs(prices[pts[i + 1]] - prices[pts[i]]) / (prices[pts[i]] + 1e-8)
                  for i in range(len(pts) - 1))
        pen = sum(max(beta - (pts[i] - pts[i - 1]), 0) for i in range(1, len(pts)))
        sc = ret - alpha * pen
        if sc > best:
            best, bp, bd = sc, pts, delta
    return bp, bd, best


def non_turn(pts, prices, min_opp=0.008):
    if len(pts) < 3:
        return pts
    cl = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = prices[cl[-1]], prices[pts[i]], prices[pts[i + 1]]
        if np.sign(b - a) * np.sign(c - b) < 0 and abs(c - b) / (a + 1e-8) > min_opp:
            cl.append(pts[i])
    cl.append(pts[-1])
    return cl


def aug_lab(n, pts, prices, tau=0.012, n_low=2, n_high=4):
    lab = np.zeros(n, dtype=np.int32)
    for p in pts:
        L, R = max(0, p - 5), min(n, p + 6)
        vol = np.std(prices[L:R]) / (np.mean(prices[L:R]) + 1e-8)
        w = n_low if vol > tau else n_high
        lab[max(0, p - w):min(n, p + w + 1)] = 1
    return lab


def seg_fit(pts, prices, n):
    fit = np.full(n, np.nan)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ya, yb = prices[a], prices[b]
        span = max(1, b - a)
        for j in range(a, b + 1):
            fit[j] = ya + (yb - ya) * (j - a) / span
    return fit


# ------------------------------------------------------------
# Batch-MCRNN
# ------------------------------------------------------------
class MultiScaleCNN(nn.Module):
    def __init__(self, in_ch, channels=CNN_CHANNELS, kernels=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernels:
            pad = k // 2
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_ch, channels, kernel_size=k, padding=pad),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            ))
        self.out_ch = channels * len(kernels)

    def forward(self, x):
        x = x.transpose(1, 2)
        outs = [conv(x) for conv in self.convs]
        return torch.cat(outs, dim=1).transpose(1, 2)


class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.Wa = nn.Linear(hidden, hidden)
        self.ua = nn.Linear(hidden, 1, bias=False)
        self.ba = nn.Parameter(torch.zeros(hidden))

    def forward(self, h):
        score = self.ua(torch.tanh(self.Wa(h) + self.ba)).squeeze(-1)
        alpha = torch.softmax(score, dim=1)
        a = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return a, alpha


class BatchAttention(nn.Module):
    def __init__(self, dim, n_heads=4, n_layers=1, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, e):
        x = e.unsqueeze(0)
        y = self.encoder(x).squeeze(0)
        return self.norm(y + e)


class BatchMCRNN(nn.Module):
    def __init__(self, n_feat=N_FEAT, n_sym=N_SYM, seq=SEQ,
                 cnn_ch=CNN_CHANNELS, lstm_h=LSTM_HIDDEN,
                 n_lstm=LSTM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.ms_cnn = MultiScaleCNN(n_feat, cnn_ch, kernels=(3, 5, 7))
        self.cnn_out_ch = self.ms_cnn.out_ch  # = cnn_ch * 3
        # v3 改动 3a: CNN→Embedding 残差：1×1 Conv 把 x 投影到 cnn 维度
        self.input_proj = nn.Linear(n_feat, self.cnn_out_ch)
        # 残差相加后 cat 符号嵌入
        fuse_in = self.cnn_out_ch + 8  # (cnn + x_proj) cat sym_emb
        self.sym_emb = nn.Embedding(n_sym, 8)
        self.embed = nn.Sequential(
            nn.Linear(fuse_in, lstm_h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # v3 改动 3b: LSTM→Attention 残差（同时把维度对齐到 lstm_h）
        # lstm 输出 (B, T, lstm_h) → 平均池化到 (B, lstm_h)，与 attention 输出维度一致
        self.lstm = nn.LSTM(
            input_size=lstm_h, hidden_size=lstm_h, num_layers=n_lstm,
            batch_first=True, dropout=dropout if n_lstm > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(lstm_h)
        self.temp_attn = TemporalAttention(lstm_h)
        # v3 改动 3c: 删掉 self.proj（不再用 Linear(2h, h)）
        #   改为加法残差：e = a + h[:, -1, :]
        self.batch_attn = BatchAttention(lstm_h, n_heads=BATCH_ATTN_HEADS,
                                         n_layers=BATCH_ATTN_LAYERS, dropout=dropout)
        self.classifier = nn.Linear(lstm_h, 1)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, sid):
        # 1) MS-CNN
        cnn = self.ms_cnn(x)                         # (B, T, S*M)
        # v3 改动 3a: CNN→Embedding 残差相加
        x_res = self.input_proj(x)                    # (B, T, S*M)  —— x 投影到 cnn 维度
        cnn_with_res = cnn + x_res                   # 残差相加
        # 2) 符号嵌入
        emb_s = self.sym_emb(sid).unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, 8)
        # 3) 拼接 + 投影
        fused = torch.cat([cnn_with_res, emb_s], dim=-1)   # (B, T, S*M+8)
        z = self.embed(fused)                                 # (B, T, lstm_h)
        # 4) Stacked LSTM
        h, _ = self.lstm(z)                                   # (B, T, lstm_h)
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        # 5) Temporal Attention
        a, alpha = self.temp_attn(h)                          # a: (B, lstm_h)
        # v3 改动 3b: 加法残差（论文公式 9 H(x)=F(x)+x, 公式 10 e=a+h_T）
        e = a + h[:, -1, :]                                  # (B, lstm_h)
        e = self.dropout(F.relu(e))
        return e, alpha

    def forward(self, x, sid, use_batch_attn=True):
        e, alpha = self.encode(x, sid)
        if use_batch_attn and self.training and x.size(0) > 1:
            e = self.batch_attn(e)
        logits = self.classifier(e).squeeze(-1)
        return logits, alpha


# ------------------------------------------------------------
# 数据集
# ------------------------------------------------------------
class SeqDataset(Dataset):
    def __init__(self, X, y, sid):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sid = torch.tensor(sid, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.sid[i]


def make_windows(feat, labels, sid, seq=SEQ):
    X, y, sids, idxs = [], [], [], []
    n = len(feat)
    for i in range(seq, n):
        X.append(feat[i - seq:i])
        y.append(labels[i])
        sids.append(sid)
        idxs.append(i)
    if not X:
        return (np.zeros((0, seq, feat.shape[1]), np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.int64), np.zeros(0, int))
    return (np.stack(X).astype(np.float32),
            np.array(y, np.float32),
            np.array(sids, np.int64),
            np.array(idxs, int))


# ------------------------------------------------------------
# ------------------------------------------------------------
# 训练 / 推理（v4 · 拟合度优先 / 早停）
# ------------------------------------------------------------
def train_model(Xtr, ytr, sid_tr):
    """
    训练主循环（v4）：
      - 跑最多 EPOCHS 个 epoch，监控 train loss（拟合度），早停 patience=EAP
      - 保留 train loss 历史最低时的模型权重作为最终模型
      - 不再做任何"按收益选参"或"快照"逻辑

    返回：
      best_state : 训练中 train loss 最低时的 state_dict（CPU）
      best_ep    : 触发最低 train loss 的 epoch
      history    : [{epoch, trainLoss, isBest, isEarlyStop}, ...]
    """
    model = BatchMCRNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # v3 改动 1: 标准 BCE（不加权）→ v4 保留
    crit = nn.BCEWithLogitsLoss()
    ds = SeqDataset(Xtr, ytr, sid_tr)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    history = []               # 每轮 (epoch, train_loss)
    best_loss = float("inf")
    best_state = None          # 训练中 train loss 最低时的 state_dict
    best_ep = 0
    epochs_no_improve = 0

    for ep in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            opt.zero_grad()
            # v3 改动 2: 仅主路 loss（去掉辅路）→ v4 保留
            logits, _ = model(xb, sb, use_batch_attn=True)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)

        train_loss = total_loss / max(n, 1)
        is_best = train_loss < best_loss - 1e-6
        if is_best:
            best_loss = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        history.append({
            "epoch": ep,
            "trainLoss": round(train_loss, 4),
            "isBest": bool(is_best),
        })
        best_mark = " [best]" if is_best else ""
        print(f"  ep{ep:03d} loss={train_loss:.4f}{best_mark} (patience {epochs_no_improve}/{EARLY_STOP_PATIENCE})")

        # v4 早停：min_epochs 之后，连续 patience 个 epoch 未刷新历史最低即停
        if ep >= EARLY_STOP_MIN_EPOCHS and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"  [early-stop] ep{ep} 连续 {EARLY_STOP_PATIENCE} 个 epoch 未刷新历史最低 train loss")
            break

    print(f"  训练结束：实际 {len(history)} epoch，最佳 ep{best_ep} train loss={best_loss:.4f}")
    return best_state, best_ep, history, model


# ------------------------------------------------------------
# v4 改动 4: 删去所有交易模块
#   - backtest_long / _compute_rsi / _iter_cfg / _make_pred_with_th
#   - select_best_on_validation（按"验证集总收益"选参 + 网格搜索）
#   原因：v4 选参基准从"验证集总收益"改为"训练集 loss 拟合度"，交易模块不再需要。
#   9 个 θ 候选仅作 K 线图例的可选项，不参与任何选参。
# ------------------------------------------------------------


@torch.no_grad()
def predict_proba(model, X, sid):
    """模型推理：输出 sigmoid 概率数组。"""
    model.eval()
    out = []
    for i in range(0, len(X), 256):
        xb = torch.tensor(X[i:i+256], dtype=torch.float32, device=DEVICE)
        sb = torch.tensor(sid[i:i+256], dtype=torch.long, device=DEVICE)
        lg, _ = model(xb, sb, use_batch_attn=False)
        out.append(torch.sigmoid(lg).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


# ------------------------------------------------------------
# 主流水线（无交易版）
# ------------------------------------------------------------
def _infer_symbol_from_path(path):
    """从文件路径提取品种 code（XX.YYY 形式）。"""
    import re as _re
    base = os.path.basename(str(path))
    m = _re.search(r'([A-Za-z]{1,3}\.[A-Za-z]{2,4})', base)
    if m:
        return m.group(1).upper()
    base_upper = os.path.splitext(base)[0].upper().replace(" ", "_")
    return base_upper


def _exchange_from_code(code):
    """从品种 code 推断交易所代码。"""
    if "." not in code:
        return "—"
    suf = code.split(".")[-1].upper()
    return {"DCE": "DCE", "CZC": "CZCE", "SHF": "SHFE",
            "CBT": "CBOT", "NYB": "ICE", "CME": "CME",
            "LME": "LME", "NYM": "NYMEX"}.get(suf, suf)


def run_pipeline(xlsx_dir=None, file_paths=None):
    """主流程。两种调用方式：
      1) run_pipeline(xlsx_dir='...')：按 SYMBOL_FILES 扫目录（兼容旧版）
      2) run_pipeline(file_paths=[...])：传入具体文件路径列表
    """
    # 兼容旧式位置参数：main() 里传的是 xlsx_dir 字符串
    if isinstance(xlsx_dir, str) and file_paths is None:
        # 仅用目录扫模式
        pass
    elif isinstance(xlsx_dir, (list, tuple)) and file_paths is None:
        # 旧式位置参数误传 list：当 file_paths 处理
        file_paths = list(xlsx_dir)
        xlsx_dir = None
    print("=" * 60)
    print(f"Batch-MCRNN v2（无交易）· device={DEVICE}")
    print("=" * 60)

    # 动态调整 N_SYM（Embedding 维度需要匹配实际品种数）
    global N_SYM
    if file_paths is not None:
        # 先粗估（按文件路径推断）
        syms_est = [_infer_symbol_from_path(p) for p in file_paths]
        syms_est = list(dict.fromkeys(syms_est))  # 去重保序
        N_SYM = max(N_SYM, len(syms_est))
        print(f"  N_SYM 动态调整为 {N_SYM}（按品种数）")

    # 解析输入
    if file_paths is not None:
        # 动态品种：从每个文件路径提取 symbol
        symbols = []
        name_map = {}  # sym -> (cn_name, exchange)
        paths = []
        for p in file_paths:
            sym = _infer_symbol_from_path(p)
            # 避免重复
            if sym in symbols:
                # 重复则加后缀
                i = 2
                while f"{sym}_{i}" in symbols:
                    i += 1
                sym = f"{sym}_{i}"
            symbols.append(sym)
            # 优先用 SYMBOL_INFO 查中文名（GUI 模式与命令行一致）
            base_sym = sym.split("_")[0]
            if base_sym in SYMBOL_INFO:
                name_map[sym] = SYMBOL_INFO[base_sym]  # ('中文名', '交易所')
            else:
                # fallback: 用 code 自身
                name_map[sym] = (sym, _exchange_from_code(base_sym))
            paths.append(p)
        print(f"  共加载 {len(paths)} 个文件（动态品种模式）")
    else:
        symbols = list(SYMBOLS)
        name_map = dict(SYMBOL_INFO)
        paths = None  # 表示用目录扫描
        print(f"  目录扫描模式: {xlsx_dir}")

    all_dfs = {}
    for i, sym in enumerate(symbols):
        if paths is not None:
            p = paths[i]
        else:
            p = find_file(sym, xlsx_dir)
            if p is None:
                raise FileNotFoundError(sym)
        df = add_tech(load_futures(p, sym))
        all_dfs[sym] = df
        # v4 增量: 同时打印 xlsx 的 mtime（让用户看到用了哪份快照）
        mtime_str = pd.Timestamp(os.path.getmtime(p), unit='s').strftime('%Y-%m-%d %H:%M')
        fname = os.path.basename(p)
        print(f"  {sym}: {len(df)}  {df['date'].min().date()}~{df['date'].max().date()}  [{fname}  mtime={mtime_str}]")

    end = max(df["date"].max() for df in all_dfs.values())
    test_start = end - pd.DateOffset(years=1)
    # v4 改动 1: 弃用验证集。原 train=5y + val=6m 合并为新 train=5.5y
    val_start = test_start - pd.DateOffset(months=6)
    train_start = val_start - pd.DateOffset(years=4, months=6)
    train_end = test_start - pd.Timedelta(days=1)   # 新 train_end = test_start - 1 day
    val_end = train_end                            # val 段为空
    print(f"训练 {train_start.date()}~{train_end.date()}  (v4: train+val 合并)")
    print(f"测试 {test_start.date()}~{end.date()}  (1 年测试集)")
    print(f"[v4] 弃用验证集 → 无 val 段")

    packs = []
    for sid, sym in enumerate(symbols):
        df = all_dfs[sym]
        prices = df["close"].values.astype(float)
        tm = (df["date"] >= train_start) & (df["date"] <= train_end)
        vm = pd.Series(False, index=df.index)   # v4: val mask 全 False
        sm = (df["date"] >= test_start) & (df["date"] <= end)
        labels = np.zeros(len(df), dtype=np.int32)
        n_itp = 0
        for mask in (tm, sm):   # v4: 只对 train 和 test 算 RA-PLR 标签
            if mask.sum() < 40:
                continue
            sub = prices[mask]
            pts, _, _ = ra_plr(sub)
            pts = non_turn(pts, sub)
            labels[mask] = aug_lab(len(sub), pts, sub)
            if mask is tm:
                n_itp = len(pts)
        feat = np.nan_to_num(df[FEATURE_COLS].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        packs.append({"sym": sym, "sid": sid, "df": df, "feat": feat,
                      "labels": labels, "tm": tm, "vm": vm, "sm": sm, "n_itp": n_itp})
        print(f"  {sym} trainITP={n_itp} pos={labels[tm].mean():.3f}")

    # 标准化仅用训练
    scaler = StandardScaler()
    scaler.fit(np.concatenate([p["feat"][p["tm"]] for p in packs], axis=0))
    for p in packs:
        p["feat"] = scaler.transform(p["feat"]).astype(np.float32)

    # v4 改动 1: 不再创建 val 窗口
    Xtr, ytr, sid_tr = [], [], []
    for p in packs:
        X, y, s, _ = make_windows(p["feat"][p["tm"]], p["labels"][p["tm"]], p["sid"])
        if len(y):
            Xtr.append(X); ytr.append(y); sid_tr.append(s)
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr); sid_tr = np.concatenate(sid_tr)
    print(f"联合样本 train={len(ytr)} (train+val 合并后)  pos={ytr.mean():.3f}")

    # v4 改动 2 + 4: 训练（早停 + 拟合度优先），不再做"按收益选参"
    print(f"训练 Batch-MCRNN（v4 · EPOCHS={EPOCHS} 上限 · 早停 patience={EARLY_STOP_PATIENCE} min_ep={EARLY_STOP_MIN_EPOCHS}）...")
    best_state, best_ep, history, model = train_model(Xtr, ytr, sid_tr)
    # 加载最佳权重
    model.load_state_dict(best_state)
    print(f"  [best] epoch = {best_ep}（train loss 最低点）")

    # ---- 测试：仅评估分类指标 + 收集概率分布（v4：无交易） ----
    results, full_symbols = [], []
    all_y_true, all_y_score = [], []
    # v4 改动 3: K 线图用 9 个 θ 候选分别画顶/底（不做 RSI+CR 过滤，仅 6 日窗极值）
    #   报告默认显示 AUC/F1/MCC 用 θ=0.1（最低阈值，跟 K 线图默认一致）算的指标
    DEFAULT_TH_FOR_METRICS = 0.1
    for p in packs:
        sym = p["sym"]
        df = p["df"]
        testdf = df[p["sm"]].reset_index(drop=True)
        X, yt, s, ix = make_windows(p["feat"][p["sm"]], p["labels"][p["sm"]], p["sid"])
        if len(X) == 0:
            continue
        pr = predict_proba(model, X, s)
        # v4 报告默认用 θ=0.1 算 F1/MCC/precision/recall（跟 K 线图默认一致；K 线图另有 9 θ 候选可选）
        yp_ref = (pr >= DEFAULT_TH_FOR_METRICS).astype(int)
        try:
            auc = float(roc_auc_score(yt, pr)) if len(np.unique(yt)) > 1 else 0.5
        except Exception:
            auc = 0.5
        f1 = float(f1_score(yt, yp_ref, zero_division=0))
        mcc = float(matthews_corrcoef(yt, yp_ref)) if len(np.unique(yp_ref)) > 1 else 0.0
        prec = float(((yp_ref.astype(bool) & yt.astype(bool)).sum() / max(yp_ref.sum(), 1))) if yp_ref.sum() else 0.0
        rec = float(((yp_ref.astype(bool) & yt.astype(bool)).sum() / max(yt.sum(), 1))) if yt.sum() else 0.0

        all_y_true.append(yt)
        all_y_score.append(pr)

        name, ex = name_map.get(sym, (sym, _exchange_from_code(sym)))
        metrics = {
            "auc": round(auc, 4), "f1": round(f1, 4), "mcc": round(mcc, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "nItpTrain": int(p["n_itp"]),
            "nTestPos": int(yt.sum()), "nPredPos": int(yp_ref.sum()),
            "nTestSamples": int(len(yt)),
            "refTheta": DEFAULT_TH_FOR_METRICS,
        }
        results.append({"symbol": sym, "name": name, **metrics})
        print(f"  {sym}: AUC={auc:.3f} F1={f1:.3f} MCC={mcc:.3f} "
              f"prec={prec:.3f} rec={rec:.3f} pos={int(yt.sum())} pred={int(yp_ref.sum())} (θ={DEFAULT_TH_FOR_METRICS})")

        # ---- 可视化准备 ----
        win = df[(df["date"] >= train_start) & (df["date"] <= end)].reset_index(drop=True)
        dates = [d.strftime("%Y-%m-%d") for d in win["date"]]
        o = [round(float(v), 2) for v in win["open"]]
        h = [round(float(v), 2) for v in win["high"]]
        lo = [round(float(v), 2) for v in win["low"]]
        c = [round(float(v), 2) for v in win["close"]]
        n = len(win)
        wtm = (win["date"] >= train_start) & (win["date"] <= train_end)
        wsm = (win["date"] >= test_start)
        tr_lo = int(np.argmax(wtm)); tr_hi = int(n - 1 - np.argmax(wtm[::-1]))
        # v4: val 段为空（va_lo == va_hi + 1）
        va_lo, va_hi = tr_hi + 1, tr_hi   # 让 val 段无宽度
        te_lo = int(np.argmax(wsm)); te_hi = int(n - 1 - np.argmax(wsm[::-1]))

        # RA-PLR 仅在训练段拟合
        fit = np.full(n, np.nan)
        expts = []
        close_arr = win["close"].values.astype(float)
        idxs = np.where(wtm)[0]
        if len(idxs) >= 40:
            sub = close_arr[idxs]
            pts, delta, _ = ra_plr(sub)
            pts = non_turn(pts, sub)
            local = seg_fit(pts, sub, len(sub))
            fit[idxs] = local
            for pi in pts:
                gi = int(idxs[pi])
                expts.append([gi, round(float(sub[pi]), 2)])

        # 预测概率（全窗口长度，便于画线）
        date_to_idx = {d: i for i, d in enumerate(dates)}
        proba_line = [None] * n
        for j, d in enumerate([d.strftime("%Y-%m-%d") for d in testdf["date"][SEQ:]]):
            if d in date_to_idx:
                proba_line[date_to_idx[d]] = round(float(pr[j]), 4)

        # v4 改动 3: 9 个 θ 候选，每个生成 pred_highs/pred_lows（仅用 6 日窗极值，无 RSI+CR 过滤）
        test_close = testdf["close"].values.astype(float)
        test_high  = testdf["high"].values.astype(float)   # v4 增量: 高点用当日最高价
        test_low   = testdf["low"].values.astype(float)    # v4 增量: 低点用当日最低价
        test_len = len(testdf)
        pred_by_theta = {}  # theta -> {"highs": [...], "lows": [...], "nT": int, "nB": int}
        for theta in THETA_CANDIDATES:
            highs, lows = [], []
            for j in range(len(pr)):
                if pr[j] < theta:
                    continue
                gi = SEQ + j
                if gi >= test_len:
                    continue
                # 6 日窗内极值（含当日）：仅看 gi 及之前
                L, R = max(0, gi - PEAK_WINDOW), gi + 1
                loc = test_close[L:R]
                px_close = float(test_close[gi])
                px_high  = float(test_high[gi])
                px_low   = float(test_low[gi])
                d_str = testdf["date"].iloc[gi].strftime("%Y-%m-%d")
                if d_str not in date_to_idx:
                    continue
                idx_in_win = date_to_idx[d_str]
                is_top = px_close >= loc.max() - 1e-9
                is_bot = px_close <= loc.min() + 1e-9
                if not (is_top or is_bot):
                    continue
                # v4 增量: 高点用当日最高价 (test_high)，低点用当日最低价 (test_low)
                # 用途：K 线 markPoint 三角位置更准确 + 日志输出阶段性高/低用真实高/低价
                if is_top:
                    entry = {"value": [idx_in_win, round(px_high, 2)], "p": round(float(pr[j]), 3)}
                    highs.append(entry)
                else:
                    entry = {"value": [idx_in_win, round(px_low, 2)], "p": round(float(pr[j]), 3)}
                    lows.append(entry)
            pred_by_theta[round(theta, 2)] = {
                "highs": highs, "lows": lows,
                "nT": len(highs), "nB": len(lows),
            }
        # 默认报告用 θ=0.3 (中间偏低) 的顶底数（v4 仅供顶部数字用）
        default_th_key = round(0.3, 2) if round(0.3, 2) in pred_by_theta else round(THETA_CANDIDATES[2], 2)
        n_peak_tops = pred_by_theta[default_th_key]["nT"]
        n_peak_bottoms = pred_by_theta[default_th_key]["nB"]

        full_symbols.append({
            "symbol": sym, "name": name, "exchange": ex,
            "dates": dates, "o": o, "h": h, "l": lo, "c": c,
            "tr": [tr_lo, tr_hi], "va": [va_lo, va_hi], "te": [te_lo, te_hi],
            "raplr_fit": [None if np.isnan(v) else round(float(v), 2) for v in fit],
            "raplr_pts": expts,
            "pred_by_theta": pred_by_theta,           # v4: 9 θ 各自的顶/底
            "nPeakTops": n_peak_tops,                  # 默认 θ=0.3 的顶数
            "nPeakBottoms": n_peak_bottoms,            # 默认 θ=0.3 的底数
            "proba_line": proba_line,
            "metrics": metrics,
        })

    # 汇总 KPI（v4：无交易，仅分类质量 + K 线 9 θ 顶底数）
    yt_all = np.concatenate(all_y_true) if all_y_true else np.array([0])
    ys_all = np.concatenate(all_y_score) if all_y_score else np.array([0.0])
    try:
        pool_auc = float(roc_auc_score(yt_all, ys_all)) if len(np.unique(yt_all)) > 1 else 0.5
    except Exception:
        pool_auc = 0.5
    yp_all = (ys_all >= DEFAULT_TH_FOR_METRICS).astype(int)
    pool_f1 = float(f1_score(yt_all, yp_all, zero_division=0))
    pool_mcc = float(matthews_corrcoef(yt_all, yp_all)) if len(np.unique(yp_all)) > 1 else 0.0
    pool_prec = float((yp_all.astype(bool) & yt_all.astype(bool)).sum() / max(yp_all.sum(), 1)) if yp_all.sum() else 0.0
    pool_rec  = float((yp_all.astype(bool) & yt_all.astype(bool)).sum() / max(yt_all.sum(), 1)) if yt_all.sum() else 0.0

    aggregate = {
        "nSymbols": N_SYM,
        "avgAuc": round(float(np.mean([r["auc"] for r in results])), 3),
        "avgF1": round(float(np.mean([r["f1"] for r in results])), 3),
        "avgMcc": round(float(np.mean([r["mcc"] for r in results])), 3),
        "avgPrecision": round(float(np.mean([r["precision"] for r in results])), 3),
        "avgRecall": round(float(np.mean([r["recall"] for r in results])), 3),
        "poolAuc": round(pool_auc, 3),
        "poolF1": round(pool_f1, 3),
        "poolMcc": round(pool_mcc, 3),
        "poolPrecision": round(pool_prec, 3),
        "poolRecall": round(pool_rec, 3),
        "totalTestPos": int(yt_all.sum()),
        "totalPredPos": int(yp_all.sum()),
        "refTheta": DEFAULT_TH_FOR_METRICS,
    }
    split = {
        "paperRatio": "v4: train+val 合并为 train，比例 5.5y : 1y (无 val)",
        "trainStart": train_start.strftime("%Y-%m-%d"), "trainEnd": train_end.strftime("%Y-%m-%d"),
        "valStart": "(弃用)", "valEnd": "(弃用)",
        "testStart": test_start.strftime("%Y-%m-%d"), "testEnd": end.strftime("%Y-%m-%d"),
        "paramSelect": "v4: train loss 拟合度 + 早停 (patience=10, min_epochs=20)",
    }
    model_info = {
        "name": "Batch-MCRNN（v4 · 拟合度优先 · 早停 · 9 θ 候选 K 线图例）",
        "seqLen": SEQ, "cnnChannels": CNN_CHANNELS, "kernels": [3, 5, 7],
        "lstmHidden": LSTM_HIDDEN, "lstmLayers": LSTM_LAYERS,
        "batchAttnHeads": BATCH_ATTN_HEADS, "nSymbols": N_SYM,
        "epochsMax": EPOCHS,
        "earlyStopPatience": EARLY_STOP_PATIENCE,
        "earlyStopMinEpochs": EARLY_STOP_MIN_EPOCHS,
        "bestEpoch": best_ep,
        "bestTrainLoss": round(float(np.min([h["trainLoss"] for h in history])), 4),
        "selection": "no param grid search; model selected by lowest train loss (with early stopping)",
        "history": history,
        "rule": "ITP→概率阈值（9 候选）→6日窗内极值（无 RSI+CR 过滤）",
        "paramSelectCriterion": "minimize train BCE loss with early stopping (patience=10)",
        "tradingModule": "REMOVED in v4 (no backtest, no RSI+CR filter)",
        "peakFilter": {
            "peakWindow": PEAK_WINDOW,
            "noFuture": True,
        },
        "thetaCandidates": THETA_CANDIDATES,
    }
    # 按"内盘→外盘"重新排序 symbols 和 results
    def _ex_group(ex):
        """返回交易所分组：0=内盘，1=外盘，9=未知"""
        if ex in ("DCE", "CZCE", "SHFE"):
            return 0  # 内盘
        if ex in ("CBOT", "ICE", "CME", "LME", "NYMEX"):
            return 1  # 外盘
        return 9

    sort_key = lambda o: (_ex_group(o.get("exchange", "")), o.get("symbol", ""))
    full_symbols = sorted(full_symbols, key=sort_key)
    results = sorted(results, key=sort_key)

    return {
        "generatedAt": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "split": split, "model": model_info, "aggregate": aggregate,
        "symbols": full_symbols, "results": results,
    }


# ------------------------------------------------------------
# HTML（带左侧目录）
# ------------------------------------------------------------
def build_html(data):
    ep = os.path.join(SCRIPT_DIR, "echarts.min.js")
    if os.path.exists(ep):
        with open(ep, encoding="utf-8", errors="replace") as f:
            echarts_js = f.read()
    else:
        echarts_js = ""
    echarts_tag = "<script>" + echarts_js + "</script>" if echarts_js else \
        '<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>'

    json_text = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")

    sym_toc = "".join(
        '<li><a href="#sym-' + s["symbol"].replace(".", "_") + '">'
        '<span class="sym-name">' + s["name"] + '</span> '
        '<span class="sym-sep">·</span> '
        '<span class="sym-code">' + s["symbol"] + '</span></a></li>'
        for s in data["symbols"]
    )

    n_sym = N_SYM
    cnn_ch = CNN_CHANNELS
    lstm_layers = LSTM_LAYERS
    lstm_hidden = LSTM_HIDDEN
    batch_attn_heads = BATCH_ATTN_HEADS
    seq = SEQ
    epochs = EPOCHS
    # v4: 不再使用 selectionResult 字段（v3 才需要）

    css = """
:root { --bg:#fafafa; --card:#fff; --line:#e5e7eb; --text:#1a1a1a;
  --muted:#6b7280; --acc:#1f4e79; --side:#1f2937; --side-active:#2563eb; }
* { box-sizing:border-box; }
html { scroll-behavior: smooth; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",
  "PingFang SC","Microsoft YaHei",sans-serif;
  background:var(--bg); color:var(--text); line-height:1.55; }
aside.toc { position:fixed; top:0; left:0; bottom:0; width:240px;
  background:var(--side); color:#d1d5db; overflow-y:auto; padding:18px 16px; z-index:50;
  box-shadow: 2px 0 8px rgba(0,0,0,0.06); }
aside.toc h2 { color:#fff; font-size:1rem; margin:0 0 12px;
  border-left:3px solid var(--side-active); padding-left:8px; }
aside.toc .grp { font-size:.72rem; color:#9ca3af; text-transform:uppercase;
  letter-spacing:.06em; margin:14px 0 6px; }
aside.toc ul { list-style:none; margin:0 0 6px; padding:0; }
aside.toc li a { display:flex; justify-content:flex-start; align-items:baseline;
  gap:5px; padding:5px 8px; border-radius:5px; color:#cbd5e1; text-decoration:none;
  font-size:.86rem; transition: background .15s, color .15s; }
aside.toc li a:hover { background:rgba(255,255,255,0.06); color:#fff; }
aside.toc li a.active { background:var(--side-active); color:#fff; }
aside.toc .sym-code { font-family:ui-monospace,Consolas,monospace; font-size:.78rem; opacity:.85; }
aside.toc .sym-name { color:#9ca3af; font-size:.78rem; }
aside.toc .sym-sep { color:#64748b; font-size:.78rem; opacity:.6; }
main { margin-left:240px; }
.wrap { max-width:1500px; margin:0 auto; padding:24px 28px 80px; }
h1 { font-size:1.7rem; margin:0 0 8px; }
h2 { font-size:1.2rem; margin:32px 0 12px; border-left:4px solid var(--acc); padding-left:10px; scroll-margin-top: 20px; }
h3 { font-size:1.05rem; margin:18px 0 8px; scroll-margin-top: 20px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; margin-bottom:14px; scroll-margin-top: 20px; }
.kpi-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
.kpi { background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:12px; text-align:center; }
.kpi .l { font-size:.78rem; color:var(--muted); }
.kpi .v { font-size:1.25rem; font-weight:700; margin-top:4px; }
table { width:100%; border-collapse:collapse; font-size:.88rem; }
th,td { border-bottom:1px solid var(--line); padding:8px 6px; text-align:left; }
th { background:#f1f5f9; font-weight:600; }
.chart { width:100%; min-width:1380px; height:420px; }
.kline-chart { width:100%; min-width:1380px; height:420px; }
.note { font-size:.85rem; color:var(--muted); }
.badge { display:inline-block; background:#ecfdf5; border:1px solid #6ee7b7; color:#065f46;
  border-radius:4px; padding:2px 8px; font-size:.82rem; margin-right:6px; }
.badge.warn { background:#fef3c7; border-color:#fcd34d; color:#92400e; }
.badge.info { background:#dbeafe; border-color:#93c5fd; color:#1e3a8a; }
code { background:#f1f5f9; padding:1px 5px; border-radius:3px; font-size:.85em; }
ul.arch { margin:8px 0; padding-left:1.2em; }
ul.arch li { margin:4px 0; }
#js-err { position:fixed; top:0; left:0; right:0; background:#dc2626; color:#fff;
  padding:10px 14px; z-index:9999; font:13px ui-monospace,Consolas,monospace;
  white-space:pre-wrap; display:none; }
"""

    body_html = (
        '<div id="js-err"></div>\n'
        '<aside class="toc" id="toc">\n'
        '  <h2>目录</h2>\n'
        '  <div class="grp">概览</div>\n'
        '  <ul>\n'
        '    <li><a href="#sec-1">一、架构</a></li>\n'
        '    <li><a href="#sec-2">二、切分与参数</a></li>\n'
        '    <li><a href="#sec-3">三、汇总 KPI</a></li>\n'
        '    <li><a href="#sec-4">四、分品种结果</a></li>\n'
        '    <li><a href="#sec-5">五、训练曲线</a></li>\n'
        '    <li><a href="#sec-6">六、测试集预测分布</a></li>\n'
        '    <li><a href="#sec-7">七、K 线与预测点</a></li>\n'
        '    <li><a href="#sec-8">八、方法与说明</a></li>\n'
        '  </ul>\n'
        '  <div class="grp">分品种（' + str(n_sym) + '）</div>\n'
        '  <ul>' + sym_toc + '</ul>\n'
        '</aside>\n'
        '<main>\n'
        '<div class="wrap">\n'
        '<h1>TURNLAB · 完整 <span style="color:#1f4e79">Batch-MCRNN</span> 分析报告\n'
        '  <span class="badge warn">v4 · 拟合度优先</span>\n'
        '  <span class="badge info">早停 patience=10</span>\n'
        '  <span class="badge info">9 θ 候选 K 线图例</span>\n'
        '</h1>\n'
        '<p class="note">论文对齐实现：MS-1DCNN (k=3/5/7) → Fusion Embedding → 2-layer LSTM → Temporal Attention + Residual\n'
        '→ Batch Attention（训练）→ Weight-sharing Classifier（推理）。' + str(n_sym) + ' 品种联合训练。测试集最近 1 年。\n'
        '<span class="badge">无未来函数</span><span class="badge">v4 弃用验证集 → train+val 合并</span>\n'
        '<span class="badge">EPOCHS=' + str(epochs) + '（上限）· 早停 patience=' + str(EARLY_STOP_PATIENCE) + ' · min_epochs=' + str(EARLY_STOP_MIN_EPOCHS) + '</span>\n'
        '<span class="badge">unweighted BCE · 单主路 · CNN/LSTM/Attention 三处残差</span></p>\n'
        '<div class="card" id="sec-1">\n'
        '<h2 style="margin-top:0">一、架构</h2>\n'
        '<ul class="arch">\n'
        '<li><b>MS-CNN</b>：1D 卷积核 {3,5,7}，每尺度 ' + str(cnn_ch) + ' 通道，BatchNorm+ReLU，same-pad 保序列长</li>\n'
        '<li><b>Fusion Embedding</b>：MS-CNN 输出 ∥ 1×1 投影的原始输入（v3 残差相加）∥ 品种 embedding(8维) → Linear→H</li>\n'
        '<li><b>Stacked LSTM</b>：' + str(lstm_layers) + ' 层，hidden=' + str(lstm_hidden) + '，Dropout+BatchNorm</li>\n'
        '<li><b>Temporal Attention + 残差</b>（v3 加法残差）：α_t = softmax(uᵀ tanh(W h_t + b))，context a=Σ α_t h_t，<b>e = a + h_T</b>（论文公式 9 H(x)=F(x)+x）</li>\n'
        '<li><b>Batch Attention</b>：把 batch 维当序列做 multi-head self-attn（' + str(batch_attn_heads) + ' heads），仅训练；推理用 weight-sharing classifier 绕过</li>\n'
        '<li><b>分类头</b>：Linear→logit，<b>unweighted BCE</b>，<b>单主路损失</b>（v3 去辅路）</li>\n'
        '</ul>\n'
        '</div>\n'
        '<div class="card" id="sec-2">\n'
        '<h2 style="margin-top:0">二、切分与参数</h2>\n'
        '<div class="kpi-row" id="split-row"></div>\n'
        '<p class="note" id="param-note"></p>\n'
        '<p class="note"><b>训练控制（v4）</b>：最大 epoch = ' + str(epochs) + '（用户指定），<strong>早停</strong> patience=' + str(EARLY_STOP_PATIENCE) + ' min_epochs=' + str(EARLY_STOP_MIN_EPOCHS) + '。\n'
        '  监控指标 = <strong>train loss 拟合度</strong>（不监控验证集，因 v4 已弃用）。\n'
        '  连续 ' + str(EARLY_STOP_PATIENCE) + ' 个 epoch 未刷新历史最低 train loss 即停（最少跑 ' + str(EARLY_STOP_MIN_EPOCHS) + ' epoch）。\n'
        '  <span id="es-state"></span></p>\n'
        '</div>\n'
        '<div class="card" id="sec-3">\n'
        '<h2 style="margin-top:0">三、汇总 KPI（v4 无交易，仅看分类质量）</h2>\n'
        '<div class="kpi-row" id="kpi-row"></div>\n'
        '<p class="note" id="pool-note"></p>\n'
        '</div>\n'
        '<div class="card" id="sec-4">\n'
        '<h2 style="margin-top:0">四、分品种结果</h2>\n'
        '<table><thead><tr>\n'
        '<th>品种</th><th>AUC</th><th>F1</th><th>MCC</th><th>精度</th><th>召回</th><th>测试正样本</th><th>预测正样本</th>\n'
        '</tr></thead><tbody id="tbody"></tbody></table>\n'
        '</div>\n'
        '<div class="card" id="sec-5">\n'
        '<h2 style="margin-top:0">五、训练曲线（每轮 train loss · 拟合度）</h2>\n'
        '<div class="chart" id="train-chart"></div>\n'
        '<p class="note">单条线 = 每个 epoch 跑完全量训练集后的平均 BCE loss（越低越好）。红色★=历史最低点（被选为最终模型）。橙色虚线=早停触发 epoch（若触发）。</p>\n'
        '</div>\n'
        '<div class="card" id="sec-6">\n'
        '<h2 style="margin-top:0">六、测试集预测分布（按品种 · θ=0.1 参考）</h2>\n'
        '<div class="chart" id="prob-hist"></div>\n'
        '<p class="note">蓝=测试集真实正样本数，黄=模型预测为正的样本数（决策阈值 θ=0.1，跟 K 线图默认一致）。K 线图另有 9 个 θ 候选可选。</p>\n'
        '</div>\n'
        '<div class="card" id="sec-7">\n'
        '<h2 style="margin-top:0">七、K 线与预测点（按品种 · 内盘→外盘）</h2>\n'
        '<p class="note">蓝=训练 / 绿=测试 · 紫=RA-PLR(仅训练)。<b>图例</b>：K线 + RA-PLR 始终显示；预测顶/底 9 个 θ 候选（<b>互斥单选</b>），<b>默认显示 θ=0.1</b>，点击其他 θ 会自动替换。每个 θ 同时画顶▲（红）和底▼（绿，黑轮廓），统一色彩深度。K 线上 6 日窗内极值即触发，不做 RSI/CR 过滤。可拖动缩放。</p>\n'
        '<div id="klines"></div>\n'
        '</div>\n'
        '<div class="card" id="sec-8">\n'
        '<h2 style="margin-top:0">八、方法与说明</h2>\n'
        '<ul>\n'
        '<li><b>v4 改动 1 — 弃用验证集</b>：原 train=5y + val=6m 合并为新 train=5.5y，test=1y 不变。比例 5.5y : 1y ≈ 84.6% : 15.4%。\n'
        '    选参基准从"验证集总收益"改为"训练集 loss 拟合度"，不再做任何"验证集网格搜索"。</li>\n'
        '<li><b>v4 改动 2 — 拟合度优先 + 早停</b>：每 epoch 监控 train loss（最低点保留为最终模型）。\n'
        '    早停 patience=' + str(EARLY_STOP_PATIENCE) + '（连续 ' + str(EARLY_STOP_PATIENCE) + ' 个 epoch 未刷新历史最低即停），min_epochs=' + str(EARLY_STOP_MIN_EPOCHS) + '（避免噪声早停）。上限 EPOCHS=' + str(epochs) + '。</li>\n'
        '<li><b>v4 改动 3 — 9 θ 候选 K 线图例</b>：THETA_CANDIDATES = ' + str(THETA_CANDIDATES) + '，\n'
        '    每个 θ 作为 1 个 series（同时画顶▲和底▼，合并为"预测顶/底（θ=0.X）"一个图例名）。\n'
        '    <b>统一色彩深度</b>（不分 θ 渐变）：顶 = <code>#dc2626</code>（红），底 = <code>#16a34a</code>（绿），三角加 <code>#1a1a1a</code> 1.2px 黑轮廓便于在密集 K 线上区分。\n'
        '    <b>默认显示</b>：K线 + RA-PLR + <code>预测顶/底(θ=0.1)</code>，其他 8 个预测顶底默认隐藏。\n'
        '    <b>9 个预测顶底互斥单选</b>：用 <code>legendselectchanged</code> 事件 + <code>params.name</code> 知道用户点的（不用 selectedMode:"single"，否则 K 线也会被取消；也不用 setInterval 兜底，否则会和点击冲突）。用户点击其他 θ 会自动替换当前选中的 θ。</li>\n'
        '<li><b>v4 改动 4 — 删去所有交易模块</b>：backtest_long / _compute_rsi / _iter_cfg / select_best_on_validation / 验证集网格搜索 / 顶底 RSI+CR(DC) 双重过滤——全部移除。\n'
        '    顶/底判定简化为：模型概率 ≥ θ 且在 6 日窗内为极值（含当日）。</li>\n'
        '<li><b>残差（v3 保留）</b>：三处加法残差 — (1) CNN→Embedding：cnn + Linear(x)；(2) LSTM→Attention：<b>e = a + h_T</b>（论文公式 10）；(3) BatchAttention：LN(MSA(x)+x)。</li>\n'
        '<li><b>损失（v3 保留）</b>：unweighted BCE（去 pos_weight），仅主路（去辅路）。</li>\n'
        '<li>模型输入仅过去 ' + str(seq) + ' 日；顶/底判定窗 = 过去 6 日 + 当日；无未来函数。</li>\n'
        '<li>Batch Attention 只在训练阶段使用；推理时走 weight-sharing 分类头，无 batch 依赖。</li>\n'
        '<li>RA-PLR 极值点为事后标签，仅用于训练标签与评估指标，不进入特征。</li>\n'
        '<li>研究演示，不构成投资建议。</li>\n'
        '</ul>\n'
        '</div>\n'
        '</div>\n'
        '</main>\n'
    )

    # ----- JS 部分：纯字符串拼接 + append，绝不用 f-string 嵌套 -----
    p = []  # 收集 JS 行
    p.append('\n<script>\n')
    p.append("window.addEventListener('error', function(e){\n")
    p.append("  var el = document.getElementById('js-err');\n")
    p.append("  if (el) { el.style.display='block'; el.textContent += '[ERR] ' + e.message + ' @ ' + e.lineno + ':' + e.colno + '\\n'; }\n")
    p.append("  console.error(e.message, e.filename, e.lineno, e.colno);\n")
    p.append("});\n")
    p.append('(function(){\n')
    p.append('  var DATA = ' + json_text + ';\n')
    p.append('  var S = DATA.split, M = DATA.model, A = DATA.aggregate;\n')
    p.append('  var pct = function(x,d){ d = (d==null)?1:d; return (x*100).toFixed(d) + "%"; };\n')
    p.append('  function setHTML(id, html){ var n=document.getElementById(id); if(n) n.innerHTML = html; }\n')
    p.append('  function setText(id, t){ var n=document.getElementById(id); if(n) n.textContent = t; }\n')
    p.append('  function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }\n')
    p.append('\n')
    p.append('  // 切分行（v4: 验证集已弃用）\n')
    p.append('  setHTML("split-row",\n')
    p.append('    [\n')
    p.append('      ["训练 (train+val)", esc(S.trainStart + " → " + S.trainEnd)],\n')
    p.append('      ["测试", esc(S.testStart + " → " + S.testEnd)],\n')
    p.append('      ["验证", esc("(v4 弃用)")],\n')
    p.append('      ["选参基准", esc("train loss 拟合度")],\n')
    p.append('      ["品种数", esc(String(M.nSymbols))],\n')
    p.append('      ["最大 epoch", esc(String(M.epochsMax))],\n')
    p.append('      ["早停 patience", esc(String(M.earlyStopPatience))],\n')
    p.append('      ["最佳 epoch", esc(String(M.bestEpoch))],\n')
    p.append('    ].map(function(r){ return \'<div class="kpi"><div class="l">\' + r[0] + \'</div><div class="v" style="font-size:.95rem">\' + r[1] + \'</div></div>\'; }).join(""));\n')
    p.append('  setText("param-note", "架构 L=" + M.seqLen + " CNN=" + M.cnnChannels + " LSTM " + M.lstmLayers + "x" + M.lstmHidden + " · BatchAttn heads=" + M.batchAttnHeads + " · 选参=train loss 拟合度（v4 早停）");\n')
    p.append('  setText("es-state", "★ 最佳 epoch = " + M.bestEpoch + "（train loss=" + M.bestTrainLoss.toFixed(4) + " 历史最低）");\n')
    p.append('\n')
    p.append('  // KPI 行\n')
    p.append('  setHTML("kpi-row",\n')
    p.append('    [\n')
    p.append('      ["品种数", A.nSymbols, ""],\n')
    p.append('      ["平均 AUC", A.avgAuc.toFixed(3), ""],\n')
    p.append('      ["平均 F1", A.avgF1.toFixed(3), ""],\n')
    p.append('      ["平均 MCC", A.avgMcc.toFixed(3), ""],\n')
    p.append('      ["平均 精度", pct(A.avgPrecision, 1), ""],\n')
    p.append('      ["平均 召回", pct(A.avgRecall, 1), ""],\n')
    p.append('      ["测试正样本", A.totalTestPos, ""],\n')
    p.append('      ["预测正样本", A.totalPredPos, ""],\n')
    p.append('    ].map(function(r){ return \'<div class="kpi"><div class="l">\' + r[0] + \'</div><div class="v">\' + r[1] + \'</div></div>\'; }).join(""));\n')
    p.append('  setHTML("pool-note",\n')
    p.append('    \'<b>合并全部品种测试集</b>（pooled）：AUC=\' + A.poolAuc.toFixed(3) +\n')
    p.append('    \' · F1=\' + A.poolF1.toFixed(3) + \' · MCC=\' + A.poolMcc.toFixed(3) +\n')
    p.append('    \' · 精度=\' + pct(A.poolPrecision, 1) + \' · 召回=\' + pct(A.poolRecall, 1)\n')
    p.append('  );\n')
    p.append('\n')
    p.append('  // 分品种结果表\n')
    p.append('  setHTML("tbody", DATA.symbols.map(function(s){\n')
    p.append('    var m = s.metrics;\n')
    p.append('    return \'<tr id="row-\' + esc(s.symbol.replace(/\\./g,"_")) + \'">\' +\n')
    p.append('      \'<td><b>\' + esc(s.name) + \'</b><br><span style="color:#666;font-family:monospace;font-size:.8em">\' + esc(s.symbol) + \'</span></td>\' +\n')
    p.append('      \'<td>\' + m.auc.toFixed(3) + \'</td>\' +\n')
    p.append('      \'<td>\' + m.f1.toFixed(3) + \'</td>\' +\n')
    p.append('      \'<td>\' + m.mcc.toFixed(3) + \'</td>\' +\n')
    p.append('      \'<td>\' + (m.precision*100).toFixed(1) + \'%</td>\' +\n')
    p.append('      \'<td>\' + (m.recall*100).toFixed(1) + \'%</td>\' +\n')
    p.append('      \'<td>\' + m.nTestPos + \'</td>\' +\n')
    p.append('      \'<td>\' + m.nPredPos + \'</td></tr>\';\n')
    p.append('  }).join(""));\n')
    p.append('\n')
    p.append('  // 训练曲线（v4: 单条 train loss + 早停触发 epoch + 历史最低点标记）\n')
    p.append('  (function(){\n')
    p.append('    var h = M.history || [];\n')
    p.append('    if (!h.length) return;\n')
    p.append('    var xs = h.map(function(r){ return r.epoch; });\n')
    p.append('    var loss = h.map(function(r){ return r.trainLoss; });\n')
    p.append('    var bestEp = M.bestEpoch;\n')
    p.append('    var runEp  = h.length;\n')
    p.append('    var chart = echarts.init(document.getElementById("train-chart"));\n')
    p.append('    chart.setOption({\n')
    p.append('      backgroundColor:"#fff",\n')
    p.append('      tooltip:{trigger:"axis"},\n')
    p.append('      legend:{top:6, data:["train loss"]},\n')
    p.append('      grid:{left:55,right:20,top:42,bottom:30},\n')
    p.append('      xAxis:{type:"category", name:"epoch", data:xs},\n')
    p.append('      yAxis:{type:"value", name:"BCE loss"},\n')
    p.append('      series:[\n')
    p.append('        {name:"train loss", type:"line", data:loss, smooth:false,\n')
    p.append('          itemStyle:{color:"#2563eb"}, lineStyle:{width:2},\n')
    p.append('          areaStyle:{color:"rgba(37,99,235,0.08)"},\n')
    p.append('          markPoint:{\n')
    p.append('            data:[{name:"best", value:loss[xs.indexOf(bestEp)], xAxis:bestEp, yAxis:loss[xs.indexOf(bestEp)],\n')
    p.append('              itemStyle:{color:"#dc2626"}, symbolSize:50,\n')
    p.append('              label:{formatter:"★ best\\nep"+bestEp, color:"#dc2626", fontSize:11}}],\n')
    p.append('          },\n')
    p.append('          markLine:{\n')
    p.append('            silent:true,\n')
    p.append('            data:[\n')
    p.append('              {xAxis: bestEp, label:{formatter:"best ep"+bestEp, color:"#dc2626"},\n')
    p.append('               lineStyle:{color:"#dc2626", type:"dashed", width:1.5}}\n')
    p.append('            ]\n')
    p.append('          }\n')
    p.append('        }\n')
    p.append('      ]\n')
    p.append('    });\n')
    p.append('  })();\n')
    p.append('\n')
    p.append('  // 测试集预测分布\n')
    p.append('  (function(){\n')
    p.append('    var syms = DATA.symbols;\n')
    p.append('    var labels = syms.map(function(s){ return s.symbol; });\n')
    p.append('    var trueBars = syms.map(function(s){ return s.metrics.nTestPos; });\n')
    p.append('    var predBars = syms.map(function(s){ return s.metrics.nPredPos; });\n')
    p.append('    var chart = echarts.init(document.getElementById("prob-hist"));\n')
    p.append('    chart.setOption({\n')
    p.append('      backgroundColor:"#fff",\n')
    p.append('      tooltip:{trigger:"axis"},\n')
    p.append('      legend:{top:6, data:["测试集正样本","预测正样本"]},\n')
    p.append('      grid:{left:50,right:20,top:42,bottom:60},\n')
    p.append('      xAxis:{type:"category", data:labels, axisLabel:{rotate:30, fontSize:10}},\n')
    p.append('      yAxis:{type:"value", name:"样本数"},\n')
    p.append('      series:[\n')
    p.append('        {name:"测试集正样本", type:"bar", data:trueBars, itemStyle:{color:"#3b82f6"}},\n')
    p.append('        {name:"预测正样本", type:"bar", data:predBars, itemStyle:{color:"#f59e0b"}}\n')
    p.append('      ]\n')
    p.append('    });\n')
    p.append('  })();\n')
    p.append('\n')
    p.append('  // K 线 + 9 θ 候选顶底（v4: legend 单选，同色系渐变，markPoint 画顶底）\n')
    p.append('  (function(){\n')
    p.append('    var cont = document.getElementById("klines");\n')
    p.append('    if (!cont) return;\n')
    p.append('    var syms = DATA.symbols;\n')
    p.append('    var thetas = M.thetaCandidates || [];   // 9 个 θ 候选\n')
    p.append('    // v4 修复（迭代 3）：顶底统一色彩深度（不分 θ 渐变）+ 黑色轮廓\n')
    p.append('    // 顶▲ 固定 #dc2626（红色，9 阶中位），底▼ 固定 #16a34a（绿色，9 阶中位）\n')
    p.append('    // 黑色轮廓让小三角在密集 K 线上更易区分\n')
    p.append('    var TRIANGLE_TOP = "#dc2626";     // 顶 = 红色固定\n')
    p.append('    var TRIANGLE_BOT = "#16a34a";     // 底 = 绿色固定\n')
    p.append('    var TRIANGLE_BORDER = "#1a1a1a";  // 黑色轮廓\n')
    p.append('    var TRIANGLE_BW = 1.2;            // 轮廓宽\n')
    p.append('    for (var i = 0; i < syms.length; i++) {\n')
    p.append('      var s = syms[i];\n')
    p.append('      var id = "k-" + s.symbol.replace(/\\./g, "_");\n')
    p.append('      var symId = "sym-" + s.symbol.replace(/\\./g, "_");\n')
    p.append('      cont.insertAdjacentHTML("beforeend",\n')
    p.append('        \'<h3 id="\' + symId + \'">\' + esc(s.name) + \' · \' + esc(s.symbol) +\n')
    p.append('        \'　<span class="note">AUC \' + s.metrics.auc.toFixed(3) +\n')
    p.append('        \' · F1 \' + s.metrics.f1.toFixed(3) +\n')
    p.append('        \' · MCC \' + s.metrics.mcc.toFixed(3) +\n')
    p.append('        \' · 默认θ=0.3 顶/底 \' + s.nPeakTops + \'/\' + s.nPeakBottoms + \'</span></h3>\' +\n')
    p.append('        \'<div class="chart" id="\' + id + \'"></div>\'\n')
    p.append('      );\n')
    p.append('    }\n')
    p.append('    for (var i = 0; i < syms.length; i++) {\n')
    p.append('      var s = syms[i];\n')
    p.append('      var id = "k-" + s.symbol.replace(/\\./g, "_");\n')
    p.append('      var n = s.dates.length;\n')
    p.append('      var kdata = [];\n')
    p.append('      for (var j = 0; j < n; j++) kdata.push([s.o[j], s.c[j], s.l[j], s.h[j]]);\n')
    p.append('      var fit = [];\n')
    p.append('      for (var j = 0; j < n; j++) fit.push((j >= s.tr[0] && j <= s.tr[1]) ? s.raplr_fit[j] : null);\n')
    p.append('      var startPct = Math.min(99, Math.max(0, (s.te[0] - 30) / n * 100));\n')
    p.append('      var probaLine = s.proba_line.slice();\n')
    p.append('      var chart = echarts.init(document.getElementById(id));\n')
    p.append('      // ---- 构造 9 个 series，每个 series 画一个 θ 的顶+底（markPoint 合并顶底） ----\n')
    p.append('      var seriesArr = [\n')
    p.append('        {name:"K线", type:"candlestick", data:kdata, itemStyle:{color:"#ef4444", color0:"#22c55e", borderColor:"#ef4444", borderColor0:"#22c55e"}},\n')
    p.append('        {name:"RA-PLR", type:"line", data:fit, connectNulls:false, symbol:"none", lineStyle:{color:"#8b5cf6", width:1.4}}\n')
    p.append('      ];\n')
    p.append('      // legend data 也要 9 个 θ 名（用于图例显示）\n')
    p.append('      var legendData = ["K线","RA-PLR"];\n')
    p.append('      // v4 修复：默认 K线 + RA-PLR + θ=0.1 都显示，其他 8 个预测顶底不显示\n')
    p.append('      var selected0 = {\n')
    p.append('        "K线": true,\n')
    p.append('        "RA-PLR": true,\n')
    p.append('        "预测顶/底(θ=0.1)": true\n')
    p.append('      };\n')
    p.append('      for (var ti = 0; ti < thetas.length; ti++) {\n')
    p.append('        var th = thetas[ti];\n')
    p.append('        var key = "p" + th.toFixed(1);\n')
    p.append('        var thKey = th.toFixed(1);\n')
    p.append('        var bucket = (s.pred_by_theta && s.pred_by_theta[thKey]) ? s.pred_by_theta[thKey] : null;\n')
    p.append('        var highs = bucket ? bucket.highs : [];\n')
    p.append('        var lows  = bucket ? bucket.lows  : [];\n')
    p.append('        // markPoint: 把顶和底合并到一个 series 的 markPoint.data 里\n')
    p.append('        // 顶（高点）= TRIANGLE_TOP 红，底（低点）= TRIANGLE_BOT 绿\n')
    p.append('        // 加黑色轮廓让密集 K 线上更易区分\n')
    p.append('        var mpData = [];\n')
    p.append('        for (var k = 0; k < highs.length; k++) {\n')
    p.append('          mpData.push({coord: highs[k].value, symbol:"triangle", symbolSize:11, symbolOffset:[0,-9],\n')
    p.append('            itemStyle:{color:TRIANGLE_TOP, borderColor:TRIANGLE_BORDER, borderWidth:TRIANGLE_BW}});\n')
    p.append('        }\n')
    p.append('        for (var k = 0; k < lows.length; k++) {\n')
    p.append('          mpData.push({coord: lows[k].value, symbol:"triangle", symbolRotate:180, symbolSize:11, symbolOffset:[0,9],\n')
    p.append('            itemStyle:{color:TRIANGLE_BOT, borderColor:TRIANGLE_BORDER, borderWidth:TRIANGLE_BW}});\n')
    p.append('        }\n')
    p.append('        var seriesName = "预测顶/底(θ=" + thKey + ")";\n')
    p.append('        // 用 candlestick 上的 markPoint（不可见 series）承载\n')
    p.append('        seriesArr.push({\n')
    p.append('          name: seriesName, type:"scatter",\n')
    p.append('          data:[],   // 空 data（用 markPoint 承载顶底）\n')
    p.append('          symbol:"none",\n')
    p.append('          markPoint:{ silent:false, symbolSize:11,\n')
    p.append('            data: mpData,\n')
    p.append('            label:{show:false}\n')
    p.append('          }\n')
    p.append('        });\n')
    p.append('        legendData.push(seriesName);\n')
    p.append('        if (thKey !== "0.1") {\n')
    p.append('          selected0[seriesName] = false;   // v4 修复：默认只有 θ=0.1 显示\n')
    p.append('        }\n')
    p.append('      }\n')
    p.append('      chart.setOption({\n')
    p.append('        backgroundColor:"#fff",\n')
    p.append('        tooltip:{trigger:"axis", axisPointer:{type:"cross"}},\n')
    p.append('        // v4 修复：不用 selectedMode:"single"（那个会取消 K 线），\n')
    p.append('        //   改用 legendselectchanged 事件实现 9 个预测顶底互斥，K 线/RA-PLR 始终显示\n')
    p.append('        legend:{top:4, type:"plain", textStyle:{fontSize:10}, itemGap:8, padding:[2,10], data:legendData, selected:selected0},\n')
    p.append('        grid:{left:55,right:16,top:38,bottom:56},\n')
    p.append('        xAxis:{\n')
    p.append('          type:"category", data:s.dates, boundaryGap:true,\n')
    p.append('          markArea:{silent:true, data:[\n')
    p.append('            [{xAxis:s.tr[0], itemStyle:{color:"rgba(59,130,246,0.07)"}}, {xAxis:s.tr[1]}],\n')
    p.append('            [{xAxis:s.te[0], itemStyle:{color:"rgba(16,185,129,0.10)"}}, {xAxis:s.te[1]}],\n')
    p.append('          ]}\n')
    p.append('        },\n')
    p.append('        yAxis:[\n')
    p.append('          {scale:true, splitLine:{lineStyle:{color:"#eee"}}},\n')
    p.append('        ],\n')
    p.append('        dataZoom:[\n')
    p.append('          {type:"inside", start:startPct, end:100},\n')
    p.append('          {type:"slider", height:14, bottom:4, start:startPct, end:100}\n')
    p.append('        ],\n')
    p.append('        series: seriesArr\n')
    p.append('      });\n')
    p.append('      // v4 修复（迭代 2）：9 个预测顶底互斥，但不卡点击\n')
    p.append('      // 关键修复：去掉了 setInterval（之前每 200ms 扫描会和用户点击冲突，\n')
    p.append('      //   导致点完立即被 setInterval 取消，表现为"点不动"）\n')
    p.append('      // 改用 legendselectchanged 事件 + params.name 知道用户点的是哪个图例\n')
    p.append('      // 然后用 dispatchAction 取消其他 8 个 theta（保留用户点的）\n')
    p.append('      (function(myChart){\n')
    p.append('        myChart.on("legendselectchanged", function(params){\n')
    p.append('          var sel = params.selected || (myChart.getOption().legend[0] || {}).selected;\n')
    p.append('          if (!sel) return;\n')
    p.append('          // 用户点的图例名（K线/RA-PLR/某个 θ）\n')
    p.append('          var clickedName = params.name;\n')
    p.append('          // 统计当前选中的 theta\n')
    p.append('          var thetaKeys = [];\n')
    p.append('          var firstTheta = null;\n')
    p.append('          for (var k in sel) {\n')
    p.append('            if (k.indexOf("预测顶/底(θ=") === 0) {\n')
    p.append('              thetaKeys.push(k);\n')
    p.append('              if (sel[k] && !firstTheta) firstTheta = k;\n')
    p.append('            }\n')
    p.append('          }\n')
    p.append('          // 决定要保留哪个 theta：\n')
    p.append('          //   1. 如果用户点的就是某个预测顶底，就保留它\n')
    p.append('          //   2. 否则（用户点的是 K线/RA-PLR），保留当前已选中的第一个 theta\n')
    p.append('          var keep = null;\n')
    p.append('          if (clickedName && clickedName.indexOf("预测顶/底(θ=") === 0 && sel[clickedName]) {\n')
    p.append('            keep = clickedName;\n')
    p.append('          } else {\n')
    p.append('            keep = firstTheta;\n')
    p.append('          }\n')
    p.append('          // 如果当前多于 1 个 theta 选中，取消其他（保留 keep）\n')
    p.append('          var count = 0;\n')
    p.append('          for (var i = 0; i < thetaKeys.length; i++) if (sel[thetaKeys[i]]) count++;\n')
    p.append('          if (count > 1 && keep) {\n')
    p.append('            for (var i = 0; i < thetaKeys.length; i++) {\n')
    p.append('              var k = thetaKeys[i];\n')
    p.append('              if (k !== keep && sel[k]) {\n')
    p.append('                myChart.dispatchAction({type:"legendUnSelect", name:k});\n')
    p.append('              }\n')
    p.append('            }\n')
    p.append('          }\n')
    p.append('        });\n')
    p.append('      })(chart);\n')
    p.append('      // 注意：不再用 setInterval 兜底，因为会和用户点击冲突\n')
    p.append('    }\n')
    p.append('  })();\n')
    p.append('\n')
    p.append('  // 滚动监听：目录高亮\n')
    p.append('  (function(){\n')
    p.append('    var tocLinks = Array.from(document.querySelectorAll("aside.toc a"));\n')
    p.append('    var map = new Map();\n')
    p.append('    tocLinks.forEach(function(a){ var h = a.getAttribute("href"); if (h) map.set(h.slice(1), a); });\n')
    p.append('    var sections = Array.from(document.querySelectorAll("[id]")).filter(function(el){ return map.has(el.id); });\n')
    p.append('    function onScroll(){\n')
    p.append('      var y = window.scrollY + 80;\n')
    p.append('      var active = sections.length ? sections[0].id : null;\n')
    p.append('      for (var i = 0; i < sections.length; i++) if (sections[i].offsetTop <= y) active = sections[i].id;\n')
    p.append('      tocLinks.forEach(function(a){ a.classList.remove("active"); });\n')
    p.append('      if (active && map.has(active)) map.get(active).classList.add("active");\n')
    p.append('    }\n')
    p.append('    window.addEventListener("scroll", onScroll, {passive:true});\n')
    p.append('    onScroll();\n')
    p.append('  })();\n')
    p.append('})();\n')
    p.append('</script>\n')
    js = ''.join(p)

    html = (
        '<!DOCTYPE html>\n'
        '<html lang="zh-CN"><head>\n'
        '<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>\n'
        '<title>TURNLAB · Batch-MCRNN 报告（v4 · 拟合度优先 / 早停 / 9 θ 候选图例）</title>\n'
        + echarts_tag + '\n'
        '<style>' + css + '</style></head><body>\n'
        + body_html + '\n'
        + js + '\n'
        '</body></html>\n'
    )
    return html


def main():
    xlsx_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\Documents\xwechat_files\wxid_qp00o1zxtoci22_d426\msg\file\2026-09"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SCRIPT_DIR, "TURNLAB_BatchMCRNN_v2.html")
    data = run_pipeline(xlsx_dir)
    html = build_html(data)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("=" * 60)
    print(f"报告: {out}  ({os.path.getsize(out)/1024:.0f} KB)")
    print(f"平均 AUC={data['aggregate']['avgAuc']}  F1={data['aggregate']['avgF1']}  MCC={data['aggregate']['avgMcc']}")
    print("=" * 60)


def run_with_files(file_paths, output_html, name_map=None, log_fn=None):
    """供 GUI / 脚本调用的高级入口：传入文件路径列表，输出 HTML 文件路径。

    参数
    ----
    file_paths : list[str]
        期货 xlsx/csv 文件路径列表（按用户拖入顺序）
    output_html : str
        生成的 HTML 报告路径
    name_map : dict[str, (str, str)] | None
        可选：symbol -> (中文名, 交易所)。不传则从文件名推断。
    log_fn : callable | None
        可选：日志回调 (str) -> None，用于 GUI 实时显示。

    返回
    ----
    dict : run_pipeline 返回的 data 字典（含 aggregate / symbols 等）
    """
    if log_fn is None:
        log_fn = print
    if name_map is not None:
        # 把 name_map 注入到 run_pipeline 的 symbols 推断里
        # 通过 monkey-patch 实现
        import builtins
        orig_infer = globals()["_infer_symbol_from_path"]

        def patched_infer(path):
            sym = orig_infer(path)
            # 找到对应的中文名
            return sym
        # 直接在 run_pipeline 内部用不到 name_map，我们换种方式：
        # 让 GUI 调用方先把 name_map 通过 path 注入。
        # 为简单起见，run_pipeline 已经能跑通，name_map 在 GUI 层补上。
        pass

    data = run_pipeline(file_paths=file_paths)
    # 如果 GUI 传入了 name_map，覆盖默认的 (sym, exchange) 显示
    if name_map:
        for sym_obj in data["symbols"]:
            s = sym_obj["symbol"]
            if s in name_map:
                cn, ex = name_map[s]
                sym_obj["name"] = cn
                sym_obj["exchange"] = ex
        for r in data["results"]:
            s = r["symbol"]
            if s in name_map:
                cn, ex = name_map[s]
                r["name"] = cn
        # HTML 模板里的 N_SYM 渲染会用到 data["model"]["nSymbols"]，但 name_map
        # 不影响结构，只是显示文本。

    html = build_html(data)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    log_fn(f"  ✅ HTML 已生成: {output_html}  ({os.path.getsize(output_html)/1024:.0f} KB)\n")
    # v4 增量: 输出"最近 2 天阶段性高/低点"提示（前 8 个内盘品种）
    # 改用从写盘后的 HTML 文件读 DATA 块作为源真相（避免 data dict 与 HTML 不一致）
    # 单次 log_fn 调用（一次性插入完整文本），确保 tkinter Text widget 的 \n 换行生效
    # （多次 splitlines 循环调 log_fn 在某些 tkinter 配置下会"折叠"成一行）
    try:
        signals_text = _format_recent_signals_from_html(output_html, theta="0.1")
        log_fn(signals_text + "\n")
    except Exception as e:
        log_fn(f"  ⚠ 阶段性提示生成失败: {e}\n")
    return data


def _format_recent_signals_from_html(html_path, theta="0.1"):
    """从写盘后的 HTML 文件读 DATA 块，生成阶段性高/低点提示文本。

    优势：HTML 是源真相（写盘后不再变化），即使 build_html 后续修改了 data，
    或 _format_recent_signals 在 build_html 之后才执行，依然能基于最终内容生成。
    """
    import re
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    m = re.search(r'var\s+DATA\s*=\s*(\{.*?\});\s*\n', html, re.DOTALL)
    if not m:
        raise ValueError(f"未在 {html_path} 中找到 DATA 块")
    data = json.loads(m.group(1).replace('\\u003c', '<'))
    return _format_recent_signals(data, theta=theta)


def _format_recent_signals(data, theta="0.1"):
    """生成"截至昨天数据"的前 8 个内盘品种阶段性高/低点提示文本。

    规则：
      - 取 data['symbols'] 中每个品种的最后 2 个交易日（dates[-2:] = 昨天/今天）
      - θ 阈值筛出的高点/低点
      - 同一品种昨天+今天都有高/低点时，仅取今天的
      - 输出格式：截至X月X日（系统当前日期的前一天）数据 + 1.~8. 豆一/豆二/.../豆油 + 备注
    """
    from datetime import date, timedelta

    # 用户指定的前 8 个内盘品种（按目录中顺序：豆一→豆二→豆粕→菜油→棕榈油→菜粕→白糖→豆油）
    # （简称, 品种 code）
    CN_ORDER = [
        ("豆一",     "A.DCE"),
        ("豆二",     "B.DCE"),
        ("豆粕",     "M.DCE"),
        ("菜油",     "OI.CZC"),
        ("棕榈油", "P.DCE"),
        ("菜粕",     "RM.CZC"),
        ("白糖",     "SR.CZC"),
        ("豆油",     "Y.DCE"),
    ]

    sym_map = {s["symbol"]: s for s in data.get("symbols", [])}
    lines = []
    for idx, (cn, code) in enumerate(CN_ORDER, start=1):
        s = sym_map.get(code)
        if not s:
            lines.append(f"{idx}.{cn}暂无阶段性高点和低点提示")
            continue
        dates = s.get("dates", [])
        bucket = (s.get("pred_by_theta") or {}).get(theta) or {}
        highs = bucket.get("highs") or []
        lows = bucket.get("lows") or []
        if len(dates) < 2:
            lines.append(f"{idx}.{cn}暂无阶段性高点和低点提示")
            continue
        d1 = dates[-2]   # 昨天
        d2 = dates[-1]   # 今天
        # value 形如 [idx_in_win, price]，用 idx_in_win 拿对应日期
        def find_in(items, target_date):
            for it in items:
                if it.get("value") and len(it["value"]) >= 1:
                    if dates[it["value"][0]] == target_date:
                        return it
            return None
        high_d1 = find_in(highs, d1)
        high_d2 = find_in(highs, d2)
        low_d1 = find_in(lows, d1)
        low_d2 = find_in(lows, d2)
        # 规则：若昨日今日都有，仅取今日
        recent_high = high_d2 or high_d1
        recent_low = low_d2 or low_d1
        # 组装
        if recent_high is None and recent_low is None:
            text = "暂无阶段性高点和低点提示"
        elif recent_high is not None and recent_low is not None:
            # 同时有高点和低点：同一天或不同天都合并输出
            text = (f"显示{int(round(recent_high['value'][1]))}附近到阶段性高点，"
                    f"显示{int(round(recent_low['value'][1]))}附近到阶段性低点")
        elif recent_high is not None:
            text = f"显示{int(round(recent_high['value'][1]))}附近到阶段性高点"
        else:
            text = f"显示{int(round(recent_low['value'][1]))}附近到阶段性低点"
        lines.append(f"{idx}.{cn}{text}")

    # 截至日期 = 昨天（v4 改：去掉"（系统当前日期的前一天）"——按用户期望简化）
    yesterday = date.today() - timedelta(days=1)
    header = f"\n截至{yesterday.month}月{yesterday.day}日数据：\n"
    body = "\n".join(lines)
    remark = ("\n备注：(1.上述数据点位是合约主连数据；"
              "2.阶段性高点代表多单减仓或空单开仓点位，"
              "阶段性低点代表空单减仓或多单开仓点位，"
              "阶段性高点和低点表示点位判断，不是趋势判断）")
    return header + body + remark


if __name__ == "__main__":
    main()
