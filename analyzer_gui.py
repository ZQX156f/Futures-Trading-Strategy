#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyzer_gui.py
Batch-MCRNN 多品种联合训练分析器 · GUI
- 批量拖入 xlsx（一次可多个），默认 12 品种
- 动态调整期望品种数（可手动加/减）
- 开始分析 → 训练 + HTML 报告 + 12 张 K 线长截图
- 自动用默认浏览器打开 HTML

依赖：
  必装：tkinter（Python 自带）、pandas、numpy、torch、sklearn
  可选：
    pip install tkinterdnd2      # 拖放支持
    pip install playwright && playwright install chromium   # K 线长截图
"""
import os, re, sys, threading, traceback, webbrowser, time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

# ======================== 可选依赖 ========================
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    HAS_DND = False

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# 核心算法
import batch_mcrnn_full_report_v2 as core


# ======================== 工具函数 ========================
def infer_symbol_from_path(path):
    """从文件路径提取品种 code（XX.YYY 形式）。"""
    base = os.path.basename(str(path))
    m = re.search(r'([A-Za-z]{1,3}\.[A-Za-z]{2,4})', base)
    if m:
        return m.group(1).upper()
    base_upper = os.path.splitext(base)[0].upper().replace(' ', '_')
    return base_upper


def exchange_from_code(code):
    suf = code.split('.')[-1].upper() if '.' in code else ''
    return {'DCE': 'DCE', 'CZC': 'CZCE', 'SHF': 'SHFE',
            'CBT': 'CBOT', 'NYB': 'ICE', 'CME': 'CME',
            'LME': 'LME', 'NYM': 'NYMEX'}.get(suf, suf or '—')


# 中文名（常用品种）
CN_NAME = {
    'SR.CZC': '郑商所白糖', 'SB.NYB': 'ICE 11号糖',
    'Y.DCE': '大商所豆油', 'OI.CZC': '郑商所菜油', 'BO.CBT': 'CBOT 豆油',
    'P.DCE': '大商所棕榈油', 'M.DCE': '大商所豆粕', 'RM.CZC': '郑商所菜粕',
    'SM.CBT': 'CBOT 豆粕', 'S.CBT': 'CBOT 大豆',
    'A.DCE': '大商所豆一', 'B.DCE': '大商所豆二',
    'CF.CZC': '郑商所棉花', 'CU.SHF': '上期所铜', 'AL.SHF': '上期所铝',
    'ZN.SHF': '上期所锌', 'AU.SHF': '上期所黄金', 'AG.SHF': '上期所白银',
    'I.DCE': '大商所铁矿石', 'J.DCE': '大商所焦炭', 'JM.DCE': '大商所焦煤',
    'TA.CZC': '郑商所PTA', 'MA.CZC': '郑商所甲醇', 'FG.CZC': '郑商所玻璃',
}


def display_name(path):
    """根据文件路径返回 (code, 中文名, 交易所) 供 UI 显示。"""
    code = infer_symbol_from_path(path)
    cn = CN_NAME.get(code, code)
    ex = exchange_from_code(code)
    return code, cn, ex


# ======================== K 线长截图 ========================
def screenshot_kline_chapter(html_path, out_png, viewport_w=1800, wait_sec=8.0):
    """用 playwright 截 HTML 报告第七章（K 线图）区域，输出长图 PNG。

    实现要点（v4 适配）：
      1. **关键：先开大 viewport 让 body 不出水平滚动条**——v4 .wrap{max-width:1500} + sidebar 240 + chart min-width 1380 + 边距
         实际页面 scrollWidth ≈ 1667-1800px，viewport 1500/1600 都会出现水平滚动条，导致右边
         167-300px 被截掉。viewport 默认设 1800。
      2. 用 page.evaluate 拿 body.scrollWidth（页面真实宽度）作为 viewport_w 实际值，
         防止任何 css 改宽后再次截不全。
      3. viewport_h 起步 12000，根据 sec-7.scrollHeight 自适应扩。
      4. scrollIntoView 把 #sec-7 滚到视口顶部（block:'start'），让 box.y 接近 0
      5. wait_sec 给够 8 秒，让 echarts 充分渲染 12 品种
    """
    if not HAS_PLAYWRIGHT:
        return None, '未安装 playwright. 安装: pip install playwright && playwright install chromium'
    try:
        html_url = 'file:///' + os.path.abspath(html_path).replace('\\', '/')
        out_png = os.path.abspath(out_png)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            # 第一次：用传入的 viewport 探测页面真实宽度
            page = browser.new_page(viewport={'width': viewport_w, 'height': 12000})
            page.goto(html_url, wait_until='domcontentloaded', timeout=120000)
            time.sleep(wait_sec)
            sec = page.query_selector('#sec-7')
            if not sec:
                browser.close()
                return None, '未找到 #sec-7 元素（报告结构异常）'
            # 关键：量 body 真实 scrollWidth（含溢出内容），让 viewport ≥ 它 + buffer
            real_w = page.evaluate(
                "Math.max(document.body.scrollWidth, document.documentElement.scrollWidth, window.innerWidth)"
            )
            cur_vw = max(viewport_w, int(real_w) + 50)
            if cur_vw > viewport_w:
                # 实际页面更宽，需要扩 viewport
                page.set_viewport_size({"width": cur_vw, "height": 12000})
                time.sleep(0.3)  # 让布局重新计算
            # 量 sec-7 实际 scrollHeight（不受 viewport 限制）
            sec_h = page.evaluate("document.getElementById('sec-7').scrollHeight")
            cur_vh = 12000
            if sec_h and sec_h + 200 > cur_vh:
                cur_vh = int(sec_h + 400)
                page.set_viewport_size({"width": cur_vw, "height": cur_vh})
                time.sleep(0.3)
            # 滚到 sec-7 顶部
            page.evaluate("document.getElementById('sec-7').scrollIntoView({block:'start'})")
            time.sleep(0.5)
            box = sec.bounding_box()
            if not box:
                browser.close()
                return None, '#sec-7 没有 bounding box（可能还未渲染）'
            # 最终校验：box.y + box.height 必须 < viewport.height
            need = box['y'] + box['height']
            if need > cur_vh:
                cur_vh = int(need + 200)
                page.set_viewport_size({"width": cur_vw, "height": cur_vh})
                time.sleep(0.3)
                page.evaluate("document.getElementById('sec-7').scrollIntoView({block:'start'})")
                time.sleep(0.5)
                box = sec.bounding_box()
                if not box:
                    browser.close()
                    return None, '#sec-7 bounding box 重读失败'
            # 截 sec-7
            page.screenshot(path=out_png, clip={
                'x': box['x'], 'y': box['y'],
                'width': box['width'], 'height': box['height'],
            })
            browser.close()
            return out_png, None
    except Exception as e:
        return None, f'截图失败: {e}'


# ======================== 文件列表组件 ========================
class FileListFrame(tk.Frame):
    """文件列表：显示每个文件的 序号 / 品种 code / 中文名 / 交易所 / 文件名 / 行数"""

    def __init__(self, parent, on_change):
        super().__init__(parent, bg='#1e293b', highlightbackground='#334155',
                         highlightthickness=2, bd=0)
        self.on_change = on_change
        self.file_paths = []  # 完整路径列表（保持顺序）

        tk.Label(self, text='已加载文件（一次拖入多个）', bg='#1e293b', fg='#f1f5f9',
                 font=('Microsoft YaHei UI', 11, 'bold')).pack(pady=(12, 6))

        # 列表 + 滚动条
        list_frame = tk.Frame(self, bg='#1e293b')
        list_frame.pack(padx=15, pady=5, fill='both', expand=True)

        self.tree = ttk.Treeview(list_frame, columns=('idx', 'code', 'cn', 'ex', 'name', 'rows'),
                                 show='headings', height=10)
        self.tree.heading('idx', text='#')
        self.tree.heading('code', text='品种 code')
        self.tree.heading('cn', text='中文名')
        self.tree.heading('ex', text='交易所')
        self.tree.heading('name', text='文件名')
        self.tree.heading('rows', text='行数')
        self.tree.column('idx', width=40, anchor='center')
        self.tree.column('code', width=80, anchor='center')
        self.tree.column('cn', width=140, anchor='w')
        self.tree.column('ex', width=70, anchor='center')
        self.tree.column('name', width=240, anchor='w')
        self.tree.column('rows', width=70, anchor='center')

        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # 配置颜色
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview', background='#0b1220', fieldbackground='#0b1220',
                        foreground='#cbd5e1', rowheight=24, font=('Consolas', 9))
        style.configure('Treeview.Heading', background='#334155', foreground='#f1f5f9',
                        font=('Microsoft YaHei UI', 9, 'bold'))
        style.map('Treeview', background=[('selected', '#2563eb')])

        # 按钮行
        btn_frame = tk.Frame(self, bg='#1e293b')
        btn_frame.pack(pady=(5, 12))
        tk.Button(btn_frame, text='➕ 添加文件', font=('Microsoft YaHei UI', 9),
                  bg='#475569', fg='white', activebackground='#334155',
                  activeforeground='white', command=self.choose_files,
                  bd=0, padx=12, pady=4, cursor='hand2').pack(side='left', padx=3)
        tk.Button(btn_frame, text='🗑 删除选中', font=('Microsoft YaHei UI', 9),
                  bg='#475569', fg='white', activebackground='#334155',
                  activeforeground='white', command=self.delete_selected,
                  bd=0, padx=12, pady=4, cursor='hand2').pack(side='left', padx=3)
        tk.Button(btn_frame, text='🧹 全部清空', font=('Microsoft YaHei UI', 9),
                  bg='#475569', fg='white', activebackground='#334155',
                  activeforeground='white', command=self.clear,
                  bd=0, padx=12, pady=4, cursor='hand2').pack(side='left', padx=3)

        # 拖放（整个区域）
        if HAS_DND:
            self.drop_target_register(DND_FILES)
            self.dnd_bind('<<Drop>>', self.on_drop)
            self.dnd_bind('<<DropEnter>>', lambda e: self.config(highlightbackground='#22c55e'))
            self.dnd_bind('<<DropLeave>>', lambda e: self.config(highlightbackground='#334155'))

        self._empty_label = None
        self._update_empty()

    def _update_empty(self):
        if not self.file_paths and self._empty_label is None:
            self._empty_label = tk.Label(self.tree, text='（空，请拖入 .xlsx 或点击 ➕ 添加文件）',
                                         bg='#0b1220', fg='#64748b',
                                         font=('Microsoft YaHei UI', 10))
        elif self.file_paths and self._empty_label is not None:
            self._empty_label.destroy()
            self._empty_label = None

    def choose_files(self):
        paths = filedialog.askopenfilenames(
            title='选择期货 .xlsx（可多选）',
            filetypes=[('Excel', '*.xlsx'), ('CSV', '*.csv'), ('All', '*.*')])
        if paths:
            self.add_files(list(paths))

    def on_drop(self, event):
        # tkinterdnd2 多文件 data 格式："{path1} {path2} ..."
        data = event.data
        # 处理花括号包围
        if data.startswith('{'):
            data = data.strip('{}')
        # 拆分（路径里可能有空格所以不能直接 split）
        paths = re.findall(r'\{([^}]+)\}|(\S+)', data)
        paths = [a or b for a, b in paths]
        if paths:
            self.add_files(paths)

    def add_files(self, paths):
        added = 0
        skipped = 0
        for p in paths:
            if not os.path.isfile(p):
                skipped += 1
                continue
            if p in self.file_paths:
                skipped += 1
                continue
            # 简单校验扩展名
            if not (p.lower().endswith('.xlsx') or p.lower().endswith('.csv')):
                skipped += 1
                continue
            self.file_paths.append(p)
            added += 1
        self._refresh()
        self._update_empty()
        self.on_change()
        return added, skipped

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo('提示', '请先在列表中选中要删除的行')
            return
        # 从后往前删
        idx_to_del = sorted([int(self.tree.item(s)['values'][0]) - 1 for s in sel], reverse=True)
        for i in idx_to_del:
            if 0 <= i < len(self.file_paths):
                del self.file_paths[i]
        self._refresh()
        self._update_empty()
        self.on_change()

    def clear(self):
        self.file_paths = []
        self._refresh()
        self._update_empty()
        self.on_change()

    def _refresh(self):
        # 清空
        for item in self.tree.get_children():
            self.tree.delete(item)
        # 填回
        for i, p in enumerate(self.file_paths, start=1):
            code, cn, ex = display_name(p)
            base = os.path.basename(p)
            rows = self._try_get_rows(p)
            self.tree.insert('', 'end', values=(i, code, cn, ex, base, rows))

    def _try_get_rows(self, p):
        """快速读 xlsx 的行数（不解析，只用 openpyxl 读 length）。失败返回 '—'。"""
        try:
            if p.lower().endswith('.xlsx'):
                # 用 openpyxl 只读维度
                from openpyxl import load_workbook
                wb = load_workbook(p, read_only=True, data_only=True)
                ws = wb.active
                n = ws.max_row
                wb.close()
                return str(n - 1) if n > 0 else '0'
            else:
                df = pd.read_csv(p)
                return str(len(df))
        except Exception:
            return '—'

    def get_paths(self):
        return list(self.file_paths)


# ======================== 主程序 ========================
class App:
    def __init__(self, root):
        self.root = root
        root.title('Batch-MCRNN · 多品种联合训练分析器')
        root.geometry('1080x780')
        root.configure(bg='#0f172a')
        root.minsize(960, 680)

        # 标题
        tk.Label(root, text='🌾 Batch-MCRNN · 多品种联合训练分析器',
                 bg='#0f172a', fg='#f1f5f9',
                 font=('Microsoft YaHei UI', 17, 'bold')).pack(pady=(18, 3))
        tk.Label(root, text='一次拖入多份期货 xlsx  ·  联合训练 + 早停  ·  无交易  ·  HTML 报告 + K 线长截图',
                 bg='#0f172a', fg='#94a3b8',
                 font=('Microsoft YaHei UI', 10)).pack(pady=(0, 12))

        # 期望品种数控制
        ctrl_top = tk.Frame(root, bg='#0f172a')
        ctrl_top.pack(pady=(0, 8), padx=20, fill='x')
        tk.Label(ctrl_top, text='期望品种数：', bg='#0f172a', fg='#cbd5e1',
                 font=('Microsoft YaHei UI', 10)).pack(side='left')
        self.spin = tk.Spinbox(ctrl_top, from_=1, to=24, width=6, justify='center',
                               font=('Consolas', 11), bg='#1e293b', fg='#f1f5f9',
                               buttonbackground='#334155', relief='flat', bd=0)
        self.spin.delete(0, 'end')
        self.spin.insert(0, '12')
        self.spin.pack(side='left', padx=5)
        tk.Label(ctrl_top, text='（拖入文件后实际数会更新；范围 1~24）',
                 bg='#0f172a', fg='#64748b',
                 font=('Microsoft YaHei UI', 9)).pack(side='left', padx=5)

        self.count_label = tk.Label(ctrl_top, text='已加载：0', bg='#0f172a', fg='#22c55e',
                                    font=('Microsoft YaHei UI', 10, 'bold'))
        self.count_label.pack(side='right')

        # 文件列表
        self.list_frame = FileListFrame(root, on_change=self._on_files_change)
        self.list_frame.pack(padx=20, pady=8, fill='both', expand=True)

        # 按钮区
        btn_frame = tk.Frame(root, bg='#0f172a')
        btn_frame.pack(pady=12)
        self.run_btn = tk.Button(btn_frame, text='▶  开始分析',
                                 font=('Microsoft YaHei UI', 13, 'bold'),
                                 bg='#3b82f6', fg='white', activebackground='#2563eb',
                                 activeforeground='white', command=self.run_analysis,
                                 width=18, height=1, bd=0, cursor='hand2')
        self.run_btn.pack(side='left', padx=6)
        self.open_btn = tk.Button(btn_frame, text='🌐 打开输出目录',
                                  font=('Microsoft YaHei UI', 10),
                                  bg='#475569', fg='white', activebackground='#334155',
                                  activeforeground='white',
                                  command=self.open_output_dir,
                                  width=14, bd=0, cursor='hand2', state='disabled')
        self.open_btn.pack(side='left', padx=6)

        # 日志区
        tk.Label(root, text='运行日志：', bg='#0f172a', fg='#cbd5e1', anchor='w',
                 font=('Microsoft YaHei UI', 10)).pack(fill='x', padx=20, pady=(8, 0))
        log_frame = tk.Frame(root, bg='#0f172a')
        log_frame.pack(pady=(2, 15), padx=20, fill='both', expand=False)
        self.log_text = tk.Text(log_frame, bg='#0b1220', fg='#cbd5e1',
                                font=('Consolas', 9), height=12, wrap='word',
                                insertbackground='#cbd5e1')
        log_scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scroll.pack(side='right', fill='y')

        # 初始提示
        self.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n')
        self.log('提示：\n')
        self.log('  1. 拖入 12 份期货 .xlsx（一次可拖多个），或点 ➕ 添加文件\n')
        self.log('  2. 文件名里需含品种 code（如 SR.CZC.xlsx、Y.DCE(1).xlsx）\n')
        self.log('  3. 点 ▶ 开始分析 → 训练 + HTML + K线长截图\n')
        if not HAS_DND:
            self.log('  ⚠️  tkinterdnd2 未安装, 拖放不可用, 请用"➕ 添加文件"按钮\n')
            self.log('     安装: pip install tkinterdnd2\n')
        if not HAS_PLAYWRIGHT:
            self.log('  ⚠️  playwright 未安装, K线长截图不可用, 仅生成 HTML\n')
            self.log('     安装: pip install playwright && playwright install chromium\n')
        self.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n')

        # 输出目录
        self.output_dir = os.path.join(SCRIPT_DIR_DEFAULT := os.path.dirname(os.path.abspath(__file__)))
        self.last_html = None
        self.last_png = None

    def _on_files_change(self):
        n = len(self.list_frame.get_paths())
        self.count_label.config(text=f'已加载：{n}')
        # 同步 spin
        if n > 0:
            self.spin.delete(0, 'end')
            self.spin.insert(0, str(n))

    def log(self, msg):
        self.log_text.insert('end', msg)
        self.log_text.see('end')
        self.root.update_idletasks()

    def open_output_dir(self):
        path = self.output_dir
        if not os.path.isdir(path):
            messagebox.showerror('错误', f'目录不存在: {path}')
            return
        try:
            if sys.platform.startswith('win'):
                os.startfile(path)
            elif sys.platform.startswith('darwin'):
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception as e:
            messagebox.showerror('错误', str(e))

    def run_analysis(self):
        paths = self.list_frame.get_paths()
        if not paths:
            messagebox.showerror('错误', '请先拖入或选择期货 xlsx 文件')
            return
        try:
            n_want = int(self.spin.get())
        except ValueError:
            n_want = len(paths)
        if len(paths) < n_want:
            if not messagebox.askyesno('确认', f'期望 {n_want} 个品种，但只加载了 {len(paths)} 个。\n按 {len(paths)} 个继续？'):
                return
        elif len(paths) > n_want:
            self.log(f'  ⚠️  加载了 {len(paths)} 个文件，多于期望 {n_want}，将使用全部 {len(paths)} 个\n')

        # 禁用按钮
        self.run_btn.config(state='disabled', text='⏳ 分析中...')
        self.open_btn.config(state='disabled')
        threading.Thread(target=self._do_analysis, args=(paths,),
                         daemon=True).start()

    def _do_analysis(self, paths):
        class StdoutRedirect:
            def __init__(self, log_fn): self.log = log_fn
            def write(self, s):
                if s and s.strip():
                    try: self.log(s)
                    except: pass
            def flush(self): pass

        old_stdout = sys.stdout
        sys.stdout = StdoutRedirect(self.log)
        out_html = None
        try:
            # 生成输出文件名
            ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            out_html = os.path.join(self.output_dir, f'BatchMCRNN_{len(paths)}品种_{ts}.html')
            self.log(f'\n▶ 开始分析 · 共 {len(paths)} 个品种\n')
            self.log(f'  输出: {out_html}\n\n')

            # 调核心算法
            data = core.run_with_files(paths, output_html=out_html, log_fn=self.log)

            # 报告统计
            agg = data['aggregate']
            self.log(f'\n━━━━━━━━━━ 分析结果 ━━━━━━━━━━\n')
            self.log(f'  平均 AUC    = {agg["avgAuc"]:.3f}\n')
            self.log(f'  平均 F1     = {agg["avgF1"]:.3f}\n')
            self.log(f'  平均 MCC    = {agg["avgMcc"]:.3f}\n')
            self.log(f'  Pool AUC    = {agg["poolAuc"]:.3f}\n')
            self.log(f'  Pool F1     = {agg["poolF1"]:.3f}\n')
            sel = data['model'].get('selectionResult')
            es = data['model'].get('earlyStopping')
            if sel:
                # v2 · 论文方法：按验证集收益选参
                self.log(f'  θ (验证收益) = {data["split"]["threshold"]}\n')
                self.log(f'  选参 (论文)  = 快照 ep{sel.get("snapEpoch")}, '
                         f'θ={sel.get("theta")} α={sel.get("alpha")} β={sel.get("beta")} '
                         f'RSI={sel.get("rsiBuy")}/{sel.get("rsiSell")}\n')
                self.log(f'  验证集收益   = {sel.get("valReturn", 0)*100:+.2f}% '
                         f'(trades={sel.get("valTrades")}, win={sel.get("valWinRate",0)*100:.0f}%, '
                         f'maxDD={sel.get("valMaxDd",0)*100:.1f}%)\n')
                self.log(f'  测试集平均收益= {agg.get("avgTestReturn", 0)*100:+.2f}% '
                         f'(trades={agg.get("totalTestTrades", 0)}, '
                         f'win={agg.get("avgTestWinRate",0)*100:.0f}%)\n')
            elif es:
                # v1 · 旧式 F1 早停（兼容老 HTML）
                self.log(f'  θ (验证F1)  = {data["split"]["threshold"]}\n')
                self.log(f'  早停        = {es["triggered"]} '
                         f'(最佳 ep{es["bestEpoch"]}, val F1={es["bestValF1"]:.3f})\n')
            self.log(f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n')

            # 截图 K 线长图
            if HAS_PLAYWRIGHT:
                out_png = os.path.splitext(out_html)[0] + '_klines.png'
                self.log('\n📸 正在截 K 线长图（第 7 章）...\n')
                png, err = screenshot_kline_chapter(out_html, out_png, wait_sec=5.0)
                if err:
                    self.log(f'   ⚠️  截图失败: {err}\n')
                else:
                    self.log(f'   ✅ K线长图: {png}\n')
                    self.log(f'      大小: {os.path.getsize(png)/1024:.1f} KB\n')
                    self.last_png = png
            else:
                self.log('\n⚠️  未安装 playwright, 跳过 K 线长截图\n')

            # 自动开浏览器
            self.log(f'\n🌐 已在默认浏览器中打开 HTML\n')
            webbrowser.open('file://' + os.path.abspath(out_html).replace('\\', '/'))

            self.last_html = out_html
            messagebox.showinfo('完成', f'分析完成！\n\n报告: {out_html}\n\n'
                                 f'平均 AUC={agg["avgAuc"]:.3f}, F1={agg["avgF1"]:.3f}, MCC={agg["avgMcc"]:.3f}')

        except Exception as e:
            self.log(f'\n❌ 错误: {e}\n')
            traceback.print_exc(file=sys.stdout)
            messagebox.showerror('错误', str(e))
        finally:
            sys.stdout = old_stdout
            self.root.after(0, lambda: self.run_btn.config(state='normal', text='▶  开始分析'))
            if self.last_html:
                self.root.after(0, lambda: self.open_btn.config(state='normal'))


# ======================== 入口 ========================
if __name__ == '__main__':
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    try:
        app = App(root)
        root.mainloop()
    except Exception as e:
        print(f'启动失败: {e}', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
