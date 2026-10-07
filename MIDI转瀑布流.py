import os
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import mido

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


PIANO_MIN = 21   # A0
PIANO_MAX = 108  # C8


def is_black_pitch(pitch: int) -> bool:
    # 黑键：C#, D#, F#, G#, A#
    return (pitch % 12) in (1, 3, 6, 8, 10)


def parse_global_tempo_events(mid: mido.MidiFile):
    """从合并后的全局时间线提取 set_tempo 事件，得到 (abs_tick, tempo) 列表。"""
    tempo = 500000
    tempo_events = [(0, tempo)]
    abs_tick = 0

    merged = mido.merge_tracks(mid.tracks)
    for msg in merged:
        if msg.time:
            abs_tick += msg.time
        if msg.type == "set_tempo":
            tempo = msg.tempo
            tempo_events.append((abs_tick, tempo))

    tempo_events = sorted(set(tempo_events), key=lambda x: x[0])
    if tempo_events and tempo_events[0][0] != 0:
        tempo_events.insert(0, (0, 500000))
    if not tempo_events:
        tempo_events = [(0, 500000)]
    return tempo_events


def build_tick_to_sec_segments(total_ticks: int, tpb: int, tempo_events):
    """将 tempo 变化分段，便于 tick→秒 的快速换算。"""
    events = sorted(tempo_events, key=lambda x: x[0])
    if not events or events[0][0] != 0:
        events = [(0, 500000)] + events

    segments = []
    sec = 0.0

    for i, (tick0, tempo) in enumerate(events):
        tick1 = events[i + 1][0] if i + 1 < len(events) else total_ticks
        tick1 = min(tick1, total_ticks)
        if tick1 < tick0:
            continue
        segments.append({"tick0": tick0, "tick1": tick1, "sec0": sec, "tempo": tempo})
        sec += mido.tick2second(tick1 - tick0, tpb, tempo)

    if not segments:
        segments = [{"tick0": 0, "tick1": total_ticks, "sec0": 0.0, "tempo": 500000}]
    return segments


def tick_to_sec(tick: int, segments, tpb: int) -> float:
    """在分段表里定位 tick 所属段并换算为秒。"""
    if tick <= 0:
        return 0.0
    for seg in segments:
        if seg["tick0"] <= tick <= seg["tick1"]:
            dt = tick - seg["tick0"]
            return seg["sec0"] + mido.tick2second(dt, tpb, seg["tempo"])
    last = segments[-1]
    return last["sec0"] + mido.tick2second(tick - last["tick0"], tpb, last["tempo"])


def parse_notes_by_track(midi_path: str):
    """
    解析 MIDI：按 track 作为声部输出音符列表
    Returns:
      notes: list of (track_idx, pitch, start_sec, end_sec, velocity)
      total_sec
    """
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    tempo_events = parse_global_tempo_events(mid)

    total_ticks = 0
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
        total_ticks = max(total_ticks, t)

    segments = build_tick_to_sec_segments(total_ticks, tpb, tempo_events)
    total_sec = tick_to_sec(total_ticks, segments, tpb)

    notes = []
    for ti, tr in enumerate(mid.tracks):
        abs_tick = 0
        active = {}  # (channel, pitch) -> list[(start_tick, vel)]，用栈处理同音重叠

        for msg in tr:
            if msg.time:
                abs_tick += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                ch = getattr(msg, "channel", 0)
                key = (ch, msg.note)
                active.setdefault(key, []).append((abs_tick, msg.velocity))

            elif msg.type in ("note_off", "note_on"):
                is_off = (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0)
                if not is_off:
                    continue

                ch = getattr(msg, "channel", 0)
                key = (ch, msg.note)
                stack = active.get(key)
                if stack:
                    st_tick, vel = stack.pop()
                    if not stack:
                        active.pop(key, None)
                    if abs_tick > st_tick:
                        st_sec = tick_to_sec(st_tick, segments, tpb)
                        en_sec = tick_to_sec(abs_tick, segments, tpb)
                        if en_sec > st_sec:
                            notes.append((ti, msg.note, st_sec, en_sec, vel))

        for (_ch, pitch), stack in active.items():
            for st_tick, vel in stack:
                if total_ticks > st_tick:
                    st_sec = tick_to_sec(st_tick, segments, tpb)
                    en_sec = total_sec
                    if en_sec > st_sec:
                        notes.append((ti, pitch, st_sec, en_sec, vel))

    return notes, total_sec


def build_variable_width_layout(white_w: float = 1.0, black_w: float = 0.9):
    """
    简单模式：每个 pitch 单独占一条竖向 lane，黑白键仅用不同宽度表达。
    Returns:
      layout[pitch] = (x_left, width, is_black)
      total_width
      octave_xs: 每个 C 的左边界 x
      white_bg, black_bg: (x,w) 列表
    """
    layout = {}
    x = 0.0
    octave_xs = []
    white_bg = []
    black_bg = []

    for pitch in range(PIANO_MIN, PIANO_MAX + 1):
        blk = is_black_pitch(pitch)
        w = black_w if blk else white_w
        layout[pitch] = (x, w, blk)

        if blk:
            black_bg.append((x, w))
        else:
            white_bg.append((x, w))

        if pitch % 12 == 0:  # C
            octave_xs.append(x)

        x += w

    return layout, x, octave_xs, white_bg, black_bg


def build_realistic_upper_layout():
    """
    真实比例模式（仅“键盘上半部分”顶面）：每个 pitch 仍是一条独立 lane，但白键会按顶面可见宽度分三类。
    规则（按钢琴顶面观感）：
      - 黑键统一较窄
      - D/G/A 这类白键左右都邻黑键，顶面更“细”
      - C/E/F/B 这类白键只有一侧邻黑键，顶面更“宽”
    """
    layout = {}
    x = 0.0
    octave_xs = []
    white_bg = []
    black_bg = []

    # 这些比例是“顶面观感”用的固定比例（不走 UI 输入）
    BLACK_W = 0.80
    WHITE_WIDE = 1.20
    WHITE_NARROW = 1.10

    narrow_white_pcs = {2, 7, 9}        # D, G, A
    wide_white_pcs = {0, 4, 5, 11}      # C, E, F, B

    for pitch in range(PIANO_MIN, PIANO_MAX + 1):
        pc = pitch % 12
        blk = is_black_pitch(pitch)

        if blk:
            w = BLACK_W
        else:
            if pc in narrow_white_pcs:
                w = WHITE_NARROW
            else:
                w = WHITE_WIDE

        layout[pitch] = (x, w, blk)

        if blk:
            black_bg.append((x, w))
        else:
            white_bg.append((x, w))

        if pc == 0:  # C
            octave_xs.append(x)

        x += w

    return layout, x, octave_xs, white_bg, black_bg


def make_track_colors(n_tracks: int):
    """瀑布流填充色：更浅一些，便于配合更明显的边框区分相邻音符。"""
    LIGHT_RED = (0.92, 0.60, 0.60, 1.0)
    LIGHT_BLUE = (0.62, 0.76, 0.94, 1.0)

    colors = []
    for i in range(max(1, n_tracks)):
        if i == 0:
            colors.append(LIGHT_BLUE)
        elif i == 1:
            colors.append(LIGHT_RED)
        else:
            g = 0.70 + 0.06 * (i % 4)
            g = float(np.clip(g, 0.65, 0.90))
            colors.append((g, g, g, 1.0))
    return colors


def render_waterfall_tracks_variablewidth(
    midi_path: str,
    out_path: str,
    fmt: str = "pdf",
    width_in: float = 14.0,
    dpi: int = 300,
    white_w: float = 1.0,
    black_w: float = 0.9,
    time_scale: float = 80.0,
    note_alpha: float = 0.90,
    edge_alpha: float = 0.95,
    edge_width: float = 1.00,
    velocity_affects_alpha: bool = False,
    show_key_borders: bool = True,
    octave_line_alpha: float = 0.65,
    octave_line_width: float = 1.5,
    use_realistic_upper_layout: bool = False,
):
    if time_scale <= 0:
        raise ValueError("纵向缩放必须 > 0")

    if not use_realistic_upper_layout:
        if white_w <= 0 or black_w <= 0:
            raise ValueError("白键宽 / 黑键宽必须 > 0")
        if black_w >= white_w:
            raise ValueError("黑键宽应小于白键宽")

    notes, total_sec = parse_notes_by_track(midi_path)

    filtered = []
    track_ids = set()
    for (ti, pitch, st_sec, en_sec, vel) in notes:
        if pitch < PIANO_MIN or pitch > PIANO_MAX:
            continue
        if en_sec <= st_sec:
            continue
        filtered.append((ti, pitch, st_sec, en_sec, vel))
        track_ids.add(ti)

    if not filtered:
        raise RuntimeError("没有可绘制的音符。")

    n_tracks = (max(track_ids) + 1) if track_ids else 1
    track_colors = make_track_colors(n_tracks)

    if use_realistic_upper_layout:
        layout, total_width, octave_xs, white_bg, black_bg = build_realistic_upper_layout()
    else:
        layout, total_width, octave_xs, white_bg, black_bg = build_variable_width_layout(
            white_w=white_w, black_w=black_w
        )

    total_y = total_sec * time_scale

    SCALED_SEC_PER_INCH = 60.0
    height_in = max(2.5, total_y / SCALED_SEC_PER_INCH)

    fig, ax = plt.subplots(figsize=(width_in, height_in), constrained_layout=False)

    white_rects = [Rectangle((x, 0), w, total_y) for (x, w) in white_bg]
    ax.add_collection(PatchCollection(
        white_rects,
        facecolor=[(0.975, 0.975, 0.975, 1.0)] * len(white_rects),
        edgecolor=(0.82, 0.82, 0.82, 1.0) if show_key_borders else "none",
        linewidth=0.45 if show_key_borders else 0.0,
        zorder=0
    ))

    black_rects = [Rectangle((x, 0), w, total_y) for (x, w) in black_bg]
    ax.add_collection(PatchCollection(
        black_rects,
        facecolor=[(0.90, 0.90, 0.90, 1.0)] * len(black_rects),
        edgecolor=(0.75, 0.75, 0.75, 1.0) if show_key_borders else "none",
        linewidth=0.40 if show_key_borders else 0.0,
        zorder=1
    ))

    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_y)

    for x in octave_xs:
        ax.axvline(x, linewidth=octave_line_width, alpha=octave_line_alpha, zorder=2)

    patches = []
    faces = []
    edges = []

    def inset_for(w: float) -> float:
        # 留出更清晰的分隔，尤其窄键时更容易看出相邻音符
        base = 0.11 if use_realistic_upper_layout else 0.10
        return min(0.10, base * w)

    for ti, pitch, st_sec, en_sec, vel in filtered:
        y = st_sec * time_scale
        h = (en_sec - st_sec) * time_scale
        if h <= 1e-6:
            continue

        x, w, _blk = layout[pitch]
        c = track_colors[ti % len(track_colors)]

        if velocity_affects_alpha:
            vel_a = float(np.clip(vel / 127.0, 0.30, 1.0))
            a = note_alpha * vel_a
        else:
            a = note_alpha

        face = (c[0], c[1], c[2], a)
        edge = (0.0, 0.0, 0.0, edge_alpha)

        inset = inset_for(w)
        rx = x + inset
        rw = max(1e-6, w - 2 * inset)

        patches.append(Rectangle((rx, y), rw, h))
        faces.append(face)
        edges.append(edge)

    ax.add_collection(PatchCollection(
        patches,
        facecolor=faces,
        edgecolor=edges,
        linewidth=edge_width,
        zorder=4
    ))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.margins(0, 0)

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower().strip(".")
    if fmt == "pdf":
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
    elif fmt in ("png", "jpg", "jpeg"):
        fig.savefig(out_path, format=fmt, dpi=dpi, bbox_inches="tight")
    else:
        plt.close(fig)
        raise ValueError(f"不支持的输出格式: {fmt}")

    plt.close(fig)

    return {
        "out_path": out_path,
        "total_sec": total_sec,
        "time_scale": time_scale,
        "notes_total": len(notes),
        "notes_plotted": len(filtered),
        "tracks_detected": n_tracks,
        "velocity_affects_alpha": velocity_affects_alpha,
        "total_width_units": total_width,
        "use_realistic_upper_layout": use_realistic_upper_layout,
    }


@dataclass
class UIState:
    midi_path: str = ""
    out_dir: str = ""
    out_name: str = "waterfall"
    out_fmt: str = "pdf"

    width_in: float = 14.0
    dpi: int = 300

    white_w: float = 1.0
    black_w: float = 0.9

    time_scale: float = 80.0

    note_alpha: float = 0.90
    edge_alpha: float = 0.95
    edge_width: float = 1.00

    velocity_affects_alpha: bool = False
    use_realistic_upper_layout: bool = True


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MIDI转瀑布流")
        self.geometry("800x600")
        self.minsize(800, 600)

        self._set_app_icon()

        self.state = UIState()
        self._build_ui()

    def _set_app_icon(self):
        """加载根目录 icon.ico（影响窗口图标与任务栏图标）。"""
        try:
            base_dir = Path(__file__).resolve().parent
        except Exception:
            base_dir = Path.cwd()

        icon_path = base_dir / "icon.ico"
        if not icon_path.exists():
            icon_path = Path.cwd() / "icon.ico"

        if os.name == "nt" and icon_path.exists():
            try:
                self.iconbitmap(str(icon_path))
            except Exception:
                pass

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, **pad)

        io_box = ttk.LabelFrame(frame, text="输入 / 输出")
        io_box.pack(fill="x", **pad)

        self.var_midi = tk.StringVar()
        r1 = ttk.Frame(io_box); r1.pack(fill="x", **pad)
        ttk.Label(r1, text="MIDI 文件:").pack(side="left")
        ttk.Entry(r1, textvariable=self.var_midi).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r1, text="选择…", command=self.pick_midi).pack(side="left")

        self.var_outdir = tk.StringVar(value=str(Path.cwd()))
        r2 = ttk.Frame(io_box); r2.pack(fill="x", **pad)
        ttk.Label(r2, text="输出目录:").pack(side="left")
        ttk.Entry(r2, textvariable=self.var_outdir).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r2, text="选择…", command=self.pick_outdir).pack(side="left")

        self.var_outname = tk.StringVar(value=self.state.out_name)
        self.var_fmt = tk.StringVar(value=self.state.out_fmt)
        r3 = ttk.Frame(io_box); r3.pack(fill="x", **pad)
        ttk.Label(r3, text="输出文件名:").pack(side="left")
        ttk.Entry(r3, textvariable=self.var_outname, width=28).pack(side="left", padx=8)
        ttk.Label(r3, text="格式:").pack(side="left", padx=(12, 0))
        ttk.Combobox(r3, textvariable=self.var_fmt, values=["pdf", "png", "jpg"], width=8, state="readonly").pack(side="left", padx=8)

        param_box = ttk.LabelFrame(frame, text="参数")
        param_box.pack(fill="x", **pad)

        self.var_width_in = tk.DoubleVar(value=self.state.width_in)
        self.var_dpi = tk.IntVar(value=self.state.dpi)

        p1 = ttk.Frame(param_box); p1.pack(fill="x", **pad)
        ttk.Label(p1, text="图宽:").pack(side="left")
        ttk.Entry(p1, textvariable=self.var_width_in, width=10).pack(side="left", padx=8)
        ttk.Label(p1, text="DPI(仅png/jpg):").pack(side="left", padx=(12, 0))
        ttk.Entry(p1, textvariable=self.var_dpi, width=10).pack(side="left", padx=8)

        self.var_timescale = tk.DoubleVar(value=self.state.time_scale)
        p2 = ttk.Frame(param_box); p2.pack(fill="x", **pad)
        ttk.Label(p2, text="纵向缩放(越大纵向越长):").pack(side="left")
        ttk.Entry(p2, textvariable=self.var_timescale, width=12).pack(side="left", padx=8)

        self.var_use_real = tk.BooleanVar(value=self.state.use_realistic_upper_layout)
        p2b = ttk.Frame(param_box); p2b.pack(fill="x", **pad)
        ttk.Checkbutton(
            p2b,
            text="使用真实钢琴上半键宽比例（勾选后忽略白/黑键宽设置）",
            variable=self.var_use_real,
            command=self._sync_key_width_state
        ).pack(side="left")

        self.var_white_w = tk.DoubleVar(value=self.state.white_w)
        self.var_black_w = tk.DoubleVar(value=self.state.black_w)
        p3 = ttk.Frame(param_box); p3.pack(fill="x", **pad)
        ttk.Label(p3, text="白键宽:").pack(side="left")
        self.ent_white = ttk.Entry(p3, textvariable=self.var_white_w, width=10)
        self.ent_white.pack(side="left", padx=8)
        ttk.Label(p3, text="黑键宽:").pack(side="left", padx=(12, 0))
        self.ent_black = ttk.Entry(p3, textvariable=self.var_black_w, width=10)
        self.ent_black.pack(side="left", padx=8)

        self.var_alpha = tk.DoubleVar(value=self.state.note_alpha)
        self.var_edge_alpha = tk.DoubleVar(value=self.state.edge_alpha)
        self.var_edge_w = tk.DoubleVar(value=self.state.edge_width)
        self.var_vel_alpha = tk.BooleanVar(value=self.state.velocity_affects_alpha)

        p4 = ttk.Frame(param_box); p4.pack(fill="x", **pad)
        ttk.Label(p4, text="音符透明度:").pack(side="left")
        ttk.Entry(p4, textvariable=self.var_alpha, width=10).pack(side="left", padx=8)
        ttk.Label(p4, text="边框透明度:").pack(side="left", padx=(12, 0))
        ttk.Entry(p4, textvariable=self.var_edge_alpha, width=10).pack(side="left", padx=8)
        ttk.Label(p4, text="边框宽度:").pack(side="left", padx=(12, 0))
        ttk.Entry(p4, textvariable=self.var_edge_w, width=10).pack(side="left", padx=8)

        p5 = ttk.Frame(param_box); p5.pack(fill="x", **pad)
        ttk.Checkbutton(p5, text="按力度渲染透明度", variable=self.var_vel_alpha).pack(side="left")

        run_box = ttk.Frame(frame)
        run_box.pack(fill="x", **pad)

        self.btn_run = ttk.Button(run_box, text="生成瀑布流", command=self.on_run)
        self.btn_run.pack(side="left")

        ttk.Button(run_box, text="打开输出目录", command=self.open_outdir).pack(side="left", padx=10)

        self.var_status = tk.StringVar(value="就绪。")
        ttk.Label(frame, textvariable=self.var_status).pack(anchor="w", **pad)

        log_box = ttk.LabelFrame(frame, text="日志")
        log_box.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(log_box, height=10)
        self.txt.pack(fill="both", expand=True, padx=8, pady=8)
        self.txt.configure(state="disabled")

        self._sync_key_width_state()

    def _sync_key_width_state(self):
        """勾选真实比例时禁用白/黑键宽输入框。"""
        use_real = bool(self.var_use_real.get())
        state = "disabled" if use_real else "normal"
        try:
            self.ent_white.configure(state=state)
            self.ent_black.configure(state=state)
        except Exception:
            pass

    def log(self, s: str):
        self.txt.configure(state="normal")
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def pick_midi(self):
        path = filedialog.askopenfilename(
            title="选择 MIDI 文件",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")]
        )
        if path:
            self.var_midi.set(path)
            stem = Path(path).stem
            self.var_outname.set(stem + "_waterfall")
            self.log(f"选择 MIDI: {path}")

    def pick_outdir(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.var_outdir.set(path)
            self.log(f"选择输出目录: {path}")

    def open_outdir(self):
        out_dir = self.var_outdir.get().strip()
        if not out_dir:
            messagebox.showwarning("提示", "请先选择输出目录。")
            return
        try:
            if os.name == "nt":
                os.startfile(out_dir)  # type: ignore
            elif sys.platform == "darwin":
                os.system(f'open "{out_dir}"')
            else:
                os.system(f'xdg-open "{out_dir}"')
        except Exception as e:
            messagebox.showerror("错误", f"无法打开目录: {e}")

    def on_run(self):
        midi_path = self.var_midi.get().strip()
        out_dir = self.var_outdir.get().strip()
        out_name = (self.var_outname.get().strip() or "waterfall")
        out_fmt = self.var_fmt.get().strip().lower()

        try:
            width_in = float(self.var_width_in.get())
            dpi = int(self.var_dpi.get())
            time_scale = float(self.var_timescale.get())
            white_w = float(self.var_white_w.get())
            black_w = float(self.var_black_w.get())
            note_alpha = float(self.var_alpha.get())
            edge_alpha = float(self.var_edge_alpha.get())
            edge_w = float(self.var_edge_w.get())
            vel_alpha = bool(self.var_vel_alpha.get())
            use_real = bool(self.var_use_real.get())
        except Exception:
            messagebox.showwarning("提示", "参数格式不正确，请检查输入。")
            return

        if not midi_path or not Path(midi_path).exists():
            messagebox.showwarning("提示", "请先选择一个有效的 MIDI 文件。")
            return
        if not out_dir:
            messagebox.showwarning("提示", "请先选择输出目录。")
            return
        if time_scale <= 0:
            messagebox.showwarning("提示", "纵向缩放必须 > 0。")
            return

        if not use_real:
            if white_w <= 0 or black_w <= 0:
                messagebox.showwarning("提示", "白键宽 / 黑键宽必须 > 0。")
                return
            if black_w >= white_w:
                messagebox.showwarning("提示", "黑键宽应小于白键宽。")
                return

        if not (0.0 < note_alpha <= 1.0):
            messagebox.showwarning("提示", "音符透明度建议在 (0,1]。")
            return
        if not (0.0 < edge_alpha <= 1.0):
            messagebox.showwarning("提示", "边框透明度建议在 (0,1]。")
            return
        if edge_w <= 0:
            messagebox.showwarning("提示", "边框宽度必须 > 0。")
            return

        out_path = Path(out_dir) / f"{out_name}.{out_fmt}"

        self.btn_run.configure(state="disabled")
        self.var_status.set("正在生成…")
        self.log("开始生成…")
        self.log(f"输入: {midi_path}")
        self.log(f"输出: {out_path}")
        if use_real:
            self.log(f"参数: time_scale={time_scale}, 真实键宽=开启, vel_alpha={vel_alpha}")
        else:
            self.log(f"参数: time_scale={time_scale}, white_w={white_w}, black_w={black_w}, vel_alpha={vel_alpha}")

        def worker():
            try:
                info = render_waterfall_tracks_variablewidth(
                    midi_path=midi_path,
                    out_path=str(out_path),
                    fmt=out_fmt,
                    width_in=width_in,
                    dpi=dpi,
                    white_w=white_w,
                    black_w=black_w,
                    time_scale=time_scale,
                    note_alpha=note_alpha,
                    edge_alpha=edge_alpha,
                    edge_width=edge_w,
                    velocity_affects_alpha=vel_alpha,
                    use_realistic_upper_layout=use_real,
                )
                self.after(0, lambda: self._done_ok(info))
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, lambda: self._done_err(e, tb))

        threading.Thread(target=worker, daemon=True).start()

    def _done_ok(self, info: dict):
        self.btn_run.configure(state="normal")
        self.var_status.set("生成完成")
        msg = (
            f"完成 输出: {info['out_path']}\n"
            f"绘制音符: {info['notes_plotted']} / 解析音符: {info['notes_total']}\n"
            f"检测到轨数: {info['tracks_detected']}\n"
        )
        self.log(msg)
        messagebox.showinfo("完成", msg)

    def _done_err(self, err: Exception, tb: str):
        self.btn_run.configure(state="normal")
        self.var_status.set("生成失败")
        self.log("生成失败：")
        self.log(str(err))
        self.log(tb)
        messagebox.showerror("错误", f"生成失败：\n{err}\n\n详情请看日志。")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
