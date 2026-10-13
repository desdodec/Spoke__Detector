#!/usr/bin/env python3
"""Clean human-in-the-loop acoustic click review UI.

Focused workflow: open a WAV -> detect clicks -> zoom/select -> tell the app how
many real clicks are in a region -> review/edit -> export/save calibration data.
"""
from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy.io import wavfile

import transient_lab as lab

BG = "#f5f2ea"
PAPER = "#fffdf8"
INK = "#1f2430"
MUTED = "#7d8490"
GRID = "#e7e2d8"
WAVE = "#3c4658"
CORAL = "#ef6a5b"
TEAL = "#1e9e91"
TEAL_SOFT = "#d9efeb"
AMBER = "#d9982b"
PURPLE = "#7658c9"
BUTTON = "#2d3748"
BUTTON_HOVER = "#404b5e"

ROOT = Path(__file__).resolve().parent.parent
QUICK_SESSIONS = [
    ("Jo Grant", ROOT / "audio" / "Jo_Grant.wav"),
    ("One Revolution", ROOT / "audio" / "One Revolution.wav"),
    ("New Revolution", ROOT / "audio" / "One revolution new.wav"),
    ("7 clicks", ROOT / "audio" / "7_clicks.wav"),
]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector — Click Review")
        self.geometry("1460x880")
        self.minsize(1080, 700)
        self.configure(bg=BG)
        self.audio: lab.AudioData | None = None
        self.filtered: np.ndarray | None = None
        self.score: np.ndarray | None = None
        self.events: list[lab.Event] = []
        self.view_start = 0.0
        self.view_end = 1.0
        self.sel_start: float | None = None
        self.sel_end: float | None = None
        self.dragging = False
        self.hover_t: float | None = None
        self._temp_files: list[str] = []
        self.file_name = tk.StringVar(value="No WAV loaded")
        self.meta = tk.StringVar(value="Choose a file or one of the example sessions.")
        self.status = tk.StringVar(value="Ready")
        self.event_count = tk.StringVar(value="0 clicks")
        self.selection_count = tk.StringVar(value="No selection")
        self.threshold = tk.DoubleVar(value=lab.DEFAULT_THRESHOLD_DBFS)
        self.min_sep = tk.DoubleVar(value=lab.DEFAULT_MIN_SEPARATION_MS)
        self.highpass = tk.DoubleVar(value=lab.DEFAULT_HIGHPASS_HZ)
        self.expected = tk.IntVar(value=4)
        self._style()
        self._build()
        self.bind("<space>", lambda _e: self.play())
        self.bind("<Escape>", lambda _e: self.clear_selection())
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _style(self) -> None:
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("Root.TFrame", background=BG)
        s.configure("Card.TFrame", background=PAPER)
        s.configure("Ink.TLabel", background=BG, foreground=INK)
        s.configure("Muted.TLabel", background=BG, foreground=MUTED)
        s.configure("Card.TLabel", background=PAPER, foreground=INK)
        s.configure("CardMuted.TLabel", background=PAPER, foreground=MUTED)
        s.configure("Title.TLabel", background=BG, foreground=INK, font=("TkDefaultFont", 22, "bold"))
        s.configure("Big.TLabel", background=BG, foreground=INK, font=("TkDefaultFont", 13, "bold"))
        s.configure("Primary.TButton", background=BUTTON, foreground="#ffffff", padding=(12, 8), borderwidth=0)
        s.map("Primary.TButton", background=[("active", BUTTON_HOVER)])
        s.configure("Soft.TButton", background=PAPER, foreground=INK, padding=(10, 7), bordercolor=GRID)
        s.map("Soft.TButton", background=[("active", "#f1ede5")])
        s.configure("Small.TButton", background=PAPER, foreground=INK, padding=(8, 5), bordercolor=GRID)
        s.map("Small.TButton", background=[("active", "#f1ede5")])
        s.configure("Clean.Horizontal.TScale", background=PAPER, troughcolor=GRID)

    def _build(self) -> None:
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)
        header = ttk.Frame(self, style="Root.TFrame", padding=(22, 18, 22, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        title = ttk.Frame(header, style="Root.TFrame")
        title.grid(row=0, column=0, sticky="w")
        ttk.Label(title, text="CLICK REVIEW", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title, text="Find acoustic clicks. Review them. Build a trustworthy calibration.", style="Muted.TLabel").pack(anchor="w")
        file_box = ttk.Frame(header, style="Root.TFrame")
        file_box.grid(row=0, column=1, sticky="ew", padx=28)
        ttk.Label(file_box, textvariable=self.file_name, style="Big.TLabel").pack(anchor="w")
        ttk.Label(file_box, textvariable=self.meta, style="Muted.TLabel").pack(anchor="w")
        ttk.Button(header, text="Open WAV", style="Primary.TButton", command=self.open_dialog).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(header, text="Export clicks.csv", style="Soft.TButton", command=self.export).grid(row=0, column=3)

        quick = ttk.Frame(self, style="Root.TFrame", padding=(22, 0, 22, 10))
        quick.grid(row=1, column=0, sticky="ew")
        ttk.Label(quick, text="Examples", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        for label, path in QUICK_SESSIONS:
            if path.exists():
                ttk.Button(quick, text=label, style="Small.TButton", command=lambda p=path: self.load_path(p)).pack(side="left", padx=(0, 6))

        stage = ttk.Frame(self, style="Card.TFrame", padding=0)
        stage.grid(row=2, column=0, sticky="nsew", padx=22)
        stage.rowconfigure(0, weight=1)
        stage.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(stage, bg=PAPER, highlightthickness=1, highlightbackground=GRID, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self.draw())
        self.canvas.bind("<ButtonPress-1>", self._select_start)
        self.canvas.bind("<B1-Motion>", self._select_drag)
        self.canvas.bind("<ButtonRelease-1>", self._select_end)
        self.canvas.bind("<Shift-Button-1>", self._add_event)
        self.canvas.bind("<Button-3>", self._remove_event)
        self.canvas.bind("<Motion>", self._motion)
        self.canvas.bind("<Leave>", lambda _e: self._leave())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Double-Button-1>", self._double_zoom)

        actions = ttk.Frame(self, style="Root.TFrame", padding=(22, 10, 22, 8))
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure(4, weight=1)
        ttk.Button(actions, text="▶ Play", style="Soft.TButton", command=self.play).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(actions, text="Zoom selection", style="Soft.TButton", command=self.zoom_selection).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(actions, text="Full file", style="Soft.TButton", command=self.full_view).grid(row=0, column=2, padx=(0, 10))
        ttk.Label(actions, textvariable=self.event_count, style="Big.TLabel").grid(row=0, column=3, padx=(0, 14))
        ttk.Label(actions, textvariable=self.selection_count, style="Muted.TLabel").grid(row=0, column=4, sticky="w")

        controls = ttk.Frame(self, style="Card.TFrame", padding=(18, 14))
        controls.grid(row=4, column=0, sticky="ew", padx=22, pady=(0, 8))
        for col in range(9):
            controls.columnconfigure(col, weight=1 if col in (1, 3, 5) else 0)
        ttk.Label(controls, text="Threshold", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        self.threshold_value = ttk.Label(controls, text="", style="Card.TLabel")
        self.threshold_value.grid(row=0, column=1, sticky="e")
        ttk.Label(controls, text="Min gap", style="CardMuted.TLabel").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.sep_value = ttk.Label(controls, text="", style="Card.TLabel")
        self.sep_value.grid(row=0, column=3, sticky="e")
        ttk.Label(controls, text="High-pass", style="CardMuted.TLabel").grid(row=0, column=4, sticky="w", padx=(20, 0))
        self.hp_value = ttk.Label(controls, text="", style="Card.TLabel")
        self.hp_value.grid(row=0, column=5, sticky="e")
        ttk.Button(controls, text="Detect file", style="Primary.TButton", command=self.detect_full).grid(row=0, column=6, rowspan=2, padx=(20, 6))
        ttk.Button(controls, text="Detect view", style="Soft.TButton", command=self.detect_view).grid(row=0, column=7, rowspan=2, padx=(0, 14))
        self._scale(controls, self.threshold, -90, -10, 1, 1, lambda: f"{self.threshold.get():.1f} dBFS", self.threshold_value)
        self._scale(controls, self.min_sep, 5, 150, 3, 3, lambda: f"{self.min_sep.get():.0f} ms", self.sep_value)
        self._scale(controls, self.highpass, 100, 5000, 5, 5, lambda: f"{self.highpass.get():.0f} Hz", self.hp_value)

        known = ttk.Frame(controls, style="Card.TFrame")
        known.grid(row=0, column=8, rowspan=2, sticky="e")
        ttk.Label(known, text="Selection contains", style="CardMuted.TLabel").grid(row=0, column=0, sticky="e", padx=(0, 5))
        tk.Spinbox(known, from_=0, to=999, textvariable=self.expected, width=5, justify="center", bg=PAPER, fg=INK, relief="solid", bd=1).grid(row=0, column=1)
        ttk.Label(known, text="real clicks", style="CardMuted.TLabel").grid(row=0, column=2, padx=(5, 8))
        ttk.Button(known, text="Fit selection", style="Soft.TButton", command=self.fit_selection).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(known, text="Save calibration", style="Soft.TButton", command=self.save_reference).grid(row=0, column=4)

        footer = ttk.Frame(self, style="Root.TFrame", padding=(22, 0, 22, 12))
        footer.grid(row=5, column=0, sticky="ew")
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(side="left")
        ttk.Label(footer, text="Drag select · wheel zoom · Shift-click add · right-click remove · Space play", style="Muted.TLabel").pack(side="right")
        self._refresh_values()

    def _scale(self, parent, var, lo, hi, row, col, formatter, value_label) -> None:
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal", style="Clean.Horizontal.TScale")
        scale.grid(row=row, column=col, sticky="ew", padx=(0, 12))
        scale.bind("<B1-Motion>", lambda _e: value_label.configure(text=formatter()))
        scale.bind("<ButtonRelease-1>", lambda _e: self._setting_changed())

    def open_dialog(self) -> None:
        p = filedialog.askopenfilename(title="Open WAV", filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")])
        if p:
            self.load_path(Path(p))

    def load_path(self, path: Path) -> None:
        try:
            self.audio = lab.load_wav(path)
            self.file_name.set(path.name)
            self.meta.set(f"{self.audio.duration_s:.2f} s · {self.audio.sample_rate:,} Hz")
            self.events = []
            self.view_start, self.view_end = 0.0, self.audio.duration_s
            self.sel_start = self.sel_end = None
            self._recompute_curve()
            self.detect_full()
            self.status.set(f"Loaded {path.name}. Automatic clicks are marked in coral.")
        except Exception as exc:
            messagebox.showerror("Cannot open WAV", str(exc))

    def _recompute_curve(self) -> None:
        if self.audio is not None:
            self.filtered, _, self.score = lab.transient_curve(self.audio, self.highpass.get())

    def _setting_changed(self) -> None:
        self._refresh_values()
        self._recompute_curve()
        self.draw()

    def _refresh_values(self) -> None:
        self.threshold_value.configure(text=f"{self.threshold.get():.1f} dBFS")
        self.sep_value.configure(text=f"{self.min_sep.get():.0f} ms")
        self.hp_value.configure(text=f"{self.highpass.get():.0f} Hz")

    def _detect(self, a: float, b: float, replace_all: bool) -> None:
        if self.audio is None:
            return
        new = lab.detect_events(self.audio, self.threshold.get(), self.min_sep.get(), self.highpass.get(), a, b)
        if replace_all:
            self.events = new
        else:
            keep = [e for e in self.events if not (a <= e.timestamp_seconds <= b)]
            self.events = sorted(keep + new, key=lambda e: e.timestamp_seconds)
        self._recompute_curve()
        self.refresh_labels()
        self.draw()

    def detect_full(self) -> None:
        if self.audio is None:
            return
        self._detect(0.0, self.audio.duration_s, True)
        self.status.set(f"Detected {len(self.events)} acoustic clicks at {self.threshold.get():.1f} dBFS.")

    def detect_view(self) -> None:
        if self.audio is None:
            return
        self._detect(self.view_start, self.view_end, False)
        self.status.set("Re-detected the visible region only.")

    def selection(self) -> tuple[float, float] | None:
        if self.sel_start is None or self.sel_end is None:
            return None
        a, b = sorted((self.sel_start, self.sel_end))
        return (a, b) if b - a >= 0.001 else None

    def fit_selection(self) -> None:
        if self.audio is None:
            return
        region = self.selection()
        if region is None:
            messagebox.showinfo("Select a region", "Drag over a short region of the waveform first.")
            return
        a, b = region
        try:
            th, new = lab.fit_threshold_for_count(self.audio, int(self.expected.get()), a, b, self.min_sep.get(), self.highpass.get())
            self.threshold.set(round(th, 3))
            self._refresh_values()
            keep = [e for e in self.events if not (a <= e.timestamp_seconds <= b)]
            self.events = sorted(keep + new, key=lambda e: e.timestamp_seconds)
            self.refresh_labels()
            self.draw()
            self.status.set(f"Selection fitted to {len(new)} clicks. Suggested threshold: {th:.2f} dBFS.")
        except Exception as exc:
            messagebox.showerror("Could not fit selection", str(exc))

    def save_reference(self) -> None:
        if self.audio is None:
            return
        region = self.selection()
        if region is None:
            messagebox.showinfo("Select a region", "Select a reviewed region before saving calibration data.")
            return
        a, b = region
        chosen = [e for e in self.events if a <= e.timestamp_seconds <= b]
        expected = int(self.expected.get())
        if len(chosen) != expected and not messagebox.askyesno("Marker count differs", f"You said the region contains {expected} real clicks, but {len(chosen)} markers are currently present. Save anyway?"):
            return
        p = filedialog.asksaveasfilename(title="Calibration library", defaultextension=".jsonl", initialfile="click_calibration.jsonl", filetypes=[("JSON Lines", "*.jsonl")])
        if p:
            lab.append_reference(Path(p), self.audio, a, b, expected, chosen, self.threshold.get(), self.min_sep.get(), self.highpass.get())
            self.status.set(f"Saved reviewed selection to {Path(p).name}.")

    def export(self) -> None:
        if self.audio is None:
            return
        p = filedialog.asksaveasfilename(title="Export reviewed clicks", defaultextension=".csv", initialfile="clicks.csv", filetypes=[("CSV", "*.csv")])
        if p:
            lab.export_events_csv(Path(p), self.events)
            self.status.set(f"Exported {len(self.events)} reviewed clicks to {Path(p).name}.")

    def x_to_time(self, x: float) -> float:
        w = max(1, self.canvas.winfo_width())
        return self.view_start + max(0.0, min(1.0, x / w)) * (self.view_end - self.view_start)

    def time_to_x(self, t: float) -> float:
        return (t - self.view_start) / max(1e-12, self.view_end - self.view_start) * max(1, self.canvas.winfo_width())

    def _select_start(self, event) -> None:
        if event.state & 0x0001:
            return
        self.dragging = True
        self.sel_start = self.sel_end = self.x_to_time(event.x)
        self.draw()

    def _select_drag(self, event) -> None:
        if self.dragging:
            self.sel_end = self.x_to_time(event.x)
            self.refresh_labels()
            self.draw()

    def _select_end(self, event) -> None:
        if self.dragging:
            self.sel_end = self.x_to_time(event.x)
            self.dragging = False
            self.refresh_labels()
            self.draw()

    def _add_event(self, event) -> None:
        if self.audio is None or self.filtered is None or self.score is None:
            return
        sr = self.audio.sample_rate
        idx = int(round(self.x_to_time(event.x) * sr))
        radius = max(1, int(round(0.007 * sr)))
        lo, hi = max(0, idx - radius), min(len(self.filtered), idx + radius + 1)
        if hi <= lo:
            return
        refined = lo + int(np.argmax(np.abs(self.filtered[lo:hi])))
        amp = float(abs(self.filtered[refined]))
        self.events.append(lab.Event(refined / sr, float(self.score[refined]), amp, amp, "manual_added"))
        self.events.sort(key=lambda e: e.timestamp_seconds)
        self.refresh_labels()
        self.draw()
        self.status.set("Manual click added.")

    def _remove_event(self, event) -> None:
        if not self.events:
            return
        t = self.x_to_time(event.x)
        idx, dist = min(((i, abs(e.timestamp_seconds - t)) for i, e in enumerate(self.events)), key=lambda z: z[1])
        tolerance = max(0.010, (self.view_end - self.view_start) * 0.012)
        if dist <= tolerance:
            removed = self.events.pop(idx)
            self.refresh_labels()
            self.draw()
            self.status.set(f"Removed marker at {removed.timestamp_seconds:.4f} s.")

    def _motion(self, event) -> None:
        if self.audio is not None:
            self.hover_t = self.x_to_time(event.x)
            self.draw()

    def _leave(self) -> None:
        self.hover_t = None
        self.draw()

    def _wheel(self, event) -> None:
        if self.audio is None:
            return
        anchor = self.x_to_time(event.x)
        factor = 0.72 if event.delta > 0 else 1.38
        span = min(self.audio.duration_s, max(0.04, (self.view_end - self.view_start) * factor))
        frac = max(0.0, min(1.0, event.x / max(1, self.canvas.winfo_width())))
        start = max(0.0, min(anchor - frac * span, self.audio.duration_s - span))
        self.view_start, self.view_end = start, start + span
        self.refresh_labels()
        self.draw()

    def _double_zoom(self, event) -> None:
        if self.audio is None:
            return
        center = self.x_to_time(event.x)
        self._set_view_center(center, max(0.08, (self.view_end - self.view_start) * 0.35))

    def _set_view_center(self, center: float, span: float) -> None:
        if self.audio is None:
            return
        span = min(self.audio.duration_s, max(0.04, span))
        start = max(0.0, min(center - span / 2, self.audio.duration_s - span))
        self.view_start, self.view_end = start, start + span
        self.refresh_labels()
        self.draw()

    def zoom_selection(self) -> None:
        region = self.selection()
        if self.audio is None or region is None:
            return
        a, b = region
        pad = max(0.02, (b - a) * 0.12)
        self.view_start, self.view_end = max(0.0, a - pad), min(self.audio.duration_s, b + pad)
        self.refresh_labels()
        self.draw()

    def full_view(self) -> None:
        if self.audio is not None:
            self.view_start, self.view_end = 0.0, self.audio.duration_s
            self.refresh_labels()
            self.draw()

    def clear_selection(self) -> None:
        self.sel_start = self.sel_end = None
        self.refresh_labels()
        self.draw()

    def play(self) -> None:
        if self.audio is None:
            return
        region = self.selection()
        a, b = region if region else (self.view_start, self.view_end)
        self._play_range(a, b)

    def _play_range(self, a: float, b: float) -> None:
        if self.audio is None:
            return
        sr = self.audio.sample_rate
        lo, hi = max(0, int(round(a * sr))), min(len(self.audio.mono), int(round(b * sr)))
        if hi <= lo:
            return
        pcm = np.round(np.clip(self.audio.mono[lo:hi], -1.0, 1.0) * 32767).astype(np.int16)
        fd, name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        wavfile.write(name, sr, pcm)
        self._temp_files.append(name)
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.PlaySound(name, winsound.SND_FILENAME | winsound.SND_ASYNC)
            elif platform.system() == "Darwin":
                subprocess.Popen(["afplay", name])
            else:
                subprocess.Popen(["aplay", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            messagebox.showerror("Playback unavailable", str(exc))

    def refresh_labels(self) -> None:
        self.event_count.set(f"{len(self.events)} clicks")
        region = self.selection()
        if region is None:
            self.selection_count.set(f"Visible: {self.view_end - self.view_start:.2f} s")
        else:
            a, b = region
            n = sum(a <= e.timestamp_seconds <= b for e in self.events)
            self.selection_count.set(f"Selection {a:.3f}–{b:.3f} s · {n} markers")

    def _ruler_step(self, span: float) -> float:
        raw = span / 8.0
        choices = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60]
        return min(choices, key=lambda x: abs(np.log(max(raw, 1e-9) / x)))

    def _column_minmax(self, y: np.ndarray, lo: int, hi: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        if hi <= lo:
            return np.array([]), np.array([])
        edges = np.linspace(lo, hi, width + 1).astype(int)
        mins, maxs = np.empty(width), np.empty(width)
        for i in range(width):
            a, b = edges[i], max(edges[i] + 1, edges[i + 1])
            seg = y[a:min(b, len(y))]
            mins[i] = float(np.min(seg)) if seg.size else 0.0
            maxs[i] = float(np.max(seg)) if seg.size else 0.0
        return mins, maxs

    def draw(self) -> None:
        c = self.canvas
        c.delete("all")
        w, h = max(20, c.winfo_width()), max(20, c.winfo_height())
        if self.audio is None:
            c.create_text(w / 2, h / 2 - 18, text="Open a WAV to begin", fill=INK, font=("TkDefaultFont", 18, "bold"))
            c.create_text(w / 2, h / 2 + 16, text="Or choose one of the example sessions above.", fill=MUTED, font=("TkDefaultFont", 11))
            return

        ruler_h, wave_top, wave_bottom = 34, 48, int(h * 0.72)
        score_top, score_bottom = wave_bottom + 18, h - 34
        region = self.selection()
        if region:
            a, b = region
            if b >= self.view_start and a <= self.view_end:
                x0, x1 = self.time_to_x(max(a, self.view_start)), self.time_to_x(min(b, self.view_end))
                c.create_rectangle(x0, wave_top - 6, x1, score_bottom, fill=TEAL_SOFT, outline="")

        span = self.view_end - self.view_start
        step = self._ruler_step(span)
        t = np.ceil(self.view_start / step) * step
        while t <= self.view_end + 1e-9:
            x = self.time_to_x(float(t))
            c.create_line(x, ruler_h, x, score_bottom, fill=GRID)
            c.create_text(x + 4, 11, anchor="nw", text=f"{t:.2f}s" if step < 1 else f"{t:.1f}s", fill=MUTED, font=("TkDefaultFont", 9))
            t += step
        c.create_line(0, ruler_h, w, ruler_h, fill=GRID)

        sr = self.audio.sample_rate
        lo, hi = max(0, int(self.view_start * sr)), min(len(self.audio.mono), int(self.view_end * sr))
        width = max(10, min(w, hi - lo))
        mins, maxs = self._column_minmax(self.audio.mono, lo, hi, width)
        peak = max(1e-9, float(max(np.max(np.abs(mins)) if mins.size else 0, np.max(np.abs(maxs)) if maxs.size else 0)))
        mid, amp = (wave_top + wave_bottom) / 2, (wave_bottom - wave_top) * 0.42
        xscale = w / max(1, width - 1)
        c.create_line(0, mid, w, mid, fill=GRID)
        for i, (mn, mx) in enumerate(zip(mins, maxs)):
            x = i * xscale
            c.create_line(x, mid - (mx / peak) * amp, x, mid - (mn / peak) * amp, fill=WAVE)
        c.create_text(10, wave_top, anchor="nw", text="WAVEFORM", fill=MUTED, font=("TkDefaultFont", 9, "bold"))

        c.create_line(0, score_top - 9, w, score_top - 9, fill=GRID)
        c.create_text(10, score_top, anchor="nw", text="TRANSIENT STRENGTH", fill=MUTED, font=("TkDefaultFont", 9, "bold"))
        if self.score is not None and hi > lo:
            edges = np.linspace(lo, hi, width + 1).astype(int)
            max_score = np.full(width, -120.0)
            for i in range(width):
                a, b = edges[i], max(edges[i] + 1, edges[i + 1])
                seg = self.score[a:min(b, len(self.score))]
                if seg.size:
                    max_score[i] = float(np.max(seg))
            floor, ceil = -100.0, -5.0
            pts = []
            for i, v in enumerate(max_score):
                frac = (np.clip(v, floor, ceil) - floor) / (ceil - floor)
                pts.extend((i * xscale, score_bottom - frac * (score_bottom - score_top - 16)))
            if len(pts) >= 4:
                c.create_line(*pts, fill=TEAL, width=1.5)
            th = float(self.threshold.get())
            frac = (np.clip(th, floor, ceil) - floor) / (ceil - floor)
            yth = score_bottom - frac * (score_bottom - score_top - 16)
            c.create_line(0, yth, w, yth, fill=AMBER, width=2)
            c.create_text(w - 8, yth - 7, anchor="se", text=f"threshold {th:.1f} dBFS", fill=AMBER, font=("TkDefaultFont", 9, "bold"))

        zoomed = span <= 20.0
        for ev in (e for e in self.events if self.view_start <= e.timestamp_seconds <= self.view_end):
            x = self.time_to_x(ev.timestamp_seconds)
            col = PURPLE if ev.source == "manual_added" else CORAL
            c.create_line(x, wave_top - 2, x, wave_bottom if zoomed else wave_top + 15, fill=col, width=2)
            c.create_polygon(x, wave_top - 7, x - 4, wave_top - 1, x + 4, wave_top - 1, fill=col, outline="")
            if span <= 4.0:
                c.create_text(x + 4, wave_top + 2, anchor="nw", text=f"{ev.timestamp_seconds:.4f}", fill=col, font=("TkDefaultFont", 8))

        if self.hover_t is not None and self.view_start <= self.hover_t <= self.view_end:
            x = self.time_to_x(self.hover_t)
            c.create_line(x, ruler_h, x, score_bottom, fill="#a9adb5", dash=(2, 3))
            idx = min(len(self.score) - 1, max(0, int(self.hover_t * sr))) if self.score is not None else 0
            extra = f" · {self.score[idx]:.1f} dBFS" if self.score is not None else ""
            label = f"{self.hover_t:.5f} s{extra}"
            c.create_rectangle(x + 7, ruler_h + 5, x + 157, ruler_h + 29, fill=PAPER, outline=GRID)
            c.create_text(x + 13, ruler_h + 17, anchor="w", text=label, fill=INK, font=("TkDefaultFont", 9))
        c.create_text(10, h - 10, anchor="sw", text=f"view {self.view_start:.3f}–{self.view_end:.3f} s", fill=MUTED, font=("TkDefaultFont", 9))

    def close(self) -> None:
        for name in self._temp_files:
            try:
                os.unlink(name)
            except OSError:
                pass
        self.destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
