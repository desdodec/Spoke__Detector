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
import time
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
PLAYHEAD = "#1575c7"

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
        self.geometry("1460x900")
        self.minsize(1080, 720)
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
        self._play_proc: subprocess.Popen | None = None
        self.playing = False
        self.paused = False
        self.play_start_t = 0.0
        self.play_end_t = 0.0
        self.play_anchor_t = 0.0
        self.play_wall = 0.0
        self.playhead_t: float | None = None
        self._play_after_id: str | None = None

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
        self.bind("<Left>", lambda _e: self.pan_view(-0.20))
        self.bind("<Right>", lambda _e: self.pan_view(0.20))
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

        self.scroll = ttk.Scrollbar(stage, orient="horizontal", command=self._scroll_command)
        self.scroll.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        actions = ttk.Frame(self, style="Root.TFrame", padding=(22, 10, 22, 8))
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure(8, weight=1)
        ttk.Button(actions, text="▶ Play", style="Primary.TButton", command=self.play).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(actions, text="Ⅱ Pause", style="Soft.TButton", command=self.pause).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(actions, text="■ Stop", style="Soft.TButton", command=self.stop).grid(row=0, column=2, padx=(0, 12))
        ttk.Button(actions, text="Zoom selection", style="Soft.TButton", command=self.zoom_selection).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(actions, text="←", style="Small.TButton", command=lambda: self.pan_view(-0.35)).grid(row=0, column=4, padx=(0, 3))
        ttk.Button(actions, text="→", style="Small.TButton", command=lambda: self.pan_view(0.35)).grid(row=0, column=5, padx=(0, 6))
        ttk.Button(actions, text="Full file", style="Soft.TButton", command=self.full_view).grid(row=0, column=6, padx=(0, 12))
        ttk.Label(actions, textvariable=self.event_count, style="Big.TLabel").grid(row=0, column=7, padx=(0, 14))
        ttk.Label(actions, textvariable=self.selection_count, style="Muted.TLabel").grid(row=0, column=8, sticky="w")

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
        ttk.Label(footer, text="Drag select · wheel zoom · arrows/scrollbar pan · Shift-click add · right-click remove · Space play", style="Muted.TLabel").pack(side="right")
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
            self.stop()
            self.audio = lab.load_wav(path)
            self.file_name.set(path.name)
            self.meta.set(f"{self.audio.duration_s:.2f} s · {self.audio.sample_rate:,} Hz")
            self.events = []
            self.view_start, self.view_end = 0.0, self.audio.duration_s
            self.sel_start = self.sel_end = None
            self.playhead_t = None
            self._recompute_curve()
            self.detect_full()
            self.status.set(f"Loaded {path.name}. Automatic clicks are marked in coral.")
            self._update_scrollbar()
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
            self.threshold.set(th)
            keep = [e for e in self.events if not (a <= e.timestamp_seconds <= b)]
            self.events = sorted(keep + new, key=lambda e: e.timestamp_seconds)
            self._refresh_values()
            self.refresh_labels()
            self.draw()
            self.status.set(f"Fitted selection to {len(new)} clicks · suggested threshold {th:.2f} dBFS")
        except Exception as exc:
            messagebox.showerror("Cannot fit selection", str(exc))

    def save_reference(self) -> None:
        if self.audio is None:
            return
        region = self.selection()
        if region is None:
            messagebox.showinfo("Select a region", "Select and review a region first.")
            return
        a, b = region
        selected = [e for e in self.events if a <= e.timestamp_seconds <= b]
        expected = int(self.expected.get())
        if len(selected) != expected and not messagebox.askyesno("Count mismatch", f"Expected {expected}, but {len(selected)} markers are in the selection. Save anyway?"):
            return
        value = filedialog.asksaveasfilename(title="Calibration library", defaultextension=".jsonl", initialfile="click_calibration.jsonl", filetypes=[("JSON Lines", "*.jsonl")])
        if value:
            lab.append_reference(Path(value), self.audio, a, b, expected, selected, self.threshold.get(), self.min_sep.get(), self.highpass.get())
            self.status.set(f"Added reviewed selection to {Path(value).name}")

    def export(self) -> None:
        if self.audio is None:
            return
        value = filedialog.asksaveasfilename(title="Export reviewed click events", defaultextension=".csv", initialfile="clicks.csv", filetypes=[("CSV", "*.csv")])
        if value:
            lab.export_events_csv(Path(value), self.events)
            self.status.set(f"Exported {len(self.events)} reviewed events to {Path(value).name}")

    def _x_to_time(self, x: float) -> float:
        w = max(1, self.canvas.winfo_width())
        return self.view_start + max(0.0, min(1.0, x / w)) * (self.view_end - self.view_start)

    def _time_to_x(self, t: float) -> float:
        return (t - self.view_start) / max(1e-12, self.view_end - self.view_start) * max(1, self.canvas.winfo_width())

    def _select_start(self, e) -> None:
        self.dragging = True
        self.sel_start = self._x_to_time(e.x)
        self.sel_end = self.sel_start
        self.draw()

    def _select_drag(self, e) -> None:
        if self.dragging:
            self.sel_end = self._x_to_time(e.x)
            self.refresh_labels()
            self.draw()

    def _select_end(self, e) -> None:
        if self.dragging:
            self.sel_end = self._x_to_time(e.x)
            self.dragging = False
            self.refresh_labels()
            self.draw()

    def _add_event(self, e) -> None:
        if self.audio is None or self.filtered is None or self.score is None:
            return
        sr = self.audio.sample_rate
        idx = int(round(self._x_to_time(e.x) * sr))
        radius = max(1, int(round(0.007 * sr)))
        lo = max(0, idx - radius)
        hi = min(len(self.filtered), idx + radius + 1)
        if hi <= lo:
            return
        refined = lo + int(np.argmax(np.abs(self.filtered[lo:hi])))
        amp = float(abs(self.filtered[refined]))
        self.events.append(lab.Event(refined / sr, float(self.score[refined]), amp, amp, "manual_added"))
        self.events.sort(key=lambda z: z.timestamp_seconds)
        self.refresh_labels()
        self.draw()

    def _remove_event(self, e) -> None:
        if not self.events:
            return
        t = self._x_to_time(e.x)
        tolerance = max(0.008, (self.view_end - self.view_start) * 0.012)
        i = min(range(len(self.events)), key=lambda j: abs(self.events[j].timestamp_seconds - t))
        if abs(self.events[i].timestamp_seconds - t) <= tolerance:
            self.events.pop(i)
            self.refresh_labels()
            self.draw()

    def _motion(self, e) -> None:
        self.hover_t = self._x_to_time(e.x) if self.audio is not None else None
        self.draw()

    def _leave(self) -> None:
        self.hover_t = None
        self.draw()

    def _wheel(self, e) -> None:
        if self.audio is None:
            return
        factor = 0.72 if e.delta > 0 else 1.38
        self.zoom_around(self._x_to_time(e.x), factor)

    def _double_zoom(self, e) -> None:
        if self.audio is not None:
            self.zoom_around(self._x_to_time(e.x), 0.35)

    def zoom_around(self, center: float, factor: float) -> None:
        if self.audio is None:
            return
        old = max(0.01, self.view_end - self.view_start)
        span = min(self.audio.duration_s, max(0.03, old * factor))
        ratio = (center - self.view_start) / old
        a = center - ratio * span
        a = min(max(0.0, a), max(0.0, self.audio.duration_s - span))
        self.view_start, self.view_end = a, a + span
        self._update_scrollbar()
        self.draw()

    def zoom_selection(self) -> None:
        if self.audio is None:
            return
        region = self.selection()
        if region is None:
            return
        a, b = region
        pad = max(0.02, (b - a) * 0.08)
        self.view_start = max(0.0, a - pad)
        self.view_end = min(self.audio.duration_s, b + pad)
        self._update_scrollbar()
        self.draw()

    def full_view(self) -> None:
        if self.audio is not None:
            self.view_start, self.view_end = 0.0, self.audio.duration_s
            self._update_scrollbar()
            self.draw()

    def pan_view(self, fraction_of_span: float) -> None:
        if self.audio is None:
            return
        span = self.view_end - self.view_start
        if span >= self.audio.duration_s:
            return
        shift = span * fraction_of_span
        start = min(max(0.0, self.view_start + shift), self.audio.duration_s - span)
        self.view_start, self.view_end = start, start + span
        self._update_scrollbar()
        self.draw()

    def _update_scrollbar(self) -> None:
        if self.audio is None or self.audio.duration_s <= 0:
            self.scroll.set(0.0, 1.0)
            return
        first = self.view_start / self.audio.duration_s
        last = self.view_end / self.audio.duration_s
        self.scroll.set(max(0.0, first), min(1.0, last))

    def _scroll_command(self, *args) -> None:
        if self.audio is None:
            return
        span = self.view_end - self.view_start
        if span >= self.audio.duration_s:
            return
        if args[0] == "moveto":
            frac = float(args[1])
            start = frac * self.audio.duration_s
            start = min(max(0.0, start), self.audio.duration_s - span)
        else:
            amount = int(args[1])
            units = args[2]
            step = span * (0.85 if units == "pages" else 0.10)
            start = self.view_start + amount * step
            start = min(max(0.0, start), self.audio.duration_s - span)
        self.view_start, self.view_end = start, start + span
        self._update_scrollbar()
        self.draw()

    def clear_selection(self) -> None:
        self.sel_start = self.sel_end = None
        self.refresh_labels()
        self.draw()

    def refresh_labels(self) -> None:
        self.event_count.set(f"{len(self.events)} clicks")
        region = self.selection()
        if region is None:
            self.selection_count.set("No selection")
        else:
            a, b = region
            n = sum(a <= ev.timestamp_seconds <= b for ev in self.events)
            self.selection_count.set(f"Selection {a:.3f}–{b:.3f} s · {n} markers")

    def _segment_for_playback(self) -> tuple[float, float] | None:
        if self.audio is None:
            return None
        region = self.selection()
        if region is not None:
            return region
        return self.view_start, self.view_end

    def _write_play_temp(self, a: float, b: float) -> str | None:
        if self.audio is None:
            return None
        sr = self.audio.sample_rate
        lo = max(0, int(round(a * sr)))
        hi = min(len(self.audio.mono), int(round(b * sr)))
        if hi <= lo:
            return None
        y = np.clip(self.audio.mono[lo:hi], -1.0, 1.0)
        pcm = np.round(y * 32767).astype(np.int16)
        fd, name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        wavfile.write(name, sr, pcm)
        self._temp_files.append(name)
        return name

    def _start_backend(self, a: float, b: float) -> bool:
        name = self._write_play_temp(a, b)
        if not name:
            return False
        try:
            system = platform.system()
            if system == "Windows":
                import winsound
                winsound.PlaySound(name, winsound.SND_FILENAME | winsound.SND_ASYNC)
                self._play_proc = None
            elif system == "Darwin":
                self._play_proc = subprocess.Popen(["afplay", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self._play_proc = subprocess.Popen(["aplay", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as exc:
            messagebox.showerror("Playback unavailable", str(exc))
            return False

    def _stop_backend(self) -> None:
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.PlaySound(None, 0)
            elif self._play_proc is not None and self._play_proc.poll() is None:
                self._play_proc.terminate()
        except Exception:
            pass
        self._play_proc = None

    def _current_play_time(self) -> float:
        if not self.playing:
            return self.playhead_t if self.playhead_t is not None else self.play_start_t
        elapsed = max(0.0, time.monotonic() - self.play_wall)
        return min(self.play_end_t, self.play_anchor_t + elapsed)

    def play(self) -> None:
        if self.audio is None:
            return
        if self.paused and self.playhead_t is not None:
            start = self.playhead_t
            end = self.play_end_t
        else:
            segment = self._segment_for_playback()
            if segment is None:
                return
            start, end = segment
            self.play_start_t, self.play_end_t = start, end
            self.playhead_t = start
        if end <= start:
            return
        self._stop_backend()
        if self._start_backend(start, end):
            self.play_anchor_t = start
            self.play_wall = time.monotonic()
            self.playing = True
            self.paused = False
            self.status.set(f"Playing {start:.3f}–{end:.3f} s")
            self._schedule_play_tick()

    def pause(self) -> None:
        if self.audio is None:
            return
        if self.playing:
            self.playhead_t = self._current_play_time()
            self._stop_backend()
            self.playing = False
            self.paused = True
            self.status.set(f"Paused at {self.playhead_t:.3f} s · press Play to resume")
            self.draw()
        elif self.paused:
            self.play()

    def stop(self) -> None:
        if self._play_after_id is not None:
            try:
                self.after_cancel(self._play_after_id)
            except Exception:
                pass
            self._play_after_id = None
        self._stop_backend()
        was_active = self.playing or self.paused
        self.playing = False
        self.paused = False
        if was_active:
            self.playhead_t = self.play_start_t
            self.status.set("Playback stopped")
            self.draw()

    def _schedule_play_tick(self) -> None:
        if self._play_after_id is None:
            self._play_after_id = self.after(40, self._play_tick)

    def _play_tick(self) -> None:
        self._play_after_id = None
        if not self.playing:
            return
        self.playhead_t = self._current_play_time()
        if self.playhead_t >= self.play_end_t - 1e-4:
            self._stop_backend()
            self.playing = False
            self.paused = False
            self.playhead_t = self.play_end_t
            self.status.set("Playback finished")
            self.draw()
            return

        span = self.view_end - self.view_start
        if self.audio is not None and span < self.audio.duration_s:
            trigger = self.view_start + 0.82 * span
            if self.playhead_t >= trigger:
                new_start = self.playhead_t - 0.22 * span
                new_start = min(max(0.0, new_start), self.audio.duration_s - span)
                self.view_start, self.view_end = new_start, new_start + span
                self._update_scrollbar()
        self.draw()
        self._schedule_play_tick()

    def draw(self) -> None:
        c = self.canvas
        c.delete("all")
        w = max(10, c.winfo_width())
        h = max(10, c.winfo_height())
        if self.audio is None:
            c.create_text(w / 2, h / 2, text="Open a WAV to begin", fill=MUTED, font=("TkDefaultFont", 15))
            return

        sr = self.audio.sample_rate
        lo = max(0, int(self.view_start * sr))
        hi = min(len(self.audio.mono), int(self.view_end * sr))
        if hi <= lo:
            return

        wave_top = 44
        wave_bottom = h * 0.72
        score_top = h * 0.78
        score_bottom = h - 30

        # Time ruler.
        span = self.view_end - self.view_start
        ticks = 8
        for i in range(ticks + 1):
            x = i * w / ticks
            t = self.view_start + i * span / ticks
            c.create_line(x, 26, x, h, fill=GRID)
            c.create_text(x + 4, 16, text=f"{t:.2f}s", anchor="w", fill=MUTED, font=("TkDefaultFont", 8))

        # Min/max waveform rendering keeps long recordings readable.
        samples = self.audio.mono[lo:hi]
        cols = min(w, max(1, len(samples)))
        edges = np.linspace(0, len(samples), cols + 1).astype(int)
        peak = max(float(np.max(np.abs(samples))), 1e-9)
        mid = (wave_top + wave_bottom) / 2
        amp = (wave_bottom - wave_top) * 0.43
        for x in range(cols):
            a, b = edges[x], edges[x + 1]
            if b <= a:
                continue
            seg = samples[a:b]
            ymin = mid - (float(np.max(seg)) / peak) * amp
            ymax = mid - (float(np.min(seg)) / peak) * amp
            c.create_line(x, ymin, x, ymax, fill=WAVE)
        c.create_line(0, mid, w, mid, fill="#d9d3c8")

        # Transient-strength lane.
        if self.score is not None:
            idx = np.linspace(lo, hi - 1, min(max(300, w), hi - lo)).astype(int)
            vals = np.clip(self.score[idx], -100.0, -5.0)
            pts = []
            for j, v in enumerate(vals):
                x = j / max(1, len(vals) - 1) * w
                y = score_bottom - (float(v) + 100.0) / 95.0 * (score_bottom - score_top)
                pts.extend((x, y))
            if len(pts) >= 4:
                c.create_line(*pts, fill=TEAL, width=1)
            th = float(self.threshold.get())
            thy = score_bottom - (max(-100.0, min(-5.0, th)) + 100.0) / 95.0 * (score_bottom - score_top)
            c.create_line(0, thy, w, thy, fill=AMBER, dash=(5, 4))
            c.create_text(8, score_top + 3, text="transient strength", anchor="nw", fill=MUTED, font=("TkDefaultFont", 8))

        # Selection first, behind markers.
        region = self.selection()
        if region is not None:
            a, b = region
            if b >= self.view_start and a <= self.view_end:
                x0 = self._time_to_x(max(a, self.view_start))
                x1 = self._time_to_x(min(b, self.view_end))
                c.create_rectangle(x0, 27, x1, h, fill=TEAL_SOFT, outline=TEAL, stipple="gray25")

        # Event markers: compact at full scale, larger once zoomed.
        dense = span > 25.0
        for ev in self.events:
            if self.view_start <= ev.timestamp_seconds <= self.view_end:
                x = self._time_to_x(ev.timestamp_seconds)
                col = PURPLE if ev.source == "manual_added" else CORAL
                if dense:
                    c.create_line(x, wave_top, x, wave_top + 13, fill=col, width=1)
                else:
                    c.create_line(x, wave_top, x, wave_bottom, fill=col, width=2)
                    c.create_oval(x - 3, wave_top - 4, x + 3, wave_top + 2, fill=col, outline="")

        if self.playhead_t is not None and self.view_start <= self.playhead_t <= self.view_end:
            x = self._time_to_x(self.playhead_t)
            c.create_line(x, 27, x, h, fill=PLAYHEAD, width=2)
            c.create_text(x + 5, 31, text=f"{self.playhead_t:.3f}s", anchor="nw", fill=PLAYHEAD, font=("TkDefaultFont", 9, "bold"))

        if self.hover_t is not None and self.view_start <= self.hover_t <= self.view_end:
            x = self._time_to_x(self.hover_t)
            c.create_line(x, 27, x, h, fill="#b5afa5", dash=(2, 3))

        c.create_text(w - 8, h - 8, text=f"view {self.view_start:.3f}–{self.view_end:.3f}s", anchor="se", fill=MUTED, font=("TkDefaultFont", 8))

    def close(self) -> None:
        self.stop()
        for name in self._temp_files:
            try:
                os.unlink(name)
            except OSError:
                pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
