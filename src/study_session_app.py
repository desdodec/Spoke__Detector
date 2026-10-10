#!/usr/bin/env python3
"""One-button desktop app for standardised study session analysis."""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import study_session_analysis as study

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = REPO_ROOT / "calibration" / "calibration_profile.json"


class StudySessionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector - Study Session Analysis")
        self.geometry("860x590")
        self.minsize(760, 520)

        self.wav_var = tk.StringVar()
        self.profile_var = tk.StringVar(value=str(DEFAULT_PROFILE))
        self.output_var = tk.StringVar()
        self.spokes_var = tk.StringVar(value="36")
        self.window_var = tk.StringVar(value=str(study.DEFAULT_COMPARISON_INTERVALS))
        self.turnaround_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)

        ttk.Label(root, text="Study Session Analysis", font=("TkDefaultFont", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )
        ttk.Label(
            root,
            text=(
                "Detects spoke clicks, identifies outward / turnaround / return, "
                "and compares an equal central wheel-travel window on both walking legs."
            ),
            wraplength=780,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 14))

        self._path_row(root, 2, "Audio WAV", self.wav_var, self._choose_wav)
        self._path_row(root, 3, "Calibration profile", self.profile_var, self._choose_profile)
        self._path_row(root, 4, "Output folder", self.output_var, self._choose_output)

        opts = ttk.LabelFrame(root, text="Standardised study settings", padding=10)
        opts.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(14, 10))
        opts.columnconfigure(5, weight=1)

        ttk.Label(opts, text="Spokes").grid(row=0, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.spokes_var, width=8).grid(row=0, column=1, padx=(6, 20))
        ttk.Label(opts, text="Central intervals per leg").grid(row=0, column=2, sticky="w")
        ttk.Entry(opts, textvariable=self.window_var, width=10).grid(row=0, column=3, padx=(6, 20))
        ttk.Label(opts, text="Manual turnaround s (optional)").grid(row=0, column=4, sticky="w")
        ttk.Entry(opts, textvariable=self.turnaround_var, width=12).grid(row=0, column=5, padx=(6, 0), sticky="w")

        ttk.Label(
            opts,
            text=(
                "Default primary comparison = central 180 inter-click intervals per leg "
                "(5 wheel revolutions for 36 spokes). The turnaround is excluded."
            ),
            wraplength=780,
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(8, 0))

        self.run_btn = ttk.Button(root, text="Analyse Session", command=self._start_analysis)
        self.run_btn.grid(row=6, column=0, sticky="w", pady=(8, 10))
        ttk.Label(root, textvariable=self.status_var).grid(row=6, column=1, columnspan=2, sticky="w", padx=(12, 0))

        ttk.Label(root, text="Results").grid(row=7, column=0, columnspan=3, sticky="w")
        self.results = tk.Text(root, height=16, wrap="word")
        self.results.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(5, 0))
        root.rowconfigure(8, weight=1)

    def _path_row(self, parent, row: int, label: str, var: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(parent, text="Browse...", command=command).grid(row=row, column=2, pady=4)

    def _choose_wav(self) -> None:
        value = filedialog.askopenfilename(
            title="Choose participant WAV",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
            initialdir=str(REPO_ROOT / "audio"),
        )
        if value:
            self.wav_var.set(value)
            if not self.output_var.get():
                p = Path(value)
                self.output_var.set(str(REPO_ROOT / "study_output" / p.stem))

    def _choose_profile(self) -> None:
        value = filedialog.askopenfilename(
            title="Choose calibration profile",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialdir=str(REPO_ROOT / "calibration"),
        )
        if value:
            self.profile_var.set(value)

    def _choose_output(self) -> None:
        value = filedialog.askdirectory(title="Choose output folder", initialdir=str(REPO_ROOT))
        if value:
            self.output_var.set(value)

    def _start_analysis(self) -> None:
        try:
            wav = Path(self.wav_var.get()).expanduser()
            profile = Path(self.profile_var.get()).expanduser()
            out = Path(self.output_var.get()).expanduser()
            spokes = int(self.spokes_var.get())
            window = int(self.window_var.get())
            turn_text = self.turnaround_var.get().strip()
            turn = None if not turn_text else float(turn_text)
            if not wav.exists():
                raise ValueError("Choose an existing WAV file.")
            if not profile.exists():
                raise ValueError("Choose an existing calibration profile.")
            if not str(out):
                raise ValueError("Choose an output folder.")
            if spokes <= 0 or window <= 0:
                raise ValueError("Spokes and comparison intervals must be positive.")
        except Exception as exc:
            messagebox.showerror("Cannot start", str(exc))
            return

        self.run_btn.configure(state="disabled")
        self.status_var.set("Analysing...")
        self.results.delete("1.0", "end")

        def worker() -> None:
            try:
                summary = study.analyse_study_session(
                    wav, out, profile, spokes=spokes,
                    comparison_intervals=window,
                    manual_turnaround_s=turn,
                )
                self.after(0, lambda: self._show_success(summary, out))
            except Exception:
                err = traceback.format_exc()
                self.after(0, lambda: self._show_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _show_success(self, summary: dict, out: Path) -> None:
        self.run_btn.configure(state="normal")
        self.status_var.set("Complete")
        comp = summary.get("standardised_primary_comparison") or {}
        split = summary.get("phase_split") or {}
        det = summary.get("detector") or {}

        lines = [
            f"Detected clicks: {det.get('detected_hits')}",
            f"Detector quality: {det.get('quality_status')}",
            f"Turnaround split: {split.get('source')}",
        ]
        if comp.get("available"):
            out_m = comp.get("outward") or {}
            ret_m = comp.get("return") or {}
            lines += [
                f"Primary comparison window: {comp.get('requested_intervals_per_leg')} intervals per leg",
                f"Equivalent wheel revolutions per leg: {comp.get('equivalent_wheel_revolutions_per_leg'):.2f}",
                f"Outward median click rate: {out_m.get('median_click_rate_hz'):.3f} Hz",
                f"Return median click rate: {ret_m.get('median_click_rate_hz'):.3f} Hz",
                f"Return vs outward change: {comp.get('return_vs_outward_click_rate_change_percent'):.2f}%",
            ]
        else:
            lines += [
                "Primary outward/return comparison: unavailable",
                f"Reason: {comp.get('reason')}",
            ]

        lines += [
            "",
            f"Output folder: {out}",
            f"Study summary: {out / study.STUDY_SUMMARY_FILENAME}",
            f"Movement plot: {out / 'movement' / 'movement_plot.svg'}",
            f"Review WAV: {out / 'detection' / 'session_isolated.wav'}",
        ]
        self.results.insert("1.0", "\n".join(lines))
        messagebox.showinfo("Analysis complete", "Study session analysis completed successfully.")

    def _show_error(self, error_text: str) -> None:
        self.run_btn.configure(state="normal")
        self.status_var.set("Failed")
        self.results.insert("1.0", error_text)
        messagebox.showerror("Analysis failed", "The analysis failed. Details are shown in the Results box.")


if __name__ == "__main__":
    StudySessionApp().mainloop()
