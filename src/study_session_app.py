#!/usr/bin/env python3
"""One-button desktop app for locked-protocol study session analysis."""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import study_session_analysis as study

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = REPO_ROOT / "calibration" / "calibration_profile.json"
DEFAULT_PROTOCOL = REPO_ROOT / "study" / "study_protocol.json"


class StudySessionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector - Standardised Study Session")
        self.geometry("900x680")
        self.minsize(800, 600)

        self.protocol = study.load_protocol(DEFAULT_PROTOCOL)
        rules = self.protocol["analysis"]

        self.wav_var = tk.StringVar()
        self.profile_var = tk.StringVar(value=str(DEFAULT_PROFILE))
        self.output_var = tk.StringVar()
        self.participant_var = tk.StringVar()
        self.trial_var = tk.StringVar(value="1")
        self.outward_position_var = tk.StringVar(value="A")
        self.return_position_var = tk.StringVar(value="B")
        self.turnaround_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.protocol_text = (
            f"{self.protocol['protocol_id']}  |  {rules['spoke_count']} spokes  |  "
            f"central {rules['primary_comparison_intervals_per_leg']} intervals per leg  |  "
            f"{rules['equivalent_wheel_revolutions_per_leg']:.1f} wheel revolutions per leg"
        )

        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)

        ttk.Label(root, text="Standardised Study Session", font=("TkDefaultFont", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        ttk.Label(
            root,
            text=(
                "Use this app for participant recordings. The study settings are locked so every subject "
                "is analysed with the same spoke geometry and the same physical-equivalent comparison window."
            ),
            wraplength=840,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))

        protocol_box = ttk.LabelFrame(root, text="Locked study protocol", padding=10)
        protocol_box.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        ttk.Label(protocol_box, text=self.protocol_text, font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            protocol_box,
            text=(
                "Primary comparison excludes the turnaround and uses the same central wheel-travel window "
                "for every participant. Change the protocol file only before data collection, not subject-by-subject."
            ),
            wraplength=820,
        ).grid(row=1, column=0, sticky="w", pady=(5, 0))

        meta = ttk.LabelFrame(root, text="Required participant metadata", padding=10)
        meta.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        meta.columnconfigure(1, weight=1)
        meta.columnconfigure(3, weight=1)

        ttk.Label(meta, text="Participant ID").grid(row=0, column=0, sticky="w")
        ttk.Entry(meta, textvariable=self.participant_var, width=20).grid(row=0, column=1, sticky="ew", padx=(6, 18))
        ttk.Label(meta, text="Trial ID").grid(row=0, column=2, sticky="w")
        ttk.Entry(meta, textvariable=self.trial_var, width=12).grid(row=0, column=3, sticky="w", padx=(6, 0))

        ttk.Label(meta, text="Outward handlebar position").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(meta, textvariable=self.outward_position_var, values=("A", "B"), state="readonly", width=8).grid(
            row=1, column=1, sticky="w", padx=(6, 18), pady=(8, 0)
        )
        ttk.Label(meta, text="Return handlebar position").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Combobox(meta, textvariable=self.return_position_var, values=("A", "B"), state="readonly", width=8).grid(
            row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0)
        )
        ttk.Button(meta, text="Swap A/B", command=self._swap_positions).grid(row=1, column=4, padx=(12, 0), pady=(8, 0))

        self._path_row(root, 4, "Audio WAV", self.wav_var, self._choose_wav)
        self._path_row(root, 5, "Calibration profile", self.profile_var, self._choose_profile)
        self._path_row(root, 6, "Output folder", self.output_var, self._choose_output)

        opts = ttk.LabelFrame(root, text="Optional protocol note", padding=10)
        opts.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(12, 10))
        ttk.Label(opts, text="Manual turnaround time, seconds (leave blank for automatic split)").grid(row=0, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.turnaround_var, width=12).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.run_btn = ttk.Button(root, text="Analyse Standardised Session", command=self._start_analysis)
        self.run_btn.grid(row=8, column=0, sticky="w", pady=(8, 10))
        ttk.Label(root, textvariable=self.status_var).grid(row=8, column=1, columnspan=2, sticky="w", padx=(12, 0))

        ttk.Label(root, text="Results").grid(row=9, column=0, columnspan=3, sticky="w")
        self.results = tk.Text(root, height=14, wrap="word")
        self.results.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=(5, 0))
        root.rowconfigure(10, weight=1)

    def _path_row(self, parent, row: int, label: str, var: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(parent, text="Browse...", command=command).grid(row=row, column=2, pady=4)

    def _swap_positions(self) -> None:
        a = self.outward_position_var.get()
        b = self.return_position_var.get()
        self.outward_position_var.set(b)
        self.return_position_var.set(a)

    def _choose_wav(self) -> None:
        value = filedialog.askopenfilename(
            title="Choose participant WAV",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
            initialdir=str(REPO_ROOT / "audio"),
        )
        if value:
            self.wav_var.set(value)
            self._suggest_output()

    def _suggest_output(self) -> None:
        participant = self.participant_var.get().strip()
        trial = self.trial_var.get().strip()
        if participant and trial:
            self.output_var.set(str(REPO_ROOT / "study_output" / participant / f"trial_{trial}"))
        elif self.wav_var.get() and not self.output_var.get():
            p = Path(self.wav_var.get())
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
            participant = self.participant_var.get().strip()
            trial = self.trial_var.get().strip()
            outward_position = self.outward_position_var.get().strip()
            return_position = self.return_position_var.get().strip()
            turn_text = self.turnaround_var.get().strip()
            turn = None if not turn_text else float(turn_text)

            if not wav.exists():
                raise ValueError("Choose an existing WAV file.")
            if not profile.exists():
                raise ValueError("Choose an existing calibration profile.")
            if not participant:
                raise ValueError("Enter a participant ID.")
            if not trial:
                raise ValueError("Enter a trial ID.")
            if outward_position == return_position:
                raise ValueError("Outward and return handlebar positions must differ.")
            if not self.output_var.get().strip():
                self._suggest_output()
                out = Path(self.output_var.get()).expanduser()
                if not self.output_var.get().strip():
                    raise ValueError("Choose an output folder.")
        except Exception as exc:
            messagebox.showerror("Cannot start", str(exc))
            return

        self.run_btn.configure(state="disabled")
        self.status_var.set("Analysing with locked protocol...")
        self.results.delete("1.0", "end")

        def worker() -> None:
            try:
                summary = study.analyse_study_session(
                    wav,
                    out,
                    profile,
                    participant_id=participant,
                    trial_id=trial,
                    outward_handlebar_position=outward_position,
                    return_handlebar_position=return_position,
                    protocol_path=DEFAULT_PROTOCOL,
                    manual_turnaround_s=turn,
                )
                self.after(0, lambda: self._show_success(summary, out))
            except Exception:
                err = traceback.format_exc()
                self.after(0, lambda: self._show_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _show_success(self, summary: dict, out: Path) -> None:
        self.run_btn.configure(state="normal")
        self.status_var.set("Complete - protocol-comparable output written")
        comp = summary.get("standardised_primary_comparison") or {}
        split = summary.get("phase_split") or {}
        det = summary.get("detector") or {}
        subject = summary.get("subject") or {}

        lines = [
            f"Participant: {subject.get('participant_id')}    Trial: {subject.get('trial_id')}",
            f"Handlebar order: {subject.get('outward_handlebar_position')} -> {subject.get('return_handlebar_position')}",
            f"Protocol: {summary.get('protocol', {}).get('protocol_id')}",
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
            f"Standard subject record: {out / study.SUBJECT_RECORD_FILENAME}",
            f"Study summary: {out / study.STUDY_SUMMARY_FILENAME}",
            f"Movement plot: {out / 'movement' / 'movement_plot.svg'}",
            f"Review WAV: {out / 'detection' / 'session_isolated.wav'}",
        ]
        self.results.insert("1.0", "\n".join(lines))
        messagebox.showinfo(
            "Analysis complete",
            "Standardised study analysis completed. A one-row subject record has been written for fair cross-subject comparison.",
        )

    def _show_error(self, error_text: str) -> None:
        self.run_btn.configure(state="normal")
        self.status_var.set("Failed")
        self.results.insert("1.0", error_text)
        messagebox.showerror("Analysis failed", "The analysis failed. Details are shown in the Results box.")


if __name__ == "__main__":
    StudySessionApp().mainloop()
