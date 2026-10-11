#!/usr/bin/env python3
"""Desktop app for locked-protocol study analysis and artwork preview/export."""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import study_session_analysis as study
import study_visualizer as visual

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = REPO_ROOT / "calibration" / "calibration_profile.json"
DEFAULT_PROTOCOL = REPO_ROOT / "study" / "study_protocol.json"


class StudySessionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector - Study Analysis + Visualiser")
        self.geometry("1180x820")
        self.minsize(980, 700)

        self.protocol = study.load_protocol(DEFAULT_PROTOCOL)
        rules = self.protocol["analysis"]
        self.current_output: Path | None = None
        self.visual_data: visual.VisualData | None = None

        self.wav_var = tk.StringVar()
        self.profile_var = tk.StringVar(value=str(DEFAULT_PROFILE))
        self.output_var = tk.StringVar()
        self.participant_var = tk.StringVar()
        self.trial_var = tk.StringVar(value="1")
        self.outward_position_var = tk.StringVar(value="A")
        self.return_position_var = tk.StringVar(value="B")
        self.turnaround_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

        self.style_var = tk.StringVar(value=visual.STYLES[0])
        self.theme_var = tk.StringVar(value=visual.THEMES[1])
        self.density_var = tk.StringVar(value="Medium")
        self.smoothing_var = tk.StringVar(value="Medium")
        self.annotated_var = tk.BooleanVar(value=False)
        self.visual_status_var = tk.StringVar(value="Run an analysis or load an existing study output.")

        self.protocol_text = (
            f"{self.protocol['protocol_id']}  |  {rules['spoke_count']} spokes  |  "
            f"central {rules['primary_comparison_intervals_per_leg']} intervals per leg  |  "
            f"{rules['equivalent_wheel_revolutions_per_leg']:.1f} wheel revolutions per leg"
        )

        self._build()

    def _build(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.analysis_tab = ttk.Frame(notebook, padding=14)
        self.visual_tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.analysis_tab, text="1. Session Analysis")
        notebook.add(self.visual_tab, text="2. Data Visualiser")

        self._build_analysis_tab(self.analysis_tab)
        self._build_visual_tab(self.visual_tab)

    def _build_analysis_tab(self, root: ttk.Frame) -> None:
        root.columnconfigure(1, weight=1)
        root.rowconfigure(10, weight=1)

        ttk.Label(root, text="Standardised Study Session", font=("TkDefaultFont", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        ttk.Label(
            root,
            text=(
                "Capture and analysis rules are locked so every subject is processed with the same spoke geometry "
                "and physical-equivalent comparison window. Visualisation choices never alter those scientific rules."
            ),
            wraplength=1080,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))

        protocol_box = ttk.LabelFrame(root, text="Locked study protocol", padding=10)
        protocol_box.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        ttk.Label(protocol_box, text=self.protocol_text, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            protocol_box,
            text="Primary comparison excludes the turnaround and uses the same central wheel-travel window for every participant.",
            wraplength=1040,
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

        button_row = ttk.Frame(root)
        button_row.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(8, 10))
        self.run_btn = ttk.Button(button_row, text="Analyse Standardised Session", command=self._start_analysis)
        self.run_btn.pack(side="left")
        ttk.Button(button_row, text="Load Existing Analysis for Visualiser", command=self._load_existing_output).pack(side="left", padx=(8, 0))
        ttk.Label(button_row, textvariable=self.status_var).pack(side="left", padx=(14, 0))

        ttk.Label(root, text="Results").grid(row=9, column=0, columnspan=3, sticky="w")
        self.results = tk.Text(root, height=14, wrap="word")
        self.results.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=(5, 0))

    def _build_visual_tab(self, root: ttk.Frame) -> None:
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(root, text="Visualisation controls", padding=10)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(controls, text="Style").grid(row=0, column=0, sticky="w")
        style_box = ttk.Combobox(controls, textvariable=self.style_var, values=visual.STYLES, state="readonly", width=24)
        style_box.grid(row=0, column=1, padx=(6, 18), sticky="w")

        ttk.Label(controls, text="Preview").grid(row=0, column=2, sticky="w")
        theme_box = ttk.Combobox(controls, textvariable=self.theme_var, values=visual.THEMES, state="readonly", width=16)
        theme_box.grid(row=0, column=3, padx=(6, 18), sticky="w")

        ttk.Label(controls, text="Line density").grid(row=0, column=4, sticky="w")
        density_box = ttk.Combobox(controls, textvariable=self.density_var, values=visual.DENSITIES, state="readonly", width=10)
        density_box.grid(row=0, column=5, padx=(6, 18), sticky="w")

        ttk.Label(controls, text="Smoothing").grid(row=0, column=6, sticky="w")
        smoothing_box = ttk.Combobox(controls, textvariable=self.smoothing_var, values=visual.SMOOTHING, state="readonly", width=10)
        smoothing_box.grid(row=0, column=7, padx=(6, 18), sticky="w")

        ttk.Checkbutton(controls, text="Show brief gait analysis", variable=self.annotated_var, command=self._refresh_preview).grid(
            row=0, column=8, sticky="w"
        )

        for box in (style_box, theme_box, density_box, smoothing_box):
            box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_preview())

        preview_frame = ttk.LabelFrame(root, text="Live preview", padding=6)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.preview_canvas = tk.Canvas(preview_frame, highlightthickness=0, background="#111111")
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", lambda _event: self._refresh_preview())

        side = ttk.Frame(root)
        side.grid(row=1, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)

        brief = ttk.LabelFrame(side, text="Brief gait analysis", padding=10)
        brief.grid(row=0, column=0, sticky="ew")
        self.brief_text = tk.Text(brief, width=37, height=12, wrap="word")
        self.brief_text.pack(fill="both", expand=True)
        self.brief_text.insert("1.0", "No analysed session loaded.")
        self.brief_text.configure(state="disabled")

        exports = ttk.LabelFrame(side, text="Artwork export", padding=10)
        exports.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(exports, text="Export Current SVG...", command=self._export_current_svg).pack(fill="x")
        ttk.Button(exports, text="Export All 3 Default Styles", command=self._export_all_styles).pack(fill="x", pady=(6, 0))
        ttk.Label(
            exports,
            text=(
                "SVG export uses the same locked central data window as the study analysis. "
                "Style, density and smoothing affect presentation only."
            ),
            wraplength=300,
        ).pack(fill="x", pady=(8, 0))

        ttk.Label(side, textvariable=self.visual_status_var, wraplength=320).grid(row=2, column=0, sticky="ew", pady=(10, 0))

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
            if not self.output_var.get().strip():
                raise ValueError("Choose an output folder.")
            out = Path(self.output_var.get()).expanduser()
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
                visual.export_all_default_styles(out)
                self.after(0, lambda: self._show_success(summary, out))
            except Exception:
                err = traceback.format_exc()
                self.after(0, lambda: self._show_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _show_success(self, summary: dict, out: Path) -> None:
        self.run_btn.configure(state="normal")
        self.status_var.set("Complete - comparable output + artwork written")
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
            lines += ["Primary outward/return comparison: unavailable", f"Reason: {comp.get('reason')}"]

        lines += [
            "",
            f"Output folder: {out}",
            f"Standard subject record: {out / study.SUBJECT_RECORD_FILENAME}",
            f"Study summary: {out / study.STUDY_SUMMARY_FILENAME}",
            f"Artwork folder: {out / 'artwork'}",
        ]
        self.results.insert("1.0", "\n".join(lines))
        self._activate_visualiser(out)
        messagebox.showinfo(
            "Analysis complete",
            "Study analysis completed. Open the Data Visualiser tab to preview and export artwork styles.",
        )

    def _show_error(self, error_text: str) -> None:
        self.run_btn.configure(state="normal")
        self.status_var.set("Failed")
        self.results.insert("1.0", error_text)
        messagebox.showerror("Analysis failed", "The analysis failed. Details are shown in the Results box.")

    def _load_existing_output(self) -> None:
        value = filedialog.askdirectory(title="Choose an existing study output folder", initialdir=str(REPO_ROOT / "study_output"))
        if not value:
            return
        try:
            self._activate_visualiser(Path(value))
            self.status_var.set("Existing analysis loaded for visualisation")
        except Exception as exc:
            messagebox.showerror("Cannot load visualiser data", str(exc))

    def _activate_visualiser(self, output_dir: Path) -> None:
        self.current_output = output_dir
        self.visual_data = visual.load_visual_data(output_dir)
        self.visual_status_var.set(f"Loaded {self.visual_data.participant_id}, trial {self.visual_data.trial_id}")
        self._update_brief_text()
        self._refresh_preview()

    def _update_brief_text(self) -> None:
        self.brief_text.configure(state="normal")
        self.brief_text.delete("1.0", "end")
        if self.visual_data is None:
            self.brief_text.insert("1.0", "No analysed session loaded.")
        else:
            lines = [
                f"Participant: {self.visual_data.participant_id}",
                f"Trial: {self.visual_data.trial_id}",
                f"Handlebar: {self.visual_data.outward_position} -> {self.visual_data.return_position}",
                f"Locked window: {self.visual_data.intervals_per_leg} intervals / leg",
                "",
            ] + visual.brief_analysis_lines(self.visual_data) + [
                "",
                "Interpretation note:",
                "Spoke timing is a wheel-motion / movement-rhythm proxy. It is not a direct heel-strike or toe-off measurement.",
            ]
            self.brief_text.insert("1.0", "\n".join(lines))
        self.brief_text.configure(state="disabled")

    def _refresh_preview(self) -> None:
        if not hasattr(self, "preview_canvas"):
            return
        if self.visual_data is None:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                max(320, self.preview_canvas.winfo_width() / 2),
                max(240, self.preview_canvas.winfo_height() / 2),
                text="Analyse a session or load an existing output to preview artwork.",
                fill="#aaaaaa",
                font=("TkDefaultFont", 13),
            )
            return
        visual.draw_on_canvas(
            self.preview_canvas,
            self.visual_data,
            self.style_var.get(),
            self.theme_var.get(),
            self.density_var.get(),
            self.smoothing_var.get(),
            self.annotated_var.get(),
        )

    def _export_current_svg(self) -> None:
        if self.visual_data is None:
            messagebox.showerror("Nothing to export", "Load or analyse a study session first.")
            return
        default_name = self.style_var.get().lower().replace(" ", "_") + ("_annotated" if self.annotated_var.get() else "") + ".svg"
        path = filedialog.asksaveasfilename(
            title="Export print-ready SVG",
            defaultextension=".svg",
            initialfile=default_name,
            filetypes=[("SVG vector artwork", "*.svg")],
        )
        if not path:
            return
        visual.export_svg(
            Path(path),
            self.visual_data,
            self.style_var.get(),
            theme=self.theme_var.get(),
            density=self.density_var.get(),
            smoothing=self.smoothing_var.get(),
            annotated=self.annotated_var.get(),
            transparent=True,
        )
        self.visual_status_var.set(f"Exported {Path(path).name}")

    def _export_all_styles(self) -> None:
        if self.visual_data is None or self.current_output is None:
            messagebox.showerror("Nothing to export", "Load or analyse a study session first.")
            return
        paths = visual.export_all_default_styles(self.current_output)
        self.visual_status_var.set(f"Exported {len(paths)} SVG files to {self.current_output / 'artwork'}")
        messagebox.showinfo("Artwork exported", f"Artwork written to:\n{self.current_output / 'artwork'}")


if __name__ == "__main__":
    StudySessionApp().mainloop()
