#!/usr/bin/env python3
"""Desktop design laboratory for art-directed Spoke Detector movement portraits."""
from __future__ import annotations

import os
import platform
import subprocess
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

import design_laboratory as lab
import study_visualizer as visual


class DesignLaboratoryApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector - Design Laboratory")
        self.geometry("1680x1040")
        self.minsize(1240, 800)

        self.output_dir: Path | None = None
        self.visual_data: visual.VisualData | None = None
        self.contact_sheet_path: Path | None = None
        self._sheet_image = None

        self.folder_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Choose a completed study output folder.")
        self.count_var = tk.IntVar(value=24)
        self._build()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=12)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Design Laboratory", font=("TkDefaultFont", 19, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(
            header,
            text=(
                "Art direction sits above the locked measurement layer. Every candidate uses the same central baseline slice; "
                "the same-count central return slice is retained for paired movement-rhythm analysis."
            ),
            wraplength=1500,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 10))

        ttk.Label(header, text="Study output").grid(row=2, column=0, sticky="w")
        ttk.Entry(header, textvariable=self.folder_var).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(header, text="Browse...", command=self._choose_folder).grid(row=2, column=2)
        ttk.Button(header, text="Load", command=self._load_folder).grid(row=2, column=3, padx=(6, 0))

        controls = ttk.Frame(self, padding=(12, 0, 12, 8))
        controls.grid(row=1, column=0, sticky="ew")
        ttk.Label(controls, text="Candidate count").pack(side="left")
        ttk.Spinbox(controls, from_=6, to=40, increment=1, textvariable=self.count_var, width=6).pack(side="left", padx=(6, 12))
        self.generate_btn = ttk.Button(controls, text="Generate Art-Directed Laboratory", command=self._generate, state="disabled")
        self.generate_btn.pack(side="left")
        ttk.Button(controls, text="Open Laboratory Folder", command=self._open_lab_folder).pack(side="left", padx=(8, 0))
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=(16, 0))

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))

        left = ttk.Frame(paned, padding=6)
        right = ttk.Frame(paned, padding=6, width=390)
        paned.add(left, weight=5)
        paned.add(right, weight=2)

        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        ttk.Label(left, text="Laboratory contact sheet", font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.canvas = tk.Canvas(left, background="#eceae4", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._show_sheet())

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        analysis_box = ttk.LabelFrame(right, text="Paired central-slice movement analysis", padding=8)
        analysis_box.grid(row=0, column=0, sticky="nsew")
        self.analysis_text = tk.Text(analysis_box, width=44, height=22, wrap="word")
        self.analysis_text.pack(fill="both", expand=True)
        self.analysis_text.insert("1.0", "Load an analysed participant to compare the same central slice from outward and return legs.")
        self.analysis_text.configure(state="disabled")

        principles = ttk.LabelFrame(right, text="Design-lab rules", padding=8)
        principles.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        text = (
            "• Baseline artwork always uses the locked central interval count.\n"
            "• The return comparison uses the same count from the centre of the return leg.\n"
            "• Art direction changes composition, palette, density and surface treatment only.\n"
            "• No decorative randomness is allowed to replace the participant data.\n"
            "• Judge candidates for impact, identity, data necessity, concept, humanity, integrity, composition, colour, materiality, detail, restraint and memorability."
        )
        ttk.Label(principles, text=text, justify="left", wraplength=360).pack(fill="x")

        directions = ttk.LabelFrame(right, text="Art directions", padding=8)
        directions.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for name, spec in lab.ART_DIRECTIONS.items():
            ttk.Label(directions, text=name, font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
            ttk.Label(directions, text=spec["description"], wraplength=360, justify="left").pack(anchor="w", pady=(0, 5))

    def _choose_folder(self) -> None:
        value = filedialog.askdirectory(title="Choose completed study output folder")
        if value:
            self.folder_var.set(value)
            self._load_folder()

    def _load_folder(self) -> None:
        path = Path(self.folder_var.get()).expanduser()
        try:
            self.visual_data = visual.load_visual_data(path)
            self.output_dir = path
        except Exception as exc:
            messagebox.showerror("Cannot load study output", str(exc))
            return
        self.generate_btn.configure(state="normal")
        self.status_var.set(f"Loaded {self.visual_data.participant_id}, trial {self.visual_data.trial_id}")
        self._update_analysis()
        existing = path / "design_lab" / "design_laboratory_contact_sheet.png"
        if existing.exists():
            self.contact_sheet_path = existing
            self._show_sheet()

    def _update_analysis(self) -> None:
        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")
        if self.visual_data is None:
            self.analysis_text.insert("1.0", "No participant loaded.")
        else:
            self.analysis_text.insert("1.0", lab.analysis_text(self.visual_data))
        self.analysis_text.configure(state="disabled")

    def _generate(self) -> None:
        if self.output_dir is None:
            return
        self.generate_btn.configure(state="disabled")
        self.status_var.set("Generating art-directed candidates...")
        count = max(6, min(40, int(self.count_var.get())))

        def worker() -> None:
            try:
                result = lab.generate_laboratory(self.output_dir, count)
                self.after(0, lambda: self._generation_done(result))
            except Exception:
                error = traceback.format_exc()
                self.after(0, lambda: self._generation_failed(error))

        threading.Thread(target=worker, daemon=True).start()

    def _generation_done(self, result: dict) -> None:
        self.generate_btn.configure(state="normal")
        self.contact_sheet_path = Path(result["contact_sheet"])
        self.status_var.set(f"Generated {result['candidate_count']} candidates")
        self._show_sheet()
        messagebox.showinfo("Design laboratory ready", f"Generated laboratory:\n{result['lab_dir']}")

    def _generation_failed(self, error: str) -> None:
        self.generate_btn.configure(state="normal")
        self.status_var.set("Generation failed")
        messagebox.showerror("Design laboratory failed", error)

    def _show_sheet(self) -> None:
        if self.contact_sheet_path is None or not self.contact_sheet_path.exists():
            self.canvas.delete("all")
            self.canvas.create_text(
                max(400, self.canvas.winfo_width() / 2),
                max(300, self.canvas.winfo_height() / 2),
                text="Generate the laboratory to see the art-directed candidate atlas.",
                fill="#555555",
                font=("TkDefaultFont", 13),
            )
            return
        im = Image.open(self.contact_sheet_path).convert("RGB")
        cw = max(200, self.canvas.winfo_width() - 20)
        ch = max(200, self.canvas.winfo_height() - 20)
        ratio = min(cw / im.width, ch / im.height)
        size = (max(1, int(im.width * ratio)), max(1, int(im.height * ratio)))
        im = im.resize(size, Image.Resampling.LANCZOS)
        self._sheet_image = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, image=self._sheet_image, anchor="center")

    def _open_path(self, path: Path) -> None:
        try:
            if platform.system() == "Windows":
                os.startfile(str(path))
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Cannot open", str(exc))

    def _open_lab_folder(self) -> None:
        if self.output_dir is None:
            messagebox.showerror("No study loaded", "Load a completed study output first.")
            return
        path = self.output_dir / "design_lab"
        path.mkdir(parents=True, exist_ok=True)
        self._open_path(path)


if __name__ == "__main__":
    DesignLaboratoryApp().mainloop()
