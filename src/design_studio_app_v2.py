#!/usr/bin/env python3
"""Generative Design Studio UI with stronger curated variation controls."""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import ImageTk

import design_laboratory as lab
import design_randomizer as rnd
import design_studio as studio
import study_visualizer as visual

rnd.install()


class DesignStudioApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector - Generative Design Studio")
        self.geometry("1780x1050")
        self.minsize(1320, 840)

        self.output_dir: Path | None = None
        self.data: studio.StudioData | None = None
        self.params = studio.DesignParams()
        self.history: list[studio.DesignParams] = []
        self._preview_image = None
        self._saved_paths: list[Path] = []

        self.folder_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Choose a completed study output folder.")
        self.family_var = tk.StringVar(value=self.params.family)
        self.palette_var = tk.StringVar(value=self.params.palette)
        self.background_var = tk.StringVar(value=self.params.background)
        self.stroke_var = tk.DoubleVar(value=self.params.stroke_width)
        self.detail_var = tk.IntVar(value=self.params.detail)
        self.jitter_var = tk.DoubleVar(value=self.params.jitter)
        self.symmetry_var = tk.IntVar(value=self.params.symmetry)
        self.scale_var = tk.DoubleVar(value=self.params.scale)
        self.rotation_var = tk.DoubleVar(value=self.params.rotation_deg)
        self.opacity_var = tk.DoubleVar(value=self.params.opacity)
        self.seed_var = tk.IntVar(value=self.params.seed)
        self.view_var = tk.StringVar(value="Artwork")
        self.saved_name_var = tk.StringVar(value="favourite_01")
        self.family_help_var = tk.StringVar(value=studio.FAMILIES[self.params.family])
        self.caption_var = tk.StringVar(value="Load a participant to begin.")
        self._build()

    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=12)
        top.grid(row=0, column=0, columnspan=3, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Generative Design Studio", font=("TkDefaultFont", 19, "bold")).grid(row=0, column=0, columnspan=5, sticky="w")
        ttk.Label(top, text="Artwork uses the locked central 180 outward intervals = five complete 36-spoke revolutions. Randomisation changes presentation only.", wraplength=1600).grid(row=1, column=0, columnspan=5, sticky="w", pady=(3, 9))
        ttk.Label(top, text="Study output").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.folder_var).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Browse...", command=self._choose_folder).grid(row=2, column=2)
        ttk.Button(top, text="Load", command=self._load_folder).grid(row=2, column=3, padx=(6, 0))
        ttk.Label(top, textvariable=self.status_var).grid(row=2, column=4, sticky="w", padx=(12, 0))

        left = ttk.Frame(self, padding=(12, 0, 8, 12), width=330)
        center = ttk.Frame(self, padding=(6, 0, 6, 12))
        right = ttk.Frame(self, padding=(8, 0, 12, 12), width=380)
        left.grid(row=1, column=0, sticky="nsw")
        center.grid(row=1, column=1, sticky="nsew")
        right.grid(row=1, column=2, sticky="nse")
        center.columnconfigure(0, weight=1); center.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1); right.rowconfigure(1, weight=1)

        family = ttk.LabelFrame(left, text="Visual family", padding=9)
        family.pack(fill="x")
        combo = ttk.Combobox(family, textvariable=self.family_var, values=tuple(studio.FAMILIES), state="readonly", width=28)
        combo.pack(fill="x"); combo.bind("<<ComboboxSelected>>", lambda _e: self._family_changed())
        ttk.Label(family, textvariable=self.family_help_var, wraplength=285, justify="left").pack(anchor="w", pady=(7, 0))

        colour = ttk.LabelFrame(left, text="Colour / surface", padding=9)
        colour.pack(fill="x", pady=(9, 0))
        ttk.Label(colour, text=f"Palette ({len(studio.PALETTES)} curated options)").pack(anchor="w")
        pal = ttk.Combobox(colour, textvariable=self.palette_var, values=tuple(studio.PALETTES), state="readonly")
        pal.pack(fill="x", pady=(3, 7)); pal.bind("<<ComboboxSelected>>", lambda _e: self._controls_changed())
        ttk.Label(colour, text="Artwork background").pack(anchor="w")
        bg = ttk.Combobox(colour, textvariable=self.background_var, values=tuple(studio.BACKGROUND_COLORS), state="readonly")
        bg.pack(fill="x", pady=(3, 0)); bg.bind("<<ComboboxSelected>>", lambda _e: self._controls_changed())

        tweak = ttk.LabelFrame(left, text="Minimal tweaks", padding=9)
        tweak.pack(fill="x", pady=(9, 0))
        self._slider(tweak, "Stroke width", self.stroke_var, .25, 5.0, .05)
        self._slider(tweak, "Detail", self.detail_var, 24, 240, 1)
        self._slider(tweak, "Jitter", self.jitter_var, 0.0, 1.0, .01)
        self._slider(tweak, "Symmetry", self.symmetry_var, 1, 16, 1)
        self._slider(tweak, "Scale", self.scale_var, .55, 1.2, .01)
        self._slider(tweak, "Rotation", self.rotation_var, 0, 359, 1)
        self._slider(tweak, "Opacity", self.opacity_var, .15, 1.0, .01)

        variation = ttk.LabelFrame(left, text="Variation", padding=9)
        variation.pack(fill="x", pady=(9, 0))
        ttk.Button(variation, text="Randomise Design", command=self._randomise_design).pack(fill="x")
        ttk.Button(variation, text="Randomise Form", command=self._randomise_form).pack(fill="x", pady=(5, 0))
        ttk.Button(variation, text="Randomise Colour", command=self._randomise_colour).pack(fill="x", pady=(5, 0))
        ttk.Button(variation, text="GO WILD", command=self._go_wild).pack(fill="x", pady=(5, 0))
        ttk.Button(variation, text="Previous Variation", command=self._undo).pack(fill="x", pady=(5, 0))
        ttk.Button(variation, text="Reset Design", command=self._reset).pack(fill="x", pady=(5, 0))
        seedrow = ttk.Frame(variation); seedrow.pack(fill="x", pady=(7, 0))
        ttk.Label(seedrow, text="Seed").pack(side="left")
        ttk.Entry(seedrow, textvariable=self.seed_var, width=12).pack(side="left", padx=(6, 0))
        ttk.Button(seedrow, text="Apply", command=self._controls_changed).pack(side="right")

        stagebar = ttk.Frame(center); stagebar.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(stagebar, text="Live design stage", font=("TkDefaultFont", 14, "bold")).pack(side="left")
        ttk.Label(stagebar, text="View").pack(side="right", padx=(10, 4))
        view = ttk.Combobox(stagebar, textvariable=self.view_var, values=("Artwork", "Black shirt", "White shirt"), state="readonly", width=13)
        view.pack(side="right"); view.bind("<<ComboboxSelected>>", lambda _e: self._refresh_preview())
        self.canvas = tk.Canvas(center, background="#dedbd4", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew"); self.canvas.bind("<Configure>", lambda _e: self._refresh_preview())
        foot = ttk.Frame(center); foot.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(foot, textvariable=self.caption_var).pack(side="left")
        ttk.Button(foot, text="Export PNG...", command=self._export_png).pack(side="right")
        ttk.Button(foot, text="Export SVG...", command=self._export_svg).pack(side="right", padx=(0, 6))

        meaning = ttk.LabelFrame(right, text="What this design means", padding=9)
        meaning.grid(row=0, column=0, sticky="ew")
        self.meaning_text = tk.Text(meaning, width=42, height=11, wrap="word"); self.meaning_text.pack(fill="both"); self.meaning_text.configure(state="disabled")

        saved = ttk.LabelFrame(right, text="Saved designs", padding=9)
        saved.grid(row=1, column=0, sticky="nsew", pady=(9, 0)); saved.rowconfigure(2, weight=1); saved.columnconfigure(0, weight=1)
        ttk.Entry(saved, textvariable=self.saved_name_var).grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.saved_list = tk.Listbox(saved, height=11); self.saved_list.grid(row=2, column=0, sticky="nsew")
        row = ttk.Frame(saved); row.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(row, text="Save Current", command=self._save_current).pack(side="left")
        ttk.Button(row, text="Load Selected", command=self._load_saved).pack(side="left", padx=(5, 0))
        ttk.Button(row, text="Open Folder", command=self._open_saved_folder).pack(side="left", padx=(5, 0))

        analysis = ttk.LabelFrame(right, text="Matched return-slice analysis", padding=9)
        analysis.grid(row=2, column=0, sticky="ew", pady=(9, 0))
        self.analysis_text = tk.Text(analysis, width=42, height=12, wrap="word"); self.analysis_text.pack(fill="both"); self.analysis_text.configure(state="disabled")
        self._update_text_panels()

    def _slider(self, parent, label, var, lo, hi, resolution) -> None:
        row = ttk.Frame(parent); row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=17).pack(side="left")
        tk.Scale(row, variable=var, from_=lo, to=hi, resolution=resolution, orient="horizontal", showvalue=True, length=155, command=lambda _v: self._controls_changed()).pack(side="right", fill="x", expand=True)

    def _choose_folder(self) -> None:
        value = filedialog.askdirectory(title="Choose completed study output folder")
        if value:
            self.folder_var.set(value); self._load_folder()

    def _load_folder(self) -> None:
        path = Path(self.folder_var.get()).expanduser()
        try:
            self.data = studio.load_studio_data(path)
            self.output_dir = path
        except Exception as exc:
            messagebox.showerror("Cannot load study output", str(exc)); return
        self.status_var.set(f"Loaded {self.data.participant_id}, trial {self.data.trial_id}")
        self._refresh_saved(); self._update_text_panels(); self._refresh_preview()

    def _read_params(self) -> studio.DesignParams:
        return studio.DesignParams(
            family=self.family_var.get(), palette=self.palette_var.get(), background=self.background_var.get(),
            stroke_width=float(self.stroke_var.get()), detail=int(self.detail_var.get()), jitter=float(self.jitter_var.get()),
            symmetry=int(self.symmetry_var.get()), scale=float(self.scale_var.get()), rotation_deg=float(self.rotation_var.get()),
            opacity=float(self.opacity_var.get()), seed=int(self.seed_var.get()),
        ).validated()

    def _apply_params(self, p: studio.DesignParams) -> None:
        p = p.validated(); self.params = p
        self.family_var.set(p.family); self.palette_var.set(p.palette); self.background_var.set(p.background)
        self.stroke_var.set(p.stroke_width); self.detail_var.set(p.detail); self.jitter_var.set(p.jitter); self.symmetry_var.set(p.symmetry)
        self.scale_var.set(p.scale); self.rotation_var.set(p.rotation_deg); self.opacity_var.set(p.opacity); self.seed_var.set(p.seed)
        self.family_help_var.set(studio.FAMILIES[p.family]); self._update_text_panels(); self._refresh_preview()

    def _push_history(self) -> None:
        try: self.history.append(studio.DesignParams(**asdict(self._read_params())))
        except Exception: return
        if len(self.history) > 40: self.history = self.history[-40:]

    def _family_changed(self) -> None:
        self.family_help_var.set(studio.FAMILIES[self.family_var.get()]); self._controls_changed()

    def _controls_changed(self) -> None:
        try: self.params = self._read_params()
        except Exception: return
        self._update_text_panels(); self._refresh_preview()

    def _randomise_design(self) -> None:
        self._push_history(); self._apply_params(rnd.randomise_design(self._read_params(), keep_family=True))

    def _randomise_form(self) -> None:
        self._push_history(); self._apply_params(rnd.randomise_form(self._read_params()))

    def _randomise_colour(self) -> None:
        self._push_history(); self._apply_params(rnd.randomise_colour(self._read_params()))

    def _go_wild(self) -> None:
        self._push_history(); self._apply_params(rnd.go_wild(self._read_params(), keep_family=False))

    def _undo(self) -> None:
        if self.history: self._apply_params(self.history.pop())

    def _reset(self) -> None:
        self._push_history(); self._apply_params(studio.DesignParams(family=self.family_var.get(), background=self.background_var.get()))

    def _preview_pil(self, size: int):
        if self.data is None: return None
        p = self._read_params()
        if self.view_var.get() == "Black shirt": return studio.render_shirt_mockup(self.data, p, size=size, shirt="Black")
        if self.view_var.get() == "White shirt": return studio.render_shirt_mockup(self.data, p, size=size, shirt="White")
        return studio.render_image(self.data, p, size=size, transparent=False)

    def _refresh_preview(self) -> None:
        if not hasattr(self, "canvas"): return
        self.canvas.delete("all")
        if self.data is None:
            self.canvas.create_text(max(350, self.canvas.winfo_width()/2), max(300, self.canvas.winfo_height()/2), text="Load a completed study output to start designing.", fill="#555", font=("TkDefaultFont", 13)); return
        side = max(420, min(self.canvas.winfo_width()-18, self.canvas.winfo_height()-18, 950))
        im = self._preview_pil(int(side))
        if im is None: return
        self._preview_image = ImageTk.PhotoImage(im.convert("RGBA"))
        self.canvas.create_image(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, image=self._preview_image, anchor="center")
        p = self._read_params()
        self.caption_var.set(f"{self.data.participant_id} · {p.family} · {p.palette} · stroke {p.stroke_width:.2f} · seed {p.seed}")

    def _update_text_panels(self) -> None:
        if hasattr(self, "meaning_text"):
            self.meaning_text.configure(state="normal"); self.meaning_text.delete("1.0", "end")
            if self.data is None: text = "Load a participant to see the design meaning."
            else: text = f"{self.family_var.get()}\n\n{studio.FAMILIES[self.family_var.get()]}\n\nDATA FOUNDATION\nCentral {len(self.data.intervals)} intervals from the first/outward leg = {len(self.data.intervals)/self.data.spoke_count:.1f} complete wheel revolutions.\n\nThe participant timing changes geometry. Stroke, palette, jitter, scale and symmetry change presentation only."
            self.meaning_text.insert("1.0", text); self.meaning_text.configure(state="disabled")
        if hasattr(self, "analysis_text"):
            self.analysis_text.configure(state="normal"); self.analysis_text.delete("1.0", "end")
            if self.output_dir is None: self.analysis_text.insert("1.0", "The paired analysis compares the same central fixed-count slice from outward and return legs.")
            else:
                try: self.analysis_text.insert("1.0", lab.analysis_text(visual.load_visual_data(self.output_dir)))
                except Exception as exc: self.analysis_text.insert("1.0", str(exc))
            self.analysis_text.configure(state="disabled")

    def _refresh_saved(self) -> None:
        self.saved_list.delete(0, "end"); self._saved_paths = []
        if self.output_dir is None: return
        self._saved_paths = studio.list_saved(self.output_dir)
        for path in self._saved_paths: self.saved_list.insert("end", path.stem)

    def _save_current(self) -> None:
        if self.output_dir is None or self.data is None: messagebox.showerror("Nothing to save", "Load a participant first."); return
        name = self.saved_name_var.get().strip() or "favourite"
        paths = studio.save_design(self.output_dir, self.data, self._read_params(), name)
        self.status_var.set(f"Saved {Path(paths['json']).stem}"); self._refresh_saved()

    def _load_saved(self) -> None:
        sel = self.saved_list.curselection()
        if not sel or not self._saved_paths: return
        try:
            self._push_history(); self._apply_params(studio.load_saved_params(self._saved_paths[sel[0]]))
        except Exception as exc: messagebox.showerror("Cannot load saved design", str(exc))

    def _export_png(self) -> None:
        if self.data is None: messagebox.showerror("Nothing to export", "Load a participant first."); return
        path = filedialog.asksaveasfilename(title="Export print PNG", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path: studio.export_png(Path(path), self.data, self._read_params(), 2400, transparent=(self.background_var.get()=="Transparent"))

    def _export_svg(self) -> None:
        if self.data is None: messagebox.showerror("Nothing to export", "Load a participant first."); return
        path = filedialog.asksaveasfilename(title="Export vector SVG", defaultextension=".svg", filetypes=[("SVG", "*.svg")])
        if path: studio.export_svg(Path(path), self.data, self._read_params(), 2400, transparent=(self.background_var.get()=="Transparent"))

    def _open_saved_folder(self) -> None:
        if self.output_dir is None: return
        path = self.output_dir / "design_studio" / "saved"; path.mkdir(parents=True, exist_ok=True)
        try:
            if platform.system() == "Windows": os.startfile(str(path))
            elif platform.system() == "Darwin": subprocess.Popen(["open", str(path)])
            else: subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc: messagebox.showerror("Cannot open", str(exc))


def main() -> None:
    DesignStudioApp().mainloop()


if __name__ == "__main__":
    main()
