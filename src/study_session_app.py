#!/usr/bin/env python3
"""Desktop app for locked study analysis and curated human-data artwork."""
from __future__ import annotations

import csv
import os
import platform
import subprocess
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
        self.title("Spoke Detector - Study Analysis + Movement Portraits")
        self.geometry("1640x1000")
        self.minsize(1240, 800)

        self.protocol = study.load_protocol(DEFAULT_PROTOCOL)
        rules = self.protocol["analysis"]
        self.current_output: Path | None = None
        self.visual_data: visual.VisualData | None = None
        self.zoom = 1.0
        self.fullscreen_window: tk.Toplevel | None = None

        self.wav_var = tk.StringVar()
        self.profile_var = tk.StringVar(value=str(DEFAULT_PROFILE))
        self.output_var = tk.StringVar()
        self.participant_var = tk.StringVar()
        self.trial_var = tk.StringVar(value="1")
        self.outward_position_var = tk.StringVar(value="A")
        self.return_position_var = tk.StringVar(value="B")
        self.turnaround_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

        self.source_var = tk.StringVar(value=visual.SOURCE_OPTIONS[0])
        self.group_var = tk.StringVar(value="Signature")
        self.style_var = tk.StringVar(value=visual.STYLE_GROUPS["Signature"][0])
        self.palette_family_var = tk.StringVar(value="Luxury")
        self.palette_var = tk.StringVar(value=visual.PALETTE_FAMILIES["Luxury"][0])
        self.theme_var = tk.StringVar(value="Dark garment")
        self.density_var = tk.StringVar(value="Medium")
        self.smoothing_var = tk.StringVar(value="Medium")
        self.annotated_var = tk.BooleanVar(value=False)
        self.visual_status_var = tk.StringVar(value="Run an analysis or load an existing study output.")
        self.style_help_var = tk.StringVar(value=visual.STYLE_HELP[self.style_var.get()])

        self.protocol_text = (
            f"{self.protocol['protocol_id']}  |  {rules['spoke_count']} spokes  |  "
            f"central {rules['primary_comparison_intervals_per_leg']} intervals per leg  |  "
            f"{rules['equivalent_wheel_revolutions_per_leg']:.1f} wheel revolutions per leg"
        )
        self._build()

    def _build(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.analysis_tab = ttk.Frame(self.notebook, padding=14)
        self.visual_tab = ttk.Frame(self.notebook, padding=10)
        self.outputs_tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(self.analysis_tab, text="1. Session Setup & Analysis")
        self.notebook.add(self.visual_tab, text="2. Data Visualiser")
        self.notebook.add(self.outputs_tab, text="3. Outputs / Subject Record")
        self._build_analysis_tab(self.analysis_tab)
        self._build_visual_tab(self.visual_tab)
        self._build_outputs_tab(self.outputs_tab)

    def _build_analysis_tab(self, root: ttk.Frame) -> None:
        root.columnconfigure(1, weight=1); root.rowconfigure(10, weight=1)
        ttk.Label(root, text="Standardised Study Session", font=("TkDefaultFont", 17, "bold")).grid(row=0,column=0,columnspan=3,sticky="w",pady=(0,8))
        ttk.Label(root, text="Capture and analysis rules are locked so every participant is processed with the same physical-equivalent comparison window. Artwork choices never alter those rules.", wraplength=1420).grid(row=1,column=0,columnspan=3,sticky="w",pady=(0,12))

        protocol_box = ttk.LabelFrame(root, text="Locked study protocol", padding=10); protocol_box.grid(row=2,column=0,columnspan=3,sticky="ew",pady=(0,12))
        ttk.Label(protocol_box,text=self.protocol_text,font=("TkDefaultFont",10,"bold")).pack(anchor="w")
        ttk.Label(protocol_box,text="The visualiser defaults to the first/outward leg as a baseline portrait. The return leg remains available as a separate visual source.",wraplength=1360).pack(anchor="w",pady=(5,0))

        meta = ttk.LabelFrame(root,text="Required participant metadata",padding=10); meta.grid(row=3,column=0,columnspan=3,sticky="ew",pady=(0,12)); meta.columnconfigure(1,weight=1); meta.columnconfigure(3,weight=1)
        ttk.Label(meta,text="Participant ID").grid(row=0,column=0,sticky="w"); ttk.Entry(meta,textvariable=self.participant_var,width=20).grid(row=0,column=1,sticky="ew",padx=(6,18))
        ttk.Label(meta,text="Trial ID").grid(row=0,column=2,sticky="w"); ttk.Entry(meta,textvariable=self.trial_var,width=12).grid(row=0,column=3,sticky="w",padx=(6,0))
        ttk.Label(meta,text="Outward handlebar position").grid(row=1,column=0,sticky="w",pady=(8,0)); ttk.Combobox(meta,textvariable=self.outward_position_var,values=("A","B"),state="readonly",width=8).grid(row=1,column=1,sticky="w",padx=(6,18),pady=(8,0))
        ttk.Label(meta,text="Return handlebar position").grid(row=1,column=2,sticky="w",pady=(8,0)); ttk.Combobox(meta,textvariable=self.return_position_var,values=("A","B"),state="readonly",width=8).grid(row=1,column=3,sticky="w",padx=(6,0),pady=(8,0)); ttk.Button(meta,text="Swap A/B",command=self._swap_positions).grid(row=1,column=4,padx=(12,0),pady=(8,0))

        self._path_row(root,4,"Audio WAV",self.wav_var,self._choose_wav)
        self._path_row(root,5,"Calibration profile",self.profile_var,self._choose_profile)
        self._path_row(root,6,"Output folder",self.output_var,self._choose_output)

        opts=ttk.LabelFrame(root,text="Optional protocol note",padding=10); opts.grid(row=7,column=0,columnspan=3,sticky="ew",pady=(12,10)); ttk.Label(opts,text="Manual turnaround time, seconds (leave blank for automatic split)").grid(row=0,column=0,sticky="w"); ttk.Entry(opts,textvariable=self.turnaround_var,width=12).grid(row=0,column=1,sticky="w",padx=(8,0))

        buttons=ttk.Frame(root); buttons.grid(row=8,column=0,columnspan=3,sticky="ew",pady=(8,10)); self.run_btn=ttk.Button(buttons,text="Analyse Standardised Session",command=self._start_analysis); self.run_btn.pack(side="left"); ttk.Button(buttons,text="Load Existing Analysis",command=self._load_existing_output).pack(side="left",padx=(8,0)); ttk.Button(buttons,text="Clear Session",command=self._clear_session).pack(side="left",padx=(8,0)); ttk.Label(buttons,textvariable=self.status_var).pack(side="left",padx=(14,0))
        ttk.Label(root,text="Results").grid(row=9,column=0,columnspan=3,sticky="w"); self.results=tk.Text(root,height=15,wrap="word"); self.results.grid(row=10,column=0,columnspan=3,sticky="nsew",pady=(5,0))

    def _build_visual_tab(self, root: ttk.Frame) -> None:
        root.columnconfigure(0,weight=1); root.rowconfigure(0,weight=1)
        paned=ttk.Panedwindow(root,orient="horizontal"); paned.grid(row=0,column=0,sticky="nsew")
        controls=ttk.Frame(paned,padding=8,width=285); preview=ttk.Frame(paned,padding=6); side=ttk.Frame(paned,padding=8,width=340)
        paned.add(controls,weight=1); paned.add(preview,weight=7); paned.add(side,weight=2)

        ttk.Label(controls,text="Movement Portrait",font=("TkDefaultFont",16,"bold")).pack(anchor="w",pady=(0,10))
        ttk.Label(controls,text="Data source").pack(anchor="w")
        source_box=ttk.Combobox(controls,textvariable=self.source_var,values=visual.SOURCE_OPTIONS,state="readonly",width=26); source_box.pack(fill="x",pady=(3,9)); source_box.bind("<<ComboboxSelected>>",self._source_changed)
        ttk.Label(controls,text="Baseline / first leg is the default personal portrait before the turn and handlebar swap.",wraplength=255,justify="left").pack(anchor="w",pady=(0,10))

        ttk.Label(controls,text="Style group").pack(anchor="w"); group_box=ttk.Combobox(controls,textvariable=self.group_var,values=tuple(visual.STYLE_GROUPS)+("All",),state="readonly",width=23); group_box.pack(fill="x",pady=(3,9)); group_box.bind("<<ComboboxSelected>>",self._group_changed)
        ttk.Label(controls,text="Style").pack(anchor="w"); self.style_box=ttk.Combobox(controls,textvariable=self.style_var,values=visual.STYLE_GROUPS["Signature"],state="readonly",width=26); self.style_box.pack(fill="x",pady=(3,8)); self.style_box.bind("<<ComboboxSelected>>",self._style_changed)
        ttk.Label(controls,textvariable=self.style_help_var,wraplength=255,justify="left").pack(anchor="w",pady=(0,10))

        ttk.Separator(controls).pack(fill="x",pady=4)
        ttk.Label(controls,text="Palette family").pack(anchor="w",pady=(6,0)); pf=ttk.Combobox(controls,textvariable=self.palette_family_var,values=tuple(visual.PALETTE_FAMILIES),state="readonly",width=22); pf.pack(fill="x",pady=(3,3)); pf.bind("<<ComboboxSelected>>",self._palette_family_changed)
        ttk.Label(controls,text="Palette").pack(anchor="w",pady=(6,0)); self.palette_box=ttk.Combobox(controls,textvariable=self.palette_var,values=visual.PALETTE_FAMILIES["Luxury"],state="readonly",width=22); self.palette_box.pack(fill="x",pady=(3,3)); self.palette_box.bind("<<ComboboxSelected>>",lambda _e:self._refresh_preview())

        for label,var,values in [("Garment preview",self.theme_var,visual.THEMES),("Line / cell density",self.density_var,visual.DENSITIES),("Smoothing",self.smoothing_var,visual.SMOOTHING)]:
            ttk.Label(controls,text=label).pack(anchor="w",pady=(6,0)); box=ttk.Combobox(controls,textvariable=var,values=values,state="readonly",width=22); box.pack(fill="x",pady=(3,3)); box.bind("<<ComboboxSelected>>",lambda _e:self._refresh_preview())
        ttk.Checkbutton(controls,text="Show participant + metric overlay",variable=self.annotated_var,command=self._refresh_preview).pack(anchor="w",pady=(8,6))

        zooms=ttk.LabelFrame(controls,text="Preview",padding=6); zooms.pack(fill="x",pady=(10,0)); ttk.Button(zooms,text="Zoom +",command=lambda:self._change_zoom(1.15)).pack(side="left",expand=True,fill="x"); ttk.Button(zooms,text="Zoom -",command=lambda:self._change_zoom(1/1.15)).pack(side="left",expand=True,fill="x",padx=4); ttk.Button(zooms,text="Fit",command=self._fit_preview).pack(side="left",expand=True,fill="x"); ttk.Button(controls,text="Full Screen Preview",command=self._open_fullscreen_preview).pack(fill="x",pady=(8,0))

        preview.rowconfigure(1,weight=1); preview.columnconfigure(0,weight=1); ttk.Label(preview,text="Large live artwork preview",font=("TkDefaultFont",13,"bold")).grid(row=0,column=0,sticky="w",pady=(0,4)); self.preview_canvas=tk.Canvas(preview,width=1000,height=780,highlightthickness=0,background="#111111"); self.preview_canvas.grid(row=1,column=0,sticky="nsew"); self.preview_canvas.bind("<Configure>",lambda _e:self._refresh_preview()); self.preview_canvas.bind("<Double-Button-1>",lambda _e:self._open_fullscreen_preview())

        side.columnconfigure(0,weight=1); side.rowconfigure(0,weight=1)
        meaning=ttk.LabelFrame(side,text="What this visual means",padding=8); meaning.grid(row=0,column=0,sticky="nsew"); self.brief_text=tk.Text(meaning,width=38,height=23,wrap="word"); self.brief_text.pack(fill="both",expand=True); self.brief_text.insert("1.0","No analysed session loaded."); self.brief_text.configure(state="disabled")
        exports=ttk.LabelFrame(side,text="Artwork export",padding=8); exports.grid(row=1,column=0,sticky="ew",pady=(8,0)); ttk.Button(exports,text="Export Current SVG...",command=self._export_current_svg).pack(fill="x"); ttk.Button(exports,text="Export Current PNG...",command=self._export_current_png).pack(fill="x",pady=(6,0)); ttk.Button(exports,text=f"Export All {len(visual.STYLES)} Baseline Styles",command=self._export_all_styles).pack(fill="x",pady=(6,0)); ttk.Label(exports,text="Palette, style and smoothing change presentation only. The underlying locked study window remains unchanged.",wraplength=300).pack(fill="x",pady=(8,0)); ttk.Label(side,textvariable=self.visual_status_var,wraplength=300).grid(row=2,column=0,sticky="ew",pady=(8,0))

    def _build_outputs_tab(self, root: ttk.Frame) -> None:
        root.columnconfigure(0,weight=2); root.columnconfigure(1,weight=3); root.rowconfigure(1,weight=1)
        ttk.Label(root,text="Outputs / Subject Record",font=("TkDefaultFont",17,"bold")).grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,10))
        files=ttk.LabelFrame(root,text="Generated files",padding=8); files.grid(row=1,column=0,sticky="nsew",padx=(0,8)); files.rowconfigure(0,weight=1); files.columnconfigure(0,weight=1); self.output_list=tk.Listbox(files); self.output_list.grid(row=0,column=0,sticky="nsew"); fb=ttk.Frame(files); fb.grid(row=1,column=0,sticky="ew",pady=(8,0)); ttk.Button(fb,text="Open Selected",command=self._open_selected_output).pack(side="left"); ttk.Button(fb,text="Open Output Folder",command=self._open_output_folder).pack(side="left",padx=(6,0))
        record=ttk.LabelFrame(root,text="Standard subject record",padding=8); record.grid(row=1,column=1,sticky="nsew"); record.rowconfigure(0,weight=1); record.columnconfigure(0,weight=1); self.record_text=tk.Text(record,wrap="word"); self.record_text.grid(row=0,column=0,sticky="nsew"); self.record_text.insert("1.0","No analysed session loaded."); self.record_text.configure(state="disabled")

    def _path_row(self,parent,row,label,var,command):
        ttk.Label(parent,text=label).grid(row=row,column=0,sticky="w",pady=4); ttk.Entry(parent,textvariable=var).grid(row=row,column=1,sticky="ew",padx=8,pady=4); ttk.Button(parent,text="Browse...",command=command).grid(row=row,column=2,pady=4)

    def _swap_positions(self):
        a,b=self.outward_position_var.get(),self.return_position_var.get(); self.outward_position_var.set(b); self.return_position_var.set(a)

    def _choose_wav(self):
        value=filedialog.askopenfilename(title="Choose participant WAV",filetypes=[("WAV audio","*.wav"),("All files","*.*")],initialdir=str(REPO_ROOT/"audio"))
        if value: self.wav_var.set(value); self._suggest_output()

    def _choose_profile(self):
        value=filedialog.askopenfilename(title="Choose calibration profile",filetypes=[("JSON","*.json"),("All files","*.*")],initialdir=str(REPO_ROOT/"calibration"))
        if value: self.profile_var.set(value)

    def _choose_output(self):
        value=filedialog.askdirectory(title="Choose output folder",initialdir=str(REPO_ROOT))
        if value: self.output_var.set(value)

    def _suggest_output(self):
        participant,trial=self.participant_var.get().strip(),self.trial_var.get().strip()
        if participant and trial: self.output_var.set(str(REPO_ROOT/"study_output"/participant/f"trial_{trial}"))
        elif self.wav_var.get() and not self.output_var.get(): self.output_var.set(str(REPO_ROOT/"study_output"/Path(self.wav_var.get()).stem))

    def _clear_session(self):
        self.wav_var.set(""); self.output_var.set(""); self.participant_var.set(""); self.trial_var.set("1"); self.current_output=None; self.visual_data=None; self.status_var.set("Ready"); self.visual_status_var.set("Run an analysis or load an existing study output."); self.results.delete("1.0","end"); self._update_brief_text(); self._update_outputs_tab(); self._refresh_preview()

    def _start_analysis(self):
        try:
            wav=Path(self.wav_var.get()).expanduser(); profile=Path(self.profile_var.get()).expanduser(); participant=self.participant_var.get().strip(); trial=self.trial_var.get().strip(); outward=self.outward_position_var.get().strip(); ret=self.return_position_var.get().strip(); turn_text=self.turnaround_var.get().strip(); turn=None if not turn_text else float(turn_text)
            if not wav.exists(): raise ValueError("Choose an existing WAV file.")
            if not profile.exists(): raise ValueError("Choose an existing calibration profile.")
            if not participant: raise ValueError("Enter a participant ID.")
            if not trial: raise ValueError("Enter a trial ID.")
            if outward==ret: raise ValueError("Outward and return handlebar positions must differ.")
            if not self.output_var.get().strip(): self._suggest_output()
            if not self.output_var.get().strip(): raise ValueError("Choose an output folder.")
            out=Path(self.output_var.get()).expanduser()
        except Exception as exc:
            messagebox.showerror("Cannot start",str(exc)); return
        self.run_btn.configure(state="disabled"); self.status_var.set("Analysing with locked protocol..."); self.results.delete("1.0","end")
        def worker():
            try:
                summary=study.analyse_study_session(wav,out,profile,participant_id=participant,trial_id=trial,outward_handlebar_position=outward,return_handlebar_position=ret,protocol_path=DEFAULT_PROTOCOL,manual_turnaround_s=turn)
                visual.export_all_default_styles(out,source="Baseline / first leg",palette_name="Gold on Black")
                self.after(0,lambda:self._show_success(summary,out))
            except Exception:
                err=traceback.format_exc(); self.after(0,lambda:self._show_error(err))
        threading.Thread(target=worker,daemon=True).start()

    def _show_success(self,summary,out):
        self.run_btn.configure(state="normal"); self.status_var.set("Complete - comparable analysis + baseline artwork written"); comp=summary.get("standardised_primary_comparison") or {}; det=summary.get("detector") or {}; subject=summary.get("subject") or {}
        lines=[f"Participant: {subject.get('participant_id')}    Trial: {subject.get('trial_id')}",f"Handlebar order: {subject.get('outward_handlebar_position')} -> {subject.get('return_handlebar_position')}",f"Detected clicks: {det.get('detected_hits')}",f"Detector quality: {det.get('quality_status')}"]
        if comp.get("available"):
            om,rm=comp.get("outward") or {},comp.get("return") or {}; lines += [f"Locked comparison window: {comp.get('requested_intervals_per_leg')} intervals per leg",f"Outward median click rate: {om.get('median_click_rate_hz'):.3f} Hz",f"Return median click rate: {rm.get('median_click_rate_hz'):.3f} Hz",f"Return vs outward change: {comp.get('return_vs_outward_click_rate_change_percent'):.2f}%"]
        lines += ["",f"Output folder: {out}","Artwork default source: Baseline / first leg",f"Artwork folder: {out/'artwork'}"]
        self.results.insert("1.0","\n".join(lines)); self._activate_visualiser(out); self.notebook.select(self.visual_tab); messagebox.showinfo("Analysis complete","Study analysis completed. The baseline movement portrait visualiser is ready.")

    def _show_error(self,error_text):
        self.run_btn.configure(state="normal"); self.status_var.set("Failed"); self.results.insert("1.0",error_text); messagebox.showerror("Analysis failed","The analysis failed. Details are shown in Results.")

    def _load_existing_output(self):
        value=filedialog.askdirectory(title="Choose an existing study output folder",initialdir=str(REPO_ROOT/"study_output"))
        if not value: return
        try: self._activate_visualiser(Path(value)); self.status_var.set("Existing analysis loaded"); self.notebook.select(self.visual_tab)
        except Exception as exc: messagebox.showerror("Cannot load analysis",str(exc))

    def _activate_visualiser(self,output_dir):
        self.current_output=output_dir; self.visual_data=visual.load_visual_data(output_dir); self.visual_status_var.set(f"Loaded {self.visual_data.participant_id}, trial {self.visual_data.trial_id}"); self._update_brief_text(); self._update_outputs_tab(); self._fit_preview()

    def _source_changed(self,_event=None): self._update_brief_text(); self._fit_preview()

    def _group_changed(self,_event=None):
        group=self.group_var.get(); values=visual.STYLES if group=="All" else visual.STYLE_GROUPS[group]; self.style_box.configure(values=values)
        if self.style_var.get() not in values: self.style_var.set(values[0])
        self._style_changed()

    def _style_changed(self,_event=None):
        self.style_help_var.set(visual.STYLE_HELP.get(self.style_var.get(),"")); self._update_brief_text(); self._fit_preview()

    def _palette_family_changed(self,_event=None):
        values=visual.palettes_for_family(self.palette_family_var.get()); self.palette_box.configure(values=values); self.palette_var.set(values[0]); self._refresh_preview()

    def _change_zoom(self,factor): self.zoom=max(.55,min(2.5,self.zoom*factor)); self._refresh_preview()
    def _fit_preview(self): self.zoom=1.0; self._refresh_preview()

    def _refresh_preview(self):
        if not hasattr(self,"preview_canvas"): return
        if self.visual_data is None:
            self.preview_canvas.delete("all"); self.preview_canvas.create_text(max(380,self.preview_canvas.winfo_width()/2),max(300,self.preview_canvas.winfo_height()/2),text="Analyse a session or load an existing output to preview artwork.",fill="#aaaaaa",font=("TkDefaultFont",13)); return
        visual.draw_on_canvas(self.preview_canvas,self.visual_data,self.style_var.get(),self.theme_var.get(),self.density_var.get(),self.smoothing_var.get(),self.annotated_var.get(),zoom=self.zoom,source=self.source_var.get(),palette_name=self.palette_var.get())
        if self.fullscreen_window and self.fullscreen_window.winfo_exists(): visual.draw_on_canvas(self.fullscreen_window._preview_canvas,self.visual_data,self.style_var.get(),self.theme_var.get(),self.density_var.get(),self.smoothing_var.get(),self.annotated_var.get(),zoom=1.0,source=self.source_var.get(),palette_name=self.palette_var.get())

    def _open_fullscreen_preview(self):
        if self.visual_data is None: messagebox.showerror("No preview","Load or analyse a session first."); return
        if self.fullscreen_window and self.fullscreen_window.winfo_exists(): self.fullscreen_window.lift(); return
        win=tk.Toplevel(self); self.fullscreen_window=win; win.title("Full Screen Movement Portrait"); win.attributes("-fullscreen",True); canvas=tk.Canvas(win,highlightthickness=0,background="#111111"); canvas.pack(fill="both",expand=True); win._preview_canvas=canvas; ttk.Button(win,text="Close Full Screen",command=win.destroy).place(relx=.98,rely=.02,anchor="ne"); win.bind("<Escape>",lambda _e:win.destroy()); canvas.bind("<Configure>",lambda _e:self._refresh_preview()); self.after(80,self._refresh_preview)

    def _update_brief_text(self):
        if not hasattr(self,"brief_text"): return
        self.brief_text.configure(state="normal"); self.brief_text.delete("1.0","end")
        if self.visual_data is None: self.brief_text.insert("1.0","No analysed session loaded.")
        else:
            lines=visual.meaning_lines(self.visual_data,self.style_var.get(),self.source_var.get())+["","MOVEMENT SUMMARY"]+visual.brief_analysis_lines(self.visual_data,self.source_var.get())
            self.brief_text.insert("1.0","\n".join(lines))
        self.brief_text.configure(state="disabled")

    def _default_art_name(self,ext):
        stem=self.style_var.get().lower().replace(" ","_").replace("-","_"); src="baseline" if self.source_var.get()=="Baseline / first leg" else "return"; return f"{src}_{stem}"+("_annotated" if self.annotated_var.get() else "")+ext

    def _export_current_svg(self):
        if self.visual_data is None: messagebox.showerror("Nothing to export","Load or analyse a study session first."); return
        path=filedialog.asksaveasfilename(title="Export print-ready SVG",defaultextension=".svg",initialfile=self._default_art_name(".svg"),filetypes=[("SVG vector artwork","*.svg")])
        if not path: return
        visual.export_svg(Path(path),self.visual_data,self.style_var.get(),theme=self.theme_var.get(),density=self.density_var.get(),smoothing=self.smoothing_var.get(),annotated=self.annotated_var.get(),transparent=True,source=self.source_var.get(),palette_name=self.palette_var.get()); self.visual_status_var.set(f"Exported {Path(path).name}")

    def _export_current_png(self):
        if self.visual_data is None: messagebox.showerror("Nothing to export","Load or analyse a study session first."); return
        path=filedialog.asksaveasfilename(title="Export transparent PNG",defaultextension=".png",initialfile=self._default_art_name(".png"),filetypes=[("PNG image","*.png")])
        if not path: return
        visual.export_png(Path(path),self.visual_data,self.style_var.get(),theme=self.theme_var.get(),density=self.density_var.get(),smoothing=self.smoothing_var.get(),annotated=self.annotated_var.get(),transparent=True,source=self.source_var.get(),palette_name=self.palette_var.get()); self.visual_status_var.set(f"Exported {Path(path).name}")

    def _export_all_styles(self):
        if self.visual_data is None or self.current_output is None: messagebox.showerror("Nothing to export","Load or analyse a study session first."); return
        paths=visual.export_all_default_styles(self.current_output,source="Baseline / first leg",palette_name=self.palette_var.get()); self.visual_status_var.set(f"Exported {len(paths)} baseline SVG files to {self.current_output/'artwork'}"); self._update_outputs_tab(); messagebox.showinfo("Artwork exported",f"Baseline artwork written to:\n{self.current_output/'artwork'}")

    def _update_outputs_tab(self):
        if not hasattr(self,"output_list"): return
        self.output_list.delete(0,"end"); self.record_text.configure(state="normal"); self.record_text.delete("1.0","end")
        if self.current_output is None: self.record_text.insert("1.0","No analysed session loaded."); self.record_text.configure(state="disabled"); return
        candidates=[self.current_output/"study_summary.json",self.current_output/"subject_record.csv",self.current_output/"detection"/"session_detections.csv",self.current_output/"detection"/"session_isolated.wav",self.current_output/"movement"/"movement_timeline.csv",self.current_output/"movement"/"movement_summary.json"]
        art=self.current_output/"artwork"
        if art.exists(): candidates += sorted(art.glob("*"))
        self._output_paths=[p for p in candidates if p.exists()]
        for p in self._output_paths: self.output_list.insert("end",str(p.relative_to(self.current_output)))
        record_path=self.current_output/"subject_record.csv"
        if record_path.exists():
            rows=list(csv.DictReader(record_path.open(newline="",encoding="utf-8"))); row=rows[0] if rows else {}; ordered=["participant_id","trial_id","outward_handlebar_position","return_handlebar_position","detected_hits","detector_quality","comparison_intervals_per_leg","equivalent_wheel_revolutions_per_leg","outward_median_click_rate_hz","return_median_click_rate_hz","outward_robust_interval_cv","return_robust_interval_cv","return_vs_outward_click_rate_change_percent","return_adaptation_change_percent","protocol_id"]; self.record_text.insert("1.0","\n".join(f"{key}: {row.get(key,'')}" for key in ordered))
        else: self.record_text.insert("1.0","Subject record not found.")
        self.record_text.configure(state="disabled")

    def _open_path(self,path):
        try:
            if platform.system()=="Windows": os.startfile(str(path))
            elif platform.system()=="Darwin": subprocess.Popen(["open",str(path)])
            else: subprocess.Popen(["xdg-open",str(path)])
        except Exception as exc: messagebox.showerror("Cannot open",str(exc))

    def _open_selected_output(self):
        sel=self.output_list.curselection()
        if sel and hasattr(self,"_output_paths"): self._open_path(self._output_paths[sel[0]])

    def _open_output_folder(self):
        if self.current_output is not None: self._open_path(self.current_output)


if __name__ == "__main__":
    StudySessionApp().mainloop()
