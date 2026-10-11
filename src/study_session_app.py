#!/usr/bin/env python3
"""Desktop app for locked-protocol study analysis, visualisation and artwork export."""
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
        self.title("Spoke Detector - Study Analysis + Visualiser")
        self.geometry("1600x980")
        self.minsize(1220, 780)

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

        self.group_var = tk.StringVar(value="Art")
        self.style_var = tk.StringVar(value=visual.STYLE_GROUPS["Art"][0])
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

    # ---------- Tab 1: analysis ----------
    def _build_analysis_tab(self, root: ttk.Frame) -> None:
        root.columnconfigure(1, weight=1)
        root.rowconfigure(10, weight=1)
        ttk.Label(root, text="Standardised Study Session", font=("TkDefaultFont", 17, "bold")).grid(row=0,column=0,columnspan=3,sticky="w",pady=(0,8))
        ttk.Label(root,text="Capture and analysis rules are locked so every subject is processed with the same physical-equivalent comparison window. Visualisation controls never alter those rules.",wraplength=1420).grid(row=1,column=0,columnspan=3,sticky="w",pady=(0,12))

        protocol_box=ttk.LabelFrame(root,text="Locked study protocol",padding=10); protocol_box.grid(row=2,column=0,columnspan=3,sticky="ew",pady=(0,12))
        ttk.Label(protocol_box,text=self.protocol_text,font=("TkDefaultFont",10,"bold")).pack(anchor="w")
        ttk.Label(protocol_box,text="Primary comparison excludes the turnaround and uses the same central wheel-travel window for every participant.",wraplength=1360).pack(anchor="w",pady=(5,0))

        meta=ttk.LabelFrame(root,text="Required participant metadata",padding=10); meta.grid(row=3,column=0,columnspan=3,sticky="ew",pady=(0,12)); meta.columnconfigure(1,weight=1); meta.columnconfigure(3,weight=1)
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

    # ---------- Tab 2: visualiser ----------
    def _build_visual_tab(self, root: ttk.Frame) -> None:
        root.columnconfigure(0,weight=1); root.rowconfigure(0,weight=1)
        paned=ttk.Panedwindow(root,orient="horizontal"); paned.grid(row=0,column=0,sticky="nsew")

        controls=ttk.Frame(paned,padding=8,width=270); preview=ttk.Frame(paned,padding=6); side=ttk.Frame(paned,padding=8,width=280)
        paned.add(controls,weight=1); paned.add(preview,weight=6); paned.add(side,weight=1)

        ttk.Label(controls,text="Visualiser",font=("TkDefaultFont",16,"bold")).pack(anchor="w",pady=(0,10))
        ttk.Label(controls,text="Style group").pack(anchor="w"); group_box=ttk.Combobox(controls,textvariable=self.group_var,values=("Art","Journey","Analysis","All"),state="readonly",width=23); group_box.pack(fill="x",pady=(3,9)); group_box.bind("<<ComboboxSelected>>",self._group_changed)
        ttk.Label(controls,text="Style").pack(anchor="w"); self.style_box=ttk.Combobox(controls,textvariable=self.style_var,values=visual.STYLE_GROUPS["Art"],state="readonly",width=25); self.style_box.pack(fill="x",pady=(3,9)); self.style_box.bind("<<ComboboxSelected>>",self._style_changed)
        ttk.Label(controls,textvariable=self.style_help_var,wraplength=245,justify="left").pack(anchor="w",pady=(0,12))

        ttk.Separator(controls).pack(fill="x",pady=5)
        for label,var,values in [("Garment preview",self.theme_var,visual.THEMES),("Line density",self.density_var,visual.DENSITIES),("Smoothing",self.smoothing_var,visual.SMOOTHING)]:
            ttk.Label(controls,text=label).pack(anchor="w",pady=(6,0)); box=ttk.Combobox(controls,textvariable=var,values=values,state="readonly",width=22); box.pack(fill="x",pady=(3,3)); box.bind("<<ComboboxSelected>>",lambda _e:self._refresh_preview())
        ttk.Checkbutton(controls,text="Show brief gait analysis",variable=self.annotated_var,command=self._refresh_preview).pack(anchor="w",pady=(8,6))

        zooms=ttk.LabelFrame(controls,text="Preview",padding=6); zooms.pack(fill="x",pady=(10,0)); ttk.Button(zooms,text="Zoom +",command=lambda:self._change_zoom(1.15)).pack(side="left",expand=True,fill="x"); ttk.Button(zooms,text="Zoom -",command=lambda:self._change_zoom(1/1.15)).pack(side="left",expand=True,fill="x",padx=4); ttk.Button(zooms,text="Fit",command=self._fit_preview).pack(side="left",expand=True,fill="x"); ttk.Button(controls,text="Full Screen Preview",command=self._open_fullscreen_preview).pack(fill="x",pady=(8,0))

        preview.rowconfigure(1,weight=1); preview.columnconfigure(0,weight=1); ttk.Label(preview,text="Large live artwork preview",font=("TkDefaultFont",13,"bold")).grid(row=0,column=0,sticky="w",pady=(0,4)); self.preview_canvas=tk.Canvas(preview,width=980,height=760,highlightthickness=0,background="#111111"); self.preview_canvas.grid(row=1,column=0,sticky="nsew"); self.preview_canvas.bind("<Configure>",lambda _e:self._refresh_preview()); self.preview_canvas.bind("<Double-Button-1>",lambda _e:self._open_fullscreen_preview())

        side.columnconfigure(0,weight=1); side.rowconfigure(0,weight=1)
        brief=ttk.LabelFrame(side,text="Brief gait analysis",padding=8); brief.grid(row=0,column=0,sticky="nsew"); self.brief_text=tk.Text(brief,width=30,height=18,wrap="word"); self.brief_text.pack(fill="both",expand=True); self.brief_text.insert("1.0","No analysed session loaded."); self.brief_text.configure(state="disabled")
        exports=ttk.LabelFrame(side,text="Artwork export",padding=8); exports.grid(row=1,column=0,sticky="ew",pady=(8,0)); ttk.Button(exports,text="Export Current SVG...",command=self._export_current_svg).pack(fill="x"); ttk.Button(exports,text="Export Current PNG...",command=self._export_current_png).pack(fill="x",pady=(6,0)); ttk.Button(exports,text=f"Export All {len(visual.STYLES)} SVG Styles",command=self._export_all_styles).pack(fill="x",pady=(6,0)); ttk.Label(exports,text="Exports use the identical locked subject data window. Visual controls change presentation only.",wraplength=250).pack(fill="x",pady=(8,0)); ttk.Label(side,textvariable=self.visual_status_var,wraplength=255).grid(row=2,column=0,sticky="ew",pady=(8,0))

    # ---------- Tab 3: outputs ----------
    def _build_outputs_tab(self, root: ttk.Frame) -> None:
        root.columnconfigure(0,weight=2); root.columnconfigure(1,weight=3); root.rowconfigure(1,weight=1)
        ttk.Label(root,text="Outputs / Subject Record",font=("TkDefaultFont",17,"bold")).grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,10))
        files=ttk.LabelFrame(root,text="Generated files",padding=8); files.grid(row=1,column=0,sticky="nsew",padx=(0,8)); files.rowconfigure(0,weight=1); files.columnconfigure(0,weight=1); self.output_list=tk.Listbox(files); self.output_list.grid(row=0,column=0,sticky="nsew"); fb=ttk.Frame(files); fb.grid(row=1,column=0,sticky="ew",pady=(8,0)); ttk.Button(fb,text="Open Selected",command=self._open_selected_output).pack(side="left"); ttk.Button(fb,text="Open Output Folder",command=self._open_output_folder).pack(side="left",padx=(6,0))
        record=ttk.LabelFrame(root,text="Standard subject record",padding=8); record.grid(row=1,column=1,sticky="nsew"); record.rowconfigure(0,weight=1); record.columnconfigure(0,weight=1); self.record_text=tk.Text(record,wrap="word"); self.record_text.grid(row=0,column=0,sticky="nsew"); self.record_text.insert("1.0","No analysed session loaded."); self.record_text.configure(state="disabled")

    # ---------- common helpers ----------
    def _path_row(self,parent,row:int,label:str,var:tk.StringVar,command)->None:
        ttk.Label(parent,text=label).grid(row=row,column=0,sticky="w",pady=4); ttk.Entry(parent,textvariable=var).grid(row=row,column=1,sticky="ew",padx=8,pady=4); ttk.Button(parent,text="Browse...",command=command).grid(row=row,column=2,pady=4)

    def _swap_positions(self)->None:
        a,b=self.outward_position_var.get(),self.return_position_var.get(); self.outward_position_var.set(b); self.return_position_var.set(a)

    def _choose_wav(self)->None:
        value=filedialog.askopenfilename(title="Choose participant WAV",filetypes=[("WAV audio","*.wav"),("All files","*.*")],initialdir=str(REPO_ROOT/"audio"));
        if value: self.wav_var.set(value); self._suggest_output()

    def _choose_profile(self)->None:
        value=filedialog.askopenfilename(title="Choose calibration profile",filetypes=[("JSON","*.json"),("All files","*.*")],initialdir=str(REPO_ROOT/"calibration"));
        if value: self.profile_var.set(value)

    def _choose_output(self)->None:
        value=filedialog.askdirectory(title="Choose output folder",initialdir=str(REPO_ROOT));
        if value: self.output_var.set(value)

    def _suggest_output(self)->None:
        participant,trial=self.participant_var.get().strip(),self.trial_var.get().strip()
        if participant and trial: self.output_var.set(str(REPO_ROOT/"study_output"/participant/f"trial_{trial}"))
        elif self.wav_var.get() and not self.output_var.get(): self.output_var.set(str(REPO_ROOT/"study_output"/Path(self.wav_var.get()).stem))

    def _clear_session(self)->None:
        self.wav_var.set(""); self.output_var.set(""); self.participant_var.set(""); self.trial_var.set("1"); self.current_output=None; self.visual_data=None; self.status_var.set("Ready"); self.visual_status_var.set("Run an analysis or load an existing study output."); self.results.delete("1.0","end"); self._update_brief_text(); self._update_outputs_tab(); self._refresh_preview()

    # ---------- analysis ----------
    def _start_analysis(self)->None:
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
        def worker()->None:
            try:
                summary=study.analyse_study_session(wav,out,profile,participant_id=participant,trial_id=trial,outward_handlebar_position=outward,return_handlebar_position=ret,protocol_path=DEFAULT_PROTOCOL,manual_turnaround_s=turn); visual.export_all_default_styles(out); self.after(0,lambda:self._show_success(summary,out))
            except Exception:
                err=traceback.format_exc(); self.after(0,lambda:self._show_error(err))
        threading.Thread(target=worker,daemon=True).start()

    def _show_success(self,summary:dict,out:Path)->None:
        self.run_btn.configure(state="normal"); self.status_var.set("Complete - comparable output + artwork written"); comp=summary.get("standardised_primary_comparison") or {}; split=summary.get("phase_split") or {}; det=summary.get("detector") or {}; subject=summary.get("subject") or {}
        lines=[f"Participant: {subject.get('participant_id')}    Trial: {subject.get('trial_id')}",f"Handlebar order: {subject.get('outward_handlebar_position')} -> {subject.get('return_handlebar_position')}",f"Protocol: {summary.get('protocol',{}).get('protocol_id')}",f"Detected clicks: {det.get('detected_hits')}",f"Detector quality: {det.get('quality_status')}",f"Turnaround split: {split.get('source')}"]
        if comp.get("available"):
            om,rm=comp.get("outward") or {},comp.get("return") or {}; lines += [f"Primary comparison window: {comp.get('requested_intervals_per_leg')} intervals per leg",f"Equivalent wheel revolutions per leg: {comp.get('equivalent_wheel_revolutions_per_leg'):.2f}",f"Outward median click rate: {om.get('median_click_rate_hz'):.3f} Hz",f"Return median click rate: {rm.get('median_click_rate_hz'):.3f} Hz",f"Return vs outward change: {comp.get('return_vs_outward_click_rate_change_percent'):.2f}%"]
        else: lines += ["Primary outward/return comparison: unavailable",f"Reason: {comp.get('reason')}"]
        lines += ["",f"Output folder: {out}",f"Standard subject record: {out/study.SUBJECT_RECORD_FILENAME}",f"Artwork folder: {out/'artwork'}"]
        self.results.insert("1.0","\n".join(lines)); self._activate_visualiser(out); self.notebook.select(self.visual_tab); messagebox.showinfo("Analysis complete","Study analysis completed. The Data Visualiser is ready.")

    def _show_error(self,error_text:str)->None:
        self.run_btn.configure(state="normal"); self.status_var.set("Failed"); self.results.insert("1.0",error_text); messagebox.showerror("Analysis failed","The analysis failed. Details are shown in Results.")

    def _load_existing_output(self)->None:
        value=filedialog.askdirectory(title="Choose an existing study output folder",initialdir=str(REPO_ROOT/"study_output"));
        if not value: return
        try: self._activate_visualiser(Path(value)); self.status_var.set("Existing analysis loaded"); self.notebook.select(self.visual_tab)
        except Exception as exc: messagebox.showerror("Cannot load analysis",str(exc))

    def _activate_visualiser(self,output_dir:Path)->None:
        self.current_output=output_dir; self.visual_data=visual.load_visual_data(output_dir); self.visual_status_var.set(f"Loaded {self.visual_data.participant_id}, trial {self.visual_data.trial_id}"); self._update_brief_text(); self._update_outputs_tab(); self._fit_preview()

    # ---------- visual controls ----------
    def _group_changed(self,_event=None)->None:
        group=self.group_var.get(); values=visual.STYLES if group=="All" else visual.STYLE_GROUPS[group]; self.style_box.configure(values=values)
        if self.style_var.get() not in values: self.style_var.set(values[0])
        self._style_changed()

    def _style_changed(self,_event=None)->None:
        self.style_help_var.set(visual.STYLE_HELP.get(self.style_var.get(),"")); self._fit_preview()

    def _change_zoom(self,factor:float)->None:
        self.zoom=max(.55,min(2.5,self.zoom*factor)); self._refresh_preview()

    def _fit_preview(self)->None:
        self.zoom=1.0; self._refresh_preview()

    def _refresh_preview(self)->None:
        if not hasattr(self,"preview_canvas"): return
        if self.visual_data is None:
            self.preview_canvas.delete("all"); self.preview_canvas.create_text(max(380,self.preview_canvas.winfo_width()/2),max(300,self.preview_canvas.winfo_height()/2),text="Analyse a session or load an existing output to preview artwork.",fill="#aaaaaa",font=("TkDefaultFont",13)); return
        visual.draw_on_canvas(self.preview_canvas,self.visual_data,self.style_var.get(),self.theme_var.get(),self.density_var.get(),self.smoothing_var.get(),self.annotated_var.get(),zoom=self.zoom)
        if self.fullscreen_window and self.fullscreen_window.winfo_exists():
            canvas=self.fullscreen_window._preview_canvas; visual.draw_on_canvas(canvas,self.visual_data,self.style_var.get(),self.theme_var.get(),self.density_var.get(),self.smoothing_var.get(),self.annotated_var.get(),zoom=1.0)

    def _open_fullscreen_preview(self)->None:
        if self.visual_data is None: messagebox.showerror("No preview","Load or analyse a session first."); return
        if self.fullscreen_window and self.fullscreen_window.winfo_exists(): self.fullscreen_window.lift(); return
        win=tk.Toplevel(self); self.fullscreen_window=win; win.title("Full Screen Artwork Preview"); win.attributes("-fullscreen",True); canvas=tk.Canvas(win,highlightthickness=0,background="#111111"); canvas.pack(fill="both",expand=True); win._preview_canvas=canvas; ttk.Button(win,text="Close Full Screen",command=win.destroy).place(relx=.98,rely=.02,anchor="ne"); win.bind("<Escape>",lambda _e:win.destroy()); canvas.bind("<Configure>",lambda _e:self._refresh_preview()); self.after(80,self._refresh_preview)

    def _update_brief_text(self)->None:
        self.brief_text.configure(state="normal"); self.brief_text.delete("1.0","end")
        if self.visual_data is None: self.brief_text.insert("1.0","No analysed session loaded.")
        else:
            lines=[f"Participant: {self.visual_data.participant_id}",f"Trial: {self.visual_data.trial_id}",f"Handlebar: {self.visual_data.outward_position} -> {self.visual_data.return_position}",f"Locked window: {self.visual_data.intervals_per_leg} intervals / leg",""]+visual.brief_analysis_lines(self.visual_data)+["","Interpretation note:","Spoke timing is a wheel-motion / movement-rhythm proxy. It is not direct heel-strike or toe-off timing."]
            self.brief_text.insert("1.0","\n".join(lines))
        self.brief_text.configure(state="disabled")

    # ---------- export ----------
    def _default_art_name(self,ext:str)->str:
        stem=self.style_var.get().lower().replace(" ","_").replace("36-","36_"); return stem+("_annotated" if self.annotated_var.get() else "")+ext

    def _export_current_svg(self)->None:
        if self.visual_data is None: messagebox.showerror("Nothing to export","Load or analyse a study session first."); return
        path=filedialog.asksaveasfilename(title="Export print-ready SVG",defaultextension=".svg",initialfile=self._default_art_name(".svg"),filetypes=[("SVG vector artwork","*.svg")]);
        if not path: return
        visual.export_svg(Path(path),self.visual_data,self.style_var.get(),theme=self.theme_var.get(),density=self.density_var.get(),smoothing=self.smoothing_var.get(),annotated=self.annotated_var.get(),transparent=True); self.visual_status_var.set(f"Exported {Path(path).name}")

    def _export_current_png(self)->None:
        if self.visual_data is None: messagebox.showerror("Nothing to export","Load or analyse a study session first."); return
        path=filedialog.asksaveasfilename(title="Export transparent PNG",defaultextension=".png",initialfile=self._default_art_name(".png"),filetypes=[("PNG image","*.png")]);
        if not path: return
        visual.export_png(Path(path),self.visual_data,self.style_var.get(),theme=self.theme_var.get(),density=self.density_var.get(),smoothing=self.smoothing_var.get(),annotated=self.annotated_var.get(),transparent=True); self.visual_status_var.set(f"Exported {Path(path).name}")

    def _export_all_styles(self)->None:
        if self.visual_data is None or self.current_output is None: messagebox.showerror("Nothing to export","Load or analyse a study session first."); return
        paths=visual.export_all_default_styles(self.current_output); self.visual_status_var.set(f"Exported {len(paths)} SVG files to {self.current_output/'artwork'}"); self._update_outputs_tab(); messagebox.showinfo("Artwork exported",f"Artwork written to:\n{self.current_output/'artwork'}")

    # ---------- outputs ----------
    def _update_outputs_tab(self)->None:
        self.output_list.delete(0,"end"); self.record_text.configure(state="normal"); self.record_text.delete("1.0","end")
        if self.current_output is None:
            self.record_text.insert("1.0","No analysed session loaded."); self.record_text.configure(state="disabled"); return
        candidates=[self.current_output/"study_summary.json",self.current_output/"subject_record.csv",self.current_output/"detection"/"session_detections.csv",self.current_output/"detection"/"session_isolated.wav",self.current_output/"movement"/"movement_timeline.csv",self.current_output/"movement"/"movement_summary.json"]
        art=self.current_output/"artwork"
        if art.exists(): candidates += sorted(art.glob("*"))
        self._output_paths=[p for p in candidates if p.exists()]
        for p in self._output_paths: self.output_list.insert("end",str(p.relative_to(self.current_output)))
        record_path=self.current_output/"subject_record.csv"
        if record_path.exists():
            rows=list(csv.DictReader(record_path.open(newline="",encoding="utf-8"))); row=rows[0] if rows else {}; ordered=["participant_id","trial_id","outward_handlebar_position","return_handlebar_position","detected_hits","detector_quality","comparison_intervals_per_leg","equivalent_wheel_revolutions_per_leg","outward_median_click_rate_hz","return_median_click_rate_hz","outward_robust_interval_cv","return_robust_interval_cv","return_vs_outward_click_rate_change_percent","return_adaptation_change_percent","protocol_id"]
            self.record_text.insert("1.0","\n".join(f"{key}: {row.get(key,'')}" for key in ordered))
        else: self.record_text.insert("1.0","Subject record not found.")
        self.record_text.configure(state="disabled")

    def _open_path(self,path:Path)->None:
        try:
            if platform.system()=="Windows": os.startfile(str(path))
            elif platform.system()=="Darwin": subprocess.Popen(["open",str(path)])
            else: subprocess.Popen(["xdg-open",str(path)])
        except Exception as exc: messagebox.showerror("Cannot open",str(exc))

    def _open_selected_output(self)->None:
        sel=self.output_list.curselection();
        if not sel or not hasattr(self,"_output_paths"): return
        self._open_path(self._output_paths[sel[0]])

    def _open_output_folder(self)->None:
        if self.current_output is not None: self._open_path(self.current_output)


if __name__ == "__main__":
    StudySessionApp().mainloop()
