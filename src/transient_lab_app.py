#!/usr/bin/env python3
"""Human-in-the-loop WAV transient analyser for acoustic click review."""
from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from scipy.io import wavfile

import transient_lab as lab


class TransientLabApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector - Acoustic Click Lab")
        self.geometry("1480x900")
        self.minsize(1050, 700)

        self.audio: lab.AudioData | None = None
        self.filtered = None
        self.score_dbfs = None
        self.events: list[lab.Event] = []
        self.view_start = 0.0
        self.view_end = 1.0
        self.sel_start: float | None = None
        self.sel_end: float | None = None
        self._dragging = False

        self.wav_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Open a WAV to begin.")
        self.threshold_var = tk.DoubleVar(value=lab.DEFAULT_THRESHOLD_DBFS)
        self.min_sep_var = tk.DoubleVar(value=lab.DEFAULT_MIN_SEPARATION_MS)
        self.highpass_var = tk.DoubleVar(value=lab.DEFAULT_HIGHPASS_HZ)
        self.expected_var = tk.IntVar(value=4)
        self.count_var = tk.StringVar(value="0 events")
        self.region_var = tk.StringVar(value="No selection")
        self._build()

    def _build(self) -> None:
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Acoustic Click Lab", font=("TkDefaultFont", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(top, text="Acoustic events only — no spoke-count or gait-timing assumptions.").grid(row=0, column=1, sticky="w", padx=12)
        ttk.Button(top, text="Open WAV...", command=self.open_wav).grid(row=0, column=2)
        ttk.Entry(top, textvariable=self.wav_var, state="readonly").grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6,0))

        body = ttk.Frame(self, padding=(10,0,10,8))
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        left = ttk.Frame(body, width=280)
        left.grid(row=0, column=0, sticky="nsw", padx=(0,10))
        stage = ttk.Frame(body)
        stage.grid(row=0, column=1, sticky="nsew")
        stage.rowconfigure(1, weight=1)
        stage.columnconfigure(0, weight=1)

        det = ttk.LabelFrame(left, text="Detector", padding=9)
        det.pack(fill="x")
        self._scale(det, "Threshold (dBFS)", self.threshold_var, -90, -10, .5)
        self._scale(det, "Min separation ms", self.min_sep_var, 5, 150, 1)
        self._scale(det, "High-pass Hz", self.highpass_var, 100, 5000, 50)
        ttk.Button(det, text="Detect Full WAV", command=self.detect_full).pack(fill="x", pady=(6,2))
        ttk.Button(det, text="Detect Current View", command=self.detect_view).pack(fill="x")
        ttk.Label(det, textvariable=self.count_var, font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(8,0))

        ref = ttk.LabelFrame(left, text="Known-count reference", padding=9)
        ref.pack(fill="x", pady=(10,0))
        ttk.Label(ref, text="Drag over a short region, listen, then say how many real clicks it contains.", wraplength=250).pack(anchor="w")
        row = ttk.Frame(ref); row.pack(fill="x", pady=(7,4))
        ttk.Label(row, text="Expected clicks").pack(side="left")
        tk.Spinbox(row, from_=0, to=999, textvariable=self.expected_var, width=6).pack(side="right")
        ttk.Button(ref, text="Fit N Clicks in Selection", command=self.fit_selection).pack(fill="x", pady=2)
        ttk.Button(ref, text="Add Selection to Calibration...", command=self.save_reference).pack(fill="x", pady=2)
        ttk.Label(ref, textvariable=self.region_var, wraplength=250).pack(anchor="w", pady=(6,0))

        edit = ttk.LabelFrame(left, text="Review / edit", padding=9)
        edit.pack(fill="x", pady=(10,0))
        ttk.Label(edit, text="Double-click: add marker\nRight-click marker: remove", justify="left").pack(anchor="w")
        ttk.Button(edit, text="Play Selection", command=self.play_selection).pack(fill="x", pady=(7,2))
        ttk.Button(edit, text="Play Current View", command=self.play_view).pack(fill="x", pady=2)
        ttk.Button(edit, text="Export Reviewed clicks.csv...", command=self.export_csv).pack(fill="x", pady=(7,2))

        nav = ttk.LabelFrame(left, text="View", padding=9)
        nav.pack(fill="x", pady=(10,0))
        ttk.Button(nav, text="Zoom to Selection", command=self.zoom_selection).pack(fill="x", pady=2)
        ttk.Button(nav, text="Zoom Out ×2", command=self.zoom_out).pack(fill="x", pady=2)
        ttk.Button(nav, text="Full WAV", command=self.full_view).pack(fill="x", pady=2)

        bar = ttk.Frame(stage)
        bar.grid(row=0, column=0, sticky="ew", pady=(0,5))
        ttk.Label(bar, text="Waveform + transient score", font=("TkDefaultFont", 13, "bold")).pack(side="left")
        ttk.Label(bar, text="waveform above · transient score below · red=automatic · purple=manual").pack(side="right")
        self.canvas = tk.Canvas(stage, bg="white", highlightthickness=1, highlightbackground="#bbb")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self._select_start)
        self.canvas.bind("<B1-Motion>", self._select_drag)
        self.canvas.bind("<ButtonRelease-1>", self._select_end)
        self.canvas.bind("<Double-Button-1>", self._add_event)
        self.canvas.bind("<Button-3>", self._remove_event)

        ttk.Label(self, textvariable=self.status_var, padding=(10,0,10,10)).grid(row=2, column=0, sticky="w")

    def _scale(self, parent, label, var, lo, hi, resolution) -> None:
        ttk.Label(parent, text=label).pack(anchor="w")
        tk.Scale(parent, from_=lo, to=hi, resolution=resolution, variable=var, orient="horizontal", length=245).pack(fill="x", pady=(0,4))

    def open_wav(self) -> None:
        value = filedialog.askopenfilename(title="Open WAV", filetypes=[("WAV audio","*.wav"),("All files","*.*")])
        if not value: return
        try:
            self.audio = lab.load_wav(Path(value))
            self.wav_var.set(value)
            self.events = []
            self.view_start, self.view_end = 0.0, self.audio.duration_s
            self.sel_start = self.sel_end = None
            self._curve()
            self.status_var.set(f"Loaded {Path(value).name} — {self.audio.duration_s:.3f}s at {self.audio.sample_rate}Hz")
            self.redraw()
        except Exception as exc:
            messagebox.showerror("Cannot open WAV", str(exc))

    def _curve(self) -> None:
        if self.audio is None: return
        self.filtered, _, self.score_dbfs = lab.transient_curve(self.audio, self.highpass_var.get())

    def _detect(self, a: float, b: float, replace_all: bool) -> None:
        if self.audio is None: return
        try:
            new = lab.detect_events(self.audio, self.threshold_var.get(), self.min_sep_var.get(), self.highpass_var.get(), a, b)
            if replace_all:
                self.events = new
            else:
                keep = [e for e in self.events if not (a <= e.timestamp_seconds <= b)]
                self.events = sorted(keep + new, key=lambda e:e.timestamp_seconds)
            self._curve(); self._update_labels(); self.redraw()
        except Exception as exc:
            messagebox.showerror("Detection failed", str(exc))

    def detect_full(self) -> None:
        if self.audio:
            self._detect(0.0, self.audio.duration_s, True)
            self.status_var.set(f"Detected {len(self.events)} events at {self.threshold_var.get():.1f} dBFS")

    def detect_view(self) -> None:
        if self.audio:
            self._detect(self.view_start, self.view_end, False)
            self.status_var.set("Re-detected current view.")

    def _selection(self):
        if self.sel_start is None or self.sel_end is None: return None
        a,b = sorted((self.sel_start,self.sel_end))
        return (a,b) if b-a >= .001 else None

    def fit_selection(self) -> None:
        if self.audio is None: return
        region = self._selection()
        if region is None:
            messagebox.showinfo("Select a region", "Drag across the waveform first."); return
        a,b = region
        try:
            th,new = lab.fit_threshold_for_count(self.audio, int(self.expected_var.get()), a,b, self.min_sep_var.get(), self.highpass_var.get())
            self.threshold_var.set(round(th,2))
            keep = [e for e in self.events if not (a <= e.timestamp_seconds <= b)]
            self.events = sorted(keep+new, key=lambda e:e.timestamp_seconds)
            self._update_labels(); self.redraw()
            self.status_var.set(f"Selection fitted to {len(new)} events; suggested threshold {th:.2f} dBFS")
        except Exception as exc:
            messagebox.showerror("Cannot fit selection", str(exc))

    def save_reference(self) -> None:
        if self.audio is None: return
        region = self._selection()
        if region is None:
            messagebox.showinfo("Select a region", "Select and review a region first."); return
        a,b = region
        selected = [e for e in self.events if a <= e.timestamp_seconds <= b]
        expected = int(self.expected_var.get())
        if len(selected) != expected and not messagebox.askyesno("Count mismatch", f"Expected {expected}, but {len(selected)} markers are in the selection. Save anyway?"):
            return
        value = filedialog.asksaveasfilename(title="Calibration library", defaultextension=".jsonl", initialfile="click_calibration.jsonl", filetypes=[("JSON Lines","*.jsonl")])
        if value:
            lab.append_reference(Path(value), self.audio, a,b,expected,selected,self.threshold_var.get(),self.min_sep_var.get(),self.highpass_var.get())
            self.status_var.set(f"Added reviewed selection to {Path(value).name}")

    def export_csv(self) -> None:
        if self.audio is None: return
        value = filedialog.asksaveasfilename(title="Export reviewed click events", defaultextension=".csv", initialfile="clicks.csv", filetypes=[("CSV","*.csv")])
        if value:
            lab.export_events_csv(Path(value), self.events)
            self.status_var.set(f"Exported {len(self.events)} reviewed events to {Path(value).name}")

    def _x_to_time(self, x: float) -> float:
        w=max(1,self.canvas.winfo_width())
        return self.view_start + max(0,min(1,x/w))*(self.view_end-self.view_start)

    def _time_to_x(self, t: float) -> float:
        return (t-self.view_start)/max(1e-12,self.view_end-self.view_start)*max(1,self.canvas.winfo_width())

    def _select_start(self,e):
        self._dragging=True; self.sel_start=self._x_to_time(e.x); self.sel_end=self.sel_start; self.redraw()
    def _select_drag(self,e):
        if self._dragging: self.sel_end=self._x_to_time(e.x); self._update_labels(); self.redraw()
    def _select_end(self,e):
        if self._dragging: self.sel_end=self._x_to_time(e.x); self._dragging=False; self._update_labels(); self.redraw()

    def _add_event(self,e):
        if self.audio is None: return
        if self.filtered is None: self._curve()
        sr=self.audio.sample_rate; idx=int(round(self._x_to_time(e.x)*sr)); r=max(1,int(.007*sr)); lo=max(0,idx-r); hi=min(len(self.filtered),idx+r+1)
        if hi<=lo: return
        refined=lo+int(np.argmax(np.abs(self.filtered[lo:hi]))); t=refined/sr; amp=float(abs(self.filtered[refined]))
        self.events.append(lab.Event(t, float(self.score_dbfs[refined]), amp, amp, "manual_added"))
        self.events.sort(key=lambda z:z.timestamp_seconds); self._update_labels(); self.redraw()

    def _remove_event(self,e):
        if not self.events: return
        t=self._x_to_time(e.x); tol=max(.008,(self.view_end-self.view_start)*.012)
        i=min(range(len(self.events)), key=lambda j:abs(self.events[j].timestamp_seconds-t))
        if abs(self.events[i].timestamp_seconds-t)<=tol:
            self.events.pop(i); self._update_labels(); self.redraw()

    def _update_labels(self):
        self.count_var.set(f"{len(self.events)} events")
        region=self._selection()
        if region is None: self.region_var.set("No selection")
        else:
            a,b=region; n=sum(a<=e.timestamp_seconds<=b for e in self.events)
            self.region_var.set(f"{a:.4f}–{b:.4f}s ({b-a:.3f}s) · {n} markers")

    def zoom_selection(self):
        region=self._selection()
        if self.audio is None or region is None: return
        a,b=region; pad=(b-a)*.08; self.view_start=max(0,a-pad); self.view_end=min(self.audio.duration_s,b+pad); self.redraw()
    def zoom_out(self):
        if self.audio is None: return
        mid=(self.view_start+self.view_end)/2; span=min(self.audio.duration_s,(self.view_end-self.view_start)*2)
        self.view_start=max(0,mid-span/2); self.view_end=min(self.audio.duration_s,self.view_start+span); self.view_start=max(0,self.view_end-span); self.redraw()
    def full_view(self):
        if self.audio: self.view_start,self.view_end=0,self.audio.duration_s; self.redraw()

    def _play(self,a:float,b:float):
        if self.audio is None: return
        sr=self.audio.sample_rate; lo=max(0,int(a*sr)); hi=min(len(self.audio.mono),int(b*sr))
        if hi<=lo: return
        y=np.clip(self.audio.mono[lo:hi],-1,1); pcm=np.round(y*32767).astype(np.int16)
        fd,name=tempfile.mkstemp(suffix='.wav'); os.close(fd); wavfile.write(name,sr,pcm)
        try:
            if platform.system()=="Windows":
                import winsound; winsound.PlaySound(name,winsound.SND_FILENAME|winsound.SND_ASYNC)
            elif platform.system()=="Darwin": subprocess.Popen(["afplay",name])
            else: subprocess.Popen(["aplay",name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except Exception as exc: messagebox.showerror("Playback unavailable",str(exc))
    def play_selection(self):
        region=self._selection()
        if region: self._play(*region)
    def play_view(self):
        if self.audio: self._play(self.view_start,self.view_end)

    def redraw(self):
        c=self.canvas; c.delete("all")
        if self.audio is None:
            c.create_text(max(300,c.winfo_width()/2),max(250,c.winfo_height()/2),text="Open a WAV to inspect acoustic transients.",fill="#555",font=("TkDefaultFont",14)); return
        w=max(10,c.winfo_width()); h=max(10,c.winfo_height()); sr=self.audio.sample_rate
        lo=max(0,int(self.view_start*sr)); hi=min(len(self.audio.mono),int(self.view_end*sr))
        if hi<=lo:return
        target=max(200,min(w*2,hi-lo)); idx=np.linspace(lo,hi-1,target).astype(int); y=self.audio.mono[idx]
        mx=max(float(np.max(np.abs(y))),1e-9); mid=h*.34; amp=h*.28; pts=[]
        for j,v in enumerate(y): pts.extend((j/max(1,len(y)-1)*w,mid-(float(v)/mx)*amp))
        if len(pts)>=4:c.create_line(*pts,fill="#222",width=1)
        c.create_line(0,h*.67,w,h*.67,fill="#bbb")
        if self.score_dbfs is not None:
            floor,ceil=-100,-5; s=self.score_dbfs[idx]; pts=[]
            for j,v in enumerate(s):
                yy=h*.95-(max(floor,min(ceil,float(v)))-floor)/(ceil-floor)*(h*.24); pts.extend((j/max(1,len(s)-1)*w,yy))
            if len(pts)>=4:c.create_line(*pts,fill="#3366aa",width=1)
            th=self.threshold_var.get(); yy=h*.95-(max(floor,min(ceil,th))-floor)/(ceil-floor)*(h*.24)
            c.create_line(0,yy,w,yy,fill="#c06060",dash=(4,3)); c.create_text(6,yy-8,anchor="w",text=f"threshold {th:.1f} dBFS",fill="#9a3333")
        for ev in self.events:
            if self.view_start<=ev.timestamp_seconds<=self.view_end:
                x=self._time_to_x(ev.timestamp_seconds); col="#7b2cbf" if ev.source=="manual_added" else "#c40000"
                c.create_line(x,6,x,h*.66,fill=col,width=2); c.create_oval(x-3,5,x+3,11,fill=col,outline="")
        region=self._selection()
        if region:
            a,b=region
            if b>=self.view_start and a<=self.view_end:
                x0=self._time_to_x(max(a,self.view_start)); x1=self._time_to_x(min(b,self.view_end)); c.create_rectangle(x0,0,x1,h,outline="#2c7a3f",width=2,dash=(5,4))
        c.create_text(6,h-8,anchor="sw",text=f"view {self.view_start:.4f}–{self.view_end:.4f}s",fill="#555")


if __name__ == "__main__":
    TransientLabApp().mainloop()
