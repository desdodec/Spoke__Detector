#!/usr/bin/env python3
from __future__ import annotations
import os, platform, subprocess, tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.io import wavfile
import transient_lab as lab

BG='#090d12'; PANEL='#111924'; PANEL2='#172231'; BORDER='#283649'; TEXT='#eff6ff'; MUTED='#8492a6'
CYAN='#39e7ff'; LIME='#b9ff57'; AMBER='#ffbd59'; RED='#ff5b75'; PURPLE='#a783ff'; GRID='#1b2634'; SEL='#153a45'

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('Spoke Detector — Acoustic Click Lab'); self.geometry('1600x920'); self.minsize(1150,720); self.configure(bg=BG)
        self.audio=None; self.filtered=None; self.score=None; self.events=[]; self.view=[0.,1.]; self.sel=[None,None]; self.drag=False; self.hover=None; self.tmp=[]
        self.path=tk.StringVar(); self.status=tk.StringVar(value='Open a WAV to begin.'); self.meta=tk.StringVar(value='No file loaded')
        self.th=tk.DoubleVar(value=lab.DEFAULT_THRESHOLD_DBFS); self.sep=tk.DoubleVar(value=lab.DEFAULT_MIN_SEPARATION_MS); self.hp=tk.DoubleVar(value=lab.DEFAULT_HIGHPASS_HZ); self.expected=tk.IntVar(value=4)
        self.count=tk.StringVar(value='0'); self.scount=tk.StringVar(value='—'); self.span=tk.StringVar(value='—'); self.hover_text=tk.StringVar(value='Hover over waveform')
        self._style(); self._ui(); self.bind('<space>',lambda e:self.play()); self.protocol('WM_DELETE_WINDOW',self.close)

    def _style(self):
        s=ttk.Style(self)
        try:s.theme_use('clam')
        except tk.TclError:pass
        for n,b,f in [('D.TFrame',BG,TEXT),('P.TFrame',PANEL,TEXT),('M.TLabel',BG,MUTED),('P.TLabel',PANEL,TEXT),('PM.TLabel',PANEL,MUTED)]:s.configure(n,background=b,foreground=f)
        s.configure('Hero.TLabel',background=BG,foreground=TEXT,font=('TkDefaultFont',20,'bold')); s.configure('Metric.TLabel',background=PANEL,foreground=TEXT,font=('TkDefaultFont',24,'bold'))
        s.configure('Accent.TButton',background=CYAN,foreground='#031216',padding=(12,8),borderwidth=0,font=('TkDefaultFont',10,'bold')); s.configure('Dark.TButton',background=PANEL2,foreground=TEXT,padding=(10,7),bordercolor=BORDER)
        s.map('Accent.TButton',background=[('active','#74efff')]); s.map('Dark.TButton',background=[('active','#223047')]); s.configure('Dark.Horizontal.TScale',background=PANEL,troughcolor='#263447'); s.configure('Dark.TEntry',fieldbackground=PANEL2,foreground=TEXT)

    def _ui(self):
        self.rowconfigure(1,weight=1); self.columnconfigure(0,weight=1)
        h=ttk.Frame(self,style='D.TFrame',padding=(18,14,18,10)); h.grid(row=0,column=0,sticky='ew'); h.columnconfigure(1,weight=1)
        t=ttk.Frame(h,style='D.TFrame'); t.grid(row=0,column=0,sticky='w'); ttk.Label(t,text='ACOUSTIC CLICK LAB',style='Hero.TLabel').pack(anchor='w'); ttk.Label(t,text='forensic transient microscope · sample-refined timing · human validation',style='M.TLabel').pack(anchor='w')
        f=ttk.Frame(h,style='D.TFrame'); f.grid(row=0,column=1,sticky='ew',padx=24); f.columnconfigure(0,weight=1); ttk.Entry(f,textvariable=self.path,state='readonly',style='Dark.TEntry').grid(row=0,column=0,sticky='ew'); ttk.Label(f,textvariable=self.meta,style='M.TLabel').grid(row=1,column=0,sticky='w',pady=(3,0))
        ttk.Button(h,text='OPEN WAV',style='Accent.TButton',command=self.open).grid(row=0,column=2,padx=(0,8)); ttk.Button(h,text='EXPORT',style='Dark.TButton',command=self.export).grid(row=0,column=3)
        body=ttk.Frame(self,style='D.TFrame',padding=(18,0,18,10)); body.grid(row=1,column=0,sticky='nsew'); body.rowconfigure(0,weight=1); body.columnconfigure(1,weight=1)
        left=ttk.Frame(body,style='P.TFrame',width=270,padding=14); left.grid(row=0,column=0,sticky='nsw',padx=(0,12)); left.grid_propagate(False); self.left=left
        mid=ttk.Frame(body,style='D.TFrame'); mid.grid(row=0,column=1,sticky='nsew'); mid.rowconfigure(1,weight=1); mid.columnconfigure(0,weight=1)
        right=ttk.Frame(body,style='P.TFrame',width=250,padding=14); right.grid(row=0,column=2,sticky='nse',padx=(12,0)); right.grid_propagate(False)
        self._controls(left); self._stage(mid); self._stats(right)
        foot=ttk.Frame(self,style='P.TFrame',padding=(18,9)); foot.grid(row=2,column=0,sticky='ew'); foot.columnconfigure(1,weight=1); ttk.Button(foot,text='▶ PLAY',style='Dark.TButton',command=self.play).grid(row=0,column=0); ttk.Label(foot,textvariable=self.status,style='PM.TLabel').grid(row=0,column=1,sticky='w',padx=15); ttk.Label(foot,text='acoustic events only',style='PM.TLabel').grid(row=0,column=2)

    def _controls(self,p):
        ttk.Label(p,text='DETECTOR',style='PM.TLabel',font=('TkDefaultFont',9,'bold')).pack(anchor='w'); ttk.Label(p,text='Tune the acoustic boundary, not the walking rhythm.',style='PM.TLabel',wraplength=235).pack(anchor='w',pady=(3,12))
        self._slider(p,'THRESHOLD',self.th,-90,-10,'dBFS','%.1f'); self._slider(p,'MIN SEPARATION',self.sep,5,150,'ms','%.0f'); self._slider(p,'HIGH-PASS',self.hp,100,5000,'Hz','%.0f')
        ttk.Button(p,text='DETECT FULL WAV',style='Accent.TButton',command=self.detect_full).pack(fill='x',pady=(7,5)); ttk.Button(p,text='RE-DETECT VIEW',style='Dark.TButton',command=self.detect_view).pack(fill='x')
        ttk.Separator(p).pack(fill='x',pady=16); ttk.Label(p,text='KNOWN-COUNT TRAINING',style='PM.TLabel',font=('TkDefaultFont',9,'bold')).pack(anchor='w'); ttk.Label(p,text='Drag-select a region you can confidently count.',style='PM.TLabel',wraplength=235).pack(anchor='w',pady=(4,8))
        r=ttk.Frame(p,style='P.TFrame'); r.pack(fill='x'); ttk.Label(r,text='Real clicks',style='P.TLabel').pack(side='left'); tk.Spinbox(r,from_=0,to=999,textvariable=self.expected,width=6,bg=PANEL2,fg=TEXT,insertbackground=TEXT,buttonbackground=PANEL2,relief='flat').pack(side='right')
        ttk.Button(p,text='FIT N CLICKS',style='Dark.TButton',command=self.fit).pack(fill='x',pady=(8,5)); ttk.Button(p,text='SAVE CALIBRATION SAMPLE',style='Dark.TButton',command=self.save_ref).pack(fill='x')
        ttk.Separator(p).pack(fill='x',pady=16); ttk.Label(p,text='GESTURES',style='PM.TLabel',font=('TkDefaultFont',9,'bold')).pack(anchor='w'); ttk.Label(p,text='Drag  select\nWheel  zoom at cursor\nDouble-click  zoom in\nShift-click  add event\nRight-click  remove nearest\nSpace  play',style='PM.TLabel',justify='left').pack(anchor='w',pady=(5,0))

    def _slider(self,p,name,var,a,b,unit,fmt):
        box=ttk.Frame(p,style='P.TFrame'); box.pack(fill='x',pady=(0,11)); row=ttk.Frame(box,style='P.TFrame'); row.pack(fill='x'); ttk.Label(row,text=name,style='PM.TLabel',font=('TkDefaultFont',8,'bold')).pack(side='left'); val=ttk.Label(row,text=(fmt%var.get())+' '+unit,style='P.TLabel',font=('TkDefaultFont',10,'bold')); val.pack(side='right')
        sc=ttk.Scale(box,from_=a,to=b,variable=var,orient='horizontal',style='Dark.Horizontal.TScale'); sc.pack(fill='x',pady=(5,0)); sc.bind('<B1-Motion>',lambda e:val.configure(text=(fmt%var.get())+' '+unit)); sc.bind('<ButtonRelease-1>',lambda e:(val.configure(text=(fmt%var.get())+' '+unit),self.curve(),self.draw()))

    def _stage(self,p):
        bar=ttk.Frame(p,style='P.TFrame',padding=(12,9)); bar.grid(row=0,column=0,sticky='ew'); bar.columnconfigure(1,weight=1); ttk.Label(bar,text='TRANSIENT MICROSCOPE',style='P.TLabel',font=('TkDefaultFont',11,'bold')).grid(row=0,column=0); ttk.Label(bar,textvariable=self.hover_text,style='PM.TLabel').grid(row=0,column=1,sticky='e')
        self.c=tk.Canvas(p,bg=BG,highlightthickness=1,highlightbackground=BORDER,cursor='crosshair'); self.c.grid(row=1,column=0,sticky='nsew'); self.c.bind('<Configure>',lambda e:self.draw()); self.c.bind('<ButtonPress-1>',self.select_start); self.c.bind('<B1-Motion>',self.select_drag); self.c.bind('<ButtonRelease-1>',self.select_end); self.c.bind('<Double-Button-1>',lambda e:self.zoom(self.x_to_t(e.x),.35)); self.c.bind('<Shift-Button-1>',self.add); self.c.bind('<Button-3>',self.remove); self.c.bind('<Motion>',self.motion); self.c.bind('<Leave>',lambda e:self.leave()); self.c.bind('<MouseWheel>',self.wheel)
        self.over=tk.Canvas(p,bg='#0c1118',height=76,highlightthickness=1,highlightbackground=BORDER); self.over.grid(row=2,column=0,sticky='ew',pady=(8,0)); self.over.bind('<Configure>',lambda e:self.draw_over()); self.over.bind('<Button-1>',self.jump)

    def _stats(self,p):
        ttk.Label(p,text='SESSION',style='PM.TLabel',font=('TkDefaultFont',9,'bold')).pack(anchor='w'); self._metric(p,'EVENTS',self.count); self._metric(p,'IN SELECTION',self.scount); self._metric(p,'VISIBLE WINDOW',self.span); ttk.Separator(p).pack(fill='x',pady=15); ttk.Label(p,text='WHY THIS VIEW',style='PM.TLabel',font=('TkDefaultFont',9,'bold')).pack(anchor='w'); ttk.Label(p,text='Top lane: the real waveform.\n\nBottom lane: transient energy used by the detector.\n\nRed markers: automatic events.\nPurple markers: manually added events.\n\nAmber line: fixed acoustic threshold.',style='PM.TLabel',wraplength=215,justify='left').pack(anchor='w',pady=(7,0))
    def _metric(self,p,n,v): ttk.Label(p,text=n,style='PM.TLabel',font=('TkDefaultFont',8,'bold')).pack(anchor='w',pady=(12,0)); ttk.Label(p,textvariable=v,style='Metric.TLabel').pack(anchor='w')

    def open(self):
        p=filedialog.askopenfilename(filetypes=[('WAV audio','*.wav'),('All files','*.*')]);
        if not p:return
        try:self.audio=lab.load_wav(Path(p)); self.path.set(p); self.events=[]; self.view=[0.,self.audio.duration_s]; self.sel=[None,None]; self.curve(); self.meta.set(f'{self.audio.duration_s:.3f} s · {self.audio.sample_rate:,} Hz · mono analysis'); self.status.set(f'Loaded {Path(p).name}'); self.update(); self.draw(); self.draw_over()
        except Exception as e:messagebox.showerror('Cannot open WAV',str(e))
    def curve(self):
        if self.audio is not None:self.filtered,_,self.score=lab.transient_curve(self.audio,self.hp.get())
    def detect(self,a,b,replace):
        if self.audio is None:return
        new=lab.detect_events(self.audio,self.th.get(),self.sep.get(),self.hp.get(),a,b); self.events=new if replace else sorted([e for e in self.events if not a<=e.timestamp_seconds<=b]+new,key=lambda e:e.timestamp_seconds); self.curve(); self.update(); self.draw(); self.draw_over()
    def detect_full(self):
        if self.audio:self.detect(0,self.audio.duration_s,True); self.status.set(f'Detected {len(self.events)} acoustic events at {self.th.get():.1f} dBFS')
    def detect_view(self):
        if self.audio:self.detect(*self.view,False); self.status.set('Re-detected current view')
    def region(self):
        if None in self.sel:return None
        a,b=sorted(self.sel); return (a,b) if b-a>.001 else None
    def fit(self):
        r=self.region();
        if self.audio is None or not r:return
        a,b=r; th,new=lab.fit_threshold_for_count(self.audio,int(self.expected.get()),a,b,self.sep.get(),self.hp.get()); self.th.set(th); self.events=sorted([e for e in self.events if not a<=e.timestamp_seconds<=b]+new,key=lambda e:e.timestamp_seconds); self.update(); self.draw(); self.draw_over(); self.status.set(f'Fitted {len(new)} clicks · threshold {th:.2f} dBFS')
    def save_ref(self):
        r=self.region();
        if self.audio is None or not r:return
        a,b=r; ev=[e for e in self.events if a<=e.timestamp_seconds<=b]; p=filedialog.asksaveasfilename(defaultextension='.jsonl',initialfile='click_calibration.jsonl',filetypes=[('JSON Lines','*.jsonl')]);
        if p:lab.append_reference(Path(p),self.audio,a,b,int(self.expected.get()),ev,self.th.get(),self.sep.get(),self.hp.get()); self.status.set(f'Saved calibration sample → {Path(p).name}')
    def export(self):
        if self.audio is None:return
        p=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='clicks.csv',filetypes=[('CSV','*.csv')]);
        if p:lab.export_events_csv(Path(p),self.events); self.status.set(f'Exported {len(self.events)} reviewed events → {Path(p).name}')

    def x_to_t(self,x):return self.view[0]+max(0,min(1,x/max(1,self.c.winfo_width())))*(self.view[1]-self.view[0])
    def t_to_x(self,t):return (t-self.view[0])/max(1e-12,self.view[1]-self.view[0])*max(1,self.c.winfo_width())
    def select_start(self,e):
        if e.state&1:return
        self.drag=True; self.sel=[self.x_to_t(e.x),self.x_to_t(e.x)]; self.draw()
    def select_drag(self,e):
        if self.drag:self.sel[1]=self.x_to_t(e.x); self.update(); self.draw()
    def select_end(self,e):
        if self.drag:self.sel[1]=self.x_to_t(e.x); self.drag=False; self.update(); self.draw()
    def add(self,e):
        if self.audio is None:return
        sr=self.audio.sample_rate; i=int(self.x_to_t(e.x)*sr); r=max(1,int(.007*sr)); a,b=max(0,i-r),min(len(self.filtered),i+r+1); j=a+int(np.argmax(np.abs(self.filtered[a:b]))); amp=float(abs(self.filtered[j])); self.events.append(lab.Event(j/sr,float(self.score[j]),amp,amp,'manual_added')); self.events.sort(key=lambda e:e.timestamp_seconds); self.update(); self.draw(); self.draw_over()
    def remove(self,e):
        if not self.events:return
        t=self.x_to_t(e.x); i=min(range(len(self.events)),key=lambda j:abs(self.events[j].timestamp_seconds-t));
        if abs(self.events[i].timestamp_seconds-t)<=max(.008,(self.view[1]-self.view[0])*.012):self.events.pop(i); self.update(); self.draw(); self.draw_over()
    def motion(self,e):
        if self.audio is None:return
        self.hover=self.x_to_t(e.x); i=min(len(self.score)-1,max(0,int(self.hover*self.audio.sample_rate))); self.hover_text.set(f'{self.hover:.6f} s · score {self.score[i]:.1f} dBFS'); self.draw()
    def leave(self):self.hover=None; self.hover_text.set('Hover over waveform'); self.draw()
    def wheel(self,e):self.zoom(self.x_to_t(e.x),.78 if e.delta>0 else 1.28)
    def zoom(self,t,f):
        if self.audio is None:return
        old=self.view[1]-self.view[0]; new=min(self.audio.duration_s,max(.025,old*f)); frac=(t-self.view[0])/max(old,1e-12); a=min(max(0,t-frac*new),max(0,self.audio.duration_s-new)); self.view=[a,a+new]; self.update(); self.draw(); self.draw_over()
    def jump(self,e):
        if self.audio is None:return
        t=e.x/max(1,self.over.winfo_width())*self.audio.duration_s; sp=self.view[1]-self.view[0]; a=min(max(0,t-sp/2),max(0,self.audio.duration_s-sp)); self.view=[a,a+sp]; self.update(); self.draw(); self.draw_over()
    def play(self):
        if self.audio is None:return
        r=self.region() or tuple(self.view); a,b=r; sr=self.audio.sample_rate; lo,hi=int(a*sr),int(b*sr); fd,p=tempfile.mkstemp(suffix='.wav'); os.close(fd); wavfile.write(p,sr,np.round(np.clip(self.audio.mono[lo:hi],-1,1)*32767).astype(np.int16)); self.tmp.append(p)
        try:
            if platform.system()=='Windows':import winsound; winsound.PlaySound(p,winsound.SND_FILENAME|winsound.SND_ASYNC)
            elif platform.system()=='Darwin':subprocess.Popen(['afplay',p])
            else:subprocess.Popen(['aplay',p],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except Exception as e:messagebox.showerror('Playback unavailable',str(e))
    def update(self):
        self.count.set(str(len(self.events))); r=self.region(); self.scount.set('—' if not r else str(sum(r[0]<=e.timestamp_seconds<=r[1] for e in self.events))); self.span.set(f'{self.view[1]-self.view[0]:.2f}s')

    def ticks(self,w):
        sp=self.view[1]-self.view[0]; raw=sp/max(4,min(12,w//120)); p=10**np.floor(np.log10(raw)); step=min((1,2,5,10),key=lambda n:abs(n*p-raw))*p; t=np.ceil(self.view[0]/step)*step; out=[]; d=max(0,int(-np.floor(np.log10(step)))+1) if step<1 else 1
        while t<=self.view[1]+step*.5:out.append((t,f'{t:.{d}f}s')); t+=step
        return out
    def draw(self):
        c=self.c; c.delete('all'); w,h=max(10,c.winfo_width()),max(10,c.winfo_height()); c.create_rectangle(0,0,w,h,fill=BG,outline='')
        if self.audio is None:c.create_text(w/2,h/2-16,text='OPEN A WAV',fill=TEXT,font=('TkDefaultFont',20,'bold')); c.create_text(w/2,h/2+18,text='Waveform, transient energy and validated markers will appear here.',fill=MUTED,font=('TkDefaultFont',11)); return
        ruler=34; wt=46; wb=int(h*.64); st=wb+18; sb=h-28; sr=self.audio.sample_rate; lo=max(0,int(self.view[0]*sr)); hi=min(len(self.audio.mono),int(self.view[1]*sr))
        for t,l in self.ticks(w):x=self.t_to_x(t); c.create_line(x,ruler,x,h,fill=GRID); c.create_text(x+4,8,text=l,anchor='nw',fill=MUTED,font=('TkDefaultFont',8))
        c.create_line(0,ruler,w,ruler,fill=BORDER); c.create_text(10,wt,text='WAVEFORM',anchor='nw',fill='#62748a',font=('TkDefaultFont',8,'bold')); c.create_text(10,st,text='TRANSIENT ENERGY',anchor='nw',fill='#62748a',font=('TkDefaultFont',8,'bold'))
        y=self.audio.mono[lo:hi]; cols=max(1,min(w,len(y))); ed=np.linspace(0,len(y),cols+1).astype(int); mid=(wt+wb)/2; amp=(wb-wt)*.44; pk=max(float(np.max(np.abs(y))),1e-9)
        for x in range(cols):
            a,b=ed[x],ed[x+1]
            if b>a:z=y[a:b]; c.create_line(x,mid-float(np.max(z))/pk*amp,x,mid-float(np.min(z))/pk*amp,fill='#5bd8eb')
        if self.score is not None:
            idx=np.linspace(lo,hi-1,max(2,min(w*2,hi-lo))).astype(int); pts=[]
            for j,v in enumerate(self.score[idx]):x=j/max(1,len(idx)-1)*w; n=(max(-100,min(-5,float(v)))+100)/95; pts.extend((x,sb-n*(sb-st-20)))
            c.create_line(*pts,fill='#1fa7c3',width=1.2); n=(max(-100,min(-5,self.th.get()))+100)/95; yy=sb-n*(sb-st-20); c.create_line(0,yy,w,yy,fill=AMBER,dash=(5,4)); c.create_text(w-10,yy-7,text=f'{self.th.get():.1f} dBFS',anchor='e',fill=AMBER,font=('TkDefaultFont',8,'bold'))
        r=self.region()
        if r and r[1]>=self.view[0] and r[0]<=self.view[1]:x0,x1=self.t_to_x(max(r[0],self.view[0])),self.t_to_x(min(r[1],self.view[1])); c.create_rectangle(x0,ruler,x1,h,fill=SEL,stipple='gray50',outline=''); c.create_line(x0,ruler,x0,h,fill=LIME); c.create_line(x1,ruler,x1,h,fill=LIME); c.create_text((x0+x1)/2,ruler+8,text=f'{r[1]-r[0]:.4f}s · {sum(r[0]<=e.timestamp_seconds<=r[1] for e in self.events)} events',anchor='n',fill=LIME,font=('TkDefaultFont',9,'bold'))
        vis=[e for e in self.events if self.view[0]<=e.timestamp_seconds<=self.view[1]]
        for e in vis:x=self.t_to_x(e.timestamp_seconds); col=PURPLE if e.source=='manual_added' else RED; c.create_line(x,ruler,x,wb,fill=col,width=2); c.create_polygon(x-5,ruler,x+5,ruler,x,ruler+8,fill=col,outline='')
        if self.hover is not None:x=self.t_to_x(self.hover); c.create_line(x,ruler,x,h,fill='#607087',dash=(2,3)); c.create_text(x,h-14,text=f'{self.hover:.6f}s',fill=TEXT,font=('TkDefaultFont',8))
    def draw_over(self):
        c=self.over; c.delete('all'); w,h=max(10,c.winfo_width()),max(10,c.winfo_height()); c.create_rectangle(0,0,w,h,fill='#0c1118',outline='')
        if self.audio is None:return
        y=self.audio.mono; ed=np.linspace(0,len(y),min(w,len(y))+1).astype(int); pk=max(float(np.max(np.abs(y))),1e-9); mid,amp=h/2,h*.34
        for x in range(len(ed)-1):
            a,b=ed[x],ed[x+1]
            if b>a:z=y[a:b]; c.create_line(x,mid-float(np.max(z))/pk*amp,x,mid-float(np.min(z))/pk*amp,fill='#335667')
        for e in self.events:x=e.timestamp_seconds/self.audio.duration_s*w; c.create_line(x,7,x,h-7,fill='#7c3a48')
        c.create_rectangle(self.view[0]/self.audio.duration_s*w,2,self.view[1]/self.audio.duration_s*w,h-2,outline=CYAN,width=2); c.create_text(8,7,text='OVERVIEW · click to recenter',anchor='nw',fill='#62748a',font=('TkDefaultFont',8,'bold'))
    def close(self):
        for p in self.tmp:
            try:os.unlink(p)
            except OSError:pass
        self.destroy()

if __name__=='__main__':App().mainloop()
