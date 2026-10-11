#!/usr/bin/env python3
"""Standardised study-data visualisation and print-ready artwork export.

Every visual style is derived from the same locked central interval block written
by the study workflow. Visual controls alter presentation only; they never alter
subject metrics or the scientific comparison window.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STYLE_GROUPS = {
    "Art": (
        "Radial fingerprint",
        "Spoke-wheel barcode",
        "Topographic field",
        "Recurrence matrix",
    ),
    "Journey": (
        "Spiral journey",
        "36-spoke wheel glyph",
    ),
    "Analysis": (
        "Difference ribbon",
        "Mirrored signature",
        "Rhythm barcode",
    ),
}
STYLES = tuple(style for group in STYLE_GROUPS.values() for style in group)
DENSITIES = ("Low", "Medium", "High")
SMOOTHING = ("Low", "Medium", "High")
THEMES = ("Light garment", "Dark garment", "Transparent")

STYLE_HELP = {
    "Radial fingerprint": "Outward and return form two halves of one movement fingerprint. Strongest all-round T-shirt style.",
    "Spoke-wheel barcode": "Each standardised interval becomes a radial spoke-like mark: a literal bridge between wheel motion and artwork.",
    "Topographic field": "Repeated movement traces become contour-like bands with a textile / landscape feel.",
    "Recurrence matrix": "A square pattern showing where similar movement rhythms recur. Highly textile-like and distinctive.",
    "Spiral journey": "Outward expands and return contracts, with a deliberate break for the turn.",
    "36-spoke wheel glyph": "Five revolutions are collapsed onto 36 wheel positions to create a compact wheel signature.",
    "Difference ribbon": "Outward and return are aligned point-for-point; the space between them shows divergence.",
    "Mirrored signature": "Outward and return are mirrored for the clearest sequence-level comparison.",
    "Rhythm barcode": "A minimal strip where each interval becomes one mark. Excellent for cross-subject comparison.",
}


@dataclass
class VisualData:
    participant_id: str
    trial_id: str
    outward_position: str
    return_position: str
    intervals_per_leg: int
    spoke_count: int
    outward_intervals: np.ndarray
    return_intervals: np.ndarray
    outward_rate_hz: float
    return_rate_hz: float
    outward_cv: float | None
    return_cv: float | None
    rate_change_pct: float | None
    turnaround_s: float | None
    adaptation_change_pct: float | None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _central(values: np.ndarray, n: int) -> np.ndarray:
    if values.size < n:
        raise ValueError(f"The selected leg has only {values.size} intervals; the locked study window requires {n}.")
    start = (values.size - n) // 2
    return values[start:start + n]


def load_visual_data(output_dir: Path) -> VisualData:
    summary_path = output_dir / "study_summary.json"
    timeline_path = output_dir / "movement" / "movement_timeline.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Study summary not found: {summary_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"Movement timeline not found: {timeline_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comp = summary.get("standardised_primary_comparison") or {}
    if not comp.get("available"):
        raise ValueError("This session has no standardised outward/return comparison to visualise.")

    rows = _read_csv(timeline_path)
    mid = np.asarray([float(r["midpoint_seconds"]) for r in rows], dtype=float)
    intervals = np.asarray([float(r["interval_seconds"]) for r in rows], dtype=float)
    phases = summary.get("phases") or []
    outward = next((p for p in phases if p.get("phase") == "outward"), None)
    ret = next((p for p in phases if p.get("phase") == "return"), None)
    if outward is None or ret is None:
        raise ValueError("Outward/return phases are unavailable.")

    out_all = intervals[(mid >= float(outward["start_s"])) & (mid <= float(outward["end_s"]))]
    ret_all = intervals[(mid >= float(ret["start_s"])) & (mid <= float(ret["end_s"]))]
    n = int(comp["requested_intervals_per_leg"])
    out_center = _central(out_all, n)
    ret_center = _central(ret_all, n)

    subject = summary.get("subject") or {}
    protocol = summary.get("protocol") or {}
    split = summary.get("phase_split") or {}
    adapt = summary.get("return_adaptation") or {}
    out_m = comp.get("outward") or {}
    ret_m = comp.get("return") or {}
    turn_s = None
    if split.get("turnaround_start_s") is not None and split.get("turnaround_end_s") is not None:
        turn_s = float(split["turnaround_end_s"]) - float(split["turnaround_start_s"])

    return VisualData(
        participant_id=str(subject.get("participant_id", "SUBJECT")),
        trial_id=str(subject.get("trial_id", "")),
        outward_position=str(subject.get("outward_handlebar_position", "A")),
        return_position=str(subject.get("return_handlebar_position", "B")),
        intervals_per_leg=n,
        spoke_count=int(protocol.get("locked_spoke_count", summary.get("study_rule", {}).get("spoke_count", 36))),
        outward_intervals=out_center,
        return_intervals=ret_center,
        outward_rate_hz=float(out_m.get("median_click_rate_hz")),
        return_rate_hz=float(ret_m.get("median_click_rate_hz")),
        outward_cv=None if out_m.get("robust_interval_cv") is None else float(out_m["robust_interval_cv"]),
        return_cv=None if ret_m.get("robust_interval_cv") is None else float(ret_m["robust_interval_cv"]),
        rate_change_pct=None if comp.get("return_vs_outward_click_rate_change_percent") is None else float(comp["return_vs_outward_click_rate_change_percent"]),
        turnaround_s=turn_s,
        adaptation_change_pct=None if adapt.get("click_rate_change_percent") is None else float(adapt["click_rate_change_percent"]),
    )


def _smooth(values: np.ndarray, level: str) -> np.ndarray:
    width = {"Low": 3, "Medium": 7, "High": 11}.get(level, 7)
    if values.size < width:
        return values.copy()
    half = width // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _normalised_deviation(intervals: np.ndarray, smoothing: str) -> np.ndarray:
    med = float(np.median(intervals))
    dev = (intervals - med) / max(med, 1e-12)
    return _smooth(np.clip(dev, -0.32, 0.32), smoothing)


def _rate_deviation(intervals: np.ndarray, smoothing: str) -> np.ndarray:
    rate = 1.0 / intervals
    med = float(np.median(rate))
    return _smooth(np.clip((rate - med) / max(med, 1e-12), -0.32, 0.32), smoothing)


def _density_offsets(level: str) -> list[float]:
    if level == "Low":
        return [0.0]
    if level == "High":
        return [0.0, 0.035, 0.070, 0.105, 0.140]
    return [0.0, 0.050, 0.100, 0.150]


def _palette(theme: str) -> dict[str, str]:
    if theme == "Dark garment":
        return {"background":"#111111","outward":"#5cc8ff","return":"#ff8a3d","echo_out":"#c17bff","echo_ret":"#ff5d73","text":"#f3f3f3","muted":"#b0b0b0","marker":"#f0d44a","matrix_low":"#17172a","matrix_mid":"#7653c6","matrix_high":"#69e6d2"}
    return {"background":"#ffffff","outward":"#1261a0","return":"#c45500","echo_out":"#6b4ba1","echo_ret":"#8a3d3d","text":"#111111","muted":"#666666","marker":"#7a6500","matrix_low":"#f5f0fb","matrix_mid":"#7251a3","matrix_high":"#0f6f78"}


def _mix_hex(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    av = tuple(int(a[i:i+2], 16) for i in (1,3,5))
    bv = tuple(int(b[i:i+2], 16) for i in (1,3,5))
    rgb = tuple(round(x + (y-x)*t) for x,y in zip(av,bv))
    return "#" + "".join(f"{v:02x}" for v in rgb)


def _radial_geometry(data: VisualData, width: int, height: int, density: str, smoothing: str) -> dict[str, Any]:
    out, ret = _normalised_deviation(data.outward_intervals, smoothing), _normalised_deviation(data.return_intervals, smoothing)
    cx, cy, size = width/2, height/2, min(width,height)
    base, amp, gap = size*0.31, size*0.27, math.radians(18)
    t_out = np.linspace(-math.pi/2-gap, -3*math.pi/2+gap, len(out))
    t_ret = np.linspace(math.pi/2+gap, -math.pi/2-gap, len(ret))
    lines=[]
    for i,off in enumerate(_density_offsets(density)):
        ro, rr = base*(1+off)+amp*out, base*(1+off)+amp*ret
        po=[(cx+float(r)*math.cos(float(t)),cy+float(r)*math.sin(float(t))) for r,t in zip(ro,t_out)]
        pr=[(cx+float(r)*math.cos(float(t)),cy+float(r)*math.sin(float(t))) for r,t in zip(rr,t_ret)]
        lines += [("outward" if i==0 else "echo_out",po,3.0 if i==0 else 1.1),("return" if i==0 else "echo_ret",pr,3.0 if i==0 else 1.1)]
    labels=[("START",cx-68,cy-base-55,"muted",12),("END",cx+68,cy-base-55,"muted",12),("TURN",cx,cy+base+60,"marker",12),(f"OUTWARD {data.outward_position}",cx-base-80,cy,"outward",13),(f"RETURN {data.return_position}",cx+base+80,cy,"return",13)]
    return {"lines":lines,"labels":labels,"polygons":[],"rects":[]}


def _spiral_geometry(data: VisualData, width: int, height: int, density: str, smoothing: str) -> dict[str, Any]:
    out, ret = _normalised_deviation(data.outward_intervals,smoothing), _normalised_deviation(data.return_intervals,smoothing)
    cx,cy,size=width/2,height/2,min(width,height); lines=[]
    t_out=np.linspace(-math.pi/2,2.5*math.pi,len(out)); t_ret=np.linspace(2.72*math.pi,5.72*math.pi,len(ret))
    bo=np.linspace(size*.10,size*.35,len(out)); br=np.linspace(size*.35,size*.12,len(ret)); amp=size*.16
    for i,off in enumerate(_density_offsets(density)):
        ro,rr=bo+size*off*.10+amp*out,br+size*off*.10+amp*ret
        po=[(cx+float(r)*math.cos(float(t)),cy+float(r)*math.sin(float(t))) for r,t in zip(ro,t_out)]
        pr=[(cx+float(r)*math.cos(float(t)),cy+float(r)*math.sin(float(t))) for r,t in zip(rr,t_ret)]
        lines += [("outward" if i==0 else "echo_out",po,3 if i==0 else 1),("return" if i==0 else "echo_ret",pr,3 if i==0 else 1)]
    labels=[("START",cx,cy-size*.08,"muted",12),("TURN",cx+size*.36,cy,"marker",12),("END",cx,cy+size*.10,"muted",12)]
    return {"lines":lines,"labels":labels,"polygons":[],"rects":[]}


def _mirrored_geometry(data: VisualData,width:int,height:int,density:str,smoothing:str)->dict[str,Any]:
    out,ret=_rate_deviation(data.outward_intervals,smoothing),_rate_deviation(data.return_intervals,smoothing)
    left,right=width*.08,width*.92; x=np.linspace(left,right,len(out)); upper,lower=height*.33,height*.67; amp=height*.30; lines=[]
    for i,off in enumerate(_density_offsets(density)):
        po=[(float(xx),upper-amp*float(v)-off*height*.14) for xx,v in zip(x,out)]
        pr=[(float(xx),lower+amp*float(v)+off*height*.14) for xx,v in zip(x,ret)]
        lines += [("outward" if i==0 else "echo_out",po,3 if i==0 else 1),("return" if i==0 else "echo_ret",pr,3 if i==0 else 1)]
    labels=[(f"OUTWARD {data.outward_position}",left,height*.12,"outward",13),(f"RETURN {data.return_position}",left,height*.88,"return",13),("TURN",width*.5,height*.5,"marker",11)]
    return {"lines":lines,"labels":labels,"polygons":[],"rects":[]}


def _topographic_geometry(data: VisualData,width:int,height:int,density:str,smoothing:str)->dict[str,Any]:
    seq=_normalised_deviation(np.concatenate([data.outward_intervals,data.return_intervals]),smoothing)
    x=np.linspace(width*.07,width*.93,len(seq)); bands={"Low":5,"Medium":9,"High":14}.get(density,9); amp=height*.055; lines=[]
    for i in range(bands):
        y=height*(.16+.68*i/max(1,bands-1)); pts=[(float(xx),y-amp*float(v)) for xx,v in zip(x,seq)]
        role="outward" if i%2==0 else "return"; lines.append((role if i in (0,bands-1,bands//2) else ("echo_out" if i%2==0 else "echo_ret"),pts,2.0 if i in (0,bands-1,bands//2) else 1.0))
    return {"lines":lines,"labels":[("OUTWARD → TURN → RETURN",width*.5,height*.08,"muted",11)],"polygons":[],"rects":[]}


def _difference_ribbon_geometry(data:VisualData,width:int,height:int,density:str,smoothing:str)->dict[str,Any]:
    out,ret=_rate_deviation(data.outward_intervals,smoothing),_rate_deviation(data.return_intervals,smoothing); n=min(len(out),len(ret)); out,ret=out[:n],ret[:n]
    x=np.linspace(width*.08,width*.92,n); mid=height*.50; amp=height*.55
    po=[(float(xx),mid-amp*float(v)) for xx,v in zip(x,out)]; pr=[(float(xx),mid-amp*float(v)) for xx,v in zip(x,ret)]
    poly=po+list(reversed(pr))
    return {"lines":[("outward",po,3),("return",pr,3)],"labels":[("OUTWARD / RETURN DIFFERENCE",width*.5,height*.10,"muted",11)],"polygons":[("echo_out",poly,0.24)],"rects":[]}


def _rhythm_barcode_geometry(data:VisualData,width:int,height:int,density:str,smoothing:str)->dict[str,Any]:
    out,ret=_normalised_deviation(data.outward_intervals,smoothing),_normalised_deviation(data.return_intervals,smoothing); n=len(out); gap=width*.04; left=width*.06; usable=width*.88; step=usable/n
    lines=[]; mult={"Low":.65,"Medium":1.0,"High":1.35}.get(density,1.0)
    for i,(ov,rv) in enumerate(zip(out,ret)):
        x=left+i*step; lo=height*(.36-.18*abs(float(ov))*mult); hi=height*.36; lr=height*.64; hr=height*(.64+.18*abs(float(rv))*mult)
        lines.append(("outward",[(x,lo),(x,hi)],max(.8,step*.55))); lines.append(("return",[(x,lr),(x,hr)],max(.8,step*.55)))
    return {"lines":lines,"labels":[(f"OUTWARD {data.outward_position}",left,height*.18,"outward",12),(f"RETURN {data.return_position}",left,height*.82,"return",12),("TURN",width*.5,height*.50,"marker",11)],"polygons":[],"rects":[]}


def _spoke_barcode_geometry(data:VisualData,width:int,height:int,density:str,smoothing:str)->dict[str,Any]:
    out,ret=_normalised_deviation(data.outward_intervals,smoothing),_normalised_deviation(data.return_intervals,smoothing); cx,cy,size=width/2,height/2,min(width,height); inner=size*.18; base=size*.34; amp=size*.25; lines=[]
    tout=np.linspace(-math.pi/2,math.pi/2,len(out),endpoint=False); tret=np.linspace(math.pi/2,3*math.pi/2,len(ret),endpoint=False); lw={"Low":1.0,"Medium":1.6,"High":2.2}.get(density,1.6)
    for th,v in zip(tout,out):
        r=base+amp*abs(float(v)); lines.append(("outward",[(cx+inner*math.cos(th),cy+inner*math.sin(th)),(cx+r*math.cos(th),cy+r*math.sin(th))],lw))
    for th,v in zip(tret,ret):
        r=base+amp*abs(float(v)); lines.append(("return",[(cx+inner*math.cos(th),cy+inner*math.sin(th)),(cx+r*math.cos(th),cy+r*math.sin(th))],lw))
    return {"lines":lines,"labels":[(f"OUT {data.outward_position}",cx+base*.65,cy,"outward",12),(f"RETURN {data.return_position}",cx-base*.65,cy,"return",12)],"polygons":[],"rects":[]}


def _wheel_glyph_geometry(data:VisualData,width:int,height:int,density:str,smoothing:str)->dict[str,Any]:
    spokes=max(1,data.spoke_count); out=np.asarray(data.outward_intervals); ret=np.asarray(data.return_intervals)
    out_bins=np.array([np.mean(out[np.arange(len(out))%spokes==i]) for i in range(spokes)]); ret_bins=np.array([np.mean(ret[np.arange(len(ret))%spokes==i]) for i in range(spokes)])
    outd=(out_bins-np.median(out_bins))/np.median(out_bins); retd=(ret_bins-np.median(ret_bins))/np.median(ret_bins); outd=_smooth(np.r_[outd,outd[:10]],smoothing)[:spokes]; retd=_smooth(np.r_[retd,retd[:10]],smoothing)[:spokes]
    cx,cy,size=width/2,height/2,min(width,height); theta=np.linspace(-math.pi/2,3*math.pi/2,spokes,endpoint=False); base1,base2=size*.24,size*.36; amp=size*.20
    po=[(cx+(base1+amp*float(v))*math.cos(float(t)),cy+(base1+amp*float(v))*math.sin(float(t))) for t,v in zip(theta,outd)]; pr=[(cx+(base2+amp*float(v))*math.cos(float(t)),cy+(base2+amp*float(v))*math.sin(float(t))) for t,v in zip(theta,retd)]; po.append(po[0]); pr.append(pr[0]); lines=[("outward",po,3),("return",pr,3)]
    if density!="Low":
        for t in theta: lines.append(("muted",[(cx+size*.12*math.cos(t),cy+size*.12*math.sin(t)),(cx+size*.44*math.cos(t),cy+size*.44*math.sin(t))],.7 if density=="Medium" else 1.0))
    return {"lines":lines,"labels":[(f"36-SPOKE MOVEMENT GLYPH",cx,height*.08,"muted",11)],"polygons":[],"rects":[]}


def _recurrence_geometry(data:VisualData,width:int,height:int,density:str,smoothing:str,theme:str)->dict[str,Any]:
    seq=np.concatenate([_normalised_deviation(data.outward_intervals,smoothing),_normalised_deviation(data.return_intervals,smoothing)])
    target={"Low":48,"Medium":72,"High":96}.get(density,72); idx=np.linspace(0,len(seq)-1,target).astype(int); s=seq[idx]; d=np.abs(s[:,None]-s[None,:]); vmax=max(float(np.quantile(d,.95)),1e-9); d=np.clip(d/vmax,0,1)
    pal=_palette(theme); size=min(width,height)*.76; x0=(width-size)/2; y0=(height-size)/2; cell=size/target; rects=[]
    for r in range(target):
        for c in range(target):
            v=float(d[r,c]); color=_mix_hex(pal["matrix_low"],pal["matrix_mid"],min(1,v*1.4)) if v<.7 else _mix_hex(pal["matrix_mid"],pal["matrix_high"],(v-.7)/.3)
            rects.append((color,x0+c*cell,y0+r*cell,x0+(c+1)*cell+0.5,y0+(r+1)*cell+0.5))
    return {"lines":[],"labels":[("RECURRENCE OF MOVEMENT RHYTHM",width*.5,height*.07,"muted",11)],"polygons":[],"rects":rects}


def build_geometry(data:VisualData,style:str,width:int,height:int,density:str="Medium",smoothing:str="Medium",theme:str="Dark garment")->dict[str,Any]:
    if style=="Spiral journey": return _spiral_geometry(data,width,height,density,smoothing)
    if style=="Mirrored signature": return _mirrored_geometry(data,width,height,density,smoothing)
    if style=="Topographic field": return _topographic_geometry(data,width,height,density,smoothing)
    if style=="Difference ribbon": return _difference_ribbon_geometry(data,width,height,density,smoothing)
    if style=="Rhythm barcode": return _rhythm_barcode_geometry(data,width,height,density,smoothing)
    if style=="Spoke-wheel barcode": return _spoke_barcode_geometry(data,width,height,density,smoothing)
    if style=="36-spoke wheel glyph": return _wheel_glyph_geometry(data,width,height,density,smoothing)
    if style=="Recurrence matrix": return _recurrence_geometry(data,width,height,density,smoothing,theme)
    return _radial_geometry(data,width,height,density,smoothing)


def brief_analysis_lines(data:VisualData)->list[str]:
    rate_change="n/a" if data.rate_change_pct is None else f"{data.rate_change_pct:+.1f}%"; out_cv="n/a" if data.outward_cv is None else f"{100*data.outward_cv:.1f}%"; ret_cv="n/a" if data.return_cv is None else f"{100*data.return_cv:.1f}%"; turn="n/a" if data.turnaround_s is None else f"{data.turnaround_s:.1f} s"; adapt="n/a" if data.adaptation_change_pct is None else f"{data.adaptation_change_pct:+.1f}%"
    return [f"OUT {data.outward_rate_hz:.2f} clicks/s   RETURN {data.return_rate_hz:.2f} clicks/s",f"RETURN vs OUTWARD {rate_change}",f"RHYTHM VARIABILITY {out_cv} / {ret_cv}",f"TURN {turn}   RETURN ADAPTATION {adapt}"]


def _role_color(palette:dict[str,str],role:str)->str:
    return palette.get(role, role if str(role).startswith("#") else palette["text"])


def draw_on_canvas(canvas:Any,data:VisualData,style:str,theme:str,density:str,smoothing:str,annotated:bool,zoom:float=1.0)->None:
    canvas.update_idletasks(); width=max(760,int(canvas.winfo_width())); height=max(620,int(canvas.winfo_height())); palette=_palette(theme); bg=palette["background"] if theme!="Transparent" else "#202020"; canvas.configure(background=bg); canvas.delete("all"); geom=build_geometry(data,style,width,height,density,smoothing,theme)
    for color,x0,y0,x1,y1 in geom.get("rects",[]): canvas.create_rectangle(x0,y0,x1,y1,fill=color,outline="")
    for role,pts,alpha in geom.get("polygons",[]):
        flat=[c for p in pts for c in p]; canvas.create_polygon(*flat,fill=_role_color(palette,role),outline="",stipple="gray50" if alpha<.5 else "")
    for role,pts,line_width in geom.get("lines",[]):
        flat=[c for p in pts for c in p]
        if len(flat)>=4: canvas.create_line(*flat,fill=_role_color(palette,role),width=line_width,smooth=False)
    if annotated:
        for text,x,y,role,size in geom.get("labels",[]): canvas.create_text(x,y,text=text,fill=_role_color(palette,role),font=("TkDefaultFont",size,"bold"))
        canvas.create_text(width/2,height*.045,text=f"{data.participant_id} / TRIAL {data.trial_id}",fill=palette["text"],font=("TkDefaultFont",15,"bold"))
        for i,line in enumerate(brief_analysis_lines(data)): canvas.create_text(width/2,height*(.86+.03*i),text=line,fill=palette["text"] if i==0 else palette["muted"],font=("TkDefaultFont",10))
    if abs(zoom-1.0)>1e-6: canvas.scale("all",width/2,height/2,zoom,zoom)


def export_svg(path:Path,data:VisualData,style:str,theme:str="Light garment",density:str="Medium",smoothing:str="Medium",annotated:bool=False,width:int=1600,height:int=1600,transparent:bool=True)->None:
    palette=_palette(theme); geom=build_geometry(data,style,width,height,density,smoothing,theme); parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    if not transparent: parts.append(f'<rect width="100%" height="100%" fill="{palette["background"]}"/>')
    for color,x0,y0,x1,y1 in geom.get("rects",[]): parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1-x0:.2f}" height="{y1-y0:.2f}" fill="{color}"/>')
    for role,pts,alpha in geom.get("polygons",[]):
        points=" ".join(f"{x:.2f},{y:.2f}" for x,y in pts); parts.append(f'<polygon points="{points}" fill="{_role_color(palette,role)}" fill-opacity="{alpha:.3f}"/>')
    for role,pts,lw in geom.get("lines",[]):
        points=" ".join(f"{x:.2f},{y:.2f}" for x,y in pts); parts.append(f'<polyline fill="none" stroke="{_role_color(palette,role)}" stroke-width="{lw:.2f}" stroke-linecap="round" stroke-linejoin="round" points="{points}"/>')
    if annotated:
        for text,x,y,role,size in geom.get("labels",[]): parts.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" fill="{_role_color(palette,role)}" font-family="sans-serif" font-size="{size*1.35:.1f}" font-weight="700">{escape(text)}</text>')
        parts.append(f'<text x="{width/2:.2f}" y="{height*.045:.2f}" text-anchor="middle" fill="{palette["text"]}" font-family="sans-serif" font-size="24" font-weight="700">{escape(data.participant_id)} / TRIAL {escape(data.trial_id)}</text>')
        for i,line in enumerate(brief_analysis_lines(data)): parts.append(f'<text x="{width/2:.2f}" y="{height*(.86+.03*i):.2f}" text-anchor="middle" fill="{palette["text"] if i==0 else palette["muted"]}" font-family="sans-serif" font-size="16">{escape(line)}</text>')
    parts.append("</svg>"); path.parent.mkdir(parents=True,exist_ok=True); path.write_text("\n".join(parts),encoding="utf-8")


def _font(size:int,bold:bool=False):
    names=["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf","Arial.ttf"]
    for name in names:
        try: return ImageFont.truetype(name,size)
        except OSError: pass
    return ImageFont.load_default()


def export_png(path:Path,data:VisualData,style:str,theme:str="Dark garment",density:str="Medium",smoothing:str="Medium",annotated:bool=False,width:int=1800,height:int=1800,transparent:bool=True)->None:
    palette=_palette(theme); bg=(0,0,0,0) if transparent else palette["background"]; image=Image.new("RGBA",(width,height),bg); draw=ImageDraw.Draw(image,"RGBA"); geom=build_geometry(data,style,width,height,density,smoothing,theme)
    for color,x0,y0,x1,y1 in geom.get("rects",[]): draw.rectangle((x0,y0,x1,y1),fill=color)
    for role,pts,alpha in geom.get("polygons",[]):
        col=_role_color(palette,role); rgb=tuple(int(col[i:i+2],16) for i in (1,3,5))+(int(255*alpha),); draw.polygon(pts,fill=rgb)
    for role,pts,lw in geom.get("lines",[]): draw.line(pts,fill=_role_color(palette,role),width=max(1,round(lw*width/1600)),joint="curve")
    if annotated:
        for text,x,y,role,size in geom.get("labels",[]): draw.text((x,y),text,fill=_role_color(palette,role),font=_font(round(size*1.7),True),anchor="mm")
        draw.text((width/2,height*.045),f"{data.participant_id} / TRIAL {data.trial_id}",fill=palette["text"],font=_font(28,True),anchor="mm")
        for i,line in enumerate(brief_analysis_lines(data)): draw.text((width/2,height*(.86+.03*i)),line,fill=palette["text"] if i==0 else palette["muted"],font=_font(20),anchor="mm")
    path.parent.mkdir(parents=True,exist_ok=True); image.save(path)


def export_all_default_styles(output_dir:Path)->list[Path]:
    data=load_visual_data(output_dir); art_dir=output_dir/"artwork"; written=[]
    for style in STYLES:
        stem=style.lower().replace(" ","_").replace("36-","36_"); svg=art_dir/f"{stem}.svg"; export_svg(svg,data,style,annotated=False,transparent=True); written.append(svg)
    annotated=art_dir/"radial_fingerprint_annotated.svg"; export_svg(annotated,data,"Radial fingerprint",annotated=True,transparent=True); written.append(annotated)
    return written
