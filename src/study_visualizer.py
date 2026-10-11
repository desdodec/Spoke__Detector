#!/usr/bin/env python3
"""Curated standardised movement visualiser and print-ready export.

The artwork layer is deliberately separated from study measurement. Every style
uses the same locked central interval block created by the study workflow. The
default artwork source is the first/outward leg, giving each participant a
baseline movement portrait before the turnaround/handlebar perturbation.
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

SOURCE_OPTIONS = ("Baseline / first leg", "Return leg")
STYLE_GROUPS = {
    "Signature": (
        "Radial fingerprint",
        "Spoke-wheel barcode",
        "Concentric rhythm rings",
        "36-spoke aggregate glyph",
        "Woven stripes",
    ),
    "Pattern": (
        "Poincare map",
        "Recurrence matrix",
        "Lag similarity map",
        "Grid weave",
    ),
}
STYLES = tuple(style for group in STYLE_GROUPS.values() for style in group)
DENSITIES = ("Low", "Medium", "High")
SMOOTHING = ("Low", "Medium", "High")
THEMES = ("Dark garment", "Light garment", "Transparent")

PALETTE_FAMILIES = {
    "Mono": ("Ink", "Chalk", "Graphite"),
    "Earth": ("Clay & Sand", "Moss & Stone", "Ochre & Slate"),
    "Electric": ("Neon Pulse", "Electric Violet", "Acid Blue"),
    "Heatmap": ("Viridis", "Plasma", "Blue Gold"),
    "Luxury": ("Gold on Black", "Ivory & Navy", "Silver Graphite"),
}
PALETTES = {
    "Ink": ("#0b0b0b", "#363636", "#737373", "#b3b3b3", "#f2f2f2"),
    "Chalk": ("#f7f4ea", "#d9d4c7", "#aaa59a", "#69665f", "#282725"),
    "Graphite": ("#17191c", "#383d42", "#626a70", "#9aa1a6", "#d7dbde"),
    "Clay & Sand": ("#5b2e24", "#9e563d", "#cc8b62", "#e7c39e", "#f5e6cf"),
    "Moss & Stone": ("#24382b", "#526647", "#7f8c68", "#b4b59a", "#e3dfcf"),
    "Ochre & Slate": ("#24323a", "#536b72", "#b18842", "#d5ad57", "#efe0b5"),
    "Neon Pulse": ("#16002e", "#5c16c7", "#cf2fff", "#ff4f9a", "#54f5ff"),
    "Electric Violet": ("#130b35", "#4527a0", "#7b49d6", "#b75cff", "#5cf2ff"),
    "Acid Blue": ("#071a36", "#0e58a8", "#00a7d4", "#42e6c4", "#c7ff54"),
    "Viridis": ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"),
    "Plasma": ("#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"),
    "Blue Gold": ("#071b33", "#155d8b", "#4da5c9", "#d4a72c", "#f7e3a1"),
    "Gold on Black": ("#17120a", "#6f5520", "#b98a2e", "#e7bd61", "#fff0b2"),
    "Ivory & Navy": ("#0d1b2a", "#18344e", "#536d7a", "#c4bca7", "#f5eddc"),
    "Silver Graphite": ("#111316", "#343a40", "#737b82", "#b7bdc2", "#eef0f2"),
}

STYLE_HELP = {
    "Radial fingerprint": "A circular signature of the selected walking leg. Small changes in radius come from changes in the timing between spoke clicks.",
    "Spoke-wheel barcode": "Each interval becomes a radial mark, linking the measured rhythm directly back to the bicycle wheel that captured it.",
    "Concentric rhythm rings": "Several rings show the same movement rhythm at different smoothing scales, from fine fluctuations to broader pacing.",
    "36-spoke aggregate glyph": "Five wheel revolutions are folded onto the wheel's 36 spoke positions to create a compact wheel-specific rhythm portrait.",
    "Woven stripes": "The interval sequence is repeated and offset into a textile-like weave, turning measured movement rhythm into fabric-like structure.",
    "Poincare map": "Each point compares one interval with the next. Tight clusters suggest steadier rhythm; wider spread shows more moment-to-moment variation.",
    "Recurrence matrix": "A map of where similar rhythmic moments reappear. Repeated structures become repeated colours and shapes across the square.",
    "Lag similarity map": "Shows how the rhythm compares with shifted versions of itself, revealing timing structures that repeat after different lags.",
    "Grid weave": "The 180 measured intervals are laid into a fixed grid. Colour represents relative timing, producing a direct textile-like movement field.",
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
        participant_id=str(subject.get("participant_id", "SUBJECT")), trial_id=str(subject.get("trial_id", "")),
        outward_position=str(subject.get("outward_handlebar_position", "A")), return_position=str(subject.get("return_handlebar_position", "B")),
        intervals_per_leg=n, spoke_count=int(protocol.get("locked_spoke_count", summary.get("study_rule", {}).get("spoke_count", 36))),
        outward_intervals=out_center, return_intervals=ret_center,
        outward_rate_hz=float(out_m.get("median_click_rate_hz")), return_rate_hz=float(ret_m.get("median_click_rate_hz")),
        outward_cv=None if out_m.get("robust_interval_cv") is None else float(out_m["robust_interval_cv"]),
        return_cv=None if ret_m.get("robust_interval_cv") is None else float(ret_m["robust_interval_cv"]),
        rate_change_pct=None if comp.get("return_vs_outward_click_rate_change_percent") is None else float(comp["return_vs_outward_click_rate_change_percent"]),
        turnaround_s=turn_s, adaptation_change_pct=None if adapt.get("click_rate_change_percent") is None else float(adapt["click_rate_change_percent"]),
    )


def _selected_intervals(data: VisualData, source: str) -> np.ndarray:
    return data.return_intervals if source == "Return leg" else data.outward_intervals


def _selected_rate_cv(data: VisualData, source: str) -> tuple[float, float | None, str]:
    if source == "Return leg":
        return data.return_rate_hz, data.return_cv, data.return_position
    return data.outward_rate_hz, data.outward_cv, data.outward_position


def _smooth(values: np.ndarray, level: str) -> np.ndarray:
    width = {"Low": 3, "Medium": 7, "High": 11}.get(level, 7)
    if values.size < width: return values.copy()
    half = width // 2; padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _normalised_deviation(intervals: np.ndarray, smoothing: str) -> np.ndarray:
    med = float(np.median(intervals)); dev = np.clip((intervals - med) / max(med, 1e-12), -0.40, 0.40)
    return _smooth(dev, smoothing)


def _density_offsets(level: str) -> list[float]:
    if level == "Low": return [0.0]
    if level == "High": return [0.0, 0.035, 0.070, 0.105, 0.140]
    return [0.0, 0.050, 0.100]


def palettes_for_family(family: str) -> tuple[str, ...]:
    return PALETTE_FAMILIES.get(family, PALETTE_FAMILIES["Luxury"])


def _mix_hex(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, float(t))); av = tuple(int(a[i:i+2], 16) for i in (1,3,5)); bv = tuple(int(b[i:i+2], 16) for i in (1,3,5))
    rgb = tuple(round(x + (y-x)*t) for x,y in zip(av,bv)); return "#" + "".join(f"{v:02x}" for v in rgb)


def _gradient_color(colors: tuple[str, ...], t: float) -> str:
    t = max(0.0, min(1.0, float(t))); pos = t * (len(colors)-1); i = min(len(colors)-2, int(math.floor(pos)))
    return _mix_hex(colors[i], colors[i+1], pos-i)


def _palette(theme: str, palette_name: str) -> dict[str, str]:
    colors = PALETTES.get(palette_name, PALETTES["Gold on Black"])
    if theme == "Dark garment": bg,text,muted = "#101010","#f2f2f2","#aaaaaa"
    elif theme == "Transparent": bg,text,muted = "#ececec","#171717","#6a6a6a"
    else: bg,text,muted = "#faf9f5","#171717","#666666"
    return {"background":bg,"text":text,"muted":muted,"primary":colors[-2],"secondary":colors[1],"echo1":colors[-1],"echo2":colors[2],"marker":colors[-1]}


def _resolve_color(role: str, palette: dict[str, str]) -> str:
    return role if role.startswith("#") else palette.get(role, palette["primary"])


def _base_geometry() -> dict[str, Any]:
    return {"lines":[],"labels":[],"rects":[],"points":[],"polygons":[]}


def _radial(data,source,width,height,density,smoothing,colors):
    seq=_normalised_deviation(_selected_intervals(data,source),smoothing); g=_base_geometry(); cx,cy,size=width/2,height/2,min(width,height); theta=np.linspace(-math.pi/2,3*math.pi/2,len(seq),endpoint=False); base,amp=size*.31,size*.27
    for i,off in enumerate(_density_offsets(density)):
        r=base*(1+off)+amp*seq; pts=[(cx+float(rr)*math.cos(float(t)),cy+float(rr)*math.sin(float(t))) for rr,t in zip(r,theta)]; pts.append(pts[0]); g["lines"].append(("primary" if i==0 else "echo2",pts,3.0 if i==0 else 1.0))
    return g


def _spoke_barcode(data,source,width,height,density,smoothing,colors):
    vals=np.abs(_normalised_deviation(_selected_intervals(data,source),smoothing)); vmax=max(float(np.max(vals)),1e-9); g=_base_geometry(); cx,cy,size=width/2,height/2,min(width,height); theta=np.linspace(-math.pi/2,3*math.pi/2,len(vals),endpoint=False); r0=size*.16
    for i,(th,v) in enumerate(zip(theta,vals)):
        vn=float(v/vmax); r1=size*(.30+.18*vn); col=_gradient_color(colors,i/max(1,len(vals)-1)); g["lines"].append((col,[(cx+r0*math.cos(th),cy+r0*math.sin(th)),(cx+r1*math.cos(th),cy+r1*math.sin(th))],1.5 if density=="High" else 1.0))
    return g


def _concentric_rings(data,source,width,height,density,smoothing,colors):
    intervals=_selected_intervals(data,source); g=_base_geometry(); cx,cy,size=width/2,height/2,min(width,height); theta=np.linspace(-math.pi/2,3*math.pi/2,len(intervals),endpoint=False); levels={"Low":3,"Medium":5,"High":7}[density]
    med=float(np.median(intervals)); raw=np.clip((intervals-med)/max(med,1e-12),-.40,.40)
    for i in range(levels):
        seq=_smooth(raw,"Low" if i<2 else "Medium" if i<4 else "High"); base=size*(.18+.04*i); r=base+size*.12*seq; pts=[(cx+float(rr)*math.cos(float(t)),cy+float(rr)*math.sin(float(t))) for rr,t in zip(r,theta)]; pts.append(pts[0]); g["lines"].append((_gradient_color(colors,i/max(1,levels-1)),pts,2.2 if i==0 else 1.1))
    return g


def _spoke_glyph(data,source,width,height,density,smoothing,colors):
    intervals=_selected_intervals(data,source); spokes=data.spoke_count; usable=(len(intervals)//spokes)*spokes
    if usable<spokes: raise ValueError("Not enough intervals for spoke aggregation.")
    mean=intervals[:usable].reshape(-1,spokes).mean(axis=0); med=float(np.median(mean)); dev=np.clip((mean-med)/max(med,1e-12),-.35,.35); g=_base_geometry(); cx,cy,size=width/2,height/2,min(width,height); th=np.linspace(-math.pi/2,3*math.pi/2,spokes,endpoint=False); r=size*.29+size*.28*dev; pts=[(cx+float(rr)*math.cos(float(t)),cy+float(rr)*math.sin(float(t))) for rr,t in zip(r,th)]; pts.append(pts[0]); g["lines"].append(("primary",pts,2.8))
    for i,t in enumerate(th): g["lines"].append((_gradient_color(colors,i/max(1,spokes-1)),[(cx+size*.11*math.cos(t),cy+size*.11*math.sin(t)),(cx+size*.39*math.cos(t),cy+size*.39*math.sin(t))],.65))
    return g


def _woven(data,source,width,height,density,smoothing,colors):
    seq=_normalised_deviation(_selected_intervals(data,source),smoothing); g=_base_geometry(); rows={"Low":7,"Medium":12,"High":18}[density]; x=np.linspace(width*.06,width*.94,len(seq)); amp=height*.055
    for row in range(rows):
        rolled=np.roll(seq,row*3); y0=height*(.10+.80*row/max(1,rows-1)); pts=[(float(xx),y0+amp*float(v)) for xx,v in zip(x,rolled)]; g["lines"].append((_gradient_color(colors,row/max(1,rows-1)),pts,1.0 if row else 2.0))
    return g


def _poincare(data,source,width,height,density,smoothing,colors):
    vals=_selected_intervals(data,source); xraw,yraw=vals[:-1],vals[1:]; lo,hi=float(vals.min()),float(vals.max()); span=max(hi-lo,1e-9); g=_base_geometry(); left,right,top,bottom=width*.12,width*.88,height*.12,height*.88
    for i,(xv,yv) in enumerate(zip(xraw,yraw)):
        x=left+(float(xv)-lo)/span*(right-left); y=bottom-(float(yv)-lo)/span*(bottom-top); g["points"].append((x,y,3.6 if density=="High" else 2.8,_gradient_color(colors,i/max(1,len(xraw)-1))))
    g["lines"].append(("muted",[(left,bottom),(right,top)],.6)); return g


def _recurrence(data,source,width,height,density,smoothing,colors):
    seq=_normalised_deviation(_selected_intervals(data,source),smoothing); D=np.abs(seq[:,None]-seq[None,:]); vmax=max(float(D.max()),1e-9); g=_base_geometry(); n=len(seq); size=min(width,height)*.78; left,top=(width-size)/2,(height-size)/2; cell=size/n; step=1 if density=="High" else 2 if density=="Medium" else 3
    for i in range(0,n,step):
        for j in range(0,n,step):
            t=1-float(D[i,j]/vmax); g["rects"].append((left+j*cell,top+i*cell,left+min(n,j+step)*cell,top+min(n,i+step)*cell,_gradient_color(colors,t)))
    return g


def _lag_similarity(data,source,width,height,density,smoothing,colors):
    seq=_normalised_deviation(_selected_intervals(data,source),smoothing); max_lag=min(data.spoke_count,len(seq)-1); M=np.vstack([np.abs(seq-np.roll(seq,lag)) for lag in range(1,max_lag+1)]); vmax=max(float(M.max()),1e-9); g=_base_geometry(); left,right,top,bottom=width*.07,width*.93,height*.12,height*.88; cw=(right-left)/len(seq); ch=(bottom-top)/max_lag; step=1 if density=="High" else 2 if density=="Medium" else 3
    for i in range(max_lag):
        for j in range(0,len(seq),step):
            t=1-float(M[i,j]/vmax); g["rects"].append((left+j*cw,top+i*ch,left+min(len(seq),j+step)*cw,top+(i+1)*ch,_gradient_color(colors,t)))
    return g


def _grid_weave(data,source,width,height,density,smoothing,colors):
    seq=_normalised_deviation(_selected_intervals(data,source),smoothing); n=len(seq); rows=12 if n%12==0 else max(2,int(round(math.sqrt(n)))); cols=math.ceil(n/rows); vals=np.pad(seq,(0,rows*cols-n),constant_values=np.nan).reshape(rows,cols); finite=vals[np.isfinite(vals)]; lo,hi=float(finite.min()),float(finite.max()); span=max(hi-lo,1e-9); g=_base_geometry(); left,right,top,bottom=width*.10,width*.90,height*.10,height*.90; cw,ch=(right-left)/cols,(bottom-top)/rows; gap=.8 if density=="High" else 2.0 if density=="Medium" else 4.0
    for r in range(rows):
        for c in range(cols):
            v=vals[r,c]
            if not np.isfinite(v): continue
            t=(float(v)-lo)/span; g["rects"].append((left+c*cw+gap,top+r*ch+gap,left+(c+1)*cw-gap,top+(r+1)*ch-gap,_gradient_color(colors,t)))
    return g


def build_geometry(data: VisualData, style: str, width: int, height: int, density: str = "Medium", smoothing: str = "Medium", theme: str = "Dark garment", source: str = "Baseline / first leg", palette_name: str = "Gold on Black") -> dict[str, Any]:
    colors=PALETTES.get(palette_name,PALETTES["Gold on Black"]); fn={"Radial fingerprint":_radial,"Spoke-wheel barcode":_spoke_barcode,"Concentric rhythm rings":_concentric_rings,"36-spoke aggregate glyph":_spoke_glyph,"Woven stripes":_woven,"Poincare map":_poincare,"Recurrence matrix":_recurrence,"Lag similarity map":_lag_similarity,"Grid weave":_grid_weave}.get(style,_radial)
    return fn(data,source,width,height,density,smoothing,colors)


def brief_analysis_lines(data: VisualData, source: str = "Baseline / first leg") -> list[str]:
    rate,cv,position=_selected_rate_cv(data,source); cv_text="n/a" if cv is None else f"{100*cv:.1f}%"; leg="first/outward leg" if source=="Baseline / first leg" else "return leg"
    return [f"Source: {source}",f"Uses the central {data.intervals_per_leg} measured intervals from the {leg}.",f"Handlebar position: {position}",f"Median movement rhythm: {rate:.2f} clicks/s",f"Robust rhythm variability: {cv_text}"]


def meaning_lines(data: VisualData, style: str, source: str = "Baseline / first leg") -> list[str]:
    source_note="This is the participant's baseline movement portrait before the turnaround and handlebar swap." if source=="Baseline / first leg" else "This portrait uses the standardised return-leg window after the turnaround and handlebar swap."
    return ["WHAT THIS VISUAL MEANS",STYLE_HELP.get(style,"A data-derived movement portrait."),"","ABOUT THIS PERSON'S DATA",source_note,f"Every mark comes from the same locked {data.intervals_per_leg}-interval study window.","Spoke timing measures wheel motion / forward-movement rhythm; it is not direct foot-strike timing."]


def _draw_geometry_canvas(canvas: Any, geom: dict[str, Any], palette: dict[str, str]) -> None:
    for x0,y0,x1,y1,col in geom.get("rects",[]): canvas.create_rectangle(x0,y0,x1,y1,outline="",fill=_resolve_color(col,palette))
    for role,pts,lw in geom.get("lines",[]):
        flat=[coord for p in pts for coord in p]
        if len(flat)>=4: canvas.create_line(*flat,fill=_resolve_color(role,palette),width=lw,smooth=False)
    for x,y,r,col in geom.get("points",[]): canvas.create_oval(x-r,y-r,x+r,y+r,outline="",fill=_resolve_color(col,palette))
    for role,pts in geom.get("polygons",[]):
        flat=[coord for p in pts for coord in p]; canvas.create_polygon(*flat,outline="",fill=_resolve_color(role,palette))


def draw_on_canvas(canvas: Any, data: VisualData, style: str, theme: str, density: str, smoothing: str, annotated: bool, zoom: float = 1.0, source: str = "Baseline / first leg", palette_name: str = "Gold on Black") -> None:
    canvas.update_idletasks(); width=max(760,int(canvas.winfo_width())); height=max(620,int(canvas.winfo_height())); palette=_palette(theme,palette_name); canvas.configure(background=palette["background"]); canvas.delete("all"); draw_w,draw_h=int(width*zoom),int(height*zoom); geom=build_geometry(data,style,draw_w,draw_h,density,smoothing,theme,source,palette_name)
    if zoom!=1.0:
        dx,dy=(width-draw_w)/2,(height-draw_h)/2
        geom["lines"]=[(r,[(x+dx,y+dy) for x,y in pts],lw) for r,pts,lw in geom["lines"]]; geom["rects"]=[(x0+dx,y0+dy,x1+dx,y1+dy,c) for x0,y0,x1,y1,c in geom["rects"]]; geom["points"]=[(x+dx,y+dy,r,c) for x,y,r,c in geom["points"]]; geom["polygons"]=[(r,[(x+dx,y+dy) for x,y in pts]) for r,pts in geom["polygons"]]
    _draw_geometry_canvas(canvas,geom,palette)
    if annotated:
        canvas.create_text(width/2,height*.045,text=f"{data.participant_id} / TRIAL {data.trial_id}",fill=palette["text"],font=("TkDefaultFont",15,"bold")); canvas.create_text(width/2,height*.082,text=f"{source} · {style}",fill=palette["muted"],font=("TkDefaultFont",10))
        for i,line in enumerate(brief_analysis_lines(data,source)[3:]): canvas.create_text(width/2,height*(.91+.03*i),text=line,fill=palette["text"],font=("TkDefaultFont",10))


def _svg_geometry(geom,palette):
    parts=[]
    for x0,y0,x1,y1,col in geom.get("rects",[]): parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1-x0:.2f}" height="{y1-y0:.2f}" fill="{_resolve_color(col,palette)}"/>')
    for role,pts,lw in geom.get("lines",[]): parts.append(f'<polyline fill="none" stroke="{_resolve_color(role,palette)}" stroke-width="{lw:.2f}" stroke-linecap="round" stroke-linejoin="round" points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in pts)}"/>')
    for x,y,r,col in geom.get("points",[]): parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{_resolve_color(col,palette)}"/>')
    for role,pts in geom.get("polygons",[]): parts.append(f'<polygon points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in pts)}" fill="{_resolve_color(role,palette)}"/>')
    return parts


def export_svg(path: Path, data: VisualData, style: str, theme: str = "Dark garment", density: str = "Medium", smoothing: str = "Medium", annotated: bool = False, width: int = 1600, height: int = 1600, transparent: bool = True, source: str = "Baseline / first leg", palette_name: str = "Gold on Black") -> None:
    palette=_palette(theme,palette_name); geom=build_geometry(data,style,width,height,density,smoothing,theme,source,palette_name); parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    if not transparent: parts.append(f'<rect width="100%" height="100%" fill="{palette["background"]}"/>')
    parts.extend(_svg_geometry(geom,palette))
    if annotated: parts += [f'<text x="{width/2:.2f}" y="{height*.045:.2f}" text-anchor="middle" fill="{palette["text"]}" font-family="sans-serif" font-size="28" font-weight="700">{escape(data.participant_id)} / TRIAL {escape(data.trial_id)}</text>',f'<text x="{width/2:.2f}" y="{height*.08:.2f}" text-anchor="middle" fill="{palette["muted"]}" font-family="sans-serif" font-size="18">{escape(source)} · {escape(style)}</text>']
    parts.append("</svg>"); path.parent.mkdir(parents=True,exist_ok=True); path.write_text("\n".join(parts),encoding="utf-8")


def export_png(path: Path, data: VisualData, style: str, theme: str = "Dark garment", density: str = "Medium", smoothing: str = "Medium", annotated: bool = False, width: int = 1800, height: int = 1800, transparent: bool = True, source: str = "Baseline / first leg", palette_name: str = "Gold on Black") -> None:
    palette=_palette(theme,palette_name); bg=(0,0,0,0) if transparent else tuple(int(palette["background"][i:i+2],16) for i in (1,3,5))+(255,); img=Image.new("RGBA",(width,height),bg); draw=ImageDraw.Draw(img); geom=build_geometry(data,style,width,height,density,smoothing,theme,source,palette_name)
    for x0,y0,x1,y1,col in geom.get("rects",[]): draw.rectangle((x0,y0,x1,y1),fill=_resolve_color(col,palette))
    for role,pts,lw in geom.get("lines",[]): draw.line(pts,fill=_resolve_color(role,palette),width=max(1,int(round(lw*2))),joint="curve")
    for x,y,r,col in geom.get("points",[]): draw.ellipse((x-r,y-r,x+r,y+r),fill=_resolve_color(col,palette))
    for role,pts in geom.get("polygons",[]): draw.polygon(pts,fill=_resolve_color(role,palette))
    if annotated:
        try: font=ImageFont.truetype("DejaVuSans.ttf",30)
        except Exception: font=ImageFont.load_default()
        draw.text((width/2,height*.035),f"{data.participant_id} / TRIAL {data.trial_id}",fill=palette["text"],anchor="ma",font=font)
    path.parent.mkdir(parents=True,exist_ok=True); img.save(path)


def export_all_default_styles(output_dir: Path, source: str = "Baseline / first leg", palette_name: str = "Gold on Black") -> list[Path]:
    data=load_visual_data(output_dir); art_dir=output_dir/"artwork"; written=[]
    for style in STYLES:
        stem=style.lower().replace(" ","_").replace("-","_"); path=art_dir/f"{stem}.svg"; export_svg(path,data,style,source=source,palette_name=palette_name,annotated=False,transparent=True); written.append(path)
    annotated=art_dir/"radial_fingerprint_annotated.svg"; export_svg(annotated,data,"Radial fingerprint",source=source,palette_name=palette_name,annotated=True,transparent=True); written.append(annotated); return written
