#!/usr/bin/env python3
"""Second-generation generative art engine for Spoke Detector.

The engine keeps the scientific layer fixed: artwork is generated from the
locked central 180-interval outward slice (five 36-spoke revolutions).  The
return slice remains available for paired movement-rhythm analysis elsewhere.
Design parameters alter rendering only; they never change the selected data.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

import study_visualizer as visual

CANVAS = 1000

FAMILIES: dict[str, str] = {
    "Tension Bloom": "A tensile network grown from similarities between moments in the five-revolution movement portrait.",
    "Orbit Armour": "Five data-driven phase rings become a dense circular object with spokes, teeth and moire-like interference.",
    "Flow Fault": "The movement sequence bends a field of horizontal paths, creating a fault-like disturbance through a calm surface.",
    "Root Signal": "Longer and shorter timing deviations grow an organic branching structure from the participant's rhythm.",
    "Glyph Constellation": "Small groups of click intervals become a field of unique data-born marks rather than a conventional chart.",
    "Spoke Loom": "The five 36-click revolutions are treated as warp and weft, making a textile-like structure tied to the wheel geometry.",
    "Resonant Dust": "The interval sequence controls a particle field whose density follows resonant nodal bands.",
    "Helix Groove": "All 180 intervals travel through a continuous spiral; timing changes alter wobble, spacing and particle emphasis.",
    "Angular Choir": "Repeated angular routes use the five revolutions as related voices inside a circular frame.",
    "Fivefold Moire": "Five revolution-specific line systems overlap and interfere, turning small rhythm differences into large optical structure.",
}

PALETTES: dict[str, tuple[str, str, str, str]] = {
    "Signal Mint": ("#00e6c7", "#c7ff33", "#139cff", "#f4fffc"),
    "Ultraviolet": ("#8a5cff", "#ff4fc8", "#56e7ff", "#fff6ff"),
    "Solar Flare": ("#ff6238", "#ffbf2f", "#ff4e8a", "#fff2dd"),
    "Deep Sea": ("#28c9e8", "#3c7dff", "#57e39f", "#efffff"),
    "Earth Metal": ("#d9b56b", "#c86842", "#6fa66a", "#fff0d5"),
    "Monochrome": ("#ffffff", "#b7c1c7", "#728087", "#e9fffb"),
}

BACKGROUND_COLORS = {
    "Black": "#0d1012",
    "White": "#f3f1eb",
    "Transparent": "#00000000",
}


@dataclass
class DesignParams:
    family: str = "Tension Bloom"
    palette: str = "Signal Mint"
    background: str = "Black"
    stroke_width: float = 1.7
    detail: int = 120
    jitter: float = 0.18
    symmetry: int = 7
    scale: float = 0.92
    rotation_deg: float = 0.0
    opacity: float = 0.92
    seed: int = 1

    def validated(self) -> "DesignParams":
        self.family = self.family if self.family in FAMILIES else next(iter(FAMILIES))
        self.palette = self.palette if self.palette in PALETTES else next(iter(PALETTES))
        self.background = self.background if self.background in BACKGROUND_COLORS else "Black"
        self.stroke_width = float(max(0.25, min(5.0, self.stroke_width)))
        self.detail = int(max(24, min(240, self.detail)))
        self.jitter = float(max(0.0, min(1.0, self.jitter)))
        self.symmetry = int(max(1, min(16, self.symmetry)))
        self.scale = float(max(0.55, min(1.2, self.scale)))
        self.rotation_deg = float(self.rotation_deg % 360.0)
        self.opacity = float(max(0.15, min(1.0, self.opacity)))
        self.seed = int(self.seed)
        return self


@dataclass(frozen=True)
class StudioData:
    participant_id: str
    trial_id: str
    intervals: np.ndarray
    revolutions: np.ndarray
    norm: np.ndarray
    rev_norm: np.ndarray
    spoke_count: int


def load_studio_data(output_dir: Path) -> StudioData:
    data = visual.load_visual_data(Path(output_dir))
    intervals = np.asarray(data.outward_intervals, dtype=float)
    if intervals.size != data.intervals_per_leg:
        raise ValueError("Artwork source does not match the locked comparison interval count.")
    if intervals.size % data.spoke_count != 0:
        raise ValueError("Locked interval count must divide cleanly into complete wheel revolutions.")
    med = float(np.median(intervals))
    norm = np.clip((intervals - med) / max(med, 1e-12), -0.45, 0.45)
    rev = intervals.reshape(-1, data.spoke_count)
    rev_med = np.median(rev, axis=1, keepdims=True)
    rev_norm = np.clip((rev - rev_med) / np.maximum(rev_med, 1e-12), -0.45, 0.45)
    return StudioData(
        participant_id=data.participant_id,
        trial_id=data.trial_id,
        intervals=intervals,
        revolutions=rev,
        norm=norm,
        rev_norm=rev_norm,
        spoke_count=data.spoke_count,
    )


def randomised_params(current: DesignParams | None = None, *, keep_family: bool = True, seed: int | None = None) -> DesignParams:
    seed = random.SystemRandom().randrange(1, 2**31 - 1) if seed is None else int(seed)
    rng = random.Random(seed)
    family = current.family if current and keep_family else rng.choice(tuple(FAMILIES))
    background = current.background if current else "Black"
    return DesignParams(
        family=family,
        palette=rng.choice(tuple(PALETTES)),
        background=background,
        stroke_width=round(rng.uniform(0.75, 2.7), 2),
        detail=rng.randint(70, 190),
        jitter=round(rng.uniform(0.05, 0.48), 3),
        symmetry=rng.randint(4, 12),
        scale=round(rng.uniform(0.78, 1.06), 3),
        rotation_deg=round(rng.uniform(0, 360), 1),
        opacity=round(rng.uniform(0.72, 1.0), 3),
        seed=seed,
    ).validated()


def _sample(values: np.ndarray, n: int) -> np.ndarray:
    n = max(2, min(int(n), len(values)))
    idx = np.linspace(0, len(values) - 1, n).round().astype(int)
    return values[idx]


def _transform_point(x: float, y: float, p: DesignParams) -> tuple[float, float]:
    x -= CANVAS / 2
    y -= CANVAS / 2
    x *= p.scale
    y *= p.scale
    a = math.radians(p.rotation_deg)
    xr = x * math.cos(a) - y * math.sin(a)
    yr = x * math.sin(a) + y * math.cos(a)
    return xr + CANVAS / 2, yr + CANVAS / 2


def _alpha_hex(color: str, opacity: float) -> str:
    color = color.lstrip("#")
    if len(color) == 8:
        color = color[:6]
    return f"#{color}{round(max(0,min(1,opacity))*255):02x}"


def _line(points: Iterable[tuple[float, float]], color: str, width: float, opacity: float = 1.0, closed: bool = False) -> dict[str, Any]:
    pts = list(points)
    if closed and pts:
        pts.append(pts[0])
    return {"kind": "line", "points": pts, "color": color, "width": width, "opacity": opacity}


def _circle(x: float, y: float, r: float, color: str, opacity: float = 1.0, fill: bool = True, width: float = 1.0) -> dict[str, Any]:
    return {"kind": "circle", "x": x, "y": y, "r": r, "color": color, "opacity": opacity, "fill": fill, "width": width}


def _rect(x0: float, y0: float, x1: float, y1: float, color: str, opacity: float = 1.0, fill: bool = True, width: float = 1.0) -> dict[str, Any]:
    return {"kind": "rect", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "color": color, "opacity": opacity, "fill": fill, "width": width}


def _palette(p: DesignParams) -> tuple[str, str, str, str]:
    colors = PALETTES[p.palette]
    if p.background == "White":
        # darken bright print colours for useful white-shirt contrast
        def darken(c: str, f: float = 0.58) -> str:
            rgb = [int(c[i:i+2], 16) for i in (1, 3, 5)]
            return "#" + "".join(f"{max(0,min(255,round(v*f))):02x}" for v in rgb)
        return tuple(darken(c) for c in colors)  # type: ignore[return-value]
    return colors


def _tension_bloom(d: StudioData, p: DesignParams) -> list[dict[str, Any]]:
    colors = _palette(p); seq = _sample(d.norm, min(p.detail, 130)); n = len(seq)
    pts = []
    for k, v in enumerate(seq):
        a = 2*math.pi*k/n
        r = 120 + 330*(k/max(1,n-1)) + 95*v
        pts.append((500+math.cos(a)*r, 500+math.sin(a)*r, float(v)))
    out: list[dict[str, Any]] = []
    step = max(2, n // 26)
    for i in range(n):
        for j in range(i+step, n, step):
            if abs(pts[i][2]-pts[j][2]) < 0.045 + p.jitter*0.12:
                out.append(_line([pts[i][:2], pts[j][:2]], colors[(i+j)%3], p.stroke_width*0.45, 0.18 + p.opacity*0.28))
    for i,(x,y,v) in enumerate(pts):
        out.append(_circle(x,y,1.5+abs(v)*8,colors[(i+1)%3],0.65+p.opacity*0.25,True))
    return out


def _orbit_armour(d: StudioData, p: DesignParams) -> list[dict[str, Any]]:
    colors = _palette(p); out=[]; n = max(72, p.detail)
    base = _sample(d.norm, n)
    for ring in range(18):
        pts=[]; r0=70+ring*21
        teeth=3+(ring%6)+p.symmetry
        for k,v in enumerate(base):
            a=2*math.pi*k/n
            r=r0 + math.sin(a*teeth + ring*.43 + p.seed*.0001)*(6+55*abs(v)) + 28*v
            pts.append((500+math.cos(a)*r,500+math.sin(a)*r))
        out.append(_line(pts, colors[ring%3], max(.35,p.stroke_width*(1-ring*.028)), .28+.035*ring, True))
    return out


def _flow_fault(d: StudioData, p: DesignParams) -> list[dict[str, Any]]:
    colors=_palette(p); vals=_sample(d.norm,60); out=[]
    rows=max(26,min(58,p.detail//3))
    for row in range(rows):
        pts=[]; y=85+row*(830/max(1,rows-1)); fault=360+row*5.2
        for k in range(130):
            x=55+k*(890/129); v=float(vals[min(len(vals)-1,int(k/129*(len(vals)-1)))])
            bend=math.exp(-((x-fault)/100)**2)*150*v*(1+p.jitter*2.5)
            wave=math.sin(k*.21+row*.47+p.seed*.003)*abs(v)*18*p.jitter
            pts.append((x,y+bend+wave))
        out.append(_line(pts, colors[row%3], p.stroke_width*.68, .48+p.opacity*.38))
    return out


def _root_signal(d: StudioData, p: DesignParams) -> list[dict[str, Any]]:
    colors=_palette(p); vals=_sample(d.norm,min(140,p.detail)); out=[]
    rng=random.Random(p.seed)
    for k,v in enumerate(vals):
        t=k/max(1,len(vals)-1); side=-1 if k%2 else 1
        x0=500 + side*(8+12*math.sin(k*.6)); y0=180+t*610
        lean=side*(70+310*abs(float(v)))*(1-.25*t)
        x1=500+lean; y1=220+t*650
        x2=x1+side*(45+110*(.5+float(v)))
        y2=y1+35+55*abs(float(v))
        mid1=(x0+lean*.28, y0+45); mid2=(x1-lean*.22, y1-55)
        # approximate cubic with 18 linear samples
        pts=[]
        for s in np.linspace(0,1,18):
            u=1-s
            x=u**3*x0+3*u*u*s*mid1[0]+3*u*s*s*mid2[0]+s**3*x2
            y=u**3*y0+3*u*u*s*mid1[1]+3*u*s*s*mid2[1]+s**3*y2
            pts.append((x,y))
        out.append(_line(pts, colors[k%3], max(.4,p.stroke_width*(.6+abs(float(v))*2.1)), .45+p.opacity*.42))
        if k%5==0:
            out.append(_circle(x2,y2,2+abs(float(v))*12,colors[(k+1)%3],.82,False,p.stroke_width*.65))
        if rng.random()<.12:
            out.append(_circle(x1,y1,1.5+abs(float(v))*6,colors[3],.55,True))
    return out


def _glyph_constellation(d: StudioData, p: DesignParams) -> list[dict[str, Any]]:
    colors=_palette(p); vals=_sample(d.norm,min(150,p.detail)); out=[]; rng=random.Random(p.seed)
    cols=11; cell=72; rows=min(12,math.ceil(len(vals)/cols)); x0=500-(cols-1)*cell/2; y0=500-(rows-1)*cell/2
    for k,v in enumerate(vals[:rows*cols]):
        cx=x0+(k%cols)*cell+(rng.random()-.5)*18*p.jitter
        cy=y0+(k//cols)*cell+(rng.random()-.5)*12*p.jitter
        v2=float(vals[(k+1)%len(vals)]); v3=float(vals[(k+2)%len(vals)])
        arms=3+int((v2+.45)/.9*5); rad=9+30*abs(float(v3))/.45
        rot=float(v)*math.pi + math.radians(p.rotation_deg)
        for aidx in range(max(3,arms)):
            a=rot+2*math.pi*aidx/max(3,arms)
            bend=(v2)*18
            p1=(cx,cy); p2=(cx+math.cos(a+.7)*bend,cy+math.sin(a+.7)*bend); p3=(cx+math.cos(a)*rad,cy+math.sin(a)*rad)
            out.append(_line([p1,p2,p3],colors[(k+aidx)%3],p.stroke_width,.52+p.opacity*.4))
        if abs(float(v))>.18:
            out.append(_circle(cx,cy,2+8*abs(float(v)),colors[3],.7,False,.8))
    return out


def _spoke_loom(d: StudioData, p: DesignParams) -> list[dict[str, Any]]:
    colors=_palette(p); out=[]; rev=d.rev_norm
    left,top,span=70,70,860; step=span/(d.spoke_count-1)
    # warp
    for s in range(d.spoke_count):
        vals=rev[:,s]; wob=float(np.mean(vals))*115
        pts=[]
        for r in range(rev.shape[0]):
            y=top+r*(span/(rev.shape[0]-1)); x=left+s*step+wob*math.sin((r+1)*.8)
            pts.append((x,y))
        out.append(_line(pts,colors[s%3],p.stroke_width*.55,.28+p.opacity*.32))
    # weft from each revolution, replicated for textile density
    copies=max(5,min(18,p.detail//12))
    for r in range(rev.shape[0]):
        base_y=top+r*(span/(rev.shape[0]-1))
        for c in range(copies):
            pts=[]
            offset=(c-(copies-1)/2)*6.5
            for s,v in enumerate(rev[r]):
                x=left+s*step; y=base_y+offset+float(v)*130
                pts.append((x,y))
            out.append(_line(pts,colors[(r+c)%3],p.stroke_width*.75,.35+p.opacity*.38))
    return out


def _resonant_dust(d: StudioData, p: DesignParams) -> list[dict[str, Any]]:
    colors=_palette(p); out=[]; rng=random.Random(p.seed); vals=_sample(d.norm,max(60,min(180,p.detail)))
    count=max(850,min(3200,p.detail*14))
    for k in range(count):
        a=rng.random()*math.pi*2; r=440*math.sqrt(rng.random()); x=500+math.cos(a)*r; y=500+math.sin(a)*r
        v=float(vals[k%len(vals)]); node=math.sin((3+p.symmetry*.25+v*7)*a)*math.cos((2+abs(v)*9)*r/85)
        if abs(node)<.12+p.jitter*.2:
            out.append(_circle(x,y,.55+2.5*abs(v),colors[k%3],.35+p.opacity*.55,True))
    out.append(_circle(500,500,445,colors[3],.5,False,.8))
    return out


def _helix_groove(d: StudioData, p: DesignParams) -> list[dict[str, Any]]:
    colors=_palette(p); seq=_sample(d.norm,max(100,min(240,p.detail))); out=[]; pts=[]
    for k,v in enumerate(seq):
        t=k/max(1,len(seq)-1); a=.35+t*math.pi*13.5
        r=18+t*435 + float(v)*95*(.5+p.jitter)
        pts.append((500+math.cos(a)*r,500+math.sin(a)*r))
    out.append(_line(pts,colors[0],p.stroke_width,p.opacity))
    # echo groove
    for e in range(1,4):
        pts2=[]
        for k,(x,y) in enumerate(pts):
            a=.35+(k/max(1,len(pts)-1))*math.pi*13.5
            pts2.append((x+math.cos(a)*e*3.2,y+math.sin(a)*e*3.2))
        out.append(_line(pts2,colors[e%3],max(.35,p.stroke_width*.42),.35+p.opacity*.2))
    for k,v in enumerate(seq[::max(1,len(seq)//85)]):
        t=k/max(1,len(seq[::max(1,len(seq)//85)])-1); a=.35+t*math.pi*13.5; r=18+t*435+float(v)*95
        out.append(_circle(500+math.cos(a)*r,500+math.sin(a)*r,1+4*abs(float(v)),colors[(k+1)%3],.75,True))
    return out


def _angular_choir(d: StudioData, p: DesignParams) -> list[dict[str, Any]]:
    colors=_palette(p); out=[]; vals=_sample(d.norm,max(72,min(180,p.detail)))
    R=430
    out.append(_circle(500,500,R,colors[3],.5,False,.8))
    voices=max(4,p.symmetry)
    for voice in range(voices):
        pts=[]
        for k,v in enumerate(vals):
            t=k/max(1,len(vals)-1)
            a=(voice/voices)*2*math.pi + t*math.pi*1.8 + float(v)*1.25
            r=R*(.08+.9*(t**.72))
            pts.append((500+math.cos(a)*r,500+math.sin(a)*r))
        out.append(_line(pts,colors[voice%3],p.stroke_width,.42+p.opacity*.44))
    return out


def _fivefold_moire(d: StudioData, p: DesignParams) -> list[dict[str, Any]]:
    colors=_palette(p); out=[]; rev=d.rev_norm
    samples=120
    theta=np.linspace(0,2*math.pi,samples,endpoint=False)
    for r in range(rev.shape[0]):
        src=np.interp(np.linspace(0,d.spoke_count-1,samples),np.arange(d.spoke_count),rev[r])
        for band in range(max(6,min(18,p.detail//12))):
            base=90+r*52+band*8.5
            pts=[]
            phase=r*.11+band*.018
            for a,v in zip(theta,src):
                rr=base + 44*float(v)*math.sin((p.symmetry+r)*a+phase)
                pts.append((500+math.cos(a+phase)*rr,500+math.sin(a+phase)*rr))
            out.append(_line(pts,colors[(r+band)%3],max(.35,p.stroke_width*.45),.18+p.opacity*.26,True))
    return out


_RENDERERS = {
    "Tension Bloom": _tension_bloom,
    "Orbit Armour": _orbit_armour,
    "Flow Fault": _flow_fault,
    "Root Signal": _root_signal,
    "Glyph Constellation": _glyph_constellation,
    "Spoke Loom": _spoke_loom,
    "Resonant Dust": _resonant_dust,
    "Helix Groove": _helix_groove,
    "Angular Choir": _angular_choir,
    "Fivefold Moire": _fivefold_moire,
}


def build_primitives(data: StudioData, params: DesignParams) -> list[dict[str, Any]]:
    p = DesignParams(**asdict(params)).validated()
    primitives = _RENDERERS[p.family](data,p)
    # Apply global transform only to geometric coordinates, leaving local radii/widths scaled naturally enough.
    transformed=[]
    for item in primitives:
        q=dict(item)
        if q["kind"]=="line":
            q["points"]=[_transform_point(x,y,p) for x,y in q["points"]]
            q["width"]=q["width"]*max(.55,p.scale)
        elif q["kind"]=="circle":
            q["x"],q["y"]=_transform_point(q["x"],q["y"],p); q["r"]*=p.scale; q["width"]*=p.scale
        elif q["kind"]=="rect":
            # use rotated opposite corners; sufficient for studio preview/export because rects are rare
            q["x0"],q["y0"]=_transform_point(q["x0"],q["y0"],p); q["x1"],q["y1"]=_transform_point(q["x1"],q["y1"],p)
        transformed.append(q)
    return transformed


def _rgba(color: str, opacity: float) -> tuple[int,int,int,int]:
    color=color.lstrip("#")[:6]; rgb=tuple(int(color[i:i+2],16) for i in (0,2,4))
    return rgb+(round(max(0,min(1,opacity))*255),)


def render_image(data: StudioData, params: DesignParams, size: int = 1200, transparent: bool | None = None) -> Image.Image:
    p=DesignParams(**asdict(params)).validated(); transparent = (p.background=="Transparent") if transparent is None else transparent
    bg=(0,0,0,0) if transparent else _rgba(BACKGROUND_COLORS[p.background],1.0)
    img=Image.new("RGBA",(size,size),bg); draw=ImageDraw.Draw(img,"RGBA"); s=size/CANVAS
    for q in build_primitives(data,p):
        if q["kind"]=="line":
            pts=[(x*s,y*s) for x,y in q["points"]]
            if len(pts)>=2: draw.line(pts,fill=_rgba(q["color"],q["opacity"]),width=max(1,round(q["width"]*s)),joint="curve")
        elif q["kind"]=="circle":
            x,y,r=q["x"]*s,q["y"]*s,q["r"]*s; box=(x-r,y-r,x+r,y+r); col=_rgba(q["color"],q["opacity"])
            if q["fill"]: draw.ellipse(box,fill=col)
            else: draw.ellipse(box,outline=col,width=max(1,round(q["width"]*s)))
        elif q["kind"]=="rect":
            box=(q["x0"]*s,q["y0"]*s,q["x1"]*s,q["y1"]*s); col=_rgba(q["color"],q["opacity"])
            if q["fill"]: draw.rectangle(box,fill=col)
            else: draw.rectangle(box,outline=col,width=max(1,round(q["width"]*s)))
    return img


def svg_text(data: StudioData, params: DesignParams, size: int = 2400, transparent: bool | None = None) -> str:
    p=DesignParams(**asdict(params)).validated(); transparent=(p.background=="Transparent") if transparent is None else transparent
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 1000 1000">']
    if not transparent: parts.append(f'<rect width="1000" height="1000" fill="{BACKGROUND_COLORS[p.background]}"/>')
    for q in build_primitives(data,p):
        op=max(0,min(1,q["opacity"]))
        if q["kind"]=="line":
            pts=" ".join(f"{x:.2f},{y:.2f}" for x,y in q["points"])
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{q["color"]}" stroke-opacity="{op:.3f}" stroke-width="{q["width"]:.2f}" stroke-linecap="round" stroke-linejoin="round"/>')
        elif q["kind"]=="circle":
            fill=q["color"] if q["fill"] else "none"; stroke="none" if q["fill"] else q["color"]
            parts.append(f'<circle cx="{q["x"]:.2f}" cy="{q["y"]:.2f}" r="{q["r"]:.2f}" fill="{fill}" stroke="{stroke}" opacity="{op:.3f}" stroke-width="{q["width"]:.2f}"/>')
        elif q["kind"]=="rect":
            fill=q["color"] if q["fill"] else "none"; stroke="none" if q["fill"] else q["color"]
            parts.append(f'<rect x="{q["x0"]:.2f}" y="{q["y0"]:.2f}" width="{q["x1"]-q["x0"]:.2f}" height="{q["y1"]-q["y0"]:.2f}" fill="{fill}" stroke="{stroke}" opacity="{op:.3f}" stroke-width="{q["width"]:.2f}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def export_png(path: Path, data: StudioData, params: DesignParams, size: int = 2400, transparent: bool | None = None) -> Path:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); render_image(data,params,size,transparent).save(path); return path


def export_svg(path: Path, data: StudioData, params: DesignParams, size: int = 2400, transparent: bool | None = None) -> Path:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(svg_text(data,params,size,transparent),encoding="utf-8"); return path


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+","_",value.strip()).strip("_") or "design"


def save_design(output_dir: Path, data: StudioData, params: DesignParams, name: str) -> dict[str, Path]:
    root=Path(output_dir)/"design_studio"/"saved"; root.mkdir(parents=True,exist_ok=True)
    stem=safe_name(name); json_path=root/f"{stem}.json"; png=root/f"{stem}.png"; svg=root/f"{stem}.svg"
    payload={"name":name,"participant_id":data.participant_id,"trial_id":data.trial_id,"locked_intervals":int(data.intervals.size),"spoke_count":data.spoke_count,"equivalent_revolutions":float(data.intervals.size/data.spoke_count),"params":asdict(params)}
    json_path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    export_png(png,data,params,1800,transparent=(params.background=="Transparent")); export_svg(svg,data,params,2400,transparent=(params.background=="Transparent"))
    return {"json":json_path,"png":png,"svg":svg}


def load_saved_params(path: Path) -> DesignParams:
    payload=json.loads(Path(path).read_text(encoding="utf-8")); return DesignParams(**payload["params"]).validated()


def list_saved(output_dir: Path) -> list[Path]:
    root=Path(output_dir)/"design_studio"/"saved"
    return sorted(root.glob("*.json")) if root.exists() else []


def render_shirt_mockup(data: StudioData, params: DesignParams, size: int = 1000, shirt: str = "Black") -> Image.Image:
    scene=Image.new("RGBA",(size,size),(231,228,220,255)); draw=ImageDraw.Draw(scene,"RGBA")
    shirt_col=_rgba(BACKGROUND_COLORS[shirt],1.0)
    # deliberately simple product-view silhouette, not a photorealistic mockup
    pts=[(.33,.16),(.42,.115),(.5,.16),(.58,.115),(.67,.16),(.845,.30),(.735,.41),(.68,.35),(.68,.87),(.32,.87),(.32,.35),(.265,.41),(.155,.30)]
    poly=[(int(x*size),int(y*size)) for x,y in pts]; draw.polygon(poly,fill=shirt_col,outline=(150,150,150,120))
    art_params=DesignParams(**asdict(params)); art_params.background="Transparent"
    art=render_image(data,art_params,int(size*.38),transparent=True)
    x=int(size*.31); y=int(size*.29); scene.alpha_composite(art,(x,y))
    return scene


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Export one generative design from a completed study output.")
    parser.add_argument("output_dir",type=Path); parser.add_argument("--family",choices=tuple(FAMILIES),default="Tension Bloom"); parser.add_argument("--out",type=Path,default=Path("design_studio_preview.png"))
    args=parser.parse_args(); data=load_studio_data(args.output_dir); params=DesignParams(family=args.family); export_png(args.out,data,params)
    print(args.out)
