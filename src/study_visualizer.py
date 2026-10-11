#!/usr/bin/env python3
"""Standardised study-data visualisation and print-ready SVG export.

The visualiser never changes the scientific comparison window.  Every style is
built from the same locked central interval block written by the study workflow.
Visual choices alter presentation only; they do not alter subject metrics.
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


STYLES = ("Radial fingerprint", "Spiral journey", "Mirrored signature")
DENSITIES = ("Low", "Medium", "High")
SMOOTHING = ("Low", "Medium", "High")
THEMES = ("Light garment", "Dark garment")


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
        raise ValueError(
            f"The selected leg has only {values.size} intervals; the locked study window requires {n}."
        )
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
    dev = np.clip(dev, -0.32, 0.32)
    return _smooth(dev, smoothing)


def _density_offsets(level: str) -> list[float]:
    if level == "Low":
        return [0.0]
    if level == "High":
        return [0.0, 0.035, 0.070, 0.105, 0.140]
    return [0.0, 0.050, 0.100, 0.150]


def _palette(theme: str) -> dict[str, Any]:
    if theme == "Dark garment":
        return {
            "background": "#111111",
            "outward": "#5cc8ff",
            "return": "#ff8a3d",
            "echo_out": "#c17bff",
            "echo_ret": "#ff5d73",
            "text": "#f3f3f3",
            "muted": "#b0b0b0",
            "marker": "#f0d44a",
        }
    return {
        "background": "#ffffff",
        "outward": "#1261a0",
        "return": "#c45500",
        "echo_out": "#6b4ba1",
        "echo_ret": "#8a3d3d",
        "text": "#111111",
        "muted": "#666666",
        "marker": "#7a6500",
    }


def _radial_geometry(data: VisualData, width: int, height: int, density: str, smoothing: str) -> dict[str, Any]:
    out = _normalised_deviation(data.outward_intervals, smoothing)
    ret = _normalised_deviation(data.return_intervals, smoothing)
    cx, cy = width / 2.0, height / 2.0
    base = min(width, height) * 0.31
    amp = min(width, height) * 0.27
    gap = math.radians(18.0)

    # Start at top. Outward moves down the left side to TURN at bottom.
    # Return moves up the right side to END at top.
    t_out = np.linspace(-math.pi / 2.0 - gap, -3.0 * math.pi / 2.0 + gap, len(out))
    t_ret = np.linspace(math.pi / 2.0 + gap, -math.pi / 2.0 - gap, len(ret))

    lines = []
    offsets = _density_offsets(density)
    for i, off in enumerate(offsets):
        ro = base * (1.0 + off) + amp * out
        rr = base * (1.0 + off) + amp * ret
        po = [(cx + float(r) * math.cos(float(t)), cy + float(r) * math.sin(float(t))) for r, t in zip(ro, t_out)]
        pr = [(cx + float(r) * math.cos(float(t)), cy + float(r) * math.sin(float(t))) for r, t in zip(rr, t_ret)]
        lines.append(("outward" if i == 0 else "echo_out", po, 3.0 if i == 0 else 1.1))
        lines.append(("return" if i == 0 else "echo_ret", pr, 3.0 if i == 0 else 1.1))

    labels = [
        ("START", cx - 68, cy - base - 55, "muted", 12),
        ("END", cx + 68, cy - base - 55, "muted", 12),
        ("TURN", cx, cy + base + 60, "marker", 12),
        (f"OUTWARD  {data.outward_position}", cx - base - 80, cy, "outward", 13),
        (f"RETURN  {data.return_position}", cx + base + 80, cy, "return", 13),
    ]
    return {"lines": lines, "labels": labels}


def _spiral_geometry(data: VisualData, width: int, height: int, density: str, smoothing: str) -> dict[str, Any]:
    out = _normalised_deviation(data.outward_intervals, smoothing)
    ret = _normalised_deviation(data.return_intervals, smoothing)
    cx, cy = width / 2.0, height / 2.0
    size = min(width, height)
    offsets = _density_offsets(density)
    lines = []

    # Outward expands; return contracts. A deliberate gap marks the turnaround.
    t_out = np.linspace(-math.pi / 2, 2.5 * math.pi, len(out))
    t_ret = np.linspace(2.72 * math.pi, 5.72 * math.pi, len(ret))
    base_out = np.linspace(size * 0.10, size * 0.35, len(out))
    base_ret = np.linspace(size * 0.35, size * 0.12, len(ret))
    amp = size * 0.16

    for i, off in enumerate(offsets):
        ro = base_out + size * off * 0.10 + amp * out
        rr = base_ret + size * off * 0.10 + amp * ret
        po = [(cx + float(r) * math.cos(float(t)), cy + float(r) * math.sin(float(t))) for r, t in zip(ro, t_out)]
        pr = [(cx + float(r) * math.cos(float(t)), cy + float(r) * math.sin(float(t))) for r, t in zip(rr, t_ret)]
        lines.append(("outward" if i == 0 else "echo_out", po, 3.0 if i == 0 else 1.0))
        lines.append(("return" if i == 0 else "echo_ret", pr, 3.0 if i == 0 else 1.0))

    labels = [
        ("START", cx, cy - size * 0.08, "muted", 12),
        ("TURN", cx + size * 0.36, cy, "marker", 12),
        ("END", cx, cy + size * 0.10, "muted", 12),
    ]
    return {"lines": lines, "labels": labels}


def _mirrored_geometry(data: VisualData, width: int, height: int, density: str, smoothing: str) -> dict[str, Any]:
    out_rate = 1.0 / data.outward_intervals
    ret_rate = 1.0 / data.return_intervals
    out = _smooth((out_rate - np.median(out_rate)) / np.median(out_rate), smoothing)
    ret = _smooth((ret_rate - np.median(ret_rate)) / np.median(ret_rate), smoothing)
    out = np.clip(out, -0.30, 0.30)
    ret = np.clip(ret, -0.30, 0.30)

    left, right = width * 0.10, width * 0.90
    x = np.linspace(left, right, len(out))
    upper, lower = height * 0.34, height * 0.66
    amp = height * 0.32
    offsets = _density_offsets(density)
    lines = []
    for i, off in enumerate(offsets):
        po = [(float(xx), upper - amp * float(v) - off * height * 0.16) for xx, v in zip(x, out)]
        pr = [(float(xx), lower + amp * float(v) + off * height * 0.16) for xx, v in zip(x, ret)]
        lines.append(("outward" if i == 0 else "echo_out", po, 3.0 if i == 0 else 1.0))
        lines.append(("return" if i == 0 else "echo_ret", pr, 3.0 if i == 0 else 1.0))
    labels = [
        (f"OUTWARD  {data.outward_position}", left, height * 0.13, "outward", 13),
        (f"RETURN  {data.return_position}", left, height * 0.87, "return", 13),
        ("START", left, height * 0.50, "muted", 11),
        ("TURN", width * 0.50, height * 0.50, "marker", 11),
        ("END", right, height * 0.50, "muted", 11),
    ]
    return {"lines": lines, "labels": labels}


def build_geometry(
    data: VisualData,
    style: str,
    width: int,
    height: int,
    density: str = "Medium",
    smoothing: str = "Medium",
) -> dict[str, Any]:
    if style == "Spiral journey":
        return _spiral_geometry(data, width, height, density, smoothing)
    if style == "Mirrored signature":
        return _mirrored_geometry(data, width, height, density, smoothing)
    return _radial_geometry(data, width, height, density, smoothing)


def brief_analysis_lines(data: VisualData) -> list[str]:
    rate_change = "n/a" if data.rate_change_pct is None else f"{data.rate_change_pct:+.1f}%"
    out_cv = "n/a" if data.outward_cv is None else f"{100.0 * data.outward_cv:.1f}%"
    ret_cv = "n/a" if data.return_cv is None else f"{100.0 * data.return_cv:.1f}%"
    turn = "n/a" if data.turnaround_s is None else f"{data.turnaround_s:.1f} s"
    adapt = "n/a" if data.adaptation_change_pct is None else f"{data.adaptation_change_pct:+.1f}%"
    return [
        f"OUT {data.outward_rate_hz:.2f} clicks/s   RETURN {data.return_rate_hz:.2f} clicks/s",
        f"RETURN vs OUTWARD {rate_change}",
        f"RHYTHM VARIABILITY {out_cv} / {ret_cv}",
        f"TURN {turn}   RETURN ADAPTATION {adapt}",
    ]


def draw_on_canvas(
    canvas: Any,
    data: VisualData,
    style: str,
    theme: str,
    density: str,
    smoothing: str,
    annotated: bool,
) -> None:
    canvas.update_idletasks()
    width = max(640, int(canvas.winfo_width()))
    height = max(520, int(canvas.winfo_height()))
    palette = _palette(theme)
    canvas.configure(background=palette["background"])
    canvas.delete("all")
    geom = build_geometry(data, style, width, height, density, smoothing)

    for role, pts, line_width in geom["lines"]:
        flat = [coord for p in pts for coord in p]
        if len(flat) >= 4:
            canvas.create_line(*flat, fill=palette[role], width=line_width, smooth=False)

    if annotated:
        for text, x, y, role, size in geom["labels"]:
            canvas.create_text(x, y, text=text, fill=palette[role], font=("TkDefaultFont", size, "bold"))
        canvas.create_text(
            width / 2,
            height * 0.08,
            text=f"{data.participant_id}  /  TRIAL {data.trial_id}",
            fill=palette["text"],
            font=("TkDefaultFont", 15, "bold"),
        )
        for i, line in enumerate(brief_analysis_lines(data)):
            canvas.create_text(
                width / 2,
                height * (0.82 + 0.035 * i),
                text=line,
                fill=palette["text"] if i == 0 else palette["muted"],
                font=("TkDefaultFont", 10),
            )


def export_svg(
    path: Path,
    data: VisualData,
    style: str,
    theme: str = "Light garment",
    density: str = "Medium",
    smoothing: str = "Medium",
    annotated: bool = False,
    width: int = 1200,
    height: int = 1200,
    transparent: bool = True,
) -> None:
    palette = _palette(theme)
    geom = build_geometry(data, style, width, height, density, smoothing)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    if not transparent:
        parts.append(f'<rect width="100%" height="100%" fill="{palette["background"]}"/>')

    for role, pts, line_width in geom["lines"]:
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        parts.append(
            f'<polyline fill="none" stroke="{palette[role]}" stroke-width="{line_width:.2f}" '
            f'stroke-linecap="round" stroke-linejoin="round" points="{points}"/>'
        )

    if annotated:
        for text, x, y, role, size in geom["labels"]:
            parts.append(
                f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" fill="{palette[role]}" '
                f'font-family="sans-serif" font-size="{size * 1.35:.1f}" font-weight="700">{escape(text)}</text>'
            )
        parts.append(
            f'<text x="{width/2:.2f}" y="{height*0.08:.2f}" text-anchor="middle" fill="{palette["text"]}" '
            f'font-family="sans-serif" font-size="24" font-weight="700">{escape(data.participant_id)} / TRIAL {escape(data.trial_id)}</text>'
        )
        for i, line in enumerate(brief_analysis_lines(data)):
            y = height * (0.82 + 0.035 * i)
            parts.append(
                f'<text x="{width/2:.2f}" y="{y:.2f}" text-anchor="middle" fill="{palette["text"] if i == 0 else palette["muted"]}" '
                f'font-family="sans-serif" font-size="16">{escape(line)}</text>'
            )

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def export_all_default_styles(output_dir: Path) -> list[Path]:
    data = load_visual_data(output_dir)
    art_dir = output_dir / "artwork"
    written: list[Path] = []
    for style in STYLES:
        stem = style.lower().replace(" ", "_")
        path = art_dir / f"{stem}.svg"
        export_svg(path, data, style, annotated=False, transparent=True)
        written.append(path)
    annotated = art_dir / "radial_fingerprint_annotated.svg"
    export_svg(annotated, data, "Radial fingerprint", annotated=True, transparent=True)
    written.append(annotated)
    return written
