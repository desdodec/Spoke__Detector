#!/usr/bin/env python3
"""Movement-rhythm analysis from accepted spoke detections.

This module intentionally treats spoke clicks as a wheel-motion / forward-movement
proxy, not as direct heel-strike or toe-off events.

Inputs
------
A session_detections.csv produced by session_analyzer.py / session_analyzer_v2.py.

Outputs
-------
- movement_timeline.csv   interval-by-interval movement proxy
- movement_summary.json  robust whole-session and phase metrics
- movement_plot.svg      lightweight plot with no extra plotting dependency

The automatic phase split is conservative.  It proposes a turnaround candidate
from a conspicuously long inter-click gap only when that gap is sufficiently
separated from the ordinary interval distribution.  A manual turnaround time can
be supplied for study records where the endpoint is known from protocol notes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np

DEFAULT_SPOKES = 36
TIMELINE_FILENAME = "movement_timeline.csv"
SUMMARY_FILENAME = "movement_summary.json"
PLOT_FILENAME = "movement_plot.svg"


@dataclass
class Phase:
    name: str
    start_s: float
    end_s: float


def _mad(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    m = float(np.median(values))
    return float(np.median(np.abs(values - m)))


def _robust_cv(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    m = float(np.median(values))
    if m <= 0:
        return None
    mad = _mad(values)
    return None if mad is None else float(1.4826 * mad / m)


def load_detection_times(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    times = np.asarray([float(r["timestamp_seconds"]) for r in rows], dtype=float)
    if times.size < 3:
        raise ValueError("At least three accepted detections are required for movement analysis.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Detection timestamps must be strictly increasing.")
    return times


def centered_running_median(values: np.ndarray, width: int = 7) -> np.ndarray:
    width = max(3, int(width) | 1)
    half = width // 2
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        a = max(0, i - half)
        b = min(len(values), i + half + 1)
        out[i] = float(np.median(values[a:b]))
    return out


def infer_turnaround(times: np.ndarray) -> dict:
    intervals = np.diff(times)
    med = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - med)))
    robust_sigma = max(1e-9, 1.4826 * mad)
    idx = int(np.argmax(intervals))
    gap = float(intervals[idx])
    z = float((gap - med) / robust_sigma)
    ratio = float(gap / max(med, 1e-9))

    # Require both an absolute/relative slowdown and robust separation.  This is
    # deliberately conservative: if confidence is weak, report no automatic split.
    accepted = bool(gap >= max(0.55, med * 2.2) and z >= 4.0)
    midpoint = float((times[idx] + times[idx + 1]) / 2.0)
    return {
        "accepted": accepted,
        "gap_index": idx,
        "gap_start_s": float(times[idx]),
        "gap_end_s": float(times[idx + 1]),
        "gap_s": gap,
        "median_interval_s": med,
        "gap_to_median_ratio": ratio,
        "robust_z": z,
        "turnaround_midpoint_s": midpoint,
        "method": "largest inter-click gap with conservative robust separation",
    }


def make_phases(times: np.ndarray, manual_turnaround_s: float | None = None) -> tuple[list[Phase], dict]:
    auto = infer_turnaround(times)
    if manual_turnaround_s is not None:
        t = float(manual_turnaround_s)
        # Use half a local median interval around the known endpoint as the
        # turnaround band.  This avoids pretending the exact manoeuvre duration
        # can be inferred from clicks alone.
        med = float(np.median(np.diff(times)))
        half = max(0.25, med)
        lo, hi = max(float(times[0]), t - half), min(float(times[-1]), t + half)
        source = "manual_protocol_time"
    elif auto["accepted"]:
        lo = float(auto["gap_start_s"])
        hi = float(auto["gap_end_s"])
        source = "automatic_gap_candidate"
    else:
        return [Phase("whole_session", float(times[0]), float(times[-1]))], {
            "source": "none",
            "automatic_candidate": auto,
            "note": "No confident turnaround split; whole-session metrics only.",
        }

    phases = [
        Phase("outward", float(times[0]), lo),
        Phase("turnaround", lo, hi),
        Phase("return", hi, float(times[-1])),
    ]
    return phases, {"source": source, "automatic_candidate": auto, "turnaround_start_s": lo, "turnaround_end_s": hi}


def phase_metrics(interval_midpoints: np.ndarray, intervals: np.ndarray, phase: Phase, spokes: int) -> dict:
    mask = (interval_midpoints >= phase.start_s) & (interval_midpoints <= phase.end_s)
    vals = intervals[mask]
    if vals.size == 0:
        return {
            "phase": phase.name,
            "start_s": phase.start_s,
            "end_s": phase.end_s,
            "interval_count": 0,
        }
    med_i = float(np.median(vals))
    click_rate = 1.0 / med_i if med_i > 0 else None
    return {
        "phase": phase.name,
        "start_s": phase.start_s,
        "end_s": phase.end_s,
        "interval_count": int(vals.size),
        "median_interval_s": med_i,
        "interval_mad_s": _mad(vals),
        "robust_interval_cv": _robust_cv(vals),
        "median_click_rate_hz": click_rate,
        "median_rotation_rate_hz": None if click_rate is None else float(click_rate / spokes),
        "median_rotation_period_s": None if click_rate is None else float(spokes / click_rate),
    }


def adaptation_metrics(interval_midpoints: np.ndarray, intervals: np.ndarray, phase: Phase) -> dict | None:
    mask = (interval_midpoints >= phase.start_s) & (interval_midpoints <= phase.end_s)
    vals = intervals[mask]
    if vals.size < 10:
        return None
    n = max(3, int(math.ceil(vals.size * 0.20)))
    early = vals[:n]
    late = vals[-n:]
    early_rate = float(1.0 / np.median(early))
    late_rate = float(1.0 / np.median(late))
    change = 100.0 * (late_rate - early_rate) / early_rate if early_rate > 0 else None
    return {
        "early_interval_count": int(n),
        "late_interval_count": int(n),
        "early_median_click_rate_hz": early_rate,
        "late_median_click_rate_hz": late_rate,
        "click_rate_change_percent": change,
        "interpretation": "movement-rhythm adaptation proxy, not a direct gait-event measure",
    }


def write_svg(path: Path, x: np.ndarray, y: np.ndarray, phases: list[Phase]) -> None:
    width, height = 1000, 360
    left, right, top, bottom = 70, 25, 30, 55
    pw, ph = width - left - right, height - top - bottom
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(np.quantile(y, 0.02)), float(np.quantile(y, 0.98))
    if ymax <= ymin:
        ymax = ymin + 1.0

    def sx(v: float) -> float:
        return left + (v - xmin) / max(xmax - xmin, 1e-9) * pw

    def sy(v: float) -> float:
        vv = min(max(v, ymin), ymax)
        return top + (ymax - vv) / (ymax - ymin) * ph

    pts = " ".join(f"{sx(float(a)):.1f},{sy(float(b)):.1f}" for a, b in zip(x, y))
    phase_lines = []
    for p in phases:
        if p.name == "whole_session":
            continue
        phase_lines.append(f'<line x1="{sx(p.start_s):.1f}" y1="{top}" x2="{sx(p.start_s):.1f}" y2="{top+ph}" stroke="gray" stroke-dasharray="5,5"/>')
        phase_lines.append(f'<text x="{sx((p.start_s+p.end_s)/2):.1f}" y="18" text-anchor="middle" font-size="12">{p.name}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="18" text-anchor="middle" font-size="15">Movement rhythm from spoke-click timing</text>
<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="black"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="black"/>
<polyline fill="none" stroke="black" stroke-width="1.5" points="{pts}"/>
{''.join(phase_lines)}
<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="12">Time (s)</text>
<text x="18" y="{height/2}" text-anchor="middle" font-size="12" transform="rotate(-90 18 {height/2})">Local click rate (Hz)</text>
<text x="{left}" y="{height-32}" font-size="10">Lower click rate = slower wheel motion; spoke clicks are a movement proxy, not heel-strike/toe-off events.</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def analyse(detections_csv: Path, output_dir: Path, spokes: int = DEFAULT_SPOKES, manual_turnaround_s: float | None = None) -> dict:
    times = load_detection_times(detections_csv)
    intervals = np.diff(times)
    mid = (times[:-1] + times[1:]) / 2.0
    instantaneous_rate = 1.0 / intervals
    local_rate = centered_running_median(instantaneous_rate, 7)
    baseline = float(np.median(local_rate))
    relative_rate = local_rate / max(baseline, 1e-12)
    local_intervals = 1.0 / local_rate

    phases, split_info = make_phases(times, manual_turnaround_s)
    metrics = [phase_metrics(mid, intervals, p, spokes) for p in phases]
    return_phase = next((p for p in phases if p.name == "return"), None)
    adaptation = None if return_phase is None else adaptation_metrics(mid, intervals, return_phase)

    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = output_dir / TIMELINE_FILENAME
    with timeline_path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["interval_index", "midpoint_seconds", "interval_seconds", "instantaneous_click_rate_hz", "local_median_click_rate_hz", "relative_movement_rate", "local_rotation_rate_hz"])
        for i, (m, dt, inst, loc, rel) in enumerate(zip(mid, intervals, instantaneous_rate, local_rate, relative_rate), 1):
            w.writerow([i, f"{m:.6f}", f"{dt:.6f}", f"{inst:.6f}", f"{loc:.6f}", f"{rel:.6f}", f"{loc/spokes:.8f}"])

    whole = phase_metrics(mid, intervals, Phase("whole_session", float(times[0]), float(times[-1])), spokes)
    summary = {
        "schema_version": 1,
        "source_detections_csv": str(detections_csv),
        "spoke_count": int(spokes),
        "detected_hits": int(times.size),
        "first_detection_s": float(times[0]),
        "last_detection_s": float(times[-1]),
        "movement_proxy_definition": "spoke click rate / timing reflects wheel motion and forward movement rhythm; it is not a direct heel-strike or toe-off measure",
        "whole_session": whole,
        "phase_split": split_info,
        "phases": metrics,
        "return_adaptation": adaptation,
        "outputs": {"timeline_csv": str(timeline_path), "plot_svg": str(output_dir / PLOT_FILENAME)},
    }
    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_svg(output_dir / PLOT_FILENAME, mid, local_rate, phases)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse movement rhythm from spoke detections")
    ap.add_argument("detections_csv", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--spokes", type=int, default=DEFAULT_SPOKES)
    ap.add_argument("--turnaround-seconds", type=float, default=None, help="Optional protocol-known turnaround midpoint")
    args = ap.parse_args()
    summary = analyse(args.detections_csv, args.output_dir, args.spokes, args.turnaround_seconds)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
