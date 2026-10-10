#!/usr/bin/env python3
"""End-to-end study analysis for spoke-click movement recordings.

This wrapper freezes the detector settings supplied by the calibration profile,
runs click detection, identifies outward/turnaround/return phases, and adds a
standardised equal-distance comparison based on the same number of spoke-click
intervals from the centre of the outward and return legs.

Spoke timing is treated as a wheel-motion / forward-movement proxy, not as a
direct heel-strike or toe-off measure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import session_analyzer as detector
import session_movement_analysis as movement

STUDY_SUMMARY_FILENAME = "study_summary.json"
DEFAULT_COMPARISON_INTERVALS = 180  # 5 wheel revolutions for a 36-spoke wheel


def _interval_metrics(values: np.ndarray, spokes: int) -> dict[str, Any]:
    if values.size == 0:
        return {"interval_count": 0}
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    robust_cv = None if med <= 0 else float(1.4826 * mad / med)
    rate = None if med <= 0 else float(1.0 / med)
    return {
        "interval_count": int(values.size),
        "median_interval_s": med,
        "interval_mad_s": mad,
        "robust_interval_cv": robust_cv,
        "median_click_rate_hz": rate,
        "median_rotation_rate_hz": None if rate is None else float(rate / spokes),
        "median_rotation_period_s": None if rate is None else float(spokes / rate),
    }


def _central_block(values: np.ndarray, n: int) -> np.ndarray:
    if values.size < n:
        return np.asarray([], dtype=float)
    start = (values.size - n) // 2
    return values[start:start + n]


def standardised_comparison(
    times: np.ndarray,
    movement_summary: dict[str, Any],
    spokes: int,
    comparison_intervals: int = DEFAULT_COMPARISON_INTERVALS,
) -> dict[str, Any]:
    phases = movement_summary.get("phases") or []
    outward = next((p for p in phases if p.get("phase") == "outward"), None)
    ret = next((p for p in phases if p.get("phase") == "return"), None)
    if not outward or not ret:
        return {
            "available": False,
            "reason": "No confident outward/return split was available.",
            "requested_intervals_per_leg": int(comparison_intervals),
        }

    intervals = np.diff(times)
    mid = (times[:-1] + times[1:]) / 2.0
    out_vals = intervals[(mid >= float(outward["start_s"])) & (mid <= float(outward["end_s"]))]
    ret_vals = intervals[(mid >= float(ret["start_s"])) & (mid <= float(ret["end_s"]))]

    n = int(comparison_intervals)
    if n <= 0:
        raise ValueError("comparison_intervals must be positive")
    if out_vals.size < n or ret_vals.size < n:
        return {
            "available": False,
            "reason": "One or both walking legs contain fewer intervals than the fixed comparison window.",
            "requested_intervals_per_leg": n,
            "outward_available_intervals": int(out_vals.size),
            "return_available_intervals": int(ret_vals.size),
        }

    out_center = _central_block(out_vals, n)
    ret_center = _central_block(ret_vals, n)
    out_metrics = _interval_metrics(out_center, spokes)
    ret_metrics = _interval_metrics(ret_center, spokes)

    out_rate = out_metrics.get("median_click_rate_hz")
    ret_rate = ret_metrics.get("median_click_rate_hz")
    rate_change = None
    if out_rate not in (None, 0.0) and ret_rate is not None:
        rate_change = float(100.0 * (ret_rate - out_rate) / out_rate)

    return {
        "available": True,
        "method": "fixed central spoke-click interval count",
        "requested_intervals_per_leg": n,
        "equivalent_wheel_revolutions_per_leg": float(n / spokes),
        "rationale": "The same number of central spoke intervals is compared on both legs, reducing start, stop and turnaround effects while approximating equal wheel travel.",
        "outward": out_metrics,
        "return": ret_metrics,
        "return_vs_outward_click_rate_change_percent": rate_change,
        "interpretation": "movement-rhythm / wheel-motion comparison; not a direct gait-event measure",
    }


def analyse_study_session(
    wav_path: Path,
    output_dir: Path,
    profile_path: Path,
    spokes: int = 36,
    comparison_intervals: int = DEFAULT_COMPARISON_INTERVALS,
    manual_turnaround_s: float | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_dir = output_dir / "detection"
    movement_dir = output_dir / "movement"

    result, csv_path, review_path, session_summary_path = detector.analyse_session(
        wav_path, detection_dir, profile_path
    )

    movement_summary = movement.analyse(
        csv_path, movement_dir, spokes=spokes, manual_turnaround_s=manual_turnaround_s
    )
    times = movement.load_detection_times(csv_path)
    comparison = standardised_comparison(
        times, movement_summary, spokes, comparison_intervals
    )

    study_summary = {
        "schema_version": 1,
        "source_wav": str(wav_path.resolve()),
        "calibration_profile": str(profile_path.resolve()),
        "detector": {
            "detected_hits": int(result.detected_hits),
            "duration_s": float(result.duration_s),
            "quality_status": result.quality_status,
            "effective_prominence_threshold": float(result.effective_prominence_threshold),
            "threshold_source": result.threshold_source,
        },
        "phase_split": movement_summary.get("phase_split"),
        "phases": movement_summary.get("phases"),
        "standardised_primary_comparison": comparison,
        "return_adaptation": movement_summary.get("return_adaptation"),
        "study_rule": {
            "primary_window": f"central {int(comparison_intervals)} inter-click intervals of each walking leg",
            "turnaround_excluded": True,
            "spoke_count": int(spokes),
            "movement_proxy_note": "Spoke timing reflects wheel motion / forward movement rhythm, not direct heel-strike or toe-off timing."
        },
        "outputs": {
            "detections_csv": str(csv_path.resolve()),
            "session_summary_json": str(session_summary_path.resolve()),
            "reference_review_wav": str(review_path.resolve()),
            "isolated_review_wav": str((detection_dir / detector.ISOLATED_FILENAME).resolve()),
            "movement_summary_json": str((movement_dir / movement.SUMMARY_FILENAME).resolve()),
            "movement_timeline_csv": str((movement_dir / movement.TIMELINE_FILENAME).resolve()),
            "movement_plot_svg": str((movement_dir / movement.PLOT_FILENAME).resolve()),
        },
    }
    summary_path = output_dir / STUDY_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(study_summary, indent=2), encoding="utf-8")
    return study_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end study session analysis")
    ap.add_argument("wav", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--profile", type=Path, default=Path("calibration/calibration_profile.json"))
    ap.add_argument("--spokes", type=int, default=36)
    ap.add_argument("--comparison-intervals", type=int, default=DEFAULT_COMPARISON_INTERVALS)
    ap.add_argument("--turnaround-seconds", type=float, default=None)
    args = ap.parse_args()
    result = analyse_study_session(
        args.wav,
        args.output_dir,
        args.profile,
        spokes=args.spokes,
        comparison_intervals=args.comparison_intervals,
        manual_turnaround_s=args.turnaround_seconds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
