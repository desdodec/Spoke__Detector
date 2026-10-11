#!/usr/bin/env python3
"""End-to-end standardised study analysis for spoke-click movement recordings.

The study protocol is deliberately separated from the detector. Capture and
analysis settings are locked in study/study_protocol.json so every participant
is processed with the same rules. Spoke timing is treated as a wheel-motion /
forward-movement proxy, not as direct heel-strike or toe-off timing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import session_analyzer as detector
import session_movement_analysis as movement

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = REPO_ROOT / "study" / "study_protocol.json"
STUDY_SUMMARY_FILENAME = "study_summary.json"
SUBJECT_RECORD_FILENAME = "subject_record.csv"
DEFAULT_COMPARISON_INTERVALS = 180


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported study protocol schema.")
    if data.get("status") != "LOCKED_FOR_COMPARABLE_CAPTURE":
        raise ValueError("Study protocol is not locked for comparable capture.")
    analysis = data.get("analysis") or {}
    if int(analysis.get("spoke_count", 0)) <= 0:
        raise ValueError("Study protocol is missing a valid spoke count.")
    if int(analysis.get("primary_comparison_intervals_per_leg", 0)) <= 0:
        raise ValueError("Study protocol is missing a valid comparison window.")
    return data


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
    comparison_intervals: int,
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
    if out_vals.size < n or ret_vals.size < n:
        return {
            "available": False,
            "reason": "One or both walking legs contain fewer intervals than the locked comparison window.",
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


def _validate_subject_metadata(participant_id: str, trial_id: str, outward_position: str, return_position: str) -> None:
    missing = [
        name for name, value in [
            ("participant_id", participant_id),
            ("trial_id", trial_id),
            ("outward_handlebar_position", outward_position),
            ("return_handlebar_position", return_position),
        ] if not str(value).strip()
    ]
    if missing:
        raise ValueError("Missing required subject metadata: " + ", ".join(missing))
    if outward_position.strip().lower() == return_position.strip().lower():
        raise ValueError("Outward and return handlebar positions must differ for the swap protocol.")


def _write_subject_record(path: Path, summary: dict[str, Any]) -> None:
    comp = summary.get("standardised_primary_comparison") or {}
    outward = comp.get("outward") or {}
    ret = comp.get("return") or {}
    adapt = summary.get("return_adaptation") or {}
    row = {
        "protocol_id": summary["protocol"]["protocol_id"],
        "participant_id": summary["subject"]["participant_id"],
        "trial_id": summary["subject"]["trial_id"],
        "outward_handlebar_position": summary["subject"]["outward_handlebar_position"],
        "return_handlebar_position": summary["subject"]["return_handlebar_position"],
        "detected_hits": summary["detector"]["detected_hits"],
        "detector_quality": summary["detector"]["quality_status"],
        "turnaround_source": (summary.get("phase_split") or {}).get("source"),
        "primary_comparison_available": comp.get("available"),
        "comparison_intervals_per_leg": comp.get("requested_intervals_per_leg"),
        "equivalent_wheel_revolutions_per_leg": comp.get("equivalent_wheel_revolutions_per_leg"),
        "outward_median_click_rate_hz": outward.get("median_click_rate_hz"),
        "return_median_click_rate_hz": ret.get("median_click_rate_hz"),
        "outward_robust_interval_cv": outward.get("robust_interval_cv"),
        "return_robust_interval_cv": ret.get("robust_interval_cv"),
        "return_vs_outward_click_rate_change_percent": comp.get("return_vs_outward_click_rate_change_percent"),
        "return_adaptation_change_percent": adapt.get("click_rate_change_percent"),
        "calibration_profile_sha256": summary["provenance"]["calibration_profile_sha256"],
        "study_protocol_sha256": summary["provenance"]["study_protocol_sha256"],
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def analyse_study_session(
    wav_path: Path,
    output_dir: Path,
    profile_path: Path,
    participant_id: str,
    trial_id: str,
    outward_handlebar_position: str,
    return_handlebar_position: str,
    protocol_path: Path = DEFAULT_PROTOCOL,
    manual_turnaround_s: float | None = None,
) -> dict[str, Any]:
    _validate_subject_metadata(participant_id, trial_id, outward_handlebar_position, return_handlebar_position)
    protocol = load_protocol(protocol_path)
    rules = protocol["analysis"]
    spokes = int(rules["spoke_count"])
    comparison_intervals = int(rules["primary_comparison_intervals_per_leg"])

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
    comparison = standardised_comparison(times, movement_summary, spokes, comparison_intervals)

    study_summary = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "status": protocol["status"],
            "path": str(protocol_path.resolve()),
            "locked_spoke_count": spokes,
            "locked_primary_comparison_intervals_per_leg": comparison_intervals,
        },
        "subject": {
            "participant_id": participant_id.strip(),
            "trial_id": trial_id.strip(),
            "outward_handlebar_position": outward_handlebar_position.strip(),
            "return_handlebar_position": return_handlebar_position.strip(),
        },
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
            "primary_window": f"central {comparison_intervals} inter-click intervals of each walking leg",
            "turnaround_excluded": True,
            "spoke_count": spokes,
            "movement_proxy_note": rules.get("movement_proxy_note"),
        },
        "provenance": {
            "calibration_profile_sha256": _sha256(profile_path),
            "study_protocol_sha256": _sha256(protocol_path),
        },
        "outputs": {
            "detections_csv": str(csv_path.resolve()),
            "session_summary_json": str(session_summary_path.resolve()),
            "reference_review_wav": str(review_path.resolve()),
            "isolated_review_wav": str((detection_dir / detector.ISOLATED_FILENAME).resolve()),
            "movement_summary_json": str((movement_dir / movement.SUMMARY_FILENAME).resolve()),
            "movement_timeline_csv": str((movement_dir / movement.TIMELINE_FILENAME).resolve()),
            "movement_plot_svg": str((movement_dir / movement.PLOT_FILENAME).resolve()),
            "subject_record_csv": str((output_dir / SUBJECT_RECORD_FILENAME).resolve()),
        },
    }
    summary_path = output_dir / STUDY_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(study_summary, indent=2), encoding="utf-8")
    _write_subject_record(output_dir / SUBJECT_RECORD_FILENAME, study_summary)
    return study_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end locked-protocol study session analysis")
    ap.add_argument("wav", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--profile", type=Path, default=Path("calibration/calibration_profile.json"))
    ap.add_argument("--protocol", type=Path, default=Path("study/study_protocol.json"))
    ap.add_argument("--participant-id", required=True)
    ap.add_argument("--trial-id", required=True)
    ap.add_argument("--outward-position", required=True)
    ap.add_argument("--return-position", required=True)
    ap.add_argument("--turnaround-seconds", type=float, default=None)
    args = ap.parse_args()
    result = analyse_study_session(
        args.wav,
        args.output_dir,
        args.profile,
        participant_id=args.participant_id,
        trial_id=args.trial_id,
        outward_handlebar_position=args.outward_position,
        return_handlebar_position=args.return_position,
        protocol_path=args.protocol,
        manual_turnaround_s=args.turnaround_seconds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
