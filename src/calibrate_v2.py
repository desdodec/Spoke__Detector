#!/usr/bin/env python3
"""Improved calibration app for the 36-spoke wheel.

Key changes from calibrate.py:
- defaults to the project's confirmed 36-spoke wheel
- treats one click per spoke as the physical calibration model
- removes expected-count information from the transient-separation PASS test
- stores wheel geometry explicitly in calibration_profile.json
- keeps expected-count boundary ratio as a diagnostic only

The underlying detector is still allowed to find clicks naturally; this module
never selects the strongest N events just because N is known.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import calibrate as legacy

DEFAULT_SPOKES = 36

# Keep a stable reference even when this module is imported by a test/script.
if not hasattr(legacy, "_ORIGINAL_CALIBRATE"):
    legacy._ORIGINAL_CALIBRATE = legacy.calibrate


def calibrate(
    wav_path: Path,
    output_dir: Path,
    spoke_count: int,
    revolutions: int,
    highpass_hz: float = legacy.d4.DEFAULT_HIGHPASS_HZ,
    min_gap_ms: float = legacy.d4.DEFAULT_MIN_GAP_MS,
    count_tolerance_pct: float = legacy.DEFAULT_COUNT_TOLERANCE_PCT,
    min_separation_ratio: float = legacy.DEFAULT_MIN_SEPARATION_RATIO,
    max_interval_outlier_pct: float = legacy.DEFAULT_MAX_INTERVAL_OUTLIER_PCT,
):
    """Run legacy calibration, then apply non-circular acceptance logic.

    The detector still operates independently. The known physical count is used
    to assess count accuracy, but not to rescue a weak natural prominence split.
    """
    metrics, hits, profile_path, csv_path, review_path = legacy._ORIGINAL_CALIBRATE(
        wav_path=wav_path,
        output_dir=output_dir,
        spoke_count=spoke_count,
        revolutions=revolutions,
        highpass_hz=highpass_hz,
        min_gap_ms=min_gap_ms,
        count_tolerance_pct=count_tolerance_pct,
        min_separation_ratio=min_separation_ratio,
        max_interval_outlier_pct=max_interval_outlier_pct,
    )

    # Methodological correction: expected_boundary_ratio uses the known expected
    # N and therefore must be diagnostic-only. PASS separation is based solely on
    # the natural acoustic split discovered without knowing N.
    metrics.pass_separation = bool(metrics.natural_split_ratio >= min_separation_ratio)
    metrics.passed = bool(metrics.pass_count and metrics.pass_separation and metrics.pass_timing)

    profile: dict[str, Any] = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["status"] = "PASS" if metrics.passed else "FAIL"
    profile["spoke_count"] = int(spoke_count)
    profile["wheel_revolutions"] = int(revolutions)
    profile["expected_hits"] = int(spoke_count * revolutions)
    profile["geometry"] = {
        "spoke_count": int(spoke_count),
        "spoke_pairs": int(spoke_count // 2) if spoke_count % 2 == 0 else None,
        "expected_clicks_per_revolution": int(spoke_count),
        "assumption": "one cable-tie click per spoke crossing",
        "detection_use": "diagnostic_only_not_count_forcing",
    }
    profile["metrics"] = asdict(metrics)
    profile["acceptance_criteria"]["separation_basis"] = "natural_split_ratio_only"
    profile["diagnostics"] = {
        "expected_boundary_ratio": float(metrics.expected_boundary_ratio),
        "expected_boundary_ratio_role": (
            "report_only; not used for PASS because it depends on the known expected count"
        ),
    }
    profile["notes"] = (
        "A PASS requires independently detected events to agree with the known physical "
        "spoke-strike count, a naturally separated prominence cluster, and acceptable "
        "timing plausibility. Expected count is never used to choose the events. Wheel "
        "geometry is stored for diagnostics and movement metrics, not to force session "
        "detections to a multiple of the spoke count."
    )
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return metrics, hits, profile_path, csv_path, review_path


class CalibrationApp(legacy.CalibrationApp):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector — 36-Spoke Calibration")
        self.spokes_var.set(str(DEFAULT_SPOKES))
        self._set_result(
            "Ready for controlled calibration.\n\n"
            "Confirmed wheel geometry: 36 spokes = 36 expected clicks per full revolution.\n"
            "Use exact complete revolutions for a formal calibration.\n"
        )


def main() -> None:
    # The inherited GUI calls legacy.calibrate; route that call through v2.
    legacy.calibrate = calibrate
    app = CalibrationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
