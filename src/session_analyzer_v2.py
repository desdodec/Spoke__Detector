#!/usr/bin/env python3
"""Geometry-aware session analyzer.

This is a conservative extension of session_analyzer.py. Detection itself is
unchanged: calibrated prominence floor + recording adaptation upward + sequence
cleanup. The confirmed 36-spoke geometry is used only to report wheel-motion
metrics; it never forces the number or timing of detections.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import session_analyzer as legacy


def geometry_metrics(result: legacy.SessionResult, profile: dict[str, Any] | None) -> dict[str, Any]:
    spoke_count = (profile or {}).get("spoke_count")
    try:
        spoke_count = int(spoke_count) if spoke_count is not None else None
    except (TypeError, ValueError):
        spoke_count = None
    if not spoke_count or spoke_count <= 0:
        return {
            "spoke_count": None,
            "geometry_available": False,
            "note": "No valid spoke count in calibration profile; no revolution metrics computed.",
        }

    median_interval = result.median_interval_s
    estimated_revolutions = float(result.detected_hits / spoke_count)
    click_rate_hz = None
    rotation_period_s = None
    rotation_rate_hz = None
    if median_interval is not None and median_interval > 0:
        click_rate_hz = float(1.0 / median_interval)
        rotation_period_s = float(median_interval * spoke_count)
        rotation_rate_hz = float(1.0 / rotation_period_s)

    return {
        "spoke_count": spoke_count,
        "spoke_pairs": spoke_count // 2 if spoke_count % 2 == 0 else None,
        "geometry_available": True,
        "one_click_per_spoke_assumed": True,
        "estimated_revolutions_from_accepted_clicks": estimated_revolutions,
        "median_click_rate_hz": click_rate_hz,
        "median_wheel_rotation_period_s": rotation_period_s,
        "median_wheel_rotation_rate_hz": rotation_rate_hz,
        "important_note": (
            "These are movement/wheel-motion diagnostics derived from accepted click timing. "
            "The analyzer does not round, add, remove, or force events to make the count a "
            "multiple of the spoke count."
        ),
    }


def analyse_session(
    wav_path: Path,
    output_dir: Path,
    profile_path: Path | None = None,
    highpass_override: float | None = None,
    min_gap_override_ms: float | None = None,
):
    result, csv_path, review_path, summary_path = legacy._ORIGINAL_ANALYSE_SESSION(
        wav_path=wav_path,
        output_dir=output_dir,
        profile_path=profile_path,
        highpass_override=highpass_override,
        min_gap_override_ms=min_gap_override_ms,
    )

    profile = legacy.load_profile(profile_path) if profile_path is not None else None
    geom = geometry_metrics(result, profile)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["geometry"] = geom
    summary["notes"] = (
        str(summary.get("notes", ""))
        + " Confirmed wheel geometry is used only for derived revolution/rate diagnostics; "
          "it is never used to force detection count or timing."
    ).strip()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result, csv_path, review_path, summary_path


class SessionAnalyzerApp(legacy.SessionAnalyzerApp):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector — 36-Spoke Session Analyzer")

    def _run(self) -> None:
        try:
            wav_text = self.wav_var.get().strip()
            out_text = self.output_var.get().strip()
            profile_text = self.profile_var.get().strip()
            if not wav_text:
                raise ValueError("Choose a session WAV first.")
            if not out_text:
                raise ValueError("Choose an output folder.")

            wav_path = Path(wav_text)
            output_dir = Path(out_text)
            profile_path = Path(profile_text) if profile_text else None

            self.run_button.configure(state="disabled")
            self.status_var.set("Analysing…")
            self.update_idletasks()

            result, csv_path, review_path, summary_path = analyse_session(
                wav_path=wav_path,
                output_dir=output_dir,
                profile_path=profile_path,
            )
            isolated_path = output_dir / legacy.ISOLATED_FILENAME
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            geom = summary.get("geometry") or {}

            lines = [
                "ANALYSIS COMPLETE",
                "",
                f"Mode: {result.analysis_mode}",
                f"Calibration status: {result.calibration_status}",
                f"Profile kind: {result.profile_kind}",
                f"Quality: {result.quality_status}",
                "",
                f"Detected hits: {result.detected_hits}",
                f"Recording duration: {result.duration_s:.2f} s",
                f"Recording-only adaptive threshold: {result.recording_adaptive_threshold:.3f}",
                f"Effective threshold used: {result.effective_prominence_threshold:.3f}",
                f"Threshold source: {result.threshold_source}",
            ]
            if result.calibration_prominence_threshold is not None:
                lines.append(f"Calibration threshold: {result.calibration_prominence_threshold:.3f}")
            if result.plateau_change_pct is not None:
                lines.append(f"Count change at +20% threshold: {result.plateau_change_pct:.1f}%")
            if result.below_floor_pressure_pct is not None:
                lines.append(f"Extra detections at -20% threshold: {result.below_floor_pressure_pct:.1f}%")

            if geom.get("geometry_available"):
                lines.extend([
                    "",
                    "36-SPOKE WHEEL DIAGNOSTICS",
                    f"Spokes: {geom['spoke_count']}",
                    f"Estimated wheel revolutions represented: {geom['estimated_revolutions_from_accepted_clicks']:.2f}",
                ])
                if geom.get("median_click_rate_hz") is not None:
                    lines.append(f"Median click rate: {geom['median_click_rate_hz']:.2f} clicks/s")
                    lines.append(f"Median revolution period: {geom['median_wheel_rotation_period_s']:.2f} s/rev")
                    lines.append(f"Median rotation rate: {geom['median_wheel_rotation_rate_hz']:.3f} rev/s")
                lines.append("Geometry is diagnostic only — detections are NOT forced to multiples of 36.")

            lines.extend([
                "",
                f"CSV: {csv_path}",
                f"Filtered review: {review_path}",
                f"Isolated review: {isolated_path}",
                f"Summary: {summary_path}",
            ])
            if result.profile_kind == "PROVISIONAL_TEST_ONLY":
                lines.extend([
                    "",
                    "WARNING: This is still the provisional Bike Test 2 acoustic profile, not a formal known-revolution calibration.",
                ])

            self._set_result("\n".join(lines))
            self.status_var.set("Complete")
        except Exception as exc:
            self.status_var.set("Failed")
            self._set_result(traceback.format_exc())
            legacy.messagebox.showerror("Session analysis failed", str(exc))
        finally:
            self.run_button.configure(state="normal")


def main() -> None:
    app = SessionAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    if not hasattr(legacy, "_ORIGINAL_ANALYSE_SESSION"):
        legacy._ORIGINAL_ANALYSE_SESSION = legacy.analyse_session
    main()
