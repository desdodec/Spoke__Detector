#!/usr/bin/env python3
"""Evaluate a broad indoor spoke-click model against Bike Test 2 and Jo Grant.

This script deliberately treats ``audio/Spoke test new.wav`` as development
calibration evidence, not physical ground truth.  It learns a *family* of click
spectra from strong indoor transients using many exemplars, then asks whether
that family helps with ambiguous events in the two outdoor/problem recordings.

Safety rules for the experiment:
- absolute pitch is never used as a hard gate;
- strong prominence detections are never deleted by the indoor model;
- the indoor model may only nominate a below-threshold event as a possible
  recovery when local timing independently indicates a likely missing click;
- all proposed recoveries are reported separately from the existing detector.

Outputs are JSON plus a CSV of proposed recovery candidates for audit.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

import detector_v4 as d4
import detector_v5 as d5

INDOOR = Path("audio/Spoke test new.wav")
TARGETS = {
    "bike_test_2": Path("audio/Bike test 2.wav"),
    "jo_grant": Path("audio/Jo_Grant_Mono.wav"),
}
OUT = Path("output_indoor_model_eval")
CALIBRATION_FLOOR = 19.647
HIGHPASS_HZ = 250.0
MIN_GAP_S = 0.025


@dataclass
class Rescue:
    recording: str
    time_seconds: float
    prominence: float
    prominence_ratio: float
    compatibility_distance: float
    compatibility_percentile: float
    left_gap_s: float
    right_gap_s: float
    local_period_s: float
    timing_score: float


def make_candidates(signal: np.ndarray, sample_rate: int) -> list[d4.Candidate]:
    times, strengths, prominences = d4.all_onset_peaks(signal, sample_rate, MIN_GAP_S)
    if len(times) == 0:
        return []
    scale = float(np.max(strengths)) + 1e-12
    return [
        d4.Candidate(
            time_seconds=float(t),
            distance=0.0,
            confidence=1.0,
            onset_strength=float(s / scale),
            normalized_prominence=float(p),
        )
        for t, s, p in zip(times, strengths, prominences)
    ]


def features_for(signal: np.ndarray, sample_rate: int, candidates: list[d4.Candidate]) -> np.ndarray:
    if not candidates:
        return np.zeros((0, 25), dtype=float)
    return np.stack([d4.feature_vector(signal, sample_rate, c.time_seconds) for c in candidates])


def robust_feature_scale(features: np.ndarray) -> np.ndarray:
    med = np.median(features, axis=0)
    mad = np.median(np.abs(features - med), axis=0)
    # The additive floor prevents very stable indoor dimensions from becoming
    # accidental hard gates when microphone/environment changes.
    return 1.4826 * mad + 0.35


def nearest_distance(query: np.ndarray, reference: np.ndarray, scale: np.ndarray, k: int = 5) -> float:
    d = np.sqrt(np.mean(((reference - query) / scale) ** 2, axis=1))
    k = min(max(1, k), len(d))
    return float(np.mean(np.partition(d, k - 1)[:k]))


def leave_one_out_distances(features: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = []
    for i, q in enumerate(features):
        ref = np.delete(features, i, axis=0)
        if len(ref) == 0:
            out.append(0.0)
        else:
            out.append(nearest_distance(q, ref, scale))
    return np.asarray(out, dtype=float)


def local_period(accepted_times: np.ndarray, left_index: int, radius: int = 5) -> float | None:
    intervals = np.diff(accepted_times)
    if len(intervals) < 3:
        return None
    lo = max(0, left_index - radius)
    hi = min(len(intervals), left_index + radius + 1)
    vals = intervals[lo:hi]
    if len(vals) < 3:
        return None
    med = float(np.median(vals))
    good = vals[(vals > 0.55 * med) & (vals < 1.60 * med)]
    return float(np.median(good)) if len(good) >= 3 else med


def timing_rescue_score(t: float, accepted_times: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return score and timing context when t plausibly fills one missing click."""
    pos = int(np.searchsorted(accepted_times, t))
    if pos <= 0 or pos >= len(accepted_times):
        return None
    left = float(t - accepted_times[pos - 1])
    right = float(accepted_times[pos] - t)
    whole = left + right
    period = local_period(accepted_times, pos - 1)
    if period is None or period <= 0:
        return None

    # A missing click normally leaves a gap near 2 x local period, and the
    # candidate should split that gap into two roughly one-period intervals.
    whole_err = abs(whole / period - 2.0)
    left_err = abs(left / period - 1.0)
    right_err = abs(right / period - 1.0)
    score = float(np.exp(-(1.4 * whole_err + left_err + right_err)))
    if whole < 1.45 * period or whole > 2.65 * period:
        return None
    if left < 0.55 * period or left > 1.45 * period:
        return None
    if right < 0.55 * period or right > 1.45 * period:
        return None
    return score, left, right, period


def percentile(value: float, reference: np.ndarray) -> float:
    return float(100.0 * np.mean(reference <= value))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    indoor_audio = d4.load_wav(INDOOR)
    indoor_signal = d4.highpass_analysis(indoor_audio.mono, indoor_audio.sample_rate, HIGHPASS_HZ)
    indoor_candidates = make_candidates(indoor_signal, indoor_audio.sample_rate)
    indoor_adaptive = d5.adaptive_prominence_threshold(indoor_candidates, 4.0)

    # Use only the strongest part of the natural indoor cluster as exemplars.
    # This is intentionally conservative because the exact physical count of
    # this development recording is unknown.
    indoor_values = np.asarray([c.normalized_prominence for c in indoor_candidates], dtype=float)
    exemplar_threshold = max(float(indoor_adaptive), float(np.quantile(indoor_values, 0.65)), 6.0)
    exemplars = [c for c in indoor_candidates if c.normalized_prominence >= exemplar_threshold]
    exemplar_features = features_for(indoor_signal, indoor_audio.sample_rate, exemplars)
    feat_scale = robust_feature_scale(exemplar_features)
    loo = leave_one_out_distances(exemplar_features, feat_scale)
    compatibility_limit = float(np.quantile(loo, 0.95))

    report: dict[str, object] = {
        "model": {
            "source": str(INDOOR),
            "physical_ground_truth": False,
            "indoor_candidates": len(indoor_candidates),
            "indoor_adaptive_threshold": float(indoor_adaptive),
            "exemplar_threshold": float(exemplar_threshold),
            "exemplars": len(exemplars),
            "compatibility_loo_median": float(np.median(loo)),
            "compatibility_loo_p95": compatibility_limit,
            "model_type": "multi_exemplar_nearest_neighbour_soft_diagnostic",
        },
        "recordings": {},
    }

    rescues: list[Rescue] = []

    for name, path in TARGETS.items():
        audio = d4.load_wav(path)
        signal = d4.highpass_analysis(audio.mono, audio.sample_rate, HIGHPASS_HZ)
        candidates = make_candidates(signal, audio.sample_rate)
        recording_adaptive = d5.adaptive_prominence_threshold(candidates, 4.0)
        effective = max(float(recording_adaptive), CALIBRATION_FLOOR)
        accepted = d5.refine_sequence(d5.adaptive_detect(candidates, effective))
        accepted_times = np.asarray([d.time_seconds for d in accepted], dtype=float)

        feats = features_for(signal, audio.sample_rate, candidates)
        distances = np.asarray([
            nearest_distance(v, exemplar_features, feat_scale) for v in feats
        ], dtype=float)
        accepted_mask = np.asarray([c.normalized_prominence >= effective for c in candidates])
        accepted_dist = distances[accepted_mask]

        # Ambiguous zone only.  Below 70% of the calibrated threshold we refuse
        # to rescue regardless of spectral similarity.
        proposed = 0
        for c, dist in zip(candidates, distances):
            ratio = c.normalized_prominence / max(effective, 1e-12)
            if ratio < 0.70 or ratio >= 1.0:
                continue
            timing = timing_rescue_score(c.time_seconds, accepted_times)
            if timing is None:
                continue
            timing_score, left, right, period = timing
            # The broad indoor family is only supporting evidence.  Require the
            # candidate to be no farther than the worst 5% of indoor exemplars,
            # plus a modest cross-recording tolerance.
            compatible = dist <= compatibility_limit * 1.35
            if compatible and timing_score >= 0.45:
                rescues.append(Rescue(
                    recording=name,
                    time_seconds=c.time_seconds,
                    prominence=c.normalized_prominence,
                    prominence_ratio=ratio,
                    compatibility_distance=float(dist),
                    compatibility_percentile=percentile(float(dist), loo),
                    left_gap_s=left,
                    right_gap_s=right,
                    local_period_s=period,
                    timing_score=timing_score,
                ))
                proposed += 1

        report["recordings"][name] = {
            "path": str(path),
            "duration_s": len(audio.mono) / audio.sample_rate,
            "candidates": len(candidates),
            "recording_adaptive_threshold": float(recording_adaptive),
            "effective_threshold": float(effective),
            "existing_refined_detections": len(accepted),
            "accepted_compatibility_median": float(np.median(accepted_dist)) if len(accepted_dist) else None,
            "accepted_compatibility_p95": float(np.quantile(accepted_dist, 0.95)) if len(accepted_dist) else None,
            "accepted_within_indoor_p95_x1_35_pct": float(100.0 * np.mean(accepted_dist <= compatibility_limit * 1.35)) if len(accepted_dist) else None,
            "proposed_soft_recoveries": proposed,
            "proposed_count_if_all_recoveries_were_added": len(accepted) + proposed,
            "important_note": "Proposed recoveries are diagnostic only and are not inserted into detector output.",
        }

    with (OUT / "indoor_model_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with (OUT / "proposed_recoveries.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(Rescue.__annotations__.keys()))
        w.writeheader()
        for r in rescues:
            w.writerow(asdict(r))

    print(json.dumps(report, indent=2))
    print("\nPROPOSED RECOVERIES")
    for r in rescues:
        print(asdict(r))


if __name__ == "__main__":
    main()
