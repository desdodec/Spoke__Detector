#!/usr/bin/env python3
"""Recording-adaptive bicycle spoke-click detector (v5).

v5 keeps the proven v4 high-pass/onset machinery but stops using the absolute
training spectral-distance gate as a hard cross-recording requirement. The
reason is empirical: Bike Test 2 contains abundant spoke-like transients but
all of them sit outside the v4 spectral profile learned from the original
7-click training clip.

Instead v5:
- high-pass filters only the analysis path
- detects all transient onset candidates
- finds a natural split in the target recording's prominence distribution
- accepts the high-prominence cluster
- applies v4's sequence-aware duplicate/split cleanup
- keeps spectral distance as diagnostic metadata rather than a hard gate

This preserves speed independence: no fixed spoke period is assumed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import detector_v4 as d4


def adaptive_prominence_threshold(candidates: list[d4.Candidate], training_floor: float) -> float:
    """Choose the natural low/high transient split in one recording.

    We search for the largest multiplicative gap in sorted prominence values
    between the training floor and the 95th percentile. A meaningful gap must
    be at least 15%; otherwise we fall back to the training-derived floor.
    """
    if not candidates:
        return float(training_floor)

    values = np.sort(np.asarray([c.normalized_prominence for c in candidates], dtype=float))
    if len(values) < 8:
        return float(training_floor)

    lower = max(float(training_floor), float(np.quantile(values, 0.20)))
    upper = float(np.quantile(values, 0.95))
    eligible = np.where((values[:-1] >= lower) & (values[:-1] <= upper))[0]
    if len(eligible) == 0:
        return float(training_floor)

    log_values = np.log(values + 1e-12)
    gaps = log_values[eligible + 1] - log_values[eligible]
    idx = int(eligible[int(np.argmax(gaps))])
    ratio = float(values[idx + 1] / max(values[idx], 1e-12))

    if ratio < 1.15:
        return float(training_floor)

    return float(np.sqrt(values[idx] * values[idx + 1]))


def adaptive_detect(candidates: list[d4.Candidate], threshold: float) -> list[d4.Detection]:
    result: list[d4.Detection] = []
    for c in candidates:
        if c.normalized_prominence < threshold:
            continue
        result.append(d4.Detection(
            time_seconds=c.time_seconds,
            distance=c.distance,
            confidence=float(min(1.0, c.normalized_prominence / max(threshold * 2.0, 1e-9))),
            onset_strength=c.onset_strength,
            normalized_prominence=c.normalized_prominence,
            source="adaptive_prominence",
            sequence_score=0.0,
        ))
    return result


def refine_sequence(detections: list[d4.Detection]) -> list[d4.Detection]:
    """Use only cleanup operations that do not require cross-recording profile matching."""
    refined = d4.merge_double_triggers(detections)
    refined = d4.suppress_split_intervals(refined)
    refined = d4.merge_double_triggers(refined)
    refined = d4.suppress_split_intervals(refined)
    return sorted(refined, key=lambda d: d.time_seconds)


def write_csv(path: Path, detections: list[d4.Detection], adaptive_threshold: float,
              profile: d4.Profile) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "hit_index", "timestamp_seconds", "timestamp_hms", "confidence",
            "profile_distance", "v4_distance_threshold", "distance_ratio",
            "onset_strength", "normalized_prominence", "adaptive_prominence_threshold",
            "prominence_ratio", "source", "sequence_score",
        ])
        for i, det in enumerate(detections, 1):
            writer.writerow([
                i,
                f"{det.time_seconds:.6f}",
                d4.timestamp_hms(det.time_seconds),
                f"{det.confidence:.6f}",
                f"{det.distance:.6f}",
                f"{profile.distance_threshold:.6f}",
                f"{det.distance / max(profile.distance_threshold, 1e-12):.6f}",
                f"{det.onset_strength:.6f}",
                f"{det.normalized_prominence:.6f}",
                f"{adaptive_threshold:.6f}",
                f"{det.normalized_prominence / max(adaptive_threshold, 1e-12):.6f}",
                det.source,
                f"{det.sequence_score:.6f}",
            ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recording-adaptive spoke click detector v5")
    p.add_argument("--training", type=Path, default=Path("short_sample/7_clicks.wav"))
    p.add_argument("--input", type=Path, default=Path("audio/Bike_test.wav"))
    p.add_argument("--output-dir", type=Path, default=Path("output_v5"))
    p.add_argument("--training-hits", type=int, default=7)
    p.add_argument("--min-gap-ms", type=float, default=d4.DEFAULT_MIN_GAP_MS)
    p.add_argument("--highpass-hz", type=float, default=d4.DEFAULT_HIGHPASS_HZ)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    training = d4.load_wav(args.training)
    training_analysis = d4.highpass_analysis(training.mono, training.sample_rate, args.highpass_hz)
    profile = d4.train_profile(training_analysis, training.sample_rate, args.training_hits)
    min_gap = args.min_gap_ms / 1000.0

    # Keep the original known-count training self-test as a hard regression check.
    self_test = d4.validate_profile(
        training_analysis, training.sample_rate, profile, args.training_hits, min_gap
    )

    target = d4.load_wav(args.input)
    if target.sample_rate != training.sample_rate:
        raise RuntimeError("Training and target sample rates must match")
    target_analysis = d4.highpass_analysis(target.mono, target.sample_rate, args.highpass_hz)
    candidates = d4.score_candidates(target_analysis, target.sample_rate, profile, min_gap)

    adaptive_threshold = adaptive_prominence_threshold(candidates, profile.prominence_threshold)
    base = adaptive_detect(candidates, adaptive_threshold)
    detections = refine_sequence(base)

    print(f"Self-test: PASS ({len(self_test)}/{args.training_hits}, no extras)")
    print(f"V4 training prominence floor: {profile.prominence_threshold:.4f} x MAD")
    print(f"Adaptive target prominence threshold: {adaptive_threshold:.4f} x MAD")
    print(f"All onset candidates: {len(candidates)}")
    print(f"Adaptive base detections: {len(base)}")
    print(f"Sequence-refined detections: {len(detections)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "detections_v5.csv"
    review_path = args.output_dir / "review_clicks_v5.wav"
    hits_path = args.output_dir / "detected_hits_v5.wav"

    write_csv(csv_path, detections, adaptive_threshold, profile)
    d4.write_review_audio(review_path, target, detections)
    d4.write_extracted_hits(hits_path, target, detections)

    print(f"CSV: {csv_path}")
    print(f"Review audio: {review_path}")
    print(f"Extracted hits: {hits_path}")


if __name__ == "__main__":
    main()
