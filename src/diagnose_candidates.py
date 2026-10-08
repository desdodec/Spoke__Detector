#!/usr/bin/env python3
"""Export onset-candidate diagnostics for Spoke Detector v4.

This script intentionally does not change detector behaviour. It runs the same
analysis path as detector_v4, then records why every onset candidate is accepted
or rejected. The output is designed to diagnose cross-recording generalisation
failures before changing thresholds.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import detector_v4 as d4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose v4 spoke-click candidates")
    p.add_argument("--training", type=Path, default=Path("short_sample/7_clicks.wav"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("candidate_diagnostics.csv"))
    p.add_argument("--training-hits", type=int, default=7)
    p.add_argument("--min-gap-ms", type=float, default=d4.DEFAULT_MIN_GAP_MS)
    p.add_argument("--highpass-hz", type=float, default=d4.DEFAULT_HIGHPASS_HZ)
    return p.parse_args()


def rejection_reason(c: d4.Candidate, profile: d4.Profile) -> str:
    prominence_ok = c.normalized_prominence >= profile.prominence_threshold
    distance_ok = c.distance <= profile.distance_threshold
    if prominence_ok and distance_ok:
        return "accepted"
    if not prominence_ok and not distance_ok:
        return "prominence+distance"
    if not prominence_ok:
        return "prominence"
    return "distance"


def main() -> None:
    args = parse_args()

    training = d4.load_wav(args.training)
    target = d4.load_wav(args.input)
    if target.sample_rate != training.sample_rate:
        raise RuntimeError("Training and target sample rates must match")

    training_analysis = d4.highpass_analysis(training.mono, training.sample_rate, args.highpass_hz)
    profile = d4.train_profile(training_analysis, training.sample_rate, args.training_hits)
    min_gap = args.min_gap_ms / 1000.0
    d4.validate_profile(training_analysis, training.sample_rate, profile, args.training_hits, min_gap)

    target_analysis = d4.highpass_analysis(target.mono, target.sample_rate, args.highpass_hz)
    candidates = d4.score_candidates(target_analysis, target.sample_rate, profile, min_gap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(candidates, key=lambda c: c.normalized_prominence, reverse=True)

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "rank_by_prominence",
            "timestamp_seconds",
            "normalized_prominence",
            "prominence_threshold",
            "prominence_ratio",
            "profile_distance",
            "distance_threshold",
            "distance_ratio",
            "onset_strength",
            "confidence",
            "decision",
        ])
        for rank, c in enumerate(ranked, 1):
            writer.writerow([
                rank,
                f"{c.time_seconds:.6f}",
                f"{c.normalized_prominence:.6f}",
                f"{profile.prominence_threshold:.6f}",
                f"{c.normalized_prominence / profile.prominence_threshold:.6f}",
                f"{c.distance:.6f}",
                f"{profile.distance_threshold:.6f}",
                f"{c.distance / profile.distance_threshold:.6f}",
                f"{c.onset_strength:.6f}",
                f"{c.confidence:.6f}",
                rejection_reason(c, profile),
            ])

    reasons: dict[str, int] = {}
    for c in candidates:
        reason = rejection_reason(c, profile)
        reasons[reason] = reasons.get(reason, 0) + 1

    prominences = np.asarray([c.normalized_prominence for c in candidates], dtype=float)
    distances = np.asarray([c.distance for c in candidates], dtype=float)
    prominence_pass = prominences >= profile.prominence_threshold
    distance_pass = distances <= profile.distance_threshold

    print(f"CANDIDATES={len(candidates)}")
    print(f"PROMINENCE_THRESHOLD={profile.prominence_threshold:.6f}")
    print(f"DISTANCE_THRESHOLD={profile.distance_threshold:.6f}")
    print(f"PASS_PROMINENCE={int(prominence_pass.sum())}")
    print(f"PASS_DISTANCE={int(distance_pass.sum())}")
    print(f"PASS_BOTH={int((prominence_pass & distance_pass).sum())}")
    for key in sorted(reasons):
        print(f"REJECT_{key.upper().replace('+', '_AND_')}={reasons[key]}")

    if len(candidates):
        top_prom = sorted(candidates, key=lambda c: c.normalized_prominence, reverse=True)[:20]
        print("TOP_PROMINENCE_CANDIDATES")
        for c in top_prom:
            print(
                f"  t={c.time_seconds:8.3f}  prom={c.normalized_prominence:8.3f} "
                f"({c.normalized_prominence/profile.prominence_threshold:5.2f}x gate) "
                f"dist={c.distance:6.3f} ({c.distance/profile.distance_threshold:5.2f}x gate) "
                f"decision={rejection_reason(c, profile)}"
            )

        top_dist = sorted(candidates, key=lambda c: c.distance)[:20]
        print("BEST_PROFILE_MATCH_CANDIDATES")
        for c in top_dist:
            print(
                f"  t={c.time_seconds:8.3f}  dist={c.distance:6.3f} "
                f"({c.distance/profile.distance_threshold:5.2f}x gate) "
                f"prom={c.normalized_prominence:8.3f} "
                f"({c.normalized_prominence/profile.prominence_threshold:5.2f}x gate) "
                f"decision={rejection_reason(c, profile)}"
            )

    print(f"DIAGNOSTICS_CSV={args.output}")


if __name__ == "__main__":
    main()
