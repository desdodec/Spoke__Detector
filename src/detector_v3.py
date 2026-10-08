#!/usr/bin/env python3
"""Sequence-aware bicycle spoke-click detector (v3).

v3 keeps v2's acoustic profile and adds conservative temporal reasoning:
- remove extra detections that split one locally-consistent spoke interval
- recover weak candidates inside gaps that are close to integer multiples of the local period
- never assumes a fixed walking speed; the local period is estimated continuously

Outputs:
  detections_v3.csv
  review_clicks_v3.wav
  detected_hits_v3.wav
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, peak_prominences, stft

FRAME_SIZE = 1024
HOP_SIZE = 128
HIT_WINDOW_SECONDS = 0.060
DEFAULT_MIN_GAP_MS = 25.0


@dataclass
class Audio:
    sample_rate: int
    mono: np.ndarray
    original: np.ndarray


@dataclass
class Profile:
    training_onsets: np.ndarray
    features: np.ndarray
    feature_scale: np.ndarray
    distance_threshold: float
    prominence_threshold: float


@dataclass
class Detection:
    time_seconds: float
    distance: float
    confidence: float
    onset_strength: float
    normalized_prominence: float
    source: str = "acoustic"
    sequence_score: float = 0.0


@dataclass
class Candidate:
    time_seconds: float
    distance: float
    confidence: float
    onset_strength: float
    normalized_prominence: float


def _to_float32(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), abs(info.max))
        return data.astype(np.float32) / float(scale)
    return data.astype(np.float32)


def load_wav(path: Path) -> Audio:
    sample_rate, data = wavfile.read(path)
    original = _to_float32(data)
    mono = original.mean(axis=1) if original.ndim == 2 else original.copy()
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 0:
        mono = mono / peak
    return Audio(sample_rate, mono.astype(np.float32), original)


def onset_curve(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    _, times, spectrum = stft(
        signal,
        fs=sample_rate,
        nperseg=FRAME_SIZE,
        noverlap=FRAME_SIZE - HOP_SIZE,
        boundary=None,
        padded=False,
    )
    magnitude = np.log1p(25.0 * np.abs(spectrum))
    flux = np.maximum(np.diff(magnitude, axis=1), 0.0).sum(axis=0)
    return times[1:], flux


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1e-12
    return median, mad


def all_onset_peaks(signal: np.ndarray, sample_rate: int, min_gap_seconds: float):
    times, flux = onset_curve(signal, sample_rate)
    _, mad = robust_scale(flux)
    distance_frames = max(1, int(np.ceil(min_gap_seconds / (HOP_SIZE / sample_rate))))
    peaks, _ = find_peaks(flux, distance=distance_frames, prominence=1e-12)
    prominences = peak_prominences(flux, peaks)[0]
    return times[peaks], flux[peaks], prominences / mad


def find_training_onsets(signal: np.ndarray, sample_rate: int, expected_hits: int):
    times, flux = onset_curve(signal, sample_rate)
    _, mad = robust_scale(flux)
    distance_frames = max(1, int(np.ceil(0.080 / (HOP_SIZE / sample_rate))))
    peaks, _ = find_peaks(flux, distance=distance_frames, prominence=max(0.25 * mad, 1e-12))
    if len(peaks) < expected_hits:
        raise RuntimeError(f"Training file yielded {len(peaks)} plausible onsets; expected {expected_hits}.")
    prominences = peak_prominences(flux, peaks)[0]
    strongest = peaks[np.argsort(flux[peaks])[-expected_hits:]]
    strongest.sort()
    prominence_by_peak = {int(p): float(pr) for p, pr in zip(peaks, prominences)}
    selected_prominence = np.asarray([prominence_by_peak[int(p)] / mad for p in strongest])
    return times[strongest], selected_prominence


def feature_vector(signal: np.ndarray, sample_rate: int, onset_seconds: float) -> np.ndarray:
    start = max(0, int(round((onset_seconds - 0.005) * sample_rate)))
    length = int(round(HIT_WINDOW_SECONDS * sample_rate))
    segment = signal[start:min(len(signal), start + length)].astype(np.float64, copy=True)
    if len(segment) < length:
        segment = np.pad(segment, (0, length - len(segment)))
    segment -= np.mean(segment)
    segment /= np.sqrt(np.mean(segment * segment)) + 1e-9

    power = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
    frequencies = np.fft.rfftfreq(len(segment), d=1.0 / sample_rate)
    low_hz, high_hz = 250.0, min(10000.0, sample_rate * 0.45)
    useful = (frequencies >= low_hz) & (frequencies <= high_hz)
    useful_power = float(power[useful].sum()) + 1e-12
    edges = np.geomspace(low_hz, high_hz, 17)
    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(np.log(float(power[mask].sum()) / useful_power + 1e-8))

    envelope = []
    for index in range(6):
        a = int(index * len(segment) / 6)
        b = int((index + 1) * len(segment) / 6)
        chunk = segment[a:b]
        envelope.append(np.log(np.sqrt(np.mean(chunk * chunk)) + 1e-6))

    zcr = float(np.mean(segment[:-1] * segment[1:] < 0))
    total = float(power.sum()) + 1e-12
    centroid = float((frequencies * power).sum() / total)
    bandwidth = float(np.sqrt((((frequencies - centroid) ** 2) * power).sum() / total))
    return np.asarray(bands + envelope + [zcr, np.log(centroid + 1), np.log(bandwidth + 1)])


def train_profile(signal: np.ndarray, sample_rate: int, expected_hits: int) -> Profile:
    onsets, training_prominence = find_training_onsets(signal, sample_rate, expected_hits)
    features = np.stack([feature_vector(signal, sample_rate, onset) for onset in onsets])
    feature_scale = features.std(axis=0) + 0.15
    loo = []
    for i, vector in enumerate(features):
        others = np.delete(features, i, axis=0)
        distances = np.sqrt(np.mean(((others - vector) / feature_scale) ** 2, axis=1))
        loo.append(float(np.min(distances)))
    return Profile(
        training_onsets=onsets,
        features=features,
        feature_scale=feature_scale,
        distance_threshold=max(1.20, max(loo) * 1.25),
        prominence_threshold=max(4.0, float(np.min(training_prominence) * 0.75)),
    )


def score_candidates(signal: np.ndarray, sample_rate: int, profile: Profile, min_gap_seconds: float) -> list[Candidate]:
    times, strengths, prominences = all_onset_peaks(signal, sample_rate, min_gap_seconds)
    if len(times) == 0:
        return []
    max_strength = float(np.max(strengths)) + 1e-12
    result = []
    for onset, strength, prominence in zip(times, strengths, prominences):
        vector = feature_vector(signal, sample_rate, float(onset))
        distances = np.sqrt(np.mean(((profile.features - vector) / profile.feature_scale) ** 2, axis=1))
        distance = float(np.min(distances))
        result.append(Candidate(
            time_seconds=float(onset),
            distance=distance,
            confidence=float(np.exp(-distance / profile.distance_threshold)),
            onset_strength=float(strength / max_strength),
            normalized_prominence=float(prominence),
        ))
    return result


def acoustic_detect(candidates: list[Candidate], profile: Profile) -> list[Detection]:
    return [Detection(c.time_seconds, c.distance, c.confidence, c.onset_strength,
                      c.normalized_prominence, "acoustic", 0.0)
            for c in candidates
            if c.normalized_prominence >= profile.prominence_threshold
            and c.distance <= profile.distance_threshold]


def local_period(times: np.ndarray, center_index: int, radius: int = 7) -> float | None:
    if len(times) < 4:
        return None
    intervals = np.diff(times)
    lo = max(0, center_index - radius)
    hi = min(len(intervals), center_index + radius + 1)
    vals = intervals[lo:hi]
    if len(vals) < 2:
        return None
    med = float(np.median(vals))
    good = vals[(vals > 0.60 * med) & (vals < 1.45 * med)]
    if len(good) >= 2:
        med = float(np.median(good))
    return med


def suppress_split_intervals(detections: list[Detection]) -> list[Detection]:
    work = sorted(detections, key=lambda d: d.time_seconds)
    changed = True
    while changed and len(work) >= 5:
        changed = False
        times = np.asarray([d.time_seconds for d in work])
        for i in range(1, len(work) - 1):
            left = times[i] - times[i - 1]
            right = times[i + 1] - times[i]
            period = local_period(times, i)
            if period is None:
                continue
            combined = left + right
            split_like = (
                left < 0.72 * period
                and right < 0.82 * period
                and abs(combined - period) <= 0.24 * period
            )
            if not split_like:
                continue
            d = work[i]
            acoustic_quality = 0.55 * (d.distance / max(1e-9, 1.2)) + 0.45 / max(d.normalized_prominence, 1e-6)
            if acoustic_quality < 0.40 and min(left, right) > 0.45 * period:
                continue
            del work[i]
            changed = True
            break
    return work


def find_recovery_candidate(candidates: list[Candidate], target_time: float, tolerance: float,
                            profile: Profile, used_times: np.ndarray) -> Candidate | None:
    pool = []
    for c in candidates:
        if abs(c.time_seconds - target_time) > tolerance:
            continue
        if used_times.size and np.min(np.abs(used_times - c.time_seconds)) < 0.020:
            continue
        if c.distance > profile.distance_threshold * 1.20:
            continue
        if c.normalized_prominence < max(2.25, profile.prominence_threshold * 0.45):
            continue
        timing_error = abs(c.time_seconds - target_time) / max(tolerance, 1e-9)
        score = (c.distance / (profile.distance_threshold * 1.20)) + 0.35 * timing_error - 0.05 * min(c.normalized_prominence, 12.0)
        pool.append((score, c))
    if not pool:
        return None
    return min(pool, key=lambda x: x[0])[1]


def recover_missing_intervals(detections: list[Detection], candidates: list[Candidate], profile: Profile) -> list[Detection]:
    work = sorted(detections, key=lambda d: d.time_seconds)
    for _ in range(3):
        if len(work) < 4:
            break
        inserted = False
        times = np.asarray([d.time_seconds for d in work])
        for i in range(len(work) - 1):
            gap = times[i + 1] - times[i]
            period = local_period(times, i)
            if period is None or period <= 0:
                continue
            multiple = int(round(gap / period))
            if multiple < 2 or multiple > 4:
                continue
            if abs(gap - multiple * period) > 0.22 * period:
                continue
            spacing = gap / multiple
            for k in range(1, multiple):
                predicted = times[i] + k * spacing
                cand = find_recovery_candidate(
                    candidates, predicted, tolerance=max(0.020, 0.22 * period),
                    profile=profile, used_times=times,
                )
                if cand is None:
                    continue
                timing_error = abs(cand.time_seconds - predicted) / period
                work.append(Detection(
                    cand.time_seconds, cand.distance, cand.confidence,
                    cand.onset_strength, cand.normalized_prominence,
                    "sequence_recovery", max(0.0, 1.0 - timing_error),
                ))
                work.sort(key=lambda d: d.time_seconds)
                inserted = True
                break
            if inserted:
                break
        if not inserted:
            break
    return work


def sequence_refine(base: list[Detection], candidates: list[Candidate], profile: Profile) -> list[Detection]:
    refined = suppress_split_intervals(base)
    refined = recover_missing_intervals(refined, candidates, profile)
    refined = suppress_split_intervals(refined)
    return sorted(refined, key=lambda d: d.time_seconds)


def validate_profile(signal: np.ndarray, sample_rate: int, profile: Profile, expected_hits: int,
                     min_gap_seconds: float) -> list[Detection]:
    candidates = score_candidates(signal, sample_rate, profile, min_gap_seconds)
    base = acoustic_detect(candidates, profile)
    refined = sequence_refine(base, candidates, profile)
    if len(refined) != expected_hits:
        raise RuntimeError(
            f"Profile self-test failed: expected {expected_hits} hits but v3 detected {len(refined)}."
        )
    return refined


def timestamp_hms(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"


def write_csv(path: Path, detections: Iterable[Detection]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hit_index", "timestamp_seconds", "timestamp_hms", "confidence",
                         "profile_distance", "onset_strength", "normalized_prominence",
                         "source", "sequence_score"])
        for i, d in enumerate(detections, 1):
            writer.writerow([i, f"{d.time_seconds:.6f}", timestamp_hms(d.time_seconds),
                             f"{d.confidence:.6f}", f"{d.distance:.6f}",
                             f"{d.onset_strength:.6f}", f"{d.normalized_prominence:.6f}",
                             d.source, f"{d.sequence_score:.6f}"])


def marker_click(sample_rate: int, duration: float = 0.025) -> np.ndarray:
    count = max(1, int(round(duration * sample_rate)))
    t = np.arange(count, dtype=np.float64) / sample_rate
    click = (np.sin(2*np.pi*2800*t) + 0.65*np.sin(2*np.pi*4200*t)) * np.exp(-t*90.0)
    return (click / (np.max(np.abs(click)) + 1e-12)).astype(np.float32)


def float_to_int16(signal: np.ndarray) -> np.ndarray:
    return np.round(np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)


def write_review_audio(path: Path, audio: Audio, detections: list[Detection]) -> None:
    source = audio.original.astype(np.float32, copy=True)
    if source.ndim == 1:
        source = source[:, None]
    review = source * 0.78
    click = marker_click(audio.sample_rate)
    for hit in detections:
        start = int(round(hit.time_seconds * audio.sample_rate))
        stop = min(len(review), start + len(click))
        if start < len(review):
            review[start:stop] += click[:stop-start, None] * 0.45
    peak = float(np.max(np.abs(review))) if review.size else 0.0
    if peak > 0.99:
        review *= 0.99 / peak
    out = review[:, 0] if audio.original.ndim == 1 else review
    wavfile.write(path, audio.sample_rate, float_to_int16(out))


def write_extracted_hits(path: Path, audio: Audio, detections: list[Detection],
                         pre_seconds: float = 0.012, post_seconds: float = 0.070,
                         gap_seconds: float = 0.025) -> None:
    pre = int(round(pre_seconds * audio.sample_rate))
    post = int(round(post_seconds * audio.sample_rate))
    gap = np.zeros(int(round(gap_seconds * audio.sample_rate)), dtype=np.float32)
    pieces = []
    for hit in detections:
        center = int(round(hit.time_seconds * audio.sample_rate))
        excerpt = audio.mono[max(0, center-pre):min(len(audio.mono), center+post)]
        if excerpt.size:
            pieces.extend([excerpt, gap])
    combined = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 0:
        combined *= 0.95 / peak
    wavfile.write(path, audio.sample_rate, float_to_int16(combined))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequence-aware bicycle spoke click detector v3")
    p.add_argument("--training", type=Path, default=Path("short_sample/7_clicks.wav"))
    p.add_argument("--input", type=Path, default=Path("audio/Bike_test.wav"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--training-hits", type=int, default=7)
    p.add_argument("--min-gap-ms", type=float, default=DEFAULT_MIN_GAP_MS)
    p.add_argument("--self-test-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    training = load_wav(args.training)
    profile = train_profile(training.mono, training.sample_rate, args.training_hits)
    min_gap = args.min_gap_ms / 1000.0
    self_test = validate_profile(training.mono, training.sample_rate, profile, args.training_hits, min_gap)
    print("Training onsets (s):", ", ".join(f"{x:.4f}" for x in profile.training_onsets))
    print(f"Distance threshold: {profile.distance_threshold:.4f}")
    print(f"Prominence threshold: {profile.prominence_threshold:.4f} x MAD")
    print(f"Self-test: PASS ({len(self_test)}/{args.training_hits}, no extras)")
    if args.self_test_only:
        return

    target = load_wav(args.input)
    if target.sample_rate != training.sample_rate:
        raise RuntimeError("Training and target sample rates must match")
    candidates = score_candidates(target.mono, target.sample_rate, profile, min_gap)
    base = acoustic_detect(candidates, profile)
    detections = sequence_refine(base, candidates, profile)
    recovered = sum(d.source == "sequence_recovery" for d in detections)
    print(f"Acoustic detections: {len(base)}")
    print(f"Sequence-refined detections: {len(detections)}")
    print(f"Recovered weak hits: {recovered}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "detections_v3.csv"
    review_path = args.output_dir / "review_clicks_v3.wav"
    hits_path = args.output_dir / "detected_hits_v3.wav"
    write_csv(csv_path, detections)
    write_review_audio(review_path, target, detections)
    write_extracted_hits(hits_path, target, detections)
    print(f"CSV: {csv_path}")
    print(f"Review audio: {review_path}")
    print(f"Extracted hits: {hits_path}")


if __name__ == "__main__":
    main()
