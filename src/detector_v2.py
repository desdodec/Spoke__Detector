#!/usr/bin/env python3
"""Robust bicycle spoke-click detector (v2).

Key differences from v1:
- learns a transient-prominence gate from labelled positive clicks
- uses a short refractory period only to suppress duplicate peaks from one impact
- validates the trained profile against the training clip before analysing target audio
- does not use walking speed or expected spoke timing as a detection requirement
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


def all_onset_peaks(
    signal: np.ndarray,
    sample_rate: int,
    min_gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    times, flux = onset_curve(signal, sample_rate)
    _, mad = robust_scale(flux)
    distance_frames = max(1, int(np.ceil(min_gap_seconds / (HOP_SIZE / sample_rate))))
    peaks, _ = find_peaks(flux, distance=distance_frames, prominence=1e-12)
    prominences = peak_prominences(flux, peaks)[0]
    return times[peaks], flux[peaks], prominences / mad, mad


def find_training_onsets(
    signal: np.ndarray,
    sample_rate: int,
    expected_hits: int,
) -> tuple[np.ndarray, np.ndarray]:
    times, flux = onset_curve(signal, sample_rate)
    _, mad = robust_scale(flux)
    distance_frames = max(1, int(np.ceil(0.080 / (HOP_SIZE / sample_rate))))
    peaks, _ = find_peaks(flux, distance=distance_frames, prominence=max(0.25 * mad, 1e-12))
    if len(peaks) < expected_hits:
        raise RuntimeError(
            f"Training file yielded only {len(peaks)} plausible onsets; expected {expected_hits}."
        )

    prominences = peak_prominences(flux, peaks)[0]
    strongest = peaks[np.argsort(flux[peaks])[-expected_hits:]]
    strongest.sort()
    prominence_by_peak = {int(p): float(pr) for p, pr in zip(peaks, prominences)}
    selected_prominence = np.asarray(
        [prominence_by_peak[int(p)] / mad for p in strongest], dtype=np.float64
    )
    return times[strongest], selected_prominence


def feature_vector(
    signal: np.ndarray,
    sample_rate: int,
    onset_seconds: float,
    window_seconds: float = HIT_WINDOW_SECONDS,
) -> np.ndarray:
    start = max(0, int(round((onset_seconds - 0.005) * sample_rate)))
    length = int(round(window_seconds * sample_rate))
    stop = min(len(signal), start + length)
    segment = signal[start:stop].astype(np.float64, copy=True)
    if len(segment) < length:
        segment = np.pad(segment, (0, length - len(segment)))

    segment -= np.mean(segment)
    segment /= np.sqrt(np.mean(segment * segment)) + 1e-9

    power = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
    frequencies = np.fft.rfftfreq(len(segment), d=1.0 / sample_rate)
    low_hz, high_hz = 250.0, min(10000.0, sample_rate * 0.45)
    useful = (frequencies >= low_hz) & (frequencies <= high_hz)
    useful_power = float(power[useful].sum()) + 1e-12

    bands = []
    edges = np.geomspace(low_hz, high_hz, 17)
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

    return np.asarray(
        bands + envelope + [zcr, np.log(centroid + 1.0), np.log(bandwidth + 1.0)],
        dtype=np.float64,
    )


def train_profile(signal: np.ndarray, sample_rate: int, expected_hits: int) -> Profile:
    onsets, training_prominence = find_training_onsets(signal, sample_rate, expected_hits)
    features = np.stack([feature_vector(signal, sample_rate, onset) for onset in onsets])
    feature_scale = features.std(axis=0) + 0.15

    leave_one_out = []
    for index, vector in enumerate(features):
        others = np.delete(features, index, axis=0)
        distances = np.sqrt(np.mean(((others - vector) / feature_scale) ** 2, axis=1))
        leave_one_out.append(float(np.min(distances)))

    distance_threshold = max(1.20, max(leave_one_out) * 1.25)
    prominence_threshold = max(4.0, float(np.min(training_prominence) * 0.75))

    return Profile(
        training_onsets=onsets,
        features=features,
        feature_scale=feature_scale,
        distance_threshold=distance_threshold,
        prominence_threshold=prominence_threshold,
    )


def detect(
    signal: np.ndarray,
    sample_rate: int,
    profile: Profile,
    min_gap_seconds: float,
) -> list[Detection]:
    times, strengths, normalized_prominence, _ = all_onset_peaks(
        signal, sample_rate, min_gap_seconds
    )
    keep = normalized_prominence >= profile.prominence_threshold
    times = times[keep]
    strengths = strengths[keep]
    normalized_prominence = normalized_prominence[keep]
    if len(times) == 0:
        return []

    max_strength = float(np.max(strengths)) + 1e-12
    detections: list[Detection] = []
    for onset, strength, prominence in zip(times, strengths, normalized_prominence):
        vector = feature_vector(signal, sample_rate, float(onset))
        distances = np.sqrt(
            np.mean(((profile.features - vector) / profile.feature_scale) ** 2, axis=1)
        )
        distance = float(np.min(distances))
        if distance <= profile.distance_threshold:
            detections.append(
                Detection(
                    time_seconds=float(onset),
                    distance=distance,
                    confidence=float(np.exp(-distance / profile.distance_threshold)),
                    onset_strength=float(strength / max_strength),
                    normalized_prominence=float(prominence),
                )
            )
    return detections


def validate_profile(
    signal: np.ndarray,
    sample_rate: int,
    profile: Profile,
    expected_hits: int,
    min_gap_seconds: float,
) -> list[Detection]:
    detections = detect(signal, sample_rate, profile, min_gap_seconds)
    if len(detections) != expected_hits:
        raise RuntimeError(
            f"Profile self-test failed: expected {expected_hits} hits but detected {len(detections)}. "
            "Do not use this profile on long audio until the training sample passes exactly."
        )
    return detections


def timestamp_hms(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"


def write_csv(path: Path, detections: Iterable[Detection]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "hit_index", "timestamp_seconds", "timestamp_hms", "confidence",
            "profile_distance", "onset_strength", "normalized_prominence"
        ])
        for index, hit in enumerate(detections, start=1):
            writer.writerow([
                index,
                f"{hit.time_seconds:.6f}",
                timestamp_hms(hit.time_seconds),
                f"{hit.confidence:.6f}",
                f"{hit.distance:.6f}",
                f"{hit.onset_strength:.6f}",
                f"{hit.normalized_prominence:.6f}",
            ])


def marker_click(sample_rate: int, duration: float = 0.025) -> np.ndarray:
    count = max(1, int(round(duration * sample_rate)))
    t = np.arange(count, dtype=np.float64) / sample_rate
    click = (
        np.sin(2.0 * np.pi * 2800.0 * t)
        + 0.65 * np.sin(2.0 * np.pi * 4200.0 * t)
    ) * np.exp(-t * 90.0)
    return (click / (np.max(np.abs(click)) + 1e-12)).astype(np.float32)


def float_to_int16(signal: np.ndarray) -> np.ndarray:
    return np.round(np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)


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
            review[start:stop] += click[: stop - start, None] * 0.45
    peak = float(np.max(np.abs(review))) if review.size else 0.0
    if peak > 0.99:
        review *= 0.99 / peak
    out = review[:, 0] if audio.original.ndim == 1 else review
    wavfile.write(path, audio.sample_rate, float_to_int16(out))


def write_extracted_hits(
    path: Path,
    audio: Audio,
    detections: list[Detection],
    pre_seconds: float = 0.012,
    post_seconds: float = 0.070,
    gap_seconds: float = 0.025,
) -> None:
    pre = int(round(pre_seconds * audio.sample_rate))
    post = int(round(post_seconds * audio.sample_rate))
    gap = np.zeros(int(round(gap_seconds * audio.sample_rate)), dtype=np.float32)
    pieces = []
    for hit in detections:
        center = int(round(hit.time_seconds * audio.sample_rate))
        excerpt = audio.mono[max(0, center - pre):min(len(audio.mono), center + post)]
        if excerpt.size:
            pieces.extend([excerpt, gap])
    combined = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 0:
        combined *= 0.95 / peak
    wavfile.write(path, audio.sample_rate, float_to_int16(combined))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile and detect bicycle spoke clicks.")
    parser.add_argument("--training", type=Path, default=Path("short_sample/7_clicks.wav"))
    parser.add_argument("--input", type=Path, default=Path("audio/Bike_test.wav"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--training-hits", type=int, default=7)
    parser.add_argument("--min-gap-ms", type=float, default=DEFAULT_MIN_GAP_MS)
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = load_wav(args.training)
    profile = train_profile(training.mono, training.sample_rate, args.training_hits)
    min_gap_seconds = args.min_gap_ms / 1000.0

    self_test = validate_profile(
        training.mono,
        training.sample_rate,
        profile,
        args.training_hits,
        min_gap_seconds,
    )

    print("Training onsets (s):", ", ".join(f"{x:.4f}" for x in profile.training_onsets))
    print(f"Distance threshold: {profile.distance_threshold:.4f}")
    print(f"Prominence threshold: {profile.prominence_threshold:.4f} x MAD")
    print(f"Self-test: PASS ({len(self_test)}/{args.training_hits}, no extras)")

    if args.self_test_only:
        return

    target = load_wav(args.input)
    if training.sample_rate != target.sample_rate:
        raise RuntimeError(
            f"Sample-rate mismatch: training={training.sample_rate} Hz, input={target.sample_rate} Hz."
        )

    detections = detect(target.mono, target.sample_rate, profile, min_gap_seconds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "detections_v2.csv"
    review_path = args.output_dir / "review_clicks_v2.wav"
    hits_path = args.output_dir / "detected_hits_v2.wav"
    write_csv(csv_path, detections)
    write_review_audio(review_path, target, detections)
    write_extracted_hits(hits_path, target, detections)

    print(f"Detected hits: {len(detections)}")
    print(f"CSV: {csv_path}")
    print(f"Review audio: {review_path}")
    print(f"Extracted hits: {hits_path}")


if __name__ == "__main__":
    main()
