#!/usr/bin/env python3
"""Spoke click detector.

Learns an acoustic profile from a short WAV containing a known number of
spoke/cable-tie impacts, then detects similar transients in a longer WAV.

Outputs:
  - detections.csv
  - review_clicks.wav (original audio + synthetic marker clicks)
  - detected_hits.wav (accepted hit excerpts concatenated with short gaps)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, stft


FRAME_SIZE = 1024
HOP_SIZE = 128
DEFAULT_HIT_WINDOW_SECONDS = 0.060
DEFAULT_MIN_GAP_SECONDS = 0.035


@dataclass
class Audio:
    sample_rate: int
    mono: np.ndarray
    original: np.ndarray


@dataclass
class Detection:
    time_seconds: float
    distance: float
    confidence: float
    onset_strength: float


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
    return Audio(sample_rate=sample_rate, mono=mono.astype(np.float32), original=original)


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
    positive_change = np.maximum(np.diff(magnitude, axis=1), 0.0)
    flux = positive_change.sum(axis=0)
    return times[1:], flux


def find_training_onsets(
    signal: np.ndarray,
    sample_rate: int,
    expected_hits: int,
) -> np.ndarray:
    times, flux = onset_curve(signal, sample_rate)
    min_distance_frames = max(1, int(0.080 / (HOP_SIZE / sample_rate)))
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median))) + 1e-12

    peaks, _ = find_peaks(
        flux,
        distance=min_distance_frames,
        prominence=max(0.25 * mad, 1e-12),
    )
    if len(peaks) < expected_hits:
        raise RuntimeError(
            f"Training file yielded only {len(peaks)} onset candidates; "
            f"expected {expected_hits}. Use a cleaner sample or lower --training-hits."
        )

    strongest = peaks[np.argsort(flux[peaks])[-expected_hits:]]
    strongest.sort()
    return times[strongest]


def candidate_onsets(
    signal: np.ndarray,
    sample_rate: int,
    min_gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    times, flux = onset_curve(signal, sample_rate)
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median))) + 1e-12
    distance_frames = max(1, int(min_gap_seconds / (HOP_SIZE / sample_rate)))

    peaks, _ = find_peaks(
        flux,
        distance=distance_frames,
        prominence=max(2.0 * mad, 1e-12),
    )
    return times[peaks], flux[peaks]


def feature_vector(
    signal: np.ndarray,
    sample_rate: int,
    onset_seconds: float,
    window_seconds: float = DEFAULT_HIT_WINDOW_SECONDS,
) -> np.ndarray:
    pre_seconds = 0.005
    start = max(0, int(round((onset_seconds - pre_seconds) * sample_rate)))
    length = int(round(window_seconds * sample_rate))
    stop = min(len(signal), start + length)
    segment = signal[start:stop].astype(np.float64, copy=True)

    if len(segment) < length:
        segment = np.pad(segment, (0, length - len(segment)))

    segment -= np.mean(segment)
    rms = np.sqrt(np.mean(segment * segment)) + 1e-9
    segment /= rms

    window = np.hanning(len(segment))
    power = np.abs(np.fft.rfft(segment * window)) ** 2
    frequencies = np.fft.rfftfreq(len(segment), d=1.0 / sample_rate)

    low_hz, high_hz = 250.0, min(10000.0, sample_rate * 0.45)
    useful = (frequencies >= low_hz) & (frequencies <= high_hz)
    total_power = float(power[useful].sum()) + 1e-12

    edges = np.geomspace(low_hz, high_hz, 17)
    spectral_bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        ratio = float(power[mask].sum()) / total_power
        spectral_bands.append(np.log(ratio + 1e-8))

    envelope = []
    envelope_bins = 6
    for index in range(envelope_bins):
        a = int(index * len(segment) / envelope_bins)
        b = int((index + 1) * len(segment) / envelope_bins)
        chunk = segment[a:b]
        chunk_rms = np.sqrt(np.mean(chunk * chunk)) + 1e-6
        envelope.append(np.log(chunk_rms))

    zero_crossing_rate = float(np.mean(segment[:-1] * segment[1:] < 0))
    spectral_total = float(power.sum()) + 1e-12
    centroid = float((frequencies * power).sum() / spectral_total)
    bandwidth = float(
        np.sqrt((((frequencies - centroid) ** 2) * power).sum() / spectral_total)
    )

    return np.asarray(
        spectral_bands
        + envelope
        + [zero_crossing_rate, np.log(centroid + 1.0), np.log(bandwidth + 1.0)],
        dtype=np.float64,
    )


def train_profile(
    training_signal: np.ndarray,
    sample_rate: int,
    expected_hits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    onsets = find_training_onsets(training_signal, sample_rate, expected_hits)
    features = np.stack(
        [feature_vector(training_signal, sample_rate, onset) for onset in onsets]
    )

    feature_scale = features.std(axis=0) + 0.15

    leave_one_out = []
    for index, vector in enumerate(features):
        others = np.delete(features, index, axis=0)
        distances = np.sqrt(np.mean(((others - vector) / feature_scale) ** 2, axis=1))
        leave_one_out.append(float(np.min(distances)))

    # High-recall operating point: allow more variation than exists inside
    # the seed sample. This threshold can later be tightened with negatives.
    threshold = max(1.35, max(leave_one_out) * 1.55)
    return onsets, features, feature_scale, threshold


def detect(
    signal: np.ndarray,
    sample_rate: int,
    training_features: np.ndarray,
    feature_scale: np.ndarray,
    threshold: float,
    min_gap_seconds: float,
) -> list[Detection]:
    times, strengths = candidate_onsets(signal, sample_rate, min_gap_seconds)
    if len(times) == 0:
        return []

    max_strength = float(np.max(strengths)) + 1e-12
    detections: list[Detection] = []

    for onset, strength in zip(times, strengths):
        vector = feature_vector(signal, sample_rate, float(onset))
        distances = np.sqrt(
            np.mean(((training_features - vector) / feature_scale) ** 2, axis=1)
        )
        distance = float(np.min(distances))
        if distance <= threshold:
            confidence = float(np.exp(-distance / max(threshold, 1e-9)))
            detections.append(
                Detection(
                    time_seconds=float(onset),
                    distance=distance,
                    confidence=confidence,
                    onset_strength=float(strength / max_strength),
                )
            )

    return detections


def timestamp_hms(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"


def write_csv(path: Path, detections: Iterable[Detection]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "hit_index",
                "timestamp_seconds",
                "timestamp_hms",
                "confidence",
                "profile_distance",
                "onset_strength",
            ]
        )
        for index, hit in enumerate(detections, start=1):
            writer.writerow(
                [
                    index,
                    f"{hit.time_seconds:.6f}",
                    timestamp_hms(hit.time_seconds),
                    f"{hit.confidence:.6f}",
                    f"{hit.distance:.6f}",
                    f"{hit.onset_strength:.6f}",
                ]
            )


def marker_click(sample_rate: int, duration: float = 0.025) -> np.ndarray:
    count = max(1, int(round(duration * sample_rate)))
    t = np.arange(count, dtype=np.float64) / sample_rate
    envelope = np.exp(-t * 90.0)
    click = np.sin(2.0 * np.pi * 2800.0 * t) + 0.65 * np.sin(2.0 * np.pi * 4200.0 * t)
    click *= envelope
    peak = np.max(np.abs(click)) + 1e-12
    return (click / peak).astype(np.float32)


def float_to_int16(signal: np.ndarray) -> np.ndarray:
    signal = np.clip(signal, -1.0, 1.0)
    return np.round(signal * 32767.0).astype(np.int16)


def write_review_audio(path: Path, audio: Audio, detections: list[Detection]) -> None:
    source = audio.original.astype(np.float32, copy=True)
    if source.ndim == 1:
        source = source[:, None]

    click = marker_click(audio.sample_rate)
    review = source.copy()
    review *= 0.78

    for hit in detections:
        start = int(round(hit.time_seconds * audio.sample_rate))
        stop = min(len(review), start + len(click))
        if start >= len(review):
            continue
        marker = click[: stop - start, None] * 0.45
        review[start:stop] += marker

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
    mono = audio.mono
    pre = int(round(pre_seconds * audio.sample_rate))
    post = int(round(post_seconds * audio.sample_rate))
    gap = np.zeros(int(round(gap_seconds * audio.sample_rate)), dtype=np.float32)

    pieces: list[np.ndarray] = []
    for hit in detections:
        center = int(round(hit.time_seconds * audio.sample_rate))
        start = max(0, center - pre)
        stop = min(len(mono), center + post)
        excerpt = mono[start:stop]
        if excerpt.size:
            pieces.extend([excerpt, gap])

    combined = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 0:
        combined = combined * (0.95 / peak)
    wavfile.write(path, audio.sample_rate, float_to_int16(combined))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile and detect bicycle spoke clicks.")
    parser.add_argument(
        "--training",
        type=Path,
        default=Path("short_sample/7_clicks.wav"),
        help="WAV containing a known number of clean spoke hits.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("audio/Bike_test.wav"),
        help="Long WAV to analyse.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for CSV and review WAV files.",
    )
    parser.add_argument(
        "--training-hits",
        type=int,
        default=7,
        help="Known number of genuine hits in the training WAV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the automatically learned profile-distance threshold.",
    )
    parser.add_argument(
        "--min-gap-ms",
        type=float,
        default=35.0,
        help="Minimum gap between onset candidates, in milliseconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training = load_wav(args.training)
    target = load_wav(args.input)

    if training.sample_rate != target.sample_rate:
        raise RuntimeError(
            f"Sample-rate mismatch: training={training.sample_rate} Hz, "
            f"input={target.sample_rate} Hz. Convert both WAVs to the same rate first."
        )

    training_onsets, features, scale, learned_threshold = train_profile(
        training.mono,
        training.sample_rate,
        args.training_hits,
    )
    threshold = args.threshold if args.threshold is not None else learned_threshold

    detections = detect(
        target.mono,
        target.sample_rate,
        features,
        scale,
        threshold,
        args.min_gap_ms / 1000.0,
    )

    csv_path = args.output_dir / "detections.csv"
    review_path = args.output_dir / "review_clicks.wav"
    hits_path = args.output_dir / "detected_hits.wav"

    write_csv(csv_path, detections)
    write_review_audio(review_path, target, detections)
    write_extracted_hits(hits_path, target, detections)

    print("Training onsets (s):", ", ".join(f"{value:.4f}" for value in training_onsets))
    print(f"Profile threshold: {threshold:.4f}")
    print(f"Detected hits: {len(detections)}")
    print(f"CSV: {csv_path}")
    print(f"Review audio: {review_path}")
    print(f"Extracted hits: {hits_path}")


if __name__ == "__main__":
    main()
