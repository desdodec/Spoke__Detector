#!/usr/bin/env python3
"""Acoustic transient review engine for Spoke Detector.

This module deliberately ignores spoke count, walking rhythm and expected timing.
It detects short acoustic events from waveform energy, refines timestamps at sample
resolution, and supports human-reviewed reference selections.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, find_peaks, sosfiltfilt

DEFAULT_HIGHPASS_HZ = 1200.0
DEFAULT_THRESHOLD_DBFS = -55.0
DEFAULT_MIN_SEPARATION_MS = 40.0


@dataclass
class AudioData:
    path: Path
    sample_rate: int
    mono: np.ndarray

    @property
    def duration_s(self) -> float:
        return float(len(self.mono) / self.sample_rate)


@dataclass
class Event:
    timestamp_seconds: float
    score_dbfs: float
    peak_amplitude: float
    event_energy: float
    source: str = "automatic"


def _to_float(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), abs(info.max))
        return data.astype(np.float64) / float(scale)
    return data.astype(np.float64)


def load_wav(path: Path) -> AudioData:
    path = Path(path)
    sample_rate, raw = wavfile.read(path)
    x = _to_float(raw)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    mono = np.nan_to_num(mono, copy=False)
    return AudioData(path=path, sample_rate=int(sample_rate), mono=mono.astype(np.float64))


def _moving_average(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples == 1:
        return values.copy()
    kernel = np.ones(samples, dtype=np.float64) / samples
    return np.convolve(values, kernel, mode="same")


def transient_curve(audio: AudioData, highpass_hz: float = DEFAULT_HIGHPASS_HZ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return high-passed waveform, transient novelty and dBFS novelty score.

    The score is referenced to digital full scale, so a threshold can be held
    fixed across recordings acquired with a standardised gain/setup.
    """
    x = audio.mono
    nyq = audio.sample_rate * 0.5
    cutoff = max(20.0, min(float(highpass_hz), nyq * 0.90))
    sos = butter(4, cutoff / nyq, btype="highpass", output="sos")
    filtered = sosfiltfilt(sos, x)
    envelope = np.abs(filtered)
    fast = _moving_average(envelope, round(0.0015 * audio.sample_rate))
    slow = _moving_average(envelope, round(0.020 * audio.sample_rate))
    novelty = np.maximum(fast - slow, 0.0)
    score_dbfs = 20.0 * np.log10(np.maximum(novelty, 1e-12))
    return filtered, novelty, score_dbfs


def _refine_sample(filtered: np.ndarray, sample_rate: int, approx_index: int, radius_ms: float = 7.0) -> int:
    radius = max(1, int(round(radius_ms * sample_rate / 1000.0)))
    lo = max(0, approx_index - radius)
    hi = min(len(filtered), approx_index + radius + 1)
    if hi <= lo:
        return int(approx_index)
    return int(lo + int(np.argmax(np.abs(filtered[lo:hi]))))


def _event_energy(filtered: np.ndarray, sample_rate: int, center: int, half_ms: float = 6.0) -> float:
    half = max(1, int(round(half_ms * sample_rate / 1000.0)))
    lo = max(0, center - half)
    hi = min(len(filtered), center + half + 1)
    seg = filtered[lo:hi]
    return float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0


def detect_events(
    audio: AudioData,
    threshold_dbfs: float = DEFAULT_THRESHOLD_DBFS,
    min_separation_ms: float = DEFAULT_MIN_SEPARATION_MS,
    highpass_hz: float = DEFAULT_HIGHPASS_HZ,
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[Event]:
    filtered, novelty, score_dbfs = transient_curve(audio, highpass_hz)
    sr = audio.sample_rate
    start = 0 if start_s is None else max(0, int(round(start_s * sr)))
    end = len(audio.mono) if end_s is None else min(len(audio.mono), int(round(end_s * sr)))
    if end <= start:
        return []
    distance = max(1, int(round(min_separation_ms * sr / 1000.0)))
    peaks, _ = find_peaks(
        novelty[start:end],
        distance=distance,
        height=10 ** (float(threshold_dbfs) / 20.0),
    )
    approx = peaks + start
    out: list[Event] = []
    used: list[int] = []
    for p in approx:
        refined = _refine_sample(filtered, sr, int(p))
        if used and min(abs(refined - u) for u in used) < max(1, int(round(0.004 * sr))):
            continue
        used.append(refined)
        out.append(Event(
            timestamp_seconds=float(refined / sr),
            score_dbfs=float(score_dbfs[int(p)]),
            peak_amplitude=float(abs(filtered[refined])),
            event_energy=_event_energy(filtered, sr, refined),
            source="automatic",
        ))
    return sorted(out, key=lambda e: e.timestamp_seconds)


def fit_threshold_for_count(
    audio: AudioData,
    expected_count: int,
    start_s: float,
    end_s: float,
    min_separation_ms: float = DEFAULT_MIN_SEPARATION_MS,
    highpass_hz: float = DEFAULT_HIGHPASS_HZ,
) -> tuple[float, list[Event]]:
    """Choose a threshold that yields the requested count when possible."""
    expected_count = max(0, int(expected_count))
    _, novelty, score_dbfs = transient_curve(audio, highpass_hz)
    sr = audio.sample_rate
    start = max(0, int(round(start_s * sr)))
    end = min(len(audio.mono), int(round(end_s * sr)))
    if end <= start or expected_count == 0:
        return 0.0, []
    distance = max(1, int(round(min_separation_ms * sr / 1000.0)))
    peaks, _ = find_peaks(novelty[start:end], distance=distance)
    peaks = peaks + start
    if len(peaks) == 0:
        return DEFAULT_THRESHOLD_DBFS, []
    heights = np.asarray([score_dbfs[p] for p in peaks], dtype=float)
    order = np.argsort(heights)[::-1]
    if expected_count >= len(order):
        threshold = float(np.min(heights) - 0.01)
    else:
        selected_floor = float(heights[order[expected_count - 1]])
        next_height = float(heights[order[expected_count]])
        threshold = float((selected_floor + next_height) / 2.0)
    events = detect_events(audio, threshold, min_separation_ms, highpass_hz, start_s, end_s)
    if len(events) != expected_count:
        best_diff = abs(len(events) - expected_count)
        best_threshold, best_events = threshold, events
        for h in sorted(set(float(v) for v in heights), reverse=True):
            th = h - 1e-6
            ev = detect_events(audio, th, min_separation_ms, highpass_hz, start_s, end_s)
            diff = abs(len(ev) - expected_count)
            if diff < best_diff or (diff == best_diff and th > best_threshold):
                best_diff, best_threshold, best_events = diff, th, ev
            if diff == 0:
                break
        threshold, events = best_threshold, best_events
    return float(threshold), events


def export_events_csv(path: Path, events: Iterable[Event]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(list(events), key=lambda e: e.timestamp_seconds)
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["click_index", "timestamp_seconds", "score_dbfs", "peak_amplitude", "event_energy", "source"])
        for i, e in enumerate(rows, 1):
            w.writerow([i, f"{e.timestamp_seconds:.9f}", f"{e.score_dbfs:.3f}", f"{e.peak_amplitude:.8f}", f"{e.event_energy:.8f}", e.source])
    return path


def append_reference(
    path: Path,
    audio: AudioData,
    start_s: float,
    end_s: float,
    expected_count: int,
    events: Iterable[Event],
    threshold_dbfs: float,
    min_separation_ms: float,
    highpass_hz: float,
) -> Path:
    """Append one human-reviewed selection to a JSONL calibration library."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wav": str(audio.path.resolve()),
        "start_s": float(start_s),
        "end_s": float(end_s),
        "expected_count": int(expected_count),
        "threshold_dbfs": float(threshold_dbfs),
        "min_separation_ms": float(min_separation_ms),
        "highpass_hz": float(highpass_hz),
        "events": [asdict(e) for e in sorted(events, key=lambda x: x.timestamp_seconds)],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Detect short acoustic events in a WAV without spoke/gait timing assumptions.")
    ap.add_argument("wav", type=Path)
    ap.add_argument("--threshold-dbfs", type=float, default=DEFAULT_THRESHOLD_DBFS)
    ap.add_argument("--min-separation-ms", type=float, default=DEFAULT_MIN_SEPARATION_MS)
    ap.add_argument("--highpass-hz", type=float, default=DEFAULT_HIGHPASS_HZ)
    ap.add_argument("--out", type=Path, default=Path("clicks.csv"))
    args = ap.parse_args()
    audio = load_wav(args.wav)
    events = detect_events(audio, args.threshold_dbfs, args.min_separation_ms, args.highpass_hz)
    export_events_csv(args.out, events)
    print(f"{len(events)} events -> {args.out}")
