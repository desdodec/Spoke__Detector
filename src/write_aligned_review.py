#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def _to_float(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), abs(info.max))
        return data.astype(np.float64) / float(scale)
    return data.astype(np.float64)


def _from_float(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scale = max(abs(info.min), abs(info.max))
        return np.round(x * scale).astype(dtype)
    return x.astype(dtype)


def read_times(csv_path: Path) -> list[float]:
    with csv_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return [float(r['timestamp_seconds']) for r in rows]


def write_aligned(input_wav: Path, detections_csv: Path, output_wav: Path) -> None:
    sr, raw = wavfile.read(input_wav)
    original_dtype = raw.dtype
    x = _to_float(raw)
    y = x.copy()
    channels = 1 if x.ndim == 1 else x.shape[1]

    # Human-audit marker: intentionally obvious, while preserving the original
    # sample timeline exactly. The marker STARTS at the detector timestamp.
    # No filtering, trimming, resampling, padding, time warping or channel conversion.
    #
    # A short 1.6 kHz beep is much easier to hear than the previous 8 kHz tick,
    # particularly against sharp spoke impacts and on ordinary laptop/headphone playback.
    marker_seconds = 0.030
    n = max(1, int(round(marker_seconds * sr)))
    t = np.arange(n, dtype=np.float64) / sr

    # 2 ms attack/release to avoid introducing an unrelated hard digital click.
    env = np.ones(n, dtype=np.float64)
    edge = max(1, int(round(0.002 * sr)))
    ramp = np.linspace(0.0, 1.0, edge, endpoint=False)
    env[:edge] = ramp
    env[-edge:] = ramp[::-1]

    # Strong but not full-scale marker. During the marker window the source is
    # gently ducked, making every accepted detection unmistakable without moving it.
    marker = 0.55 * np.sin(2 * np.pi * 1600.0 * t) * env
    source_gain_during_marker = 0.45

    times = read_times(detections_csv)
    for sec in times:
        start = int(round(sec * sr))
        if start < 0 or start >= len(y):
            continue
        stop = min(len(y), start + n)
        m = marker[: stop - start]
        if channels == 1:
            y[start:stop] = source_gain_during_marker * y[start:stop] + m
        else:
            y[start:stop, :] = (
                source_gain_during_marker * y[start:stop, :] + m[:, None]
            )

    # Preserve exact frame count, sample rate, channel count and original dtype.
    wavfile.write(output_wav, sr, _from_float(y, original_dtype))

    # Hard invariants for human audit alignment.
    sr2, check = wavfile.read(output_wav)
    assert sr2 == sr
    assert check.shape == raw.shape
    assert check.dtype == raw.dtype
    print(
        f'{output_wav}: sr={sr}, shape={raw.shape}, dtype={raw.dtype}, '
        f'markers={len(times)}, marker=30ms@1600Hz'
    )


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('usage: write_aligned_review.py INPUT.wav DETECTIONS.csv OUTPUT.wav')
    write_aligned(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
