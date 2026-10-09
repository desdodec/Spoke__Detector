#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import find_peaks

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import detector_v4 as d4
import detector_v5 as d5

WAV = Path('audio/Spoke test new.wav')
OUT = Path('output_spoke_pairs')
OUT.mkdir(exist_ok=True)


def spectral_features(x: np.ndarray, sr: int, t: float, win_s: float = 0.035) -> dict:
    n = max(256, int(win_s * sr))
    c = int(round(t * sr))
    a = max(0, c - n // 4)
    b = min(len(x), a + n)
    seg = np.zeros(n)
    raw = x[a:b]
    seg[:len(raw)] = raw
    seg *= np.hanning(n)
    mag = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mask = (freqs >= 250) & (freqs <= 10000)
    m = mag[mask]
    f = freqs[mask]
    if m.size == 0 or m.sum() <= 1e-12:
        return {'peak_hz': 0.0, 'centroid_hz': 0.0, 'low_high_logratio': 0.0}
    peak = float(f[np.argmax(m)])
    centroid = float(np.sum(f * m) / np.sum(m))
    low = float(np.sum(m[(f >= 250) & (f < 1400)])) + 1e-12
    high = float(np.sum(m[(f >= 1400) & (f <= 6000)])) + 1e-12
    return {'peak_hz': peak, 'centroid_hz': centroid, 'low_high_logratio': float(np.log(low / high))}


def split_active_sections(times: np.ndarray) -> list[tuple[float, float]]:
    if len(times) == 0:
        return []
    gaps = np.diff(times)
    # Long gaps separate deliberate wheel-spin sessions.
    cut = np.where(gaps > 1.0)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(times) - 1]
    sections = [(float(times[s]), float(times[e])) for s, e in zip(starts, ends) if e - s + 1 >= 8]
    return sections


def two_cluster_1d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Tiny deterministic 1-D k-means, no sklearn dependency.
    c = np.quantile(values, [0.3, 0.7]).astype(float)
    labels = np.zeros(len(values), dtype=int)
    for _ in range(50):
        d = np.abs(values[:, None] - c[None, :])
        new_labels = np.argmin(d, axis=1)
        new_c = np.array([
            values[new_labels == k].mean() if np.any(new_labels == k) else c[k]
            for k in range(2)
        ])
        if np.array_equal(new_labels, labels) and np.allclose(new_c, c):
            break
        labels, c = new_labels, new_c
    order = np.argsort(c)
    remap = np.zeros(2, dtype=int)
    remap[order] = [0, 1]
    return remap[labels], c[order]


def alternation_stats(labels: np.ndarray) -> dict:
    if len(labels) < 2:
        return {'n': int(len(labels)), 'alternation_rate': None, 'same_same': 0, 'transitions': 0}
    same = int(np.sum(labels[1:] == labels[:-1]))
    transitions = len(labels) - 1
    return {
        'n': int(len(labels)),
        'alternation_rate': float(1.0 - same / transitions),
        'same_same': same,
        'transitions': int(transitions),
    }


audio = d4.load_wav(WAV)
analysis = d4.highpass_analysis(audio.mono, audio.sample_rate, 250.0)
times, strengths, prominences = d4.all_onset_peaks(analysis, audio.sample_rate, 0.025)
max_strength = float(np.max(strengths)) + 1e-12
cands = [d4.Candidate(float(t), 0.0, 1.0, float(s/max_strength), float(p)) for t,s,p in zip(times,strengths,prominences)]
# Use a conservative threshold for this clean lab-like sample; adaptive if stronger.
adaptive = d5.adaptive_prominence_threshold(cands, 4.0)
threshold = max(8.0, adaptive)
base = d5.adaptive_detect(cands, threshold)
dets = d5.refine_sequence(base)
det_times = np.array([d.time_seconds for d in dets], dtype=float)

rows = []
for d in dets:
    feat = spectral_features(analysis, audio.sample_rate, d.time_seconds)
    rows.append({
        'time': float(d.time_seconds),
        'prominence': float(d.normalized_prominence),
        **feat,
    })

# Try three different 1-D descriptors; choose each independently and report alternation.
report = {
    'duration_s': float(len(audio.mono)/audio.sample_rate),
    'candidate_count': int(len(cands)),
    'adaptive_threshold': float(adaptive),
    'used_threshold': float(threshold),
    'detection_count': int(len(dets)),
    'sections': [],
    'global': {},
}

sections = split_active_sections(det_times)
if len(sections) < 3:
    # fallback: split on the two largest detection gaps
    gaps = np.diff(det_times)
    if len(gaps) >= 2:
        idx = np.sort(np.argsort(gaps)[-2:])
        bounds = [0, idx[0]+1, idx[1]+1, len(det_times)]
        sections = [(float(det_times[bounds[i]]), float(det_times[bounds[i+1]-1])) for i in range(3)]

for feature in ['peak_hz', 'centroid_hz', 'low_high_logratio']:
    vals = np.array([r[feature] for r in rows], dtype=float)
    labels, centers = two_cluster_1d(vals)
    st = alternation_stats(labels)
    st['cluster_centers'] = [float(x) for x in centers]
    report['global'][feature] = st
    for r, lab in zip(rows, labels):
        r[f'{feature}_class'] = int(lab)

for si, (start, end) in enumerate(sections, 1):
    mask = (det_times >= start) & (det_times <= end)
    indices = np.where(mask)[0]
    sec = {
        'section': si,
        'start_s': start,
        'end_s': end,
        'n': int(len(indices)),
        'median_interval_s': float(np.median(np.diff(det_times[indices]))) if len(indices) > 1 else None,
        'features': {},
    }
    for feature in ['peak_hz', 'centroid_hz', 'low_high_logratio']:
        labs = np.array([rows[i][f'{feature}_class'] for i in indices], dtype=int)
        sec['features'][feature] = alternation_stats(labs)
    report['sections'].append(sec)

# Pairwise diagnostic: adjacent hit frequency ratios and sign changes.
if len(rows) >= 2:
    peaks = np.array([r['peak_hz'] for r in rows])
    cent = np.array([r['centroid_hz'] for r in rows])
    report['adjacent_peak_ratio_median'] = float(np.median(np.maximum(peaks[1:], peaks[:-1]) / np.maximum(np.minimum(peaks[1:], peaks[:-1]), 1e-9)))
    report['adjacent_centroid_absdiff_median_hz'] = float(np.median(np.abs(np.diff(cent))))

(OUT/'spoke_pair_report.json').write_text(json.dumps(report, indent=2))

import csv
with (OUT/'spoke_pair_events.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['time'])
    w.writeheader(); w.writerows(rows)

print(json.dumps(report, indent=2))
