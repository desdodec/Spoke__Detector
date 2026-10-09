#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import detector_v4 as d4
import detector_v5 as d5

WAV = Path('audio/Spoke test new.wav')
OUT = Path('output_wheel_fingerprint')
OUT.mkdir(exist_ok=True)


def band_signature(signal: np.ndarray, sr: int, t: float) -> np.ndarray:
    # 35 ms window around impact, represented by log energy in broad bands.
    half = int(round(0.0175 * sr))
    c = int(round(t * sr))
    lo = max(0, c - half)
    hi = min(len(signal), c + half)
    x = signal[lo:hi]
    if len(x) < 64:
        return np.zeros(18, dtype=float)
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    edges = np.geomspace(250.0, min(10000.0, sr * 0.45), 19)
    vals = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (freqs >= a) & (freqs < b)
        vals.append(np.log1p(float(spec[m].sum())))
    v = np.asarray(vals, dtype=float)
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def continuous_sections(times: np.ndarray) -> list[tuple[int, int]]:
    if len(times) == 0:
        return []
    if len(times) == 1:
        return [(0, 1)]
    dt = np.diff(times)
    # Large pauses separate the deliberate spin sessions. Use an absolute floor
    # so slow sections are not fragmented merely because speed changes.
    typical = float(np.median(dt[dt < 1.0])) if np.any(dt < 1.0) else float(np.median(dt))
    gap = max(1.25, typical * 6.0)
    cuts = np.where(dt > gap)[0]
    starts = [0] + [int(i + 1) for i in cuts]
    ends = [int(i + 1) for i in cuts] + [len(times)]
    return [(a, b) for a, b in zip(starts, ends) if b - a >= 8]


def lag_similarity(sigs: np.ndarray, sections: list[tuple[int, int]], lag: int) -> tuple[float | None, int]:
    vals = []
    for a, b in sections:
        if b - a <= lag:
            continue
        # cosine similarity because signatures are unit-normalized
        vals.extend(np.sum(sigs[a:b-lag] * sigs[a+lag:b], axis=1).tolist())
    if not vals:
        return None, 0
    return float(np.median(vals)), len(vals)


def timing_repeatability(times: np.ndarray, sections: list[tuple[int, int]], lag: int) -> tuple[float | None, int]:
    # Compare local intervals separated by one candidate wheel cycle.
    dt = np.diff(times)
    vals = []
    for a, b in sections:
        # interval indexes run a..b-2
        n = (b - a - 1) - lag
        if n <= 0:
            continue
        x = dt[a:a+n]
        y = dt[a+lag:a+lag+n]
        denom = np.maximum((x + y) * 0.5, 1e-9)
        vals.extend((np.abs(x-y) / denom).tolist())
    if not vals:
        return None, 0
    return float(np.median(vals)), len(vals)


audio = d4.load_wav(WAV)
analysis = d4.highpass_analysis(audio.mono, audio.sample_rate, 250.0)
times, strengths, prominences = d4.all_onset_peaks(analysis, audio.sample_rate, 0.025)
max_strength = float(np.max(strengths)) + 1e-12
all_candidates = [
    d4.Candidate(float(t), 0.0, 1.0, float(s/max_strength), float(p))
    for t, s, p in zip(times, strengths, prominences)
]

results = {
    'duration_s': len(audio.mono) / audio.sample_rate,
    'raw_candidate_count': len(all_candidates),
    'thresholds': {},
}

for threshold in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]:
    base = d5.adaptive_detect(all_candidates, threshold)
    dets = d5.refine_sequence(base)
    t = np.asarray([d.time_seconds for d in dets], dtype=float)
    secs = continuous_sections(t)
    sigs = np.vstack([band_signature(analysis, audio.sample_rate, x) for x in t]) if len(t) else np.empty((0,18))

    lag_rows = []
    for lag in range(8, 49):
        acoustic, na = lag_similarity(sigs, secs, lag)
        timing, nt = timing_repeatability(t, secs, lag)
        if acoustic is None:
            continue
        lag_rows.append({
            'lag_hits': lag,
            'acoustic_similarity_median': acoustic,
            'acoustic_pairs': na,
            'timing_relative_error_median': timing,
            'timing_pairs': nt,
        })

    # Rank mostly by acoustic repetition. Timing is included as supporting evidence,
    # but speed changes mean it must not dominate.
    ranked = sorted(
        lag_rows,
        key=lambda r: (r['acoustic_similarity_median'], -(r['timing_relative_error_median'] if r['timing_relative_error_median'] is not None else 99)),
        reverse=True,
    )
    results['thresholds'][str(threshold)] = {
        'detection_count': len(t),
        'sections': [
            {
                'start_s': float(t[a]),
                'end_s': float(t[b-1]),
                'hits': int(b-a),
                'median_interval_s': float(np.median(np.diff(t[a:b]))) if b-a >= 2 else None,
            }
            for a,b in secs
        ],
        'best_lags': ranked[:10],
    }

# Consensus: collect top-5 lags at each threshold and count support.
support = {}
for th, item in results['thresholds'].items():
    for rank, row in enumerate(item['best_lags'][:5], 1):
        lag = int(row['lag_hits'])
        d = support.setdefault(lag, {'threshold_support':0, 'weighted_score':0.0, 'examples':[]})
        d['threshold_support'] += 1
        d['weighted_score'] += (6-rank) * float(row['acoustic_similarity_median'])
        d['examples'].append({'threshold': float(th), 'rank': rank, 'similarity': row['acoustic_similarity_median']})
results['consensus_lags'] = [
    {'lag_hits': lag, **data}
    for lag, data in sorted(support.items(), key=lambda kv: (kv[1]['threshold_support'], kv[1]['weighted_score']), reverse=True)
]

path = OUT / 'wheel_fingerprint_analysis.json'
path.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
