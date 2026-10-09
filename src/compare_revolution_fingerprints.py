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
OUT = Path('output_revolution_compare')
OUT.mkdir(exist_ok=True)

# Three visually/sequentially distinct spin regions from the earlier analysis.
# Windows are deliberately broad; only accepted transient events inside them are used.
WINDOWS = [
    (2.5, 9.0),
    (26.0, 34.5),
    (45.0, 51.7),
]


def signature(signal: np.ndarray, sr: int, t: float) -> np.ndarray:
    half = int(round(0.018 * sr))
    c = int(round(t * sr))
    x = signal[max(0,c-half):min(len(signal),c+half)]
    if len(x) < 64:
        return np.zeros(20, dtype=float)
    spec = np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
    f = np.fft.rfftfreq(len(x), 1/sr)
    edges = np.geomspace(250, min(11000, sr*0.45), 21)
    vals = []
    for a,b in zip(edges[:-1],edges[1:]):
        m=(f>=a)&(f<b)
        vals.append(np.log1p(float(spec[m].sum())))
    v=np.asarray(vals,float)
    v-=v.mean()
    n=np.linalg.norm(v)
    return v/n if n>1e-12 else v


def seq_for_threshold(cands, threshold, analysis, sr):
    dets=d5.refine_sequence(d5.adaptive_detect(cands,threshold))
    out=[]
    for lo,hi in WINDOWS:
        ds=[d for d in dets if lo<=d.time_seconds<=hi]
        out.append({
            'times': np.asarray([d.time_seconds for d in ds],float),
            'sigs': np.vstack([signature(analysis,sr,d.time_seconds) for d in ds]) if ds else np.empty((0,20)),
        })
    return out


def align_score(A: np.ndarray, B: np.ndarray, gap_penalty: float=0.35):
    # Global sequence alignment. Match score is cosine similarity remapped to [0,1].
    n,m=len(A),len(B)
    dp=np.full((n+1,m+1),-1e9,float)
    matches=np.zeros((n+1,m+1),int)
    dp[0,0]=0
    for i in range(1,n+1): dp[i,0]=dp[i-1,0]-gap_penalty
    for j in range(1,m+1): dp[0,j]=dp[0,j-1]-gap_penalty
    for i in range(1,n+1):
        for j in range(1,m+1):
            sim=(float(np.dot(A[i-1],B[j-1]))+1.0)/2.0
            opts=[
                (dp[i-1,j-1]+sim, matches[i-1,j-1]+1),
                (dp[i-1,j]-gap_penalty, matches[i-1,j]),
                (dp[i,j-1]-gap_penalty, matches[i,j-1]),
            ]
            best=max(opts,key=lambda x:x[0])
            dp[i,j],matches[i,j]=best
    k=max(matches[n,m],1)
    return float(dp[n,m]/k), int(matches[n,m])


def circular_compare(A: np.ndarray, B: np.ndarray):
    # Rotate B through all possible phase offsets and retain best sequence alignment.
    if len(A)==0 or len(B)==0:
        return None
    best=None
    for shift in range(len(B)):
        Br=np.roll(B,shift,axis=0)
        score,matched=align_score(A,Br)
        row={'shift':shift,'score':score,'matched':matched}
        if best is None or score>best['score']:
            best=row
    return best


def null_scores(A: np.ndarray, B: np.ndarray, repeats=200, seed=123):
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(repeats):
        perm=rng.permutation(len(B))
        score,_=align_score(A,B[perm])
        vals.append(score)
    return np.asarray(vals,float)

audio=d4.load_wav(WAV)
analysis=d4.highpass_analysis(audio.mono,audio.sample_rate,250.0)
times,strengths,proms=d4.all_onset_peaks(analysis,audio.sample_rate,0.025)
mx=float(np.max(strengths))+1e-12
cands=[d4.Candidate(float(t),0.0,1.0,float(s/mx),float(p)) for t,s,p in zip(times,strengths,proms)]

report={'windows':WINDOWS,'thresholds':{}}
for threshold in [4.0,5.0,6.0,7.0,8.0,10.0,12.0]:
    seqs=seq_for_threshold(cands,threshold,analysis,audio.sample_rate)
    item={'counts':[len(s['times']) for s in seqs],'pairs':[]}
    for i,j in [(0,1),(0,2),(1,2)]:
        A,B=seqs[i]['sigs'],seqs[j]['sigs']
        best=circular_compare(A,B)
        if best is None:
            continue
        # Compare against random order as a conservative specificity check.
        ns=null_scores(A,np.roll(B,best['shift'],axis=0))
        pct=float(np.mean(ns < best['score']))*100.0
        item['pairs'].append({
            'sections':[i+1,j+1],
            'best_shift':best['shift'],
            'alignment_score':best['score'],
            'matched_events':best['matched'],
            'null_median':float(np.median(ns)),
            'null_p95':float(np.quantile(ns,0.95)),
            'percentile_vs_random_order':pct,
        })
    if item['pairs']:
        item['mean_percentile_vs_random_order']=float(np.mean([p['percentile_vs_random_order'] for p in item['pairs']]))
        item['mean_alignment_score']=float(np.mean([p['alignment_score'] for p in item['pairs']]))
    report['thresholds'][str(threshold)]=item

ranked=sorted(
    [
        {'threshold':float(th),**x}
        for th,x in report['thresholds'].items()
        if 'mean_percentile_vs_random_order' in x
    ],
    key=lambda x:(x['mean_percentile_vs_random_order'],x['mean_alignment_score']),
    reverse=True,
)
report['ranked_thresholds']=ranked

path=OUT/'revolution_fingerprint_comparison.json'
path.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
