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
OUT = Path('output_36_spoke_template')
OUT.mkdir(exist_ok=True)

SPOKES = 36
# Main rotation regions identified in the earlier analyses. These are intentionally
# broad and are not assumed to contain exactly one revolution.
WINDOWS = [
    (2.5, 9.0),
    (26.0, 34.5),
    (45.0, 51.7),
]


def signature(signal: np.ndarray, sr: int, t: float) -> np.ndarray:
    half = int(round(0.018 * sr))
    c = int(round(t * sr))
    x = signal[max(0, c-half):min(len(signal), c+half)]
    if len(x) < 64:
        return np.zeros(24, dtype=float)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1/sr)
    edges = np.geomspace(250, min(12000, sr*0.45), 25)
    vals=[]
    for a,b in zip(edges[:-1], edges[1:]):
        m=(f>=a)&(f<b)
        vals.append(np.log1p(float(spec[m].sum())))
    v=np.asarray(vals,float)
    v-=np.median(v)
    n=np.linalg.norm(v)
    return v/n if n>1e-12 else v


def robust_interval(times: np.ndarray) -> float | None:
    if len(times) < 5:
        return None
    dt=np.diff(times)
    med=float(np.median(dt))
    # Remove obvious doubles/large pauses before recomputing.
    good=dt[(dt>0.45*med)&(dt<1.8*med)]
    return float(np.median(good)) if len(good)>=3 else med


def phase_fit(times: np.ndarray, strengths: np.ndarray, interval: float):
    """Fit events to a 36-slot cyclic clock without forcing one event per slot.

    Searches a phase origin. Each event maps to its nearest integer spoke index.
    Events farther than 0.34 of a spoke interval from a slot are not considered a
    clean geometric match. Multiple events can map to one slot; strongest wins.
    """
    if interval is None or len(times)==0:
        return None
    # Search phase within one spoke interval.
    phases=np.linspace(0.0, interval, 361, endpoint=False)
    best=None
    for phase in phases:
        z=(times-phase)/interval
        k=np.rint(z).astype(int)
        residual=np.abs(z-k)
        good=residual<=0.34
        # Reward many geometrically plausible events, high prominence and small residual.
        score=float(np.sum(good*(1.0-residual/0.34)*(1.0+0.08*np.log1p(strengths))))
        if best is None or score>best['score']:
            best={'phase':float(phase),'score':score,'k':k,'residual':residual,'good':good}
    return best


def build_slots(times, strengths, sigs, interval, fit):
    # slot key is modulo 36; retain strongest geometrically plausible event for a
    # given absolute spoke crossing, then aggregate repeated turns if present.
    by_slot={i:[] for i in range(SPOKES)}
    abs_seen={}
    for idx,(t,s) in enumerate(zip(times,strengths)):
        if not fit['good'][idx]:
            continue
        k=int(fit['k'][idx])
        r=float(fit['residual'][idx])
        prev=abs_seen.get(k)
        quality=float(s)/(1.0+4*r)
        if prev is None or quality>prev[0]:
            abs_seen[k]=(quality,idx)
    for k,(_,idx) in abs_seen.items():
        by_slot[k % SPOKES].append(idx)
    slots=[]
    for s in range(SPOKES):
        ids=by_slot[s]
        if ids:
            V=np.vstack([sigs[i] for i in ids])
            v=np.mean(V,axis=0)
            n=np.linalg.norm(v)
            v=v/n if n>1e-12 else v
            slots.append(v)
        else:
            slots.append(None)
    return slots, abs_seen


def compare_slots(A, B):
    # Search circular spoke offset because the recordings do not share a known valve phase.
    best=None
    for shift in range(SPOKES):
        sims=[]
        matched=[]
        for s in range(SPOKES):
            a=A[s]
            b=B[(s+shift)%SPOKES]
            if a is None or b is None:
                continue
            sim=(float(np.dot(a,b))+1.0)/2.0
            sims.append(sim)
            matched.append(s)
        if not sims:
            continue
        row={'shift':shift,'mean_similarity':float(np.mean(sims)),
             'median_similarity':float(np.median(sims)),'matched_slots':len(sims)}
        # Require useful coverage before letting a tiny overlap win.
        row['score']=row['mean_similarity']*(min(len(sims),24)/24.0)
        if best is None or row['score']>best['score']:
            best=row
    return best


def random_order_null(A,B,repeats=400,seed=20261009):
    rng=np.random.default_rng(seed)
    valid=[x for x in B if x is not None]
    mask=[x is not None for x in B]
    vals=[]
    for _ in range(repeats):
        shuffled=list(valid)
        rng.shuffle(shuffled)
        it=iter(shuffled)
        Br=[next(it) if m else None for m in mask]
        cmp=compare_slots(A,Br)
        if cmp is not None:
            vals.append(cmp['score'])
    return np.asarray(vals,float)


audio=d4.load_wav(WAV)
analysis=d4.highpass_analysis(audio.mono,audio.sample_rate,250.0)
times,strengths,proms=d4.all_onset_peaks(analysis,audio.sample_rate,0.025)
mx=float(np.max(strengths))+1e-12
cands=[d4.Candidate(float(t),0.0,1.0,float(s/mx),float(p)) for t,s,p in zip(times,strengths,proms)]

report={'spokes':SPOKES,'windows':WINDOWS,'thresholds':{}}
for threshold in [4.0,5.0,6.0,7.0,8.0,9.0,10.0,12.0]:
    dets=d5.refine_sequence(d5.adaptive_detect(cands,threshold))
    sections=[]
    for lo,hi in WINDOWS:
        ds=[d for d in dets if lo<=d.time_seconds<=hi]
        tt=np.asarray([d.time_seconds for d in ds],float)
        pp=np.asarray([d.prominence for d in ds],float)
        sig=np.vstack([signature(analysis,audio.sample_rate,t) for t in tt]) if len(tt) else np.empty((0,24))
        interval=robust_interval(tt)
        fit=phase_fit(tt,pp,interval) if interval is not None else None
        slots,abs_seen=build_slots(tt,pp,sig,interval,fit) if fit is not None else ([None]*SPOKES,{})
        # Number of distinct modulo-36 positions represented and number of clean absolute crossings.
        occupied=sum(x is not None for x in slots)
        clean=len(abs_seen)
        rev_span=None
        if abs_seen:
            ks=sorted(abs_seen)
            rev_span=float((ks[-1]-ks[0]+1)/SPOKES)
        sections.append({
            'times':tt,'proms':pp,'slots':slots,
            'summary':{
                'raw_hits':int(len(tt)),
                'median_interval_s':interval,
                'estimated_revolution_s':None if interval is None else float(interval*SPOKES),
                'geometrically_clean_hits':int(clean),
                'occupied_spoke_positions':int(occupied),
                'observed_spoke_span_revolutions':rev_span,
                'phase_score':None if fit is None else float(fit['score']),
            }
        })
    item={'sections':[s['summary'] for s in sections],'pairs':[]}
    for i,j in [(0,1),(0,2),(1,2)]:
        cmp=compare_slots(sections[i]['slots'],sections[j]['slots'])
        if cmp is None:
            continue
        null=random_order_null(sections[i]['slots'],sections[j]['slots'])
        pct=float(np.mean(null < cmp['score']))*100.0 if len(null) else None
        cmp.update({
            'sections':[i+1,j+1],
            'percentile_vs_random_slot_order':pct,
            'null_median_score':None if not len(null) else float(np.median(null)),
            'null_p95_score':None if not len(null) else float(np.quantile(null,0.95)),
        })
        item['pairs'].append(cmp)
    if item['pairs']:
        item['mean_percentile']=float(np.mean([p['percentile_vs_random_slot_order'] for p in item['pairs']]))
        item['mean_matched_slots']=float(np.mean([p['matched_slots'] for p in item['pairs']]))
        item['mean_similarity']=float(np.mean([p['mean_similarity'] for p in item['pairs']]))
    report['thresholds'][str(threshold)]=item

ranked=[]
for th,item in report['thresholds'].items():
    if 'mean_percentile' not in item:
        continue
    counts=[s['raw_hits'] for s in item['sections']]
    occupied=[s['occupied_spoke_positions'] for s in item['sections']]
    ranked.append({
        'threshold':float(th),
        'raw_counts':counts,
        'occupied_positions':occupied,
        'mean_percentile':item['mean_percentile'],
        'mean_matched_slots':item['mean_matched_slots'],
        'mean_similarity':item['mean_similarity'],
    })
ranked.sort(key=lambda x:(x['mean_percentile'],x['mean_matched_slots'],x['mean_similarity']),reverse=True)
report['ranked']=ranked

# JSON-safe copy stripping arrays/vectors.
safe={'spokes':report['spokes'],'windows':report['windows'],'thresholds':{},'ranked':ranked}
for th,item in report['thresholds'].items():
    safe['thresholds'][th]={'sections':item['sections'],'pairs':item['pairs']}
    for k in ('mean_percentile','mean_matched_slots','mean_similarity'):
        if k in item: safe['thresholds'][th][k]=item[k]

path=OUT/'analysis_36_spoke_template.json'
path.write_text(json.dumps(safe,indent=2),encoding='utf-8')
print(json.dumps(safe,indent=2))
