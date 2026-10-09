#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

THIS_DIR=Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path: sys.path.insert(0,str(THIS_DIR))
import detector_v4 as d4
import detector_v5 as d5

WAV=Path('audio/Spoke test new.wav')
OUT=Path('output_36_spoke_template_v2'); OUT.mkdir(exist_ok=True)
SPOKES=36
WINDOWS=[(2.5,9.0),(26.0,34.5),(45.0,51.7)]


def sig(x,sr,t):
    c=int(round(t*sr)); h=int(round(.018*sr)); y=x[max(0,c-h):min(len(x),c+h)]
    if len(y)<64: return np.zeros(24)
    p=np.abs(np.fft.rfft(y*np.hanning(len(y))))**2; f=np.fft.rfftfreq(len(y),1/sr)
    edges=np.geomspace(250,min(12000,sr*.45),25)
    v=np.array([np.log1p(float(p[(f>=a)&(f<b)].sum())) for a,b in zip(edges[:-1],edges[1:])])
    v-=np.median(v); n=np.linalg.norm(v); return v/n if n>1e-12 else v


def interval(tt):
    if len(tt)<5:return None
    dt=np.diff(tt); m=float(np.median(dt)); g=dt[(dt>.45*m)&(dt<1.8*m)]
    return float(np.median(g)) if len(g)>=3 else m


def phasefit(tt,prom,ival):
    best=None
    for ph in np.linspace(0,ival,361,endpoint=False):
        z=(tt-ph)/ival; k=np.rint(z).astype(int); r=np.abs(z-k); good=r<=.34
        score=float(np.sum(good*(1-r/.34)*(1+.08*np.log1p(prom))))
        if best is None or score>best[0]: best=(score,ph,k,r,good)
    return best


def slots(tt,prom,V,ival,fit):
    _,ph,k,r,good=fit; chosen={}
    for i in range(len(tt)):
        if not good[i]: continue
        q=float(prom[i])/(1+4*float(r[i])); kk=int(k[i])
        if kk not in chosen or q>chosen[kk][0]: chosen[kk]=(q,i)
    out=[None]*SPOKES
    grouped={s:[] for s in range(SPOKES)}
    for kk,(_,i) in chosen.items(): grouped[kk%SPOKES].append(i)
    for s,ids in grouped.items():
        if ids:
            v=np.mean(V[ids],axis=0); n=np.linalg.norm(v); out[s]=v/n if n>1e-12 else v
    return out,chosen


def compare(A,B):
    best=None
    for shift in range(SPOKES):
        vals=[]
        for s in range(SPOKES):
            if A[s] is None or B[(s+shift)%SPOKES] is None: continue
            vals.append((float(np.dot(A[s],B[(s+shift)%SPOKES]))+1)/2)
        if not vals: continue
        mean=float(np.mean(vals)); matched=len(vals); score=mean*min(matched,24)/24
        row=(score,shift,mean,float(np.median(vals)),matched)
        if best is None or row[0]>best[0]: best=row
    return best


def null(A,B,reps=300):
    rng=np.random.default_rng(20261009); idx=[i for i,x in enumerate(B) if x is not None]; vals=[]
    base=[B[i] for i in idx]
    for _ in range(reps):
        z=[None]*SPOKES; order=rng.permutation(len(base))
        for dst,src in zip(idx,order): z[dst]=base[src]
        c=compare(A,z)
        if c: vals.append(c[0])
    return np.asarray(vals)

A=d4.load_wav(WAV); x=d4.highpass_analysis(A.mono,A.sample_rate,250)
t,s,p=d4.all_onset_peaks(x,A.sample_rate,.025); mx=float(np.max(s))+1e-12
cands=[d4.Candidate(float(tt),0,1,float(ss/mx),float(pp)) for tt,ss,pp in zip(t,s,p)]
report={'spokes':36,'windows':WINDOWS,'thresholds':{}}
for th in [4,5,6,7,8,9,10,12]:
    det=d5.refine_sequence(d5.adaptive_detect(cands,float(th))); secs=[]
    for lo,hi in WINDOWS:
        ds=[q for q in det if lo<=q.time_seconds<=hi]; tt=np.array([q.time_seconds for q in ds]); pr=np.array([q.normalized_prominence for q in ds])
        V=np.vstack([sig(x,A.sample_rate,z) for z in tt]) if len(tt) else np.empty((0,24)); iv=interval(tt)
        if iv is None: sl=[None]*36; chosen={}; pf=None
        else: pf=phasefit(tt,pr,iv); sl,chosen=slots(tt,pr,V,iv,pf)
        ks=sorted(chosen); span=((ks[-1]-ks[0]+1)/36) if ks else None
        secs.append({'slots':sl,'summary':{'raw_hits':len(tt),'median_interval_s':iv,'estimated_revolution_s':None if iv is None else iv*36,'geometrically_clean_hits':len(chosen),'occupied_positions':sum(v is not None for v in sl),'observed_span_revolutions':span}})
    item={'sections':[q['summary'] for q in secs],'pairs':[]}
    for i,j in [(0,1),(0,2),(1,2)]:
        c=compare(secs[i]['slots'],secs[j]['slots'])
        if not c: continue
        ns=null(secs[i]['slots'],secs[j]['slots']); pct=float(np.mean(ns<c[0])*100) if len(ns) else None
        item['pairs'].append({'sections':[i+1,j+1],'best_shift':c[1],'mean_similarity':c[2],'median_similarity':c[3],'matched_slots':c[4],'percentile_vs_random':pct})
    if item['pairs']:
        item['mean_percentile']=float(np.mean([q['percentile_vs_random'] for q in item['pairs']])); item['mean_matched_slots']=float(np.mean([q['matched_slots'] for q in item['pairs']]))
    report['thresholds'][str(th)]=item
report['ranked']=sorted([{'threshold':float(k),'counts':[s['raw_hits'] for s in v['sections']],'occupied':[s['occupied_positions'] for s in v['sections']],'mean_percentile':v.get('mean_percentile'),'mean_matched_slots':v.get('mean_matched_slots')} for k,v in report['thresholds'].items()],key=lambda z:(z['mean_percentile'] or -1,z['mean_matched_slots'] or -1),reverse=True)
(OUT/'analysis.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
