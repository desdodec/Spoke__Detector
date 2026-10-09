#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np

THIS=Path(__file__).resolve().parent
if str(THIS) not in sys.path: sys.path.insert(0,str(THIS))
import detector_v4 as d4
import detector_v5 as d5

WAV=Path('audio/Spoke test new.wav')
OUT=Path('output_36_spoke_repeat'); OUT.mkdir(exist_ok=True)
# Broad sessions inferred from long pauses between the user's three speed tests.
SESSIONS=[(2.5,25.9),(26.0,39.2),(40.0,51.8)]
TARGET=36


def signature(x,sr,t):
    c=int(round(t*sr)); h=int(round(.02*sr)); y=x[max(0,c-h):min(len(x),c+h)]
    if len(y)<64:return np.zeros(28)
    p=np.abs(np.fft.rfft(y*np.hanning(len(y))))**2; f=np.fft.rfftfreq(len(y),1/sr)
    edges=np.geomspace(220,min(12000,.45*sr),29)
    v=np.array([np.log1p(float(p[(f>=a)&(f<b)].sum())) for a,b in zip(edges[:-1],edges[1:])])
    v-=np.mean(v); n=np.linalg.norm(v); return v/n if n>1e-12 else v


def lag_stats(V,lag):
    if len(V)<=lag+2:return None
    sims=np.sum(V[:-lag]*V[lag:],axis=1)
    sims=(sims+1)/2
    return {'pairs':int(len(sims)),'mean':float(np.mean(sims)),'median':float(np.median(sims)),'p25':float(np.quantile(sims,.25))}


def null_percentile(V,lag,observed,reps=600,seed=20261009):
    if len(V)<=lag+2:return None
    rng=np.random.default_rng(seed+lag+len(V)); vals=[]; n=len(V)-lag
    A=V[:n]
    for _ in range(reps):
        idx=rng.choice(len(V),size=n,replace=False)
        B=V[idx]
        vals.append(float(np.mean((np.sum(A*B,axis=1)+1)/2)))
    vals=np.asarray(vals)
    return {'percentile':float(np.mean(vals<observed)*100),'null_mean':float(np.mean(vals)),'null_p95':float(np.quantile(vals,.95))}

A=d4.load_wav(WAV); x=d4.highpass_analysis(A.mono,A.sample_rate,250)
t,s,p=d4.all_onset_peaks(x,A.sample_rate,.025); mx=float(np.max(s))+1e-12
cands=[d4.Candidate(float(tt),0,1,float(ss/mx),float(pp)) for tt,ss,pp in zip(t,s,p)]
report={'target_spokes':36,'sessions':SESSIONS,'thresholds':{}}
for th in [4,5,6,7,8,10,12]:
    det=d5.refine_sequence(d5.adaptive_detect(cands,float(th)))
    item={'sessions':[]}
    for si,(lo,hi) in enumerate(SESSIONS,1):
        ds=[q for q in det if lo<=q.time_seconds<=hi]; tt=np.array([q.time_seconds for q in ds])
        V=np.vstack([signature(x,A.sample_rate,z) for z in tt]) if len(tt) else np.empty((0,28))
        lags=[]
        for lag in range(30,43):
            st=lag_stats(V,lag)
            if st is None: continue
            nu=null_percentile(V,lag,st['mean'])
            lags.append({'lag':lag,**st,**(nu or {})})
        lags.sort(key=lambda z:(z.get('percentile',-1),z['mean']),reverse=True)
        target=next((z for z in lags if z['lag']==36),None)
        item['sessions'].append({'session':si,'count':len(tt),'duration_s':hi-lo,'target36':target,'best_lags':lags[:6],'rank36':(1+next((i for i,z in enumerate(lags) if z['lag']==36),999)) if target else None})
    ranks=[s['rank36'] for s in item['sessions'] if s['rank36']]
    pcts=[s['target36']['percentile'] for s in item['sessions'] if s['target36']]
    item['mean_rank36']=float(np.mean(ranks)) if ranks else None; item['mean_percentile36']=float(np.mean(pcts)) if pcts else None
    report['thresholds'][str(th)]=item
report['ranked']=sorted([{'threshold':float(k),'counts':[s['count'] for s in v['sessions']],'mean_rank36':v['mean_rank36'],'mean_percentile36':v['mean_percentile36']} for k,v in report['thresholds'].items()], key=lambda z:((z['mean_percentile36'] or -1),-(z['mean_rank36'] or 999)), reverse=True)
(OUT/'repeat36.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
