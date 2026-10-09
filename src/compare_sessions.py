#!/usr/bin/env python3
from pathlib import Path
import csv, json
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector_v4 as d4
import detector_v5 as d5

FILES = [Path('audio/Bike test 2.wav'), Path('audio/Jo_Grant_Mono.wav')]
CAL = 19.647
THRESHOLDS = [4.0, 8.0, 12.0, 16.0, CAL, 24.0, 28.0]

rows=[]
summary={}
for path in FILES:
    a=d4.load_wav(path)
    x=d4.highpass_analysis(a.mono,a.sample_rate,250.0)
    times,strengths,prom=d4.all_onset_peaks(x,a.sample_rate,0.025)
    candidates=[]
    mx=float(np.max(strengths))+1e-12
    for t,s,p in zip(times,strengths,prom):
        candidates.append(d4.Candidate(time_seconds=float(t),distance=0.0,confidence=1.0,onset_strength=float(s/mx),normalized_prominence=float(p)))
    vals=np.asarray([c.normalized_prominence for c in candidates],float)
    adaptive=d5.adaptive_prominence_threshold(candidates,4.0)
    q={str(k):float(np.quantile(vals,k)) for k in [0.5,0.75,0.9,0.95,0.975,0.99]}
    rec={'duration_s':len(a.mono)/a.sample_rate,'candidates':len(candidates),'adaptive_threshold':adaptive,'quantiles':q,'counts':{}}
    for th in THRESHOLDS+[adaptive]:
        base=d5.adaptive_detect(candidates,float(th))
        refined=d5.refine_sequence(base)
        rec['counts'][f'{th:.3f}']={'base':len(base),'refined':len(refined)}
        rows.append([path.name,th,len(base),len(refined)])
    summary[path.name]=rec

Path('output_compare').mkdir(exist_ok=True)
with open('output_compare/summary.json','w') as f: json.dump(summary,f,indent=2)
with open('output_compare/counts.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['file','threshold','base','refined']); w.writerows(rows)
print(json.dumps(summary,indent=2))
