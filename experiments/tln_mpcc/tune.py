"""Tune cruise only on development seeds; held-out test layouts are untouched."""
import json,time
import numpy as np
from pathlib import Path
from world import World
from expert import MPCC
w=World();sim=w.sim();rows=[]
for cruise in [1.3,1.7,2.0]:
 for seed in [90,91,92]:
  obs,_,_=w.layouts(seed);o=w.configure(sim,obs);m=MPCC(w,cruise=cruise);prev=0;progress=0;lat=[];last=[];stalls=0;t0=time.time()
  for k in range(600):
   st=sim.agents[0].state.copy();u,info=m.act(st,o['scans'][0],obs);lat.append(info['solve_ms']);o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2],hint=prev);progress+=(s-prev+w.length/2)%w.length-w.length/2;prev=s;stalls=stalls+1 if not info['ok'] else 0
   if o['collisions'][0] or progress>w.length-.5 or stalls>12:break
  row={'cruise':cruise,'seed':seed,'success':bool(progress>w.length-.5 and not o['collisions'][0]),'collision':sim.agents[0].collision_kind,'progress':progress,'time':(k+1)*.1,'failures':m.failures,'latency_p95':float(np.percentile(lat,95)),'wall_seconds':time.time()-t0};rows.append(row);Path('experiments/tln_mpcc/results/tuning.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
