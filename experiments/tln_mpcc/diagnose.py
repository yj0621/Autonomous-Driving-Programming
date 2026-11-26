import json,time
from pathlib import Path
import numpy as np
from world import World
from expert import MPCC
w=World();sim=w.sim();rows=[]
for seed in [92,101,104,2026]:
 obs,recipe,_=w.layouts(seed);o=w.configure(sim,obs);m=MPCC(w);prev=0;progress=0;trace=[];failed=0
 for k in range(600):
  st=sim.agents[0].state.copy();u,info=m.act(st,o['scans'][0],obs);trace.append({'state':st.tolist(),'u':u.tolist(),'ms':info['solve_ms'],'status':info['status'],'ok':info['ok'],'visible':info['visible']});o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2],hint=prev);progress+=(s-prev+w.length/2)%w.length-w.length/2;prev=s;failed=failed+1 if not info['ok'] else 0
  if o['collisions'][0] or progress>w.length-.5 or failed>15:break
 row={'seed':seed,'success':bool(progress>w.length-.5 and not o['collisions'][0]),'collision':sim.agents[0].collision_kind,'progress':progress,'time':(k+1)*.1,'failures':m.failures,'latency_p95':float(np.percentile([x['ms'] for x in trace],95)),'trace':trace,'obstacles':obs};rows.append(row);Path('experiments/tln_mpcc/results/expert_latest_diagnostics.json').write_text(json.dumps(rows));print({k:v for k,v in row.items() if k not in ('trace','obstacles')},flush=True)
