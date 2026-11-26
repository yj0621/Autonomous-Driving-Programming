import json,time
import numpy as np
from world import World
from expert import MPCC
w=World();sim=w.sim();obs,recipe,_=w.layouts(100);o=w.configure(sim,obs);m=MPCC(w);progress=0;prev=0;records=[]
for k in range(650):
 state=sim.agents[0].state.copy();u,info=m.act(state,o['scans'][0],obs);o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2]);progress+=(s-prev+w.length/2)%w.length-w.length/2;prev=s
 records.append({'k':k,'u':u.tolist(),'state':state.tolist(),'ok':info['ok'],'solve_ms':info['solve_ms'],'progress':progress})
 if k%25==0:print(k,np.round(state[:5],2),np.round(u,2),'prog',round(progress,1),'fail',m.failures,flush=True)
 if o['collisions'][0] or progress>w.length-.5:break
out={'seed':100,'recipe':recipe,'obstacles':obs,'progress':progress,'collision':sim.agents[0].collision_kind,'failures':m.failures,'steps':records};open('experiments/tln_mpcc/results/probe.json','w').write(json.dumps(out,indent=2));print('RESULT',out['progress'],out['collision'],out['failures'],flush=True)
