import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,argparse
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent));sys.path.insert(0,str(root))
import numpy as np
from world import World
from inputs import InputsWide
from sampled_mpcc import SampledMPCC
p=argparse.ArgumentParser();p.add_argument('--cruise',type=float,default=4.2);p.add_argument('--layout',default='basic');p.add_argument('--start',type=float,default=0.);a=p.parse_args();w=World();sim=w.sim();obs=[]
if a.layout!='basic':
 for s,d in [(16.,-.22),(35.5,-.20),(49.,.25)] if a.layout=='layout02' else [(10.,.22),(33.1,.28),(46.,-.20)]:
  xy=w.xy(s)+w.normal(s)*d;obs.append(dict(x=float(xy[0]),y=float(xy[1]),theta=float(w.tangent(s)),s=s,d=d))
o=w.configure(sim,obs,a.start);teacher=SampledMPCC(w,a.cruise);inputs=InputsWide(w);last=np.zeros(2);previous=a.start;progress=0.;rows=[];begin=time.perf_counter()
for k in range(450):
 st=sim.agents[0].state.copy();h,f=inputs.push(o['scans'][0],st,last,previous);u,info=teacher.act(st,o['scans'][0],inputs.obstacles,previous);rows.append({'state':st.tolist(),'command':u.tolist(),'ok':info['ok'],'status':info['status'],'solve_ms':info['solve_ms'],'objects':inputs.obstacles});last=u;o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2],hint=previous);progress+=(s-previous+w.length/2)%w.length-w.length/2;previous=s
 if k%25==0:print(k,'progress',round(progress,2),'v',round(st[3],2),'u',u,'ok',info['ok'],'objects',len(inputs.obstacles),'ms',round(info['solve_ms'],1),flush=True)
 if o['collisions'][0] or progress>=w.length:break
report={'layout':a.layout,'cruise':a.cruise,'success':bool(progress>=w.length and not o['collisions'][0]),'collision':sim.agents[0].collision_kind,'progress':progress,'sim_time':(k+1)*.1,'wall_s':time.perf_counter()-begin,'rows':rows};(root/'results'/f'probe_{a.layout}_{a.cruise}.json').write_text(json.dumps(report));print('FINAL', {k:v for k,v in report.items() if k!='rows'},flush=True)
