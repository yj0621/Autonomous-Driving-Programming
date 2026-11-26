"""Fresh MPCC recordings. No previous weights/episodes, no reverse/resets in laps."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,argparse,json,time,hashlib
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent));sys.path.insert(0,str(root))
import numpy as np
from world import World
from inputs import InputsWide
from sampled_mpcc import SampledMPCC
from decode import nominal,FUTURE_STEPS
p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val','test'],required=True);p.add_argument('--seed-base',type=int);p.add_argument('--offset',type=int,default=0);p.add_argument('--count',type=int,required=True);a=p.parse_args();w=World();sim=w.sim();source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')};base=a.seed_base if a.seed_base is not None else {'train':91000,'val':92000,'test':93000}[a.split];summaries=[]
def layout(seed,index):
 rng=np.random.default_rng(seed);basic=index<({'train':12,'val':4,'test':4}[a.split]);obs=[]
 if not basic:
  count=1+(index%3)
  for trial in range(3000):
   if len(obs)==count:break
   so=float(rng.uniform(1,w.length-1))
   if not obs and index%2==0:so=(float(rng.choice([25.1,32.1,35.8,54.5]))+rng.uniform(.35,2.5))%w.length
   d=float(rng.uniform(-.35,.35));xy=w.xy(so)+w.normal(so)*d;theta=float(w.tangent(so)+rng.uniform(-.65,.65));corners=xy+np.array([[-.15,-.15],[-.15,.15],[.15,-.15],[.15,.15]])@np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
   if np.min(w.clearance(corners[:,0],corners[:,1]))<.10:continue
   if any(np.linalg.norm(xy-[b['x'],b['y']])<2.3 or abs((so-b['s']+w.length/2)%w.length-w.length/2)<3.0 for b in obs):continue
   left,right=InputsWide(w).connected_width(so)[0]
   if max(left-d-.22,right+d-.22)<.34:continue
   obs.append(dict(x=float(xy[0]),y=float(xy[1]),theta=theta,s=so,d=d,size_m=.3))
  if len(obs)!=count:raise RuntimeError('Layout construction failed')
 start=float(rng.uniform(0,w.length))
 for _ in range(300):
  xy=w.xy(start)
  if all(np.linalg.norm(xy-[b['x'],b['y']])>3.2 for b in obs) and w.clearance(*xy)>.28:break
  start=(start-1.3)%w.length
 return obs,start,basic
for index in range(a.offset,a.offset+a.count):
 seed=base+index;stem=root/'data'/f'{a.split}_{seed}'
 if stem.with_suffix('.json').exists():continue
 obs,start,basic=layout(seed,index);rng=np.random.default_rng(seed);o=w.configure(sim,obs,start,lateral=float(rng.uniform(-.035,.035)) if a.split=='train' else 0.,yaw_error=float(rng.uniform(-.02,.02)) if a.split=='train' else 0.);inputs=InputsWide(w);teacher=SampledMPCC(w,cruise=4.5,seed=seed+500);last=np.zeros(2);previous=start;progress=0.;states=[];scans=[];history=[];features=[];targets=[];masks=[];commands=[];logs=[];stationary=0;failures=0;begin=time.perf_counter()
 for k in range(450):
  st=sim.agents[0].state.copy();raw=np.asarray(o['scans'][0],np.float32);h,f=inputs.push(raw,st,last,previous);u,info=teacher.act(st,raw,inputs.obstacles,previous);actions=np.array(info['actions'][:FUTURE_STEPS]) if info.get('actions') else np.tile(u,(FUTURE_STEPS,1));y=actions.copy();y[:,0]/=.36;y[0,0]=(u[0]-nominal(f))/.72;y[:,1]/=4.5;mask=np.full(FUTURE_STEPS,float(info['ok']));states.append(st);scans.append(raw);history.append(h);features.append(f);targets.append(y.ravel());masks.append(mask);commands.append(u);logs.append({key:info[key] for key in ['ok','status','solve_ms']});failures+=not info['ok'];last=u;o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2],hint=previous);progress+=(s-previous+w.length/2)%w.length-w.length/2;previous=s;stationary=stationary+1 if abs(sim.agents[0].state[3])<.05 and k>20 else 0
  if o['collisions'][0] or progress>=w.length or stationary>=10:break
 success=bool(progress>=w.length and not o['collisions'][0]);record=dict(seed=seed,index=index,split=a.split,mode='basic_mpcc' if basic else 'avoidance_mpcc',success=success,collision=sim.agents[0].collision_kind,start_s=start,obstacles=obs,progress_m=progress,sim_time_s=(k+1)*.1,wall_s=time.perf_counter()-begin,samples=len(states),teacher_failed_frames=failures,mean_speed_mps=float(np.mean(np.array(states)[:,3])),max_speed_mps=float(np.max(np.array(states)[:,3])),reverse_steps=0,pose_resets=0,source_sha256=source,teacher_gt_obstacle_input=False,teacher_state='same measured pose/speed/steer/yaw-rate; slip estimated from speed and yaw-rate',solver_log=logs)
 np.savez_compressed(stem.with_suffix('.npz'),state=np.array(states,np.float32),raw_scan=np.array(scans,np.float32),scan=np.array(history,np.float16),features=np.array(features,np.float32),target=np.array(targets,np.float32),mask=np.array(masks,np.float32),command=np.array(commands,np.float32));stem.with_suffix('.json').write_text(json.dumps(record));summaries.append({k:v for k,v in record.items() if k!='solver_log'});(root/'results'/f'collection_{a.split}_{a.offset}.json').write_text(json.dumps(summaries,indent=2));print(seed,record['mode'],'success',success,'collision',record['collision'],'progress',round(progress,1),'sim_s',record['sim_time_s'],'failed_frames',failures,'wall',round(record['wall_s'],1),flush=True)
