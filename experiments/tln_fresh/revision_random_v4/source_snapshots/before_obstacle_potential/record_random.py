"""Entirely new basic/avoidance/MPCC recordings; teacher sees perceived objects only."""
from pathlib import Path
import sys,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
root=Path(__file__).parent;sys.path.insert(0,str(root.parent));sys.path.insert(0,str(root))
import argparse,json,time,hashlib,collections
import numpy as np
from world import World
from sensors import Inputs,nominal_steer,SPEED_SCALE
from mpcc_teacher import Expert
from fast_basic import FastTracker
p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=['train','val','test']);p.add_argument('--count',type=int,default=48);p.add_argument('--offset',type=int,default=0);args=p.parse_args();w=World();sim=w.sim();dest=root/'data';dest.mkdir(exist_ok=True);base={'train':81000,'val':82000,'test':83000}[args.split];reports=[]
def layout(seed,index):
 rng=np.random.default_rng(seed);count=[1,3,2,3][index%4];items=[];biased=index%3==0;corner=float(rng.choice([25.1,32.1,35.8,54.5]));target=(corner+float(rng.uniform(.5,2.6)))%w.length
 for trial in range(1500):
  if len(items)>=count:break
  s=target if biased and len(items)==0 else float(rng.uniform(2.,w.length-1.));d=float(rng.uniform(-.38,.38));xy=w.xy(s)+w.normal(s)*d;theta=float(w.tangent(s)+rng.uniform(-.65,.65));half=.15*(abs(np.cos(theta-w.tangent(s)))+abs(np.sin(theta-w.tangent(s))))
  if w.clearance(*xy)<.40:continue
  if any(np.linalg.norm(xy-[b['x'],b['y']])<2.3 for b in items):continue
  if any(abs((s-b['s']+w.length/2)%w.length-w.length/2)<3. for b in items):continue
  left,right=Inputs(w).connected_width(s)[0]
  if max(left-d-half,right+d-half)<.45:continue
  items.append({'x':float(xy[0]),'y':float(xy[1]),'theta':theta,'s':s,'d':d,'size_m':.3})
 if len(items)!=count:raise RuntimeError('Cannot construct requested layout')
 start=(corner-float(rng.uniform(4.,6.)))%w.length if biased else float(rng.uniform(0,w.length))
 for _ in range(100):
  xy=w.xy(start)
  if all(np.linalg.norm(xy-[b['x'],b['y']])>2.2 for b in items):break
  start=(start-2.)%w.length
 return items,start,biased
for index in range(args.offset,args.offset+args.count):
 seed=base+index;mode='mpcc' if index%4==3 else 'avoidance';stem=dest/f'{args.split}_{mode}_{seed}'
 if stem.with_suffix('.json').exists():print('exists',stem.name,flush=True);continue
 obs,start,biased=layout(seed,index);rng=np.random.default_rng(seed+900000);o=w.configure(sim,obs,start,lateral=float(rng.uniform(-.055,.055)) if args.split=='train' else 0.,yaw_error=float(rng.uniform(-.04,.04)) if args.split=='train' else 0.);expert=Expert(w,cruise=3.0);tracker=FastTracker(w,3.0);inputs=Inputs(w);progress=0.;previous=start;last=np.zeros(2);states=[];raws=[];frames=[];features=[];labels=[];commands=[];executed=[];info_rows=[];errors=[];stalled=0;failure_streak=0;failures=0;recovery_steps=0;t=time.perf_counter()
 for k in range(900):
  st=sim.agents[0].state.copy();raw=np.asarray(o['scans'][0],np.float32);h,f=inputs.push(raw,st,last,previous)
  if mode=='avoidance' and not inputs.obstacles and not expert.known:
   u,profile=tracker.act(st,previous);info={'ok':True,'status':'fast_raceline_until_observed','solve_ms':0.,'plan':[]}
  else:
   expert.previous[0]=last[0];expert.previous[1]=max(0,last[1]);u,info=expert.act(st,raw,inputs.obstacles)
  target=np.array([(u[0]-nominal_steer(f))/.72,u[1]/SPEED_SCALE],np.float32);behavior=u.copy()
  if args.split=='train' and u[1]>.25:behavior[0]=np.clip(behavior[0]+rng.normal(0,.004),-.36,.36)
  states.append(st.tolist());raws.append(raw);frames.append(h);features.append(f);labels.append(target);commands.append(u.tolist());executed.append(behavior.tolist());info_rows.append({key:value for key,value in info.items() if key in ['ok','status','solve_ms','beam_ms','recovery']});recovery_steps+=int(info.get('recovery',False));failures+=not info['ok'];failure_streak=failure_streak+1 if not info['ok'] else 0;errors.append(float(np.dot(st[:2]-w.xy(previous),w.normal(previous))));last=behavior;o=w.step(sim,behavior);s=w.project(*sim.agents[0].state[:2],hint=previous);progress+=(s-previous+w.length/2)%w.length-w.length/2;previous=s;stalled=stalled+1 if abs(sim.agents[0].state[3])<.05 and k>25 else 0
  if o['collisions'][0] or progress>=w.length or stalled>=15 or failure_streak>=25:break
 success=bool(progress>=w.length and not o['collisions'][0]);record={'split':args.split,'mode':mode,'seed':seed,'success':success,'collision':sim.agents[0].collision_kind,'start_s':start,'obstacles':obs,'corner_biased':biased,'progress_m':progress,'sim_time_s':(k+1)*.1,'wall_time_s':time.perf_counter()-t,'samples':len(labels),'mean_actual_speed_mps':float(np.mean(np.array(states)[:,3])),'max_actual_speed_mps':float(np.max(np.array(states)[:,3])),'mean_abs_raceline_error_m':float(np.mean(np.abs(errors))),'solver_failures':failures,'recovery_steps':recovery_steps,'max_observed_object_count':max(int(np.sum(f[34:66].reshape(4,8)[:,3]>.5)) for f in features),'teacher_gt_obstacle_input':False,'teacher_obstacles':'LiDAR/map connected-corridor perception; uncertainty inflated; no layout GT supplied to Expert.act','source':'entirely new direct physical simulation, no earlier episode reused','trajectory':states,'commands':commands,'executed_commands':executed,'solver_log':info_rows,'teacher_source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['mpcc_teacher.py','mpcc_core.py','beam_path.py','recovery.py','sensors.py']},'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 np.savez_compressed(stem.with_suffix('.npz'),state=np.array(states,np.float32),raw_scan=np.array(raws,np.float32),scan=np.array(frames,np.float16),features=np.array(features,np.float32),target=np.array(labels,np.float32),command=np.array(commands,np.float32),executed=np.array(executed,np.float32));stem.with_suffix('.json').write_text(json.dumps(record));reports.append({k:v for k,v in record.items() if k not in ['trajectory','commands','executed_commands','solver_log']});(root/'results'/f'recording_{args.split}_{args.offset}.json').write_text(json.dumps(reports,indent=2));print(args.split,index,mode,seed,'success',success,'collision',record['collision'],'progress',round(progress,1),'speed',round(record['mean_actual_speed_mps'],2),'time',record['sim_time_s'],'reverse_steps',recovery_steps,'solver_failures',failures,'wall_s',round(time.perf_counter()-t,1),flush=True)
