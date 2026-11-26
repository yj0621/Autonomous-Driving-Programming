"""New seeded, no-reset corner-exit/MPCC rollouts, with raw sensor/action logs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from world import World,DT
from expert_corner import CornerMPCC
from perception import nominal_steer
from perception_corner import Inputs,decode_action
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
from policy import plan_label,basic_action
p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val','test'],required=True);p.add_argument('--limit',type=int);p.add_argument('--offset',type=int,default=0);a=p.parse_args()
root=Path(__file__).parent;dest=root/'corner_revision/dagger_data';dest.mkdir(parents=True,exist_ok=True);w=World();sim=w.sim();model=tf.keras.models.load_model(root/'corner_revision/results/fresh_tln_corner.keras',compile=False);predict=tf.function(lambda h,f:model([h,f],training=False));splits={'train':(51000,[-.28,0.,.28],[.45,1.,1.65]),'val':(42000,[-.18,.18],[.75,1.35]),'test':(43000,[-.1,.1],[.60,1.5])};base,lats,exits=splits[a.split];schedule=[]
for corner in [25.1,32.1,35.8,54.5]:
 for exit in exits:
  for lateral in lats:
   so=(corner+exit)%w.length;xy=w.xy(so)+w.normal(so)*lateral
   if w.clearance(*xy)<.40:continue
   obs=[{'x':float(xy[0]),'y':float(xy[1]),'theta':float(w.tangent(so)), 's':so,'d':lateral}]
   schedule.append({'mode':'corner_exit','obstacles':obs,'start_s':(corner-4.5)%w.length,'required_progress_m':11.,'corner_s':corner,'exit_distance_m':exit,'lateral_m':lateral})
for j in range(6 if a.split=='train' else 3):
 obs,recipe,relaxed=w.layouts(base+900+j,count=2+j%3)
 schedule.append({'mode':'mpcc','obstacles':obs,'start_s':0.,'required_progress_m':w.length,'recipe':recipe,'relaxed':relaxed})
for j in range(2 if a.split=='train' else 1):schedule.append({'mode':'basic','obstacles':[],'start_s':float(j*15),'required_progress_m':w.length})
for index,r in enumerate(schedule):
 if index<a.offset or (a.limit is not None and index>=a.offset+a.limit):continue
 seed=base+index;path=dest/f'{a.split}_{r["mode"]}_{seed}.npz'
 if path.exists() and path.with_suffix('.json').exists():print('exists',path.name,flush=True);continue
 t0=time.time();rng=np.random.default_rng(seed);start=r['start_s'];o=w.configure(sim,r['obstacles'],start,lateral=float(rng.uniform(-.10,.10)) if a.split=='train' else 0.,yaw_error=float(rng.uniform(-.07,.07)) if a.split=='train' else 0.)
 teacher=CornerMPCC(w,cruise=2.2);inputs=Inputs(w);prev=start;last=np.zeros(2);progress=0.;stall=0;executed=[];hs=[];fs=[];ys=[];raws=[];states=[];commands=[];trajectory=[];solvers=[];failures=0;errors=[]
 for k in range(850 if r['mode']!='corner_exit' else 280):
  st=sim.agents[0].state.copy();scan=np.asarray(o['scans'][0],np.float32);f0=w.route_features(st,previous_steer=last[0],s_hint=prev);h,f=inputs.push(scan,st,f0)
  if r['mode']=='basic':u,_=basic_action(w,st,prev);target=np.r_[np.zeros(3),u[1]/2.8,0.];info={'ok':True,'status':'basic_tracker','solve_ms':0.}
  else:
   u,info=teacher.act(st,scan,r['obstacles']);label=plan_label(w,teacher,info,st)
   target=np.r_[label[0][:3],u[1]/2.8,np.clip((u[0]-nominal_steer(f0))/.72,-1,1)] if label is not None else None
  trajectory.append(st.tolist());commands.append(u.tolist());solvers.append({'ok':info['ok'],'status':info['status'],'ms':info['solve_ms']})
  if target is not None:
   hs.append(h);fs.append(f);ys.append(target);raws.append(scan);states.append(st)
  else:failures+=1
  behavior=u.copy()
  if k%4==0 and target is not None:
   student=decode_action(predict(h[None],f[None]).numpy()[0],f);behavior=.65*u+.35*student
  behavior[0]=np.clip(behavior[0]+rng.normal(0,.012),-.36,.36);executed.append(behavior.tolist())
  errors.append(float(np.dot(st[:2]-w.xy(prev),w.normal(prev))));last=behavior;o=w.step(sim,behavior);s=w.project(*sim.agents[0].state[:2],hint=prev);progress+=(s-prev+w.length/2)%w.length-w.length/2;prev=s
  stall=stall+1 if k>20 and sim.agents[0].state[3]<.05 else 0
  if o['collisions'][0] or progress>=r['required_progress_m'] or stall>=40 or failures>=12:break
 success=bool(progress>=r['required_progress_m'] and not o['collisions'][0])
 report={**r,'split':a.split,'seed':seed,'success':success,'collision':sim.agents[0].collision_kind,'progress_m':progress,'sim_time_s':(k+1)*DT,'wall_time_s':time.time()-t0,'solver_failures':failures,'samples':len(hs),'mean_abs_raceline_error_m':float(np.mean(np.abs(errors))),'trajectory':trajectory,'commands':commands,'solver_log':solvers,'behavior_policy':'65 percent expert / 35 percent current fresh student query blend on every fourth step plus 0.012 rad noise; expert labels kept separate','executed_commands':executed,'teacher':'CornerMPCC' if r['mode']!='basic' else 'basic_tracker','teacher_sha256':hashlib.sha256((root/'expert_corner.py').read_bytes()).hexdigest(),'no_reset':True,'raw_lidar_recorded':True,'obstacle_gt_student_input':False}
 np.savez_compressed(path,scan=np.asarray(hs,np.float32),features=np.asarray(fs,np.float32),target=np.asarray(ys,np.float32),raw_scan=np.asarray(raws,np.float32),state=np.asarray(states,np.float32))
 path.with_suffix('.json').write_text(json.dumps(report));print(a.split,index,r['mode'],seed,'ok',success,'collision',report['collision'],'progress',round(progress,2),'samples',len(hs),'failures',failures,'wall_s',round(time.time()-t0,1),flush=True)
print('COLLECTION DONE',a.split,len(schedule),flush=True)
