"""Fresh, episode-split recordings of basic, avoidance and MPCC driving."""
import os,json,time,hashlib,argparse
from pathlib import Path
import numpy as np
from world import World,History,DT,MAP
from expert import MPCC
from policy import basic_action,plan_label,SPEED_SCALE
from safety import shield
p=argparse.ArgumentParser();p.add_argument('--small',action='store_true');args=p.parse_args();root=Path(__file__).parent;data=root/'data';w=World();sim=w.sim();rng=np.random.default_rng(8127)
# All seeds are new; neither legacy model weights nor legacy driving arrays are read.
schedule=[]
for split,base,n_basic,n_avoid,n_mpcc in [('train',1000,8,12,12),('val',2000,2,3,3),('test',3000,2,3,3)]:
 for mode,n in [('basic',n_basic),('avoidance',n_avoid),('mpcc',n_mpcc)]:
  for j in range(n):schedule.append((split,mode,base,j));base+=1
manifest={'created_at':time.time(),'map':'hangar_1905_v0','map_sha256':hashlib.sha256((MAP/'global_waypoints.json').read_bytes()).hexdigest(),'expert_sha256':hashlib.sha256((root/'expert.py').read_bytes()).hexdigest(),'legacy_data_used':False,'legacy_weights_used':False,'episodes':[],'control_dt_s':DT}
for split,mode,seed,index in schedule:
 path=data/f'{split}_{mode}_{seed}.npz';report=path.with_suffix('.json')
 if path.exists() and report.exists():manifest['episodes'].append(json.loads(report.read_text()));continue
 obs,recipe,relaxed=w.layouts(seed,count=0 if mode=='basic' else (1 if mode=='avoidance' and index%2==0 else 4));start=[0.,10.,23.,40.][index%4] if split=='train' else 0.
 if any(np.linalg.norm(w.xy(start)-[b['x'],b['y']])<2.0 for b in obs):start=0.
 r=np.random.default_rng(seed);o=w.configure(sim,obs,start,lateral=float(r.uniform(-.08,.08)) if split=='train' else 0.,yaw_error=float(r.uniform(-.03,.03)) if split=='train' else 0.)
 expert=MPCC(w,cruise=1.8 if mode=='avoidance' else 2.2) if mode!='basic' else None;hist=History();last=np.zeros(2);prev=start;progress=0.;scans=[];features=[];targets=[];masks=[];states=[];raw_scans=[];commands=[];trajectory=[];infos=[];failures=0;stalled=0;start_time=time.time();deviations=[]
 for k in range(700):
  st=sim.agents[0].state.copy();scan=np.asarray(o['scans'][0]);h=hist.push(scan);f=w.route_features(st,previous_steer=last[0],s_hint=prev)
  if mode=='basic':u,_=basic_action(w,st,prev);label=np.r_[0.,0.,0.,u[1]/SPEED_SCALE].astype(np.float32);mask=np.ones(3);ok=True;status='race_line_tracker';ms=0.
  else:
   u,info=expert.act(st,scan,obs);l=plan_label(w,expert,info,st);ok=l is not None;label,mask=l if ok else (None,None);status=info['status'];ms=info['solve_ms']
  # Recorded command and speed target agree after the same shield.
  u,safety=shield(scan,u)
  if ok:
   label[3]=u[1]/SPEED_SCALE;scans.append(h);features.append(f);targets.append(label);masks.append(mask);states.append(st);raw_scans.append(scan.astype(np.float32));infos.append([ms,int(safety['limited'])])
  else:failures+=1
  commands.append(u.tolist());trajectory.append(st.tolist());deviations.append(float(np.dot(st[:2]-w.xy(prev),w.normal(prev))))
  last=u;o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2],hint=prev);progress+=(s-prev+w.length/2)%w.length-w.length/2;prev=s;stalled=stalled+1 if sim.agents[0].state[3]<.05 and k>20 else 0
  if o['collisions'][0] or progress>w.length-.5 or stalled>40 or failures>20:break
 success=bool(progress>w.length-.5 and not o['collisions'][0]);rep={'split':split,'mode':mode,'seed':seed,'start_s':start,'success':success,'collision':sim.agents[0].collision_kind,'progress_m':progress,'sim_time_s':(k+1)*DT,'wall_time_s':time.time()-start_time,'solver_failures':failures,'samples':len(scans),'mean_abs_raceline_error_m':float(np.mean(np.abs(deviations))),'obstacles':obs,'recipe':recipe,'relaxed':relaxed,'trajectory':trajectory,'commands':commands}
 np.savez_compressed(path,scan=np.asarray(scans,np.float32),features=np.asarray(features,np.float32),target=np.asarray(targets,np.float32),plan_mask=np.asarray(masks,np.float32),state=np.asarray(states,np.float32),raw_scan=np.asarray(raw_scans,np.float32),info=np.asarray(infos,np.float32))
 report.write_text(json.dumps(rep));manifest['episodes'].append(rep);(data/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(split,mode,seed,'success',success,'collision',rep['collision'],'progress',round(progress,1),'samples',len(scans),'seconds',round(time.time()-start_time,1),flush=True)
print('NEW RECORDINGS COMPLETE',flush=True)
