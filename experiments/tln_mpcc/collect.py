"""Collect immutable episode-level BC splits with simulator MPCC success/failure audit."""
import os,json,time,argparse,hashlib
from pathlib import Path
import numpy as np
from world import World,History,ROOT,MAP,DT
from expert import MPCC
p=argparse.ArgumentParser();p.add_argument('--expert',default='expert_v4');p.add_argument('--train',type=int,default=24);p.add_argument('--val',type=int,default=6);p.add_argument('--test',type=int,default=8);a=p.parse_args()
import importlib
MPCC=importlib.import_module(a.expert).MPCC
out=Path(__file__).parent/'results';out.mkdir(exist_ok=True);w=World();sim=w.sim();splits={'train':list(range(100,100+a.train)),'val':list(range(200,200+a.val)),'test':[2026]+list(range(300,300+a.test-1))};manifest={'expert_module':a.expert,'expert_sha256':hashlib.sha256((Path(__file__).parent/(a.expert+'.py')).read_bytes()).hexdigest(),'map':'hangar_1905_v0','splits':splits,'episodes':[],'dt':DT,'history':3,'scan_normalization':'clip(0,10)/10','features':'8 local route points /5, 8 left/right widths /3, speed/2.5, previous commanded steer/.36','targets':'steer_rad/.36, target_speed_mps/2.5','map_sha256':hashlib.sha256((MAP/'global_waypoints.json').read_bytes()).hexdigest()}
for split,seeds in splits.items():
 for seed in seeds:
  path=out/f'{split}_{seed}.npz';reportpath=out/f'{split}_{seed}.json'
  if path.exists() and reportpath.exists():manifest['episodes'].append(json.loads(reportpath.read_text()));continue
  obstacles,recipe,relaxed=w.layouts(seed);rng=np.random.default_rng(seed);start_s=0. if split=='test' else float(rng.choice([0.,8.,22.,39.]))
  # Never start within 2m of a cube; route start near line always cleared by spawner.
  if any(np.linalg.norm(w.xy(start_s)-[b['x'],b['y']])<2 for b in obstacles):start_s=0.
  o=w.configure(sim,obstacles,start_s,lateral=float(rng.uniform(-.08,.08)) if split=='train' else 0.,yaw_error=float(rng.uniform(-.04,.04)) if split=='train' else 0.)
  expert=MPCC(w);history=History();prev=start_s;progress=0;scans=[];features=[];targets=[];states=[];infos=[];traj=[];speed=[];t0=time.time();min_clear=10.;consecutive_fail=0
  for k in range(600):
   st=sim.agents[0].state.copy();scan=o['scans'][0];h=history.push(scan);f=w.route_features(st,previous_steer=expert.previous[0],s_hint=prev);u,info=expert.act(st,scan,obstacles)
   if info['ok']:
    scans.append(h);features.append(f);targets.append([u[0]/.36,u[1]/2.5]);states.append(st);infos.append([info['solve_ms'],info['visible']]);consecutive_fail=0
   else:consecutive_fail+=1
   traj.append(st.tolist());speed.append(u.tolist());min_clear=min(min_clear,float(w.clearance(st[0],st[1]))-.29)
   o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2],hint=prev);progress+=(s-prev+w.length/2)%w.length-w.length/2;prev=s
   if o['collisions'][0] or progress>w.length-.5 or consecutive_fail>15:break
  success=bool(progress>w.length-.5 and not o['collisions'][0]);rep={'split':split,'seed':seed,'start_s':start_s,'obstacles':obstacles,'recipe':recipe,'relaxed':relaxed,'success':success,'collision':sim.agents[0].collision_kind,'progress_m':progress,'sim_time_s':(k+1)*DT,'samples':len(scans),'solver_failures':expert.failures,'solver_ms_p50':float(np.median(np.array(infos)[:,0])) if infos else None,'wall_clearance_center_minus_half_length':min_clear,'wall_time_s':time.time()-t0,'trajectory':traj,'commands':speed}
  # Failed episodes remain auditable but are excluded from primary BC data.
  np.savez_compressed(path,scan=np.asarray(scans,dtype=np.float32),features=np.asarray(features,dtype=np.float32),target=np.asarray(targets,dtype=np.float32),state=np.asarray(states,dtype=np.float32),info=np.asarray(infos,dtype=np.float32))
  reportpath.write_text(json.dumps(rep));manifest['episodes'].append(rep);(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
  print(split,seed,'success',success,'collision',rep['collision'],'progress',round(progress,1),'samples',len(scans),'failures',expert.failures,'seconds',round(time.time()-t0,1),flush=True)
print('COLLECTION COMPLETE',flush=True)
