"""Freeze the latest actual controller session; physical-time metrics, no invented ROS scans."""
from pathlib import Path
import sys,json,hashlib
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent))
import numpy as np
from world import World
w=World();rows=[]
with (root/'results/live_v5.jsonl').open() as f:
 for line in f:
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  if r['event']=='ready':rows=[]
  rows.append(r)
controls=[x for x in rows if x['event']=='control'];assert controls
start=controls[0]['physics_s'];last_s=None;progress=0.;lap_start=start;laps=[];speed=[];straight=[];corners=[];error=[];clear_error=[];accels=[];reverse=0;stationary=0.;max_stationary=0.;last=None
for x in controls:
 st=np.array(x['state']);s=w.project(*st[:2],hint=last_s)
 if last_s is not None:progress+=(s-last_s+w.length/2)%w.length-w.length/2
 last_s=s
 if progress>=w.length:laps.append(x['physics_s']-lap_start);progress-=w.length;lap_start=x['physics_s']
 d=float(np.dot(st[:2]-w.xy(s),w.normal(s)));error.append(abs(d));speed.append(float(st[3]));reverse+=x['command'][1]<-.05
 if not x.get('objects'):clear_error.append(abs(d))
 d1=w.path(s,1);d2=w.path(s,2);k=abs(d1[0]*d2[1]-d1[1]*d2[0])/max(float(np.linalg.norm(d1))**3,.01)
 if k<.08:straight.append(float(st[3]))
 if k>.55:corners.append(float(st[3]))
 if last is not None:
  dt=x['physics_s']-last['physics_s']
  if .05<dt<.3:
   accels.append((st[3]-last['state'][3])/dt);stationary=stationary+dt if abs(st[3])<.05 and x['physics_s']-start>2 else 0.;max_stationary=max(max_stationary,stationary)
 last=x
resets=[x for x in rows if x['event']=='pose_reset_observed'];collisions=[x for x in rows if x['event']=='collision'];snapshot={'model':rows[0].get('model'),'session_start':rows[0],'control_samples':len(controls),'physics_duration_s':controls[-1]['physics_s']-start,'full_continuous_laps':len(laps),'lap_physics_s':laps,'mean_lap_physics_s':float(np.mean(laps)),'collisions':collisions,'startup_pose_resets':min(1,len(resets)),'pose_resets_after_startup':max(0,len(resets)-1),'max_stationary_physics_s_after_startup':max_stationary,'reverse_command_steps':int(reverse),'mean_speed_mps':float(np.mean(speed)),'max_speed_mps':float(max(speed)),'straight_mean_mps':float(np.mean(straight)),'tight_corner_mean_mps':float(np.mean(corners)),'mean_abs_raceline_error_m':float(np.mean(error)),'clear_road_mean_abs_error_m':float(np.mean(clear_error)),'actual_acceleration_p95_mps2':float(np.percentile(accels,95)),'actual_deceleration_p05_mps2':float(np.percentile(accels,5)),'control_latency_p95_ms':float(np.percentile([x['latency_ms'] for x in controls],95)),'mpcc_intervention_steps':sum(x['guard'].get('intervened',False) for x in controls),'simulation_mode':'unchanged vehicle physics, command-step synchronization, monotonic real-time target pacing','data_source':'actual /tln/control_debug JSONL records, not a fabricated raw-scan ROS bag','model_sha256':hashlib.sha256((root/rows[0]['model']).read_bytes()).hexdigest()}
(root/'results/final_runtime_snapshot.json').write_text(json.dumps(snapshot,indent=2));out=root/'results/final_runtime_controls.jsonl'
with out.open('w') as f:
 for x in rows:f.write(json.dumps(x)+'\n')
print({k:v for k,v in snapshot.items() if k not in ['lap_physics_s','session_start','collisions']},flush=True)
