"""No-reset closed-loop evaluation of all layouts, including expert failures."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import tensorflow as tf
from world import World
from perception_corner import Inputs,decode_action
from guard_corner import Guard
p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--controllers',default='neural,protected');p.add_argument('--include-old',action='store_true');p.add_argument('--baseline',action='store_true');p.add_argument('--tag',default='');p.add_argument('--model',default='corner_revision/results/fresh_tln_corner.keras');p.add_argument('--decoder',choices=['residual','path'],default='residual');a=p.parse_args();
if a.decoder=='path':
 from perception_corner import decode_path_action as decode_action
if a.baseline:
 from perception import Inputs,decode_action
 from guard_v2 import Guard
root=Path(__file__).parent;out=root/'corner_revision/results';w=World();sim=w.sim();tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
path=root/a.model;m=tf.keras.models.load_model(path,compile=False)
predict=tf.function(lambda h,f:m([h,f],training=False),input_signature=[tf.TensorSpec((None,541,6),tf.float32),tf.TensorSpec((None,42),tf.float32)]);predict(tf.zeros((1,541,6)),tf.zeros((1,42)))
files=sorted((root/'corner_revision/data').glob(a.split+'_*.json'))
if a.include_old:files+=sorted((root/'data').glob(a.split+'_*.json'))
rows=[]
for controller in a.controllers.split(','):
 for p in files:
  r=json.loads(p.read_text());start=r['start_s'];required=r.get('required_progress_m',w.length);o=w.configure(sim,r['obstacles'],start);inputs=Inputs(w);guard=Guard();prev=start;last=np.zeros(2);progress=0.;stalled=0;trace=[];lat=[];errors=[];interventions=0;recovery=0
  for k in range(850 if required>=w.length else 280):
   t=time.perf_counter();st=sim.agents[0].state.copy();scan=o['scans'][0];base=w.route_features(st,previous_steer=last[0],s_hint=prev);h,f=inputs.push(scan,st,base);y=predict(h[None],f[None]).numpy()[0]
   if controller=='protected':u,detail=guard.act(scan,f,y,inputs.foreground.observations);interventions+=int(detail['braking']);recovery+=int(detail.get('safe_path_recovery',False))
   else:u=decode_action(y,f);detail={}
   lat.append((time.perf_counter()-t)*1000);errors.append(float(np.dot(st[:2]-w.xy(prev),w.normal(prev))));trace.append({'state':st.tolist(),'command':u.tolist(),'foreground':f[34:].tolist(),'guard':detail});last=u;o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2],hint=prev);progress+=(s-prev+w.length/2)%w.length-w.length/2;prev=s;stalled=stalled+1 if sim.agents[0].state[3]<.05 and k>20 else 0
   if o['collisions'][0] or progress>=required or stalled>=40:break
  row={'controller':controller,'mode':r['mode'],'seed':r['seed'],'success':bool(progress>=required and not o['collisions'][0]),'collision':sim.agents[0].collision_kind,'sim_time_s':(k+1)*.1,'progress_m':progress,'required_progress_m':required,'mean_abs_raceline_error_m':float(np.mean(np.abs(errors))),'max_abs_raceline_error_m':float(np.max(np.abs(errors))),'end_to_end_control_ms_p95':float(np.percentile(lat,95)),'braking_steps':interventions,'recovery_steps':recovery,'trace':trace,'scenario':str(p.relative_to(root))};rows.append(row);(out/f'evaluation{a.tag}_{a.split}.json').write_text(json.dumps(rows));print(controller,r['mode'],r['seed'],'ok',row['success'],'collision',row['collision'],'progress',round(progress,1),'time',row['sim_time_s'],flush=True)
summary={}
for c in a.controllers.split(','):
 rr=[r for r in rows if r['controller']==c];summary[c]={}
 for mode in ['all','corner_exit','mpcc','basic']:
  rs=rr if mode=='all' else [r for r in rr if r['mode']==mode];ok=[r for r in rs if r['success']];summary[c][mode]={'trials':len(rs),'successes':len(ok),'wall_collisions':sum(r['collision']=='wall' for r in rs),'obstacle_collisions':sum(r['collision']=='obstacle' for r in rs),'stops_or_timeouts':sum(not r['success'] and not r['collision'] for r in rs),'success_mean_sim_time_s':float(np.mean([r['sim_time_s'] for r in ok])) if ok else None,'mean_abs_raceline_error_m':float(np.mean([r['mean_abs_raceline_error_m'] for r in rs])) if rs else None,'end_to_end_ms_p95_mean':float(np.mean([r['end_to_end_control_ms_p95'] for r in rs])) if rs else None}
report={'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'summary':summary,'map':'hangar_1905_v0','no_resets':True,'counts_expert_failures_as_test_scenarios':True,'baseline_previous_model':a.baseline,'decoder':a.decoder,'time_basis':'fixed-step simulation; end-to-end CPU feature+inference+guard, excludes ROS I/O'};(out/f'evaluation{a.tag}_{a.split}_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
