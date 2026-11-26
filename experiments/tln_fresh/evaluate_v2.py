"""No-reset, episode-level evaluation. Never opens legacy TLN models."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json,time,argparse,hashlib
import numpy as np
import tensorflow as tf
from world import World,History
from policy import decode,basic_action
from guard_v2 import Guard
from perception import Inputs,decode_action
p=argparse.ArgumentParser();p.add_argument('--split',default='test',choices=['val','test']);p.add_argument('--model',default='fresh_tln_v2.keras');p.add_argument('--tag',default='');p.add_argument('--modes',default='neural,protected');a=p.parse_args();root=Path(__file__).parent;out=root/'results';w=World();sim=w.sim();tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
m=tf.keras.models.load_model(out/a.model,compile=False);predict=tf.function(lambda h,f:m([h,f],training=False),input_signature=[tf.TensorSpec((None,541,6),tf.float32),tf.TensorSpec((None,42),tf.float32)]);predict(tf.zeros((1,541,6)),tf.zeros((1,42)));results=[]
for mode in a.modes.split(','):
 for p in sorted((root/'data').glob(f'{a.split}_*.json')):
  r=json.loads(p.read_text());obs=r['obstacles'];start=r['start_s'];o=w.configure(sim,obs,start);inputs=Inputs(w);supervisor=Guard();prev=start;last=np.zeros(2);progress=0;stalled=0;trace=[];lat=[];interventions=0;errors=[]
  for k in range(700):
   st=sim.agents[0].state.copy();scan=o['scans'][0];base_features=w.route_features(st,previous_steer=last[0],s_hint=prev);h,f=inputs.push(scan,st,base_features);t=time.perf_counter()
   if mode=='tracker':u,_=basic_action(w,st,prev);detail={}
   else:
    y=predict(h[None],f[None]).numpy()[0]
    if mode=='protected':u,detail=supervisor.act(scan,f,y,inputs.foreground.observations);interventions+=int(detail['braking'])
    else:u=decode_action(y,f);detail={}
   lat.append((time.perf_counter()-t)*1000);errors.append(float(np.dot(st[:2]-w.xy(prev),w.normal(prev))));trace.append({'state':st.tolist(),'command':u.tolist(),'supervisor':detail});last=u;o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2],hint=prev);progress+=(s-prev+w.length/2)%w.length-w.length/2;prev=s;stalled=stalled+1 if sim.agents[0].state[3]<.05 and k>20 else 0
   if o['collisions'][0] or progress>w.length-.5 or stalled>40:break
  row={'controller':mode,'scenario_mode':r['mode'],'seed':r['seed'],'success':bool(progress>w.length-.5 and not o['collisions'][0]),'collision':sim.agents[0].collision_kind,'sim_time_s':(k+1)*.1,'progress_m':progress,'mean_abs_raceline_error_m':float(np.mean(np.abs(errors))),'max_abs_raceline_error_m':float(np.max(np.abs(errors))),'controller_ms_p50':float(np.median(lat)),'controller_ms_p95':float(np.percentile(lat,95)),'path_interventions':interventions,'trace':trace};results.append(row);(out/f'evaluation_v2{a.tag}_{a.split}.json').write_text(json.dumps(results));print(mode,r['mode'],r['seed'],'success',row['success'],'collision',row['collision'],'progress',round(progress,1),'time',row['sim_time_s'],flush=True)
summary={}
for mode in a.modes.split(','):
 rr=[r for r in results if r['controller']==mode];ok=[r for r in rr if r['success']];summary[mode]={'trials':len(rr),'successes':len(ok),'wall_collisions':sum(r['collision']=='wall' for r in rr),'obstacle_collisions':sum(r['collision']=='obstacle' for r in rr),'timeouts':sum(not r['success'] and not r['collision'] for r in rr),'clean_lap_mean_s':float(np.mean([r['sim_time_s'] for r in ok])) if ok else None,'mean_abs_raceline_error_m':float(np.mean([r['mean_abs_raceline_error_m'] for r in rr])),'controller_p95_ms_mean':float(np.mean([r['controller_ms_p95'] for r in rr]))}
report={'model_sha256':hashlib.sha256((out/a.model).read_bytes()).hexdigest(),'pretrained_weights_used':False,'split':a.split,'summary':summary};(out/f'evaluation_v2{a.tag}_{a.split}_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
