"""Paired, no-reset evaluation on held-out scenario seeds. Test never feeds training."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import json,time,argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from world import World,History,preprocess,ROOT,DT
from expert import MPCC
from safety import shield
p=argparse.ArgumentParser();p.add_argument('--models',default='baseline,baseline_slow,student');p.add_argument('--suffix',default='');a=p.parse_args()
tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
out=Path(__file__).parent/'results';manifest=json.loads((out/'manifest.json').read_text());seeds=manifest['splits']['test'];w=World();sim=w.sim();base=tf.keras.models.load_model(ROOT/'ofc/ofc/models/TLN_aug.keras',compile=False,safe_mode=False)
student=tf.keras.models.load_model(out/'tln_temporal_mpcc.keras',compile=False) if 'student' in a.models else None
base_predict=tf.function(lambda x:base(x,training=False),input_signature=[tf.TensorSpec((None,541,1),tf.float32)])
base_predict(tf.zeros((1,541,1)))
if student is not None:
 student_predict=tf.function(lambda x,f:student([x,f],training=False),input_signature=[tf.TensorSpec((None,541,3),tf.float32),tf.TensorSpec((None,34),tf.float32)])
 student_predict(tf.zeros((1,541,3)),tf.zeros((1,34)))
results=[]
for name in a.models.split(','):
 for seed in seeds:
  # Use the expert manifest's immutable initial layout, not a re-draw.
  report=json.loads((out/f'test_{seed}.json').read_text());obs=report['obstacles'];o=w.configure(sim,obs,s=0);hist=History();prev=0.;progress=0;ema=0.;last_u=np.zeros(2);trajectory=[];commands=[];times=[];clearance=10;stopped=0;shield_count=0;lateral_errors=[];min_wall=100.;min_obstacle=100.;expert=MPCC(w) if name=='expert' else None
  for k in range(600):
   st=sim.agents[0].state.copy();scan=o['scans'][0];t=time.perf_counter()
   if name in ('student','student_safe'):
    h=hist.push(scan);f=w.route_features(st,previous_steer=last_u[0],s_hint=prev);y=student_predict(h[None],f[None]).numpy()[0];u=np.clip([y[0]*.36,y[1]*2.5],[-.36,0],[.36,2.5])
   elif name=='expert':u,info=expert.act(st,scan,obs)
   else:
    y=base_predict((preprocess(scan)*10).reshape(1,541,1)).numpy()[0];ema=.3*ema+.7*y[0]
    vmin,vmax=(3,6.5) if name=='baseline' else (0.,2.5)
    speed=vmin+(float(y[1])+1)*.5*(vmax-vmin);speed*=1-.65*min(abs(ema)/.36,1)
    u=np.array([np.clip(ema,-.36,.36),np.clip(speed,vmin,vmax)])
   if name=='student_safe':
    u,safety=shield(scan,u);shield_count+=int(safety['limited'])
   times.append((time.perf_counter()-t)*1000)
   sp=w.project(st[0],st[1],hint=prev);lateral_errors.append(float(np.dot(st[:2]-w.xy(sp),w.normal(sp))))
   for offset in [-.16,.06,.28]:
    centre=st[:2]+offset*np.array([np.cos(st[4]),np.sin(st[4])]);min_wall=min(min_wall,float(w.clearance(*centre))-.16)
    for cube in obs:
     dx,dy=centre-[cube['x'],cube['y']];co,si=np.cos(cube['theta']),np.sin(cube['theta']);local=np.array([co*dx+si*dy,-si*dx+co*dy]);q=np.maximum(abs(local)-.15,0);min_obstacle=min(min_obstacle,float(np.linalg.norm(q))-.16)
   trajectory.append(st.tolist());commands.append(u.tolist());clearance=min(clearance,float(w.clearance(st[0],st[1]))-.29);last_u=u;o=w.step(sim,u)
   s=w.project(*sim.agents[0].state[:2],hint=prev);delta=(s-prev+w.length/2)%w.length-w.length/2
   progress+=delta;prev=s;stopped=stopped+1 if sim.agents[0].state[3]<.05 and k>20 else 0
   if o['collisions'][0] or progress>w.length-.5 or stopped>50:break
  rep={'shield_interventions':shield_count,'mean_abs_raceline_error_m':float(np.mean(abs(np.asarray(lateral_errors)))),'max_abs_raceline_error_m':float(np.max(abs(np.asarray(lateral_errors)))),'min_envelope_wall_clearance_m':min_wall,'min_envelope_obstacle_clearance_m':min_obstacle,'compiled_tf_function':True,'model':name,'seed':seed,'success':bool(progress>w.length-.5 and not o['collisions'][0]),'collision':sim.agents[0].collision_kind,'progress_m':progress,'sim_time_s':(k+1)*DT,'inference_ms_p50':float(np.median(times)),'inference_ms_p95':float(np.percentile(times,95)),'min_wall_center_clearance_minus_half_length':clearance,'mean_speed_mps':float(np.mean(np.array(trajectory)[:,3])),'stopped_timeout':stopped>50,'trajectory':trajectory,'commands':commands};results.append(rep)
  (out/f'evaluation{a.suffix}.json').write_text(json.dumps(results,indent=2));print(name,seed,'success',rep['success'],'collision',rep['collision'],'progress',round(progress,1),'time',rep['sim_time_s'],flush=True)
summary={}
for name in a.models.split(','):
 r=[x for x in results if x['model']==name];ok=[x for x in r if x['success']];summary[name]={'trials':len(r),'successes':len(ok),'success_rate':len(ok)/len(r),'wall_collisions':sum(x['collision']=='wall' for x in r),'obstacle_collisions':sum(x['collision']=='obstacle' for x in r),'timeouts':sum(not x['success'] and not x['collision'] for x in r),'mean_progress_m':float(np.mean([x['progress_m'] for x in r])),'clean_lap_mean_s':float(np.mean([x['sim_time_s'] for x in ok])) if ok else None,'mean_abs_raceline_error_m':float(np.mean([x['mean_abs_raceline_error_m'] for x in r])),'inference_p95_ms_mean':float(np.mean([x['inference_ms_p95'] for x in r]))}
(out/f'evaluation_summary{a.suffix}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
