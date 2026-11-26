"""One audited DAgger round on training-only seeds; no test states/labels used."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json,time
import numpy as np
import tensorflow as tf
from world import World,History
from expert_v4 import MPCC
out=Path(__file__).parent/'results';tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
m=tf.keras.models.load_model(out/'tln_temporal_mpcc.keras',compile=False);w=World();sim=w.sim();reports=[]
for seed in range(400,406):
 rng=np.random.default_rng(seed);obs,recipe,_=w.layouts(seed);o=w.configure(sim,obs);e=MPCC(w);hist=History();h=[];f=[];y=[];overrides=0;prev=0.;progress=0;last_u=np.zeros(2);fail=0
 for k in range(500):
  st=sim.agents[0].state.copy();scan=o['scans'][0];hh=hist.push(scan);ff=w.route_features(st,previous_steer=last_u[0],s_hint=prev);pp=m([hh[None],ff[None]],training=False).numpy()[0];student=np.clip([pp[0]*.36,pp[1]*2.5],[-.36,0],[.36,2.5]);teacher,info=e.act(st,scan,obs)
  if info['ok']:
   h.append(hh);f.append(ff);y.append(teacher/[.36,2.5]);fail=0
  else:fail+=1
  use_expert=info['ok'] and (rng.random()<.45 or abs(student[0]-teacher[0])>.10 or w.clearance(st[0],st[1])<.35)
  u=teacher if use_expert else student;overrides+=int(use_expert);last_u=u;o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2],hint=prev);progress+=(s-prev+w.length/2)%w.length-w.length/2;prev=s
  if o['collisions'][0] or progress>w.length-.5 or fail>15:break
 if h:np.savez_compressed(out/f'dagger_train_{seed}.npz',scan=np.asarray(h,np.float32),features=np.asarray(f,np.float32),target=np.asarray(y,np.float32))
 r={'seed':seed,'samples':len(h),'teacher_interventions':overrides,'collision':sim.agents[0].collision_kind,'progress_m':progress,'solver_failures':e.failures,'success':bool(progress>w.length-.5 and not o['collisions'][0])};reports.append(r);(out/'dagger_collection.json').write_text(json.dumps(reports,indent=2));print(r,flush=True)
