"""New-generation on-policy correction data. No legacy models or recordings."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np,tensorflow as tf
from world import World
from expert import MPCC
from policy import plan_label
from perception import Inputs,decode_action,nominal_steer
root=Path(__file__).parent;out=root/'data_v2';w=World();sim=w.sim();tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
model=tf.keras.models.load_model(root/'results/fresh_tln_v2.keras',compile=False);predict=tf.function(lambda h,f:model([h,f],training=False));predict(tf.zeros((1,541,6)),tf.zeros((1,42)))
rows=[]
for seed in [2026,4000,4001,4002,4003,4004,4005]:
 obs,recipe,_=w.layouts(seed);o=w.configure(sim,obs);expert=MPCC(w,cruise=2.2);inputs=Inputs(w);last=np.zeros(2);prev=0.;progress=0;hs=[];fs=[];ys=[];rng=np.random.default_rng(seed);overrides=0;failed=0
 for k in range(600):
  st=sim.agents[0].state.copy();scan=o['scans'][0];base=w.route_features(st,previous_steer=last[0],s_hint=prev);h,f=inputs.push(scan,st,base);y=predict(h[None],f[None]).numpy()[0];student=decode_action(y,f);teacher,info=expert.act(st,scan,obs);label=plan_label(w,expert,info,st)
  if label is not None:
   target=label[0];target=np.r_[target,np.clip((teacher[0]-nominal_steer(f))/.72,-1,1)];hs.append(h);fs.append(f);ys.append(target);failed=0
  else:failed+=1
  override=label is not None and (rng.random()<.6 or abs(student[0]-teacher[0])>.08 or f[37]>.5);u=teacher if override else student;overrides+=int(override);last=u;o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2],hint=prev);progress+=(s-prev+w.length/2)%w.length-w.length/2;prev=s
  if o['collisions'][0] or progress>w.length-.5 or failed>15:break
 success=bool(progress>w.length-.5 and not o['collisions'][0]);r={'split':'train','mode':'dagger_new','seed':seed,'success':success,'collision':sim.agents[0].collision_kind,'samples':len(hs),'teacher_interventions':overrides,'progress_m':progress,'source':'current randomly-initialized generation only','legacy_model_used':False}
 np.savez_compressed(out/f'train_dagger_{seed}.npz',scan=np.asarray(hs,np.float32),features=np.asarray(fs,np.float32),target=np.asarray(ys,np.float32));(out/f'train_dagger_{seed}.json').write_text(json.dumps(r,indent=2));rows.append(r);(root/'results/dagger_new_collection.json').write_text(json.dumps(rows,indent=2));print(r,flush=True)
