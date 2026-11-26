"""Recover initial training metrics after the checkpoint was saved but JSON lr serialization failed."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import re,ast,json,time
from pathlib import Path
import numpy as np,tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
out=Path(__file__).parent/'results';hist=[]
for line in Path('/tmp/tln_mpcc_train.log').read_text().splitlines():
 if line.startswith('epoch '):hist.append(ast.literal_eval(line[line.index('{'):]))
model=tf.keras.models.load_model(out/'tln_temporal_mpcc_bc.keras',compile=False);val=[];vf=[];vy=[];trainseeds=[];valseeds=[];n=0
for p in out.glob('train_*.npz'):
 r=json.loads(p.with_suffix('.json').read_text())
 if r['success']:trainseeds.append(r['seed']);n+=len(np.load(p)['target'])
for p in out.glob('val_*.npz'):
 r=json.loads(p.with_suffix('.json').read_text())
 if r['success']:
  z=np.load(p);val.append(z['scan']);vf.append(z['features']);vy.append(z['target']);valseeds.append(r['seed'])
h,f,y=map(np.concatenate,[val,vf,vy]);pred=model.predict([h,f],batch_size=256,verbose=0);err=abs(pred-y);times=[]
for _ in range(40):
 t=time.perf_counter();model([h[:1],f[:1]],training=False).numpy();times.append((time.perf_counter()-t)*1000)
r={'train_episode_seeds':sorted(trainseeds),'validation_episode_seeds':sorted(valseeds),'train_samples_before_mirror':n,'train_samples_after_mirror':2*n,'validation_samples':len(y),'epochs':len(hist),'history':hist,'validation_steer_mae_rad':float(err[:,0].mean()*.36),'validation_speed_mae_mps':float(err[:,1].mean()*2.5),'inference_ms_p50':float(np.median(times)),'inference_ms_p95':float(np.percentile(times,95)),'parameter_count':model.count_params(),'report_recovered_after_numpy_scalar_serialization_error':True}
(out/'training.json').write_text(json.dumps(r,indent=2));print(r['validation_steer_mae_rad'],r['validation_speed_mae_mps'],flush=True)
