"""Select only by validation error, then freeze before held-out evaluation."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,shutil
from pathlib import Path
import numpy as np,tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
out=Path(__file__).parent/'results';hh=[];ff=[];yy=[];ww=[];seeds=[]
for p in list(out.glob('val_*.npz'))+list((out/'refined').glob('val_*.npz')):
 r=json.loads(p.with_suffix('.json').read_text())
 if not r['success']:continue
 z=np.load(p);hh.append(z['scan']);ff.append(z['features']);yy.append(z['target']);ww.append(1+1.5*(z['info'][:,1]>0));seeds.append(r['seed'])
h,f,y,w=map(np.concatenate,[hh,ff,yy,ww]);rows=[]
for name in ['tln_temporal_mpcc_bc.keras','tln_temporal_mpcc_dagger.keras','tln_temporal_mpcc_refined.keras']:
 p=out/name
 if not p.exists():continue
 m=tf.keras.models.load_model(p,compile=False);pred=m.predict([h,f],batch_size=256,verbose=0);loss=float(np.mean(tf.keras.losses.Huber(delta=.2,reduction='none')(y,pred).numpy()*w));error=abs(y-pred)
 rows.append({'checkpoint':name,'weighted_huber':loss,'steer_mae_rad':float(error[:,0].mean()*.36),'speed_mae_mps':float(error[:,1].mean()*2.5)})
best=min(rows,key=lambda x:x['weighted_huber']);shutil.copy2(out/best['checkpoint'],out/'tln_temporal_mpcc.keras');r={'criterion':'minimum validation weighted Huber; no test metrics used','validation_seeds':seeds,'validation_samples':len(y),'candidates':rows,'selected':best};(out/'model_selection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
