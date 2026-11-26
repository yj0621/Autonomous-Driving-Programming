"""Temporal TLN transfer learning with route-conditioning and physical-unit output contract."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time,argparse
import numpy as np
import tensorflow as tf
from world import ROOT
from tracking import nominal
p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=55);p.add_argument('--resume',action='store_true');p.add_argument('--stage',choices=['dagger','refined'],default='dagger');a=p.parse_args()
tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1);tf.keras.utils.set_random_seed(417)
out=Path(__file__).parent/'results'
def load(split):
 hs=[];fs=[];ys=[];ws=[];used=[]
 for p in sorted(list(out.glob(f'{split}_*.npz'))+list((out/'refined').glob(f'{split}_*.npz'))):
  r=json.loads(p.with_suffix('.json').read_text())
  if not r['success']:continue
  z=np.load(p);hs.append(z['scan']);fs.append(z['features']);ys.append(z['target']);ws.append(1+1.5*(z['info'][:,1]>0));used.append(int(r['seed']))
 if not hs:raise RuntimeError(f'No successful episodes in {split}')
 return np.concatenate(hs),np.concatenate(fs),np.concatenate(ys),np.concatenate(ws),used
h,f,y,weights,used=load('train');vh,vf,vy,vw,vused=load('val')
y[:,0]=(y[:,0]*.36-nominal(f))/.72;vy[:,0]=(vy[:,0]*.36-nominal(vf))/.72
m=tf.keras.models.load_model(out/'tln_temporal_mpcc.keras',compile=False)
layer=m.get_layer('steer_fraction');ww=layer.get_weights();layer.set_weights([np.zeros_like(ww[0]),np.zeros_like(ww[1])])
m.compile(optimizer=tf.keras.optimizers.Adam(2e-4,clipnorm=1.),loss=tf.keras.losses.Huber(delta=.2),metrics=['mae'])
# Mirror scan, local y, left/right widths and signed steering consistently.
mh=h[:,::-1,:].copy();mf=f.copy();mf[:,:16]=f[:,:16].reshape(-1,8,2).__mul__(np.array([1,-1])).reshape(-1,16);mf[:,16:32]=f[:,16:32].reshape(-1,8,2)[:,:,::-1].reshape(-1,16);mf[:,33]*=-1;my=y.copy();my[:,0]*=-1
h=np.concatenate([h,mh]);f=np.concatenate([f,mf]);y=np.concatenate([y,my]);weights=np.r_[weights,weights]
# Also aggregate explicitly identified student-visited states if available; never use test.
for pth in sorted(out.glob('dagger_train_*.npz')):
 z=np.load(pth);h=np.concatenate([h,z['scan']]);f=np.concatenate([f,z['features']]);zy=z['target'].copy();zy[:,0]=(zy[:,0]*.36-nominal(z['features']))/.72;y=np.concatenate([y,zy]);weights=np.r_[weights,np.full(len(z['target']),2.)]
class Log(tf.keras.callbacks.Callback):
 def on_epoch_end(self,epoch,logs=None):print('epoch',epoch+1,{k:round(float(v),5) for k,v in logs.items()},flush=True)
checkpoint=out/'tln_residual_mpcc.keras'
previous_val=float(m.evaluate([vh,vf],vy,verbose=0,return_dict=True)['loss']) if a.resume else None
t0=time.time();hist=m.fit([h,f],y,sample_weight=weights,validation_data=([vh,vf],vy,vw),epochs=a.epochs,batch_size=128,verbose=0,callbacks=[Log(),tf.keras.callbacks.ModelCheckpoint(str(checkpoint),monitor='val_loss',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=4,factor=.5,min_lr=1e-5)])
m=tf.keras.models.load_model(checkpoint,compile=False);pred=m.predict([vh,vf],batch_size=256,verbose=0);err=np.abs(pred-vy)
times=[]
for _ in range(60):
 t=time.perf_counter();m([vh[:1],vf[:1]],training=False).numpy();times.append((time.perf_counter()-t)*1000)
report={'selected_checkpoint':str(checkpoint.name),'previous_validation_loss':previous_val,'seed':417,'train_episode_seeds':used,'validation_episode_seeds':vused,'train_samples_augmented':len(h),'validation_samples':len(vh),'epochs':len(hist.history['loss']),'history':hist.history,'validation_steer_mae_rad':float(err[:,0].mean()*.72),'validation_speed_mae_mps':float(err[:,1].mean()*2.5),'eager_inference_ms_p50':float(np.median(times)),'eager_inference_ms_p95':float(np.percentile(times,95)),'training_seconds':time.time()-t0,'parameters':m.count_params(),'route_conditioned':True,'inference_ema':False,'outputs':'nominal raceline feedback + steer_fraction*.72 rad; speed_fraction*2.5 m/s; clip only; zero allowed'}
(out/'training_residual.json').write_text(json.dumps(report,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)));print('TRAINING COMPLETE',report['validation_steer_mae_rad'],report['validation_speed_mae_mps'],flush=True)
