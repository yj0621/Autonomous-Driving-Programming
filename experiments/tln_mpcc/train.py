"""Temporal TLN transfer learning with route-conditioning and physical-unit output contract."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time,argparse
import numpy as np
import tensorflow as tf
from world import ROOT
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
if a.resume and (out/'tln_temporal_mpcc.keras').exists():m=tf.keras.models.load_model(out/'tln_temporal_mpcc.keras',compile=False)
else:
 old=tf.keras.models.load_model(ROOT/'ofc/ofc/models/TLN_aug.keras',compile=False,safe_mode=False)
 scan=tf.keras.Input((541,3),name='scan_history');feat=tf.keras.Input((34,),name='route_state');x=scan
 for i,(filters,kernel,stride) in enumerate([(24,10,4),(36,8,4),(48,4,2),(64,3,1),(64,3,1)]):x=tf.keras.layers.Conv1D(filters,kernel,strides=stride,activation='relu',name=f'tln_conv_{i}')(x)
 x=tf.keras.layers.Flatten()(x);x=tf.keras.layers.Dense(100,activation='relu',name='tln_dense')(x)
 route=tf.keras.layers.Dense(48,activation='relu',name='route_embedding')(feat);x=tf.keras.layers.Concatenate()([x,route]);x=tf.keras.layers.Dense(96,activation='relu')(x);x=tf.keras.layers.Dense(48,activation='relu')(x)
 steer=tf.keras.layers.Dense(1,activation='tanh',name='steer_fraction')(x);speed=tf.keras.layers.Dense(1,activation='sigmoid',name='speed_fraction')(x);m=tf.keras.Model([scan,feat],tf.keras.layers.Concatenate(name='action_fraction')([steer,speed]))
 convs=[l for l in old.layers if isinstance(l,tf.keras.layers.Conv1D)]
 for i,l in enumerate(convs):
  ww=l.get_weights()
  if i==0:ww[0]=np.repeat(ww[0],3,axis=1)*10/3 # input raw metres -> normalized history; replicate old current-frame response initially.
  m.get_layer(f'tln_conv_{i}').set_weights(ww)
 m.get_layer('tln_dense').set_weights(next(l for l in old.layers if isinstance(l,tf.keras.layers.Dense)).get_weights())
m.compile(optimizer=tf.keras.optimizers.Adam(2e-4,clipnorm=1.),loss=tf.keras.losses.Huber(delta=.2),metrics=['mae'])
# Mirror scan, local y, left/right widths and signed steering consistently.
mh=h[:,::-1,:].copy();mf=f.copy();mf[:,:16]=f[:,:16].reshape(-1,8,2).__mul__(np.array([1,-1])).reshape(-1,16);mf[:,16:32]=f[:,16:32].reshape(-1,8,2)[:,:,::-1].reshape(-1,16);mf[:,33]*=-1;my=y.copy();my[:,0]*=-1
h=np.concatenate([h,mh]);f=np.concatenate([f,mf]);y=np.concatenate([y,my]);weights=np.r_[weights,weights]
# Also aggregate explicitly identified student-visited states if available; never use test.
for pth in sorted(out.glob('dagger_train_*.npz')):
 z=np.load(pth);h=np.concatenate([h,z['scan']]);f=np.concatenate([f,z['features']]);y=np.concatenate([y,z['target']]);weights=np.r_[weights,np.full(len(z['target']),2.)]
class Log(tf.keras.callbacks.Callback):
 def on_epoch_end(self,epoch,logs=None):print('epoch',epoch+1,{k:round(float(v),5) for k,v in logs.items()},flush=True)
checkpoint=out/('tln_temporal_mpcc_'+a.stage+'.keras' if a.resume else 'tln_temporal_mpcc.keras')
previous_val=float(m.evaluate([vh,vf],vy,verbose=0,return_dict=True)['loss']) if a.resume else None
t0=time.time();hist=m.fit([h,f],y,sample_weight=weights,validation_data=([vh,vf],vy,vw),epochs=a.epochs,batch_size=128,verbose=0,callbacks=[Log(),tf.keras.callbacks.ModelCheckpoint(str(checkpoint),monitor='val_loss',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=4,factor=.5,min_lr=1e-5)])
m=tf.keras.models.load_model(checkpoint,compile=False);pred=m.predict([vh,vf],batch_size=256,verbose=0);err=np.abs(pred-vy)
times=[]
for _ in range(60):
 t=time.perf_counter();m([vh[:1],vf[:1]],training=False).numpy();times.append((time.perf_counter()-t)*1000)
report={'selected_checkpoint':str(checkpoint.name),'previous_validation_loss':previous_val,'seed':417,'train_episode_seeds':used,'validation_episode_seeds':vused,'train_samples_augmented':len(h),'validation_samples':len(vh),'epochs':len(hist.history['loss']),'history':hist.history,'validation_steer_mae_rad':float(err[:,0].mean()*.36),'validation_speed_mae_mps':float(err[:,1].mean()*2.5),'eager_inference_ms_p50':float(np.median(times)),'eager_inference_ms_p95':float(np.percentile(times,95)),'training_seconds':time.time()-t0,'parameters':m.count_params(),'route_conditioned':True,'inference_ema':False,'outputs':'steer_fraction*.36 rad; speed_fraction*2.5 m/s; clip only; zero allowed'}
if a.resume:
 import shutil
 shutil.copy2(checkpoint,out/'tln_temporal_mpcc.keras')
(out/('training_'+a.stage+'.json' if a.resume else 'training.json')).write_text(json.dumps(report,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)));print('TRAINING COMPLETE',report['validation_steer_mae_rad'],report['validation_speed_mae_mps'],flush=True)
