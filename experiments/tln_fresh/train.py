"""Train a NEW temporal LiDAR path policy entirely from random initialization.
Never imports, loads, copies or resumes any existing TLN/checkpoint weights.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json,time,hashlib,argparse
import numpy as np
import tensorflow as tf
p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=80);a=p.parse_args()
root=Path(__file__).parent;data=root/'data';out=root/'results';out.mkdir(exist_ok=True)
tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1);tf.keras.utils.set_random_seed(93271)
def load(split):
 hs=[];fs=[];ys=[];ws=[];episodes=[]
 for p in sorted(list(data.glob(f'{split}_*.npz'))+list((data/'ros_fast').glob(f'{split}_*.npz'))):
  r=json.loads(p.with_suffix('.json').read_text())
  if not r['success']:continue
  z=np.load(p);h=z['scan'];f=z['features'];y=z['target'];weight=(.7 if r['mode']=='basic' else 1.)*(1+4*(np.max(abs(y[:,:3]),axis=1)>.10))
  hs.append(h);fs.append(f);ys.append(y);ws.append(weight);episodes.append({'path':str(p.relative_to(root)),'seed':r['seed'],'mode':r['mode'],'samples':len(y),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 if not hs:raise RuntimeError(f'No successful fresh {split} recordings')
 return np.concatenate(hs),np.concatenate(fs),np.concatenate(ys),np.concatenate(ws),episodes
h,f,y,w,episodes=load('train');vh,vf,vy,vw,validation=load('val')
scan=tf.keras.Input((541,3),name='scan_history');state=tf.keras.Input((34,),name='route_state');x=scan
for i,(filters,kernel,stride) in enumerate([(16,9,2),(24,7,2),(32,5,2),(48,3,2)]):
 x=tf.keras.layers.Conv1D(filters,kernel,strides=stride,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-5),name=f'fresh_conv_{i}')(x)
x=tf.keras.layers.Flatten()(x);x=tf.keras.layers.Dense(96,activation='relu')(x);r=tf.keras.layers.Dense(48,activation='relu')(state);x=tf.keras.layers.Concatenate()([x,r]);x=tf.keras.layers.Dense(96,activation='relu')(x);x=tf.keras.layers.Dropout(.05)(x);x=tf.keras.layers.Dense(48,activation='relu')(x)
path=tf.keras.layers.Dense(3,activation='tanh',kernel_initializer='zeros',bias_initializer='zeros',name='future_path_offsets')(x);speed=tf.keras.layers.Dense(1,activation='sigmoid',name='target_speed')(x)
m=tf.keras.Model([scan,state],tf.keras.layers.Concatenate(name='path_and_speed')([path,speed]),name='fresh_temporal_lidar_path_policy')
initial_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in m.weights)).hexdigest();m.save(out/'random_initialization.keras')
# Mirror all spatial information consistently. Route y and previous steering change sign.
mh=h[:,::-1,:].copy();mf=f.copy();mf[:,:16]=f[:,:16].reshape(-1,8,2).__mul__(np.array([1,-1])).reshape(-1,16);mf[:,16:32]=f[:,16:32].reshape(-1,8,2)[:,:,::-1].reshape(-1,16);mf[:,33]*=-1;my=y.copy();my[:,:3]*=-1
h=np.r_[h,mh];f=np.r_[f,mf];y=np.r_[y,my];w=np.r_[w,w]
m.compile(optimizer=tf.keras.optimizers.Adam(6e-4,clipnorm=1.),loss=tf.keras.losses.Huber(delta=.15),metrics=['mae'],weighted_metrics=[])
class Log(tf.keras.callbacks.Callback):
 def on_epoch_end(self,epoch,logs=None):print('epoch',epoch+1,{k:round(float(v),6) for k,v in logs.items()},flush=True)
t0=time.time();history=m.fit([h,f],y,sample_weight=w,validation_data=([vh,vf],vy,vw),epochs=a.epochs,batch_size=128,verbose=0,callbacks=[Log(),tf.keras.callbacks.ModelCheckpoint(str(out/'fresh_tln.keras'),monitor='val_loss',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=4,factor=.5,min_lr=1e-5)])
pred=m.predict([vh,vf],batch_size=256,verbose=0);err=abs(pred-vy)
report={'pretrained_weights_used':False,'legacy_tln_model_used':False,'legacy_driving_data_used':False,'initialization':'random seed 93271; zero path head to start at raceline; all layers trainable','initial_weights_sha256':initial_hash,'train_episodes':episodes,'validation_episodes':validation,'training_samples_after_mirror':len(h),'validation_samples':len(vh),'epochs':len(history.history['loss']),'history':history.history,'parameters':m.count_params(),'validation_offset_mae_m':(err[:,:3].mean(0)*.9).tolist(),'validation_speed_mae_mps':float(err[:,3].mean()*2.8),'training_seconds':time.time()-t0,'input_contract':'541x3 normalized scans; 34 route/state features','output_contract':'3 lateral offsets /0.9m at 0.75,1.5,2.5m and speed /2.8mps','decoding':'race-line preview + learned offsets, geometric path tracking, curvature speed cap; no old decoder, no EMA, no positive minimum speed'}
(out/'training.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));print('FRESH TRAINING COMPLETE',report['validation_offset_mae_m'],report['validation_speed_mae_mps'],flush=True)
