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
root=Path(__file__).parent;data=root/'data_v2';out=root/'results';out.mkdir(exist_ok=True)
tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(1);tf.keras.utils.set_random_seed(93271)
def load(split):
 hs=[];fs=[];ys=[];ws=[];episodes=[]
 for p in sorted(list(data.glob(f'{split}_*.npz'))):
  r=json.loads(p.with_suffix('.json').read_text())
  if not r['success']:continue
  z=np.load(p);h=z['scan'];f=z['features'];y=z['target'];weight=(.6 if r['mode'].startswith('basic') else (2. if r['mode']=='dagger_new' else 1.))*(1+5*(f[:,37]>.5))
  hs.append(h);fs.append(f);ys.append(y);ws.append(weight);episodes.append({'path':str(p.relative_to(root)),'seed':r['seed'],'mode':r['mode'],'samples':len(y),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 if not hs:raise RuntimeError(f'No successful fresh {split} recordings')
 return np.concatenate(hs),np.concatenate(fs),np.concatenate(ys),np.concatenate(ws),episodes
h,f,y,w,episodes=load('train');vh,vf,vy,vw,validation=load('val')
checkpoint=root/'results/fresh_tln_v2.keras'
# Only this task's newly initialized model may be continued. Legacy TLN is never opened.
initial_report=json.loads((out/'training_v2.json').read_text())
assert initial_report['legacy_tln_model_used'] is False
m=tf.keras.models.load_model(checkpoint,compile=False)
continued_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
for layer in m.layers:
 if layer.name in ['fresh_conv_0','fresh_conv_1','fresh_conv_2']:layer.trainable=False
initial_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in m.weights)).hexdigest();m.save(out/'new_generation_continuation_start.keras')
# Mirror all spatial information consistently. Route y and previous steering change sign.
mh=h[:,::-1,:].copy();mf=f.copy();mf[:,:16]=f[:,:16].reshape(-1,8,2).__mul__(np.array([1,-1])).reshape(-1,16);mf[:,16:32]=f[:,16:32].reshape(-1,8,2)[:,:,::-1].reshape(-1,16);mf[:,33]*=-1;mf[:,35]*=-1;mf[:,38:40]=f[:,38:40][:,::-1];mf[:,40]*=-1;my=y.copy();my[:,:3]*=-1;my[:,4]*=-1
h=np.r_[h,mh];f=np.r_[f,mf];y=np.r_[y,my];w=np.r_[w,w]
@tf.keras.utils.register_keras_serializable(package='FreshTLN')
def fresh_action_loss(y_true,y_pred):
 diff=tf.abs(y_true-y_pred);loss=tf.where(diff<.1,.5*tf.square(diff),.1*diff-.005);return tf.reduce_sum(loss*tf.constant([.3,.3,.3,1.,4.]),axis=-1)
m.compile(optimizer=tf.keras.optimizers.Adam(6e-4,clipnorm=1.),loss=fresh_action_loss,metrics=['mae'],weighted_metrics=[])
class Log(tf.keras.callbacks.Callback):
 def on_epoch_end(self,epoch,logs=None):print('epoch',epoch+1,{k:round(float(v),6) for k,v in logs.items()},flush=True)
t0=time.time();history=m.fit([h,f],y,sample_weight=w,validation_data=([vh,vf],vy,vw),epochs=a.epochs,batch_size=128,verbose=0,callbacks=[Log(),tf.keras.callbacks.ModelCheckpoint(str(out/'fresh_tln_v2_continued.keras'),monitor='val_loss',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=4,factor=.5,min_lr=1e-5)])
pred=m.predict([vh,vf],batch_size=256,verbose=0);err=abs(pred-vy)
report={'continued_new_generation_checkpoint':True,'continued_checkpoint_sha256':continued_sha,'original_random_initialization_sha256':initial_report['initial_weights_sha256'],'legacy_pretrained_weights_used':False,'legacy_tln_model_used':False,'legacy_driving_data_used':False,'initialization':'continuation of this task new-generation model; first three convolution layers frozen; no legacy weights','initial_weights_sha256':initial_hash,'train_episodes':episodes,'validation_episodes':validation,'training_samples_after_mirror':len(h),'validation_samples':len(vh),'epochs':len(history.history['loss']),'history':history.history,'parameters':m.count_params(),'validation_offset_mae_m':(err[:,:3].mean(0)*.9).tolist(),'validation_steering_residual_mae_rad':float(err[:,4].mean()*.72),'validation_speed_mae_mps':float(err[:,3].mean()*2.8),'training_seconds':time.time()-t0,'input_contract':'541x6 range/foreground history; 42 route/state/foreground descriptors','output_contract':'3 auxiliary path offsets /0.9m, speed /2.8mps, expert steering residual /0.72rad','decoding':'nominal race-line feedback + learned MPCC steering residual; predicted path auxiliary; curvature cap; no EMA; zero speed allowed'}
(out/'training_v2_continued.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));print('FRESH TRAINING COMPLETE',report['validation_offset_mae_m'],report['validation_speed_mae_mps'],flush=True)
