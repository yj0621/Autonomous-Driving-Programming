"""Fresh temporal CNN/GRU from random weights, no legacy model/data imports."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time,hashlib,argparse
import numpy as np,tensorflow as tf
p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=90);args=p.parse_args();root=Path(__file__).parent;out=root/'models_mixture';out.mkdir(exist_ok=True);manifest=json.loads((root/'results/dataset_manifest_mixture.json').read_text());tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1);tf.keras.utils.set_random_seed(917203)
def load(split):
 hs=[];fs=[];ys=[];ws=[]
 for e in manifest[split]:
  z=np.load(root/e['path']);h=z['scan'].astype(np.float32);f=z['features'];y=z['target'];near=np.max(f[:,34:66].reshape(-1,4,8)[:,:,3],axis=1)>.5;weight=np.where(near,2.8,1.)*np.where(y[:,1]<0,2.,1.)*np.where(y[:,1]>.75,1.3,1.)
  hs.append(h);fs.append(f);ys.append(y);ws.append(weight)
 return np.concatenate(hs),np.concatenate(fs),np.concatenate(ys),np.concatenate(ws)
h,f,y,w=load('train');vh,vf,vy,vw=load('val')
mh=h[:,:,::-1,:].copy();mf=f.copy();mf[:,:16]=(f[:,:16].reshape(-1,8,2)*np.array([1,-1])).reshape(-1,16)
mf[:,16:32]=f[:,16:32].reshape(-1,8,2)[:,:,::-1].reshape(-1,16);mf[:,33]*=-1;mf[:,66]*=-1;ob=f[:,34:66].reshape(-1,4,8).copy();ob[:,:,1]*=-1;ob[:,:,[4,5]]=ob[:,:,[5,4]];ob[:,:,6]*=-1;mf[:,34:66]=ob.reshape(-1,32);my=y.copy();my[:,0]*=-1;my[:,2:7]*=-1;my[:,12:14]=y[:,12:14][:,::-1];mf[:,68:76]=(f[:,68:76].reshape(-1,4,2)*np.array([1,-1])).reshape(-1,8);mf[:,76:84]=f[:,76:84].reshape(-1,4,2)[:,:,::-1].reshape(-1,8)
h=np.r_[h,mh];f=np.r_[f,mf];y=np.r_[y,my];w=np.r_[w,w]
history=tf.keras.Input((4,541,2),name='lidar_foreground_history');features=tf.keras.Input((84,),name='route_motion_objects');x=history
for i,(filters,kernel,stride) in enumerate([(12,9,4),(20,5,4),(24,3,2)]):x=tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters,kernel,strides=stride,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(5e-6)),name=f'shared_scan_conv_{i}')(x)
x=tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x);x=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(48,activation='relu'))(x);x=tf.keras.layers.GRU(32,name='temporal_motion_encoder')(x);r=tf.keras.layers.Dense(64,activation='relu')(features);r=tf.keras.layers.Dense(48,activation='relu')(r);x=tf.keras.layers.Concatenate()([x,r]);x=tf.keras.layers.Dense(96,activation='relu')(x);x=tf.keras.layers.Dense(48,activation='relu')(x);x=tf.keras.layers.Dropout(.03)(x)
steering=tf.keras.layers.Dense(1,activation='tanh',kernel_initializer='zeros',bias_initializer='zeros',name='steering_residual')(x);speed=tf.keras.layers.Dense(1,activation='tanh',bias_initializer=tf.keras.initializers.Constant(float(np.arctanh(1.5/3.2))),name='signed_target_speed')(x);path_left=tf.keras.layers.Dense(5,activation='tanh',kernel_initializer='zeros',bias_initializer='zeros',name='left_conditioned_path')(x);path_right=tf.keras.layers.Dense(5,activation='tanh',kernel_initializer='zeros',bias_initializer='zeros',name='right_conditioned_path')(x);side=tf.keras.layers.Dense(3,activation='softmax',name='passing_side')(x);model=tf.keras.Model([history,features],tf.keras.layers.Concatenate()([steering,speed,path_left,path_right,side]),name='fresh_tln_random_layouts_temporal_gru')
initial_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in model.weights)).hexdigest();model.save(out/'random_initialization.keras')
@tf.keras.utils.register_keras_serializable(package='FreshRandomTLN')
def action_loss(truth,pred):
 chosen_path=truth[:,12:13]*pred[:,2:7]+truth[:,13:14]*pred[:,7:12];continuous=tf.concat([pred[:,:2],chosen_path],axis=1);d=tf.abs(truth[:,:7]-continuous);threshold=tf.constant([.05,.10,.08,.08,.08,.08,.08]);loss=tf.where(d<threshold,.5*d*d,threshold*d-.5*threshold*threshold);mask=tf.concat([tf.ones_like(truth[:,:2]),truth[:,7:12]*(truth[:,12:13]+truth[:,13:14])],axis=1);regression=tf.reduce_sum(loss*tf.constant([4.,1.5,1.,1.,1.,1.,1.])*mask,axis=-1);classification=-tf.reduce_sum(truth[:,12:15]*tf.math.log(tf.clip_by_value(pred[:,12:15],1e-6,1.)),axis=-1)*truth[:,15];return regression+.05*classification

model.compile(optimizer=tf.keras.optimizers.Adam(5e-4,clipnorm=1.),loss=action_loss)
class Progress(tf.keras.callbacks.Callback):
 def on_epoch_end(self,epoch,logs=None):
  print('epoch',epoch+1,{k:round(float(v),6) for k,v in (logs or {}).items()},flush=True)
  if (epoch+1)%20==0:self.model.save(out/f'candidate_epoch_{epoch+1}.keras')
t=time.perf_counter();history=model.fit([h,f],y,sample_weight=w,validation_data=([vh,vf],vy,vw),epochs=args.epochs,batch_size=128,verbose=0,callbacks=[Progress(),tf.keras.callbacks.ModelCheckpoint(str(out/'best_regression.keras'),save_best_only=True,monitor='val_loss'),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=4,min_lr=1e-5)]);model.save(out/'last_restored.keras');prediction=model.predict([vh,vf],batch_size=256,verbose=0)
report={'from_random_initialization':True,'pretrained_weights_used':False,'earlier_episode_reuse':False,'random_seed':917203,'initial_weights_sha256':initial_hash,'training_manifest':manifest,'training_samples_after_mirror':len(y),'validation_samples':len(vy),'epochs':len(history.history['loss']),'history':history.history,'parameters':model.count_params(),'training_wall_s':time.perf_counter()-t,'validation_steering_residual_mae_rad':float(np.mean(abs(prediction[:,0]-vy[:,0]))*.72),'validation_speed_mae_mps':float(np.mean(abs(prediction[:,1]-vy[:,1]))*3.2),'input_contract':'4 x 541 x 2 temporal range/foreground; 84 measured route/motion/four-object/9m-preview features','output_contract':'residual steering x0.72 rad, speed x3.2 m/s, left/right conditional five-point paths x1.2 m and passing-side probabilities; future labels and masks never enter inference','final_selection':'choose using closed-loop validation, never test outcomes'};(root/'results/training_mixture.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));print('TRAINING DONE',report['epochs'],report['validation_steering_residual_mae_rad'],report['validation_speed_mae_mps'],flush=True)
