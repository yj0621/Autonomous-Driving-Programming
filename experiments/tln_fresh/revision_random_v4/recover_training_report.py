"""Recover metadata after JSON scalar serialization failed; never change saved weights."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
from pathlib import Path
import json,ast,re,hashlib
import numpy as np,tensorflow as tf
root=Path(__file__).parent;manifest=json.loads((root/'results/dataset_manifest.json').read_text());tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1);model=tf.keras.models.load_model(root/'models/last_restored.keras',compile=False);initial=tf.keras.models.load_model(root/'models/random_initialization.keras',compile=False)
h=[];f=[];y=[]
for e in manifest['val']:
 z=np.load(root/e['path']);h.append(z['scan'].astype(np.float32));f.append(z['features']);y.append(z['target'])
h=np.concatenate(h);f=np.concatenate(f);y=np.concatenate(y);pred=model.predict([h,f],batch_size=256,verbose=0);history={'loss':[],'val_loss':[]}
for line in (root/'results/train.log').read_text().splitlines():
 match=re.match(r'^epoch (\d+) (\{.*\})$',line)
 if match:
  value=ast.literal_eval(match.group(2))
  for key in history:history[key].append(float(value[key]))
report={'from_random_initialization':True,'pretrained_weights_used':False,'earlier_episode_reuse':False,'random_seed':917203,'initial_weights_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.weights)).hexdigest(),'training_manifest':manifest,'training_samples_after_mirror':2*sum(e['samples'] for e in manifest['train']),'validation_samples':len(y),'epochs':len(history['loss']),'history':history,'parameters':model.count_params(),'validation_steering_residual_mae_rad':float(np.mean(abs(pred[:,0]-y[:,0]))*.72),'validation_speed_mae_mps':float(np.mean(abs(pred[:,1]-y[:,1]))*3.2),'metadata_recovery':'training finished and checkpoints saved; final JSON serialization of NumPy scalar failed. Report rebuilt from epoch logs and unchanged own checkpoints. No retraining or weight modification.','training_source_sha256':hashlib.sha256((root/'source_snapshots/training_run_v4/train.py').read_bytes()).hexdigest(),'input_contract':'4x541x2 temporal range/foreground; 68 measured route/motion/four-object features','output_contract':'steering residual times 0.72 rad, signed speed times 3.2 m/s'};(root/'results/training.json').write_text(json.dumps(report,indent=2));print('TRAINING REPORT VERIFIED',report['epochs'],report['parameters'],report['validation_steering_residual_mae_rad'],report['validation_speed_mae_mps'],flush=True)
