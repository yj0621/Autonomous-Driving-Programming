"""Freeze successful fresh recordings; rebuild identical online inputs for basic data."""
from pathlib import Path
import sys,json,hashlib
root=Path(__file__).parent;sys.path.insert(0,str(root.parent));sys.path.insert(0,str(root))
import numpy as np
from world import World
from sensors import Inputs,nominal_steer,SPEED_SCALE
w=World();out=root/'prepared';out.mkdir(exist_ok=True);manifest={'train':[],'val':[],'test':[],'excluded':[],'legacy_episodes_used':False,'legacy_weights_used':False,'input_shapes':[[4,541,2],[68]],'output':'steering residual /0.72 rad, signed speed /3.2 mps'}
for p in sorted((root/'data').glob('*.npz')):
 record=json.loads(p.with_suffix('.json').read_text())
 if record['split']=='test':continue
 if not record['success']:
  manifest['excluded'].append({'raw':str(p.relative_to(root)),'mode':record['mode'],'seed':record['seed'],'collision':record['collision'],'progress_m':record['progress_m']});continue
 z=np.load(p);dest=out/p.name
 if 'scan' in z:
  h=z['scan'];f=z['features'];y=z['target']
 else:
  inputs=Inputs(w);last=np.zeros(2);previous=record['start_s'];hs=[];fs=[];ys=[]
  for st,raw,u in zip(z['state'],z['raw_scan'],z['command']):
   hh,ff=inputs.push(raw,st,last,previous);previous=inputs.s;hs.append(hh);fs.append(ff);ys.append([(u[0]-nominal_steer(ff))/.72,u[1]/SPEED_SCALE]);last=u
  h=np.asarray(hs,np.float16);f=np.asarray(fs,np.float32);y=np.asarray(ys,np.float32)
 assert h.shape[1:]==(4,541,2) and f.shape[1:]==(68,) and y.shape[1:]==(2,),(p,h.shape,f.shape,y.shape)
 assert np.isfinite(h).all() and np.isfinite(f).all() and np.isfinite(y).all()
 np.savez_compressed(dest,scan=h,features=f,target=y);summary={k:v for k,v in record.items() if k not in ['trajectory','commands','executed_commands','solver_log','profile']};dest.with_suffix('.json').write_text(json.dumps(summary,indent=2));manifest[record['split']].append({'path':str(dest.relative_to(root)),'raw':str(p.relative_to(root)),'seed':record['seed'],'mode':record['mode'],'samples':len(y),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest()});print(p.name,len(y),flush=True)
(root/'results/dataset_manifest.json').write_text(json.dumps(manifest,indent=2));print('FROZEN',{k:len(v) for k,v in manifest.items() if isinstance(v,list)},flush=True)
