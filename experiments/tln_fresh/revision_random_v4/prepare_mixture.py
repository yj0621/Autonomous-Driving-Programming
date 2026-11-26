"""New data only. Re-encode persistent static memory / 9m route, label separate sides."""
from pathlib import Path
import sys,json,hashlib
root=Path(__file__).parent;sys.path.insert(0,str(root.parent));sys.path.insert(0,str(root))
import numpy as np
from world import World
from inputs_wide import InputsWide,nominal_steer,SPEED_SCALE
w=World();original=json.loads((root/'results/dataset_manifest.json').read_text());dest=root/'prepared_mixture';dest.mkdir(exist_ok=True);manifest={'train':[],'val':[],'test':[],'excluded':original['excluded'],'legacy_episodes_used':False,'future_positions_are_labels_only':True,'input_shapes':[[4,541,2],[84]],'persistent_static_memory':True,'route_preview_m':9,'path_scale_m':1.2,'side_classes':['left','right','clear']}
for split in ['train','val']:
 for e in original[split]:
  z=np.load(root/e['raw']);meta=json.loads((root/e['raw']).with_suffix('.json').read_text());states=z['state'];commands=z['command'];executed=z['executed'] if 'executed' in z else commands;inputs=InputsWide(w);hs=[];fs=[];ys=[];native=[];arc=[];progress=0.;last=meta['start_s'];previous=np.zeros(2)
  for st,raw,u,behavior in zip(states,z['raw_scan'],commands,executed):
   h,f=inputs.push(raw,st,previous,last);s=inputs.s;progress+=(s-last+w.length/2)%w.length-w.length/2;native.append(s);arc.append(progress);last=s;hs.append(h);fs.append(f);ys.append([(u[0]-nominal_steer(f))/.72,u[1]/3.2]);previous=behavior
  fs=np.asarray(fs,np.float32);native=np.asarray(native);arc=np.asarray(arc);lateral=((states[:,:2]-w.xy(native))*w.normal(native)).sum(-1);path=np.zeros((len(states),5),np.float32);masks=np.ones_like(path);classes=np.zeros((len(states),3),np.float32);valid_class=np.ones((len(states),1),np.float32)
  for i,st in enumerate(states):
   desc=fs[i,34:66].reshape(4,8);visible=desc[desc[:,3]>.5];active=[b for b in visible if b[7]*6>=-.65]
   if not active:classes[i,2]=1;continue
   b=min(active,key=lambda b:max(float(b[7]*6),0));co,si=np.cos(st[4]),np.sin(st[4]);rotation=np.array([[co,-si],[si,co]]);observed=st[:2]+np.array([[q[0]*6,q[1]*3] for q in visible])@rotation.T;chosen_xy=st[:2]+rotation@np.array([b[0]*6,b[1]*3]);gt=min(meta['obstacles'],key=lambda ob:np.linalg.norm(chosen_xy-[ob['x'],ob['y']])) if meta['obstacles'] else None
   if gt is None:classes[i,2]=1;valid_class[i]=0;continue
   so=gt['s'];difference=(so-native[i]+w.length/2)%w.length-w.length/2
   if abs(difference-b[7]*6)>1.:
    ahead=(w.track.s-native[i]+w.length/2)%w.length-w.length/2;ids=np.flatnonzero((ahead>-.8)&(ahead<8.5));j=ids[np.argmin((w.track.x[ids]-gt['x'])**2+(w.track.y[ids]-gt['y'])**2)];so=float(w.track.s[j]);difference=float(ahead[j])
   obj_d=float(np.dot(np.array([gt['x'],gt['y']])-w.xy(so),w.normal(so)));goal=arc[i]+difference
   if difference>=0:
    ids=np.flatnonzero(arc[i:]>=goal)
    if not len(ids):valid_class[i]=0;classes[i,2]=1;continue
    k=i+ids[0]
   else:k=int(np.argmin(abs(arc[:i+1]-goal)))
   separation=float(lateral[k]-obj_d)
   if abs(separation)<.15:valid_class[i]=0
   classes[i,0 if separation>=0 else 1]=1
   for j,ds in enumerate([.5,1.,1.5,2.,3.]):
    goal=arc[i]+ds;ids=np.flatnonzero(arc[i+1:]>=goal)
    if not len(ids):masks[i,j]=0;continue
    k=i+1+ids[0];fraction=np.clip((goal-arc[k-1])/max(arc[k]-arc[k-1],1e-5),0,1);path[i,j]=np.clip(((1-fraction)*lateral[k-1]+fraction*lateral[k])/1.2,-1,1);future_s=(native[i]+ds)%w.length
    for ob in meta['obstacles']:
     if abs((ob['s']-future_s+w.length/2)%w.length-w.length/2)<1.5 and np.min(np.linalg.norm(observed-[ob['x'],ob['y']],axis=1))>.65:masks[i,j]=0
  target=np.c_[ys,path,masks,classes,valid_class].astype(np.float32);out=dest/Path(e['path']).name;np.savez_compressed(out,scan=np.asarray(hs,np.float16),features=fs,target=target);manifest[split].append({**e,'path':str(out.relative_to(root)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()});print(out.name,len(target),'classes',classes.sum(0).tolist(),'valid',float(valid_class.mean()),flush=True)
(root/'results/dataset_manifest_mixture.json').write_text(json.dumps(manifest,indent=2));print('MIXTURE LABELS READY',flush=True)
