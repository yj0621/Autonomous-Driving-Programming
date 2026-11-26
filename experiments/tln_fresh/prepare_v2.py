"""Enrich NEW recordings with LiDAR/map foreground and expert action targets."""
from pathlib import Path
import json
import numpy as np
from world import World
from perception import Foreground,nominal_steer
root=Path(__file__).parent;dest=root/'data_v2';dest.mkdir(exist_ok=True);w=World();manifest=[]
files=list((root/'data').glob('*_*.npz'))+list((root/'data/ros_fast').glob('train_*.npz'))
for p in sorted(files):
 if p.name.startswith('manifest'):continue
 r=json.loads(p.with_suffix('.json').read_text());split=r['split']
 if not r['success'] and split!='test':continue
 z=np.load(p);scan_history=z['scan'];foreground=Foreground(w);hh=[];ff=[];yy=[];cmds=np.asarray(r.get('commands',[]));traj=np.asarray(r.get('trajectory',[]));pos=0
 for i,(st,f,raw,target) in enumerate(zip(z['state'],z['features'],z['raw_scan'],z['target'])):
  mask,features=foreground.compute(raw,st,f);h=np.concatenate([scan_history[i],mask],axis=-1);nom=nominal_steer(f)
  if r['mode'].startswith('basic'):residual=0.
  else:
   while pos<len(traj)-1 and not np.allclose(traj[pos,:5],st[:5],atol=1e-5):pos+=1
   if pos>=len(cmds):raise RuntimeError(f'Cannot align expert commands {p}:{i}')
   residual=(float(cmds[pos,0])-nom)/.72;pos+=1
  hh.append(h);ff.append(features);yy.append(np.r_[target,np.clip(residual,-1,1)])
 np.savez_compressed(dest/p.name,scan=np.asarray(hh,np.float32),features=np.asarray(ff,np.float32),target=np.asarray(yy,np.float32))
 rr={**{k:v for k,v in r.items() if k not in ['trajectory','commands']},'source_npz':str(p),'map_foreground_from_lidar_and_pose_only':True};(dest/p.with_suffix('.json').name).write_text(json.dumps(rr,indent=2));manifest.append(rr)
 print(p.name,len(hh),'foreground',round(float(np.mean(np.asarray(ff)[:,37])),3),flush=True)
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
