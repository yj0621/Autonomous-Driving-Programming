"""Rebuild fresh-only data with identical online memory, no GT obstacle inputs."""
from pathlib import Path
import numpy as np,json,hashlib
from world import World
from perception_corner import Inputs
from perception import nominal_steer
root=Path(__file__).parent;dest=root/'corner_revision/prepared';dest.mkdir(exist_ok=True);w=World();old=json.loads((root/'results/training_v2.json').read_text());paths=[]
for key in ['train_episodes','validation_episodes']:
 for e in old[key]:
  name=Path(e['path']).name;raw=root/'data'/name
  if not raw.exists():raw=root/'data/ros_fast'/name
  paths.append(raw)
paths+=list((root/'corner_revision/data').glob('*.npz'))+list((root/'corner_revision/dagger_data').glob('*.npz'));manifest={'train_episodes':[],'validation_episodes':[],'test_episodes':[],'excluded':[],'obstacle_gt_inputs':False,'memory_steps':30}
for p in sorted(set(paths)):
 r=json.loads(p.with_suffix('.json').read_text());split=r['split']
 if not r['success']:
  manifest['excluded'].append({'path':str(p.relative_to(root)),'reason':'not_completed','collision':r['collision']});continue
 output=dest/p.name
 if output.exists() and output.with_suffix('.json').exists():
  rr=json.loads(output.with_suffix('.json').read_text());manifest[{'train':'train_episodes','val':'validation_episodes','test':'test_episodes'}[split]].append({'path':str(output.relative_to(root)),'mode':r['mode'],'seed':r['seed'],'samples':rr['samples'],'sha256':hashlib.sha256(output.read_bytes()).hexdigest()});continue
 z=np.load(p);inputs=Inputs(w);hs=[];fs=[];ys=[];pos=0;traj=np.asarray(r.get('trajectory',[]));cmds=np.asarray(r.get('commands',[]))
 for i,(st,raw,f) in enumerate(zip(z['state'],z['raw_scan'],z['features'])):
  f0=f[:34];h,ff=inputs.push(raw,st,f0)
  if z['target'].shape[1]==5:target=z['target'][i]
  else:
   if r['mode'].startswith('basic'):res=0.
   else:
    while pos<len(traj)-1 and not np.allclose(traj[pos,:5],st[:5],atol=1e-5):pos+=1
    if pos>=len(cmds):raise RuntimeError('command alignment failed '+str(p))
    res=np.clip((cmds[pos,0]-nominal_steer(f0))/.72,-1,1);pos+=1
   target=np.r_[z['target'][i],res]
  hs.append(h);fs.append(ff);ys.append(target)
 output=dest/p.name;np.savez_compressed(output,scan=np.asarray(hs,np.float32),features=np.asarray(fs,np.float32),target=np.asarray(ys,np.float32))
 rr={k:v for k,v in r.items() if k not in ['trajectory','commands','solver_log']};rr['source_raw']=str(p.relative_to(root));output.with_suffix('.json').write_text(json.dumps(rr,indent=2));manifest[{'train':'train_episodes','val':'validation_episodes','test':'test_episodes'}[split]].append({'path':str(output.relative_to(root)),'mode':r['mode'],'seed':r['seed'],'samples':len(ys),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()});print(output.name,len(ys),flush=True)
(root/'corner_revision/results/dataset_manifest.json').write_text(json.dumps(manifest,indent=2));print('PREPARED',[(k,len(manifest[k])) for k in ['train_episodes','validation_episodes','test_episodes']],flush=True)
