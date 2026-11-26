"""New high-speed straight / anticipatory corner-braking basic recordings."""
from pathlib import Path
import sys,json,time,argparse,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from world import World
from policy import basic_action
class FastTracker:
 def __init__(self,w,cruise=3.0):self.w=w;self.cruise=cruise;self.previous=np.zeros(2)
 def profile(self,s):
  ahead=np.arange(0,6.01,.15);ss=(s+ahead)%self.w.length;v=self.w.path(ss,1);a=self.w.path(ss,2);k=abs(v[:,0]*a[:,1]-v[:,1]*a[:,0])/np.maximum(np.linalg.norm(v,axis=1)**3,.01)
  lateral_cap=np.minimum(self.cruise,np.sqrt(1.9/np.maximum(k,.025)))
  cap=float(np.min(np.sqrt(lateral_cap**2+2*2.8*ahead)))
  return cap,float(max(k))
 def act(self,st,s):
  u,f=basic_action(self.w,st,s,cruise=2.8);cap,curvature=self.profile(s);u[0]=np.clip(u[0],self.previous[0]-.16,self.previous[0]+.16)
  speed=min(self.cruise,cap,np.sqrt(1.9/max(abs(np.tan(u[0])/.3302),.025)),self.previous[1]+.25)
  u[1]=max(0.,speed);self.previous=u.copy();return u,{'profile_cap_mps':cap,'preview_max_curvature':curvature}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',default='train',choices=['train','val','test']);p.add_argument('--count',type=int,default=8);p.add_argument('--cruise',type=float,default=3.0);args=p.parse_args();root=Path(__file__).parent;dest=root/'data';dest.mkdir(exist_ok=True);w=World();sim=w.sim();summaries=[]
 for j in range(args.count):
  seed={'train':71000,'val':72000,'test':73000}[args.split]+j;start=(j*w.length/args.count+.37*(args.split!='train'))%w.length;rng=np.random.default_rng(seed);o=w.configure(sim,[],start,lateral=float(rng.uniform(-.035,.035)) if args.split=='train' else 0.,yaw_error=float(rng.uniform(-.025,.025)) if args.split=='train' else 0.);tracker=FastTracker(w,args.cruise);progress=0.;prev=start;states=[];raw=[];commands=[];profiles=[];error=[];wall_start=time.perf_counter()
  for k in range(650):
   st=sim.agents[0].state.copy();u,detail=tracker.act(st,prev);states.append(st.tolist());raw.append(np.asarray(o['scans'][0],np.float32));commands.append(u.tolist());profiles.append(detail);error.append(float(np.dot(st[:2]-w.xy(prev),w.normal(prev))));o=w.step(sim,u);s=w.project(*sim.agents[0].state[:2],hint=prev);progress+=(s-prev+w.length/2)%w.length-w.length/2;prev=s
   if o['collisions'][0] or progress>=w.length:break
  success=bool(progress>=w.length and not o['collisions'][0]);report={'seed':seed,'split':args.split,'mode':'basic_fast_new','success':success,'collision':sim.agents[0].collision_kind,'start_s':start,'progress_m':progress,'sim_time_s':(k+1)*.1,'wall_elapsed_s':time.perf_counter()-wall_start,'target_straight_mps':args.cruise,'mean_measured_speed_mps':float(np.mean(np.array(states)[:,3])),'max_measured_speed_mps':float(np.max(np.array(states)[:,3])),'mean_abs_raceline_error_m':float(np.mean(np.abs(error))),'max_abs_raceline_error_m':float(np.max(np.abs(error))),'samples':len(states),'obstacles':[],'source':'new direct physical rollout; no previous episode reuse','profile':profiles,'commands':commands,'trajectory':states,'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};path=dest/f'{args.split}_basic_{seed}.npz';np.savez_compressed(path,state=np.array(states,np.float32),raw_scan=np.array(raw,np.float32),command=np.array(commands,np.float32));path.with_suffix('.json').write_text(json.dumps(report));summaries.append({k:v for k,v in report.items() if k not in ['profile','commands','trajectory']});print(seed,'success',success,'collision',report['collision'],'mean_v',round(report['mean_measured_speed_mps'],2),'max_v',round(report['max_measured_speed_mps'],2),'error',round(report['mean_abs_raceline_error_m'],3),'sim_s',report['sim_time_s'],flush=True)
 (root/'results'/f'basic_{args.split}_summary.json').write_text(json.dumps(summaries,indent=2))
