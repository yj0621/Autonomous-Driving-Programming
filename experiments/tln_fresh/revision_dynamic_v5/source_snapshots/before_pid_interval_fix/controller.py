"""V5 physical decoder, jerk-limited actuator, full-path predictive correction.
The neural-only and MPCC-assisted modes are evaluated and reported separately.
"""
from pathlib import Path
import sys,time
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent));sys.path.insert(0,str(root))
import numpy as np
from decode import decode
from sampled_mpcc import SampledMPCC,rollout
from neural_rollout import rollout_actions
class Controller:
 def __init__(self,w,protected=True):self.w=w;self.protected=protected;self.mpc=SampledMPCC(w);self.previous=np.zeros(2)
 def actuator(self,state,features,action):
  u=np.asarray(action,dtype=np.float64).copy();u[0]=np.clip(u[0],max(-.36,self.previous[0]-.16),min(.36,self.previous[0]+.16));kp=4.755 if u[1]>=state[3] else 19.02;requested=np.clip((u[1]-state[3])*kp,-3.5,5.);actual=features[85]*6.;accel=np.clip(requested,actual-1.2,actual+1.2);u[1]=np.clip(state[3]+accel/(4.755 if accel>=0 else 19.02),0,4.5);return u
 def act(self,state,scan,features,prediction,obstacles,s):
  actions=decode(prediction,features);u=self.actuator(state,features,actions[0]);detail={'neural_command':actions[0].tolist(),'intervened':False,'reason':'neural','planned_path_feasible':None}
  if self.protected:
   ref=self.mpc.reference(state,obstacles,s);detail['planned_path_feasible']=ref is not None
   if ref is None:ref=self.mpc.reference(state,[],s)
   measured=state.copy();measured[6]=features[84]*.25;origin=np.array(self.w.mapcfg['origin'][:2]);ob=np.array([[b['x'],b['y'],b.get('theta',0),b.get('half_size',.18)] for b in obstacles],float).reshape(-1,4);knots=np.zeros((6,2));args=(measured,self.previous,float(features[85]*6),ref,knots,self.mpc.params,self.w.signed,origin,self.w.mapcfg['resolution'],ob);neural_cost,_,_=rollout_actions(*args,actions);base_cost,_,_=rollout(*args);detail['neural_cost']=float(neural_cost);detail['baseline_cost']=float(base_cost)
   if neural_cost>=1e6 or neural_cost>base_cost+30.:
    self.mpc.previous=self.previous.copy();self.mpc.last_v=float(state[3]-features[85]*.6);u,info=self.mpc.act(state,scan,obstacles,s);detail['intervened']=True;detail['reason']=info['status'];detail['mpcc_solve_ms']=info['solve_ms'];u=self.actuator(state,features,u)
  self.previous=u.copy();return u,detail
