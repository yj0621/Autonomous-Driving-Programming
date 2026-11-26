"""V5 physical decoder, jerk-limited actuator, full-path predictive correction.
The neural-only and MPCC-assisted modes are evaluated and reported separately.
"""
from pathlib import Path
import sys,time
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent));sys.path.insert(0,str(root))
import numpy as np
from decode import decode,nominal
from sampled_mpcc import SampledMPCC,rollout,footprint_gap
from physics import step
from numba import njit
@njit(cache=True)
def emergency(st,previous,params,sdf,origin,res,ob,goal):
 best_score=-1e20;best_delta=previous[0];all_valid=False
 for delta in np.linspace(max(-.36,previous[0]-.16),min(.36,previous[0]+.16),17):
  state=st.copy();q0=previous[0];q1=previous[0];valid=True;survived=0
  for k in range(60):
   effective=q0;q0=q1;q1=delta;state=step(state,effective,0.,params)
   if k%2==1 and footprint_gap(state,sdf,origin,res,ob)<0:valid=False;break
   survived=k+1
   if abs(state[3])<.015:break
  score=(100000. if valid else 0.)+survived*10-20*abs(delta-goal)
  if score>best_score:best_score=score;best_delta=delta;all_valid=valid
 return np.array([best_delta,0.]),all_valid

from neural_rollout import rollout_actions
class Controller:
 def __init__(self,w,protected=True):self.w=w;self.protected=protected;self.mpc=SampledMPCC(w);self.previous=np.zeros(2);self.hazard_steps=0;self.safe_plan=None
 def actuator(self,state,features,action):
  u=np.asarray(action,dtype=np.float64).copy();u[0]=np.clip(u[0],max(-.36,self.previous[0]-.16),min(.36,self.previous[0]+.16));kp=4.755 if u[1]>=state[3] else 19.02;gain=(1-np.exp(-kp*.1))/.1;requested=np.clip((u[1]-state[3])*gain,-3.5,5.);actual=features[85]*6.;accel=np.clip(requested,actual-1.2,actual+1.2);inverse=(1-np.exp(-(4.755 if accel>=0 else 19.02)*.1))/.1;u[1]=np.clip(state[3]+accel/inverse,0,4.5);return u
 def act(self,state,scan,features,prediction,obstacles,s):
  prediction=np.asarray(prediction);side=int(np.argmax(prediction[36:39])) if len(prediction)==39 else None
  selected=prediction[side*12:side*12+12] if side is not None else prediction
  actions=decode(selected,features);u=self.actuator(state,features,actions[0]);detail={'neural_command':actions[0].tolist(),'intervened':False,'reason':'neural','planned_path_feasible':None,'passing_side':side}
  if self.protected:
   ref=self.mpc.reference(state,obstacles,s);detail['planned_path_feasible']=ref is not None
   if ref is None:ref=self.mpc.reference(state,[],s)
   measured=state.copy();measured[6]=features[84]*.25;origin=np.array(self.w.mapcfg['origin'][:2]);ob=np.array([[b['x'],b['y'],b.get('theta',0),b.get('half_size',.18)] for b in obstacles],float).reshape(-1,4);knots=np.zeros((6,2));args=(measured,self.previous,float(features[85]*6),ref,knots,self.mpc.params,self.w.signed,origin,self.w.mapcfg['resolution'],ob)
   neural_cost,neural_plan,_=rollout_actions(*args,actions);base_cost,_,_=rollout(*args);detail['neural_cost']=float(neural_cost);detail['baseline_cost']=float(base_cost)
   old=self.safe_plan;self.safe_plan=neural_plan.copy() if neural_cost<1e6 else None
   threshold=8. if obstacles else 30.;trigger=neural_cost>=1e6 or neural_cost>base_cost+threshold
   if trigger:self.hazard_steps=4
   if trigger or (self.hazard_steps>0 and obstacles):
    self.mpc.previous=self.previous.copy();self.mpc.last_v=float(state[3]-features[85]*.6);proposal,info=self.mpc.act(state,scan,obstacles,s);detail['intervened']=True;detail['reason']=info['status'];detail['mpcc_solve_ms']=info['solve_ms'];valid=False
    if info['ok']:
     proposed=np.asarray(info['actions'],dtype=np.float64);proposed[0]=self.actuator(state,features,proposal);checked,plan,_=rollout_actions(*args,proposed)
     if checked<1e6:u=plan[0];self.safe_plan=plan.copy();valid=True
    if not valid and old is not None and len(old)>1:
     shifted=old[1:].copy();shifted[0]=self.actuator(state,features,shifted[0]);checked,plan,_=rollout_actions(*args,shifted)
     if checked<1e6:u=plan[0];self.safe_plan=plan.copy();valid=True;detail['continued_verified_plan']=True
    if not valid and neural_cost<1e6:
     u=neural_plan[0];self.safe_plan=neural_plan.copy();valid=True;detail['checked_neural_fallback']=True
    if not valid:
     u,brake_valid=emergency(measured,self.previous,self.mpc.params,self.w.signed,origin,self.w.mapcfg['resolution'],ob,nominal(features));detail['emergency_braking']=True;detail['checked_braking_valid']=bool(brake_valid);self.safe_plan=None
  self.hazard_steps=max(0,self.hazard_steps-1);self.previous=u.copy();return u,detail
