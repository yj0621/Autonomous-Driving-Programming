"""Explicit hybrid early-path guard. No forced forward command into blocked space."""
import numpy as np
from corridor import Corridor
from perception_corner import decode_action
from swept_envelope import free_distance
class Guard:
 def __init__(self):self.corridor=Corridor();self.previous=np.zeros(2);self.no_obstacle=0;self.slow_steps=0
 def act(self,scan,f,output,observations=None):
  present=bool(f[37]>.5);out=np.asarray(output).copy();self.no_obstacle=0 if present else self.no_obstacle+1
  if self.no_obstacle>=3:out[:3]=0;out[4]=0
  u=decode_action(out,f);neural_speed=float(u[1]);plan,feasible=self.corridor.plan(f,observations or []);planned=False
  if present and feasible and plan is not None:
   ss,offsets,road,normals=plan;look=float(np.clip(.7+.25*f[32]*2.8,.75,1.25));path=road+normals*offsets[:,None];target=np.array([np.interp(look,ss,path[:,i]) for i in range(2)])
   u[0]=np.clip(np.arctan2(2*.3302*target[1],max(float(target@target),.10)),-.36,.36);planned=True
  u[0]=np.clip(u[0],self.previous[0]-.20,self.previous[0]+.20)
  distance=free_distance(scan,.8*u[0]+.2*self.previous[0]);cap=max(0.,np.sqrt(.4**2+8*distance)-.4);u[1]=min(u[1],cap,np.sqrt(1.3/max(abs(np.tan(u[0])/.3302),.02)))
  self.slow_steps=self.slow_steps+1 if f[32]*2.8<.1 and u[1]<.15 else 0
  recovery=False
  # A low network speed is recoverable only with a checked forward path, never a blocked one.
  if self.slow_steps>=5 and cap>.6 and ((planned and feasible) or not present):u[1]=min(.35,cap);recovery=True
  self.previous=u.copy()
  return u,{'corridor_planned':planned,'corridor_feasible':bool(feasible),'foreground_detected':present,'ttc_stopping_distance_m':distance,'speed_cap_mps':cap,'braking':bool(u[1]<neural_speed-1e-5),'safe_path_recovery':recovery}
