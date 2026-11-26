"""Explicit hybrid deployment guard: raceline feedback, learned avoidance, braking.
This is reported separately from the unguarded neural model.
"""
import numpy as np
from perception import decode_action
from supervisor import Supervisor
from safety import shield
from corridor import Corridor
class Guard:
 def __init__(self):self.geometry=Supervisor();self.missing=10;self.previous=np.zeros(2);self.corridor=Corridor()
 def act(self,scan,features,output,observations=None):
  out=np.asarray(output,dtype=float).copy();present=bool(features[37]>.5)
  self.missing=0 if present else self.missing+1
  # Deterministic base tracking prevents spurious learned offsets in a clear road.
  if self.missing>=3:out[:3]=0;out[4]=0
  u=decode_action(out,features);before=u[1];planned=False;feasible=True
  ranges=np.nan_to_num(np.asarray(scan,dtype=float),nan=0.,posinf=30.,neginf=0.);angles=np.linspace(-2.35,2.35,len(ranges));co=np.cos(angles);si=np.abs(np.sin(angles));envelope=np.minimum(.1651/np.maximum(abs(co),1e-5),.1016/np.maximum(si,1e-5));valid=co>1e-3
  distance=float(np.min((ranges[valid]-envelope[valid])/co[valid]));available=max(0.,distance-.025);decel=4.;latency=.10;cap=max(0.,np.sqrt((decel*latency)**2+2*decel*available)-decel*latency)
  if cap<.05:cap=0.
  free,clear,_=self.geometry.measure(scan,features[:34],out[:3]*.9)
  observations=observations if observations is not None else ([features[34:]] if present else [])
  plan,feasible=self.corridor.plan(features,observations)
  if plan is not None and feasible and present and free<1.8 and features[41]*6>1.2:
   ss,offsets,road,normals=plan;look=float(np.clip(.8+.35*features[32]*2.8,.9,1.6));target=np.array([np.interp(look,ss,road[:,i]+normals[:,i]*offsets) for i in range(2)])
   steering=float(np.clip(np.arctan2(2*.3302*target[1],max(np.dot(target,target),.15)),-.36,.36));u[0]=.6*steering+.4*u[0];planned=True
  # No global infeasibility claim or forced stop just because a 5m lattice failed.
  u[1]=min(u[1],cap)
  u[0]=float(np.clip(u[0],self.previous[0]-.30,self.previous[0]+.30));self.previous=u
  preferred='left' if features[38]>features[39] else 'right'
  return u,{'corridor_planned':planned,'corridor_feasible':feasible,'foreground_detected':present,'base_tracking_only':self.missing>=3,'preferred_side':preferred if present else 'none','ttc_stopping_distance_m':distance,'speed_cap_mps':cap,'free_path_m':free,'path_clearance_m':clear,'braking':bool(u[1]<before-1e-5)}
