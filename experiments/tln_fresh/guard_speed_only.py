"""Explicit hybrid deployment guard: raceline feedback, learned avoidance, braking.
This is reported separately from the unguarded neural model.
"""
import numpy as np
from perception import decode_action
from supervisor import Supervisor
from safety import shield
class Guard:
 def __init__(self):self.geometry=Supervisor();self.missing=10;self.previous=np.zeros(2)
 def act(self,scan,features,output):
  out=np.asarray(output,dtype=float).copy();present=bool(features[37]>.5)
  self.missing=0 if present else self.missing+1
  # Deterministic base tracking prevents spurious learned offsets in a clear road.
  if self.missing>=3:out[:3]=0;out[4]=0
  u=decode_action(out,features);free,clear,_=self.geometry.measure(scan,features[:34],out[:3]*.9)
  latency=.12;decel=2.;cap=max(0.,np.sqrt((decel*latency)**2+2*decel*free)-decel*latency)
  if cap<.12:cap=0.
  before=u[1];u[1]=min(u[1],cap)
  if present:
   immediate,detail=shield(scan,u)
   # Constant-steer projection is used only as an immediate stop envelope;
   # long-range protection uses the predicted curved path instead.
   if detail['free_m']<.12:u[1]=0.
  u[0]=float(np.clip(u[0],self.previous[0]-.30,self.previous[0]+.30));self.previous=u
  preferred='left' if features[38]>features[39] else 'right'
  return u,{'foreground_detected':present,'base_tracking_only':self.missing>=3,'preferred_side':preferred if present else 'none','free_path_m':free,'path_clearance_m':clear,'braking':bool(u[1]<before-1e-5)}
