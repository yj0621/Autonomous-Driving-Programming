"""Map-assisted foreground features from LiDAR + measured pose, never obstacle GT."""
import numpy as np
from world import History
class Foreground:
 def __init__(self,w):self.w=w;self.masks=[];self.observations=[]
 def compute(self,raw,state,f):
  w=self.w;r=np.asarray(raw,dtype=float);a=np.linspace(-2.35,2.35,1080);yaw=state[4];origin=state[:2]+.27*np.array([np.cos(yaw),np.sin(yaw)])
  valid=np.isfinite(r)&(r>.15)&(r<8);rr=np.nan_to_num(r,nan=0.,posinf=30.,neginf=0.);points=origin+rr[:,None]*np.c_[np.cos(yaw+a),np.sin(yaw+a)]
  foreground=valid&(w.clearance(points[:,0],points[:,1])>.18);mask=np.r_[foreground,foreground[-1]][::2].astype(np.float32)
  if not self.masks:self.masks=[mask.copy(),mask.copy()]
  self.masks=(self.masks+[mask])[-3:]
  extra=np.zeros(8,np.float32);indices=np.flatnonzero(foreground);s0=w.project(*state[:2]);choices=[]
  if len(indices):
   for ids in np.split(indices,np.flatnonzero(np.diff(indices)>2)+1):
    if len(ids)<3:continue
    centre=np.median(points[ids],axis=0);ray=centre-origin;centre+=.15*ray/max(np.linalg.norm(ray),.01)
    so=w.project(*centre);ds=(so-s0+w.length/2)%w.length-w.length/2
    if not -.4<ds<6.5:continue
    d=float(np.dot(centre-w.xy(so),w.normal(so)));left=float(np.interp(so,w.ts,w.left));right=float(np.interp(so,w.ts,w.right))
    if not -right-.2<d<left+.2:continue
    delta=centre-state[:2];co,si=np.cos(yaw),np.sin(yaw);local=np.array([co*delta[0]+si*delta[1],-si*delta[0]+co*delta[1]])
    descriptor=np.array([local[0]/6,local[1]/3,np.linalg.norm(delta)/6,1.,(left-.15-d-.21)/3,(d-.21+right-.15)/3,d/2,ds/6],np.float32);choices.append((max(ds,0),descriptor))
   if choices:extra=min(choices,key=lambda p:p[0])[1]
  self.observations=[d for _,d in choices]
  return np.stack(self.masks,axis=-1),np.r_[f,extra].astype(np.float32)
class Inputs:
 def __init__(self,w):self.history=History();self.foreground=Foreground(w)
 def push(self,raw,state,f):
  h=self.history.push(raw);mask,ff=self.foreground.compute(raw,state,f);return np.concatenate([h,mask],axis=-1).astype(np.float32),ff

def nominal_steer(f):
 from policy import decode
 return decode(np.zeros(4),np.asarray(f)[:34])[0]

def decode_action(output,f):
 from policy import decode
 base=nominal_steer(f);u=np.array([np.clip(base+float(output[4])*.72,-.36,.36),np.clip(float(output[3])*2.8,0,2.8)])
 u[1]=min(u[1],np.sqrt(1.8/max(abs(np.tan(u[0])/.3302),.02)))
 return u
