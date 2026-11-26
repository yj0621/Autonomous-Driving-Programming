"""Explicit LiDAR/path supervisor, evaluated separately from the neural policy.
Does not use simulator obstacle ground truth. Compares left/right path clearance.
"""
import numpy as np
from policy import decode,PATH_SCALE
DS=np.array([0,.5,1.,1.5,2.,3.,4.,5.])
SAMPLES=np.linspace(.05,3.5,70)
class Supervisor:
 def __init__(self):self.side=0;self.clear_frames=0
 def geometry(self,f,offsets):
  road=np.asarray(f[:16]).reshape(8,2)*5.;x=np.interp(SAMPLES,DS,road[:,0]);y=np.interp(SAMPLES,DS,road[:,1]);v=np.c_[x,y];t=np.gradient(v,axis=0);normal=np.c_[-t[:,1],t[:,0]]/np.maximum(np.linalg.norm(t,axis=1)[:,None],.01)
  left=np.interp(SAMPLES,DS,np.asarray(f[16:32]).reshape(8,2)[:,0]*3)-.24;right=np.interp(SAMPLES,DS,np.asarray(f[16:32]).reshape(8,2)[:,1]*3)-.24
  d=np.interp(SAMPLES,[.75,1.5,2.5],offsets);d=np.clip(d,-right,left);v+=normal*d[:,None]
  # Continuous connection from measured current pose, not an instantaneous lateral jump.
  blend=np.minimum(SAMPLES/.6,1);v*=blend[:,None]
  return v
 def measure(self,scan,f,offsets):
  r=np.nan_to_num(np.asarray(scan)[::2],nan=0.,posinf=30.,neginf=0.);a=np.linspace(-2.35,2.35,1080)[::2];points=np.c_[.27+r*np.cos(a),r*np.sin(a)];v=self.geometry(f,offsets)
  dist2=((v[:,None,:]-points[None,:,:])**2).sum(-1);nearest=np.sqrt(dist2.min(1));hit=np.flatnonzero(nearest<.18);free=max(0.,float(SAMPLES[hit[0]])-.48) if len(hit) else 3.2
  clearance=float(np.min(nearest[(SAMPLES>.5)&(SAMPLES<2.5)]))-.18
  heading=np.unwrap(np.arctan2(np.gradient(v[:,1]),np.gradient(v[:,0])));curvature=np.gradient(heading)/np.maximum(np.linalg.norm(np.gradient(v,axis=0),axis=1),.025)
  max_k=float(np.percentile(abs(curvature[(SAMPLES>.4)&(SAMPLES<2.)]),85))
  return free,clearance,max_k
 def act(self,scan,f,output):
  out=np.asarray(output,dtype=float).copy();base=out[:3]*PATH_SCALE;free,clear,kappa=self.measure(scan,f,base);chosen=base;changed=False;left=right=None
  if free<2.0:
   options=[]
   for sign in [-1,1]:
    best=None
    for amount in [.3,.5,.7]:
     candidate=base+sign*amount*np.array([.6,1.,1.]);l=np.interp([.75,1.5,2.5],DS,np.asarray(f[16:32]).reshape(8,2)[:,0]*3)-.25;rr=np.interp([.75,1.5,2.5],DS,np.asarray(f[16:32]).reshape(8,2)[:,1]*3)-.25;candidate=np.clip(candidate,-rr,l)
     ff,cc,kk=self.measure(scan,f,candidate);score=2*ff+.6*cc-.20*float(np.mean(candidate*candidate))+( .10 if self.side==sign else 0.)
     item=(score,candidate,ff,cc,kk,sign)
     if best is None or score>best[0]:best=item
    options.append(best)
   right,left=options;best=max(options,key=lambda x:x[0]);base_score=2*free+.6*clear-.20*float(np.mean(base*base))
   if best[0]>base_score+.12:
    _,chosen,free,clear,kappa,self.side=best;changed=True
   self.clear_frames=0
  else:
   self.clear_frames+=1
   if self.clear_frames>4:self.side=0
  out[:3]=chosen/PATH_SCALE;u=decode(out,f)
  decel=2.;latency=.12;speed_cap=max(0.,np.sqrt((decel*latency)**2+2*decel*free)-decel*latency)
  if speed_cap<.12:speed_cap=0.
  u[1]=min(u[1],speed_cap,np.sqrt(1.7/max(kappa,.03)))
  return u,{'path_changed':changed,'side':'LEFT' if self.side>0 else ('RIGHT' if self.side<0 else 'NONE'),'free_path_m':free,'minimum_path_clearance_m':clear,'left_free_path_m':left[2] if left else None,'right_free_path_m':right[2] if right else None,'speed_cap_mps':speed_cap}
