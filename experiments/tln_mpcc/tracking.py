"""Deterministic raceline feedback used only in explicitly-labelled residual policies."""
import numpy as np
from world import STEER_MAX

def nominal(features):
 f=np.asarray(features);single=f.ndim==1
 if single:f=f[None]
 points=f[:,:16].reshape(-1,8,2)*5.;speed=f[:,-2]*2.5
 distances=np.array([0,.5,1.,1.5,2.,3.,4.,5.]);look=np.clip(.8+.35*speed,.8,1.6)
 xs=np.array([np.interp(d,distances,p[:,0]) for d,p in zip(look,points)]);ys=np.array([np.interp(d,distances,p[:,1]) for d,p in zip(look,points)])
 result=np.clip(np.arctan2(2*.3302*ys,np.maximum(xs*xs+ys*ys,.15)),-STEER_MAX,STEER_MAX)
 return float(result[0]) if single else result

def decode(output,features,residual=False):
 out=np.asarray(output);steer=float(out[0])*(.72 if residual else .36)+(nominal(features) if residual else 0.)
 return np.array([np.clip(steer,-.36,.36),np.clip(float(out[1])*2.5,0,2.5)])
