"""LiDAR-only swept-arc speed shield. Separate from learned policy performance."""
import numpy as np
from world import STEER_MAX,V_MAX

def shield(scan,command,latency=.15):
 u=np.asarray(command,dtype=float).copy();u=np.clip(u,[-STEER_MAX,0],[STEER_MAX,V_MAX]);r=np.asarray(scan,dtype=float)
 if len(r)!=1080 or not np.isfinite(u).all():return np.zeros(2),{'limited':True,'reason':'invalid_input','free_m':0.}
 r=np.nan_to_num(r,nan=0.,posinf=30.,neginf=0.);a=np.linspace(-2.35,2.35,len(r));points=np.c_[.27+r*np.cos(a),r*np.sin(a)]
 s=np.linspace(.05,3.5,70);beta=np.arctan(.5*np.tan(u[0]));curvature=np.cos(beta)*np.tan(u[0])/.3302
 if abs(curvature)<1e-4:centres=np.c_[s,np.zeros_like(s)]
 else:centres=np.c_[(np.sin(beta+curvature*s)-np.sin(beta))/curvature,(np.cos(beta)-np.cos(beta+curvature*s))/curvature]
 distance2=((centres[:,None,:]-points[None,:,:])**2).sum(-1);hit=np.any(distance2<.19**2,axis=1)
 free=max(0.,float(s[np.flatnonzero(hit)[0]])-.50) if hit.any() else 3.2
 # v*latency + v^2/(2*deceleration) <= free, permitting zero target speed.
 decel=2.;cap=max(0.,np.sqrt((decel*latency)**2+2*decel*free)-decel*latency);cap=0. if cap<.20 else cap;limited=bool(u[1]>cap);u[1]=min(u[1],cap)
 return u,{'limited':limited,'reason':'swept_arc_clearance' if limited else 'none','free_m':free,'speed_cap_mps':cap}
