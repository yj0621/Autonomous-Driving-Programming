"""Predict contact of the same sensor-centred rectangle used by simulator iTTC.
No double subtraction of front overhang, and steering sideslip is included.
"""
import numpy as np

def free_distance(scan,steer):
 r=np.nan_to_num(np.asarray(scan,dtype=float)[::2],nan=0.,posinf=30.,neginf=0.);a=np.linspace(-2.35,2.35,1080)[::2];points=np.c_[.27+r*np.cos(a),r*np.sin(a)];s=np.linspace(0.,2.5,101);beta=np.arctan(.5*np.tan(steer));k=np.cos(beta)*np.tan(steer)/.3302;theta=k*s
 if abs(k)<1e-5:base=np.c_[s,np.zeros_like(s)]
 else:base=np.c_[(np.sin(beta+theta)-np.sin(beta))/k,(np.cos(beta)-np.cos(beta+theta))/k]
 co=np.cos(theta);si=np.sin(theta);centres=base+.27*np.c_[co,si];delta=points[None,:,:]-centres[:,None,:];x=delta[:,:,0]*co[:,None]+delta[:,:,1]*si[:,None];y=-delta[:,:,0]*si[:,None]+delta[:,:,1]*co[:,None]
 hit=np.any((abs(x)<.1751)&(abs(y)<.1116),axis=1);ids=np.flatnonzero(hit)
 return max(0.,float(s[ids[0]])-.025) if len(ids) else 2.5
