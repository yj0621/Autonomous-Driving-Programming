"""New path-predicting temporal LiDAR policy; no legacy TLN weights or decoder."""
import numpy as np
from world import STEER_MAX
PATH_DS=np.array([.75,1.5,2.5])
PATH_SCALE=.9
SPEED_SCALE=2.8

def decode(output,features):
 f=np.asarray(features);p=f[:16].reshape(8,2)*5.;v=f[-2]*2.8
 ds=np.array([0,.5,1.,1.5,2.,3.,4.,5.]);look=np.clip(.8+.35*v,.9,1.65)
 base=np.array([np.interp(look,ds,p[:,i]) for i in range(2)])
 tangent=np.array([np.interp(look+.15,ds,p[:,i])-np.interp(look-.15,ds,p[:,i]) for i in range(2)]);normal=np.array([-tangent[1],tangent[0]])/max(np.linalg.norm(tangent),.01)
 offsets=np.asarray(output[:3])*PATH_SCALE;target=base+normal*np.interp(look,PATH_DS,offsets)
 steer=float(np.clip(np.arctan2(2*.3302*target[1],max(np.dot(target,target),.15)),-.36,.36));speed=float(np.clip(output[3]*SPEED_SCALE,0,SPEED_SCALE))
 curvature=abs(np.tan(steer)/.3302);speed=min(speed,np.sqrt(1.8/max(curvature,.02)))
 return np.array([steer,speed])

def basic_action(w,state,prev_s=None,cruise=2.2):
 f=w.route_features(state,s_hint=prev_s);u=decode(np.array([0.,0.,0.,cruise/SPEED_SCALE]),f)
 return u,f

def plan_label(w,expert,info,state):
 if not info['ok'] or not info['plan']:return None
 plan=np.asarray(info['plan']);s0=plan[0,4];arc=plan[:,4]-s0
 normals=w.normal(plan[:,4]);d=((plan[:,:2]-w.xy(plan[:,4]))*normals).sum(-1)
 # Predicted virtual progress is nondecreasing; extrapolated tail is explicitly masked.
 target=np.interp(PATH_DS,arc,d)
 mask=(PATH_DS<=max(float(arc[-1]),.01)).astype(np.float32)
 # For distant targets use the shared reference corridor, flagged as guide supervision.
 ob=expert.op.value(expert.ob)
 guide=expert.make_guide(s0+np.r_[0,PATH_DS],ob)[0,1:]
 target=np.where(mask>0,target,guide)
 return np.r_[np.clip(target/PATH_SCALE,-1,1),np.clip(expert.previous[1]/SPEED_SCALE,0,1)].astype(np.float32),mask
