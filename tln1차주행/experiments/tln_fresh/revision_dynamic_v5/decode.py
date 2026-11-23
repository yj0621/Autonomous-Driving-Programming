"""One physical-unit contract shared by collection, training and runtime."""
import numpy as np
SPEED_SCALE=4.5
STEER_SCALE=.72
FUTURE_STEPS=6
def nominal(features):
 f=np.asarray(features);road=f[:16].reshape(8,2)*5;v=f[32]*SPEED_SCALE;beta=f[84]*.25;look=float(np.clip(.55+.15*v,.6,1.15));ds=[0,.5,1,1.5,2,3,4,5];point=np.array([np.interp(look,ds,road[:,i]) for i in range(2)]);lateral=-np.sin(beta)*point[0]+np.cos(beta)*point[1];curve=2*lateral/max(float(point@point),.1);kus=(1/4.718-1/5.4562)/(1.0489*9.81);delta=(.3302+kus*v*v)*curve if v>.5 else np.arctan(.3302*curve);return float(np.clip(delta,-.36,.36))
def decode(prediction,features):
 p=np.asarray(prediction,dtype=np.float64).reshape(FUTURE_STEPS,2);actions=p.copy();actions[:,0]*=.36;actions[0,0]=nominal(features)+p[0,0]*STEER_SCALE;actions[:,0]=np.clip(actions[:,0],-.36,.36);actions[:,1]=np.clip(p[:,1]*SPEED_SCALE,0,4.5);return actions
