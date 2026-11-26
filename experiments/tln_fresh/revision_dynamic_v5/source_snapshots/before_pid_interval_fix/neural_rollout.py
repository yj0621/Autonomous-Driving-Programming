import numpy as np
from numba import njit
from sampled_mpcc import clip,footprint_gap
from physics import step
@njit(cache=True)
def rollout_actions(st,previous,last_accel,ref,knots,params,sdf,origin,res,ob,forced,horizon=20):
 state=st.copy();idx=0;cost=0.;queue0=previous[0];queue1=previous[0];prev=previous.copy();accel=last_accel;actions=np.zeros((horizon,2));trajectory=np.zeros((horizon,7));start_arc=ref[0,5]
 for j in range(horizon):
  best=1e20;begin=max(0,idx-3);end=min(len(ref),idx+18)
  for i in range(begin,end):
   dist=(state[0]-ref[i,0])**2+(state[1]-ref[i,1])**2
   if dist<best:best=dist;idx=i
  v=state[3];look=min(1.15,max(.60,.55+.15*v));target_idx=min(len(ref)-1,idx+int(look/.06));point=ref[target_idx];dx=point[0]-state[0];dy=point[1]-state[1];beta=0.
  if v>.5:beta=clip(params[4]*state[5]/v-v*state[5]/(params[0]*9.81*params[2]),-.25,.25)
  lateral=-np.sin(state[4]+beta)*dx+np.cos(state[4]+beta)*dy;kappa=2*lateral/max(dx*dx+dy*dy,.10);kus=(1/params[1]-1/params[2])/(params[0]*9.81)
  base=(params[3]+params[4]+kus*v*v)*kappa if v>.5 else np.arctan((params[3]+params[4])*kappa)
  interval=min(4,j//4);fraction=(j%4)/4.;off=(1-fraction)*knots[interval,0]+fraction*knots[interval+1,0];shift=(1-fraction)*knots[interval,1]+fraction*knots[interval+1,1]
  delta=clip(base+off,max(-.36,prev[0]-.16),min(.36,prev[0]+.16));desired=max(0.,min(4.2,ref[idx,4]+shift));requested=clip(4*(desired-v),-3.5,5.);
  if j<len(forced):
   delta=clip(forced[j,0],max(-.36,prev[0]-.16),min(.36,prev[0]+.16));requested=clip((forced[j,1]-v)*(4.755 if forced[j,1]>=v else 19.02),-3.5,5.)
  accel=clip(requested,accel-1.2,accel+1.2);target=clip(v+accel/(4.755 if accel>=0 else 19.02),0.,4.5);u=np.array([delta,target]);actions[j]=u
  for k in range(10):
   effective=queue0;queue0=queue1;queue1=delta;state=step(state,effective,target,params)
   if k%5==4:
    gap=footprint_gap(state,sdf,origin,res,ob)
    if gap<0:return 1e6+1000*(horizon-j),actions,trajectory
    cost+=.001/max(gap+.015,.015)
  trajectory[j]=state
  co=np.cos(ref[idx,2]);si=np.sin(ref[idx,2]);ex=state[0]-ref[idx,0];ey=state[1]-ref[idx,1];ec=-si*ex+co*ey;el=co*ex+si*ey-max(v,0)*.1;heading=(state[4]+state[6]-ref[idx,2]+np.pi)%(2*np.pi)-np.pi
  cost+=110*ec*ec+18*el*el+2.*heading*heading+1.2*(state[3]-ref[idx,4])**2+1.5*(delta-prev[0])**2+.015*accel*accel
  prev=u
 cost-=5*(ref[idx,5]-start_arc)
 return cost,actions,trajectory
