"""Sampling MPCC: contour/lag tracking and progress optimization with exact PID/RK4.
Reference path is from static-map / perceived-obstacle DP, never obstacle ground truth.
Teacher/student share measurable inputs. No reverse or teleport recovery in teacher.
"""
from pathlib import Path
import sys,time
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent));sys.path.insert(0,str(root))
import numpy as np,yaml
from numba import njit
from physics import step
from frenet_guide import FrenetGuide
from scipy.interpolate import PchipInterpolator
from world import ROOT
def perceived_geometry(w,state,scan,obstacles):
 boxes=np.array([[b['x'],b['y'],b.get('theta',0),b.get('half_size',.18)] for b in obstacles],float).reshape(-1,4);r=np.asarray(scan,dtype=float);a=np.linspace(-2.35,2.35,len(r))+state[4];origin=state[:2]+.27*np.array([np.cos(state[4]),np.sin(state[4])]);points=origin+r[:,None]*np.c_[np.cos(a),np.sin(a)];valid=np.isfinite(points).all(1)&np.isfinite(r)&(r>.05)&(r<7.)
 points=points[valid];points=points[w.clearance(points[:,0],points[:,1])>.10]
 if len(points)>48:points=points[np.linspace(0,len(points)-1,48).astype(int)]
 return np.r_[boxes,np.c_[points,np.zeros((len(points),2))]]
@njit(cache=True)
def clip(x,lo,hi):return min(hi,max(lo,x))
@njit(cache=True)
def footprint_gap(st,sdf,origin,res,ob):
 co=np.cos(st[4]);si=np.sin(st[4]);gap=3.
 for x in [-.27,-.12,.06,.25,.445]:
  for y in [-.11,0.,.11]:
   px=st[0]+co*x-si*y;py=st[1]+si*x+co*y;ix=int(np.floor((px-origin[0])/res));iy=sdf.shape[0]-1-int(np.floor((py-origin[1])/res))
   if ix<0 or iy<0 or ix>=sdf.shape[1] or iy>=sdf.shape[0]:return -1.
   gap=min(gap,sdf[iy,ix]-.035)
 # SAT clearance proxy for the full rectangle, not inflated circles.
 cx=st[0]+.0875*co;cy=st[1]+.0875*si
 for b in ob:
  if b[3]<.01:
   dx=b[0]-cx;dy=b[1]-cy;gap=min(gap,max(abs(co*dx+si*dy)-.3575,abs(-si*dx+co*dy)-.115)-.020);continue
  cb=np.cos(b[2]);sb=np.sin(b[2]);dx=b[0]-cx;dy=b[1]-cy;dot=co*cb+si*sb;cross=-si*cb+co*sb;half=b[3]
  a=abs(co*dx+si*dy)-.3575-half*(abs(dot)+abs(cross));bb=abs(-si*dx+co*dy)-.115-half*(abs(dot)+abs(cross));c=abs(cb*dx+sb*dy)-half-.3575*abs(dot)-.115*abs(cross);d=abs(-sb*dx+cb*dy)-half-.3575*abs(cross)-.115*abs(dot);gap=min(gap,max(a,bb,c,d)-.025)
 return gap
@njit(cache=True)
def rollout(st,previous,last_accel,ref,knots,params,sdf,origin,res,ob,horizon=20):
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
  delta=clip(base+off,max(-.36,prev[0]-.16),min(.36,prev[0]+.16));desired=max(0.,min(4.2,ref[idx,4]+shift));requested=clip(4*(desired-v),-3.5,5.);accel=clip(requested,accel-1.2,accel+1.2);target=clip(v+accel/(4.755 if accel>=0 else 19.02),0.,4.5);u=np.array([delta,target]);actions[j]=u
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
@njit(cache=True)
def batch(st,previous,accel,ref,samples,params,sdf,origin,res,ob):
 costs=np.zeros(len(samples))
 for i in range(len(samples)):costs[i]=rollout(st,previous,accel,ref,samples[i],params,sdf,origin,res,ob)[0]
 return costs
class SampledMPCC:
 def __init__(self,w,cruise=4.2,seed=123):
  self.w=w;self.cruise=cruise;cfg=yaml.safe_load((ROOT/'stack_master/config/SIM/sim_params.yaml').read_text());keys=['mu','C_Sf','C_Sr','lf','lr','h','m','I','s_min','s_max','sv_min','sv_max','v_switch','a_max','v_min','v_max'];self.params=np.array([cfg[k] for k in keys],float);self.guide=FrenetGuide(w);self.previous=np.zeros(2);self.accel=0.;self.last_v=0.;self.knots=np.zeros((6,2));self.rng=np.random.default_rng(seed);self.s=None;self.last_plan=None
 def reference(self,state,obstacles,s):
  w=self.w;ss,d,ok=self.guide.plan(state,obstacles,s)
  if not obstacles:ok=True;d=np.zeros(len(ss))
  if not ok:return None
  dense=s+np.arange(0,8.401,.06);offset=PchipInterpolator(ss,d)(dense);xy=w.xy(dense)+w.normal(dense)*offset[:,None];derivative=np.gradient(xy,.06,axis=0);angle=np.unwrap(np.arctan2(derivative[:,1],derivative[:,0]));curv=np.gradient(angle,.06)/np.maximum(np.linalg.norm(derivative,axis=1),.2);cap=np.minimum(self.cruise,np.sqrt(2.1/np.maximum(abs(curv),.025)))
  # Continuous braking envelope; no arbitrary speed drop simply because an obstacle exists.
  for i in range(len(cap)-2,-1,-1):cap[i]=min(cap[i],np.sqrt(cap[i+1]**2+2*2.0*.06))
  return np.c_[xy,angle,curv,cap,dense]
 def act(self,state,scan,obstacles,s=None):
  start=time.perf_counter();s=self.w.project(*state[:2],hint=self.s) if s is None else s;self.s=s;ref=self.reference(state,obstacles,s);ob=perceived_geometry(self.w,state,scan,obstacles);origin=np.array(self.w.mapcfg['origin'][:2]);measured=state.copy()
  if state[3]>.5:measured[6]=np.clip(self.params[4]*state[5]/state[3]-state[3]*state[5]/(self.params[0]*9.81*self.params[2]),-.25,.25)
  else:measured[6]=0.
  self.accel=np.clip((state[3]-self.last_v)/.1,-3.5,5.);self.last_v=state[3]
  guide_fallback=ref is None
  if guide_fallback:ref=self.reference(state,[],s)
  mean=self.knots.copy();std=np.tile([.045,.28],(6,1));best_cost=1e30;best=None
  for iteration in range(3):
   samples=mean+self.rng.normal(size=(32,6,2))*std;samples[:,:,0]=np.clip(samples[:,:,0],-.13,.13);samples[:,:,1]=np.clip(samples[:,:,1],-.7,.3);samples[0]=mean;samples[1]=0
   costs=batch(measured,self.previous,self.accel,ref,samples,self.params,self.w.signed,origin,self.w.mapcfg['resolution'],ob);order=np.argsort(costs);elite=samples[order[:6]]
   if costs[order[0]]<best_cost:best_cost=float(costs[order[0]]);best=samples[order[0]].copy()
   mean=elite.mean(0);std=np.maximum(elite.std(0),[.008,.05])
  cost,actions,trajectory=rollout(measured,self.previous,self.accel,ref,best,self.params,self.w.signed,origin,self.w.mapcfg['resolution'],ob);ok=cost<1e6
  if ok:u=actions[0];self.knots=.8*best;self.last_plan=trajectory
  else:u=np.array([self.previous[0],max(0.,state[3]-min(state[3],3.5/19.02))]);self.last_plan=None
  self.previous=u.copy();return u,{'ok':bool(ok),'status':'sampling_contouring_progress' if ok else 'no_collision_free_rollout','solve_ms':(time.perf_counter()-start)*1000,'cost':float(cost),'plan':trajectory.tolist() if ok else [],'reference':ref.tolist(),'actions':actions.tolist() if ok else []}
