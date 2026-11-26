"""Map/observed-obstacle collision checked nonholonomic beam path, no teleports."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from numba import njit
from frenet_guide import FrenetGuide
@njit(cache=True)
def plan_beam(initial,road,arcs,tangents,sdf,origin,res,obstacles,target_d,steps=50,width=80):
 previous=np.zeros((width,6));previous[0,:5]=initial;previous[0,5]=0.;count=1
 history=np.zeros((steps,width,6));parents=np.full((steps,width),-1,np.int32);sizes=np.zeros(steps,np.int32)
 for depth in range(steps):
  candidates=np.zeros((width*5,6));parent=np.zeros(width*5,np.int32);nc=0
  for b in range(count):
   x,y,yaw,old_delta,old_s,cost=previous[b]
   for action in range(5):
    delta=min(.36,max(-.36,old_delta+(action-2)*.06));beta=np.arctan(.5*np.tan(delta));k=np.cos(beta)*np.tan(delta)/.3302;angle=k*.15
    xx=x+.15*np.cos(yaw+beta+angle*.5);yy=y+.15*np.sin(yaw+beta+angle*.5);heading=yaw+angle
    best=1e20;idx=0
    for i in range(len(road)):
     if arcs[i]<old_s-.3 or arcs[i]>old_s+.65:continue
     dist=(xx-road[i,0])**2+(yy-road[i,1])**2
     if dist<best:best=dist;idx=i
    ss=arcs[idx];err=(heading-tangents[idx]+np.pi)%(2*np.pi)-np.pi
    if best>.95**2 or abs(err)>1.1:continue
    clearance=10.;blocked=False
    for offset in [-.16,.06,.28]:
     cx=xx+offset*np.cos(heading);cy=yy+offset*np.sin(heading);ix=int(np.floor((cx-origin[0])/res));iy=sdf.shape[0]-1-int(np.floor((cy-origin[1])/res))
     if ix<0 or iy<0 or ix>=sdf.shape[1] or iy>=sdf.shape[0]:blocked=True;break
     wall=sdf[iy,ix]-.165;clearance=min(clearance,wall)
     if wall<.015:blocked=True;break
     for ob in obstacles:
      co=np.cos(ob[2]);si=np.sin(ob[2]);dx=cx-ob[0];dy=cy-ob[1];qx=max(abs(co*dx+si*dy)-ob[3],0.);qy=max(abs(-si*dx+co*dy)-ob[3],0.);gap=np.sqrt(qx*qx+qy*qy)-.165;clearance=min(clearance,gap)
      if gap<.035:blocked=True;break
     if blocked:break
    if blocked:continue
    lateral=-(xx-road[idx,0])*np.sin(tangents[idx])+(yy-road[idx,1])*np.cos(tangents[idx]);value=cost+2.0*(lateral-target_d[idx])**2+.08*err*err+.10*delta*delta+.25*(delta-old_delta)**2+.003/max(clearance,.02)-1.2*(ss-old_s)
    for obstacle in obstacles:
     blocked_lateral=max(0.,obstacle[3]+.22-abs(lateral-obstacle[5]));value+=2.5*np.exp(-.5*((ss-obstacle[4])/2.)**2)*blocked_lateral*blocked_lateral
    candidates[nc]=np.array([xx,yy,heading,delta,ss,value]);parent[nc]=b;nc+=1
  if nc==0:
   # A failed distant branch does not invalidate a checked local braking corridor.
   if depth<6:return np.zeros((0,6)),False
   end=depth-1;index=0;best=1e20
   for j in range(sizes[end]):
    value=history[end,j,5]-2*(history[end,j,4]-initial[4])
    if value<best:best=value;index=j
   if history[end,index,4]-initial[4]<.4:return np.zeros((0,6)),False
   result=np.zeros((depth+1,6));result[0,:5]=initial
   for level in range(end,-1,-1):
    result[level+1]=history[level,index];index=parents[level,index]
   return result,True
  ids=np.argsort(candidates[:nc,5]);count=0
  for ii in ids:
   duplicate=False
   for j in range(count):
    if abs(candidates[ii,0]-previous[j,0])<.08 and abs(candidates[ii,1]-previous[j,1])<.08 and abs(candidates[ii,2]-previous[j,2])<.15 and abs(candidates[ii,3]-previous[j,3])<.08:duplicate=True;break
   if duplicate:continue
   previous[count]=candidates[ii];history[depth,count]=candidates[ii];parents[depth,count]=parent[ii];count+=1
   if count>=width:break
  sizes[depth]=count
 index=0;best=1e20
 for j in range(count):
  value=previous[j,5]-2*(previous[j,4]-initial[4])
  if value<best:best=value;index=j
 result=np.zeros((steps+1,6));result[0,:5]=initial
 for depth in range(steps-1,-1,-1):
  result[depth+1]=history[depth,index];index=parents[depth,index]
 return result,True
class BeamPath:
 def __init__(self,w):self.w=w;self.guide=FrenetGuide(w);self.guide_ok=False
 def plan(self,state,obstacles,s=None):
  w=self.w;s=w.project(*state[:2]) if s is None else s;arcs=s+np.arange(-.8,9.01,.10);road=w.xy(arcs);tangent=w.tangent(arcs);obs=np.array([[b['x'],b['y'],b.get('theta',0.),b.get('half_size',.15),s+(b.get('s',w.project(b['x'],b['y']))-s+w.length/2)%w.length-w.length/2,float(np.dot(np.array([b['x'],b['y']])-w.xy(b.get('s',w.project(b['x'],b['y']))),w.normal(b.get('s',w.project(b['x'],b['y'])))))] for b in obstacles],float).reshape(-1,6)
  guide_s,guide_d,self.guide_ok=self.guide.plan(state,obstacles,s);target_d=np.interp(arcs,guide_s,guide_d) if self.guide_ok else np.zeros(len(arcs))
  initial=np.array([state[0],state[1],state[4],state[2],s]);return plan_beam(initial,road,arcs,tangent,w.signed,np.array(w.mapcfg['origin'][:2]),w.mapcfg['resolution'],obs,target_d)
