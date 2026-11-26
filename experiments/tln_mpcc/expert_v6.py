"""Nonlinear MPCC: bicycle dynamics, virtual progress, contour/lag costs,
track/obstacle constraints. Visible obstacles only; no hidden obstacle labels.
Independent implementation informed by github.com/alexliniger/MPCC.
"""
import time,math
import numpy as np
import casadi as ca
from world import DT,V_MAX,STEER_MAX
class MPCC:
 def __init__(self,world,horizon=24,cruise=1.7):
  self.w=world;self.N=N=horizon;self.last=None;self.previous=np.zeros(3);self.failures=0;self.progress_s=None
  # Sequential local path linearization, refreshed each receding-horizon solve.
  # Avoid symbolic spline derivatives in the NLP to keep control latency low.
  cfg=world.mapcfg;res=cfg['resolution'];height,width=world.signed.shape
  xs=cfg['origin'][0]+(np.arange(width)+.5)*res;ys=cfg['origin'][1]+(np.arange(height)+.5)*res
  self.sdf=ca.interpolant('map_sdf','linear',[xs,ys],world.signed[::-1,:].T.ravel(order='F'))
  o=ca.Opti();self.op=o;X=o.variable(5,N+1);U=o.variable(3,N);self.X=X;self.U=U
  # state = x,y,yaw,velocity,virtual arc progress; u=steer,target speed,progress rate.
  p=o.parameter(5);ob=o.parameter(6,4);up=o.parameter(3);self.p=p;self.ob=ob;self.wall=o.parameter(5,3*N);self.up=up;self.ref=o.parameter(7,N+1);self.cruise=o.parameter();o.set_value(self.cruise,cruise)
  o.subject_to(X[:,0]==p);o.subject_to(o.bounded(-STEER_MAX,U[0,:],STEER_MAX));o.subject_to(o.bounded(0,U[1,:],V_MAX));o.subject_to(o.bounded(0,U[2,:],V_MAX))
  o.subject_to(o.bounded(0,X[3,:],V_MAX+.2));o.subject_to(o.bounded(-world.length/2,X[4,:],1.5*world.length))
  slack=o.variable(N);o.subject_to(slack>=0);cost=20000*ca.sumsqr(slack)
  for k in range(N):
   x,y,yaw,v,s=ca.vertsplit(X[:,k]);delta,vc,nu=ca.vertsplit(U[:,k]);beta=ca.atan(.5*ca.tan(delta))
   dx=ca.vertcat(v*ca.cos(yaw+beta),v*ca.sin(yaw+beta),v*ca.cos(beta)*ca.tan(delta)/.3302,4.0*(vc-v),nu)
   o.subject_to(X[:,k+1]==X[:,k]+DT*dx)
   o.subject_to(o.bounded(-2.5,v*v*ca.tan(delta)/.3302,2.5))
   o.subject_to(o.bounded(-.12,U[0,k]-(up[0] if k==0 else U[0,k-1]),.12))
   o.subject_to(o.bounded(-.45,U[1,k]-(up[1] if k==0 else U[1,k-1]),.22))
   ds=U[:,k]-(up if k==0 else U[:,k-1]);cost+=.25*delta**2+9*ds[0]**2+.4*ds[1]**2+.3*(nu-v)**2+3.0*(v-self.cruise)**2-.7*nu
  for k in range(1,N+1):
   x,y,yaw,v,s=ca.vertsplit(X[:,k]);rx,ry,co,si,sref,left,right=ca.vertsplit(self.ref[:,k]);ex=x-rx;ey=y-ry
   ec=-si*ex+co*ey;el=co*ex+si*ey-(s-sref);target=0
   for j in range(4):
    ox,oy,so,goal,active,theta=ca.vertsplit(ob[:,j]);target+=active*goal*ca.exp(-((s-so)/2.0)**2)
    # Two-circle vehicle footprint + square circumscribed obstacle + safety margin.
    for offset in [-.19,0.,.19]:
     cx=x+offset*ca.cos(yaw);cy=y+offset*ca.sin(yaw)
     dx=ca.cos(theta)*(cx-ox)+ca.sin(theta)*(cy-oy);dy=-ca.sin(theta)*(cx-ox)+ca.cos(theta)*(cy-oy)
     qx=ca.fmax(ca.fabs(dx)-.15,0);qy=ca.fmax(ca.fabs(dy)-.15,0)
     o.subject_to(qx*qx+qy*qy >= active*.16**2)
   for circle,offset in enumerate([-.19,0.,.19]):
    wx,wy,dist,gx,gy=ca.vertsplit(self.wall[:,3*(k-1)+circle]);cx=x+offset*ca.cos(yaw);cy=y+offset*ca.sin(yaw)
    o.subject_to(self.sdf(ca.vertcat(cx,cy)) >= .17-slack[k-1])
   o.subject_to(ec >= -right+.28-slack[k-1]);o.subject_to(ec <= left-.28+slack[k-1])
   # Visible-obstacle approach speed, independent of chosen lateral side.
   for j in range(4):
    ox,oy,so,goal,active,theta=ca.vertsplit(ob[:,j])
    cost+=active*6.0*ca.exp(-((s-so)/1.5)**2)*(v-1.1)**2
   cost+=22*(ec-target)**2+45*el**2+.6*(1-co*ca.cos(yaw)-si*ca.sin(yaw))
  o.minimize(cost);o.solver('ipopt',{'expand':True,'print_time':False},{'print_level':0,'max_iter':160,'tol':2e-3,'acceptable_tol':1e-2,'sb':'yes','warm_start_init_point':'yes'})
 def visible(self,state,scan,obstacles):
  x,y,delta,v,yaw=state[:5];lx=x+.27*np.cos(yaw);ly=y+.27*np.sin(yaw);out=[]
  for b in obstacles:
   dx=b['x']-lx;dy=b['y']-ly;d=math.hypot(dx,dy);a=(math.atan2(dy,dx)-yaw+np.pi)%(2*np.pi)-np.pi
   if abs(a)>2.34 or d>6:continue
   i=int(round((a+2.35)/4.7*1079));near=np.asarray(scan)[max(0,i-2):min(1080,i+3)]
   if len(near) and abs(float(np.median(near))-d)<.5:out.append(b)
  return sorted(out,key=lambda b:math.hypot(b['x']-x,b['y']-y))[:4]
 def act(self,state,scan,obstacles):
  start=time.perf_counter();w=self.w;x,y,delta,v,yaw=state[:5];s0=w.project(x,y)
  if self.progress_s is not None:
   near=np.abs((w.track.s-self.progress_s+w.length/2)%w.length-w.length/2)<2.0
   ids=np.flatnonzero(near);j=ids[np.argmin((w.track.x[ids]-x)**2+(w.track.y[ids]-y)**2)];s0=float(w.track.s[j])
  self.progress_s=s0
  p=np.array([x,y,yaw,max(0,v),s0]);ob=np.zeros((6,4));ob[:2,:]=100
  visible=self.visible(state,scan,obstacles)
  for j,b in enumerate(visible):
   so=b.get('s',w.project(b['x'],b['y']));so=s0+(so-s0+w.length/2)%w.length-w.length/2
   normal=w.normal(so);d=float(np.dot(np.array([b['x'],b['y']])-w.xy(so),normal));left=float(np.interp(so%w.length,w.ts,w.left));right=float(np.interp(so%w.length,w.ts,w.right))
   goals=[d+.62,d-.62];valid=[g for g in goals if -right+.30<g<left-.30 and w.clearance(*(w.xy(so)+w.normal(so)*g))>.28]
   goal=min(valid,key=abs) if valid else min(goals,key=abs)
   ob[:,j]=[b['x'],b['y'],so,goal,1,b.get('theta',0)]
  self.op.set_value(self.p,p);self.op.set_value(self.ob,ob);self.op.set_value(self.up,self.previous)
  if self.last is not None:
   xx,uu=self.last;xx=np.c_[xx[:,1:],xx[:,-1]];uu=np.c_[uu[:,1:],uu[:,-1]]
   if abs(xx[4,0]-s0)>w.length/2:xx[4,:]+=s0-xx[4,0]
   xx[2,:]+=round((yaw-xx[2,0])/(2*np.pi))*2*np.pi;xx[:,0]=p
  else:
   ss=s0+np.arange(self.N+1)*DT*max(v,1.4)
   lateral=np.zeros_like(ss)
   for j in range(4):
    lateral+=ob[4,j]*ob[3,j]*np.exp(-((ss-ob[2,j])/2.0)**2)
   coords=w.xy(ss)+w.normal(ss)*lateral[:,None]
   dy=np.gradient(coords,axis=0);head=np.unwrap(np.arctan2(dy[:,1],dy[:,0]));head+=round((yaw-head[0])/(2*np.pi))*2*np.pi
   xx=np.vstack([coords.T,head,np.full(self.N+1,max(v,.1)),ss]);xx[:,0]=p
   curvature=np.gradient(head)/np.maximum(np.linalg.norm(dy,axis=1),.03)
   uu=np.vstack([np.clip(np.arctan(.3302*curvature[:-1]),-.36,.36),np.full(self.N,1.2),np.full(self.N,1.2)])
  ss=xx[4,:].copy();coords=w.xy(ss);angles=w.tangent(ss)
  ref=np.vstack([coords.T,np.cos(angles),np.sin(angles),ss,np.interp(ss%w.length,w.ts,w.left),np.interp(ss%w.length,w.ts,w.right)])
  wall=np.zeros((5,3*self.N))
  for k in range(1,self.N+1):
   for j,offset in enumerate([-.19,0.,.19]):
    cx=xx[0,k]+offset*np.cos(xx[2,k]);cy=xx[1,k]+offset*np.sin(xx[2,k]);dist,gx,gy=w.wall_linearization(cx,cy)
    wall[:,3*(k-1)+j]=[cx,cy,float(dist[0]),float(gx[0]),float(gy[0])]
  self.op.set_value(self.wall,wall)
  self.op.set_value(self.ref,ref)
  self.op.set_initial(self.X,xx);self.op.set_initial(self.U,uu)
  try:
   sol=self.op.solve();xx=sol.value(self.X);uu=sol.value(self.U);self.last=(xx,uu);u=uu[:,0];ok=True
  except RuntimeError:
   self.failures+=1;self.last=None;u=np.array([np.clip(delta,-.36,.36),0.,0.]);ok=False
  self.previous=u
  return np.array([np.clip(u[0],-.36,.36),np.clip(u[1],0,V_MAX)]),{'ok':ok,'status':self.op.stats().get('return_status','unknown'),'visible':len(visible),'solve_ms':(time.perf_counter()-start)*1000,'plan':self.last[0].T.tolist() if self.last else []}
