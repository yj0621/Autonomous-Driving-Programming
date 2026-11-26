"""Unit-matched neural decoder and separately reported map/LiDAR safety controller."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sensors import nominal_steer,SPEED_SCALE
from fast_basic import FastTracker
from beam_path import BeamPath
from recovery import ReverseRecovery
from swept_envelope import free_distance

def decode(prediction,features):
 p=np.asarray(prediction);return np.array([np.clip(nominal_steer(features)+p[0]*.72,-.36,.36),np.clip(p[1]*SPEED_SCALE,-.60,3.2)])

class Controller:
 def __init__(self,w,protected=True):self.w=w;self.protected=protected;self.previous=np.zeros(2);self.tracker=FastTracker(w,3.0);self.planner=BeamPath(w);self.recovery=ReverseRecovery(w);self.stopped_steps=0;self.interventions=0
 def map_distance(self,state,delta):
  s=np.linspace(.025,2.50,100);beta=np.arctan(.5*np.tan(delta));k=np.cos(beta)*np.tan(delta)/.3302;theta=state[4]+k*s
  if abs(k)<1e-5:xy=state[:2]+s[:,None]*np.array([np.cos(state[4]+beta),np.sin(state[4]+beta)])
  else:xy=state[:2]+np.c_[(np.sin(state[4]+beta+k*s)-np.sin(state[4]+beta))/k,(np.cos(state[4]+beta)-np.cos(state[4]+beta+k*s))/k]
  co=np.cos(theta);si=np.sin(theta);hit=np.zeros(len(s),bool)
  for x in np.arange(-.25,.436,.07):
   for y in [-.115,0.,.115]:
    px=xy[:,0]+co*x-si*y;py=xy[:,1]+si*x+co*y;hit|=self.w.clearance(px,py)<.025
  ids=np.flatnonzero(hit);return max(0.,float(s[ids[0]])-.025) if len(ids) else 2.5
 def act(self,state,scan,features,prediction,obstacles,s):
  self.recovery.observe(state);raw=decode(prediction,features);u=raw.copy();cap,_=self.tracker.profile(s);detail={'neural_command':raw.tolist(),'path_correction':False,'path_feasible':None,'braking':False,'reverse_recovery':False,'safe_path_resume':False}
  if not self.protected:
   u[0]=np.clip(u[0],self.previous[0]-.16,self.previous[0]+.16);u[1]=min(u[1],cap,np.sqrt(1.9/max(abs(np.tan(u[0])/.3302),.025)));u[1]=min(u[1],self.previous[1]+.25);self.previous=u.copy();return u,detail
  if self.recovery.active:
   u,done=self.recovery.act(state,obstacles);detail['reverse_recovery']=True;self.previous=u.copy();return u,detail
  if not obstacles:
   nominal=nominal_steer(features);u[0]=np.clip(u[0],nominal-.025,nominal+.025)
  else:
   path,ok=self.planner.plan(state,obstacles,s);detail['path_feasible']=bool(ok)
   if ok:
    distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1))];look=float(np.clip(.8+.25*abs(state[3]),.8,1.5));point=np.array([np.interp(look,distances,path[:,i]) for i in range(2)]);rel=point-state[:2];y=-np.sin(state[4])*rel[0]+np.cos(state[4])*rel[1];geometric=np.clip(np.arctan2(2*.3302*y,max(float(rel@rel),.1)),-.36,.36)
    free=free_distance(scan,.8*u[0]+.2*self.previous[0]);need=abs(state[3])*.12+state[3]**2/8+.10;opposite=(u[0]*geometric<0 and abs(u[0]-geometric)>.10)
    if opposite or abs(u[0]-geometric)>.18 or free<max(.45,need):u[0]=geometric;detail['path_correction']=True
    ahead=np.maximum(path[:,4]-s,0);curve=np.maximum(abs(np.tan(path[:,3])/.3302),.025);local_speed=np.minimum(3.,np.sqrt(1.9/curve));cap=min(cap,float(np.min(np.sqrt(local_speed**2+2*2.8*ahead))))
    if len(path)<51:cap=min(cap,float(np.sqrt(2*2.8*max(path[-1,4]-s-.25,0.))))
    # Forward motion is restored only after checking a complete local vehicle path.
    if u[1]<.25 and cap>.5:u[1]=min(cap,.65);detail['safe_path_resume']=True
   else:
    u[1]=0.;self.stopped_steps+=1
    if abs(state[3])<.15 and self.stopped_steps>=2 and self.recovery.start(state):u,done=self.recovery.act(state,obstacles);detail['reverse_recovery']=True;self.previous=u.copy();return u,detail
  u[0]=np.clip(u[0],self.previous[0]-.16,self.previous[0]+.16);effective=.8*u[0]+.2*self.previous[0];distance=min(free_distance(scan,effective),self.map_distance(state,effective));brake_cap=max(0.,np.sqrt(.4**2+8*distance)-.4)
  u[1]=max(0.,min(u[1],cap,brake_cap,np.sqrt(1.9/max(abs(np.tan(u[0])/.3302),.025)),self.previous[1]+.25));self.stopped_steps=self.stopped_steps+1 if u[1]<.08 and abs(state[3])<.1 else 0
  if self.stopped_steps>=4 and obstacles and self.recovery.start(state):u,done=self.recovery.act(state,obstacles);detail['reverse_recovery']=True
  if self.stopped_steps>=4 and not obstacles and distance>.75:u[1]=min(.5,cap);detail['safe_path_resume']=True
  detail['braking']=bool(u[1]<raw[1]-1e-5);detail['free_distance_m']=distance;detail['speed_cap_mps']=min(cap,brake_cap);self.interventions+=int(detail['path_correction'] or detail['braking'] or detail['safe_path_resume']);self.previous=u.copy();return u,detail
