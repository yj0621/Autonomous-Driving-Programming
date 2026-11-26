"""Unit-matched neural decoder and separately reported map/LiDAR safety controller."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sensors import nominal_steer,SPEED_SCALE
from world import ROOT
import yaml
VEHICLE=yaml.safe_load((ROOT/"stack_master/config/SIM/sim_params.yaml").read_text())
from fast_basic import FastTracker
from beam_path import BeamPath
from recovery import ReverseRecovery
from swept_envelope import free_distance

def decode(prediction,features,method='direct'):
 quantized=method.endswith('_grid');method=method.removesuffix('_grid')
 p=np.asarray(prediction);direct=float(np.clip(nominal_steer(features)+p[0]*.72,-.36,.36));speed=float(np.clip(p[1]*SPEED_SCALE,-.60,3.2))
 if method=='direct' or len(p)<7 or speed<0:return np.array([np.round(direct/.032)*.032 if quantized else direct,speed])
 offsets=p[2:7]
 if method in ('mixture','mixture_space'):
  side=int(np.argmax(p[12:15]));offsets=p[2:7] if side==0 else p[7:12] if side==1 else np.zeros(5)
  if method=='mixture_space':
   ds_path=np.array([.5,1,1.5,2,3]);widths=np.asarray(features[16:32]).reshape(8,2)*3;route_ds=np.array([0,.5,1,1.5,2,3,4,5]);left=np.interp(ds_path,route_ds,widths[:,0])-.32;right=np.interp(ds_path,route_ds,widths[:,1])-.32;objects=features[34:66].reshape(4,8);active=[b for b in objects if b[3]>.5 and -.45<b[7]*6<4.]
   if active:
    b=min(active,key=lambda x:max(0,x[7]));scores=[]
    for k,path in enumerate([p[2:7]*1.2,p[7:12]*1.2]):
     margin=np.minimum(left-path,right+path);separation=abs(float(np.interp(np.clip(b[7]*6,.5,3),ds_path,path))-b[6]*2);room=float(b[4 if k==0 else 5]*3);score=-20.*float(np.sum(np.maximum(-margin,0)**2))-10.*max(.44-separation,0)**2+.04*room+.01*np.log(max(float(p[12+k]),1e-5));scores.append(score)
    side=int(np.argmax(scores));offsets=p[2:7] if side==0 else p[7:12]
   offsets=np.clip(offsets*1.2,-np.maximum(right,0),np.maximum(left,0))/1.2
   if active:
    distance=max(0.,float(min(active,key=lambda b:max(0,b[7]))[7]*6));speed=min(speed,float(np.sqrt(1.15**2+2*2.0*max(distance-1.8,0))))
 road=np.asarray(features[:16]).reshape(8,2)*5;ds=np.array([0,.5,1,1.5,2,3,4,5]);v=float(features[32]*3.2);rate=float(features[66]*5);look=float(np.clip(.65+.22*abs(v),.75,1.3));target=np.array([np.interp(look,ds,road[:,i]) for i in range(2)]);tangent=np.array([np.interp(look+.1,ds,road[:,i])-np.interp(look-.1,ds,road[:,i]) for i in range(2)]);normal=np.array([-tangent[1],tangent[0]])/max(np.linalg.norm(tangent),.01);target+=normal*np.interp(look,[.5,1,1.5,2,3],offsets*1.2)
 beta=0.
 if v>.5:beta=float(np.clip(VEHICLE['lr']*rate/v-v*rate/(VEHICLE['mu']*9.81*VEHICLE['C_Sr']),-.25,.25))
 lateral=-np.sin(beta)*target[0]+np.cos(beta)*target[1];curvature=2*lateral/max(float(target@target),.12);wheelbase=VEHICLE['lf']+VEHICLE['lr']
 if v>.5:
  understeer=(1/VEHICLE['C_Sf']-1/VEHICLE['C_Sr'])/(VEHICLE['mu']*9.81);steering=(wheelbase+understeer*v*v)*curvature
 else:steering=np.arctan(wheelbase*curvature)
 steering=float(np.clip(steering,-.36,.36))
 if method=='blend':steering=.70*steering+.30*direct
 return np.array([np.round(steering/.032)*.032 if quantized else steering,speed])

class Controller:
 def __init__(self,w,protected=True,style='dynamic',decoder='direct'):
  self.decoder=decoder;self.style=style;self.w=w;self.protected=protected;self.previous=np.zeros(2);self.tracker=FastTracker(w,3.0);self.planner=BeamPath(w);self.recovery=ReverseRecovery(w);self.stopped_steps=0;self.interventions=0;self.static_memory=[];self.hazard_steps=0;self.reverse_brake=False;self.resteer_goal=None;self.reverse_override=0;self.reverse_override_delta=0.;self.previous_backup=0.
  if self.style=='dynamic' and self.protected:
   from dynamics_shield import DynamicShield
   self.dynamic=DynamicShield(w)
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
  self.recovery.observe(state);raw=decode(prediction,features,self.decoder);u=raw.copy();cap,_=self.tracker.profile(s);detail={'neural_command':raw.tolist(),'path_correction':False,'path_feasible':None,'braking':False,'reverse_recovery':False,'safe_path_resume':False}
  if self.protected and self.style=='dynamic':return self.dynamic_act(state,scan,features,prediction,obstacles,s)
  if not self.protected:
   u[0]=np.clip(u[0],self.previous[0]-.16,self.previous[0]+.16);u[1]=min(u[1],cap,np.sqrt(1.9/max(abs(np.tan(u[0])/.3302),.025)));u[1]=min(u[1],self.previous[1]+.25);
   if self.decoder.endswith('_grid'):u[0]=np.round(u[0]/.032)*.032
   self.previous=u.copy();return u,detail
  if self.recovery.active:
   u,done=self.recovery.act(state,obstacles);detail['reverse_recovery']=True;self.previous=u.copy();return u,detail
  if not obstacles:
   nominal=nominal_steer(features);u[0]=np.clip(u[0],nominal-.025,nominal+.025)
  else:
   path,ok=self.planner.plan(state,obstacles,s);detail['path_feasible']=bool(ok)
   if ok:
    distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1))];look=float(np.clip(.8+.25*abs(state[3]),.8,1.5));point=np.array([np.interp(look,distances,path[:,i]) for i in range(2)]);rel=point-state[:2];y=-np.sin(state[4])*rel[0]+np.cos(state[4])*rel[1];geometric=np.clip(np.arctan2(2*.3302*y,max(float(rel@rel),.1)),-.36,.36)
    if self.style=='beam':geometric=float(np.interp(max(.15,abs(state[3])*.1),distances,path[:,3]))
    free=free_distance(scan,.8*u[0]+.2*self.previous[0]);need=abs(state[3])*.12+state[3]**2/8+.10;opposite=(u[0]*geometric<0 and abs(u[0]-geometric)>.10)
    if self.style!='speed_only' and (opposite or abs(u[0]-geometric)>.18 or free<max(.45,need)):u[0]=geometric;detail['path_correction']=True
    ahead=np.maximum(path[:,4]-s,0);curve=np.maximum(abs(np.tan(path[:,3])/.3302),.025);local_speed=np.minimum(3.,np.sqrt(1.9/curve));cap=min(cap,float(np.min(np.sqrt(local_speed**2+2*2.8*ahead))))
    if len(path)<51:cap=min(cap,float(np.sqrt(2*2.8*max(path[-1,4]-s-.25,0.))))
    # Forward motion is restored only after checking a complete local vehicle path.
    if u[1]<.25 and cap>.5:u[1]=min(cap,.65);detail['safe_path_resume']=True
   else:
    # Distant search failure is not an emergency if the current measured path is free.
    detail['local_search_failed']=True
  u[0]=np.clip(u[0],self.previous[0]-.16,self.previous[0]+.16)
  if obstacles and detail.get('local_search_failed',False):
   current=min(free_distance(scan,.8*u[0]+.2*self.previous[0]),self.map_distance(state,.8*u[0]+.2*self.previous[0]));need=abs(state[3])*.12+state[3]**2/8+.25
   if current<need:
    options=[];nominal=nominal_steer(features);near=features[34:66].reshape(4,8);present=near[near[:,3]>.5];preferred=(1 if present[0,4]>present[0,5] else -1) if len(present) else 0
    for delta in np.unique(np.clip(np.r_[u[0],np.linspace(-.36,.36,13)],self.previous[0]-.16,self.previous[0]+.16)):
     effective=.8*delta+.2*self.previous[0];free=min(free_distance(scan,effective),self.map_distance(state,effective));score=free-.20*abs(delta-u[0])-.10*abs(delta-self.previous[0])+.05*preferred*np.sign(delta-nominal);options.append((score,free,delta))
    best=max(options)
    if best[1]>current+.15:u[0]=best[2];detail['path_correction']=True
  effective=.8*u[0]+.2*self.previous[0];distance=min(free_distance(scan,effective),self.map_distance(state,effective));brake_cap=max(0.,np.sqrt(.4**2+8*distance)-.4)
  u[1]=max(0.,min(u[1],cap,brake_cap,np.sqrt(1.9/max(abs(np.tan(u[0])/.3302),.025)),self.previous[1]+.25));self.stopped_steps=self.stopped_steps+1 if u[1]<.08 and abs(state[3])<.1 else 0
  if self.stopped_steps>=4 and obstacles and self.recovery.start(state):u,done=self.recovery.act(state,obstacles);detail['reverse_recovery']=True
  if self.stopped_steps>=4 and not obstacles and distance>.75:u[1]=min(.5,cap);detail['safe_path_resume']=True
  detail['braking']=bool(u[1]<raw[1]-1e-5);detail['free_distance_m']=distance;detail['speed_cap_mps']=min(cap,brake_cap);self.interventions+=int(detail['path_correction'] or detail['braking'] or detail['safe_path_resume']);self.previous=u.copy();return u,detail
 def dynamic_act(self,state,scan,f,prediction,obstacles,s):
  if not self.recovery.active and self.recovery.anchor is not None and np.linalg.norm(state[:2]-self.recovery.anchor)>3.:self.recovery.attempts=0;self.recovery.anchor=None
  raw=decode(prediction,f,self.decoder);nominal=nominal_steer(f);cap,_=self.tracker.profile(s);desired=raw.copy();desired[0]=np.clip(desired[0],self.previous[0]-.16,self.previous[0]+.16);desired[1]=max(0.,min(desired[1],cap,np.sqrt(1.9/max(abs(np.tan(desired[0])/.3302),.025)),self.previous[1]+.25))
  if not obstacles:desired[0]=np.clip(desired[0],nominal-.025,nominal+.025)
  for b in self.static_memory:b['age']+=1
  for b in obstacles:
   match=next((m for m in self.static_memory if np.hypot(m['x']-b['x'],m['y']-b['y'])<.45),None)
   if match is None:self.static_memory.append({**b,'age':0})
   else:match.update(b);match['age']=0
  self.static_memory=[b for b in self.static_memory if b['age']<80 and np.linalg.norm(state[:2]-[b['x'],b['y']])<7.]
  prepared=self.dynamic.prepare(state,scan,self.static_memory);detail={'neural_command':raw.tolist(),'path_correction':False,'path_feasible':None,'braking':False,'reverse_recovery':False,'safe_path_resume':False,'dynamic_check':True}
  if self.recovery.active:
   self.recovery.count+=1
   if np.linalg.norm(state[:2]-self.recovery.anchor)>.78 or np.linalg.norm(state[:2]-self.recovery.target)<.25 or self.recovery.count>40:self.reverse_brake=True
   if self.reverse_brake:
    u=np.array([self.previous[0],.8 if state[3]<-.03 else 0.])
    if state[3]>=-.03:
     self.recovery.active=False
     if self.recovery.history:
      j=int(np.argmin([np.linalg.norm(x[:2]-state[:2]) for x in self.recovery.history]));self.recovery.history=self.recovery.history[:j+1]
     self.reverse_brake=False;self.hazard_steps=20;self.stopped_steps=0
   else:
    rel=self.recovery.target-state[:2];y=-np.sin(state[4])*rel[0]+np.cos(state[4])*rel[1];u=np.array([np.clip(np.arctan2(2*.3302*y,max(float(rel@rel),.08)),-.36,.36),-.45])
   if self.resteer_goal is not None:
    if abs(state[2]-self.resteer_goal)<.03:self.reverse_override=4;self.reverse_override_delta=self.resteer_goal;self.resteer_goal=None
    else:u=np.array([self.resteer_goal,0.]);detail['stationary_resteer']=True
   if self.reverse_override>0 and not self.reverse_brake:
    u=np.array([self.reverse_override_delta,-.35]);self.reverse_override-=1
   u[0]=np.clip(u[0],self.previous[0]-.16,self.previous[0]+.16);ok,ttc=self.dynamic.safe(state,u,self.previous,prepared)
   if not ok:
    candidates=[]
    for delta in np.unique(np.clip(np.r_[u[0],self.previous[0],np.linspace(-.36,.36,13)],self.previous[0]-.16,self.previous[0]+.16)):
     test=np.array([delta,u[1]]);safe,_=self.dynamic.safe(state,test,self.previous,prepared)
     if safe:candidates.append((abs(delta-u[0]),delta))
    if candidates:u[0]=min(candidates)[1]
    else:
     prepared_choices=[]
     if abs(state[3])<.08:
      from dynamics_shield import segment
      for delta in np.linspace(-.36,.36,13):
       future=segment(state,np.array([delta,0.]),self.previous[0],self.dynamic.params,45);passed,_=self.dynamic.safe(future,np.array([delta,-.35]),np.array([delta,0.]),prepared)
       if passed:prepared_choices.append((abs(delta-u[0]),delta))
     if prepared_choices:
      self.resteer_goal=float(min(prepared_choices)[1]);u=np.array([np.clip(self.resteer_goal,self.previous[0]-.16,self.previous[0]+.16),0.]);detail['stationary_resteer']=True
     else:
      # A blocked reverse must not latch recovery when a checked forward exit exists.
      exits=[]
      for delta in np.unique(np.clip(np.r_[raw[0],nominal,np.linspace(-.36,.36,13)],self.previous[0]-.16,self.previous[0]+.16)):
       command=np.array([delta,.35]);passed,_=self.dynamic.safe(state,command,self.previous,prepared)
       if passed:exits.append((abs(delta-raw[0]),delta))
      if exits:
       self.recovery.active=False;self.reverse_brake=False;self.resteer_goal=None;self.stopped_steps=0;self.hazard_steps=15;u=np.array([min(exits)[1],.35]);detail['forward_recovery_exit']=True
      else:u=np.array([self.previous[0],.8 if state[3]<-.03 else 0.]);detail['no_recovery_candidate']=True
   detail['reverse_recovery']=True;self.previous=u.copy();return u,detail
  safe,ttc=self.dynamic.safe(state,desired,self.previous,prepared);detail['predicted_contact_s']=None if safe else ttc
  if safe and self.hazard_steps==0 and self.stopped_steps<3:
   self.stopped_steps=self.stopped_steps+1 if abs(state[3])<.05 and desired[1]<.2 else 0
   detail['braking']=bool(desired[1]<raw[1]-1e-5);self.previous_backup=self.dynamic.backup_for(desired);self.previous=desired.copy();return desired,detail
  path,ok=self.planner.plan(state,obstacles,s);detail['path_feasible']=bool(ok);self.hazard_steps=max(self.hazard_steps-1,0)
  if not safe:self.hazard_steps=15
  if ok:
   distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1))];goal=float(np.interp(max(.15,abs(state[3])*.1),distance,path[:,3]));path_cap=float(np.min(np.sqrt(np.minimum(3.,np.sqrt(1.9/np.maximum(abs(np.tan(path[:,3])/.3302),.025)))**2+2*2.8*np.maximum(path[:,4]-s,0))))
   if len(path)<51:path_cap=min(path_cap,float(np.sqrt(2*2.8*max(path[-1,4]-s-.25,0))))
   cap=min(cap,path_cap)
  else:
   near=f[34:66].reshape(4,8);present=near[near[:,3]>.5];side=(1 if present[0,4]>present[0,5] else -1) if len(present) else 0;goal=np.clip(nominal+.18*side,-.36,.36)
  deltas=np.unique(np.clip(np.r_[goal,desired[0],nominal,self.previous[0],np.linspace(-.36,.36,13)],self.previous[0]-.16,self.previous[0]+.16));speeds=np.unique(np.maximum(0,np.minimum(np.r_[desired[1],cap,1.2,.65,0.],min(cap,self.previous[1]+.25))));choices=[]
  # No unplanned forward creep from rest when the complete passage is infeasible.
  if not ok and abs(state[3])<.20:speeds=np.array([0.])
  for delta in deltas:
   for speed in speeds:
    command=np.array([delta,speed]);passed,contact=self.dynamic.safe(state,command,self.previous,prepared)
    if not passed:continue
    # During a verified hazard, follow the checked passage rather than repeating it.
    score=.25*speed-6.*(delta-goal)**2-.20*(delta-desired[0])**2-.08*(delta-self.previous[0])**2
    if speed<.15:score-=.20
    choices.append((score,float(delta),float(speed)))
  if choices:
   _,delta,speed=max(choices);u=np.array([delta,speed]);detail['path_correction']=bool(abs(delta-raw[0])>.015);detail['braking']=bool(speed<raw[1]-.03);detail['safe_path_resume']=bool(raw[1]<.2 and speed>.2)
   if speed<.15 and abs(state[3])<.15:
    self.stopped_steps+=1
    if self.stopped_steps>=2 and self.recovery.start(state):self.reverse_brake=False;u=np.array([self.previous[0],-.20]);passed,_=self.dynamic.safe(state,u,self.previous,prepared);detail['reverse_recovery']=True
    if detail['reverse_recovery'] and not passed:u=np.array([self.previous[0],0.])
   else:self.stopped_steps=0
  else:
   u=np.array([self.previous_backup,.8 if state[3]<-.03 else 0.]);detail['emergency_backup']=True;self.stopped_steps+=1
   if abs(state[3])<.15 and self.stopped_steps>=2 and self.recovery.start(state):self.reverse_brake=False;detail['reverse_recovery']=True
  self.previous=u.copy();return u,detail
