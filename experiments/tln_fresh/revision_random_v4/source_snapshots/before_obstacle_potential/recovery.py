"""Bounded reverse along recently travelled static free space; no pose reset."""
import numpy as np
class ReverseRecovery:
 def __init__(self,w):self.w=w;self.history=[];self.active=False;self.anchor=None;self.target=None;self.count=0;self.attempts=0
 def observe(self,state):
  if self.active:return
  if not self.history or np.linalg.norm(state[:2]-self.history[-1][:2])>.07:self.history.append(state.copy());self.history=self.history[-40:]
 def clear(self,state,delta,obstacles):
  beta=np.arctan(.5*np.tan(delta));k=np.cos(beta)*np.tan(delta)/.3302;s=-np.linspace(.03,.20,8);yaw=state[4]+k*s
  if abs(k)<1e-5:xy=state[:2]+s[:,None]*np.array([np.cos(state[4]),np.sin(state[4])])
  else:xy=state[:2]+np.c_[(np.sin(state[4]+beta+k*s)-np.sin(state[4]+beta))/k,(np.cos(state[4]+beta)-np.cos(state[4]+beta+k*s))/k]
  for offset in [-.16,.06,.28]:
   pts=xy+offset*np.c_[np.cos(yaw),np.sin(yaw)]
   if np.min(self.w.clearance(pts[:,0],pts[:,1]))<.16:return False
   for b in obstacles:
    d=pts-[b['x'],b['y']];theta=b.get('theta',0.);co,si=np.cos(theta),np.sin(theta);qx=np.maximum(abs(co*d[:,0]+si*d[:,1])-b.get('half_size',.15),0);qy=np.maximum(abs(-si*d[:,0]+co*d[:,1])-b.get('half_size',.15),0)
    if np.min(qx*qx+qy*qy)<.16**2:return False
  return True
 def start(self,state):
  if len(self.history)<6 or self.attempts>=3:return False
  choices=[st for st in self.history if .90<np.linalg.norm(st[:2]-state[:2])<1.20]
  if not choices:return False
  self.target=min(choices,key=lambda st:abs(np.linalg.norm(st[:2]-state[:2])-1.05))[:2].copy();self.anchor=state[:2].copy();self.active=True;self.count=0;self.attempts+=1;return True
 def act(self,state,obstacles):
  self.count+=1
  if np.linalg.norm(state[:2]-self.anchor)>=.95 or np.linalg.norm(state[:2]-self.target)<.13 or self.count>40:
   self.active=False;self.history=[];return np.array([state[2],0.]),True
  rel=self.target-state[:2];yaw=state[4];local_y=-np.sin(yaw)*rel[0]+np.cos(yaw)*rel[1];desired=np.clip(np.arctan2(2*.3302*local_y,max(float(rel@rel),.08)),-.36,.36)
  choices=sorted(np.r_[desired,np.linspace(-.36,.36,13)],key=lambda d:abs(d-desired))
  for delta in choices:
   effective=.8*delta+.2*state[2]
   if self.clear(state,effective,obstacles):return np.array([delta,-.45]),False
  self.active=False;return np.array([state[2],0.]),True
