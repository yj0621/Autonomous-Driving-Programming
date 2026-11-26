"""Shared measured inputs: four LiDAR frames, connected-map widths, four objects."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from world import preprocess
from perception import nominal_steer as previous_nominal
SPEED_SCALE=3.2
class Inputs:
 def __init__(self,w):self.w=w;self.ranges=[];self.masks=[];self.memory=[];self.obstacles=[];self.s=None
 def connected_width(self,s):
  w=self.w;ss=np.atleast_1d(s);p=w.xy(ss);n=w.normal(ss);ds=np.arange(.05,2.51,.05);result=[]
  for sign in [1,-1]:
   points=p[:,None,:]+sign*n[:,None,:]*ds[None,:,None];valid=w.clearance(points[:,:,0],points[:,:,1])>.13;width=np.zeros(len(ss))
   for i,row in enumerate(valid):
    bad=np.flatnonzero(~row);width[i]=ds[bad[0]] if len(bad) else 2.5
   result.append(width)
  return np.stack(result,axis=-1)
 def push(self,raw,state,last,s_hint=None):
  w=self.w;s=w.project(*state[:2],hint=s_hint);self.s=s;f=w.route_features(state,previous_steer=last[0],s_hint=s);ds=np.array([0,.5,1,1.5,2,3,4,5]);f[16:32]=(self.connected_width(s+ds)/3).ravel();f[32]=state[3]/SPEED_SCALE
  scan=np.asarray(raw,dtype=float);angles=np.linspace(-2.35,2.35,len(scan));yaw=state[4];co,si=np.cos(yaw),np.sin(yaw);rotation=np.array([[co,-si],[si,co]]);origin=state[:2]+.27*np.array([co,si]);valid=np.isfinite(scan)&(scan>.15)&(scan<8.5);rr=np.nan_to_num(scan,nan=0.,posinf=30.,neginf=0.);points=origin+rr[:,None]*np.c_[np.cos(yaw+angles),np.sin(yaw+angles)];fg=valid&(w.clearance(points[:,0],points[:,1])>.18);mask=np.r_[fg,fg[-1]][::2].astype(np.float32)
  for m in self.memory:m['age']+=1
  ids=np.flatnonzero(fg)
  if len(ids):
   for group in np.split(ids,np.flatnonzero(np.diff(ids)>2)+1):
    if len(group)<3:continue
    centre=np.median(points[group],axis=0);ray=centre-origin;centre+=.16*ray/max(np.linalg.norm(ray),.01);ahead=(w.track.s-s+w.length/2)%w.length-w.length/2;eligible=np.flatnonzero((ahead>-.8)&(ahead<8.5))
    j=eligible[np.argmin((w.track.x[eligible]-centre[0])**2+(w.track.y[eligible]-centre[1])**2)];so=float(w.track.s[j]);d=float(np.dot(centre-w.xy(so),w.normal(so)));line=w.xy(so)[None,:]+np.linspace(0,1,12)[:,None]*(centre-w.xy(so))[None,:]
    if abs(d)>1.8 or np.min(w.clearance(line[:,0],line[:,1]))<.12:continue
    match=next((m for m in self.memory if np.linalg.norm(m['xy']-centre)<.42),None)
    if match is None:self.memory.append({'xy':centre,'age':0})
    else:match['xy']=.3*match['xy']+.7*centre;match['age']=0
  choices=[];kept=[];observed=[]
  for m in self.memory:
   ahead=(w.track.s-s+w.length/2)%w.length-w.length/2;eligible=np.flatnonzero((ahead>-.8)&(ahead<8.5));j=eligible[np.argmin((w.track.x[eligible]-m['xy'][0])**2+(w.track.y[eligible]-m['xy'][1])**2)];so=float(w.track.s[j]);arc=float(ahead[j])
   if m['age']>25 or arc<-.7 or arc>8:continue
   kept.append(m);d=float(np.dot(m['xy']-w.xy(so),w.normal(so)));left,right=self.connected_width(so)[0];local=rotation.T@(m['xy']-state[:2]);descriptor=np.array([local[0]/6,local[1]/3,np.linalg.norm(local)/6,1.,(left-d-.21)/3,(right+d-.21)/3,d/2,arc/6],np.float32);choices.append((max(arc,0),descriptor));observed.append({'x':float(m['xy'][0]),'y':float(m['xy'][1]),'theta':float(w.tangent(so)),'s':so,'half_size':.21})
  self.memory=kept;self.obstacles=observed;extra=np.zeros((4,8),np.float32)
  for i,(_,b) in enumerate(sorted(choices,key=lambda x:x[0])[:4]):extra[i]=b
  distance=preprocess(raw)
  if not self.ranges:self.ranges=[distance.copy()]*3;self.masks=[mask.copy()]*3
  self.ranges=(self.ranges+[distance])[-4:];self.masks=(self.masks+[mask])[-4:];history=np.stack([np.stack([a,b],axis=-1) for a,b in zip(self.ranges,self.masks)])
  features=np.r_[f,extra.ravel(),(state[5]/5 if len(state)>5 else 0.),last[1]/SPEED_SCALE].astype(np.float32);return history,features

def nominal_steer(f):
 base=np.asarray(f[:34]).copy();base[32]*=SPEED_SCALE/2.8;return previous_nominal(base)
