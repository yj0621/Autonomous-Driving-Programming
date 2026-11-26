"""Shared measured inputs: four LiDAR frames, connected-map widths, four objects."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import least_squares
def fit_square(points,prior,theta):
 points=np.asarray(points);points=points[np.isfinite(points).all(1)]
 if len(points)<4:return np.asarray(prior),float(theta)
 def residual(z):
  co,si=np.cos(z[2]),np.sin(z[2]);q=(points-z[:2])@np.array([[co,-si],[si,co]]);return np.r_[np.max(abs(q),axis=1)-.15,.05*(z[:2]-prior)]
 best=None
 for angle in [theta,theta-.45,theta+.45]:
  result=least_squares(residual,np.r_[prior,angle],bounds=(np.r_[np.asarray(prior)-.24,theta-1.2],np.r_[np.asarray(prior)+.24,theta+1.2]),loss="soft_l1",f_scale=.012,max_nfev=35)
  score=np.mean(np.minimum(residual(result.x)[:len(points)]**2,.01))
  if best is None or score<best[0]:best=(score,result.x)
 return best[1][:2],float(best[1][2])

from world import preprocess
from perception import nominal_steer as previous_nominal
SPEED_SCALE=4.5
class InputsWide:
 def __init__(self,w):self.w=w;self.ranges=[];self.masks=[];self.memory=[];self.obstacles=[];self.s=None;self.width_cache={};self.previous_v=None
 def _uncached_width(self,s):
  w=self.w;ss=np.atleast_1d(s);p=w.xy(ss);n=w.normal(ss);ds=np.arange(.05,2.51,.05);result=[]
  for sign in [1,-1]:
   points=p[:,None,:]+sign*n[:,None,:]*ds[None,:,None];valid=w.clearance(points[:,:,0],points[:,:,1])>.13;width=np.zeros(len(ss))
   for i,row in enumerate(valid):
    bad=np.flatnonzero(~row);width[i]=ds[bad[0]] if len(bad) else 2.5
   result.append(width)
  return np.stack(result,axis=-1)
 def connected_width(self,s):
  values=np.atleast_1d(s);keys=[round(float(x%self.w.length),2) for x in values]
  for key in keys:
   if key not in self.width_cache:self.width_cache[key]=self._uncached_width(key)[0]
  return np.asarray([self.width_cache[key] for key in keys])
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
    surface=points[group][::max(1,len(group)//30)]
    if match is None:
     fitted,theta=fit_square(surface,centre,float(w.tangent(so)));self.memory.append({'xy':fitted,'age':0,'points':surface,'theta':theta,'updates':0})
    else:
     match['age']=0;match['updates']+=1;combined=np.r_[match['points'],surface];match['points']=combined[np.linspace(0,len(combined)-1,min(160,len(combined))).astype(int)]
     if match['updates']%3==0:match['xy'],match['theta']=fit_square(match['points'],match['xy'],match['theta'])
  choices=[];observed=[]
  for m in self.memory:
   ahead=(w.track.s-s+w.length/2)%w.length-w.length/2;eligible=np.flatnonzero((ahead>-.8)&(ahead<8.5));j=eligible[np.argmin((w.track.x[eligible]-m['xy'][0])**2+(w.track.y[eligible]-m['xy'][1])**2)];so=float(w.track.s[j]);arc=float(ahead[j])
   delta=m['xy']-w.xy(so);direction=w.path(so,1);direction=direction/max(np.linalg.norm(direction),.01)
   if arc<-.7 or arc>8 or abs(float(np.dot(delta,direction)))>.30:continue
   d=float(np.dot(m['xy']-w.xy(so),w.normal(so)));line=w.xy(so)[None,:]+np.linspace(0,1,12)[:,None]*delta[None,:]
   if abs(d)>1.8 or np.min(w.clearance(line[:,0],line[:,1]))<.12:continue
   left,right=self.connected_width(so)[0];local=rotation.T@(m['xy']-state[:2]);descriptor=np.array([local[0]/6,local[1]/3,np.linalg.norm(local)/6,1.,(left-d-.18)/3,(right+d-.18)/3,d/2,arc/6],np.float32);choices.append((max(arc,0),descriptor,float(m.get("theta",w.tangent(so)))) );observed.append({'x':float(m['xy'][0]),'y':float(m['xy'][1]),'theta':float(m.get('theta',w.tangent(so))),'s':so,'half_size':.18})
  self.memory=self.memory[-32:];self.obstacles=observed;extra=np.zeros((4,8),np.float32)
  orientation=np.zeros((4,2),np.float32)
  for i,(_,b,theta) in enumerate(sorted(choices,key=lambda x:x[0])[:4]):extra[i]=b;orientation[i]=[np.cos(4*(theta-yaw)),np.sin(4*(theta-yaw))]
  distance=preprocess(raw)
  if not self.ranges:self.ranges=[distance.copy()]*3;self.masks=[mask.copy()]*3
  self.ranges=(self.ranges+[distance])[-4:];self.masks=(self.masks+[mask])[-4:];history=np.stack([np.stack([a,b],axis=-1) for a,b in zip(self.ranges,self.masks)])
  features=np.r_[f,extra.ravel(),(state[5]/5 if len(state)>5 else 0.),last[1]/SPEED_SCALE].astype(np.float32);preview=w.xy(s+np.array([6.,7.,8.,9.]))-state[:2];preview=preview@rotation;future_width=self.connected_width(s+np.array([6.,7.,8.,9.]));beta=0. if abs(state[3])<.5 else np.clip(.17145*state[5]/state[3]-state[3]*state[5]/(1.0489*9.81*5.4562),-.25,.25);acceleration=0. if self.previous_v is None else np.clip((state[3]-self.previous_v)/.1,-6.,6.);self.previous_v=float(state[3]);return history,np.r_[features,preview.ravel()/5,future_width.ravel()/3,beta/.25,acceleration/6,orientation.ravel()].astype(np.float32)

def nominal_steer(f):
 base=np.asarray(f[:34]).copy();base[32]*=SPEED_SCALE/2.8;return previous_nominal(base)
