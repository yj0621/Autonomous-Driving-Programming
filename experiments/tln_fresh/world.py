"""Fixed hangar_1905_v0 simulator, route features, shared units and seeded layouts."""
from pathlib import Path
import os,sys,json,math,random
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parents[2]
# Override with TLN_MAP_DIR for map-specific collection and evaluation while
# keeping the historical hangar default for existing datasets.
MAP=Path(os.environ.get('TLN_MAP_DIR',ROOT/'stack_master/maps/hangar_1905_v0')).resolve()
sys.path.insert(0,str(ROOT/'utilities/nodes/race_manager'))
from race_manager.track import Track
from race_manager.placement import generate_candidates,draw,parse_recipe
DT=.1
V_MAX=2.8
STEER_MAX=.36
class World:
 def __init__(self):
  map_yaml=MAP/(MAP.name+'.yaml')
  self.track=Track(str(MAP/'global_waypoints.json'));t=self.track;self.length=t.length
  # Close the loop at the recorded last arc coordinate (same scorer convention).
  s=np.r_[t.s[:-1],t.length];xy=np.c_[np.r_[t.x[:-1],t.x[0]],np.r_[t.y[:-1],t.y[0]]]
  self.path=CubicSpline(s,xy,bc_type='periodic');self.ts=s
  self.left=np.r_[t.d_left[:-1],t.d_left[0]];self.right=np.r_[t.d_right[:-1],t.d_right[0]]
  cfg=yaml.safe_load(map_yaml.read_text());im=np.asarray(Image.open(MAP/cfg['image']).convert('L'))
  self.clear=distance_transform_edt(im>205)*cfg['resolution'];self.signed=self.clear-distance_transform_edt(im<=205)*cfg['resolution'];self.grad_y,self.grad_x=np.gradient(self.signed,cfg['resolution']);self.mapcfg=cfg
  self.candidates=None
 def xy(self,s):return self.path(np.asarray(s)%self.length)
 def tangent(self,s):
  d=self.path(np.asarray(s)%self.length,1);return np.arctan2(d[...,1],d[...,0])
 def normal(self,s):
  a=self.tangent(s);return np.stack([-np.sin(a),np.cos(a)],axis=-1)
 def project(self,x,y,hint=None):
  ids=np.arange(len(self.track.s))
  if hint is not None:
   ids=ids[np.abs((self.track.s-hint+self.length/2)%self.length-self.length/2)<2.0]
  i=ids[np.argmin((self.track.x[ids]-x)**2+(self.track.y[ids]-y)**2)];return float(self.track.s[i])
 def clearance(self,x,y):
  c=self.mapcfg;ix=np.floor((np.asarray(x)-c['origin'][0])/c['resolution']).astype(int);iy=self.clear.shape[0]-1-np.floor((np.asarray(y)-c['origin'][1])/c['resolution']).astype(int)
  valid=(ix>=0)&(iy>=0)&(ix<self.clear.shape[1])&(iy<self.clear.shape[0]);return np.where(valid,self.clear[np.clip(iy,0,self.clear.shape[0]-1),np.clip(ix,0,self.clear.shape[1]-1)],0)
 def wall_linearization(self,x,y):
  c=self.mapcfg;ix=(np.asarray(x)-c['origin'][0])/c['resolution'];iy=self.clear.shape[0]-1-(np.asarray(y)-c['origin'][1])/c['resolution']
  return tuple(map_coordinates(a,[np.atleast_1d(iy),np.atleast_1d(ix)],order=1,mode='nearest') for a in [self.signed,self.grad_x,-self.grad_y])
 def layouts(self,seed,count=4):
  if self.candidates is None:
   self.candidates=generate_candidates(self.track,size=.3,wall_margin=.15,car_width=.31,squeeze_clearance=.05,spacing=.5,d_step=.1,thetas=[0,np.pi/8,np.pi/4,3*np.pi/8],exclusion_ahead=5,exclusion_behind=1,min_severity=0)
  rng=random.Random(seed);cfg=yaml.safe_load((ROOT/'utilities/nodes/race_manager/config/race_manager_params.yaml').read_text())['obstacle_spawner']['ros__parameters']
  recipe=rng.choice(cfg['recipes']);spread,slots=parse_recipe(recipe)
  items,relaxed=draw(self.candidates,count,rng,self.track,wall_margin=.15,size=.3,min_obstacle_gap=.5,slots=slots,chicane_span=2.5,car_width=.31,squeeze_clearance=.05,spread=spread)
  return [dict(o._asdict()) for o in items],recipe,relaxed
 def route_features(self,state,previous_steer=None,s_hint=None):
  """Map route preview shared by teacher/student. No obstacle GT in this vector."""
  x,y,delta,v,yaw=state[:5];s=self.project(x,y,hint=s_hint);ds=np.array([0,.5,1.,1.5,2.,3.,4.,5.])
  pts=self.xy(s+ds)-[x,y];co,si=np.cos(yaw),np.sin(yaw)
  local=np.c_[co*pts[:,0]+si*pts[:,1],-si*pts[:,0]+co*pts[:,1]]
  # offsets in metres/5; widths in metres/3; v m/s/2.5; measured steer rad/.36.
  widths=np.c_[np.interp((s+ds)%self.length,self.ts,self.left),np.interp((s+ds)%self.length,self.ts,self.right)]
  return np.r_[local.ravel()/5,widths.ravel()/3,v/V_MAX,(delta if previous_steer is None else previous_steer)/STEER_MAX].astype('float32')
 def sim(self,seed=1):
  import pyglet
  pyglet.options['headless']=True
  from f110_gym.envs.base_classes import Simulator
  params=yaml.safe_load((ROOT/'stack_master/config/SIM/sim_params.yaml').read_text())
  self.params=params
  sim=Simulator(params,1,seed,time_step=.01);sim.set_map(str(MAP/(MAP.name+'.yaml')),'.png');return sim
 def configure(self,sim,obs,s=0.,lateral=0.,yaw_error=0.):
  from f110_gym.envs.base_classes import RaceCar
  RaceCar.static_obstacles=[(o['x'],o['y'],o['theta'],.3,.3) for o in obs]
  xy=self.xy(s)+self.normal(s)*lateral;yaw=float(self.tangent(s))+yaw_error
  sim.reset(np.array([[xy[0],xy[1],yaw]]));return sim.step(np.array([[0.,0.]]))
 def step(self,sim,cmd):
  for _ in range(10):
   o=sim.step(np.asarray(cmd).reshape(1,2))
   if o['collisions'][0]:break
  return o

def preprocess(scan):
 a=np.asarray(scan,dtype=np.float32)
 if len(a)==1080:a=np.r_[a,a[-1]]
 if len(a)!=1081:raise ValueError(f'Expected 1080/1081 beams, got {len(a)}')
 return np.clip(np.nan_to_num(a,nan=10.,posinf=10.,neginf=0.),0.,10.)[::2]/10.

class History:
 def __init__(self):self.frames=[]
 def push(self,scan):
  a=preprocess(scan)
  if not self.frames:self.frames=[a.copy(),a.copy()]
  self.frames.append(a);self.frames=self.frames[-3:]
  return np.stack(self.frames,axis=-1)
