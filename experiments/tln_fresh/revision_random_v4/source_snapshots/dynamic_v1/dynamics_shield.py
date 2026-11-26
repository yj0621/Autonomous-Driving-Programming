"""Check candidate + backup braking with the unchanged project PID/RK4 dynamics.
Uses static map, observed LiDAR foreground and remembered perceived cubes only.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,yaml,pyglet
pyglet.options['headless']=True
from numba import njit
from f110_gym.envs.dynamic_models import pid,vehicle_dynamics_st
@njit(cache=True)
def step(state,delta,speed,p):
 accel,sv=pid(speed,delta,state[3],state[2],p[11],p[13],p[15],p[14]);u=np.array([sv,accel]);k1=vehicle_dynamics_st(state,u,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);k2=vehicle_dynamics_st(state+.005*k1,u,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);k3=vehicle_dynamics_st(state+.005*k2,u,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);k4=vehicle_dynamics_st(state+.01*k3,u,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);result=state+.01*(1./6.)*(k1+2*k2+2*k3+k4)
 if result[4]>2*np.pi:result[4]-=2*np.pi
 elif result[4]<0:result[4]+=2*np.pi
 return result
@njit(cache=True)
def occupied(state,points,obstacles,sdf,origin,res):
 x,y,delta,v,yaw=state[:5];co=np.cos(yaw);si=np.sin(yaw)
 for xx in np.linspace(-.29,.4451,20):
  for yy in [-.1116,0.,.1116]:
   px=x+co*xx-si*yy;py=y+si*xx+co*yy;ix=int((px-origin[0])/res);iy=sdf.shape[0]-1-int((py-origin[1])/res)
   if ix<0 or iy<0 or ix>=sdf.shape[1] or iy>=sdf.shape[0] or sdf[iy,ix]<.025:return True
 for point in points:
  dx=point[0]-x;dy=point[1]-y;px=co*dx+si*dy;py=-si*dx+co*dy
  if .27-.1801<px<.27+.1801 and abs(py)<.1166:return True
 # SAT between conservative full body and uncertainty-inflated observed square.
 cx=x+.07755*co;cy=y+.07755*si;hl=.36755;hw=.1166
 for b in obstacles:
  cb=np.cos(b[2]);sb=np.sin(b[2]);dx=b[0]-cx;dy=b[1]-cy;dot=co*cb+si*sb;cross=-si*cb+co*sb;half=b[3]
  if abs(co*dx+si*dy)>hl+half*(abs(dot)+abs(cross)):continue
  if abs(-si*dx+co*dy)>hw+half*(abs(dot)+abs(cross)):continue
  if abs(cb*dx+sb*dy)>half+hl*abs(dot)+hw*abs(cross):continue
  if abs(-sb*dx+cb*dy)>half+hl*abs(cross)+hw*abs(dot):continue
  return True
 return False
@njit(cache=True)
def check(initial,command,previous,p,points,obstacles,sdf,origin,res,latency_steps=10):
 state=initial.copy();d0=previous[0];d1=previous[0];minimum=1e9;backup_speed=0.
 for k in range(latency_steps+10+260):
  if k<latency_steps:target_delta=previous[0];target_speed=previous[1]
  elif k<latency_steps+10:target_delta=command[0];target_speed=command[1]
  else:
   target_delta=command[0]
   if (k-latency_steps-10)%10==0:backup_speed=.8 if state[3]<-.03 else 0.
   target_speed=backup_speed
  effective=d0;d0=d1;d1=target_delta;state=step(state,effective,target_speed,p)
  if occupied(state,points,obstacles,sdf,origin,res):return False,(k+1)*.01
  if k>latency_steps+20 and abs(state[3])<.015:return True,(k+1)*.01
 return True,(latency_steps+270)*.01
@njit(cache=True)
def segment(initial,command,previous_delta,p,n=10):
 state=initial.copy();d0=previous_delta;d1=previous_delta
 for k in range(n):
  effective=d0;d0=d1;d1=command[0];state=step(state,effective,command[1],p)
 return state
class DynamicShield:
 def __init__(self,w):
  self.w=w;cfg=yaml.safe_load((Path(__file__).resolve().parents[3]/'stack_master/config/SIM/sim_params.yaml').read_text());keys=['mu','C_Sf','C_Sr','lf','lr','h','m','I','s_min','s_max','sv_min','sv_max','v_switch','a_max','v_min','v_max'];self.params=np.array([cfg[k] for k in keys],float)
 def prepare(self,state,scan,obstacles):
  self.valid_input=len(scan)==1080 and float(np.mean(np.isfinite(scan)))>.95
  r=np.asarray(scan)[::2];a=np.linspace(-2.35,2.35,1080)[::2]+state[4];origin=state[:2]+.27*np.array([np.cos(state[4]),np.sin(state[4])]);points=origin+r[:,None]*np.c_[np.cos(a),np.sin(a)];valid=np.isfinite(points).all(1)&np.isfinite(r)&(r>.05)&(r<10.);points=points[valid];points=points[self.w.clearance(points[:,0],points[:,1])>.10];obs=np.array([[b['x'],b['y'],b.get('theta',0.),b.get('half_size',.21)] for b in obstacles],float).reshape(-1,4);return points,obs
 def safe(self,state,command,previous,prepared):
  if not getattr(self,'valid_input',True) or not np.isfinite(state).all() or not np.isfinite(command).all():return False,0.
  points,obstacles=prepared;return check(np.asarray(state,dtype=float),np.asarray(command,dtype=float),np.asarray(previous,dtype=float),self.params,points,obstacles,self.w.signed,np.array(self.w.mapcfg['origin'][:2]),self.w.mapcfg['resolution'])
