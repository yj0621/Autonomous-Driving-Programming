"""Map-connected lateral DP reference; beam search enforces the actual vehicle path."""
import numpy as np
from numba import njit
GRID=np.arange(-1.5,1.5001,.05)
@njit(cache=True)
def footprint(x,y,yaw,sdf,origin,res,obs):
 for offset in [-.16,.06,.28]:
  cx=x+offset*np.cos(yaw);cy=y+offset*np.sin(yaw);ix=int((cx-origin[0])/res);iy=sdf.shape[0]-1-int((cy-origin[1])/res)
  if ix<0 or iy<0 or ix>=sdf.shape[1] or iy>=sdf.shape[0] or sdf[iy,ix]<.175:return False
  for b in obs:
   dx=cx-b[0];dy=cy-b[1];co=np.cos(b[2]);si=np.sin(b[2]);qx=max(abs(co*dx+si*dy)-b[3],0.);qy=max(abs(-si*dx+co*dy)-b[3],0.)
   if qx*qx+qy*qy<.185**2:return False
 return True
@njit(cache=True)
def search(road,normal,initial,yaw,d0,sdf,origin,res,obs):
 n=len(road);m=len(GRID);cost=np.full((n,m,7),np.inf);back=np.full((n,m,7),-1,np.int16);j0=np.argmin(np.abs(GRID-d0))
 for di in range(7):
  j=j0+di-3
  if j<0 or j>=m:continue
  p=road[1]+normal[1]*GRID[j];angle=np.arctan2(p[1]-initial[1],p[0]-initial[0]);error=(angle-yaw+np.pi)%(2*np.pi)-np.pi
  if abs(error)>.4:continue
  if not footprint(p[0],p[1],angle,sdf,origin,res,obs):continue
  cost[1,j,di]=.15*GRID[j]*GRID[j]+.4*error*error
 for k in range(1,n-1):
  for j in range(m):
   for di in range(7):
    old=cost[k,j,di]
    if not np.isfinite(old):continue
    jp=j-(di-3)
    if jp<0 or jp>=m:continue
    a=road[k-1]+normal[k-1]*GRID[jp]
    if k==1:a=initial
    b=road[k]+normal[k]*GRID[j]
    for nd in range(7):
     jj=j+nd-3
     if jj<0 or jj>=m:continue
     c=road[k+1]+normal[k+1]*GRID[jj];u=b-a;v=c-b;den=np.sqrt((u@u)*(v@v)*((u+v)@(u+v)));curvature=2*(u[0]*v[1]-u[1]*v[0])/max(den,1e-8)
     if abs(curvature)>1.12:continue
     heading=np.arctan2(v[1],v[0])-.5*np.arctan(.3302*curvature)
     if not footprint(c[0],c[1],heading,sdf,origin,res,obs):continue
     middle=(b+c)*.5
     if not footprint(middle[0],middle[1],heading,sdf,origin,res,obs):continue
     value=old+.10*GRID[jj]**2+.10*curvature**2+.12*(nd-di)**2
     if value<cost[k+1,jj,nd]:cost[k+1,jj,nd]=value;back[k+1,jj,nd]=di
 flat=np.argmin(cost[-1]);j=flat//7;di=flat%7
 if not np.isfinite(cost[-1,j,di]):return np.zeros(n),False
 result=np.zeros(n);result[-1]=GRID[j]
 for k in range(n-1,1,-1):
  previous=back[k,j,di];j-=di-3;di=previous;result[k-1]=GRID[j]
 result[0]=GRID[j0];return result,True
class FrenetGuide:
 def __init__(self,w):self.w=w
 def plan(self,state,obstacles,s):
  w=self.w;ss=s+np.arange(0,9.01,.25);road=w.xy(ss);normal=w.normal(ss);d0=float(np.dot(state[:2]-road[0],normal[0]));obs=np.array([[b['x'],b['y'],b.get('theta',0.),b.get('half_size',.15)] for b in obstacles],float).reshape(-1,4);d,ok=search(road,normal,state[:2],float(state[4]),d0,w.signed,np.array(w.mapcfg['origin'][:2]),w.mapcfg['resolution'],obs);return ss,d,ok
