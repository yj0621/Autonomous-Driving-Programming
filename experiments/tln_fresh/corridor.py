"""Nonholonomic local corridor supervisor from LiDAR/map perception, not obstacle GT."""
import numpy as np
from numba import njit
DS=.25
GRID=np.arange(-1.8,1.8001,.05)
@njit(cache=True)
def search(road,normals,left,right,obstacles,d0):
 n=len(road);m=len(GRID);cost=np.full((n,m,7),np.inf);back=np.full((n,m,7),-1,np.int16);j0=int(np.argmin(np.abs(GRID-d0)))
 for delta in range(-3,4):
  j=j0+delta
  if j<0 or j>=m:continue
  x=road[1,0]+normals[1,0]*GRID[j];y=road[1,1]+normals[1,1]*GRID[j]
  if x<=0 or abs(np.arctan2(y,x))>.35:continue
  if GRID[j]<-right[1]+.25 or GRID[j]>left[1]-.25:continue
  cost[1,j,delta+3]=.15*GRID[j]**2
 for k in range(1,n-1):
  for j in range(m):
   for di in range(7):
    old=cost[k,j,di]
    if not np.isfinite(old):continue
    jp=j-(di-3)
    if jp<0 or jp>=m:continue
    x0=road[k-1,0]+normals[k-1,0]*GRID[jp];y0=road[k-1,1]+normals[k-1,1]*GRID[jp]
    if k==1:x0=0.;y0=0.
    x1=road[k,0]+normals[k,0]*GRID[j];y1=road[k,1]+normals[k,1]*GRID[j]
    for nd in range(7):
     jj=j+(nd-3)
     if jj<0 or jj>=m:continue
     d=GRID[jj]
     if d<-right[k+1]+.25 or d>left[k+1]-.25:continue
     penalty=0.;blocked=False;s=(k+1)*DS
     for b in obstacles:
      if b[3]<.5:continue
      if b[0]-.65<s<b[0]+.25 and abs(d-b[1])<.40:blocked=True;break
      if abs(s-b[0])<1.:
       desired=1 if b[2]>0 else -1
       if (d-b[1])*desired<0:penalty+=.6
     if blocked:continue
     x2=road[k+1,0]+normals[k+1,0]*d;y2=road[k+1,1]+normals[k+1,1]*d
     ax=x1-x0;ay=y1-y0;bx=x2-x1;by=y2-y1;den=np.sqrt((ax*ax+ay*ay)*(bx*bx+by*by)*((ax+bx)**2+(ay+by)**2))
     curvature=2*(ax*by-ay*bx)/max(den,1e-6)
     if abs(curvature)>1.13:continue
     value=old+.15*d*d+.20*curvature*curvature+.15*(nd-di)**2+penalty
     if value<cost[k+1,jj,nd]:cost[k+1,jj,nd]=value;back[k+1,jj,nd]=di
 end=cost[n-1];flat=np.argmin(end);j=flat//7;di=flat%7
 if not np.isfinite(end[j,di]):return np.zeros(n),False
 result=np.zeros(n);result[n-1]=GRID[j]
 for k in range(n-1,1,-1):
  previous_di=back[k,j,di];j=j-(di-3);di=previous_di;result[k-1]=GRID[j]
 result[0]=GRID[j0]
 return result,True
class Corridor:
 def __init__(self):self.obstacles=[]
 def plan(self,f,observations):
  speed=float(f[32]*2.8)
  for b in self.obstacles:b[0]-=max(speed,0)*.1
  self.obstacles=[b for b in self.obstacles if b[0]>-.8]
  for desc in observations:
   ds=float(desc[7]*6);d=float(desc[6]*2);side=1 if desc[4]>desc[5] else -1
   if ds<-.5 or ds>5.8:continue
   match=next((b for b in self.obstacles if abs(b[0]-ds)<.5 and abs(b[1]-d)<.3),None)
   if match is None:self.obstacles.append([ds,d,side,1.])
   else:match[:]=[ds,d,side,1.]
  if not self.obstacles:return None,False
  ahead=[b[0] for b in self.obstacles if b[0]>.0];horizon=np.clip((min(ahead)+1.0) if ahead else 2.5,2.5,5.0);ss=np.arange(int(np.ceil(horizon/DS))+1)*DS;ds=np.array([0,.5,1,1.5,2,3,4,5]);p=np.asarray(f[:16]).reshape(8,2)*5;road=np.c_[np.interp(ss,ds,p[:,0]),np.interp(ss,ds,p[:,1])];t=np.gradient(road,axis=0);normals=np.c_[-t[:,1],t[:,0]]/np.maximum(np.linalg.norm(t,axis=1)[:,None],.01);width=np.asarray(f[16:32]).reshape(8,2)*3;left=np.interp(ss,ds,width[:,0]);right=np.interp(ss,ds,width[:,1]);d0=-float(np.dot(road[0],normals[0]));ob=np.asarray(self.obstacles[:10],np.float64)
  offsets,ok=search(road,normals,left,right,ob,d0)
  return (ss,offsets,road,normals),ok
