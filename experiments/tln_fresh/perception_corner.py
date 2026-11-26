"""Temporal LiDAR/map foreground with ego-motion-compensated obstacle memory."""
import numpy as np
from perception import Inputs as BaseInputs,decode_action as base_decode
class Inputs(BaseInputs):
 def __init__(self,w):super().__init__(w);self.w=w;self.memory=[]
 def push(self,raw,state,f):
  h,ff=super().push(raw,state,f);w=self.w;yaw=state[4];co,si=np.cos(yaw),np.sin(yaw);rotation=np.array([[co,-si],[si,co]]);s0=w.project(*state[:2])
  for m in self.memory:m['age']+=1
  for b in self.foreground.observations:
   xy=state[:2]+rotation@np.array([b[0]*6,b[1]*3]);match=next((m for m in self.memory if np.linalg.norm(m['xy']-xy)<.5),None)
   if match is None:self.memory.append({'xy':xy.copy(),'age':0})
   else:match['xy']=.35*match['xy']+.65*xy;match['age']=0
  choices=[];kept=[]
  for m in self.memory:
   so=w.project(*m['xy']);ds=(so-s0+w.length/2)%w.length-w.length/2
   if m['age']>30 or ds<-.7 or ds>6.5:continue
   kept.append(m);d=float(np.dot(m['xy']-w.xy(so),w.normal(so)));left=np.interp(so,w.ts,w.left);right=np.interp(so,w.ts,w.right);local=rotation.T@(m['xy']-state[:2])
   b=np.array([local[0]/6,local[1]/3,np.linalg.norm(local)/6,1.,(left-.15-d-.21)/3,(d-.21+right-.15)/3,d/2,ds/6],np.float32)
   choices.append((max(ds,0),b))
  self.memory=kept;self.foreground.observations=[b for _,b in choices];ff[34:]=min(choices,key=lambda x:x[0])[1] if choices else np.zeros(8)
  return h,ff

def corner_speed_cap(features):
 p=np.asarray(features[:16]).reshape(8,2)*5;ds=np.array([0,.5,1,1.5,2,3,4,5]);a=np.diff(p,axis=0);head=np.unwrap(np.arctan2(a[:,1],a[:,0]));curv=abs(np.diff(head))/np.maximum((np.linalg.norm(a,axis=1)[1:]+np.linalg.norm(a,axis=1)[:-1])*.5,.1)
 # Visible map preview, independent of hidden obstacle placement.
 return float(min(2.8,np.sqrt(1.10/max(float(max(curv[:5])),.03))))
def decode_action(output,f):
 u=base_decode(output,f);u[1]=min(u[1],corner_speed_cap(f));return u

def decode_path_action(output,f):
 from policy import decode
 out=np.asarray(output).copy()
 if f[37]<.5:out[:3]=0.
 u=decode(out[:4],f[:34]);u[1]=min(u[1],corner_speed_cap(f));return u
