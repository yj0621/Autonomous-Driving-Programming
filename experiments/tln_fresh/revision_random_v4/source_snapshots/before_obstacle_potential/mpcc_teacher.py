"""MPCC following a collision-checked nonholonomic route through observed obstacles."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpcc_core import CornerMPCC
from beam_path import BeamPath
from recovery import ReverseRecovery
class Expert(CornerMPCC):
 def __init__(self,w,cruise=3.0):
  self.recovery=ReverseRecovery(w);self.slow_count=0;self.beam=BeamPath(w);self.path=None;self.path_ok=False;self.path_ms=0.;super().__init__(w,horizon=22,cruise=cruise)
 def choose_goal(self,so,d,half,choices,chosen):
  if self.path is not None and self.path_ok:
   ss=self.path[:,4];offset=((self.path[:,:2]-self.w.xy(ss))*self.w.normal(ss)).sum(-1);return float(np.interp(so,ss,offset))
  return super().choose_goal(so,d,half,choices,chosen)
 def make_guide(self,ss,ob):
  if self.path is None or not self.path_ok:return super().make_guide(ss,ob)
  path=self.path;arc=np.maximum.accumulate(path[:,4]);ids=np.r_[True,np.diff(arc)>1e-5];arc=arc[ids];path=path[ids];offset=((path[:,:2]-self.w.xy(arc))*self.w.normal(arc)).sum(-1);d=np.interp(ss,arc,offset);head=np.interp(ss,arc,np.unwrap(path[:,2]));curvature=abs(np.tan(path[:,3])/.3302)
  caps=np.minimum(self.cruise_value,np.sqrt(1.9/np.maximum(curvature,.025)))
  for i in range(len(caps)):
   ahead=np.maximum(arc[i:]-arc[i],0);caps[i]=min(caps[i],float(np.min(np.sqrt(caps[i:]**2+2*2.8*ahead))))
  for b in ob[:,ob[4]>0].T:
   # A local cap at the pass point, not a low speed across every straight segment.
   caps=np.minimum(caps,np.sqrt(1.25**2+2*2.5*np.maximum(abs(arc-b[2])-.4,0)))
  speed=np.interp(ss,arc,caps)
  if len(self.path)<51:speed=np.minimum(speed,np.sqrt(2*2.8*np.maximum(arc[-1]-ss-.25,0.)))
  return np.vstack([d,speed,head])
 def act(self,state,scan,obstacles):
  self.recovery.observe(state)
  if self.recovery.active:
   u,done=self.recovery.act(state,list(self.known.values()))
   if done:self.previous[:]=0;self.last=None;self.slow_count=0
   return u,{'ok':True,'status':'bounded_reverse_recovery','plan':[],'solve_ms':0.,'beam_ms':0.,'safety':{'limited':True},'recovery':True}
  start=time.perf_counter();s=self.w.project(*state[:2],hint=self.progress_s)
  visible=self.visible(state,scan,obstacles)
  for b in visible:self.remember(b)
  active=[b for b in self.known.values() if -1.5<(b.get('s',self.w.project(b['x'],b['y']))-s+self.w.length/2)%self.w.length-self.w.length/2<8.5]
  previous_path=self.path
  self.path,self.path_ok=self.beam.plan(state,active,s);self.path_ms=(time.perf_counter()-start)*1000
  if not self.path_ok:
   self.last=None
   self.slow_count+=1
   if self.slow_count>=2 and abs(state[3])<.15 and self.recovery.start(state):
    u,done=self.recovery.act(state,active);return u,{'ok':True,'status':'bounded_reverse_recovery','plan':[],'solve_ms':self.path_ms,'beam_ms':self.path_ms,'safety':{'limited':True},'recovery':True}
   return np.array([state[2],0.]),{'ok':False,'status':'no_checked_path','plan':[],'solve_ms':self.path_ms,'beam_ms':self.path_ms,'safety':{'limited':True}}
  # Refresh from the checked route to avoid locking an obsolete passage side.
  if self.last is not None and previous_path is not None and len(previous_path)>0:
   probe=s+np.array([1.,2.]);old_s=previous_path[:,4];new_s=self.path[:,4];old_d=((previous_path[:,:2]-self.w.xy(old_s))*self.w.normal(old_s)).sum(-1);new_d=((self.path[:,:2]-self.w.xy(new_s))*self.w.normal(new_s)).sum(-1)
   if np.max(abs(np.interp(probe,old_s,old_d)-np.interp(probe,new_s,new_d)))>.12:self.last=None
  u,info=super().act(state,scan,obstacles)
  self.slow_count=self.slow_count+1 if (not info['ok'] or u[1]<.10) else 0
  if self.slow_count>=3 and abs(state[3])<.15 and self.recovery.start(state):
   u,done=self.recovery.act(state,active);info['ok']=True;info['status']='bounded_reverse_recovery';info['recovery']=True
  info['beam_ms']=self.path_ms;info['solve_ms']+=self.path_ms;return u,info
