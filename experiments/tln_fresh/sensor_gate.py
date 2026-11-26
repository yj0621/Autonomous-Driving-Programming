"""Drain ready sensor callbacks once after an executor scheduling blackout.
Normal sensor-loss timeout stays at 0.3 s; blackout grace is bounded to 0.05 s.
"""
class SensorGate:
 def __init__(self,timeout=.3,grace=.05):self.timeout=timeout;self.grace=grace;self.last_tick=None;self.resume_until=0.;self.grace_used=False
 def check(self,now,scan_time,odom_time):
  gap=0. if self.last_tick is None else now-self.last_tick;self.last_tick=now
  stale=max(now-scan_time,now-odom_time)>self.timeout
  if not stale:self.resume_until=0.;self.grace_used=False;return 'fresh',gap
  if gap>self.timeout and not self.grace_used:self.resume_until=now+self.grace;self.grace_used=True
  if now<self.resume_until:return 'drain_pending_callbacks',gap
  return 'stop_sensor_stale',gap
