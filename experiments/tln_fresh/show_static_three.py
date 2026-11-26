"""Three fixed obstacles for RViz student demonstration; records failures honestly."""
import json,math,time
from pathlib import Path
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy
from geometry_msgs.msg import PoseArray,Pose
from visualization_msgs.msg import Marker,MarkerArray
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from world import World
class Layout(Node):
 def __init__(self):
  super().__init__('tln_three_static_obstacles');self.w=World();self.state=None;self.active=False;self.previous=None;self.last_xy=None;self.progress=0.;self.collision='none';self.low_since=None;self.stop_reported=False;self.events=[];self.obstacles=[];self.laps=[];self.started=time.monotonic();self.lap_started=self.started
  for i,(s,d,label) in enumerate([(10.,.22,'straight'),(33.1,.28,'corner_exit'),(46.,-.20,'return')]):
   xy=self.w.xy(s)+self.w.normal(s)*d;theta=float(self.w.tangent(s));assert self.w.clearance(*xy)>.3
   self.obstacles.append({'id':i+1,'s':s,'d':d,'x':float(xy[0]),'y':float(xy[1]),'theta':theta,'size_m':.3,'location':label})
  qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL);self.pub=self.create_publisher(PoseArray,'/sim/obstacles',qos);self.markers=self.create_publisher(MarkerArray,'/sim/obstacles/markers',qos)
  self.create_subscription(Odometry,'/car_state/odom_GT',self.odom,1);self.create_subscription(String,'/sim/collision',self.col,10);self.create_timer(.5,self.tick);print('THREE STATIC OBSTACLES',json.dumps(self.obstacles),flush=True)
 def event(self,name,**fields):
  value={'event':name,'time':time.time(),**fields};self.events.append(value);print(json.dumps(value),flush=True)
 def col(self,m):
  if self.active and m.data in ('wall','obstacle') and m.data!=self.collision:self.event('collision',kind=m.data,progress_m=self.progress);self.progress=0.;self.lap_started=time.monotonic()
  self.collision=m.data
 def odom(self,m):
  xy=np.array([m.pose.pose.position.x,m.pose.pose.position.y]);v=m.twist.twist.linear.x;self.state=(xy,v)
  if not self.active:return
  if self.last_xy is not None and np.linalg.norm(xy-self.last_xy)>1.5:self.event('pose_reset',xy=xy.tolist());self.previous=None;self.progress=0.;self.lap_started=time.monotonic()
  s=self.w.project(*xy,hint=self.previous)
  if self.previous is not None:self.progress+=(s-self.previous+self.w.length/2)%self.w.length-self.w.length/2
  self.previous=s;self.last_xy=xy
  if self.progress>=self.w.length:
   row={'elapsed_wall_s':time.monotonic()-self.lap_started,'progress_m':self.progress,'collision_free':True};self.laps.append(row);self.event('full_lap',**row);self.progress=0.;self.lap_started=time.monotonic()
 def tick(self):
  now=time.monotonic()
  if not self.active:
   if self.state is None:return
   xy,v=self.state;s=self.w.project(*xy);ahead=[(b['s']-s)%self.w.length for b in self.obstacles]
   if min(ahead)<5. or any(np.linalg.norm(xy-[b['x'],b['y']])<2.5 for b in self.obstacles):return
   self.active=True;self.previous=s;self.last_xy=xy;self.lap_started=now;self.event('activated',car_s=s)
  msg=PoseArray();msg.header.frame_id='map';msg.header.stamp=self.get_clock().now().to_msg();ma=MarkerArray();delete=Marker();delete.action=Marker.DELETEALL;ma.markers.append(delete)
  for b in self.obstacles:
   p=Pose();p.position.x=b['x'];p.position.y=b['y'];p.orientation.z=math.sin(b['theta']/2);p.orientation.w=math.cos(b['theta']/2);msg.poses.append(p);m=Marker();m.header=msg.header;m.ns='tln_static_three';m.id=b['id'];m.type=Marker.CUBE;m.action=Marker.ADD;m.pose.position.x=b['x'];m.pose.position.y=b['y'];m.pose.position.z=.15;m.pose.orientation=p.orientation;m.scale.x=m.scale.y=m.scale.z=.3;m.color.r=1.;m.color.g=.15;m.color.b=.05;m.color.a=1.;ma.markers.append(m)
  self.pub.publish(msg);self.markers.publish(ma)
  if self.state[1]<.05:
   if self.low_since is None:self.low_since=now
   if now-self.low_since>2 and not self.stop_reported:self.event('stopped',progress_m=self.progress);self.stop_reported=True
  else:self.low_since=None;self.stop_reported=False
  output={'active':self.active,'map':'hangar_1905_v0','controller':'new TLN + geometric guard','obstacles':self.obstacles,'current_progress_m':self.progress,'laps':self.laps,'events':self.events};(Path(__file__).parent/'corner_revision/results/rviz_three_obstacles.json').write_text(json.dumps(output,indent=2))
rclpy.init();n=Layout()
try:rclpy.spin(n)
except KeyboardInterrupt:pass
finally:
 n.destroy_node()
 if rclpy.ok():rclpy.shutdown()
