import sys,time,math,json
sys.path.insert(0,'/home/misys/forza_ws/race_stack/experiments/tln_fresh')
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy
from geometry_msgs.msg import PoseArray,Pose
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker,MarkerArray
from world import World
class RandomThree(Node):
 def __init__(self):
  super().__init__('random_three_static');self.w=World();self.xy=self.w.xy(0);self.rng=np.random.default_rng();self.changed=-100.;self.items=[]
  q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL);self.pub=self.create_publisher(PoseArray,'/sim/obstacles',q);self.mark=self.create_publisher(MarkerArray,'/sim/obstacles/markers',q);self.create_subscription(Odometry,'/car_state/odom_GT',self.odom,1);self.create_timer(.5,self.tick)
 def odom(self,m):self.xy=np.array([m.pose.pose.position.x,m.pose.pose.position.y])
 def tick(self):
  if time.monotonic()-self.changed>=20:
   items=[]
   for _ in range(3000):
    s=float(self.rng.uniform(0,self.w.length));xy=self.w.xy(s)+self.w.normal(s)*self.rng.uniform(-.35,.35)
    if self.w.clearance(*xy)<.4 or np.linalg.norm(xy-self.xy)<5 or any(np.linalg.norm(xy-b[0])<4 for b in items):continue
    items.append((xy,float(self.w.tangent(s))))
    if len(items)==3:break
   if len(items)==3:self.items=items;self.changed=time.monotonic();print('RELOCATED',[(xy.tolist(),a) for xy,a in items],flush=True)
  msg=PoseArray();msg.header.frame_id='map';msg.header.stamp=self.get_clock().now().to_msg();ma=MarkerArray()
  for i,(xy,a) in enumerate(self.items):
   p=Pose();p.position.x=float(xy[0]);p.position.y=float(xy[1]);p.orientation.z=math.sin(a/2);p.orientation.w=math.cos(a/2);msg.poses.append(p)
   m=Marker();m.header=msg.header;m.ns='random_static';m.id=i;m.type=Marker.CUBE;m.pose=p;m.pose.position.z=.15;m.scale.x=m.scale.y=m.scale.z=.3;m.color.r=1.;m.color.g=.15;m.color.a=1.;ma.markers.append(m)
  self.pub.publish(msg);self.mark.publish(ma)
rclpy.init();rclpy.spin(RandomThree())
