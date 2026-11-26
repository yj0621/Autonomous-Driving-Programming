"""ROS demo for freshly trained V4; no obstacle ground truth enters controller.
Reset and collision events remain failures in the log. Physics clock drives control.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,time,math,json,argparse
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent));sys.path.insert(0,str(root))
import numpy as np,tensorflow as tf,rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String,Float64MultiArray
from ackermann_msgs.msg import AckermannDriveStamped
from world import World
from inputs_wide import InputsWide
from sensors import Inputs
from controller import Controller
from sensor_gate import SensorGate
class Live(Node):
 def __init__(self,args):
  super().__init__('tln_fresh_v4');self.args=args;self.w=World();self.state=None;self.scan=None;self.physics=0.;self.steer=0.;self.control_time=-1.;self.scan_time=0.;self.odom_time=0.;self.previous_xy=None;self.last=np.zeros(2);self.s=None;self.collision='none';self.collision_since=None;self.reset_time=time.monotonic();self.started=False;self.gate=SensorGate();self.log=open(root/'results/live_v4.jsonl','a',buffering=1)
  tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1);self.model=tf.keras.models.load_model(root/args.model,compile=False);self.features=int(self.model.inputs[1].shape[-1]);self.predict=tf.function(lambda h,f:self.model([h,f],training=False),input_signature=[tf.TensorSpec((None,4,541,2),tf.float32),tf.TensorSpec((None,self.features),tf.float32)]);self.predict(tf.zeros((1,4,541,2)),tf.zeros((1,self.features)));self.clear()
  self.pub=self.create_publisher(AckermannDriveStamped,'/drive',1);self.pose=self.create_publisher(PoseWithCovarianceStamped,'/initialpose',10);self.debug=self.create_publisher(String,'/tln/control_debug',1)
  self.create_subscription(LaserScan,'/scan',self.lidar,qos_profile_sensor_data);self.create_subscription(Odometry,'/car_state/odom_GT',self.odom,qos_profile_sensor_data);self.create_subscription(Float64MultiArray,'/sim/physics_state',self.clock_cb,1);self.create_subscription(String,'/sim/collision',self.collision_cb,10);self.create_subscription(PoseWithCovarianceStamped,'/initialpose',self.pose_cb,10);self.create_timer(.01,self.tick);self.event('ready',model=args.model,decoder=args.decoder,protected=not args.raw);print('V4 READY',args.model,flush=True)
 def event(self,name,**fields):self.log.write(json.dumps({'wall_time':time.time(),'physics_s':self.physics,'event':name,**fields})+'\n')
 def clear(self):
  self.inputs=InputsWide(self.w) if self.features==84 else Inputs(self.w);self.controller=Controller(self.w,protected=not self.args.raw,decoder=self.args.decoder);self.last=np.zeros(2);self.s=None
 def clock_cb(self,m):
  if len(m.data)==2:self.physics,self.steer=map(float,m.data)
 def lidar(self,m):self.scan=np.asarray(m.ranges);self.scan_time=time.monotonic()
 def odom(self,m):
  q=m.pose.pose.orientation;yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));xy=np.array([m.pose.pose.position.x,m.pose.pose.position.y]);v=m.twist.twist.linear.x;rate=m.twist.twist.angular.z;beta=0.
  if self.previous_xy is not None and abs(v)>.5:
   delta=xy-self.previous_xy;length=np.linalg.norm(delta)
   if .002<length<.15:
    direction=math.atan2(delta[1],delta[0])-(math.pi if v<0 else 0);beta=float(np.clip((direction-yaw+math.pi)%(2*math.pi)-math.pi,-.25,.25))
  self.previous_xy=xy;self.state=np.r_[xy,self.steer,v,yaw,rate,beta];self.odom_time=time.monotonic()
 def collision_cb(self,m):
  if m.data in ('wall','obstacle') and m.data!=self.collision:self.event('collision',kind=m.data,state=self.state.tolist() if self.state is not None else None);self.collision_since=time.monotonic()
  self.collision=m.data
 def pose_cb(self,m):self.clear();self.previous_xy=None;self.reset_time=time.monotonic();self.control_time=self.physics;self.event('pose_reset_observed')
 def publish(self,u):
  m=AckermannDriveStamped();m.header.stamp=self.get_clock().now().to_msg();m.drive.steering_angle=float(u[0]);m.drive.speed=float(u[1]);self.pub.publish(m);self.last=np.asarray(u)
 def reset(self):
  # Start on clear raceline. LiDAR endpoint proximity excludes known obstacle vicinity.
  start=0. if self.s is None else self.s+3.
  for s in np.arange(start,start+self.w.length,1.):
   xy=self.w.xy(s);yaw=float(self.w.tangent(s));candidates=xy+np.array([[-.25,0],[.43,0],[0,.15],[0,-.15]])@np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
   if np.min(self.w.clearance(candidates[:,0],candidates[:,1]))<.18:continue
   known=self.inputs.memory
   if any(np.linalg.norm(xy-b['xy'])<2. for b in known):continue
   m=PoseWithCovarianceStamped();m.header.frame_id='map';m.header.stamp=self.get_clock().now().to_msg();m.pose.pose.position.x=float(xy[0]);m.pose.pose.position.y=float(xy[1]);m.pose.pose.orientation.z=math.sin(yaw/2);m.pose.pose.orientation.w=math.cos(yaw/2);self.publish(np.zeros(2));self.pose.publish(m);self.event('reset_requested',s=float(s%self.w.length));self.reset_time=time.monotonic();return
  self.event('reset_unavailable')
 def tick(self):
  if self.state is None or self.scan is None:return
  if self.args.arm_file and not Path(self.args.arm_file).exists():self.publish(np.zeros(2));return
  now=time.monotonic();status,_=self.gate.check(now,self.scan_time,self.odom_time)
  if status=='drain_pending_callbacks':return
  if status!='fresh':self.publish(np.zeros(2));return
  if not self.started:self.started=True;self.publish(np.zeros(2));self.reset();return
  if now-self.reset_time<.6:self.publish(np.zeros(2));return
  if self.collision in ('wall','obstacle'):
   self.publish(np.zeros(2))
   if now-self.reset_time>1.5 and not self.args.no_reset:self.reset()
   return
  if self.physics-self.control_time<.095:return
  self.control_time=self.physics;t=time.perf_counter();h,f=self.inputs.push(self.scan,self.state,self.last,self.s);self.s=self.inputs.s;p=self.predict(h[None].astype(np.float32),f[None]).numpy()[0];u,detail=self.controller.act(self.state,self.scan,f,p,self.inputs.obstacles,self.s);self.publish(u);row={'state':self.state.tolist(),'command':u.tolist(),'prediction':p.tolist(),'guard':detail,'latency_ms':(time.perf_counter()-t)*1000};self.event('control',**row);m=String();m.data=json.dumps(row);self.debug.publish(m)
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--decoder',default='mixture');p.add_argument('--raw',action='store_true');p.add_argument('--no-reset',action='store_true');p.add_argument('--arm-file');args=p.parse_args();rclpy.init();node=Live(args)
try:rclpy.spin(node)
except KeyboardInterrupt:pass
finally:
 if rclpy.ok():node.publish(np.zeros(2))
 node.destroy_node()
 if rclpy.ok():rclpy.shutdown()
