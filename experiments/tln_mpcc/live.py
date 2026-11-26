"""ROS2 observer/demo for MPCC or trained student. No training in live mode."""
import os,math,time,json,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile,DurabilityPolicy
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import PoseArray,PoseStamped,PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from world import World,History,STEER_MAX,V_MAX
from expert import MPCC
from safety import shield
class Live(Node):
 def __init__(self,mode,model):
  super().__init__('mpcc_teacher_demo' if mode=='expert' else 'tln_temporal_mpcc');self.w=World();self.sensors=ReentrantCallbackGroup();self.solve_delay=.05;self.yaw_rate=0.;self.mode=mode;self.scan=None;self.state=None;self.obstacles=[];self.last_u=np.zeros(2);self.route_s=None;self.kind='none';self.last_reset=0;self.reset_index=0;self.hist=History();self.event_log=open(str(__import__("pathlib").Path(__file__).parent/"results/live_events.jsonl"),"a",buffering=1);self.last_scan_time=0;self.last_control=0;self.fail_since=None
  if mode=='expert':self.controller=MPCC(self.w)
  else:
   import tensorflow as tf
   tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
   self.model=tf.keras.models.load_model(model,compile=False);self.predict=tf.function(lambda h,f:self.model([h,f],training=False),input_signature=[tf.TensorSpec((None,541,3),tf.float32),tf.TensorSpec((None,34),tf.float32)]);self.predict(tf.zeros((1,541,3)),tf.zeros((1,34)))
  self.pub=self.create_publisher(AckermannDriveStamped,'/drive',1);self.pose_pub=self.create_publisher(PoseWithCovarianceStamped,'/initialpose',10);self.plan_pub=self.create_publisher(Path,'/mpcc/plan',1)
  self.create_subscription(Odometry,'/car_state/odom_GT',self.odom,1,callback_group=self.sensors);self.create_subscription(LaserScan,'/scan',self.lidar,1,callback_group=self.sensors)
  self.create_subscription(PoseArray,'/sim/obstacles',self.obstacles_cb,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL),callback_group=self.sensors);self.create_subscription(String,'/sim/collision',self.collision,10,callback_group=self.sensors)
  self.started=time.monotonic();self.create_timer(.1,self.tick);print('LIVE',mode,'started; GT localization in simulation; route-conditioned student',flush=True)
 def odom(self,m):
  self.yaw_rate=m.twist.twist.angular.z
  q=m.pose.pose.orientation;yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));self.state=np.array([m.pose.pose.position.x,m.pose.pose.position.y,self.last_u[0],m.twist.twist.linear.x,yaw])
 def lidar(self,m):self.scan=np.asarray(m.ranges);self.last_scan_time=time.monotonic()
 def obstacles_cb(self,m):self.obstacles=[{'x':p.position.x,'y':p.position.y,'s':self.w.project(p.position.x,p.position.y),'theta':2*math.atan2(p.orientation.z,p.orientation.w)} for p in m.poses]
 def collision(self,m):
  if m.data!=self.kind and m.data in ('wall','obstacle'):
   self.event_log.write(json.dumps({'time':time.time(),'event':'collision','kind':m.data,'state':self.state.tolist() if self.state is not None else None,'command':self.last_u.tolist(),'obstacles':self.obstacles})+'\n')
  self.kind=m.data
 def reset(self):
  for _ in range(len(self.w.track.s)):
   s=self.reset_index%self.w.length;self.reset_index+=2.;xy=self.w.xy(s);yaw=float(self.w.tangent(s))
   if self.w.clearance(*xy)<.48:continue
   if self.w.clearance(*(xy+.5*np.array([np.cos(yaw),np.sin(yaw)])))<.3:continue
   if any(np.linalg.norm(xy-[o['x'],o['y']])<2 for o in self.obstacles):continue
   if self.state is not None and np.linalg.norm(xy-self.state[:2])<2 and self.last_reset>0:continue
   break
  else:print('No clear reset candidate; keeping stopped',flush=True);return
  xy=self.w.xy(s);yaw=float(self.w.tangent(s));m=PoseWithCovarianceStamped();m.header.frame_id='map';m.header.stamp=self.get_clock().now().to_msg();m.pose.pose.position.x=float(xy[0]);m.pose.pose.position.y=float(xy[1]);m.pose.pose.orientation.z=math.sin(yaw/2);m.pose.pose.orientation.w=math.cos(yaw/2);self.pose_pub.publish(m);self.last_reset=time.monotonic();self.last_u[:]=0;self.hist=History();self.route_s=s
  if self.mode=='expert':self.controller.last=None;self.controller.previous[:]=0;self.controller.progress_s=None;self.controller.known.clear()
  self.event_log.write(json.dumps({'time':time.time(),'event':'reset','x':float(xy[0]),'y':float(xy[1]),'yaw':yaw})+'\n');print('pose reset',xy.tolist(),flush=True)
 def publish(self,u):
  msg=AckermannDriveStamped();msg.header.stamp=self.get_clock().now().to_msg();msg.drive.steering_angle=float(u[0]);msg.drive.speed=float(u[1]);self.pub.publish(msg);self.last_u=u
 def tick(self):
  now=time.monotonic()
  if self.state is None or self.scan is None:return
  if self.last_reset==0 and now-self.started>2:self.publish(np.zeros(2));self.reset();return
  if now-self.last_reset<.6:self.publish(np.zeros(2));return
  if self.kind in ('wall','obstacle'):
   self.publish(np.zeros(2))
   if now-self.last_reset>8:self.reset()
   return
  if now-self.last_scan_time>.3:self.publish(np.zeros(2));return
  if self.mode=='expert':
   predicted=self.state.copy();delay=min(self.solve_delay,.15);beta=np.arctan(.5*np.tan(self.last_u[0]));predicted[0]+=predicted[3]*np.cos(predicted[4]+beta)*delay;predicted[1]+=predicted[3]*np.sin(predicted[4]+beta)*delay;predicted[4]+=self.yaw_rate*delay
   u,info=self.controller.act(predicted,self.scan,self.obstacles);self.solve_delay=.7*self.solve_delay+.3*info['solve_ms']/1000
   if not info['ok']:
    if self.fail_since is None:self.fail_since=now;print('MPCC solver failure; stop, not success',flush=True)
    if now-self.fail_since>4 and self.state[3]<.05:self.reset();self.fail_since=None;return
   else:self.fail_since=None
   if info['plan']:
    path=Path();path.header.frame_id='map';path.header.stamp=self.get_clock().now().to_msg()
    for v in info['plan']:
     p=PoseStamped();p.header=path.header;p.pose.position.x=v[0];p.pose.position.y=v[1];p.pose.orientation.w=1.;path.poses.append(p)
    self.plan_pub.publish(path)
  else:
   h=self.hist.push(self.scan);self.route_s=self.w.project(*self.state[:2],hint=self.route_s);f=self.w.route_features(self.state,previous_steer=self.last_u[0],s_hint=self.route_s);out=self.predict(h[None],f[None]).numpy()[0];u=np.array([float(out[0])*STEER_MAX,float(out[1])*V_MAX]);u=np.clip(u,[-STEER_MAX,0],[STEER_MAX,V_MAX])
  u,safety=shield(self.scan,u);self.publish(u)
  if now-self.last_control>3:print(self.mode,'xy',np.round(self.state[:2],2),'u',np.round(u,3),flush=True);self.last_control=now
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['expert','student'],default='expert');p.add_argument('--model');args=p.parse_args();rclpy.init();node=Live(args.mode,args.model)
executor=MultiThreadedExecutor(num_threads=3);executor.add_node(node)
try:executor.spin()
except KeyboardInterrupt:node.publish(np.zeros(2))
finally:executor.shutdown();node.destroy_node();rclpy.shutdown()
