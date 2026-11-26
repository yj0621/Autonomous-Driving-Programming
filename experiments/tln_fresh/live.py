"""ROS2 observer/demo for MPCC or trained student. No training in live mode."""
import os,math,time,json,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import PoseArray,PoseStamped,PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String,Float64MultiArray
from world import World,History,STEER_MAX,V_MAX
from policy import decode,basic_action
from supervisor import Supervisor
from perception import Inputs,decode_action
from guard_v2 import Guard
from expert import MPCC
from safety import shield
from sensor_gate import SensorGate
class Live(Node):
 def __init__(self,mode,model,cruise=1.3,revision="previous",start_s=0.,auto_reset=True):
  self.revision=revision;self.auto_reset=auto_reset
  if revision=='corner':
   global Inputs,Guard
   from perception_corner import Inputs
   from guard_corner import Guard
  super().__init__('mpcc_teacher_demo' if mode=='expert' else 'tln_temporal_mpcc');self.cruise=cruise;self.w=World();self.sensors=ReentrantCallbackGroup();self.solve_delay=.05;self.physics_clock=False;self.measured_steer=0.;self.sim_elapsed=0.;self.last_sim_control=0.;self.last_control_wall=0.;self.previous_odom=None;self.previous_stamp=None;self.rtf=1.;self.yaw_rate=0.;self.mode=mode;self.scan=None;self.state=None;self.obstacles=[];self.last_u=np.zeros(2);self.route_s=None;self.kind='none';self.last_reset=0;self.reset_index=start_s;self.hist=History();self.event_log=open(str(__import__("pathlib").Path(__file__).parent/"results/live_events.jsonl"),"a",buffering=1);self.last_scan_time=0;self.last_odom_time=0;self.last_control=0;self.fail_since=None;self.sensor_gate=SensorGate();self.command_reason="startup";self.previous_reason=None;self.last_reason_log=0
  if revision=='v4_basic':
   import sys
   sys.path.insert(0,str(__import__('pathlib').Path(__file__).parent/'revision_random_v4'))
   from fast_basic import FastTracker
   self.fast_tracker=FastTracker(self.w,cruise=cruise)
  if mode=='expert':
   if revision=='corner':
    from expert_corner import CornerMPCC
    self.controller=CornerMPCC(self.w,cruise=cruise)
   else:self.controller=MPCC(self.w)
  elif mode=='student':
   import tensorflow as tf
   tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
   self.model=tf.keras.models.load_model(model,compile=False);self.inputs=Inputs(self.w);self.guard=Guard();self.predict=tf.function(lambda h,f:self.model([h,f],training=False),input_signature=[tf.TensorSpec((None,541,6),tf.float32),tf.TensorSpec((None,42),tf.float32)]);self.predict(tf.zeros((1,541,6)),tf.zeros((1,42)))
  self.debug_pub=self.create_publisher(String,'/tln/control_debug',1);self.debug_log=open(str(__import__('pathlib').Path(__file__).parent/'revision_random_v4/results/live_debug.jsonl'),'a',buffering=1) if mode=='student' else None
  self.pub=self.create_publisher(AckermannDriveStamped,'/drive',1);self.pose_pub=self.create_publisher(PoseWithCovarianceStamped,'/initialpose',10);self.plan_pub=self.create_publisher(Path,'/mpcc/plan',1)
  sensor_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
  self.create_subscription(Float64MultiArray,'/sim/physics_state',self.physics,1,callback_group=self.sensors)
  self.create_subscription(Odometry,'/car_state/odom_GT',self.odom,sensor_qos,callback_group=self.sensors);self.create_subscription(LaserScan,'/scan',self.lidar,sensor_qos,callback_group=self.sensors)
  self.create_subscription(PoseArray,'/sim/obstacles',self.obstacles_cb,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL),callback_group=self.sensors);self.create_subscription(String,'/sim/collision',self.collision,10,callback_group=self.sensors)
  self.supervisor=Supervisor();self.started=time.monotonic();self.create_timer((.1 if revision=='v4_basic' else .05) if mode=='basic' else .01,self.tick);print('LIVE',mode,'pid',os.getpid(),'started; GT localization in simulation; route-conditioned student',flush=True)
 def physics(self,m):
  if len(m.data)!=2:return
  self.physics_clock=True;self.sim_elapsed=float(m.data[0]);self.measured_steer=float(m.data[1])
 def odom(self,m):
  self.last_odom_time=time.monotonic()
  stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9
  current=np.array([m.pose.pose.position.x,m.pose.pose.position.y,m.twist.twist.linear.x])
  if self.previous_stamp is not None and stamp>self.previous_stamp:
   distance=np.linalg.norm(current[:2]-self.previous_odom[:2]);velocity=(abs(current[2])+abs(self.previous_odom[2]))/2
   simulated=distance/max(velocity,.05) if velocity>.1 and distance<.5 else .01
   if simulated<.1 and not self.physics_clock:
    self.sim_elapsed+=simulated;ratio=np.clip(simulated/(stamp-self.previous_stamp),.05,1.5);self.rtf=.9*self.rtf+.1*ratio
  self.previous_odom=current;self.previous_stamp=stamp
  self.yaw_rate=m.twist.twist.angular.z
  q=m.pose.pose.orientation;yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));self.state=np.array([m.pose.pose.position.x,m.pose.pose.position.y,(self.measured_steer if self.physics_clock else self.last_u[0]),m.twist.twist.linear.x,yaw])
 def lidar(self,m):self.scan=np.asarray(m.ranges);self.last_scan_time=time.monotonic()
 def obstacles_cb(self,m):self.obstacles=[{'x':p.position.x,'y':p.position.y,'s':self.w.project(p.position.x,p.position.y),'theta':2*math.atan2(p.orientation.z,p.orientation.w)} for p in m.poses]
 def collision(self,m):
  if m.data!=self.kind and m.data in ('wall','obstacle'):
   self.event_log.write(json.dumps({'time':time.time(),'event':'collision','kind':m.data,'state':self.state.tolist() if self.state is not None else None,'command':self.last_u.tolist(),'obstacles':self.obstacles})+'\n')
  self.kind=m.data
 def reset(self):
  for _ in range(len(self.w.track.s)):
   s=self.reset_index%self.w.length;self.reset_index+=2.;xy=self.w.xy(s);yaw=float(self.w.tangent(s))
   if self.w.clearance(*xy)<.28:continue
   if self.w.clearance(*(xy+.43*np.array([np.cos(yaw),np.sin(yaw)])))<.16:continue
   if any(np.linalg.norm(xy-[o['x'],o['y']])<2 for o in self.obstacles):continue
   if self.state is not None and np.linalg.norm(xy-self.state[:2])<2 and self.last_reset>0:continue
   break
  else:print('No clear reset candidate; keeping stopped',flush=True);return
  xy=self.w.xy(s);yaw=float(self.w.tangent(s));m=PoseWithCovarianceStamped();m.header.frame_id='map';m.header.stamp=self.get_clock().now().to_msg();m.pose.pose.position.x=float(xy[0]);m.pose.pose.position.y=float(xy[1]);m.pose.pose.orientation.z=math.sin(yaw/2);m.pose.pose.orientation.w=math.cos(yaw/2);self.pose_pub.publish(m);self.last_reset=time.monotonic();self.last_u[:]=0;self.hist=History();self.route_s=s;self.last_sim_control=self.sim_elapsed
  if self.mode=='student':self.inputs=Inputs(self.w);self.guard=Guard()
  if self.mode=='expert':self.controller.last=None;self.controller.previous[:]=0;self.controller.progress_s=None;self.controller.known.clear()
  self.event_log.write(json.dumps({'time':time.time(),'event':'reset','x':float(xy[0]),'y':float(xy[1]),'yaw':yaw})+'\n');print('pose reset',xy.tolist(),flush=True)
 def publish(self,u):
  now=time.monotonic()
  if self.command_reason!=self.previous_reason or now-self.last_reason_log>2:
   self.event_log.write(json.dumps({'time':time.time(),'event':'control_status','mode':self.mode,'reason':self.command_reason,'scan_age_s':now-self.last_scan_time,'odom_age_s':now-self.last_odom_time,'state':self.state.tolist() if self.state is not None else None,'command':np.asarray(u).tolist()})+'\n');self.previous_reason=self.command_reason;self.last_reason_log=now
  msg=AckermannDriveStamped();msg.header.stamp=self.get_clock().now().to_msg();msg.drive.steering_angle=float(u[0]);msg.drive.speed=float(u[1]);self.pub.publish(msg);self.last_u=u
 def tick(self):
  now=time.monotonic()
  if self.state is None or self.scan is None:return
  self.command_reason="tracking"
  if self.last_reset==0:
   self.command_reason="startup";self.publish(np.zeros(2))
   if now-self.started>1:self.reset()
   return
  if now-self.last_reset<.6:self.command_reason="reset_settle";self.publish(np.zeros(2));return
  if self.kind in ('wall','obstacle'):
   self.command_reason='collision_'+self.kind;self.publish(np.zeros(2))
   if self.auto_reset and now-self.last_reset>8:self.reset()
   return
  sensor_status,gap=self.sensor_gate.check(now,self.last_scan_time,self.last_odom_time)
  if sensor_status=="drain_pending_callbacks":
   self.event_log.write(json.dumps({'time':time.time(),'event':'executor_resume','gap_s':gap,'grace_s':.05})+'\n');return
  if sensor_status=="stop_sensor_stale":
   self.command_reason="sensor_stale";self.publish(np.zeros(2));return
  if self.mode!='basic' and self.sim_elapsed-self.last_sim_control<.098 and (self.physics_clock or now-self.last_control_wall<.18):return
  self.last_sim_control=self.sim_elapsed;self.last_control_wall=now
  if self.mode=='expert':
   predicted=self.state.copy();delay=0. if self.physics_clock else min(self.solve_delay*self.rtf,.10);beta=np.arctan(.5*np.tan(self.last_u[0]));predicted[0]+=predicted[3]*np.cos(predicted[4]+beta)*delay;predicted[1]+=predicted[3]*np.sin(predicted[4]+beta)*delay;predicted[4]+=self.yaw_rate*delay
   u,info=self.controller.act(predicted,self.scan,self.obstacles);self.solve_delay=.7*self.solve_delay+.3*info['solve_ms']/1000
   if not info['ok']:
    if self.fail_since is None:self.fail_since=now;print('MPCC solver failure; stop, not success',flush=True)
    if self.auto_reset and now-self.fail_since>4 and self.state[3]<.05:self.reset();self.fail_since=None;return
   else:self.fail_since=None
   if info['plan']:
    path=Path();path.header.frame_id='map';path.header.stamp=self.get_clock().now().to_msg()
    for v in info['plan']:
     p=PoseStamped();p.header=path.header;p.pose.position.x=v[0];p.pose.position.y=v[1];p.pose.orientation.w=1.;path.poses.append(p)
    self.plan_pub.publish(path)
  elif self.mode=='basic':
   self.route_s=self.w.project(*self.state[:2],hint=self.route_s)
   if self.revision=='v4_basic':u,f=self.fast_tracker.act(self.state,self.route_s)
   else:u,f=basic_action(self.w,self.state,self.route_s,cruise=self.cruise)
  else:
   control_started=time.perf_counter()
   self.route_s=self.w.project(*self.state[:2],hint=self.route_s);base=self.w.route_features(self.state,previous_steer=self.last_u[0],s_hint=self.route_s);h,f=self.inputs.push(self.scan,self.state,base);out=self.predict(h[None],f[None]).numpy()[0];u,detail=self.guard.act(self.scan,f,out,self.inputs.foreground.observations)
   debug={'wall_time':time.time(),'monotonic':now,'physics_time':self.sim_elapsed,'state':self.state.tolist(),'raw_output':out.tolist(),'command':u.tolist(),'guard':detail,'foreground':f[34:].tolist(),'route_features':f[:34].tolist(),'scan_age_s':now-self.last_scan_time,'odom_age_s':now-self.last_odom_time,'compute_ms':(time.perf_counter()-control_started)*1000};message=String();message.data=json.dumps(debug);self.debug_pub.publish(message);self.debug_log.write(message.data+'\n')

  if self.mode=='expert' and self.revision!='corner':u,safety=shield(self.scan,u)
  if u[1]<.05:self.command_reason="planner_or_guard_stop"
  self.publish(u)
  # Control telemetry goes to the event file; a full terminal pipe cannot block control.
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['expert','student','basic'],default='expert');p.add_argument('--model');p.add_argument('--cruise',type=float,default=1.3);p.add_argument('--revision',choices=['previous','corner','v4_basic'],default='previous');p.add_argument('--start-s',type=float,default=0.);p.add_argument('--no-auto-reset',action='store_true');args=p.parse_args();rclpy.init();node=Live(args.mode,args.model,args.cruise,args.revision,args.start_s,not args.no_auto_reset)
executor=MultiThreadedExecutor(num_threads=3);executor.add_node(node)
try:executor.spin()
except KeyboardInterrupt:
 if rclpy.ok():node.publish(np.zeros(2))
finally:
 executor.shutdown();node.destroy_node()
 if rclpy.ok():rclpy.shutdown()
