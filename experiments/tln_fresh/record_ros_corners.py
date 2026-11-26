"""Record visible ROS expert episodes; separate resets and failures from clean segments."""
import os,time,json,subprocess,signal,math,argparse
from pathlib import Path
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy
from geometry_msgs.msg import PoseArray,Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker,MarkerArray
from world import World
parser=argparse.ArgumentParser();parser.add_argument('--dest',default='corner_revision/ros_recordings_background');parser.add_argument('--episodes',default='train_corner_exit_41000,train_corner_exit_41011,train_corner_exit_41018,train_mpcc_41028');parser.add_argument('--max-wall',type=float,default=600.);args=parser.parse_args()
root=Path(__file__).parent;dest=root/args.dest;dest.mkdir(exist_ok=True)
rclpy.init();n=Node('corner_recording_layout');pub=n.create_publisher(PoseArray,'/sim/obstacles',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL));markers=n.create_publisher(MarkerArray,'/sim/obstacle_markers',10);meta=n.create_publisher(String,'/tln/recording_episode',10);stop=n.create_publisher(AckermannDriveStamped,'/drive',1);state={'xy':None,'collision':'none'};w=World()
def odom(m):state['xy']=[m.pose.pose.position.x,m.pose.pose.position.y];state['v']=m.twist.twist.linear.x
subs=[n.create_subscription(Odometry,'/car_state/odom_GT',odom,1),n.create_subscription(String,'/sim/collision',lambda m:state.update(collision=m.data),10)]
reports=[]
# Successful training layouts; no validation or test samples enter training.
for name in args.episodes.split(','):
 r=json.loads((root/'corner_revision/data'/f'{name}.json').read_text())
 if not r['success']:continue
 def publish_layout():
  msg=PoseArray();msg.header.frame_id='map';msg.header.stamp=n.get_clock().now().to_msg();ma=MarkerArray();delete=Marker();delete.action=Marker.DELETEALL;ma.markers.append(delete)
  for j,b in enumerate(r['obstacles']):
   p=Pose();p.position.x=b['x'];p.position.y=b['y'];p.orientation.z=math.sin(b['theta']/2);p.orientation.w=math.cos(b['theta']/2);msg.poses.append(p);m=Marker();m.header=msg.header;m.ns='recording_obstacles';m.id=j;m.type=Marker.CUBE;m.action=Marker.ADD;m.pose=p;m.pose.position.z=.15;m.scale.x=m.scale.y=m.scale.z=.3;m.color.r=1.;m.color.g=.25;m.color.a=1.;ma.markers.append(m)
  pub.publish(msg);markers.publish(ma)
 stop.publish(AckermannDriveStamped());publish_layout();rclpy.spin_once(n,timeout_sec=.1)
 bag=dest/name
 if bag.exists():raise RuntimeError('Refusing to overwrite '+str(bag))
 baglog=open(dest/f'{name}_bag.log','w');blog=subprocess.Popen(['ros2','bag','record','-o',str(bag),'/scan','/car_state/odom_GT','/drive','/sim/obstacles','/sim/collision','/initialpose','/mpcc/plan','/tln/recording_episode','/sim/physics_state'],stdout=baglog,stderr=subprocess.STDOUT)
 log=open(dest/f'{name}_controller.log','w');control=subprocess.Popen(['python3',str(root/'live.py'),'--mode','expert','--revision','corner','--cruise','2.2','--start-s',str(r['start_s']),'--no-auto-reset'],stdout=log,stderr=subprocess.STDOUT)
 begun=time.monotonic();prev=None;progress=0.;lastpub=0.;active=False;collisions=[];stall=0.;success=False
 try:
  while time.monotonic()-begun<args.max_wall:
   now=time.monotonic();rclpy.spin_once(n,timeout_sec=.02)
   if now-lastpub>.5:publish_layout();lastpub=now;message=String();message.data=json.dumps({'episode':name,'phase':'recording','progress_m':progress});meta.publish(message)
   xy=state['xy']
   if xy is not None:
    if not active and np.linalg.norm(np.array(xy)-w.xy(r['start_s']))<.4 and now-begun>1:active=True;prev=w.project(*xy,hint=r['start_s'])
    if active:
     s=w.project(*xy,hint=prev);delta=(s-prev+w.length/2)%w.length-w.length/2
     if abs(delta)>.8:collisions.append('unexpected_teleport');break
     progress+=delta;prev=s
     if state['collision'] in ('wall','obstacle'):collisions.append(state['collision']);break
     if progress>=r['required_progress_m']:success=True;break
     if abs(state.get('v',0))<.05 and now-begun>8:
      if not stall:stall=now
      if now-stall>8:break
     else:stall=0.
   if control.poll() is not None:break
 finally:
  # rclpy's SIGINT handler may raise before a child is reaped. Always close bag.
  try:
   if control.poll() is None:control.send_signal(signal.SIGINT)
   try:control.wait(timeout=8)
   except subprocess.TimeoutExpired:control.terminate();control.wait(timeout=5)
  finally:
   if rclpy.ok():stop.publish(AckermannDriveStamped())
   if blog.poll() is None:blog.send_signal(signal.SIGINT)
   try:blog.wait(timeout=15)
   except subprocess.TimeoutExpired:blog.terminate();blog.wait(timeout=5)
   log.close();baglog.close()
 report={'episode':name,'success':success,'collision_events':collisions,'progress_m':progress,'required_progress_m':r['required_progress_m'],'wall_elapsed_s':time.monotonic()-begun,'wall_budget_s':args.max_wall,'training_inclusion':'diagnostic_only; not automatically imported','obstacles':r['obstacles']};reports.append(report);(dest/'manifest.json').write_text(json.dumps(reports,indent=2));print(report,flush=True)
n.destroy_node()
if rclpy.ok():rclpy.shutdown()
