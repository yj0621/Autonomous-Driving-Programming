"""Import new clean ROS commands, using physics time and record order for alignment."""
from pathlib import Path
import sys,json,sqlite3,math
root=Path(__file__).parent;sys.path.insert(0,str(root.parent));sys.path.insert(0,str(root))
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from world import World
from sensors import Inputs,nominal_steer,SPEED_SCALE
w=World();audit=json.loads((root/'results/basic_ros_audit.json').read_text());assert audit['collision_messages']==0 and audit['reset_messages']==0
bag=root/'ros_basic_fast_3p0';c=sqlite3.connect(str(next(bag.glob('*.db3'))));topics={i:(n,get_message(t)) for i,n,t in c.execute('SELECT id,name,type FROM topics')};physical=0.;steer=0.;state=None;scan=None;odom_time=None;scan_time=None;last=np.zeros(2);last_sample=-1.;previous=None;progress=0.;inputs=Inputs(w);records=[];states=[];raw=[];hs=[];fs=[];ys=[];commands=[];start_s=None
for i,topic_id,blob in c.execute('SELECT id,topic_id,data FROM messages ORDER BY id'):
 topic,typ=topics[topic_id]
 if topic not in ['/sim/physics_state','/car_state/odom_GT','/scan','/drive','/sim/obstacles']:continue
 m=deserialize_message(blob,typ)
 if topic=='/sim/physics_state':physical=float(m.data[0]);steer=float(m.data[1])
 elif topic=='/sim/obstacles':assert len(m.poses)==0,'Unexpected obstacles in basic recording'
 elif topic=='/scan':scan=np.asarray(m.ranges,np.float32);scan_time=physical
 elif topic=='/car_state/odom_GT':
  q=m.pose.pose.orientation;yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));state=np.array([m.pose.pose.position.x,m.pose.pose.position.y,steer,m.twist.twist.linear.x,yaw,m.twist.twist.angular.z,0.]);odom_time=physical
 elif topic=='/drive':
  u=np.array([m.drive.steering_angle,m.drive.speed])
  if state is None or scan is None or physical-last_sample<.085 or physical-odom_time>.03 or physical-scan_time>.03:last=u;continue
  s=w.project(*state[:2],hint=previous)
  if start_s is None:start_s=s
  if previous is not None:progress+=(s-previous+w.length/2)%w.length-w.length/2
  previous=s;h,f=inputs.push(scan,state,last,s);states.append(state.copy());raw.append(scan.copy());hs.append(h);fs.append(f);ys.append([(u[0]-nominal_steer(f))/.72,u[1]/SPEED_SCALE]);commands.append(u.copy());last=u;last_sample=physical
  if progress>=w.length:
   seed=71500+len(records);stem=root/'data'/f'train_basic_ros_{seed}';np.savez_compressed(stem.with_suffix('.npz'),state=np.array(states,np.float32),raw_scan=np.array(raw,np.float32),scan=np.array(hs,np.float16),features=np.array(fs,np.float32),target=np.array(ys,np.float32),command=np.array(commands,np.float32));report={'seed':seed,'split':'train','mode':'basic_fast_ros_new','success':True,'collision':None,'source_bag':str(bag),'samples':len(ys),'start_s':start_s,'progress_m':progress,'obstacles':[],'mean_measured_speed_mps':float(np.mean(np.array(states)[:,3])),'max_measured_speed_mps':float(np.max(np.array(states)[:,3])),'alignment':'record order; command uses latest scan/state no older than 0.03 physics s; 10 physics Hz; no future samples','source':'new recorded ROS data only; no earlier recording reused'};stem.with_suffix('.json').write_text(json.dumps(report,indent=2));records.append(report);states=[];raw=[];hs=[];fs=[];ys=[];commands=[];progress=0.;start_s=None;inputs=Inputs(w)
assert len(records)>=2,records
(root/'results/basic_ros_import.json').write_text(json.dumps({'episodes':records,'partial_final_segment_excluded':True},indent=2));print('NEW ROS EPISODES',[(x['seed'],x['samples']) for x in records])
