"""Audit closed real ROS MPCC recording; header stamps deduplicate paused frames."""
from pathlib import Path
import sys,sqlite3,json
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent))
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from world import World
bag=root/'ros_teacher_synced';db=sqlite3.connect(next(bag.glob('*.db3')));topics={i:(name,get_message(typ)) for i,name,typ in db.execute('select id,name,type from topics')};counts={name:db.execute('select count(*) from messages where topic_id=?',(i,)).fetchone()[0] for i,(name,typ) in topics.items()};w=World();seen=set();speed=[];progress=0.;previous=None;lastxy=None;laps=0;resets=0;collision_messages=0;events=0;kind='none';physics=[];obstacle_counts=[]
for tid,blob in db.execute('select topic_id,data from messages order by id'):
 name,typ=topics[tid]
 if name not in ['/initialpose','/sim/physics_state','/sim/collision','/sim/obstacles','/car_state/odom_GT']:continue
 m=deserialize_message(blob,typ)
 if name=='/initialpose':resets+=1;previous=None;lastxy=None;progress=0.
 elif name=='/sim/physics_state':physics.append(m.data[0])
 elif name=='/sim/obstacles':obstacle_counts.append(len(m.poses))
 elif name=='/sim/collision':
  collision_messages+=m.data!='none';events+=m.data in ['wall','obstacle'] and m.data!=kind;kind=m.data
 elif name=='/car_state/odom_GT':
  stamp=(m.header.stamp.sec,m.header.stamp.nanosec)
  if stamp in seen:continue
  seen.add(stamp);xy=np.array([m.pose.pose.position.x,m.pose.pose.position.y]);speed.append(m.twist.twist.linear.x)
  if lastxy is not None and np.linalg.norm(xy-lastxy)>1.5:previous=None;progress=0.
  s=w.project(*xy,hint=previous)
  if previous is not None:progress+=(s-previous+w.length/2)%w.length-w.length/2
  if progress>=w.length:laps+=1;progress-=w.length
  previous=s;lastxy=xy
report={'bag':str(bag),'topic_counts':counts,'unique_odom_frames':len(seen),'collision_messages':int(collision_messages),'collision_events':int(events),'pose_resets':resets,'full_path_traversals':laps,'physics_duration_s':max(physics)-min(physics),'mean_speed_mps':float(np.mean(speed)),'max_speed_mps':float(max(speed)),'observed_obstacle_counts':sorted(set(obstacle_counts)),'role':'MPCC teacher runtime validation; not claimed as final student raw-scan recording or used for new training'};(root/'results/ros_teacher_audit.json').write_text(json.dumps(report,indent=2));print(report)
