"""Audit a closed fresh high-speed ROS bag in record order, not mutable wall time."""
from pathlib import Path
import sys,sqlite3,json,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from world import World
root=Path(__file__).parent;bag=root/'ros_candidate_three';db=next(bag.glob('*.db3'));c=sqlite3.connect(str(db));topics={i:(name,get_message(type_)) for i,name,type_ in c.execute('SELECT id,name,type FROM topics')};w=World();speed=[];straight=[];corners=[];commands=[];resets=0;collisions=[];previous=None;lastxy=None;progress=0.;laps=[];physics=[];count=0
for index,topic_id,blob in c.execute('SELECT id,topic_id,data FROM messages ORDER BY id'):
 topic,typ=topics[topic_id]
 if topic not in ['/car_state/odom_GT','/drive','/sim/collision','/initialpose','/sim/physics_state']:continue
 m=deserialize_message(blob,typ)
 if topic=='/initialpose':resets+=1;previous=None;progress=0.;lastxy=None
 elif topic=='/sim/physics_state':physics.append(m.data[0])
 elif topic=='/sim/collision' and m.data!='none':collisions.append({'index':index,'kind':m.data})
 elif topic=='/drive':commands.append([m.drive.steering_angle,m.drive.speed])
 elif topic=='/car_state/odom_GT':
  xy=np.array([m.pose.pose.position.x,m.pose.pose.position.y]);v=m.twist.twist.linear.x
  if lastxy is not None and np.linalg.norm(xy-lastxy)>1.5:previous=None;progress=0.
  s=w.project(*xy,hint=previous);speed.append(v);d1=w.path(s,1);d2=w.path(s,2);curvature=abs(d1[0]*d2[1]-d1[1]*d2[0])/max(np.linalg.norm(d1)**3,.01)
  if curvature<.08:straight.append(v)
  if curvature>.55:corners.append(v)
  if previous is not None:progress+=(s-previous+w.length/2)%w.length-w.length/2
  previous=s;lastxy=xy
  if progress>=w.length:laps.append({'progress_m':progress});progress=0.
counts={name:c.execute('SELECT COUNT(*) FROM messages WHERE topic_id=?',(i,)).fetchone()[0] for i,(name,typ) in topics.items()};report={'bag':str(bag),'topic_message_counts':counts,'collision_messages':len(collisions),'collision_kinds':sorted({x['kind'] for x in collisions}),'reset_messages':resets,'full_path_traversals_without_teleport':len(laps),'mean_measured_speed_mps':float(np.mean(speed)),'max_measured_speed_mps':float(max(speed)),'straight_curvature_below_0p08_mean_mps':float(np.mean(straight)),'tight_corner_curvature_above_0p55_mean_mps':float(np.mean(corners)),'physics_time_span_s':float(max(physics)-min(physics)) if physics else None,'command_steer_delta_p95_rad':float(np.percentile(abs(np.diff(np.array(commands)[:,0])),95))};(root/'results/candidate_three_ros_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
