"""Convert genuinely recorded ROS basic driving into aligned NEW training episodes."""
import argparse,json,math,bisect
from pathlib import Path
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from world import World,History
p=argparse.ArgumentParser();p.add_argument('bag');p.add_argument('--prefix',type=int,default=1100);a=p.parse_args();root=Path(__file__).parent;out=root/'data/ros_fast';out.mkdir(exist_ok=True);w=World()
reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=a.bag,storage_id='sqlite3'),rosbag2_py.ConverterOptions('',''));types={t.name:get_message(t.type) for t in reader.get_all_topics_and_types()};scans=[];odoms=[];commands=[];resets=[];collisions=[];last_scan=-1
while reader.has_next():
 topic,blob,t=reader.read_next();t=t/1e9;m=deserialize_message(blob,types[topic])
 if topic=='/scan' and t-last_scan>=.095:scans.append((t,np.asarray(m.ranges,np.float32)));last_scan=t
 elif topic=='/car_state/odom_GT':
  q=m.pose.pose.orientation;yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));odoms.append((t,np.array([m.pose.pose.position.x,m.pose.pose.position.y,0,m.twist.twist.linear.x,yaw],float)))
 elif topic=='/drive':commands.append((t,np.array([m.drive.steering_angle,m.drive.speed])))
 elif topic=='/initialpose':resets.append(t)
 elif topic=='/sim/collision' and m.data!='none':collisions.append(t)
if not scans or not odoms or not commands:raise RuntimeError('Missing scan/odom/drive recording')
ot=[v[0] for v in odoms];ct=[v[0] for v in commands];hist=History();episodes=[];hs=[];fs=[];ys=[];states=[];raw=[];prev=None;progress=0;episode=0;last_reset=None;actual_speeds=[]
def save():
 global hs,fs,ys,states,raw,episode
 if len(hs)<40:return
 seed=a.prefix+episode;stem=out/f'train_basic_fast_{seed}'
 np.savez_compressed(stem.with_suffix('.npz'),scan=np.asarray(hs,np.float32),features=np.asarray(fs,np.float32),target=np.asarray(ys,np.float32),state=np.asarray(states,np.float32),raw_scan=np.asarray(raw,np.float32),info=np.zeros((len(hs),2),np.float32),plan_mask=np.ones((len(hs),3),np.float32))
 report={'split':'train','mode':'basic_fast_ros','seed':seed,'success':True,'collision':None,'samples':len(hs),'source_bag':str(Path(a.bag).resolve()),'recording_type':'actual ROS bag, nearest-state alignment <=40ms, command age <=100ms','progress_m':progress,'mean_actual_speed_mps':float(np.mean(np.array(states)[:,3])),'max_actual_speed_mps':float(np.max(np.array(states)[:,3]))};stem.with_suffix('.json').write_text(json.dumps(report,indent=2));episodes.append(report);episode+=1;hs=[];fs=[];ys=[];states=[];raw=[]
for t,scan in scans:
 if any(abs(t-r)<.7 for r in resets) or any(abs(t-c)<.8 for c in collisions):
  # Never label reset/collision transitions as normal driving.
  hs=[];fs=[];ys=[];states=[];raw=[];hist=History();prev=None;progress=0;continue
 j=bisect.bisect_right(ot,t)-1;k=bisect.bisect_right(ct,t)-1
 if j<0 or k<0 or t-ot[j]>.04 or t-ct[k]>.1:continue
 st=odoms[j][1];u=commands[k][1];s=w.project(st[0],st[1],hint=prev)
 if prev is not None:progress+=(s-prev+w.length/2)%w.length-w.length/2
 prev=s
 hs.append(hist.push(scan));fs.append(w.route_features(st,previous_steer=u[0],s_hint=s));ys.append([0.,0.,0.,np.clip(u[1]/2.8,0,1)]);states.append(st);raw.append(scan);actual_speeds.append(st[3])
 if progress>w.length-.5:save();progress=0;hist=History()
# Partial final fragment is intentionally excluded from success-labelled training.
summary={'source_bag':a.bag,'raw_scan_samples_10hz':len(scans),'odometry_messages':len(odoms),'command_messages':len(commands),'collision_messages':len(collisions),'reset_messages':len(resets),'full_lap_episodes':episodes,'used_for_training_only':True};(out/'import_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
