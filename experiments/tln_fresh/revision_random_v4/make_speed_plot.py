"""Plot actual ROS odometry and drive commands against raceline arc."""
from pathlib import Path
import sys,sqlite3
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from world import World
w=World();bag=next((root/'ros_basic_fast_3p0').glob('*.db3'));db=sqlite3.connect(bag);topics=dict(db.execute('select id,name from topics'));last=None;command=0.;rows=[]
for tid,raw in db.execute('select topic_id,data from messages order by id'):
 name=topics[tid]
 if name=='/drive':command=deserialize_message(raw,AckermannDriveStamped).drive.speed
 if name!='/car_state/odom_GT':continue
 m=deserialize_message(raw,Odometry);s=w.project(m.pose.pose.position.x,m.pose.pose.position.y,hint=last);last=s;rows.append((s,m.twist.twist.linear.x,command))
a=np.array(rows)[::10];fig,ax=plt.subplots(figsize=(11,4));ax.scatter(a[:,0],a[:,1],s=5,alpha=.3,color='#167d8d',label='Measured ROS speed');ax.scatter(a[:,0],a[:,2],s=3,alpha=.18,color='#df8f2d',label='Speed command');ax.set(xlabel='Raceline arc (m)',ylabel='Speed (m/s)',ylim=(0,3.25),title='Fresh ROS recording: fast straights, braking before corners');ax.grid(alpha=.2);ax.legend(loc='lower right');fig.tight_layout();out=root/'deliverables/ros_speed_profile.png';fig.savefig(out,dpi=180);print(out)
