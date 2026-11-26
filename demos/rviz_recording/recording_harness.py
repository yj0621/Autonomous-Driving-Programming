#!/usr/bin/env python3
"""Actual ROS test routing, obstacle injection, and measured telemetry for RViz recording.

No controller or planner outputs are synthesized. Ground-truth obstacle observations
are injected to isolate planner behavior from perception errors.
"""
import json, math, time
from pathlib import Path
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped, Point
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import Obstacle, ObstacleArray, OTWpntArray

ROOT=Path(__file__).resolve().parent
WP=json.loads((ROOT.parents[1]/'stack_master/maps/hyhall12/global_waypoints.json').read_text())['global_traj_wpnts_iqp']['wpnts']
def obstacle_geometry(spec):
    idx=spec if isinstance(spec,int) else spec['index']
    offset=0. if isinstance(spec,int) else spec.get('d',0.)
    w=WP[idx]
    return w,offset,w['x_m']-offset*math.sin(w['psi_rad']),w['y_m']+offset*math.cos(w['psi_rad'])

class Recorder(Node):
    def __init__(self):
        super().__init__('recording_harness')
        self.mode='stop'; self.case='Standby'; self.obstacles=[]; self.trace=[]
        self.pose=None; self.drive=None; self.speed=0.; self.state=''; self.collisions=[]
        self.spline_count=0; self.last_cmd={}; self.origin=time.monotonic(); self.last_reset=0.; self.last_collision='none'
        self.pub=self.create_publisher(AckermannDriveStamped,'/drive',10)
        self.resetpub=self.create_publisher(PoseWithCovarianceStamped,'/initialpose',10)
        self.obspub=self.create_publisher(PoseArray,'/sim/obstacles',10)
        self.perpub=self.create_publisher(ObstacleArray,'/perception/obstacles',10)
        self.markerpub=self.create_publisher(MarkerArray,'/recording/markers',10)
        self.create_subscription(AckermannDriveStamped,'/pp_drive',lambda m:self.route('pp',m),10)
        self.create_subscription(AckermannDriveStamped,'/tln_drive',lambda m:self.route('tln',m),10)
        self.create_subscription(Odometry,'/car_state/odom_GT',self.odom,10)
        self.create_subscription(String,'/sim/collision',self.collision,10)
        self.create_subscription(String,'/state',lambda m:setattr(self,'state',m.data),10)
        self.create_subscription(OTWpntArray,'/planner/avoidance/otwpnts',lambda m:setattr(self,'spline_count',len(m.wpnts)),10)
        self.log=(ROOT/'telemetry.jsonl').open('a',buffering=1)
        self.create_timer(.05,self.tick)

    def route(self,mode,msg):
        if self.mode==mode and time.monotonic()-self.last_reset>.3:
            self.pub.publish(msg); self.drive=msg.drive

    def odom(self,m):
        p=m.pose.pose.position; self.pose=(p.x,p.y)
        self.speed=m.twist.twist.linear.x
        if self.mode!='stop' and time.monotonic()-self.last_reset>.35:
            self.trace.append(self.pose); self.trace=self.trace[-1500:]

    def collision(self,m):
        # The simulator publishes actual collision events, not inferred annotations.
        if time.monotonic()-self.last_reset>.35 and m.data and m.data not in ['none','False','false',''] and m.data != self.last_collision:
            event={'wall_time':time.time(),'case':self.case,'kind':m.data,'pose':self.pose}
            self.collisions.append(event); self.log.write(json.dumps({'collision_event':event})+'\n')
        self.last_collision=m.data

    def tick(self):
        try: cmd=json.loads((ROOT/'command.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError): cmd=self.last_cmd
        if cmd!=self.last_cmd:
            self.last_cmd=cmd; self.mode=cmd.get('mode','stop'); self.case=cmd.get('case','Standby')
            self.obstacles=cmd.get('obstacles',[])
            if 'reset' in cmd:
                self.pub.publish(AckermannDriveStamped())
                p=WP[cmd['reset']]; m=PoseWithCovarianceStamped(); m.header.frame_id='map'
                yaw=p['psi_rad']+cmd.get('yaw_offset',0.)
                m.pose.pose.position.x=p['x_m'];m.pose.pose.position.y=p['y_m']
                m.pose.pose.orientation.z=math.sin(yaw/2);m.pose.pose.orientation.w=math.cos(yaw/2)
                self.resetpub.publish(m); self.trace=[];self.collisions=[];self.drive=None
                self.origin=time.monotonic();self.last_reset=self.origin;self.last_collision='none'
            self.log.write(json.dumps({'command':cmd,'wall_time':time.time()})+'\n')
        if self.mode=='stop' or time.monotonic()-self.last_reset<.3:
            self.pub.publish(AckermannDriveStamped())
        pa=PoseArray();pa.header.frame_id='map';pa.header.stamp=self.get_clock().now().to_msg()
        oa=ObstacleArray();oa.header=pa.header
        for k,spec in enumerate(self.obstacles):
            w,offset,ox,oy=obstacle_geometry(spec);p=Pose();p.position.x=ox;p.position.y=oy;p.orientation.w=1.;pa.poses.append(p)
            o=Obstacle();o.id=k;o.s_center=w['s_m'];o.s_start=o.s_center-.15;o.s_end=o.s_center+.15
            o.d_center=offset;o.d_left=offset+.15;o.d_right=offset-.15;o.size=.3;o.is_static=True;o.is_visible=True;oa.obstacles.append(o)
        self.obspub.publish(pa);self.perpub.publish(oa)
        markers=MarkerArray()
        def marker(idx,kind):
            m=Marker();m.header=pa.header;m.ns='recording';m.id=idx;m.type=kind;m.pose.orientation.w=1.;m.color.a=1.;return m
        trail=marker(0,Marker.LINE_STRIP);trail.scale.x=.07;trail.color.g=1.;trail.color.b=.95
        trail.points=[Point(x=x,y=y,z=.13) for x,y in self.trace[::2]];markers.markers.append(trail)
        if self.pose:
            ring=marker(1,Marker.CYLINDER);ring.pose.position.x,ring.pose.position.y=self.pose;ring.pose.position.z=.12
            ring.scale.x=.60;ring.scale.y=.60;ring.scale.z=.04;ring.color.g=1.;ring.color.b=.8;ring.color.a=.6;markers.markers.append(ring)
        for k in range(5):
            cube=marker(10+k,Marker.CUBE)
            if k<len(self.obstacles):
                w,offset,ox,oy=obstacle_geometry(self.obstacles[k]);cube.pose.position.x=ox;cube.pose.position.y=oy;cube.pose.position.z=.2
                cube.scale.x=.3;cube.scale.y=.3;cube.scale.z=.4;cube.color.r=1.;cube.color.g=.15
            else: cube.action=Marker.DELETE
            markers.markers.append(cube)
        speed=self.drive.speed if self.drive else 0.
        steer=self.drive.steering_angle if self.drive else 0.
        text=marker(2,Marker.TEXT_VIEW_FACING);text.pose.position.x=3.0;text.pose.position.y=8.0;text.pose.position.z=.2
        text.scale.z=.34;text.color.r=1.;text.color.g=1.;text.color.b=1.
        text.text=f'{self.case} | v_cmd {speed:.2f} m/s | steer {math.degrees(steer):+.1f} deg'
        markers.markers.append(text)
        text2=marker(3,Marker.TEXT_VIEW_FACING);text2.pose.position.x=3.;text2.pose.position.y=7.5;text2.pose.position.z=.2
        text2.scale.z=.27;text2.color.r=1.;text2.color.g=.6;text2.color.b=.3
        text2.text=f'spline: {self.spline_count} points | {self.state} | collision events: {len(self.collisions)}'
        markers.markers.append(text2)
        self.markerpub.publish(markers)
        self.log.write(json.dumps({'wall_time':time.time(),'elapsed':time.monotonic()-self.origin,'case':self.case,'mode':self.mode,'xy':self.pose,'speed':self.speed,'v_cmd':speed,'steer':steer,'state':self.state,'spline_points':self.spline_count,'collision_count':len(self.collisions)})+'\n')

if __name__=='__main__':
    rclpy.init();n=Recorder()
    try:rclpy.spin(n)
    finally:n.log.close();n.destroy_node();rclpy.shutdown()
