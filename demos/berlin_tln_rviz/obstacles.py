#!/usr/bin/env python3
"""Publish five periodically randomized obstacles on the Berlin track."""
import json
import math
import random
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

HERE = Path(__file__).resolve().parent
WP_PATH = HERE.parents[1] / 'stack_master/maps/berlin/global_waypoints.json'
WAYPOINTS = json.loads(WP_PATH.read_text())['centerline_waypoints']['wpnts']
TRACK_LENGTH = float(WAYPOINTS[-1]['s_m'])


class BerlinObstacles(Node):
    def __init__(self):
        super().__init__('berlin_demo_obstacles')
        self.specs = []
        self.rng = random.Random()
        self.last_reposition = 0.0
        self.reposition_period = 12.0
        self.obstacle_count = 5
        self.create_subscription(Odometry, '/car_state/odom_GT', self.odom, 10)
        self.car_x = self.car_y = None
        transient = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pose_pub = self.create_publisher(PoseArray, '/sim/obstacles', transient)
        self.marker_pub = self.create_publisher(MarkerArray, '/sim/obstacles/markers', transient)
        self.speed = 0.0
        self.collisions = []
        self.started = time.time()
        self.log = (HERE / 'obstacle_demo_telemetry.jsonl').open('a', buffering=1)
        self.reposition()
        self.create_subscription(String, '/sim/collision', self.collision, 10)
        self.create_subscription(String, '/tln/control_debug', self.control, 10)
        self.create_timer(.25, self.publish)
        self.get_logger().info(f"Berlin randomized obstacle scene: {self.obstacle_count} obstacles, reposition every {self.reposition_period:.0f}s")

    def odom(self, msg):
        self.speed = float(msg.twist.twist.linear.x)
        self.car_x = float(msg.pose.pose.position.x)
        self.car_y = float(msg.pose.pose.position.y)

    def reposition(self):
        # Spread obstacles around the lap, with randomized order and lateral offset.
        # Keep a generous gap from the car at each change to avoid teleporting onto it.
        anchor = self.rng.uniform(0.0, TRACK_LENGTH)
        slots = [((anchor + i * TRACK_LENGTH / self.obstacle_count + self.rng.uniform(-1.0, 1.0)) % TRACK_LENGTH)
                 for i in range(self.obstacle_count)]
        points = []
        for s in slots:
            wp = min(WAYPOINTS, key=lambda p: abs(((float(p['s_m']) - s + TRACK_LENGTH / 2) % TRACK_LENGTH) - TRACK_LENGTH / 2))
            psi = float(wp['psi_rad'])
            d = self.rng.uniform(-0.38, 0.38)
            x = float(wp['x_m']) - math.sin(psi) * d
            y = float(wp['y_m']) + math.cos(psi) * d
            if self.car_x is not None and math.hypot(x - self.car_x, y - self.car_y) < 6.0:
                # Move the obstacle farther around the lap; repeat until safely clear.
                for shift in (55, 90, 125, 160):
                    candidate = WAYPOINTS[(int(wp['id']) + shift) % len(WAYPOINTS)]
                    cpsi = float(candidate['psi_rad'])
                    cx = float(candidate['x_m']) - math.sin(cpsi) * d
                    cy = float(candidate['y_m']) + math.cos(cpsi) * d
                    if math.hypot(cx - self.car_x, cy - self.car_y) >= 6.0:
                        wp, psi, x, y = candidate, cpsi, cx, cy
                        break
            points.append({'x': x, 'y': y, 'theta': psi, 's': float(wp['s_m']), 'd': d})
        self.specs = points
        self.last_reposition = time.time()
        self.log.write(json.dumps({'t': self.last_reposition, 'event': 'obstacles_repositioned',
                                   'count': len(points), 'positions': points}) + '\n')

    def collision(self, msg):
        if msg.data in ('wall', 'obstacle') and (not self.collisions or self.collisions[-1]['kind'] != msg.data):
            row = {'t': time.time(), 'kind': msg.data, 'speed_mps': self.speed}
            self.collisions.append(row)
            self.log.write(json.dumps({'collision': row}) + '\n')

    def control(self, msg):
        try:
            row = json.loads(msg.data)
            row['t'] = time.time()
            self.log.write(json.dumps({'control': row}) + '\n')
        except (ValueError, TypeError):
            pass

    def publish(self):
        if time.time() - self.last_reposition >= self.reposition_period:
            self.reposition()
        stamp = self.get_clock().now().to_msg()
        poses = PoseArray()
        poses.header.frame_id = 'map'
        poses.header.stamp = stamp
        markers = MarkerArray()
        for i, spec in enumerate(self.specs):
            theta = float(spec['theta'])
            pose = Pose()
            pose.position.x = float(spec['x'])
            pose.position.y = float(spec['y'])
            pose.orientation.z = math.sin(theta / 2)
            pose.orientation.w = math.cos(theta / 2)
            poses.poses.append(pose)

            marker = Marker()
            marker.header = poses.header
            marker.ns = 'berlin_static_obstacles'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = pose
            marker.pose.position.z = .20
            marker.scale.x = marker.scale.y = .30
            marker.scale.z = .40
            marker.color.r = 1.0
            marker.color.g = .12
            marker.color.b = .04
            marker.color.a = 1.0
            markers.markers.append(marker)
        self.pose_pub.publish(poses)
        self.marker_pub.publish(markers)
        self.log.write(json.dumps({'t': time.time(), 'elapsed_s': time.time()-self.started,
                                   'speed_mps': self.speed, 'collision_events': self.collisions}) + '\n')


def main():
    rclpy.init()
    node = BerlinObstacles()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
