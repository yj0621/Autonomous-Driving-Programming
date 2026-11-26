#!/usr/bin/env python3
"""Walks the pipeline boundary by boundary and says where it breaks.

Detection depends on a chain, and any one broken link makes it look like
"nothing is detected". This checks each link in the order the data flows:

    /scan -> obstacles actually in the beams
          -> TF map <- laser
          -> /global_waypoints (the detector will not even start without it)
          -> /car_state/frenet/odom (car_s; a lost filter rejects every point)
          -> /perception/detection/raw_obstacles
          -> /perception/obstacles

    ros2 run race_manager diagnose
"""

import math
import sys
import time

import numpy as np
import rclpy
import tf2_ros
from f110_msgs.msg import ObstacleArray, WpntArray
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST)
OK, BAD, WARN = '  OK  ', ' FAIL ', ' WARN '


class Diagnose(Node):
    def __init__(self):
        super().__init__('diagnose')
        self.seen = {}
        self.counts = {}

        def track(name, msg_type, topic, qos=10):
            def cb(m, n=name):
                self.seen[n] = m
                self.counts[n] = self.counts.get(n, 0) + 1
            self.create_subscription(msg_type, topic, cb, qos)

        track('scan', LaserScan, '/scan')
        track('gt', Odometry, '/car_state/odom_GT')
        track('pf', PoseStamped, '/car_state/pose')
        track('frenet', Odometry, '/car_state/frenet/odom')
        track('wpnts', WpntArray, '/global_waypoints')
        track('raw', ObstacleArray, '/perception/detection/raw_obstacles')
        track('obs', ObstacleArray, '/perception/obstacles')
        track('bench', PoseArray, '/sim/obstacles', LATCHED)
        track('collision', String, '/sim/collision')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def gather(self, seconds=6.0):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)


def line(status, label, detail=''):
    print(f'[{status}] {label:<42} {detail}')


def main():
    rclpy.init()
    node = Diagnose()
    print('\nlistening for 6 s ...\n')
    node.gather()

    names = [n.split('/')[-1] for n in node.get_node_names()]
    seen, counts = node.seen, node.counts

    print('=== nodes ===')
    for want, why in (('bridge', 'simulator'), ('detect', 'perception (detection)'),
                      ('tracking', 'perception (tracking)'), ('scorer', 'race manager'),
                      ('obstacle_spawner', 'race manager'), ('particle_filter', 'localisation')):
        n = names.count(want)
        if n == 0:
            line(BAD, f'{want}', f'not running — {why}')
        elif n > 1:
            line(WARN, f'{want}', f'{n} copies running! kill the stale ones')
        else:
            line(OK, f'{want}', why)

    print('\n=== topics ===')
    for key, topic in (('scan', '/scan'), ('gt', '/car_state/odom_GT'),
                       ('frenet', '/car_state/frenet/odom'), ('wpnts', '/global_waypoints'),
                       ('bench', '/sim/obstacles'),
                       ('raw', '/perception/detection/raw_obstacles'),
                       ('obs', '/perception/obstacles')):
        c = counts.get(key, 0)
        line(OK if c else BAD, topic, f'{c} msgs' if c else 'silent')

    print('\n=== is the obstacle actually in the beams? ===')
    scan, gt, bench = seen.get('scan'), seen.get('gt'), seen.get('bench')
    if not (scan and gt and bench):
        line(BAD, 'cannot check', 'need /scan, /car_state/odom_GT and /sim/obstacles')
    elif not bench.poses:
        line(WARN, 'no obstacles placed', 'obstacle_spawner published an empty set')
    else:
        p, q = gt.pose.pose.position, gt.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        lx, ly = p.x + 0.27 * math.cos(yaw), p.y + 0.27 * math.sin(yaw)
        ranges = np.array(scan.ranges)
        for i, pose in enumerate(bench.poses):
            dx, dy = pose.position.x - lx, pose.position.y - ly
            dist = math.hypot(dx, dy)
            bearing = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(bearing) > scan.angle_max:
                line(WARN, f'obstacle {i+1}', f'{dist:.1f} m, behind the lidar')
                continue
            idx = int(round((bearing - scan.angle_min) / scan.angle_increment))
            idx = max(0, min(len(ranges) - 1, idx))
            measured = ranges[idx]
            hit = measured < dist + 0.25
            line(OK if hit else BAD, f'obstacle {i+1}',
                 f'{dist:.1f} m away, beam {idx} reads {measured:.2f} m '
                 f'-> {"in the scan" if hit else "NOT in the scan"}')

    print('\n=== TF the detector needs ===')
    frame = scan.header.frame_id if scan else 'car_state/laser'
    try:
        node.tf_buffer.lookup_transform('map', frame, rclpy.time.Time())
        line(OK, f'map <- {frame}', 'available')
    except Exception as e:
        line(BAD, f'map <- {frame}', str(e)[:70])

    print('\n=== localisation ===')
    pf = seen.get('pf')
    if not counts.get('gt'):
        # Without ground truth there is nothing to compare against. Reporting an
        # error here would be meaningless -- and worse, a stale filter estimate
        # left over from an earlier run reads as a huge bogus error.
        line(WARN, 'cannot judge localisation',
             '/car_state/odom_GT is silent — the simulator is not running')
    elif not pf:
        line(BAD, '/car_state/pose', 'silent — particle filter not publishing')
    else:
        g = gt.pose.pose.position
        err = math.hypot(pf.pose.position.x - g.x, pf.pose.position.y - g.y)
        line(OK if err < 0.5 else BAD, 'estimated vs true position',
             f'PF ({pf.pose.position.x:.2f}, {pf.pose.position.y:.2f})  '
             f'GT ({g.x:.2f}, {g.y:.2f})  off by {err:.2f} m')
        if err >= 0.5:
            print('         the filter is not tracking this run. Most often it was never '
                  'seeded:\n'
                  '         set an initial pose (RViz "2D Pose Estimate") after the '
                  'simulator starts.')

    fr = seen.get('frenet')
    if not fr:
        line(BAD, '/car_state/frenet/odom', 'silent — detect rejects every point without car_s')
    else:
        line(OK, '/car_state/frenet/odom',
             f's={fr.pose.pose.position.x:.2f} m, d={fr.pose.pose.position.y:.2f} m')

    print('\n=== detection output ===')
    raw, obs = seen.get('raw'), seen.get('obs')
    line(OK if raw and raw.obstacles else BAD, '/perception/detection/raw_obstacles',
         f'{len(raw.obstacles)} obstacles' if raw else 'never published')
    line(OK if obs and obs.obstacles else WARN, '/perception/obstacles',
         f'{len(obs.obstacles)} obstacles' if obs else 'never published')

    print('\n=== collision channel ===')
    col = seen.get('collision')
    if col is None:
        line(BAD, '/sim/collision', 'silent — is the patched bridge running?')
    else:
        line(OK, '/sim/collision', f'"{col.data}"'
             + ('  (car is sitting in a collision and stays there until it is reset '
                'by hand; no further collision is counted)' if col.data != 'none' else ''))

    print()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
