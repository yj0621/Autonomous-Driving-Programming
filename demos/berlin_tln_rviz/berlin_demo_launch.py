"""Paced Berlin simulator, RViz, map route, TF and odometry for live TLN demo."""
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import Command


def generate_launch_description():
    root = Path(__file__).resolve().parents[2]
    map_dir = root / 'stack_master' / 'maps' / 'berlin'
    map_yaml = str(map_dir / 'berlin.yaml')
    stack_share = Path(get_package_share_directory('stack_master'))
    gym_share = Path(get_package_share_directory('f1tenth_gym_ros'))
    sim_yaml = stack_share / 'config' / 'SIM' / 'sim.yaml'
    sim_params = stack_share / 'config' / 'SIM' / 'sim_params.yaml'
    xacro = gym_share / 'config' / 'ego_racecar.xacro'

    paced_bridge = ExecuteProcess(
        cmd=[sys.executable, str(root / 'experiments/tln_fresh/timed_bridge.py'),
             '--ros-args', '-r', '__node:=bridge', '--params-file', str(sim_yaml),
             '-p', f'map_path:={map_yaml}', '-p', f'sim_params:={sim_params}',
             '-p', 'synchronous_control:=true', '-p', 'paced_control:=true'],
        output='screen')
    rviz = Node(package='rviz2', executable='rviz2', name='rviz',
                arguments=['-d', str(root / 'demos/berlin_tln_rviz/berlin_rviz.rviz')], output='screen')
    map_server = Node(package='nav2_map_server', executable='map_server', name='map_server',
                      parameters=[{'yaml_filename': map_yaml, 'topic': 'map',
                                   'frame_id': 'map', 'use_sim_time': True}], output='screen')
    lifecycle = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                     name='lifecycle_manager_localization',
                     parameters=[{'use_sim_time': True, 'autostart': True,
                                  'node_names': ['map_server']}], output='screen')
    robot = Node(package='robot_state_publisher', executable='robot_state_publisher',
                 name='ego_robot_state_publisher',
                 parameters=[{'robot_description': Command(['xacro ', str(xacro)])}],
                 remappings=[('/robot_description', 'ego_robot_description')], output='screen')
    frenet = Node(package='frenet_odom_republisher',
                  executable='frenet_odom_republisher_node',
                  name='frenet_odom_republisher',
                  parameters=[{'use_sim_time': True, 'pose_topic': '/car_state/pose_GT'}],
                  output='screen')
    trajectory = Node(package='global_planner', executable='global_trajectory_publisher',
                      name='global_republisher', parameters=[{'map_path': str(map_dir)}],
                      output='screen')
    speed = Node(package='speed_scaler', executable='speed_scaler', name='speed_scaler',
                 parameters=[str(map_dir / 'speed_scaling.yaml')], output='screen')
    return LaunchDescription([paced_bridge, rviz, map_server, lifecycle, robot,
                              frenet, trajectory, speed])
