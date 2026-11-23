from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import subprocess
R='/home/misys/forza_ws/race_stack/'
def generate_launch_description():
 desc=subprocess.check_output(['xacro',R+'base_system/f110_simulator/f1tenth_gym_ros/config/ego_racecar.xacro']).decode()
 return LaunchDescription([
 ExecuteProcess(cmd=['python3',R+'experiments/tln_fresh/timed_bridge.py','--ros-args','-r','__node:=bridge','--params-file',R+'stack_master/config/SIM/sim.yaml','-p','map_path:='+R+'stack_master/maps/hangar_1905_v0/hangar_1905_v0.yaml','-p','sim_params:='+R+'stack_master/config/SIM/sim_params.yaml','-p','synchronous_control:=true','-p','paced_control:=true'],output='screen'),
 Node(package='nav2_map_server',executable='map_server',name='map_server',parameters=[{'yaml_filename':R+'stack_master/maps/hangar_1905_v0/hangar_1905_v0.yaml','use_sim_time':False}]),
 Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='demo_lifecycle',parameters=[{'autostart':True,'node_names':['map_server']}]),
 Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[{'robot_description':desc}],remappings=[('/robot_description','/ego_robot_description')]),
 ExecuteProcess(cmd=['python3','/tmp/tln_random_three.py'],output='screen'),
 ExecuteProcess(cmd=['python3',R+'experiments/tln_fresh/revision_dynamic_v5/live.py','--model','models_moe96/best_regression.keras'],output='screen'),
 Node(package='rviz2',executable='rviz2',arguments=['-d','/tmp/tln_random_demo.rviz'])])
