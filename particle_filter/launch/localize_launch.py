# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def resolve_map(map_path, map_name, localize_config):
    """Pick the map yaml the localisation should run against.

    map_path and map_name come down from base_system_launch.xml, so a map switch there
    reaches the particle filter too. Localisation wants the raw SLAM map (<name>_orig)
    where one exists -- the cleaned <name>.png has had features removed that the scan
    match needs. Falls back to the name written in localize.yaml so mapping runs, which
    have no map on disk yet, keep working as before.
    """
    if map_path and map_name:
        for candidate in (map_name + '_orig', map_name):
            path = os.path.join(map_path, candidate + '.yaml')
            if os.path.exists(path):
                return path

    name = yaml.safe_load(open(localize_config))['map_server']['ros__parameters']['map']
    return os.path.join(get_package_share_directory('particle_filter'), 'maps', name + '.yaml')


def setup(context, *args, **kwargs):
    get = lambda key: LaunchConfiguration(key).perform(context)
    map_yaml = resolve_map(get('map_path'), get('map_name'), get('localize_config'))
    print(f'[localize_launch] localising against {map_yaml}')

    return [
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{'use_sim_time': True},
                        {'autostart': True},
                        {'node_names': ['map_server']}]
        ),
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            parameters=[{'yaml_filename': map_yaml},
                        {'topic': 'map'},
                        {'frame_id': 'map'},
                        {'output': 'screen'},
                        {'use_sim_time': True}]
        ),
        Node(
            package='particle_filter',
            executable='particle_filter',
            name='particle_filter',
            parameters=[LaunchConfiguration('localize_config')]
        ),
    ]


def generate_launch_description():
    localize_config = os.path.join(
        get_package_share_directory('particle_filter'),
        'config',
        'localize.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument('localize_config', default_value=localize_config,
                              description='Localization configs'),
        DeclareLaunchArgument('map_path', default_value='',
                              description='Map folder. Empty falls back to localize.yaml'),
        DeclareLaunchArgument('map_name', default_value='',
                              description='Map basename inside map_path'),
        OpaqueFunction(function=setup),
    ])
