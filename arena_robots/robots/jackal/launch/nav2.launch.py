# Copyright 2022 Clearpath Robotics, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from arena_rclpy_mixins.launch_argument import LaunchArgument
from launch import LaunchDescription
from launch.actions import (GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Get the launch directory
    pkg_nav2_bringup = FindPackageShare('nav2_bringup')
    pkg_jackal_navigation = FindPackageShare('jackal_navigation')

    # Create the launch configuration variables
    namespace = LaunchConfiguration('namespace')
    use_namespace = LaunchConfiguration('use_namespace')
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    use_composition = LaunchConfiguration('use_composition')
    localization = LaunchConfiguration('localization')

    remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'yaml_filename': map_yaml_file}

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True)

    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    declare_namespace_cmd = LaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace.')

    declare_use_namespace_cmd = LaunchArgument(
        'use_namespace',
        default_value='false',
        description='Apply the namespace to the navigation stack.')

    declare_slam_cmd = LaunchArgument(
        'slam',
        default_value='false',
        choices=['true', 'false'],
        description='Run SLAM.')

    declare_localization_cmd = LaunchArgument(
        'localization',
        default_value='false',
        choices=['true', 'false'],
        description='Run localization.')

    declare_nav2_cmd = LaunchArgument(
        'nav2',
        default_value='true',
        choices=['true', 'false'],
        description='Run Nav2.')

    declare_map_yaml_cmd = LaunchArgument(
        'map',
        default_value=PathJoinSubstitution(
            [pkg_jackal_navigation,
             'maps',
             'depot.yaml']),
        description='Map yaml file to load.')

    declare_use_sim_time_cmd = LaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use the sim clock.')

    declare_params_file_cmd = LaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution(
            [pkg_jackal_navigation,
             'config',
             'nav2.yaml']),
        description='Parameters file for all launched nodes.')

    declare_autostart_cmd = LaunchArgument(
        'autostart', default_value='true',
        description='Start the Nav2 stack automatically.')

    declare_use_composition_cmd = LaunchArgument(
        'use_composition', default_value='True',
        description='Use composed bringup.')

    # Specify the actions
    bringup_cmd_group = GroupAction([
        PushRosNamespace(
            condition=IfCondition(use_namespace),
            namespace=namespace),

        Node(
            condition=IfCondition(use_composition),
            name='nav2_container',
            package='rclcpp_components',
            executable='component_container',
            parameters=[configured_params, {'autostart': autostart}],
            remappings=remappings,
            output='screen'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [pkg_nav2_bringup,
                     'launch',
                     'localization_launch.py'])),
            condition=IfCondition(localization),
            launch_arguments={'namespace': namespace,
                              'map': map_yaml_file,
                              'use_sim_time': use_sim_time,
                              'autostart': autostart,
                              'params_file': params_file,
                              'use_composition': use_composition,
                              'container_name': 'nav2_container'}.items()),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [pkg_nav2_bringup,
                     'launch',
                     'navigation_launch.py'])),
            launch_arguments={'namespace': namespace,
                              'use_sim_time': use_sim_time,
                              'autostart': autostart,
                              'params_file': params_file,
                              'use_composition': use_composition,
                              'container_name': 'nav2_container'}.items()),
    ])

    # Create the launch description and populate
    ld = LaunchDescription()

    # Set environment variables
    ld.add_action(stdout_linebuf_envvar)

    # Declare the launch options
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_namespace_cmd)
    ld.add_action(declare_slam_cmd)
    ld.add_action(declare_localization_cmd)
    ld.add_action(declare_nav2_cmd)
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_composition_cmd)

    # Add the actions to launch all of the navigation nodes
    ld.add_action(bringup_cmd_group)

    return ld
