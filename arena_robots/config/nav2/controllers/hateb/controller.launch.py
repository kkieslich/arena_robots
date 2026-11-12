"""Auxiliary nodes for the HATEB local planner (the CoHAN controller component)."""

import os

from arena_rclpy_mixins.launch_argument import LaunchArgument
from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ld_items: list = []
    LaunchArgument.auto_append(ld_items)

    namespace = LaunchArgument('namespace', description='ROS namespace of the robot.')
    env_namespace = LaunchArgument('env_namespace', default_value='', description='ROS namespace of the env the robot belongs to.')
    use_sim_time = LaunchArgument('use_sim_time', description='Whether nodes follow the simulator clock.')

    def _setup(context, *args, **kwargs):
        ns = namespace.substitution.perform(context)
        env_ns = env_namespace.substitution.perform(context)
        sim_time = use_sim_time.substitution.perform(context)

        target_frame = os.path.basename(env_ns.rstrip('/')) + '/map'
        arena_peds_abs = env_ns.rstrip('/') + '/arena_peds'

        bridge = Node(
            package='arena_planners',
            executable='cohan_peds_bridge',
            name='cohan_peds_bridge',
            namespace=ns,
            output='screen',
            parameters=[
                {'use_sim_time': sim_time.lower() in ('true', '1', 'yes')},
                {'target_frame': target_frame},
            ],
            remappings=[('arena_peds', arena_peds_abs)],
        )

        return [bridge]

    return LaunchDescription([*ld_items, OpaqueFunction(function=_setup)])
