"""Auxiliary node for the rosnav_rl local planner (the action server behind the DRLController get_command service)."""

from arena_rclpy_mixins.launch_argument import LaunchArgument
from launch import LaunchContext, LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ld_items: list = []
    LaunchArgument.auto_append(ld_items)

    namespace = LaunchArgument('namespace', description='ROS namespace of the robot.')
    frame = LaunchArgument('frame', description='Frame prefix of the robot.')
    base_frame = LaunchArgument('base_frame', description='Base link frame of the robot.')
    use_sim_time = LaunchArgument('use_sim_time', description='Whether nodes follow the simulator clock.')
    agent = LaunchArgument('agent', description='Name of the DRL agent directory the controller loads weights from.')

    def _setup(context: LaunchContext, *args: object, **kwargs: object) -> list[Node]:
        ns = namespace.substitution.perform(context)
        sim_time = use_sim_time.substitution.perform(context)

        action_server = Node(
            package='rosnav_rl',
            executable='action_server.py',
            name='rosnav_action_server',
            namespace=ns,
            output='screen',
            parameters=[
                {'use_sim_time': sim_time.lower() in ('true', '1', 'yes')},
                {'agent_name': agent.substitution.perform(context)},
                {'namespace': ns},
                {'frame': frame.substitution.perform(context)},
                {'base_frame': base_frame.substitution.perform(context)},
            ],
        )

        return [action_server]

    return LaunchDescription([*ld_items, OpaqueFunction(function=_setup)])
