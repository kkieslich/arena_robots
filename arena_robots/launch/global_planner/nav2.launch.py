"""nav2 family handler for `mobile.global_planner:=nav2/<kind>`.

Bootstraps a planner_server-only nav2 stack and a goal-pose -> compute_path_to_pose bridge.
"""

from arena_rclpy_mixins.launch_argument import LaunchArgument
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    ld_items = []
    LaunchArgument.auto_append(ld_items)

    robot = LaunchArgument('robot', description='Robot model name.')
    arena_env_node = LaunchArgument('arena_env_node', default_value='', description='Name of the env node whose map and services the robot uses.')
    namespace = LaunchArgument('namespace', description='ROS namespace of the robot.')
    frame = LaunchArgument('frame', description='Frame prefix of the robot.')
    use_sim_time = LaunchArgument('use_sim_time', description='Whether nodes follow the simulator clock.')
    kind = LaunchArgument('kind', description='Nav2 global planner kind.')

    nav2_launch = PathJoinSubstitution([
        FindPackageShare('arena_robots'),
        'launch', 'adapters', 'mobile', 'nav2.launch.py',
    ])

    nav2_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch),
        launch_arguments={
            'robot': robot.substitution,
            'namespace': namespace.substitution,
            'use_sim_time': use_sim_time.substitution,
            'frame': frame.substitution,
            'global_planner': kind.substitution,
            'local_planner': 'dwb',
            'inter_planner': 'empty',
            'train_mode': 'true',
            'planner_only': 'true',
            'arena_env_node': arena_env_node.substitution,
        }.items(),
    )

    bridge_node = Node(
        package='arena_robots',
        executable='goal_to_plan_bridge',
        name='goal_to_plan_bridge',
        namespace=namespace.substitution,
        output='screen',
        parameters=[{'use_sim_time': use_sim_time.substitution}],
    )

    return LaunchDescription([*ld_items, nav2_include, bridge_node])
