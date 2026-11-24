"""rosnav_rl adapter launch: single inference node, no nav2."""

from arena_rclpy_mixins.launch_argument import LaunchArgument
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    ld_items: list = []
    LaunchArgument.auto_append(ld_items)

    LaunchArgument("robot", description="Robot model name.")
    namespace = LaunchArgument("namespace", description="ROS namespace of the robot.")
    use_sim_time = LaunchArgument("use_sim_time", description="Whether nodes follow the simulator clock.")
    frame = LaunchArgument("frame", description="Frame prefix of the robot.")
    base_frame = LaunchArgument("base_frame", description="Base link frame of the robot.")
    arena_env_node = LaunchArgument("arena_env_node", default_value="", description="Name of the env node whose map and services the robot uses.")
    agent = LaunchArgument("agent", description="Name of the DRL agent directory the controller loads weights from.")
    node_name = LaunchArgument("node_name", default_value="rosnav_rl_inference", description="Name of the DRL inference node.")
    control_rate = LaunchArgument("control_rate", default_value="0.0", description="Control loop rate of the inference node in Hz, 0 for the rate the agent was trained at.")
    min_lookahead_dist = LaunchArgument("min_lookahead_dist", default_value="0.5", description="Minimum lookahead distance along the global plan in meters.")
    max_lookahead_dist = LaunchArgument("max_lookahead_dist", default_value="2.5", description="Maximum lookahead distance along the global plan in meters.")
    lookahead_time = LaunchArgument("lookahead_time", default_value="1.5", description="Lookahead time along the global plan in seconds.")
    train_mode = LaunchArgument("train_mode", default_value="false", description="Whether the robot is driven by a DRL agent in training (true or false).")

    inference_node = Node(
        package="rosnav_rl",
        executable="arena_inference_node.py",
        name=node_name.substitution,
        namespace=namespace.substitution,
        output="screen",
        parameters=[
            {
                **agent.str_param,
                **namespace.str_param,
                **arena_env_node.str_param,
                **frame.str_param,
                **base_frame.str_param,
                **use_sim_time.param(bool),
                **control_rate.param(float),
                **min_lookahead_dist.param(float),
                **max_lookahead_dist.param(float),
                **lookahead_time.param(float),
                **train_mode.param(bool),
            }
        ],
    )

    return LaunchDescription([*ld_items, inference_node])
