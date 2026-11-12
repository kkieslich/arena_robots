"""A1 bringup: CHAMP quadruped locomotion sidecar (cmd_vel -> leg joints).

Included by `RobotManager._launch_robot` once per robot spawn, inside the
already-pushed robot namespace, gated on this file's presence. It hosts the
open-source gait controller that
consumes the arena `cmd_vel` (Twist) and streams joint trajectories to the
`joint_group_position_controller` brought up from `control.yaml`. Sim-agnostic:
the same nodes drive the joints whether the controller_manager is backed by
Gazebo's gz_ros2_control or Isaac's topic_bridge.
"""

import os

from ament_index_python.packages import get_package_share_directory
from arena_rclpy_mixins.launch_argument import LaunchArgument
from launch import LaunchDescription
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_share = get_package_share_directory('arena_robots')
    a1_dir = os.path.join(robot_share, 'robots', 'a1')

    description_path = os.path.join(a1_dir, 'urdf', 'a1.urdf.xacro')
    joints_map = os.path.join(a1_dir, 'champ', 'joints.yaml')
    links_map = os.path.join(a1_dir, 'champ', 'links.yaml')
    gait_config = os.path.join(a1_dir, 'champ', 'gait.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_namespace = LaunchArgument('namespace', default_value='', description='Robot namespace.')
    declare_base_frame = LaunchArgument('base_frame', default_value='base', description='Base link frame.')
    declare_use_sim_time = LaunchArgument('use_sim_time', default_value='true', description='Use the sim clock.')

    urdf = {'urdf': Command(['xacro ', description_path])}

    # The gait engine: cmd_vel -> foot trajectories -> joint angles. arena owns
    # joint_states (joint_state_broadcaster) and odom TF (pose_to_tf from the sim
    # pose), so this only publishes joint control; foot contacts and odom TF are
    # left off to avoid double-publishing.
    quadruped_controller = Node(
        package='champ_base',
        executable='quadruped_controller_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'gazebo': True},
            {'publish_joint_states': False},
            {'publish_joint_control': True},
            {'publish_foot_contacts': False},
            {'publish_odom_tf': False},
            {'joint_controller_topic': 'joint_group_position_controller/joint_trajectory'},
            urdf,
            joints_map,
            links_map,
            gait_config,
        ],
        remappings=[('cmd_vel/smooth', 'cmd_vel')],
    )

    # Body-velocity / odometry estimate from the gait state. Publishes `odom` for
    # nav; TF fusion (robot_localization EKF) is intentionally deferred to tuning.
    state_estimator = Node(
        package='champ_base',
        executable='state_estimation_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'orientation_from_imu': False},
            urdf,
            joints_map,
            links_map,
            gait_config,
        ],
    )

    return LaunchDescription(
        [
            declare_namespace,
            declare_base_frame,
            declare_use_sim_time,
            quadruped_controller,
            state_estimator,
        ]
    )
