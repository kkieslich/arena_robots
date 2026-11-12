"""Nav2 adapter launch."""

import os

from arena_bringup.future import PythonExpression
from arena_rclpy_mixins.launch_argument import LaunchArgument
from arena_bringup.substitutions import YAMLDropSubstitution, YAMLFileSubstitution, YAMLMergeSubstitution, YAMLReplaceSubstitution, YAMLRetrieveSubstitution, YAMLSelectSubstitution
from arena_robots.nav2 import (
    Nav2CollisionDerivedYAML,
    Nav2KinematicsDerivedYAML,
    Nav2SubBlockYAML,
    SensorsDerivedYAML,
)
from launch import LaunchDescription, SomeSubstitutionsType
from launch.actions import GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


MAPPING_MODES = ('static', 'slam')  # static = the env's map server, slam = built from the lidar while driving


def generate_launch_description():
    robots_root = FindPackageShare('arena_robots')

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    robot = LaunchArgument('robot', description='Robot model name.')
    arena_env_node = LaunchArgument('arena_env_node', default_value='', description='Name of the env node whose map and services the robot uses.')
    env_namespace = LaunchArgument('env_namespace', default_value='', description='ROS namespace of the env the robot belongs to.')
    namespace = LaunchArgument('namespace', description='ROS namespace of the robot.')
    frame = LaunchArgument('frame', description='Frame prefix of the robot.')
    use_sim_time = LaunchArgument('use_sim_time', description='Whether nodes follow the simulator clock.')
    global_planner = LaunchArgument('global_planner', description='Nav2 global planner kind.')
    local_planner = LaunchArgument('local_planner', description='Nav2 controller kind.')
    inter_planner = LaunchArgument('inter_planner', description='Nav2 inter planner kind.')
    train_mode = LaunchArgument('train_mode', default_value='false', description='Whether the robot is driven by a DRL agent in training (true or false).')
    planner_only = LaunchArgument('planner_only', default_value='false', description='Run only the global planner.')
    sensors_json = LaunchArgument('sensors_json', default_value='', description='JSON list of the robot sensor specs used to derive the costmap observation sources.')
    params_overlay = LaunchArgument('params_overlay', default_value='', description='Nav2 params overlay, a path or an overlay name (empty = none).')
    agent = LaunchArgument('agent', default_value='', description='Name of the DRL agent directory the controller loads weights from.')
    mapping = LaunchArgument('mapping', default_value='static', description='Map source: static = the env map server, slam = built from the lidar while driving.')
    scan_topic = LaunchArgument('scan_topic', default_value='', description='Laserscan topic of the SLAM mapper (empty = none).')

    def nav2_cfg(*parts):
        return PathJoinSubstitution([robots_root, 'config', 'nav2', *parts])

    mobile_path = PathJoinSubstitution([
        robots_root, 'robots', robot.substitution, 'caps', 'mobile.yaml'
    ])
    model_params_path = PathJoinSubstitution([
        robots_root, 'robots', robot.substitution, 'model_params.yaml'
    ])
    interplanner_cfg = nav2_cfg('interplanners', inter_planner.substitution, 'interplanner_config.yaml')
    interplanner_yaml = YAMLFileSubstitution(interplanner_cfg)
    overlay = YAMLFileSubstitution(params_overlay.substitution, default={})

    def retrieve(key):
        return YAMLRetrieveSubstitution(interplanner_yaml, key)

    substitutions = YAMLMergeSubstitution(
        YAMLFileSubstitution(nav2_cfg('defaults', 'model_params.yaml')),
        YAMLFileSubstitution(mobile_path),
        Nav2SubBlockYAML(mobile_path),
        Nav2CollisionDerivedYAML(mobile_path),
        Nav2KinematicsDerivedYAML(mobile_path),
        SensorsDerivedYAML(model_params_path, mobile_path, sensors_json.substitution, frame.substitution),
        YAMLFileSubstitution(nav2_cfg('defaults', 'controller_config.yaml')),
        YAMLFileSubstitution(nav2_cfg('controllers', local_planner.substitution, 'controller_config.yaml')),
        YAMLFileSubstitution(nav2_cfg('defaults', 'planner_config.yaml')),
        YAMLFileSubstitution(nav2_cfg('planners', global_planner.substitution, 'planner_config.yaml')),
        YAMLFileSubstitution(nav2_cfg('defaults', 'interplanner_config.yaml')),
        YAMLFileSubstitution(interplanner_cfg),
        YAMLSelectSubstitution(overlay, 'substitutions'),
        YAMLFileSubstitution.from_dict(
            {
                'frame': frame.substitution,
                'base_frame': YAMLRetrieveSubstitution(YAMLFileSubstitution(model_params_path), 'base_frame'),
                **arena_env_node.dict,
                **env_namespace.dict,
                'namespace': namespace.substitution,
                # In train_mode the RL environment publishes cmd_vel directly.
                # Redirect the collision_monitor output to a dead topic so it
                # never overwrites the RL agent's velocity commands.
                'map_topic': PythonExpression(
                    ['"', namespace.substitution, '/map" if "', mapping.substitution, '" == "slam" else "', arena_env_node.substitution, '/map"']
                ),
                'scan_topic': scan_topic.substitution,
                'cmd_vel_out_topic': PythonExpression(
                    ['"cmd_vel_sink" if "', train_mode.substitution, '" == "true" else "cmd_vel"']
                ),
                'default_nav_to_pose_bt_xml': retrieve('bt_navigator/ros__parameters/default_nav_to_pose_bt_xml'),
                'default_nav_through_poses_bt_xml': retrieve('bt_navigator/ros__parameters/default_nav_through_poses_bt_xml'),
                'plugin_lib_names': retrieve('bt_navigator/ros__parameters/plugin_lib_names'),
            },
            substitute=True
        ),
    )

    substituted_parameters = YAMLReplaceSubstitution(
        obj=YAMLFileSubstitution(YAMLMergeSubstitution(
            YAMLFileSubstitution(nav2_cfg('nav2.yaml')),
            YAMLDropSubstitution(overlay, 'substitutions'),
        )),
        substitutions=YAMLFileSubstitution(substitutions)
    )

    slam_params = ParameterFile(
        RewrittenYaml(
            source_file=YAMLReplaceSubstitution(obj=YAMLFileSubstitution(nav2_cfg('slam.yaml')), substitutions=YAMLFileSubstitution(substitutions)),
            root_key=namespace.substitution,
            param_rewrites={'use_sim_time': use_sim_time.substitution},
            convert_types=True,
        ),
        allow_substs=True,
    )

    nav2_configured_params = ParameterFile(
        RewrittenYaml(
            source_file=substituted_parameters,
            root_key=namespace.substitution,
            param_rewrites={
                'use_sim_time': use_sim_time.substitution,
            },
            convert_types=True
        ),
        allow_substs=True,
    )

    full_lifecycle_nodes = [
        'controller_server',
        'smoother_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'waypoint_follower',
        'velocity_smoother',
        'collision_monitor'
    ]

    def launch_setup(context, *args, **kwargs):
        tgn = arena_env_node.substitution.perform(context)
        is_planner_only = planner_only.substitution.perform(context).lower() == 'true'
        mapping_mode = mapping.substitution.perform(context)
        if mapping_mode not in MAPPING_MODES:
            raise ValueError(f"mapping {mapping_mode!r} is none of {', '.join(MAPPING_MODES)}")
        slam = mapping_mode == 'slam'
        if slam and not scan_topic.substitution.perform(context):
            raise ValueError('mapping:=slam needs a laserscan sensor on the robot')
        remappings: list[tuple[SomeSubstitutionsType, SomeSubstitutionsType]] = [
            ('map_server', '/map_server'),
        ]
        if tgn and not slam:
            remappings.append(('map', PathJoinSubstitution([tgn, 'map'])))

        planner_server_node = Node(
            package='nav2_planner', executable='planner_server', name='planner_server',
            output='screen', parameters=[nav2_configured_params]
        )

        if is_planner_only:
            lifecycle_nodes = ['planner_server']
            nav2_nodes = [planner_server_node]
        else:
            lifecycle_nodes = full_lifecycle_nodes
            # cmd_vel chain: controller -> cmd_vel_nav -> smoother -> cmd_vel_smoothed <- behaviors
            #                cmd_vel_smoothed -> collision_monitor -> cmd_vel (-> twist_stamper)
            nav2_nodes = [
                Node(
                    package='nav2_controller', executable='controller_server', name='controller_server',
                    output='screen', parameters=[nav2_configured_params],
                    remappings=[('cmd_vel', 'cmd_vel_nav')],
                ),
                Node(
                    package='nav2_smoother', executable='smoother_server', name='smoother_server',
                    output='screen', parameters=[nav2_configured_params]
                ),
                planner_server_node,
                Node(
                    package='nav2_behaviors', executable='behavior_server', name='behavior_server',
                    output='screen', parameters=[nav2_configured_params],
                    remappings=[('cmd_vel', 'cmd_vel_smoothed')],
                ),
                Node(
                    package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
                    output='screen', parameters=[nav2_configured_params]
                ),
                Node(
                    package='nav2_waypoint_follower', executable='waypoint_follower', name='waypoint_follower',
                    output='screen', parameters=[nav2_configured_params]
                ),
                Node(
                    package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother',
                    output='screen', parameters=[nav2_configured_params],
                    remappings=[('cmd_vel', 'cmd_vel_nav'), ('smoothed_cmd_vel', 'cmd_vel_smoothed')],
                ),
                Node(
                    package='nav2_collision_monitor', executable='collision_monitor', name='collision_monitor',
                    output='screen', parameters=[nav2_configured_params]
                ),
            ]

        if slam:
            lifecycle_nodes = ['slam_toolbox', *lifecycle_nodes]

        bringup_cmd_group = GroupAction([
            *(SetRemap(src=r[0], dst=r[1]) for r in remappings),
            *nav2_nodes,
            Node(
                package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation',
                output='screen',
                parameters=[
                    {'autostart': True},
                    {'node_names': lifecycle_nodes},
                    nav2_configured_params
                ]
            ),
        ])
        result: list = [bringup_cmd_group]
        if slam:
            result.append(
                Node(
                    package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
                    output='screen', parameters=[slam_params],
                    remappings=[(f'/{topic}', PathJoinSubstitution([namespace.substitution, topic])) for topic in ('map', 'map_metadata')],
                )
            )

        controller_launch = nav2_cfg(
            'controllers', local_planner.substitution, 'controller.launch.py'
        ).perform(context)
        if os.path.isfile(controller_launch):
            result.append(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(controller_launch),
                    launch_arguments={
                        'namespace': namespace.substitution,
                        'env_namespace': env_namespace.substitution,
                        'frame': frame.substitution,
                        'base_frame': YAMLRetrieveSubstitution(YAMLFileSubstitution(model_params_path), 'base_frame'),
                        'use_sim_time': use_sim_time.substitution,
                        'agent': agent.substitution,
                    }.items(),
                )
            )

        return result

    return LaunchDescription([*ld_items, OpaqueFunction(function=launch_setup)])
