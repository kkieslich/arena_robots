"""Mapping modes of the nav2 adapter launch: the env's map server, or slam_toolbox mapping from the lidar while driving."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

NAV2_LAUNCH = Path(__file__).resolve().parents[2] / "launch" / "adapters" / "mobile" / "nav2.launch.py"
NAMESPACE = "/arena/env_0/arena_env_node/jackal"
ENV_NODE = "/arena/env_0/arena_env_node"


def _entities(mapping: str, scan_topic: str = f"{NAMESPACE}/lidar") -> tuple[list, object]:
    from launch import LaunchContext
    from launch.actions import OpaqueFunction

    spec = importlib.util.spec_from_file_location("nav2_adapter_launch", NAV2_LAUNCH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ld = module.generate_launch_description()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "robot": "jackal",
            "namespace": NAMESPACE,
            "frame": "env_0/jackal/",
            "use_sim_time": "true",
            "global_planner": "navfn",
            "local_planner": "dwb",
            "inter_planner": "empty",
            "planner_only": "true",
            "train_mode": "true",
            "arena_env_node": ENV_NODE,
            "env_namespace": "",
            "sensors_json": "",
            "params_overlay": "",
            "agent": "",
            "mapping": mapping,
            "scan_topic": scan_topic,
        }
    )
    (setup,) = [e for e in ld.entities if isinstance(e, OpaqueFunction)]
    return list(setup.execute(context)), context


def _params(node: object, context: object) -> dict:
    from launch.substitutions import TextSubstitution
    from launch_ros.actions import Node

    assert isinstance(node, Node)
    node._perform_substitutions(context)
    cmd = ["".join(s.text for s in part) if all(isinstance(s, TextSubstitution) for s in part) else None for part in node.cmd]
    (root,) = yaml.safe_load(Path(str(cmd[cmd.index("--params-file") + 1])).read_text()).values()
    return root


def _node(entities: list, executable: str) -> object:
    from launch.actions import GroupAction
    from launch_ros.actions import Node

    nodes = [e for e in entities if isinstance(e, Node)] + [e for g in entities if isinstance(g, GroupAction) for e in g.get_sub_entities() if isinstance(e, Node)]
    (found,) = [n for n in nodes if n.node_executable == executable]
    return found


def _remaps(entities: list) -> list[str]:
    from launch.actions import GroupAction
    from launch_ros.actions import SetRemap

    return ["".join(part.text for part in e._SetRemap__src) for g in entities if isinstance(g, GroupAction) for e in g.get_sub_entities() if isinstance(e, SetRemap)]


def _static_map_topic(mapping: str) -> str:
    entities, context = _entities(mapping)
    return _params(_node(entities, "planner_server"), context)["global_costmap"]["global_costmap"]["ros__parameters"]["static_layer"]["map_topic"]


def test_static_mapping_reads_the_env_map_server():
    assert _static_map_topic("static") == f"{ENV_NODE}/map"


def test_slam_mapping_reads_the_robots_own_map():
    assert _static_map_topic("slam") == f"{NAMESPACE}/map"


def test_static_mapping_starts_no_slam():
    entities, _ = _entities("static")
    with pytest.raises(ValueError):
        _node(entities, "async_slam_toolbox_node")


def test_slam_maps_from_the_lidar_and_leaves_the_tf_to_the_simulator():
    entities, context = _entities("slam")
    params = _params(_node(entities, "async_slam_toolbox_node"), context)["slam_toolbox"]["ros__parameters"]
    assert params["scan_topic"] == f"{NAMESPACE}/lidar"
    assert params["odom_frame"] == "map"
    assert params["base_frame"] == "env_0/jackal/base_link"
    assert params["map_frame"] == "map"
    assert params["transform_publish_period"] == 0.0
    assert params["use_scan_matching"] is False


def _managed(mapping: str) -> list[str]:
    from launch.substitutions import TextSubstitution
    from launch_ros.actions import Node

    entities, context = _entities(mapping)
    manager = _node(entities, "lifecycle_manager")
    assert isinstance(manager, Node)
    manager._perform_substitutions(context)
    cmd = ["".join(s.text for s in part) if all(isinstance(s, TextSubstitution) for s in part) else "" for part in manager.cmd]
    files = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--params-file"]
    found = [params["node_names"] for path in files for root in yaml.load(Path(path).read_text(), Loader=yaml.FullLoader).values() for params in [root.get("ros__parameters", {})] if "node_names" in params]
    (names,) = found
    return list(names)


def test_the_lifecycle_manager_brings_slam_up_before_the_planner():
    entities, context = _entities("slam")
    params = _params(_node(entities, "async_slam_toolbox_node"), context)["slam_toolbox"]["ros__parameters"]
    assert params["use_lifecycle_manager"] is True
    names = _managed("slam")
    assert names.index("slam_toolbox") < names.index("planner_server")


def test_slam_publishes_its_map_in_the_robot_namespace():
    from launch_ros.actions import Node

    entities, context = _entities("slam")
    slam = _node(entities, "async_slam_toolbox_node")
    assert isinstance(slam, Node)
    slam._perform_substitutions(context)
    remaps = dict(slam.expanded_remapping_rules or [])
    assert remaps["/map"] == f"{NAMESPACE}/map"
    assert remaps["/map_metadata"] == f"{NAMESPACE}/map_metadata"


def test_static_mapping_manages_no_slam():
    assert "slam_toolbox" not in _managed("static")


def test_slam_mapping_keeps_nav2_off_the_env_map():
    assert "map" not in _remaps(_entities("slam")[0])
    assert "map" in _remaps(_entities("static")[0])


def test_slam_without_a_laserscan_is_refused():
    with pytest.raises(ValueError, match="laserscan"):
        _entities("slam", scan_topic="")


def test_an_unknown_mapping_mode_is_refused():
    with pytest.raises(ValueError, match="none of"):
        _entities("octomap")


def test_the_scan_topic_is_the_first_laserscan_with_its_namespace():
    from arena_robots.bringup.mobile.nav2 import scan_topic
    from arena_robots.Sensor import SensorSpec, SensorType

    sensors = [
        SensorSpec(name="imu", type=SensorType.IMU, topic="${namespace}/imu/data", frame="imu_link"),
        SensorSpec(name="lidar", type=SensorType.LASERSCAN, topic="${namespace}/lidar", frame="lidar_link"),
        SensorSpec(name="rear", type=SensorType.LASERSCAN, topic="${namespace}/rear", frame="rear_link"),
    ]
    assert scan_topic(sensors, NAMESPACE) == f"{NAMESPACE}/lidar"
    assert scan_topic(sensors[:1], NAMESPACE) == ""
