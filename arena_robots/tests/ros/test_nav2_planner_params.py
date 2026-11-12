"""Planner parameters resolved by the nav2 adapter launch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

NAV2_LAUNCH = Path(__file__).resolve().parents[2] / "launch" / "adapters" / "mobile" / "nav2.launch.py"


def _planner_params(robot: str, global_planner: str) -> dict:
    from launch import LaunchContext
    from launch.actions import GroupAction, OpaqueFunction
    from launch.substitutions import TextSubstitution
    from launch_ros.actions import Node

    spec = importlib.util.spec_from_file_location("nav2_adapter_launch", NAV2_LAUNCH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ld = module.generate_launch_description()

    context = LaunchContext()
    context.launch_configurations.update(
        {
            "robot": robot,
            "namespace": f"/env_0/{robot}",
            "frame": f"env_0/{robot}/",
            "use_sim_time": "true",
            "global_planner": global_planner,
            "local_planner": "dwb",
            "inter_planner": "empty",
            "planner_only": "true",
            "train_mode": "true",
            "arena_env_node": "",
            "env_namespace": "",
            "sensors_json": "",
            "params_overlay": "",
            "agent": "",
            "mapping": "static",
            "scan_topic": "",
        }
    )
    (setup,) = [e for e in ld.entities if isinstance(e, OpaqueFunction)]
    (group, *_) = setup.execute(context)
    assert isinstance(group, GroupAction)
    (planner_server,) = [e for e in group.get_sub_entities() if isinstance(e, Node) and e.node_executable == "planner_server"]
    planner_server._perform_substitutions(context)
    cmd = ["".join(s.text for s in part) if all(isinstance(s, TextSubstitution) for s in part) else None for part in planner_server.cmd]
    params_file = cmd[cmd.index("--params-file") + 1]
    (root,) = yaml.safe_load(Path(params_file).read_text()).values()
    return root["planner_server"]["ros__parameters"]


@pytest.mark.parametrize("robot", ["jackal", "husky"])
def test_navfn_plans_with_dijkstra(robot: str) -> None:
    params = _planner_params(robot, "navfn")
    assert params["planner_plugins"] == ["GridBased"]
    assert params["GridBased"]["plugin"] == "nav2_navfn_planner::NavfnPlanner"
    assert params["GridBased"]["use_astar"] is False


def test_selected_planner_sets_the_planner_plugin() -> None:
    params = _planner_params("jackal", "smac_2d")
    assert params["GridBased"]["plugin"] == "nav2_smac_planner::SmacPlanner2D"
    assert params["GridBased"]["tolerance"] == 0.125
