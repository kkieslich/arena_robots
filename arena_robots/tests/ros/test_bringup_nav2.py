"""Tests for arena_robots.bringup.mobile.nav2 - Nav2Bringup."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from arena_robots.bringup.mobile.nav2 import Nav2Bringup
    from launch import LaunchDescription

ROSNAV_RL_SIDECAR = Path(__file__).resolve().parents[2] / "config" / "nav2" / "controllers" / "rosnav_rl" / "controller.launch.py"


def _make_mock_robot(name: str = "test_robot") -> object:
    mock_robot = MagicMock()
    mock_robot.name = name
    mock_robot.caps.mobile.min_turning_radius = 0.0
    return mock_robot


class TestNav2BringupAttributes:
    def test_kind_value(self):
        from arena_robots.bringup.mobile.nav2 import Nav2Bringup

        assert Nav2Bringup.kind == "nav2"

    def test_requires_mobile(self):
        from arena_robots.bringup.mobile.nav2 import Nav2Bringup

        assert "mobile" in Nav2Bringup._bringup_meta.requires

    def test_requires_frozenset(self):
        from arena_robots.bringup.mobile.nav2 import Nav2Bringup

        assert isinstance(Nav2Bringup._bringup_meta.requires, frozenset)


class TestNav2BringupProperties:
    def setup_method(self):
        robot = _make_mock_robot("myrobot")
        from arena_robots.bringup.mobile.nav2 import Nav2Bringup

        self.b = Nav2Bringup(robot=robot, namespace="/robot1")

    def test_native_action_name_contains_namespace(self):
        ep = self.b.native_action_name
        assert "navigate_to_pose" in ep
        assert "robot1" in ep

    def test_bt_node_name_contains_namespace(self):
        ep = self.b.bt_node_name
        assert "bt_navigator" in ep
        assert "robot1" in ep


class TestNav2LaunchActions:
    def _make_bringup(self, namespace: str = "/robot1"):
        robot = _make_mock_robot("myrobot")
        from arena_robots.bringup.mobile.nav2 import Nav2Bringup

        return Nav2Bringup(robot=robot, namespace=namespace)

    def test_returns_list(self):
        b = self._make_bringup()
        actions = b._launch_actions()
        assert isinstance(actions, list)
        assert len(actions) == 1

    def test_use_sim_time_true_lowercase(self):
        b = self._make_bringup()
        actions = b._launch_actions(use_sim_time=True)
        action = actions[0]
        args = dict(action.launch_arguments)
        assert args["use_sim_time"] == "true"

    def test_use_sim_time_false_lowercase(self):
        b = self._make_bringup()
        actions = b._launch_actions(use_sim_time=False)
        action = actions[0]
        args = dict(action.launch_arguments)
        assert args["use_sim_time"] == "false"

    def test_planner_arg_forwarded(self):
        b = self._make_bringup()
        actions = b._launch_actions(global_planner="NavFn")
        args = dict(actions[0].launch_arguments)
        assert args["global_planner"] == "NavFn"

    def test_local_planner_arg_forwarded(self):
        b = self._make_bringup()
        actions = b._launch_actions(local_planner="TEB")
        args = dict(actions[0].launch_arguments)
        assert args["local_planner"] == "TEB"

    def test_inter_planner_arg_forwarded(self):
        b = self._make_bringup()
        actions = b._launch_actions(inter_planner="smac")
        args = dict(actions[0].launch_arguments)
        assert args["inter_planner"] == "smac"

    @pytest.mark.parametrize(("robot", "inter_planner"), [("jackal", "default"), ("f1tenth", "default_no_spin")])
    def test_default_inter_planner_spins_only_for_a_robot_that_turns_in_place(self, robot: str, inter_planner: str) -> None:
        from arena_robots.bringup.mobile.nav2 import Nav2Bringup
        from arena_robots.Robot import RobotIdentifier

        bringup = Nav2Bringup(robot=RobotIdentifier(robot).resolve_sync(), namespace="/robot1")
        assert dict(bringup._launch_actions()[0].launch_arguments)["inter_planner"] == inter_planner

    def test_explicit_inter_planner_is_kept_for_a_car_like_robot(self) -> None:
        from arena_robots.bringup.mobile.nav2 import Nav2Bringup
        from arena_robots.Robot import RobotIdentifier

        bringup = Nav2Bringup(robot=RobotIdentifier("f1tenth").resolve_sync(), namespace="/robot1")
        assert dict(bringup._launch_actions(inter_planner="navigate_w_replanning_time")[0].launch_arguments)["inter_planner"] == "navigate_w_replanning_time"

    def test_extra_kwargs_ignored(self):
        b = self._make_bringup()
        actions = b._launch_actions(unknown_kwarg="ignored")
        assert isinstance(actions, list)

    def test_robot_name_forwarded(self):
        b = self._make_bringup()
        actions = b._launch_actions()
        args = dict(actions[0].launch_arguments)
        assert args["robot"] == "myrobot"

    def test_namespace_forwarded(self):
        b = self._make_bringup()
        actions = b._launch_actions()
        args = dict(actions[0].launch_arguments)
        assert "robot1" in args["namespace"]


class TestNav2RosnavRlAgent:
    def _make_bringup(self, temp_robot_dir: Callable[..., Path], mobile_cap: dict) -> Nav2Bringup:
        from arena_robots.bringup.mobile.nav2 import Nav2Bringup
        from arena_robots.Robot import RobotView

        rd = temp_robot_dir(model_params={"base_frame": "base_link"}, mobile_cap=mobile_cap, name="myrobot")
        return Nav2Bringup(robot=RobotView(rd), namespace="/robot1")

    def test_agent_kwarg_forwarded(self, temp_robot_dir: Callable[..., Path]) -> None:
        b = self._make_bringup(temp_robot_dir, {"odom_frame": "odom"})
        actions = b._launch_actions(local_planner="rosnav_rl", agent="my_agent")
        assert dict(actions[0].launch_arguments)["agent"] == "my_agent"

    def test_agent_from_mobile_cap(self, temp_robot_dir: Callable[..., Path]) -> None:
        b = self._make_bringup(temp_robot_dir, {"odom_frame": "odom", "rosnav_rl": {"agent": "cap_agent"}})
        actions = b._launch_actions(local_planner="rosnav_rl")
        assert dict(actions[0].launch_arguments)["agent"] == "cap_agent"

    def test_agent_kwarg_overrides_mobile_cap(self, temp_robot_dir: Callable[..., Path]) -> None:
        b = self._make_bringup(temp_robot_dir, {"odom_frame": "odom", "rosnav_rl": {"agent": "cap_agent"}})
        actions = b._launch_actions(local_planner="rosnav_rl", agent="cli_agent")
        assert dict(actions[0].launch_arguments)["agent"] == "cli_agent"

    def test_rosnav_rl_without_agent_raises(self, temp_robot_dir: Callable[..., Path]) -> None:
        b = self._make_bringup(temp_robot_dir, {"odom_frame": "odom"})
        with pytest.raises(ValueError, match="robot.mobile.agent"):
            b._launch_actions(local_planner="rosnav_rl")

    def test_other_local_planner_without_agent(self, temp_robot_dir: Callable[..., Path]) -> None:
        b = self._make_bringup(temp_robot_dir, {"odom_frame": "odom"})
        actions = b._launch_actions(local_planner="mppi")
        assert dict(actions[0].launch_arguments)["agent"] == ""


class TestRosnavRlControllerSidecar:
    def _load(self) -> LaunchDescription:
        spec = importlib.util.spec_from_file_location("rosnav_rl_controller_launch", ROSNAV_RL_SIDECAR)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.generate_launch_description()

    def test_declares_nav2_sidecar_args(self) -> None:
        ld = self._load()
        assert {a.name for a in ld.get_launch_arguments()} == {"namespace", "frame", "base_frame", "use_sim_time", "agent"}

    def test_starts_rosnav_rl_action_server(self) -> None:
        from launch import LaunchContext
        from launch.actions import OpaqueFunction

        ld = self._load()
        context = LaunchContext()
        context.launch_configurations.update({"namespace": "/env0/robot1", "frame": "env0/robot1/", "base_frame": "base_link", "use_sim_time": "true", "agent": "my_agent"})
        (setup,) = [e for e in ld.entities if isinstance(e, OpaqueFunction)]
        (node,) = setup.execute(context)
        assert node.node_package == "rosnav_rl"
        assert node.node_executable == "action_server.py"
