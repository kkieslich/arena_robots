"""Domain fixtures for arena_robots ROS-gated tests."""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

GOAL_TIMEOUT = 15.0

_rig_ids = itertools.count()


def until(pred: Callable[[], bool], what: str, timeout: float = GOAL_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        time.sleep(0.01)


class Nav2Server:
    """A NavigateToPose action server scripted per goal."""

    def __init__(self, node: Any, name: str, *, reject_first: int = 0, outcomes: tuple[str, ...] = ("succeed",), hold: bool = False, distance: float | None = None) -> None:
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionServer, CancelResponse

        self._node = node
        self._reject_first = reject_first
        self._outcomes = list(outcomes)
        self.holding = hold
        self.distance = distance
        self.received = 0
        self.canceled = 0
        self._server = ActionServer(node, NavigateToPose, name, execute_callback=self._execute, goal_callback=self._on_goal, cancel_callback=lambda _handle: CancelResponse.ACCEPT)

    def _on_goal(self, _goal: Any) -> Any:
        from rclpy.action import GoalResponse

        self.received += 1
        return GoalResponse.REJECT if self.received <= self._reject_first else GoalResponse.ACCEPT

    async def _execute(self, goal_handle: Any) -> Any:
        from nav2_msgs.action import NavigateToPose

        from arena_robots.task_server_handlers import _executor_sleep

        while self.holding and not goal_handle.is_cancel_requested:
            if self.distance is not None:
                feedback = NavigateToPose.Feedback()
                feedback.distance_remaining = self.distance
                feedback.estimated_time_remaining.sec = 3
                goal_handle.publish_feedback(feedback)
            await _executor_sleep(self._node, 0.02, wall=True)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self.canceled += 1
        elif (self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]) == "abort":
            goal_handle.abort()
        else:
            goal_handle.succeed()
        return NavigateToPose.Result()

    def destroy(self) -> None:
        self._server.destroy()


class GotoRig:
    """A goto_pose handler behind a real action server, a client, and the lockstep register service."""

    def __init__(self, namespace: str) -> None:
        from arena_runtime_msgs.srv import LockstepRegister
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        self.namespace = namespace
        self.bringup = SimpleNamespace(
            native_action_name=f"{namespace}/navigate_to_pose",
            goal_topic=f"{namespace}/goal_pose",
            namespace=lambda *parts: "/".join((namespace, *parts)),
            frame="env_0/",
            robot=SimpleNamespace(model_params=SimpleNamespace(base_frame="base_link")),
        )
        self.node = Node("task_server", namespace=namespace)
        self.peer = Node("task_peer", namespace=namespace)
        self.registrations: list[list[Any]] = []
        self.handles: list[Any] = []
        self.feedback: list[Any] = []
        self._stamps = itertools.count(1)
        self._server: Any = None
        self._client: Any = None
        self._nav2: list[Nav2Server] = []
        self.peer.create_service(LockstepRegister, "/arena/sim_lifecycle/lockstep/register", self._on_register)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._executor.add_node(self.peer)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _on_register(self, request: Any, response: Any) -> Any:
        self.registrations.append(list(request.registration.channels))
        response.success = True
        return response

    def nav2(self, **script: Any) -> Nav2Server:
        server = Nav2Server(self.peer, self.bringup.native_action_name, **script)
        self._nav2.append(server)
        return server

    def controller(self, **params: float) -> list[dict[str, float]]:
        """Serve controller_server get/set_parameters over `params`, returning the list every set lands in."""
        from rcl_interfaces.msg import ParameterType, ParameterValue, SetParametersResult
        from rcl_interfaces.srv import GetParameters, SetParameters

        sets: list[dict[str, float]] = []

        def on_get(request: Any, response: Any) -> Any:
            response.values = [ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=params[name]) if name in params else ParameterValue() for name in request.names]
            return response

        def on_set(request: Any, response: Any) -> Any:
            values = {p.name: p.value.double_value for p in request.parameters}
            params.update(values)
            sets.append(values)
            response.results = [SetParametersResult(successful=True) for _ in request.parameters]
            return response

        self.peer.create_service(GetParameters, f"{self.namespace}/controller_server/get_parameters", on_get)
        self.peer.create_service(SetParameters, f"{self.namespace}/controller_server/set_parameters", on_set)
        return sets

    def serve(self, handler_cls: type, tf_buffer: Any = None) -> Any:
        from arena_robots_msgs.action import GotoPose
        from rclpy.action import ActionClient, ActionServer

        handler = handler_cls(self.bringup, tf_buffer=tf_buffer, node=self.node)

        async def execute(goal_handle: Any) -> Any:
            self.handles.append(goal_handle)
            return await handler.execute(goal_handle)

        self._server = ActionServer(self.node, GotoPose, "goto_pose", execute_callback=execute, cancel_callback=handler.on_cancel)
        self._client = ActionClient(self.peer, GotoPose, "goto_pose")
        assert self._client.wait_for_server(timeout_sec=GOAL_TIMEOUT)
        return handler

    def _wait(self, future: Any, what: str) -> Any:
        until(future.done, what)
        return future.result()

    def send(self, x: float = 0.0, *, pose_tolerance: float = 0.0, yaw_tolerance: float = 0.0) -> Any:
        from arena_robots_msgs.action import GotoPose

        goal = GotoPose.Goal()
        goal.target.pose.position.x = x
        goal.pose_tolerance = pose_tolerance
        goal.yaw_tolerance = yaw_tolerance
        handle = self._wait(self._client.send_goal_async(goal, feedback_callback=self.feedback.append), "goal response")
        assert handle.accepted
        return handle

    def result(self, handle: Any) -> Any:
        return self._wait(handle.get_result_async(), "goal result")

    def cancel(self, handle: Any) -> None:
        self._wait(handle.cancel_goal_async(), "cancel response")

    def listen(self, handler: Any) -> list[Any]:
        from geometry_msgs.msg import PoseStamped

        received: list[Any] = []
        self.peer.create_subscription(PoseStamped, self.bringup.goal_topic, received.append, 10)
        until(lambda: handler._pub.get_subscription_count() == 1, "goal subscription")
        return received

    def robot_at(self, buffer: Any, x: float) -> None:
        from geometry_msgs.msg import TransformStamped

        transform = TransformStamped()
        transform.header.stamp.sec = next(self._stamps)
        transform.header.frame_id = "map"
        transform.child_frame_id = "env_0/base_link"
        transform.transform.translation.x = x
        transform.transform.rotation.w = 1.0
        buffer.set_transform(transform, "test")

    def close(self) -> None:
        for server in self._nav2:
            server.holding = False
            server.destroy()
        if self._client is not None:
            self._client.destroy()
        if self._server is not None:
            self._server.destroy()
        self._executor.shutdown()
        self._thread.join(timeout=GOAL_TIMEOUT)
        self.node.destroy_node()
        self.peer.destroy_node()


@pytest.fixture()
def goto_rig() -> Iterator[GotoRig]:
    rig = GotoRig(f"/env_0/robot{next(_rig_ids)}")
    yield rig
    rig.close()


@pytest.fixture()
def temp_robot_dir(tmp_path: Path):
    """Return a factory that writes a minimal robot directory tree.

    Usage::

        def test_foo(temp_robot_dir):
            robot_dir = temp_robot_dir(
                model_params={"base_frame": "base_link"},
                control={"cmd_vel": "/cmd_vel"},
                mobile_cap={"odom_frame": "odom"},
            )
    """

    def _factory(
        *,
        model_params: dict[str, Any] | None = None,
        control: dict[str, Any] | None = None,
        mobile_cap: dict[str, Any] | None = None,
        extra_caps: dict[str, dict[str, Any]] | None = None,
        name: str = "test_robot",
    ) -> Path:
        robot_dir = tmp_path / name
        robot_dir.mkdir(parents=True, exist_ok=True)

        if model_params is not None:
            (robot_dir / "model_params.yaml").write_text(yaml.dump(model_params))

        if control is not None:
            (robot_dir / "control.yaml").write_text(yaml.dump(control))

        if mobile_cap is not None or extra_caps is not None:
            caps_dir = robot_dir / "caps"
            caps_dir.mkdir(exist_ok=True)
            if mobile_cap is not None:
                (caps_dir / "mobile.yaml").write_text(yaml.dump(mobile_cap))
            for cap_name, cap_data in (extra_caps or {}).items():
                (caps_dir / f"{cap_name}.yaml").write_text(yaml.dump(cap_data))

        return robot_dir

    return _factory


@pytest.fixture()
def stub_node():
    """Minimal Node-like SimpleNamespace for tests that only read one or two attrs."""
    node = SimpleNamespace()
    node.get_logger = lambda: SimpleNamespace(
        info=lambda msg: None,
        warn=lambda msg: None,
        error=lambda msg: None,
        debug=lambda msg: None,
    )
    node.create_publisher = MagicMock()
    node.create_timer = MagicMock()
    node.destroy_timer = MagicMock()
    return node
