"""Tests for the MoveIt reach handler: goal building, request validation and the pure helpers of linear motion, relative targets and contact."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

rclpy = pytest.importorskip("rclpy")

from arena_robots_msgs.action import ReachPose
from geometry_msgs.msg import Pose
from control_msgs.msg import JointTrajectoryControllerState
from moveit_msgs.msg import AllowedCollisionEntry, AllowedCollisionMatrix
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import Buffer

from arena_robots.caps import ArmSpec

_ARM_ENTRY: dict = {
    "base_link": "chassis_link",
    "tip_link": "tool0",
    "chain": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
    "controller": "arm_controller",
    "moveit": {"package": "ur_moveit_config", "planning_group": "ur_manipulator"},
}


def _wait(future, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while not future.done():
        assert time.monotonic() < deadline, "future did not complete"
        time.sleep(0.02)
    return future.result()


def _robot_dir(tmp_path: Path) -> Path:
    robot_dir = tmp_path / "robot"
    (robot_dir / "caps").mkdir(parents=True)
    (robot_dir / "model_params.yaml").write_text(yaml.dump({}))
    (robot_dir / "caps" / "arm.yaml").write_text(yaml.dump({"arm": _ARM_ENTRY}))
    return robot_dir


def _goal(**fields) -> ReachPose.Goal:
    goal = ReachPose.Goal()
    goal.target.header.frame_id = "base"
    goal.target.pose.position.x = 0.4
    goal.target.pose.orientation.w = 1.0
    for name, value in fields.items():
        setattr(goal, name, value)
    return goal


def _yaw(angle: float) -> Pose:
    pose = Pose()
    pose.orientation.z = math.sin(angle / 2)
    pose.orientation.w = math.cos(angle / 2)
    return pose


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[tuple[object, ActionClient, ArmSpec]]:
    from arena_robots.bringup.arm.moveit import MoveItArmBringup
    from arena_robots.Robot import RobotView
    from arena_robots.task_server_handlers.reach_pose.moveit import ReachPoseHandlerMoveIt

    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    ns = f"/reach_pose_{uuid.uuid4().hex[:8]}"
    node = rclpy.create_node("reach_pose_moveit", namespace=ns)
    bringup = MoveItArmBringup(RobotView(_robot_dir(tmp_path)), ns)
    handler = ReachPoseHandlerMoveIt(bringup, tf_buffer=Buffer(), node=node)
    group = ReentrantCallbackGroup()
    ActionServer(node, ReachPose, f"{ns}/reach_pose", execute_callback=handler.execute, cancel_callback=handler.on_cancel, callback_group=group)
    client = ActionClient(node, ReachPose, f"{ns}/reach_pose", callback_group=group)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        assert client.wait_for_server(timeout_sec=30.0)
        yield handler, client, bringup.arms()["arm"]
    finally:
        executor.shutdown()
        node.destroy_node()
        if owned:
            rclpy.shutdown()


def _run(client: ActionClient, goal: ReachPose.Goal):
    handle = _wait(client.send_goal_async(goal))
    assert handle.accepted
    return _wait(handle.get_result_async()).result


class TestBuildGoalMotion:
    def test_default_goal_uses_the_default_pipeline_at_full_speed(self, rig) -> None:
        handler, _, arm = rig
        request = handler._build_goal(_goal(), arm).request
        assert request.pipeline_id == ""
        assert request.planner_id == ""
        assert request.max_velocity_scaling_factor == 1.0
        assert request.max_acceleration_scaling_factor == 1.0

    def test_linear_goal_selects_the_pilz_lin_planner(self, rig) -> None:
        handler, _, arm = rig
        request = handler._build_goal(_goal(motion="linear"), arm).request
        assert request.pipeline_id == "pilz_industrial_motion_planner"
        assert request.planner_id == "LIN"

    def test_free_goal_keeps_the_default_pipeline(self, rig) -> None:
        handler, _, arm = rig
        request = handler._build_goal(_goal(motion="free"), arm).request
        assert request.pipeline_id == ""

    def test_speed_scales_velocity_and_acceleration(self, rig) -> None:
        handler, _, arm = rig
        request = handler._build_goal(_goal(speed=0.25), arm).request
        assert request.max_velocity_scaling_factor == pytest.approx(0.25)
        assert request.max_acceleration_scaling_factor == pytest.approx(0.25)

    def test_goal_only_plans_so_the_handler_executes_from_its_own_start_state(self, rig) -> None:
        handler, _, arm = rig
        assert handler._build_goal(_goal(), arm).planning_options.plan_only is True


class TestExecuteValidation:
    def test_unknown_motion_is_aborted(self, rig) -> None:
        _, client, _ = rig
        result = _run(client, _goal(motion="spiral"))
        assert result.status == ReachPose.Result.STATUS_ABORTED
        assert "spiral" in result.reason

    def test_linear_motion_with_a_named_target_is_aborted(self, rig) -> None:
        _, client, _ = rig
        result = _run(client, _goal(motion="linear", named_target="ready"))
        assert result.status == ReachPose.Result.STATUS_ABORTED
        assert "named target" in result.reason

    def test_relative_goal_without_a_transform_names_both_frames(self, rig) -> None:
        _, client, _ = rig
        result = _run(client, _goal(relative=True))
        assert result.status == ReachPose.Result.STATUS_ABORTED
        assert "tool0" in result.reason
        assert "base" in result.reason


class TestComposeOffset:
    def test_positions_add(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _compose_offset

        tool, offset = Pose(), Pose()
        tool.position.x, tool.position.y, tool.position.z = 1.0, 2.0, 3.0
        offset.position.x, offset.position.y, offset.position.z = 0.1, -0.2, 0.3
        tool.orientation.w = offset.orientation.w = 1.0
        out = _compose_offset(tool, offset)
        assert (out.position.x, out.position.y, out.position.z) == pytest.approx((1.1, 1.8, 3.3))

    def test_offset_rotation_multiplies_the_tool_rotation(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _compose_offset

        out = _compose_offset(_yaw(math.pi / 2), _yaw(math.pi / 2))
        assert (out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w) == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-9)

    def test_identity_offset_keeps_the_tool_orientation(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _compose_offset

        offset = Pose()
        offset.orientation.w = 1.0
        tool = _yaw(0.7)
        out = _compose_offset(tool, offset)
        assert (out.orientation.z, out.orientation.w) == pytest.approx((tool.orientation.z, tool.orientation.w))


class TestDescendantLinks:
    _URDF = """<robot name="r">
      <link name="base"/><link name="tip"/><link name="hand"/><link name="finger_l"/><link name="finger_r"/><link name="elbow"/>
      <joint name="j0" type="fixed"><parent link="base"/><child link="elbow"/></joint>
      <joint name="j1" type="fixed"><parent link="elbow"/><child link="tip"/></joint>
      <joint name="j2" type="fixed"><parent link="tip"/><child link="hand"/></joint>
      <joint name="j3" type="fixed"><parent link="hand"/><child link="finger_l"/></joint>
      <joint name="j4" type="fixed"><parent link="hand"/><child link="finger_r"/></joint>
    </robot>"""

    def test_returns_the_tip_and_everything_below_it(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _descendant_links

        assert _descendant_links(self._URDF, "tip") == {"tip", "hand", "finger_l", "finger_r"}

    def test_a_leaf_link_is_its_own_subtree(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _descendant_links

        assert _descendant_links(self._URDF, "finger_l") == {"finger_l"}


def _matrix(names: list[str], pairs: set[tuple[str, str]]) -> AllowedCollisionMatrix:
    acm = AllowedCollisionMatrix()
    acm.entry_names = names
    acm.entry_values = [AllowedCollisionEntry(enabled=[(a, b) in pairs or (b, a) in pairs for b in names]) for a in names]
    return acm


def _allowed(acm: AllowedCollisionMatrix, a: str, b: str) -> bool:
    return bool(acm.entry_values[acm.entry_names.index(a)].enabled[acm.entry_names.index(b)])


class TestArmLinks:
    _URDF = """<robot name="r">
      <link name="base"/><link name="shoulder"/><link name="elbow"/><link name="wrist"/><link name="tip"/><link name="hand"/><link name="finger"/>
      <joint name="j0" type="revolute"><parent link="base"/><child link="shoulder"/></joint>
      <joint name="j1" type="revolute"><parent link="shoulder"/><child link="elbow"/></joint>
      <joint name="j2" type="revolute"><parent link="elbow"/><child link="wrist"/></joint>
      <joint name="j3" type="revolute"><parent link="wrist"/><child link="tip"/></joint>
      <joint name="j4" type="fixed"><parent link="tip"/><child link="hand"/></joint>
      <joint name="j5" type="fixed"><parent link="hand"/><child link="finger"/></joint>
    </robot>"""

    def test_chain_joint_links_join_the_tip_subtree(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _arm_links

        assert _arm_links(self._URDF, "tip", ["j0", "j1", "j2", "j3"]) == {"shoulder", "elbow", "wrist", "tip", "hand", "finger"}

    def test_a_chain_naming_unknown_joints_adds_nothing(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _arm_links

        assert _arm_links(self._URDF, "tip", ["missing"]) == {"tip", "hand", "finger"}


class TestJointBounds:
    _URDF = """<robot name="r">
      <link name="a"/><link name="b"/><link name="c"/>
      <joint name="j1" type="revolute"><parent link="a"/><child link="b"/><limit lower="-1.5" upper="1.5" effort="1" velocity="1"/></joint>
      <joint name="j2" type="continuous"><parent link="b"/><child link="c"/></joint>
    </robot>"""

    def test_returns_the_limited_joints_only(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _joint_bounds

        assert _joint_bounds(self._URDF) == {"j1": (-1.5, 1.5)}


class TestStartState:
    def _handler(self, positions: dict[str, float], bounds: dict[str, tuple[float, float]]):
        from arena_robots.task_server_handlers.reach_pose.moveit import ReachPoseHandlerMoveIt

        handler = object.__new__(ReachPoseHandlerMoveIt)
        handler._positions = {"top": positions}
        handler._models = {"top": (frozenset(), bounds)}
        return handler

    class _Arm:
        chain = ["j1", "j2"]

    def test_a_joint_resting_a_hair_outside_its_limit_is_clamped(self) -> None:
        handler = self._handler({"j1": -1.5708001, "j2": 0.5}, {"j1": (-1.5707963, 1.5707963), "j2": (-1.0, 1.0)})
        state = handler._start_state("top", self._Arm())
        assert list(state.name) == ["j1", "j2"]
        assert list(state.position) == [-1.5707963, 0.5]

    def test_a_joint_far_outside_its_limit_gets_no_start_state(self) -> None:
        handler = self._handler({"j1": -1.8, "j2": 0.5}, {"j1": (-1.0, 1.0), "j2": (-1.0, 1.0)})
        assert handler._start_state("top", self._Arm()) is None

    def test_a_joint_without_a_reading_gets_no_start_state(self) -> None:
        handler = self._handler({"j1": 0.0}, {"j1": (-1.0, 1.0), "j2": (-1.0, 1.0)})
        assert handler._start_state("top", self._Arm()) is None


class TestAllowContact:
    def test_object_may_touch_exactly_the_given_links(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _allow_contact

        acm = _matrix(["base", "tip", "hand"], {("base", "tip")})
        out = _allow_contact(acm, "arena/chest/drawer", frozenset({"tip", "hand"}))
        assert _allowed(out, "arena/chest/drawer", "tip")
        assert _allowed(out, "hand", "arena/chest/drawer")
        assert not _allowed(out, "arena/chest/drawer", "base")
        assert not _allowed(out, "arena/chest/drawer", "arena/chest/drawer")

    def test_existing_entries_survive(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _allow_contact

        acm = _matrix(["base", "tip"], {("base", "tip")})
        out = _allow_contact(acm, "obj", frozenset({"tip"}))
        assert _allowed(out, "base", "tip")
        assert _allowed(out, "tip", "base")

    def test_links_missing_from_the_matrix_are_added(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _allow_contact

        acm = _matrix(["base"], set())
        out = _allow_contact(acm, "obj", frozenset({"finger"}))
        assert set(out.entry_names) == {"base", "obj", "finger"}
        assert _allowed(out, "obj", "finger")
        assert all(len(entry.enabled) == len(out.entry_names) for entry in out.entry_values)

    def test_reusing_an_object_resets_its_row(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _allow_contact

        acm = _matrix(["base", "tip", "obj"], {("obj", "base")})
        out = _allow_contact(acm, "obj", frozenset({"tip"}))
        assert not _allowed(out, "obj", "base")
        assert _allowed(out, "obj", "tip")
        assert len(out.entry_names) == 3

    def test_input_matrix_is_not_modified(self) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _allow_contact

        acm = _matrix(["base"], set())
        _allow_contact(acm, "obj", frozenset({"tip"}))
        assert list(acm.entry_names) == ["base"]


class TestArmSpecStallTunables:
    def test_defaults(self, tmp_path: Path) -> None:
        arm = ArmSpec(path=tmp_path, raw=dict(_ARM_ENTRY), name="arm")
        assert (arm.stall_error, arm.stall_speed, arm.stall_time, arm.scene_radius) == (0.2, 0.05, 0.5, 1.5)

    def test_arm_block_values_win(self, tmp_path: Path) -> None:
        arm = ArmSpec(path=tmp_path, raw={**_ARM_ENTRY, "stall_error": 0.35, "stall_time": 1.25, "scene_radius": 2.0}, name="arm")
        assert (arm.stall_error, arm.stall_time, arm.scene_radius) == (0.35, 1.25, 2.0)


def _state(t: float, error: float, speed: float, position: float = 0.0) -> JointTrajectoryControllerState:
    msg = JointTrajectoryControllerState()
    msg.header.stamp.sec = int(t)
    msg.header.stamp.nanosec = int((t - int(t)) * 1e9)
    msg.joint_names = ["arm_1_joint", "arm_2_joint"]
    msg.feedback.positions = [0.0, position]
    msg.feedback.velocities = [0.0, speed]
    msg.error.positions = [0.01, error]
    return msg


class TestGuardStep:
    @pytest.fixture
    def guard(self, tmp_path: Path):
        from arena_robots.task_server_handlers.reach_pose.moveit import _Guard

        return _Guard(arm=ArmSpec(path=tmp_path, raw=dict(_ARM_ENTRY), name="arm"), native_handle=None)

    def test_a_lagging_joint_that_still_moves_is_not_blocked(self, guard) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _guard_step

        assert [_guard_step(guard, _state(t / 10, 0.5, 0.8)) for t in range(20)] == [""] * 20

    def test_a_stopped_joint_with_a_large_error_is_blocked_after_the_stall_time(self, guard) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _guard_step

        reasons = [_guard_step(guard, _state(t / 10, 0.5, 0.0)) for t in range(7)]
        assert reasons[:5] == [""] * 5
        assert reasons[5].startswith("arm blocked: joint arm_2_joint")

    def test_moving_again_restarts_the_stall_clock(self, guard) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _guard_step

        for t in range(4):
            assert _guard_step(guard, _state(t / 10, 0.5, 0.0)) == ""
        assert _guard_step(guard, _state(0.4, 0.5, 0.8)) == ""
        assert [_guard_step(guard, _state(0.5 + t / 10, 0.5, 0.0)) for t in range(5)] == [""] * 5

    def test_a_non_finite_position_is_reported_at_once(self, guard) -> None:
        from arena_robots.task_server_handlers.reach_pose.moveit import _guard_step

        assert _guard_step(guard, _state(0.0, 0.0, 0.0, position=math.nan)).startswith("arm state not finite: joint arm_2_joint")


class _FakeMoveIt:
    """move_group's parameter, plan and execute endpoints, execution ending only on a trajectory_execution_event stop."""

    def __init__(self, node, bringup, group) -> None:
        from moveit_msgs.action import ExecuteTrajectory, MoveGroup
        from rcl_interfaces.msg import ParameterType, ParameterValue
        from rcl_interfaces.srv import GetParameters
        from std_msgs.msg import String

        ns = bringup.arm_namespace("arm")
        self.executing = False
        self.events: list[str] = []

        def on_parameters(_request, response):
            response.values = [ParameterValue(type=ParameterType.PARAMETER_STRING, string_value='<robot name="r"><link name="tool0"/></robot>')]
            return response

        def on_plan(goal_handle):
            goal_handle.succeed()
            result = MoveGroup.Result()
            result.error_code.val = 1
            return result

        async def on_execute(goal_handle):
            from arena_robots.task_server_handlers import _executor_sleep

            self.executing = True
            while "stop" not in self.events:
                await _executor_sleep(node, 0.02, wall=True)
            goal_handle.abort()
            result = ExecuteTrajectory.Result()
            result.error_code.val = -7
            return result

        node.create_service(GetParameters, str(ns("move_group", "get_parameters")), on_parameters, callback_group=group)
        node.create_subscription(String, str(ns("trajectory_execution_event")), lambda msg: self.events.append(msg.data), 10, callback_group=group)
        self._servers = [
            ActionServer(node, MoveGroup, str(ns("move_action")), execute_callback=on_plan, callback_group=group),
            ActionServer(node, ExecuteTrajectory, str(ns("execute_trajectory")), execute_callback=on_execute, cancel_callback=lambda _handle: rclpy.action.CancelResponse.ACCEPT, callback_group=group),
        ]


@pytest.fixture
def moveit_rig(tmp_path: Path) -> Iterator[tuple[ActionClient, _FakeMoveIt]]:
    import rclpy.action

    from arena_robots.bringup.arm.moveit import MoveItArmBringup
    from arena_robots.Robot import RobotView
    from arena_robots.task_server_handlers.reach_pose.moveit import ReachPoseHandlerMoveIt

    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    ns = f"/reach_pose_{uuid.uuid4().hex[:8]}"
    node = rclpy.create_node("reach_pose_moveit", namespace=ns)
    peer = rclpy.create_node("move_group_fake", namespace=ns)
    bringup = MoveItArmBringup(RobotView(_robot_dir(tmp_path)), ns)
    handler = ReachPoseHandlerMoveIt(bringup, tf_buffer=Buffer(), node=node)
    group = ReentrantCallbackGroup()
    moveit = _FakeMoveIt(peer, bringup, group)
    ActionServer(node, ReachPose, f"{ns}/reach_pose", execute_callback=handler.execute, cancel_callback=handler.on_cancel, callback_group=group)
    client = ActionClient(node, ReachPose, f"{ns}/reach_pose", callback_group=group)
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    executor.add_node(peer)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        assert client.wait_for_server(timeout_sec=30.0)
        yield client, moveit
    finally:
        executor.shutdown()
        peer.destroy_node()
        node.destroy_node()
        if owned:
            rclpy.shutdown()


class TestCancelDuringExecution:
    def test_cancel_stops_the_executing_trajectory(self, moveit_rig) -> None:
        client, moveit = moveit_rig
        handle = _wait(client.send_goal_async(_goal()))
        assert handle.accepted
        deadline = time.monotonic() + 30.0
        while not moveit.executing:
            assert time.monotonic() < deadline, "execution never started"
            time.sleep(0.02)
        _wait(handle.cancel_goal_async())
        result = _wait(handle.get_result_async()).result
        assert moveit.events == ["stop"]
        assert result.status == ReachPose.Result.STATUS_CANCELED
