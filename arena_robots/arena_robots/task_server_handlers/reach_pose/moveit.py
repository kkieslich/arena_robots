from __future__ import annotations

import copy
import dataclasses
import math
import time
import xml.etree.ElementTree as ET
from functools import partial
from typing import TYPE_CHECKING

import rclpy.time
from action_msgs.msg import GoalStatus
from arena_robots_msgs.action import ReachPose
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningScene,
    PlanningSceneComponents,
    PositionConstraint,
)
from moveit_msgs.srv import GetPlanningScene
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from tf2_ros import TransformException

from arena_robots.lockstep_beat import LockstepBeat
from arena_robots.task_server_handlers import NativeGoals, TaskHandler, _executor_sleep

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rclpy.action import CancelResponse
    from rclpy.action.client import ClientGoalHandle
    from rclpy.action.server import ServerGoalHandle
    from rclpy.client import Client
    from rclpy.node import Node
    from tf2_ros import Buffer

    from arena_robots.bringup.arm.moveit import MoveItArmBringup
    from arena_robots.caps import ArmSpec

_DEFAULT_POSITION_TOLERANCE = 0.01
_DEFAULT_ORIENTATION_TOLERANCE = 0.1
_DEFAULT_PLANNING_TIME = 5.0
_ARM_BEAT_PERIOD = 0.1
_PILZ = "pilz_industrial_motion_planner"
_MOTIONS = ("", "free", "linear")
_SERVICE_TIMEOUT = 10.0
_START_CLAMP = 0.05


class _ReachAbort(Exception):
    pass


@dataclasses.dataclass
class _Guard:
    arm: ArmSpec
    native_handle: ClientGoalHandle
    stall_since: float | None = None
    violation: str = ""


def _quat_mul(a: Pose, b: Pose) -> tuple[float, float, float, float]:
    qa, qb = a.orientation, b.orientation
    return (
        qa.w * qb.x + qa.x * qb.w + qa.y * qb.z - qa.z * qb.y,
        qa.w * qb.y - qa.x * qb.z + qa.y * qb.w + qa.z * qb.x,
        qa.w * qb.z + qa.x * qb.y - qa.y * qb.x + qa.z * qb.w,
        qa.w * qb.w - qa.x * qb.x - qa.y * qb.y - qa.z * qb.z,
    )


def _compose_offset(tool: Pose, offset: Pose) -> Pose:
    """The pose `offset` applied to `tool`: positions add, the offset rotation multiplies on the left."""
    out = Pose()
    out.position.x = tool.position.x + offset.position.x
    out.position.y = tool.position.y + offset.position.y
    out.position.z = tool.position.z + offset.position.z
    out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w = _quat_mul(offset, tool)
    return out


def _descendant_links(urdf: str, root_link: str) -> frozenset[str]:
    children: dict[str, list[str]] = {}
    for joint in ET.fromstring(urdf).iter("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is not None and child is not None:
            children.setdefault(parent.get("link", ""), []).append(child.get("link", ""))
    found = {root_link}
    frontier = [root_link]
    while frontier:
        for child in children.get(frontier.pop(), []):
            if child not in found:
                found.add(child)
                frontier.append(child)
    return frozenset(found)


def _arm_links(urdf: str, root_link: str, chain: Sequence[str]) -> frozenset[str]:
    """The tip subtree plus the link every chain joint drives, the whole arm from below its base."""
    child_of: dict[str, str] = {}
    for joint in ET.fromstring(urdf).iter("joint"):
        child = joint.find("child")
        if child is not None:
            child_of[joint.get("name", "")] = child.get("link", "")
    links = set(_descendant_links(urdf, root_link))
    links.update(child_of[joint] for joint in chain if joint in child_of)
    return frozenset(links)


def _joint_bounds(urdf: str) -> dict[str, tuple[float, float]]:
    """Position bounds of every joint the URDF limits."""
    bounds: dict[str, tuple[float, float]] = {}
    for joint in ET.fromstring(urdf).iter("joint"):
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            continue
        bounds[joint.get("name", "")] = (float(limit.get("lower", "0")), float(limit.get("upper", "0")))
    return bounds


def _allow_contact(acm: AllowedCollisionMatrix, object_id: str, links: frozenset[str]) -> AllowedCollisionMatrix:
    """A copy of `acm` where `object_id` may collide with exactly `links` and every other entry is kept."""
    names = list(acm.entry_names)
    rows = [list(entry.enabled) for entry in acm.entry_values]
    for name in (object_id, *sorted(links)):
        if name not in names:
            names.append(name)
            for row in rows:
                row.append(False)
            rows.append([False] * len(names))
    index = names.index(object_id)
    for i, name in enumerate(names):
        allowed = name in links
        rows[index][i] = allowed
        rows[i][index] = allowed
    rows[index][index] = False
    out = copy.deepcopy(acm)
    out.entry_names = names
    out.entry_values = [AllowedCollisionEntry(enabled=row) for row in rows]
    return out


def _guard_step(guard: _Guard, msg: JointTrajectoryControllerState) -> str:
    """The violation one controller state message shows for `guard`, empty while the arm is fine."""
    for name, position in zip(msg.joint_names, msg.feedback.positions, strict=False):
        if not math.isfinite(position):
            return f"arm state not finite: joint {name} position {position}"
    errors = [(abs(error), name, index) for index, (name, error) in enumerate(zip(msg.joint_names, msg.error.positions, strict=False))]
    if not errors:
        return ""
    worst, joint, index = max(errors)
    if not math.isfinite(worst):
        return f"arm state not finite: joint {joint} error {worst}"
    velocities = msg.feedback.velocities
    speed = abs(velocities[index]) if index < len(velocities) else 0.0
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if worst <= guard.arm.stall_error or speed > guard.arm.stall_speed:
        guard.stall_since = None
    elif guard.stall_since is None:
        guard.stall_since = stamp
    elif stamp - guard.stall_since >= guard.arm.stall_time:
        return f"arm blocked: joint {joint} error {worst:.3f} rad above {guard.arm.stall_error} at {speed:.3f} rad/s for {guard.arm.stall_time} s"
    return ""


def _translate_moveit_status(code: int) -> tuple[int, str]:
    if code == MoveItErrorCodes.SUCCESS:
        return ReachPose.Result.STATUS_SUCCEEDED, ""
    if code == MoveItErrorCodes.NO_IK_SOLUTION:
        return ReachPose.Result.STATUS_NO_IK, "MoveIt: no IK solution"
    if code == MoveItErrorCodes.PLANNING_FAILED:
        return ReachPose.Result.STATUS_NO_PLAN, "MoveIt: planning failed"
    if code == MoveItErrorCodes.INVALID_MOTION_PLAN:
        return ReachPose.Result.STATUS_NO_PLAN, "MoveIt: motion plan invalid"
    if code == MoveItErrorCodes.TIMED_OUT:
        return ReachPose.Result.STATUS_TIMEOUT, "MoveIt: timed out"
    return ReachPose.Result.STATUS_ABORTED, f"MoveIt: error code {code}"


class ReachPoseHandlerMoveIt(TaskHandler[ReachPose.Goal, ReachPose.Feedback, ReachPose.Result]):
    def __init__(self, bringup: MoveItArmBringup, *, tf_buffer: Buffer, node: Node) -> None:
        self._bringup = bringup
        self._tf_buffer = tf_buffer
        self._node = node
        self._tf_prefix = bringup.frame
        self._native_clients: dict[str, ActionClient] = {mount: ActionClient(node, MoveGroup, str(bringup.arm_namespace(mount)("move_action"))) for mount in bringup.arms()}
        self._exec_clients: dict[str, ActionClient] = {mount: ActionClient(node, ExecuteTrajectory, str(bringup.arm_namespace(mount)("execute_trajectory"))) for mount in bringup.arms()}
        self._stop_pubs = {mount: node.create_publisher(String, str(bringup.arm_namespace(mount)("trajectory_execution_event")), 10) for mount in bringup.arms()}
        self._executing: dict[bytes, str] = {}
        self._beat = LockstepBeat(node, "reach")
        self._native_goals = NativeGoals()
        self._guards: dict[str, _Guard] = {}
        self._scene_clients = {mount: node.create_client(GetPlanningScene, str(bringup.arm_namespace(mount)("get_planning_scene"))) for mount in bringup.arms()}
        self._param_clients = {mount: node.create_client(GetParameters, str(bringup.arm_namespace(mount)("move_group", "get_parameters"))) for mount in bringup.arms()}
        self._models: dict[str, tuple[frozenset[str], dict[str, tuple[float, float]]]] = {}
        self._positions: dict[str, dict[str, float]] = {}
        for mount, arm in bringup.arms().items():
            node.create_subscription(
                JointTrajectoryControllerState,
                str(bringup.arm_namespace(mount)(arm.controller, "controller_state")),
                partial(self._on_controller_state, mount),
                10,
            )

    def _on_controller_state(self, mount: str, msg: JointTrajectoryControllerState) -> None:
        self._beat.pulse(msg.header.stamp)
        self._positions[mount] = dict(zip(msg.joint_names, msg.feedback.positions, strict=False))
        guard = self._guards.get(mount)
        if guard is None or guard.violation:
            return
        reason = _guard_step(guard, msg)
        if reason:
            guard.violation = reason
            self._halt(mount)
            guard.native_handle.cancel_goal_async()

    def _halt(self, mount: str) -> None:
        """Stop the trajectory MoveIt executes on `mount`."""
        self._stop_pubs[mount].publish(String(data="stop"))

    def _tool_pose(self, arm: ArmSpec, frame: str) -> Pose | None:
        try:
            transform = self._tf_buffer.lookup_transform(frame, self._tf_prefix + arm.tip_link, rclpy.time.Time())
        except TransformException:
            return None
        pose = Pose()
        pose.position.x = transform.transform.translation.x
        pose.position.y = transform.transform.translation.y
        pose.position.z = transform.transform.translation.z
        pose.orientation = transform.transform.rotation
        return pose

    async def _call[Req, Res](self, client: Client, request: Req, what: str) -> Res:
        deadline = time.monotonic() + _SERVICE_TIMEOUT
        while not client.service_is_ready():
            if time.monotonic() > deadline:
                raise _ReachAbort(f"{what} service unavailable")
            await _executor_sleep(self._node, 0.1, wall=True)
        response = await client.call_async(request)
        if response is None:
            raise _ReachAbort(f"{what} returned no response")
        return response

    async def _model(self, mount: str, arm: ArmSpec) -> tuple[frozenset[str], dict[str, tuple[float, float]]]:
        """Links that may touch a contact object and the joint bounds, both from move_group's URDF."""
        if mount not in self._models:
            request = GetParameters.Request()
            request.names = ["robot_description"]
            response = await self._call(self._param_clients[mount], request, "move_group get_parameters")
            urdf = response.values[0].string_value if response.values else ""
            if not urdf:
                raise _ReachAbort("move_group has no robot_description")
            self._models[mount] = (_arm_links(urdf, self._tf_prefix + arm.tip_link, arm.chain), _joint_bounds(urdf))
        return self._models[mount]

    def _start_state(self, mount: str, arm: ArmSpec) -> JointState | None:
        """The current arm state with joints clamped into their bounds, None when a joint is far outside or unknown."""
        positions = self._positions.get(mount)
        model = self._models.get(mount)
        if not positions or model is None:
            return None
        bounds = model[1]
        names: list[str] = []
        values: list[float] = []
        for joint in arm.chain:
            value = positions.get(joint)
            if value is None:
                return None
            low, high = bounds.get(joint, (value, value))
            clamped = min(max(value, low), high)
            if abs(clamped - value) > _START_CLAMP:
                return None
            names.append(joint)
            values.append(clamped)
        return JointState(name=names, position=values)

    async def _contact_scene(self, mount: str, arm: ArmSpec, object_ids: Sequence[str]) -> PlanningScene:
        links, _ = await self._model(mount, arm)
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        response = await self._call(self._scene_clients[mount], request, "move_group get_planning_scene")
        scene = PlanningScene()
        scene.is_diff = True
        acm = response.scene.allowed_collision_matrix
        for object_id in object_ids:
            acm = _allow_contact(acm, object_id, links)
        scene.allowed_collision_matrix = acm
        return scene

    def on_cancel(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        mount = self._executing.get(bytes(goal_handle.goal_id.uuid))
        if mount is not None:
            self._halt(mount)
        return self._native_goals.cancel(goal_handle)

    async def execute(self, goal_handle: ServerGoalHandle) -> ReachPose.Result:
        arena_goal: ReachPose.Goal = goal_handle.request
        result = ReachPose.Result()

        arms = self._bringup.arms()
        if arena_goal.instance:
            if arena_goal.instance not in arms:
                result.status = ReachPose.Result.STATUS_ABORTED
                result.reason = f"unknown arm instance {arena_goal.instance!r}; available: {sorted(arms)}"
                return result
            mount = arena_goal.instance
        else:
            if len(arms) != 1:
                result.status = ReachPose.Result.STATUS_ABORTED
                result.reason = f"'instance' required; multiple arms available: {sorted(arms)}"
                return result
            (mount,) = arms
        arm = arms[mount]
        native_client = self._native_clients[mount]

        if arena_goal.motion not in _MOTIONS or (arena_goal.motion == "linear" and arena_goal.named_target):
            goal_handle.abort()
            result.status = ReachPose.Result.STATUS_ABORTED
            result.reason = f"motion {arena_goal.motion!r} unsupported{' with a named target' if arena_goal.named_target else ''}, known: {list(_MOTIONS)}"
            return result

        if arena_goal.relative and not arena_goal.named_target:
            frame = arena_goal.target.header.frame_id
            tool = self._tool_pose(arm, frame)
            if tool is None:
                goal_handle.abort()
                result.status = ReachPose.Result.STATUS_ABORTED
                result.reason = f"relative reach: no transform from {self._tf_prefix + arm.tip_link!r} to {frame!r}"
                return result
            arena_goal = copy.deepcopy(arena_goal)
            arena_goal.target.pose = _compose_offset(tool, arena_goal.target.pose)

        mg_goal = self._build_goal(arena_goal, arm)
        if mg_goal is None:
            goal_handle.abort()
            result.status = ReachPose.Result.STATUS_UNKNOWN_NAMED
            result.reason = f"named pose {arena_goal.named_target!r} not in caps/arm.yaml.named_poses"
            return result

        while not (native_client.server_is_ready() and self._exec_clients[mount].server_is_ready()):
            if not goal_handle.is_active:
                result.status = ReachPose.Result.STATUS_CANCELED
                result.reason = "goal canceled by client"
                return result
            await _executor_sleep(self._node, 0.1, wall=True)

        await self._beat.acquire(_ARM_BEAT_PERIOD)
        try:
            await self._model(mount, arm)
            start = self._start_state(mount, arm)
            if start is not None:
                mg_goal.request.start_state.is_diff = False
                mg_goal.request.start_state.joint_state = start
            if arena_goal.contact:
                mg_goal.planning_options.planning_scene_diff = await self._contact_scene(mount, arm, list(arena_goal.contact))
            frame = self._tf_prefix + arm.base_link if arena_goal.named_target else arena_goal.target.header.frame_id
            return await self._dispatch(goal_handle, native_client, mg_goal, result, mount, arm, frame)
        except _ReachAbort as exc:
            goal_handle.abort()
            result.status = ReachPose.Result.STATUS_ABORTED
            result.reason = str(exc)
            return result
        finally:
            await self._beat.release()

    @staticmethod
    def _publish_feedback(goal_handle: ServerGoalHandle, state: str, progress: float) -> None:
        feedback = ReachPose.Feedback()
        feedback.state = state
        feedback.progress = progress
        goal_handle.publish_feedback(feedback)

    async def _dispatch(self, goal_handle: ServerGoalHandle, native_client: ActionClient, mg_goal: MoveGroup.Goal, result: ReachPose.Result, mount: str, arm: ArmSpec, frame: str) -> ReachPose.Result:
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.status = ReachPose.Result.STATUS_CANCELED
            result.reason = "goal canceled by client"
            return result

        self._publish_feedback(goal_handle, "PLANNING", 0.0)
        plan_handle = await native_client.send_goal_async(mg_goal)
        if plan_handle is None or not plan_handle.accepted:
            goal_handle.abort()
            result.status = ReachPose.Result.STATUS_ABORTED
            result.reason = "MoveIt MotionPlanRequest aborted"
            return result
        with self._native_goals.tracking(goal_handle, plan_handle):
            planned = await plan_handle.get_result_async()
        if not goal_handle.is_active:
            result.status = ReachPose.Result.STATUS_CANCELED
            result.reason = "goal canceled by client"
            return result
        status, reason = self._outcome(planned)
        if status != ReachPose.Result.STATUS_SUCCEEDED:
            return self._finish(goal_handle, result, status, reason, arm, frame)

        execute = ExecuteTrajectory.Goal()
        execute.trajectory = planned.result.planned_trajectory
        exec_handle = await self._exec_clients[mount].send_goal_async(execute)
        if exec_handle is None or not exec_handle.accepted:
            goal_handle.abort()
            result.status = ReachPose.Result.STATUS_ABORTED
            result.reason = "MoveIt trajectory execution refused"
            return result

        self._publish_feedback(goal_handle, "EXECUTING", 0.5)
        guard = _Guard(arm=arm, native_handle=exec_handle)
        self._guards[mount] = guard
        key = bytes(goal_handle.goal_id.uuid)
        self._executing[key] = mount
        if goal_handle.is_cancel_requested:
            self._halt(mount)
        try:
            with self._native_goals.tracking(goal_handle, exec_handle):
                wrapped = await exec_handle.get_result_async()
        finally:
            self._executing.pop(key, None)
            self._guards.pop(mount, None)
        if guard.violation and goal_handle.is_active:
            return self._finish(goal_handle, result, ReachPose.Result.STATUS_ABORTED, guard.violation, arm, frame)
        if not goal_handle.is_active:
            result.status = ReachPose.Result.STATUS_CANCELED
            result.reason = "goal canceled by client"
            return result
        if goal_handle.is_cancel_requested:
            return self._finish(goal_handle, result, ReachPose.Result.STATUS_CANCELED, "goal canceled by client", arm, frame)
        status, reason = self._outcome(wrapped)
        return self._finish(goal_handle, result, status, reason, arm, frame)

    @staticmethod
    def _outcome(wrapped: MoveGroup.Impl.GetResultService.Response | ExecuteTrajectory.Impl.GetResultService.Response) -> tuple[int, str]:
        """The arena status and reason of a MoveIt action result."""
        if wrapped.status == GoalStatus.STATUS_CANCELED:
            return ReachPose.Result.STATUS_CANCELED, "goal canceled by client"
        if wrapped.result is None:
            return ReachPose.Result.STATUS_ABORTED, "MoveIt MotionPlanRequest aborted"
        return _translate_moveit_status(wrapped.result.error_code.val)

    def _finish(self, goal_handle: ServerGoalHandle, result: ReachPose.Result, status: int, reason: str, arm: ArmSpec, frame: str) -> ReachPose.Result:
        if status == ReachPose.Result.STATUS_SUCCEEDED:
            self._publish_feedback(goal_handle, "DONE", 1.0)
            goal_handle.succeed()
        elif status == ReachPose.Result.STATUS_CANCELED:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        result.status = status
        result.reason = reason
        self._fill_final_pose(result, arm, frame)
        return result

    def _fill_final_pose(self, result: ReachPose.Result, arm: ArmSpec, frame: str) -> None:
        pose = self._tool_pose(arm, frame)
        if pose is not None:
            result.final_pose = PoseStamped()
            result.final_pose.header.frame_id = frame
            result.final_pose.pose = pose

    def _build_goal(self, arena_goal: ReachPose.Goal, arm: ArmSpec) -> MoveGroup.Goal | None:
        constraints = Constraints()

        if arena_goal.named_target:
            joints = arm.named_poses.get(arena_goal.named_target)
            if joints is None:
                return None
            for joint_name, position in joints.items():
                jc = JointConstraint()
                jc.joint_name = joint_name
                jc.position = float(position)
                jc.tolerance_above = 0.01
                jc.tolerance_below = 0.01
                jc.weight = 1.0
                constraints.joint_constraints.append(jc)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            pos_tol = arena_goal.position_tolerance or _DEFAULT_POSITION_TOLERANCE
            ori_tol = arena_goal.orientation_tolerance or _DEFAULT_ORIENTATION_TOLERANCE

            pc = PositionConstraint()
            pc.header = arena_goal.target.header
            pc.link_name = self._tf_prefix + arm.tip_link
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.SPHERE
            prim.dimensions = [max(float(pos_tol), 0.001)]
            pc.constraint_region.primitives.append(prim)  # pyright: ignore[reportAttributeAccessIssue]
            pc.constraint_region.primitive_poses.append(arena_goal.target.pose)  # pyright: ignore[reportAttributeAccessIssue]
            pc.weight = 1.0
            constraints.position_constraints.append(pc)  # pyright: ignore[reportAttributeAccessIssue]

            oc = OrientationConstraint()
            oc.header = arena_goal.target.header
            oc.link_name = self._tf_prefix + arm.tip_link
            oc.orientation = arena_goal.target.pose.orientation
            oc.absolute_x_axis_tolerance = float(ori_tol)
            oc.absolute_y_axis_tolerance = float(ori_tol)
            oc.absolute_z_axis_tolerance = float(ori_tol)
            oc.weight = 1.0
            constraints.orientation_constraints.append(oc)  # pyright: ignore[reportAttributeAccessIssue]

        mg_goal = MoveGroup.Goal()
        mg_goal.request = MotionPlanRequest()
        mg_goal.request.group_name = arm.planning_group or arm.name
        mg_goal.request.goal_constraints.append(constraints)  # pyright: ignore[reportAttributeAccessIssue]
        mg_goal.request.allowed_planning_time = float(arena_goal.planning_time or _DEFAULT_PLANNING_TIME)
        mg_goal.request.num_planning_attempts = 5
        speed = float(arena_goal.speed or 1.0)
        mg_goal.request.max_velocity_scaling_factor = speed
        mg_goal.request.max_acceleration_scaling_factor = speed
        if arena_goal.motion == "linear":
            mg_goal.request.pipeline_id = _PILZ
            mg_goal.request.planner_id = "LIN"
        # is_diff=true means "use the current robot state as start; ignore the (empty) joint_state I'm sending".
        # Without this, MoveIt warns every goal that it's ignoring our supplied start state.
        mg_goal.request.start_state.is_diff = True
        ws = arm.workspace
        if ws is not None and ws.get("type") == "box":
            ws_min, ws_max = ws["min"], ws["max"]
            if not isinstance(ws_min, list) or not isinstance(ws_max, list):
                raise ValueError(f"{arm.path}: arm '{arm.name}' workspace 'min' and 'max' must be lists")
            mg_goal.request.workspace_parameters.header.frame_id = self._tf_prefix + str(ws.get("frame", "base_link"))
            mg_goal.request.workspace_parameters.min_corner.x = float(ws_min[0])
            mg_goal.request.workspace_parameters.min_corner.y = float(ws_min[1])
            mg_goal.request.workspace_parameters.min_corner.z = float(ws_min[2])
            mg_goal.request.workspace_parameters.max_corner.x = float(ws_max[0])
            mg_goal.request.workspace_parameters.max_corner.y = float(ws_max[1])
            mg_goal.request.workspace_parameters.max_corner.z = float(ws_max[2])
        mg_goal.planning_options.plan_only = True
        return mg_goal
