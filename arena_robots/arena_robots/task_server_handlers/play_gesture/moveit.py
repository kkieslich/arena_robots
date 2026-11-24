from __future__ import annotations

from typing import TYPE_CHECKING

import builtin_interfaces.msg
from action_msgs.msg import GoalStatus
from arena_robots_msgs.action import PlayGesture
from arena_simulation_setup.tree.Gesture import GestureIdentifier, GestureSpec
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from arena_robots.lockstep_beat import LockstepBeat
from arena_robots.task_server_handlers import NativeGoals, TaskHandler, _executor_sleep

if TYPE_CHECKING:
    from arena_rclpy_mixins.Time import TimeNode
    from rclpy.action import CancelResponse
    from rclpy.action.server import ServerGoalHandle
    from tf2_ros import Buffer

    from arena_robots.bringup.arm.moveit import MoveItArmBringup
    from arena_robots.caps import ArmSpec

_ARM_BEAT_PERIOD = 0.1


def _translate_fjt_status(error_code: int) -> tuple[int, str]:
    if error_code == FollowJointTrajectory.Result.SUCCESSFUL:
        return PlayGesture.Result.STATUS_SUCCEEDED, ""
    if error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
        return PlayGesture.Result.STATUS_ABORTED, "FJT: PATH_TOLERANCE_VIOLATED"
    if error_code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED:
        return PlayGesture.Result.STATUS_ABORTED, "FJT: GOAL_TOLERANCE_VIOLATED"
    if error_code == FollowJointTrajectory.Result.INVALID_GOAL:
        return PlayGesture.Result.STATUS_ABORTED, "FJT: INVALID_GOAL"
    if error_code == FollowJointTrajectory.Result.INVALID_JOINTS:
        return PlayGesture.Result.STATUS_ABORTED, "FJT: INVALID_JOINTS"
    if error_code == FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP:
        return PlayGesture.Result.STATUS_ABORTED, "FJT: OLD_HEADER_TIMESTAMP"
    return PlayGesture.Result.STATUS_ABORTED, f"FJT: error code {error_code}"


def _build_movegroup_goal(arm: ArmSpec, joints: dict[str, float]) -> MoveGroup.Goal:
    constraints = Constraints()
    for joint_name, position in joints.items():
        jc = JointConstraint()
        jc.joint_name = joint_name
        jc.position = float(position)
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight = 1.0
        constraints.joint_constraints.append(jc)  # pyright: ignore[reportAttributeAccessIssue]

    mg = MoveGroup.Goal()
    mg.request = MotionPlanRequest()
    mg.request.group_name = arm.planning_group or arm.name
    mg.request.goal_constraints.append(constraints)  # pyright: ignore[reportAttributeAccessIssue]
    mg.request.allowed_planning_time = 5.0
    mg.request.num_planning_attempts = 5
    mg.request.max_velocity_scaling_factor = 1.0
    mg.request.max_acceleration_scaling_factor = 1.0
    mg.request.start_state.is_diff = True
    mg.planning_options.plan_only = False
    return mg


def _translate_moveit_status(code: int) -> str:
    if code == MoveItErrorCodes.SUCCESS:
        return ""
    if code == MoveItErrorCodes.NO_IK_SOLUTION:
        return "MoveIt: no IK solution"
    if code == MoveItErrorCodes.PLANNING_FAILED:
        return "MoveIt: planning failed"
    if code == MoveItErrorCodes.INVALID_MOTION_PLAN:
        return "MoveIt: motion plan invalid"
    if code == MoveItErrorCodes.TIMED_OUT:
        return "MoveIt: timed out"
    return f"MoveIt: error code {code}"


def _build_trajectory(gesture: GestureSpec, arm: ArmSpec) -> JointTrajectory:
    chain: list[str] = arm.chain
    named_poses: dict[str, dict[str, float]] = arm.named_poses

    traj = JointTrajectory()
    traj.joint_names = chain

    base_t = gesture.keyframes[0].t
    for kf in gesture.keyframes[1:]:
        pose_joints = named_poses[kf.pose]
        pt = JointTrajectoryPoint()
        pt.positions = [pose_joints.get(j, 0.0) for j in chain]
        pt.velocities = [0.0] * len(chain)
        t_total = kf.t - base_t
        sec = int(t_total)
        pt.time_from_start = builtin_interfaces.msg.Duration(
            sec=sec,
            nanosec=int((t_total - sec) * 1e9),
        )
        traj.points.append(pt)  # pyright: ignore[reportAttributeAccessIssue]

    return traj


class PlayGestureHandlerMoveIt(TaskHandler[PlayGesture.Goal, PlayGesture.Feedback, PlayGesture.Result]):
    def __init__(self, bringup: MoveItArmBringup, *, tf_buffer: Buffer, node: TimeNode) -> None:
        del tf_buffer
        self._bringup = bringup
        self._node = node
        self._mg_clients: dict[str, ActionClient] = {mount: ActionClient(node, MoveGroup, str(bringup.arm_namespace(mount)("move_action"))) for mount in bringup.arms()}
        self._beat = LockstepBeat(node, "gesture")
        self._native_goals = NativeGoals()
        for mount, arm in bringup.arms().items():
            node.create_subscription(
                JointTrajectoryControllerState,
                str(bringup.arm_namespace(mount)(arm.controller, "controller_state")),
                self._on_controller_state,
                10,
            )

    def _on_controller_state(self, msg: JointTrajectoryControllerState) -> None:
        self._beat.pulse(msg.header.stamp)

    def on_cancel(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        return self._native_goals.cancel(goal_handle)

    async def execute(self, goal_handle: ServerGoalHandle) -> PlayGesture.Result:
        arena_goal: PlayGesture.Goal = goal_handle.request
        result = PlayGesture.Result()

        arms = self._bringup.arms()
        if arena_goal.instance:
            if arena_goal.instance not in arms:
                result.status = PlayGesture.Result.STATUS_ABORTED
                result.reason = f"unknown arm instance {arena_goal.instance!r}; available: {sorted(arms)}"
                goal_handle.abort()
                return result
            mount = arena_goal.instance
        else:
            if len(arms) != 1:
                result.status = PlayGesture.Result.STATUS_ABORTED
                result.reason = f"'instance' required; multiple arms available: {sorted(arms)}"
                goal_handle.abort()
                return result
            (mount,) = arms
        arm = arms[mount]
        mg_client = self._mg_clients[mount]

        gesture: GestureSpec | None = arm.gestures.get(arena_goal.gesture)
        if gesture is None:
            try:
                gesture = GestureIdentifier(arena_goal.gesture).resolve_sync()
            except FileNotFoundError:
                goal_handle.abort()
                result.status = PlayGesture.Result.STATUS_UNKNOWN_GESTURE
                result.reason = f"gesture {arena_goal.gesture!r} not found in arm overrides or shared library"
                return result

        missing = gesture.required_poses() - set(arm.named_poses.keys())
        if missing:
            goal_handle.abort()
            result.status = PlayGesture.Result.STATUS_UNSUPPORTED
            result.reason = f"arm '{arm.name}' missing named poses: {sorted(missing)}"
            return result

        first_kf_joints = arm.named_poses[gesture.keyframes[0].pose]
        mg_goal = _build_movegroup_goal(arm, first_kf_joints)

        while not mg_client.server_is_ready():
            if not goal_handle.is_active:
                result.status = PlayGesture.Result.STATUS_CANCELED
                result.reason = "goal canceled by client"
                return result
            await _executor_sleep(self._node, 0.1, wall=True)

        await self._beat.acquire(_ARM_BEAT_PERIOD)
        try:
            return await self._dispatch(goal_handle, mg_client, mg_goal, gesture, arm, mount, result)
        finally:
            await self._beat.release()

    async def _dispatch(
        self,
        goal_handle: ServerGoalHandle,
        mg_client: ActionClient,
        mg_goal: MoveGroup.Goal,
        gesture: GestureSpec,
        arm: ArmSpec,
        mount: str,
        result: PlayGesture.Result,
    ) -> PlayGesture.Result:
        mg_handle = await mg_client.send_goal_async(mg_goal)
        if mg_handle is None or not mg_handle.accepted:
            goal_handle.abort()
            result.status = PlayGesture.Result.STATUS_ABORTED
            result.reason = "MoveIt transition goal rejected"
            return result

        with self._native_goals.tracking(goal_handle, mg_handle):
            mg_wrapped = await mg_handle.get_result_async()
        if not goal_handle.is_active:
            result.status = PlayGesture.Result.STATUS_CANCELED
            result.reason = "goal canceled by client"
            return result
        if mg_wrapped.status == GoalStatus.STATUS_CANCELED:
            goal_handle.canceled()
            result.status = PlayGesture.Result.STATUS_CANCELED
            result.reason = "goal canceled by client"
            return result
        if mg_wrapped.result is None or mg_wrapped.result.error_code.val != MoveItErrorCodes.SUCCESS:
            code = mg_wrapped.result.error_code.val if mg_wrapped.result is not None else -1
            goal_handle.abort()
            result.status = PlayGesture.Result.STATUS_ABORTED
            result.reason = f"transition to first keyframe failed: {_translate_moveit_status(code)}"
            return result

        if len(gesture.keyframes) <= 1:
            goal_handle.succeed()
            result.status = PlayGesture.Result.STATUS_SUCCEEDED
            return result

        traj = _build_trajectory(gesture, arm)

        fjt_action = str(self._bringup.arm_namespace(mount)(arm.controller, "follow_joint_trajectory"))
        fjt_client: ActionClient = ActionClient(self._node, FollowJointTrajectory, fjt_action)

        while not fjt_client.server_is_ready():
            if not goal_handle.is_active:
                result.status = PlayGesture.Result.STATUS_CANCELED
                result.reason = "goal canceled by client"
                return result
            await _executor_sleep(self._node, 0.1, wall=True)

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.status = PlayGesture.Result.STATUS_CANCELED
            result.reason = "goal canceled by client"
            return result

        fjt_goal = FollowJointTrajectory.Goal()
        fjt_goal.trajectory = traj

        fjt_handle = await fjt_client.send_goal_async(fjt_goal)
        if fjt_handle is None or not fjt_handle.accepted:
            goal_handle.abort()
            result.status = PlayGesture.Result.STATUS_ABORTED
            result.reason = "FollowJointTrajectory goal rejected"
            return result

        with self._native_goals.tracking(goal_handle, fjt_handle):
            fjt_wrapped = await fjt_handle.get_result_async()
        if not goal_handle.is_active:
            result.status = PlayGesture.Result.STATUS_CANCELED
            result.reason = "goal canceled by client"
            return result
        if fjt_wrapped.status == GoalStatus.STATUS_CANCELED:
            goal_handle.canceled()
            result.status = PlayGesture.Result.STATUS_CANCELED
            result.reason = "goal canceled by client"
            return result

        if fjt_wrapped.result is not None:
            arena_status, arena_reason = _translate_fjt_status(fjt_wrapped.result.error_code)
        else:
            arena_status = PlayGesture.Result.STATUS_ABORTED
            arena_reason = "FollowJointTrajectory aborted with no result"

        if arena_status == PlayGesture.Result.STATUS_SUCCEEDED:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        result.status = arena_status
        result.reason = arena_reason
        return result
