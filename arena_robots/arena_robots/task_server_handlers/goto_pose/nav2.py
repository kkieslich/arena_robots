from __future__ import annotations

import math
from typing import TYPE_CHECKING

import rclpy.time
import tf2_ros
from action_msgs.msg import GoalStatus
from arena_robots_msgs.action import GotoPose
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient

from arena_robots.lockstep_beat import LockstepBeat
from arena_robots.task_server_handlers import NativeGoals, TaskHandler, _executor_sleep

if TYPE_CHECKING:
    from rclpy.action import CancelResponse
    from rclpy.action.server import ServerGoalHandle
    from rclpy.client import Client
    from rclpy.node import Node
    from tf2_ros import Buffer

    from arena_robots.bringup.mobile.nav2 import Nav2Bringup

_NAV2_RETRY_BACKOFF_SEC = 0.5
_CONTROLLER_PERIOD_FALLBACK = 0.1
_CHECKER_XY = "goal_checker.xy_goal_tolerance"
_CHECKER_YAW = "goal_checker.yaw_goal_tolerance"
_SERVICE_WAIT_S = 5.0
_SETTLE_POLL_S = 0.25
_SETTLE_POLLS = 20
_SETTLE_STEP_M = 0.001
_SETTLE_STEP_RAD = 0.002
_PRECISE_RESENDS = 5


def _yaw(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _translate_nav2_status(nav2_status: int) -> tuple[int, str]:
    if nav2_status == GoalStatus.STATUS_SUCCEEDED:
        return GotoPose.Result.STATUS_SUCCEEDED, ""
    if nav2_status == GoalStatus.STATUS_CANCELED:
        return GotoPose.Result.STATUS_CANCELED, "Nav2: canceled during execution"
    return GotoPose.Result.STATUS_ABORTED, "Nav2: planning or controller aborted"


class GotoPoseHandlerNav2(TaskHandler[GotoPose.Goal, GotoPose.Feedback, GotoPose.Result]):
    def __init__(self, bringup: Nav2Bringup, *, tf_buffer: Buffer, node: Node) -> None:
        self._bringup = bringup
        self._tf_buffer = tf_buffer
        self._node = node
        self._native_client = ActionClient(node, NavigateToPose, bringup.native_action_name)
        self._beat = LockstepBeat(node, "nav")
        self._beat_period: float | None = None
        self._nav2_goals = NativeGoals()
        self._base_frame = bringup.frame + bringup.robot.model_params.base_frame
        self._checker_defaults: dict[str, float] | None = None
        node.create_subscription(Twist, str(bringup.namespace("cmd_vel_nav")), lambda _msg: self._beat.pulse(), 10)

    async def _controller_period(self) -> float:
        """One controller tick in sim seconds, read once from controller_server."""
        if self._beat_period is not None:
            return self._beat_period
        client = self._node.create_client(GetParameters, str(self._bringup.namespace("controller_server", "get_parameters")))
        period = None
        if client.service_is_ready():
            response = await client.call_async(GetParameters.Request(names=["controller_frequency"]))
            assert response is not None
            values = response.values
            if values and values[0].type == ParameterType.PARAMETER_DOUBLE and values[0].double_value > 0.0:
                period = 1.0 / values[0].double_value
        self._node.destroy_client(client)
        if period is None:
            return _CONTROLLER_PERIOD_FALLBACK
        self._beat_period = period
        return period

    def on_cancel(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        return self._nav2_goals.cancel(goal_handle)

    async def execute(self, goal_handle: ServerGoalHandle) -> GotoPose.Result:
        arena_goal: GotoPose.Goal = goal_handle.request
        nav2_goal = NavigateToPose.Goal()
        nav2_goal.pose = arena_goal.target

        result = GotoPose.Result()
        result.final_pose = PoseStamped()

        while not self._native_client.server_is_ready():
            if not goal_handle.is_active:
                result.status = GotoPose.Result.STATUS_CANCELED
                result.reason = "canceled before action server accepted goal"
                return result
            await _executor_sleep(self._node, 0.1, wall=True)

        await self._beat.acquire(await self._controller_period())
        restore = await self._tighten_checker(arena_goal)
        try:
            return await self._dispatch(goal_handle, nav2_goal, result, precise=bool(restore))
        finally:
            if restore:
                await self._set_parameters(restore)
            await self._beat.release()

    async def _service(self, srv_type: type, name: str) -> Client | None:
        """A ready controller_server parameter client, None after `_SERVICE_WAIT_S` s of wall time."""
        client = self._node.create_client(srv_type, str(self._bringup.namespace("controller_server", name)))
        for _ in range(int(_SERVICE_WAIT_S / 0.1)):
            if client.service_is_ready():
                return client
            await _executor_sleep(self._node, 0.1, wall=True)
        self._node.destroy_client(client)
        return None

    async def _tighten_checker(self, goal: GotoPose.Goal) -> dict[str, float]:
        """Set the goal checker to the goal's tolerances where they are tighter, returning the configured values to restore."""
        wanted = {_CHECKER_XY: float(goal.pose_tolerance), _CHECKER_YAW: float(goal.yaw_tolerance)}
        if not any(value > 0.0 for value in wanted.values()):
            return {}
        if self._checker_defaults is None:
            self._checker_defaults = {}
            client = await self._service(GetParameters, "get_parameters")
            if client is None:
                return {}
            response = await client.call_async(GetParameters.Request(names=[_CHECKER_XY, _CHECKER_YAW]))
            self._node.destroy_client(client)
            if response is not None and len(response.values) == 2 and all(v.type == ParameterType.PARAMETER_DOUBLE for v in response.values):
                self._checker_defaults = {_CHECKER_XY: response.values[0].double_value, _CHECKER_YAW: response.values[1].double_value}
        tighter = {name: value for name, value in wanted.items() if 0.0 < value < self._checker_defaults.get(name, 0.0)}
        if not tighter or not await self._set_parameters(tighter):
            return {}
        return {name: self._checker_defaults[name] for name in tighter}

    async def _set_parameters(self, values: dict[str, float]) -> bool:
        client = await self._service(SetParameters, "set_parameters")
        if client is None:
            return False
        request = SetParameters.Request(parameters=[Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value)) for name, value in values.items()])
        response = await client.call_async(request)
        self._node.destroy_client(client)
        return response is not None and all(r.successful for r in response.results)

    def _base_pose(self) -> tuple[float, float, float] | None:
        try:
            t = self._tf_buffer.lookup_transform("map", self._base_frame, rclpy.time.Time()).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        return t.translation.x, t.translation.y, _yaw(t.rotation)

    async def _settled_within(self, goal: GotoPose.Goal) -> bool:
        """Whether the base, once it stops moving, rests within the goal's position and yaw tolerances."""
        pose = self._base_pose()
        for _ in range(_SETTLE_POLLS):
            await _executor_sleep(self._node, _SETTLE_POLL_S, wall=True)
            last, pose = pose, self._base_pose()
            if pose is None:
                return False
            if last is not None and math.hypot(pose[0] - last[0], pose[1] - last[1]) < _SETTLE_STEP_M and abs(math.remainder(pose[2] - last[2], math.tau)) < _SETTLE_STEP_RAD:
                break
        if pose is None:
            return False
        target = goal.target.pose
        yaw = _yaw(target.orientation)
        near = goal.pose_tolerance <= 0.0 or math.hypot(pose[0] - target.position.x, pose[1] - target.position.y) <= goal.pose_tolerance
        aligned = goal.yaw_tolerance <= 0.0 or abs(math.remainder(pose[2] - yaw, math.tau)) <= goal.yaw_tolerance
        return near and aligned

    async def _dispatch(self, goal_handle: ServerGoalHandle, nav2_goal: NavigateToPose.Goal, result: GotoPose.Result, *, precise: bool = False) -> GotoPose.Result:
        def _on_nav2_feedback(fb_msg: NavigateToPose.Impl.FeedbackMessage) -> None:
            fb = fb_msg.feedback
            arena_fb = GotoPose.Feedback()
            arena_fb.current_pose = fb.current_pose
            arena_fb.distance_remaining = fb.distance_remaining
            eta = fb.estimated_time_remaining
            arena_fb.eta_seconds = eta.sec + eta.nanosec * 1e-9
            goal_handle.publish_feedback(arena_fb)

        resends = 0
        while True:
            if not goal_handle.is_active:
                result.status = GotoPose.Result.STATUS_CANCELED
                result.reason = "goal preempted by new submit_task"
                return result
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.status = GotoPose.Result.STATUS_CANCELED
                result.reason = "goal canceled"
                return result

            send_future = self._native_client.send_goal_async(nav2_goal, feedback_callback=_on_nav2_feedback)
            nav2_goal_handle = await send_future

            if nav2_goal_handle is None or not nav2_goal_handle.accepted:
                self._node.get_logger().info("nav2 rejected goal; retrying")
                await _executor_sleep(self._node, _NAV2_RETRY_BACKOFF_SEC)
                continue

            with self._nav2_goals.tracking(goal_handle, nav2_goal_handle):
                wrapped = await nav2_goal_handle.get_result_async()

            arena_status, arena_reason = _translate_nav2_status(wrapped.status)
            if not goal_handle.is_active:
                result.status = GotoPose.Result.STATUS_CANCELED
                result.reason = "goal preempted by new submit_task"
                return result

            if arena_status == GotoPose.Result.STATUS_SUCCEEDED:
                if precise and resends < _PRECISE_RESENDS and not await self._settled_within(goal_handle.request):
                    resends += 1
                    self._node.get_logger().info("nav2 settled outside the goal tolerance; resending")
                    continue
                goal_handle.succeed()
                result.status = arena_status
                result.reason = arena_reason
                return result

            if arena_status == GotoPose.Result.STATUS_CANCELED:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.status = arena_status
                    result.reason = arena_reason
                else:
                    goal_handle.abort()
                    result.status = GotoPose.Result.STATUS_ABORTED
                    result.reason = "Nav2: goal replaced by another navigate_to_pose goal"
                return result

            self._node.get_logger().info("nav2 aborted goal; replanning")
            await _executor_sleep(self._node, _NAV2_RETRY_BACKOFF_SEC)
