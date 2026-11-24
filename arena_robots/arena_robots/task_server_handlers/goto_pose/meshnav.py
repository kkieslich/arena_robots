from __future__ import annotations

from typing import TYPE_CHECKING

from action_msgs.msg import GoalStatus
from arena_robots_msgs.action import GotoPose
from geometry_msgs.msg import PoseStamped, TwistStamped
from mbf_msgs.action import MoveBase
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient

from arena_robots.lockstep_beat import LockstepBeat
from arena_robots.task_server_handlers import NativeGoals, TaskHandler, _executor_sleep

if TYPE_CHECKING:
    from arena_rclpy_mixins.Time import TimeNode
    from rclpy.action import CancelResponse
    from rclpy.action.server import ServerGoalHandle
    from tf2_ros import Buffer

    from arena_robots.bringup.mobile.meshnav import MeshNavBringup

_RETRY_BACKOFF_SEC = 0.5
_CONTROLLER_PERIOD_FALLBACK = 0.1


def _translate_outcome(status: int, result: MoveBase.Result) -> tuple[int, str]:
    if status == GoalStatus.STATUS_SUCCEEDED and result.outcome == MoveBase.Result.SUCCESS:
        return GotoPose.Result.STATUS_SUCCEEDED, ""
    if status == GoalStatus.STATUS_CANCELED or result.outcome == MoveBase.Result.CANCELED:
        return GotoPose.Result.STATUS_CANCELED, "MeshNav: canceled during execution"
    return GotoPose.Result.STATUS_ABORTED, f"MeshNav: outcome {result.outcome} {result.message}".strip()


class GotoPoseHandlerMeshNav(TaskHandler[GotoPose.Goal, GotoPose.Feedback, GotoPose.Result]):
    def __init__(self, bringup: MeshNavBringup, *, tf_buffer: Buffer, node: TimeNode) -> None:
        self._bringup = bringup
        self._tf_buffer = tf_buffer
        self._node = node
        self._native_client = ActionClient(node, MoveBase, bringup.native_action_name)
        self._beat = LockstepBeat(node, "nav")
        self._beat_period: float | None = None
        self._native_goals = NativeGoals()
        node.create_subscription(TwistStamped, str(bringup.namespace("move_base_flex", "cmd_vel")), lambda _msg: self._beat.pulse(), 10)

    async def _controller_period(self) -> float:
        """One controller tick in sim seconds, read once from the mesh server."""
        if self._beat_period is not None:
            return self._beat_period
        client = self._node.create_client(GetParameters, str(self._bringup.namespace("move_base_flex", "get_parameters")))
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
        return self._native_goals.cancel(goal_handle)

    async def execute(self, goal_handle: ServerGoalHandle) -> GotoPose.Result:
        arena_goal: GotoPose.Goal = goal_handle.request
        native_goal = MoveBase.Goal()
        native_goal.target_pose = arena_goal.target
        native_goal.planner = self._bringup.planner
        native_goal.controller = "mesh_controller"

        result = GotoPose.Result()
        result.final_pose = PoseStamped()

        while not self._native_client.server_is_ready():
            if not goal_handle.is_active:
                result.status = GotoPose.Result.STATUS_CANCELED
                result.reason = "canceled before action server accepted goal"
                return result
            await _executor_sleep(self._node, 0.1, wall=True)

        await self._beat.acquire(await self._controller_period())
        try:
            return await self._dispatch(goal_handle, native_goal, result)
        finally:
            await self._beat.release()

    async def _dispatch(self, goal_handle: ServerGoalHandle, native_goal: MoveBase.Goal, result: GotoPose.Result) -> GotoPose.Result:
        def _on_feedback(fb_msg: MoveBase.Impl.FeedbackMessage) -> None:
            fb = fb_msg.feedback
            arena_fb = GotoPose.Feedback()
            arena_fb.current_pose = fb.current_pose
            arena_fb.distance_remaining = fb.dist_to_goal
            goal_handle.publish_feedback(arena_fb)

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

            native_handle = await self._native_client.send_goal_async(native_goal, feedback_callback=_on_feedback)
            if native_handle is None or not native_handle.accepted:
                self._node.get_logger().info("meshnav rejected goal; retrying")
                await _executor_sleep(self._node, _RETRY_BACKOFF_SEC)
                continue

            with self._native_goals.tracking(goal_handle, native_handle):
                wrapped = await native_handle.get_result_async()

            status, reason = _translate_outcome(wrapped.status, wrapped.result)
            result.final_pose = wrapped.result.final_pose
            if not goal_handle.is_active:
                result.status = GotoPose.Result.STATUS_CANCELED
                result.reason = "goal preempted by new submit_task"
                return result

            if status == GotoPose.Result.STATUS_SUCCEEDED:
                goal_handle.succeed()
                result.status = status
                result.reason = reason
                return result

            if status == GotoPose.Result.STATUS_CANCELED:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.status = status
                    result.reason = reason
                else:
                    goal_handle.abort()
                    result.status = GotoPose.Result.STATUS_ABORTED
                    result.reason = "MeshNav: goal replaced by another move_base goal"
                return result

            self._node.get_logger().info(f"meshnav aborted goal ({reason}); replanning")
            await _executor_sleep(self._node, _RETRY_BACKOFF_SEC)
