from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import rclpy.time
import tf2_ros
from arena_robots_msgs.action import GotoPose
from geometry_msgs.msg import PoseStamped, Quaternion, Twist

from arena_robots.lockstep_beat import LockstepBeat
from arena_robots.task_server_handlers import TaskHandler, _executor_sleep

if TYPE_CHECKING:
    from arena_rclpy_mixins.Time import TimeNode
    from rclpy.action.server import ServerGoalHandle
    from tf2_ros import Buffer

    from arena_robots.bringup.mobile.drl import DrlBringup
    from arena_robots.bringup.mobile.external import ExternalBringup
    from arena_robots.bringup.mobile.none import NoneBringup
    from arena_robots.bringup.mobile.rosnav_rl import RosnavRlBringup

_ARRIVAL_POLL_S = 0.2
_DEFAULT_TOLERANCE_M = 1.0  # matches task_generator's task.episode.goto_pose.tolerance.radius default
_PASSTHROUGH_BEAT_PERIOD = 0.25


class _PassthroughHandler(TaskHandler[GotoPose.Goal, GotoPose.Feedback, GotoPose.Result]):
    """Publish the goal and succeed instantly, for bringups where no stack drives toward it."""

    def __init__(self, bringup: NoneBringup | ExternalBringup | RosnavRlBringup | DrlBringup, *, tf_buffer: Buffer, node: TimeNode) -> None:
        self._bringup = bringup
        self._tf_buffer = tf_buffer
        self._node = node
        self._pub = node.create_publisher(PoseStamped, bringup.goal_topic, 1)

    async def execute(self, goal_handle: ServerGoalHandle) -> GotoPose.Result:
        arena_goal: GotoPose.Goal = goal_handle.request
        self._pub.publish(arena_goal.target)
        goal_handle.succeed()
        result = GotoPose.Result()
        result.status = GotoPose.Result.STATUS_SUCCEEDED
        result.reason = ""
        result.final_pose = arena_goal.target
        return result


class _GoalWindowHandler(_PassthroughHandler):
    """Publish the goal, then hold the action open until arrival,
    supersession, or cancel. Stacks whose control loop Arena does not own
    are beat-gated by proxy (one cmd_vel per window) for the whole motion."""

    _PROXY_BEAT: ClassVar[bool] = True

    def __init__(self, bringup: ExternalBringup | RosnavRlBringup | DrlBringup, *, tf_buffer: Buffer, node: TimeNode) -> None:
        super().__init__(bringup, tf_buffer=tf_buffer, node=node)
        self._base_frame = bringup.frame + bringup.robot.model_params.base_frame
        self._active_token: object | None = None
        self._beat: LockstepBeat | None = None
        if self._PROXY_BEAT:
            beat = LockstepBeat(node, "nav")
            self._beat = beat
            node.create_subscription(Twist, str(bringup.namespace("cmd_vel")), lambda _msg: beat.pulse(), 10)

    async def execute(self, goal_handle: ServerGoalHandle) -> GotoPose.Result:
        arena_goal: GotoPose.Goal = goal_handle.request
        self._pub.publish(arena_goal.target)
        token = object()
        self._active_token = token

        result = GotoPose.Result()
        result.final_pose = arena_goal.target
        tolerance = float(arena_goal.pose_tolerance) or _DEFAULT_TOLERANCE_M

        if self._beat is not None:
            await self._beat.acquire(_PASSTHROUGH_BEAT_PERIOD)
        try:
            while True:
                if self._active_token is not token:
                    goal_handle.abort()
                    result.status = GotoPose.Result.STATUS_CANCELED
                    result.reason = "superseded by new goal"
                    return result
                if not goal_handle.is_active:
                    result.status = GotoPose.Result.STATUS_CANCELED
                    result.reason = "goal preempted"
                    return result
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.status = GotoPose.Result.STATUS_CANCELED
                    result.reason = "goal canceled"
                    return result

                pose = self._current_pose()
                if pose is not None:
                    dx = pose.pose.position.x - arena_goal.target.pose.position.x
                    dy = pose.pose.position.y - arena_goal.target.pose.position.y
                    distance = math.hypot(dx, dy)
                    fb = GotoPose.Feedback()
                    fb.current_pose = pose
                    fb.distance_remaining = distance
                    goal_handle.publish_feedback(fb)
                    if distance <= tolerance and self._yaw_ok(pose, arena_goal):
                        goal_handle.succeed()
                        result.status = GotoPose.Result.STATUS_SUCCEEDED
                        result.reason = ""
                        result.final_pose = pose
                        return result

                await _executor_sleep(self._node, _ARRIVAL_POLL_S, wall=True)
        finally:
            if self._active_token is token:
                self._active_token = None
            if self._beat is not None:
                await self._beat.release()

    def _current_pose(self) -> PoseStamped | None:
        try:
            t = self._tf_buffer.lookup_transform("map", self._base_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        pose = PoseStamped()
        pose.header = t.header
        pose.pose.position.x = t.transform.translation.x
        pose.pose.position.y = t.transform.translation.y
        pose.pose.position.z = t.transform.translation.z
        pose.pose.orientation = t.transform.rotation
        return pose

    @staticmethod
    def _yaw_ok(pose: PoseStamped, arena_goal: GotoPose.Goal) -> bool:
        tol = float(arena_goal.yaw_tolerance)
        if tol <= 0.0:
            return True

        def yaw(q: Quaternion) -> float:
            return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        delta = yaw(pose.pose.orientation) - yaw(arena_goal.target.pose.orientation)
        return abs(math.atan2(math.sin(delta), math.cos(delta))) <= tol


class GotoPoseHandlerNone(_PassthroughHandler):
    pass


class GotoPoseHandlerExternal(_GoalWindowHandler):
    pass


class GotoPoseHandlerRosnavRl(_GoalWindowHandler):
    pass


class GotoPoseHandlerDrl(_GoalWindowHandler):
    """The planner bridge registers its own `planner/<robot>` beat."""

    _PROXY_BEAT = False
