from __future__ import annotations

from typing import TYPE_CHECKING

import tf2_ros
from arena_robots_msgs.action import ReachPose
from rclpy.task import Future

from arena_robots.clients import Client
from arena_robots.Robot import RobotView
from arena_robots.task_kinds import TaskKind, action_type, endpoint

if TYPE_CHECKING:
    from arena_rclpy_mixins.Async import AsyncNode


class ReachPoseClient(Client):
    task_kind = TaskKind.REACH_POSE

    def __init__(self, robot: RobotView, namespace: str, *, node: AsyncNode, tf_buffer: tf2_ros.Buffer) -> None:
        super().__init__(robot, namespace, node=node, tf_buffer=tf_buffer)
        self._action = node.create_action_client_wrapper(
            action_type(self.task_kind),
            self.action_endpoint(),
        )
        self._goal_handle = None
        self._result = None
        self._status = None
        self._reason: str | None = None
        self._feedback = None
        self._done = None
        self._result_future = None

    def action_endpoint(self) -> str:
        return endpoint(self.namespace, self.task_kind)

    async def wait_ready(self) -> None:
        await self._action.ensure()

    def reset(self) -> None:
        """Forget the previous goal's result, so a phase still preparing its goal reads as running."""
        self._done = False
        self._result = None
        self._status = None
        self._reason = None
        self._feedback = None
        self._result_future = None

    async def send_goal(self, goal: object) -> object:
        self.reset()
        self._goal_handle = await self._action.send_goal(goal, feedback_callback=self._on_feedback)
        self._result_future = self._goal_handle.get_result_async()
        self._result_future.add_done_callback(self._on_result)
        return self._goal_handle

    async def await_result(self) -> object:
        if self._goal_handle is None:
            raise RuntimeError("No goal in flight; call send_goal first.")
        result_response = await self._action.await_result(self._goal_handle)
        if not isinstance(result_response, ReachPose.Impl.GetResultService.Response):
            raise TypeError(f"unexpected result type {type(result_response).__name__} from {self.action_endpoint()!r}")
        self._result = result_response.result
        if result_response.result is not None:
            self._status = result_response.result.status
            self._reason = result_response.result.reason
        else:
            self._status = None
            self._reason = None
        self._done = True
        return self._result

    def is_done(self) -> bool | None:
        return self._done

    def cancel(self) -> Future | None:
        return None if self._goal_handle is None else self._goal_handle.cancel_goal_async()

    @property
    def status(self) -> int | None:
        return self._status

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def feedback(self) -> object:
        return self._feedback

    def _on_feedback(self, msg: ReachPose.Impl.FeedbackMessage) -> None:
        self._feedback = msg.feedback

    def _on_result(self, future: Future) -> None:
        if future is not self._result_future:
            return
        try:
            result_response = future.result()
        except Exception:
            self._done = True
            return
        if result_response is None:
            self._done = True
            return
        self._result = result_response.result
        if result_response.result is not None:
            self._status = result_response.result.status
            self._reason = result_response.result.reason
        else:
            self._status = None
            self._reason = None
        self._done = True
