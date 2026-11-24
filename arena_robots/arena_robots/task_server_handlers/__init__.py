"""TaskHandler ABC and shared utilities for task_server handler implementations.

Handler registration is owned by the Bringup subclass via its ``task_handlers``
ClassVar (see ``arena_robots.bringup.Bringup``); this module only exposes the
shared base class and the ``_executor_sleep`` helper.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

from rclpy.action import CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.task import Future

if TYPE_CHECKING:
    from rclpy.action.client import ClientGoalHandle
    from tf2_ros import Buffer

    from arena_robots.bringup import Bringup


class TaskHandler[GoalT, FeedbackT, ResultT](ABC):
    @abstractmethod
    def __init__(self, bringup: Bringup, *, tf_buffer: Buffer, node: Node) -> None: ...

    @abstractmethod
    async def execute(self, goal_handle: ServerGoalHandle) -> ResultT: ...

    def on_cancel(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        return CancelResponse.ACCEPT


class NativeGoals:
    """The native action goal each server goal waits on, so that a cancel of the server goal reaches it."""

    def __init__(self) -> None:
        self._goals: dict[bytes, ClientGoalHandle] = {}

    def cancel(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        native_handle = self._goals.get(bytes(goal_handle.goal_id.uuid))
        if native_handle is not None:
            native_handle.cancel_goal_async()
        return CancelResponse.ACCEPT

    def others(self, goal_handle: ServerGoalHandle) -> list[ClientGoalHandle]:
        key = bytes(goal_handle.goal_id.uuid)
        return [native_handle for uuid, native_handle in self._goals.items() if uuid != key]

    @contextlib.contextmanager
    def tracking(self, goal_handle: ServerGoalHandle, native_handle: ClientGoalHandle) -> Iterator[None]:
        key = bytes(goal_handle.goal_id.uuid)
        self._goals[key] = native_handle
        if goal_handle.is_cancel_requested:
            native_handle.cancel_goal_async()
        try:
            yield
        finally:
            self._goals.pop(key, None)


async def _executor_sleep(node: Node, seconds: float, *, wall: bool = False) -> None:
    """Timer-backed sleep that yields to rclpy's executor. Works inside action
    server callbacks (which run under rclpy.spin, not asyncio).

    ``wall=True`` uses a wall-clock timer so the sleep still ticks while sim
    time is paused, use it for readiness/discovery polling, not for retry
    rate-limiting where sim-time semantics are preferred.
    """
    fut: Future = Future()

    def _fire():
        if not fut.done():
            fut.set_result(None)

    timer = node.create_timer(seconds, _fire, clock=Clock() if wall else None)
    try:
        await fut
    finally:
        node.destroy_timer(timer)


async def _await_within(node: Node, future: Future, seconds: float) -> None:
    """Wait for `future`, giving up after `seconds` of wall time."""
    gate: Future = Future()

    def _release(*_args: object) -> None:
        if not gate.done():
            gate.set_result(None)

    future.add_done_callback(_release)
    timer = node.create_timer(seconds, _release, clock=Clock())
    try:
        await gate
    finally:
        node.destroy_timer(timer)
