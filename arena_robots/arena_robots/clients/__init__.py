"""arena_robots.clients: action client abstractions for all TaskKinds.

Two surfaces, one shared state machine:
- Awaitable: ``send_goal`` + ``await_result``, suited for notebooks and
  remote tooling where the caller can await the full round-trip.
- Polling: ``is_done`` / ``status`` / ``feedback``, suited for
  the env node's tick-based loop that cannot block awaiting a future.

Both surfaces operate on the same in-flight goal; mixing them is safe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import tf2_ros
    from arena_rclpy_mixins.Async import AsyncNode
    from rclpy.task import Future

    from arena_robots.Robot import RobotView

from arena_rclpy_mixins.registry import ClassRegistry

from arena_robots.task_kinds import TaskKind


class Client(ABC):
    task_kind: ClassVar[TaskKind]

    def __init__(
        self,
        robot: RobotView,
        namespace: str,
        *,
        node: AsyncNode,
        tf_buffer: tf2_ros.Buffer,
    ) -> None:
        self.robot = robot
        self.namespace = namespace
        self.node = node
        self.tf_buffer = tf_buffer

    @abstractmethod
    def action_endpoint(self) -> str: ...

    @abstractmethod
    async def wait_ready(self) -> None: ...

    @abstractmethod
    async def send_goal(self, goal: object) -> object:
        """Send goal; return once accepted by the server (returns GoalHandle)."""
        ...

    @abstractmethod
    async def await_result(self) -> object: ...

    @abstractmethod
    def is_done(self) -> bool | None: ...

    def cancel(self) -> Future | None:
        """Request cancellation of the in-flight goal, the future resolves when the server answers."""
        raise NotImplementedError

    @property
    def status(self) -> int | None:
        return None

    @property
    def reason(self) -> str | None:
        return None

    @property
    def feedback(self) -> object:
        return None


CLIENTS: ClassRegistry[TaskKind, type[Client]] = ClassRegistry()


@CLIENTS.register(TaskKind.GOTO_POSE)
def _load_goto_pose() -> type[Client]:
    from .goto_pose import GotoPoseClient

    return GotoPoseClient


@CLIENTS.register(TaskKind.REACH_POSE)
def _load_reach_pose() -> type[Client]:
    from .reach_pose import ReachPoseClient

    return ReachPoseClient


@CLIENTS.register(TaskKind.PLAY_GESTURE)
def _load_play_gesture() -> type[Client]:
    from .play_gesture import PlayGestureClient

    return PlayGestureClient
