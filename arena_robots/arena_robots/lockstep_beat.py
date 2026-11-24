"""Lockstep heartbeat for per-robot producers.

acquire registers a hard channel for this robot, pulse republishes a coverage
stamp, release deregisters. A wall-clock watchdog covers legitimately silent
phases (MoveIt planning, nav2 recoveries) with keepalive stamps after `grace_s`
that pace the sim at wall rate, `grace_s=None` lets silence hold the sim.
"""

from __future__ import annotations

import contextlib
import typing

from arena_rclpy_mixins.Time import ClockEcho, TimeNode
from arena_runtime_msgs.msg import LockstepChannel, LockstepHeartbeat, LockstepRegistration
from arena_runtime_msgs.srv import LockstepRegister
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.clock import Clock
from rclpy.duration import Duration
from rclpy.exceptions import InvalidHandle
from rclpy.time import Time

from arena_robots.task_server_handlers import _executor_sleep

_REGISTER_SERVICE = "/arena/sim_lifecycle/lockstep/register"
_REGISTER_TIMEOUT = 3.0
_WATCHDOG_PERIOD = 0.2
_GRACE_S = 2.0


class LockstepBeat:
    def __init__(
        self,
        node: TimeNode,
        kind: str,
        *,
        grace_s: float | None = _GRACE_S,
        sleep: typing.Callable[[float], typing.Awaitable[None]] | None = None,
    ) -> None:
        ns = node.get_namespace().rstrip("/")
        self._node = node
        self._grace_s = grace_s
        self._sleep = sleep
        self._name = f"{kind}/{ns.rsplit('/', 1)[-1]}"
        self._topic = f"{ns}/lockstep/{kind}"
        self._env = ns.rsplit("/", 1)[0] or "/"
        self._pub: ClockEcho[LockstepHeartbeat] = ClockEcho(node, node.create_publisher(LockstepHeartbeat, self._topic, 10))
        self._client = node.create_client(LockstepRegister, _REGISTER_SERVICE)
        self._active = 0
        self._registered = False
        self._starved = False
        self._wall = Clock()
        self._last_pulse = self._wall.now()
        self._period_ns = 0
        self._horizon_ns = 0
        self._keepalive_at = self._wall.now()
        if grace_s is not None:
            node.create_timer(_WATCHDOG_PERIOD, self._watchdog, clock=self._wall)

    async def acquire(self, period_s: float) -> None:
        """Register the channel on the first concurrent acquisition."""
        self._active += 1
        if self._active > 1:
            return
        self._starved = False
        self._last_pulse = self._wall.now()
        self._period_ns = round(period_s * 1e9)
        self._registered = await self._apply(
            [
                LockstepChannel(
                    name=self._name,
                    topic=self._topic,
                    type="arena_runtime_msgs/msg/LockstepHeartbeat",
                    period_s=period_s,
                    hard=True,
                )
            ]
        )
        if self._registered and self._active == 0:
            # released while the register call was in flight
            self._registered = False
            await self._apply([])

    async def release(self) -> None:
        """Deregister when the last concurrent acquisition ends."""
        self._active -= 1
        if self._active > 0 or not self._registered:
            return
        self._registered = False
        with contextlib.suppress(InvalidHandle):
            await self._apply([])

    def pulse(self, stamp: TimeMsg | None = None) -> None:
        """Publish one coverage stamp, defaulting to current sim time."""
        if self._active <= 0:
            return
        self._last_pulse = self._wall.now()
        self._starved = False
        msg = LockstepHeartbeat()
        msg.header.stamp = stamp if stamp is not None else self._node.sim_time.to_msg()
        self._pub.publish(msg)

    def _watchdog(self) -> None:
        if self._active <= 0 or not self._registered or self._grace_s is None:
            return
        wall = self._wall.now()
        silent = (wall - self._last_pulse).nanoseconds * 1e-9
        if silent < self._grace_s:
            return
        sim_ns = self._node.sim_time.to_nanoseconds()
        if not self._starved:
            self._starved = True
            self._horizon_ns = sim_ns
            self._keepalive_at = wall - Duration(seconds=_WATCHDOG_PERIOD)
            self._node.get_logger().warning(f"lockstep beat {self._name}: silent {silent:.1f}s, keepalive engaged")
        self._horizon_ns = min(self._horizon_ns, sim_ns + self._period_ns) + (wall - self._keepalive_at).nanoseconds
        self._keepalive_at = wall
        msg = LockstepHeartbeat()
        msg.header.stamp = Time(nanoseconds=max(self._horizon_ns - self._period_ns, 0)).to_msg()
        self._pub.publish(msg)

    async def _wait(self, seconds: float) -> None:
        if self._sleep is not None:
            await self._sleep(seconds)
            return
        await _executor_sleep(self._node, seconds, wall=True)

    async def _apply(self, channels: list[LockstepChannel]) -> bool:
        deadline = self._wall.now().nanoseconds * 1e-9 + _REGISTER_TIMEOUT
        while not self._client.service_is_ready():
            if self._wall.now().nanoseconds * 1e-9 >= deadline:
                self._node.get_logger().info("lockstep register service not available, skipping beat registration")
                return False
            await self._wait(0.1)
        request = LockstepRegister.Request()
        request.registration = LockstepRegistration(caller=self._topic, env=self._env, channels=channels)
        call = self._client.call_async(request)
        while not call.done():
            if self._wall.now().nanoseconds * 1e-9 >= deadline:
                self._node.get_logger().warning(f"lockstep beat {self._name}: register call timed out")
                return False
            await self._wait(0.1)
        response = call.result()
        assert response is not None
        if not response.success:
            self._node.get_logger().warning(f"lockstep beat {self._name}: registration failed: {response.error_msg}")
            return False
        return bool(channels)
