"""Drive `planner_server` from `<ns>/goal_pose`.

`planner_server` only publishes `/plan` as a side effect of the
`compute_path_to_pose` action. When we run nav2 in `planner_only` mode there is
no `bt_navigator` calling it, so this node bridges PoseStamped goals to the
action and re-issues them so `/plan` tracks the robot as it moves.
"""

from __future__ import annotations

import rclpy
from action_msgs.msg import GoalStatus
from arena_rclpy_mixins.params import Param, Range, declare
from arena_rclpy_mixins.spin import spin_node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.task import Future


class GoalToPlanBridge(Node):
    def __init__(self) -> None:
        super().__init__("goal_to_plan_bridge")

        goal_topic = declare(self, Param[str]("goal_topic", "goal_pose", "Topic of the PoseStamped goals to plan to.")).value
        action_name = declare(self, Param[str]("action_name", "compute_path_to_pose", "Name of the ComputePathToPose action served by planner_server.")).value
        replan_period = declare(self, Param[float]("replan_period", 1.0, "Period in wall seconds between plan requests for the current goal.", constraint=Range(lo=0.0, lo_open=True))).value
        self._inflight_timeout = declare(self, Param[float]("inflight_timeout", 5.0, "Wall seconds without an answer after which a plan request is re-issued.", constraint=Range(lo=0.0, lo_open=True))).value
        self._planner_id = declare(self, Param[str]("planner_id", "", "Planner plugin id used for the requests (empty = planner_server default).")).value

        self._goal: PoseStamped | None = None
        self._goal_seq: int = 0
        self._sent_seq: int = -1
        self._in_flight_since: float | None = None
        self._plans: int = 0
        self._wall = Clock(clock_type=ClockType.STEADY_TIME)
        self._client: ActionClient = ActionClient(self, ComputePathToPose, action_name)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PoseStamped, goal_topic, self._on_goal, qos)
        self.create_timer(replan_period, self._replan, clock=self._wall)

    def _now(self) -> float:
        return self._wall.now().nanoseconds * 1e-9

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg
        self._goal_seq += 1
        self._replan()

    def _replan(self) -> None:
        if self._goal is None:
            return
        if self._in_flight_since is not None:
            if self._now() - self._in_flight_since < self._inflight_timeout:
                return
            self.get_logger().warning(f"compute_path_to_pose silent for {self._inflight_timeout:.0f}s, re-issuing")
        if not self._client.wait_for_server(timeout_sec=0.0):
            if self._sent_seq < 0:
                self.get_logger().info("waiting for compute_path_to_pose", throttle_duration_sec=10.0)
            return
        goal = ComputePathToPose.Goal()
        goal.goal = self._goal
        goal.use_start = False
        if self._planner_id:
            goal.planner_id = self._planner_id
        self._in_flight_since = self._now()
        self._sent_seq = self._goal_seq
        self._client.send_goal_async(goal).add_done_callback(self._on_sent)

    def _on_sent(self, future: Future) -> None:
        handle = future.result()
        if handle is None or not handle.accepted:
            self._in_flight_since = None
            self.get_logger().warning("compute_path_to_pose rejected the goal", throttle_duration_sec=5.0)
            return
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future: Future) -> None:
        self._in_flight_since = None
        wrapped = future.result()
        if wrapped is None:
            return
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warning(f"compute_path_to_pose failed (status {wrapped.status}, error {wrapped.result.error_code} {wrapped.result.error_msg!r})", throttle_duration_sec=5.0)
            return
        self._plans += 1
        if self._plans == 1:
            self.get_logger().info(f"first plan: {len(wrapped.result.path.poses)} poses")


def main() -> None:
    rclpy.init()
    node = GoalToPlanBridge()
    try:
        spin_node(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
