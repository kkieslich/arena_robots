"""Descriptions and constraints of the task server and goal bridge parameters."""

from __future__ import annotations

from rclpy.parameter import Parameter


def test_goal_to_plan_bridge_describes_replan_period() -> None:
    from arena_robots.goal_to_plan_bridge import GoalToPlanBridge

    node = GoalToPlanBridge()
    try:
        [descriptor] = node.describe_parameters(["replan_period"])
        assert descriptor.description
        assert node.get_parameter("replan_period").value == 1.0
    finally:
        node.destroy_node()


def test_task_server_describes_robot_name() -> None:
    from arena_robots.task_server import TaskServerNode

    node = TaskServerNode(parameter_overrides=[Parameter("robot_name", value="jackal")])
    try:
        [descriptor] = node.describe_parameters(["robot_name"])
        assert descriptor.description
        assert node.get_parameter("robot_name").value == "jackal"
    finally:
        node.destroy_node()
