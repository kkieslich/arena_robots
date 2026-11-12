"""Tests for arena_robots.task_server_handlers.goto_pose.nav2."""

from __future__ import annotations

import pytest

from conftest import until


def _handler():
    from arena_robots.task_server_handlers.goto_pose.nav2 import GotoPoseHandlerNav2

    return GotoPoseHandlerNav2


class TestTranslateNav2Status:
    def test_succeeded(self):
        from action_msgs.msg import GoalStatus

        from arena_robots.task_server_handlers.goto_pose.nav2 import _translate_nav2_status
        from arena_robots_msgs.action import GotoPose

        status, reason = _translate_nav2_status(GoalStatus.STATUS_SUCCEEDED)
        assert status == GotoPose.Result.STATUS_SUCCEEDED
        assert reason == ""

    def test_canceled(self):
        from action_msgs.msg import GoalStatus

        from arena_robots.task_server_handlers.goto_pose.nav2 import _translate_nav2_status
        from arena_robots_msgs.action import GotoPose

        status, reason = _translate_nav2_status(GoalStatus.STATUS_CANCELED)
        assert status == GotoPose.Result.STATUS_CANCELED
        assert reason

    def test_aborted_on_unknown(self):
        from arena_robots.task_server_handlers.goto_pose.nav2 import _translate_nav2_status
        from arena_robots_msgs.action import GotoPose

        status, reason = _translate_nav2_status(99)
        assert status == GotoPose.Result.STATUS_ABORTED
        assert reason

    def test_aborted_on_executing(self):
        from action_msgs.msg import GoalStatus

        from arena_robots.task_server_handlers.goto_pose.nav2 import _translate_nav2_status
        from arena_robots_msgs.action import GotoPose

        status, reason = _translate_nav2_status(GoalStatus.STATUS_EXECUTING)
        assert status == GotoPose.Result.STATUS_ABORTED
        assert reason


class TestGotoPoseHandlerNav2Execute:
    def test_goal_terminated_while_nav2_is_down_returns_canceled(self, goto_rig):
        from action_msgs.msg import GoalStatus

        from arena_robots_msgs.action import GotoPose

        goto_rig.serve(_handler())
        handle = goto_rig.send()
        until(lambda: bool(goto_rig.handles), "server goal")
        goto_rig.handles[0].abort()
        wrapped = goto_rig.result(handle)
        assert wrapped.status == GoalStatus.STATUS_ABORTED
        assert wrapped.result.status == GotoPose.Result.STATUS_CANCELED
        assert goto_rig.registrations == []

    def test_cancel_reaches_nav2_and_cancels_the_goal(self, goto_rig):
        from action_msgs.msg import GoalStatus

        from arena_robots_msgs.action import GotoPose

        nav2 = goto_rig.nav2(hold=True)
        goto_rig.serve(_handler())
        handle = goto_rig.send()
        until(lambda: nav2.received == 1, "nav2 goal")
        goto_rig.cancel(handle)
        wrapped = goto_rig.result(handle)
        assert nav2.canceled == 1
        assert wrapped.status == GoalStatus.STATUS_CANCELED
        assert wrapped.result.status == GotoPose.Result.STATUS_CANCELED

    def test_goal_terminated_during_navigation_returns_canceled(self, goto_rig):
        from arena_robots_msgs.action import GotoPose

        nav2 = goto_rig.nav2(hold=True)
        goto_rig.serve(_handler())
        handle = goto_rig.send()
        until(lambda: nav2.received == 1 and bool(goto_rig.handles), "nav2 goal")
        goto_rig.handles[0].abort()
        nav2.holding = False
        wrapped = goto_rig.result(handle)
        assert wrapped.result.status == GotoPose.Result.STATUS_CANCELED
        assert wrapped.result.reason == "goal preempted by new submit_task"

    def test_nav2_success_succeeds_the_goal(self, goto_rig):
        from action_msgs.msg import GoalStatus
        from geometry_msgs.msg import PoseStamped

        from arena_robots_msgs.action import GotoPose

        goto_rig.nav2()
        goto_rig.serve(_handler())
        wrapped = goto_rig.result(goto_rig.send())
        assert wrapped.status == GoalStatus.STATUS_SUCCEEDED
        assert wrapped.result.status == GotoPose.Result.STATUS_SUCCEEDED
        assert wrapped.result.final_pose == PoseStamped()

    def test_rejected_goal_is_sent_again(self, goto_rig):
        from arena_robots_msgs.action import GotoPose

        nav2 = goto_rig.nav2(reject_first=1)
        goto_rig.serve(_handler())
        wrapped = goto_rig.result(goto_rig.send())
        assert nav2.received == 2
        assert wrapped.result.status == GotoPose.Result.STATUS_SUCCEEDED

    def test_aborted_goal_is_replanned(self, goto_rig):
        from arena_robots_msgs.action import GotoPose

        nav2 = goto_rig.nav2(outcomes=("abort", "succeed"))
        goto_rig.serve(_handler())
        wrapped = goto_rig.result(goto_rig.send())
        assert nav2.received == 2
        assert wrapped.result.status == GotoPose.Result.STATUS_SUCCEEDED

    def test_nav2_feedback_is_translated(self, goto_rig):
        nav2 = goto_rig.nav2(hold=True, distance=1.5)
        goto_rig.serve(_handler())
        handle = goto_rig.send()
        until(lambda: bool(goto_rig.feedback), "translated feedback")
        nav2.holding = False
        goto_rig.result(handle)
        feedback = goto_rig.feedback[0].feedback
        assert feedback.distance_remaining == pytest.approx(1.5)
        assert feedback.eta_seconds == pytest.approx(3.0)

    def test_beat_is_registered_for_the_goal_and_released_after_it(self, goto_rig):
        goto_rig.nav2()
        goto_rig.serve(_handler())
        goto_rig.result(goto_rig.send())
        until(lambda: len(goto_rig.registrations) == 2, "beat release")
        acquired, released = goto_rig.registrations
        assert [(channel.hard, channel.period_s) for channel in acquired] == [(True, pytest.approx(0.1))]
        assert released == []


_XY = "goal_checker.xy_goal_tolerance"
_YAW = "goal_checker.yaw_goal_tolerance"


class TestGotoPoseHandlerNav2Tolerance:
    def test_tighter_tolerances_set_the_goal_checker_for_the_goal_and_restore_it(self, goto_rig):
        from tf2_ros import Buffer

        from arena_robots_msgs.action import GotoPose

        sets = goto_rig.controller(**{_XY: 0.25, _YAW: 0.25})
        goto_rig.nav2()
        buffer = Buffer()
        goto_rig.robot_at(buffer, 0.0)
        goto_rig.serve(_handler(), tf_buffer=buffer)
        wrapped = goto_rig.result(goto_rig.send(pose_tolerance=0.05, yaw_tolerance=0.04))
        assert wrapped.result.status == GotoPose.Result.STATUS_SUCCEEDED
        assert sets == [{_XY: pytest.approx(0.05), _YAW: pytest.approx(0.04)}, {_XY: 0.25, _YAW: 0.25}]

    def test_looser_tolerances_leave_the_goal_checker_alone(self, goto_rig):
        from arena_robots_msgs.action import GotoPose

        sets = goto_rig.controller(**{_XY: 0.25, _YAW: 0.25})
        goto_rig.nav2()
        goto_rig.serve(_handler())
        wrapped = goto_rig.result(goto_rig.send(pose_tolerance=1.0, yaw_tolerance=0.5))
        assert wrapped.result.status == GotoPose.Result.STATUS_SUCCEEDED
        assert sets == []

    def test_base_settled_outside_the_tolerance_is_sent_again(self, goto_rig):
        from arena_robots.task_server_handlers.goto_pose.nav2 import _PRECISE_RESENDS
        from arena_robots_msgs.action import GotoPose
        from tf2_ros import Buffer

        goto_rig.controller(**{_XY: 0.25, _YAW: 0.25})
        nav2 = goto_rig.nav2()
        buffer = Buffer()
        goto_rig.robot_at(buffer, 0.1)
        goto_rig.serve(_handler(), tf_buffer=buffer)
        wrapped = goto_rig.result(goto_rig.send(pose_tolerance=0.05))
        assert nav2.received == 1 + _PRECISE_RESENDS
        assert wrapped.result.status == GotoPose.Result.STATUS_SUCCEEDED

    def test_base_settled_within_the_tolerance_is_not_sent_again(self, goto_rig):
        from tf2_ros import Buffer

        goto_rig.controller(**{_XY: 0.25, _YAW: 0.25})
        nav2 = goto_rig.nav2()
        buffer = Buffer()
        goto_rig.robot_at(buffer, 0.03)
        goto_rig.serve(_handler(), tf_buffer=buffer)
        goto_rig.result(goto_rig.send(pose_tolerance=0.05))
        assert nav2.received == 1
