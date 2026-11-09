"""Tests for PowerPublisher ROS 2 Node."""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

if TYPE_CHECKING:
    from collections.abc import Callable

    from arena_robots.power_publisher import PowerPublisher


def test_power_publisher_initialization() -> None:
    from arena_robots.power_publisher import PowerPublisher

    node = PowerPublisher(
        parameter_overrides=[
            Parameter("static_power_w", value=35.0),
            Parameter("drivetrain_efficiency", value=0.70),
            Parameter("heating_coefficient_ch", value=1.25),
            Parameter("battery_capacity_wh", value=270.0),
        ]
    )

    try:
        assert node._efficiency == 0.70
        assert node._heating_coeff == 1.25
        assert node._battery_capacity_wh == 270.0
        assert node._static_power_w == 35.0
    finally:
        node.destroy_node()


def test_power_publisher_on_valid_joint_state() -> None:
    from arena_robots.power_publisher import PowerPublisher

    node = PowerPublisher(
        parameter_overrides=[
            Parameter("static_power_w", value=25.0),
            Parameter("drivetrain_efficiency", value=0.80),
            Parameter("heating_coefficient_ch", value=2.0),
            Parameter("battery_capacity_wh", value=100.0),
        ]
    )

    try:
        msg = JointState()
        msg.header.stamp.sec = 10
        msg.header.stamp.nanosec = 0
        msg.name = ["left_wheel_joint", "right_wheel_joint"]
        msg.velocity = [2.0, 2.0]
        msg.effort = [3.0, 3.0]

        # First callback sets _last_time
        node._on_joint_state(msg)

        # Expected instantaneous:
        # P_mech = 2 * (3.0 * 2.0 / 0.80) = 2 * 7.5 = 15.0 W
        # P_heat = 2 * (2.0 * 3.0^2) = 2 * 18.0 = 36.0 W
        # P_static = 25.0 W
        # P_total = 25.0 + 15.0 + 36.0 = 76.0 W

        # Second callback after 1.0 second
        msg2 = JointState()
        msg2.header.stamp.sec = 11
        msg2.header.stamp.nanosec = 0
        msg2.name = ["left_wheel_joint", "right_wheel_joint"]
        msg2.velocity = [2.0, 2.0]
        msg2.effort = [3.0, 3.0]
        node._on_joint_state(msg2)

        # dt = 1.0 s -> E += 76.0 * 1.0 / 3600 Wh
        expected_energy = 76.0 / 3600.0
        assert abs(node._total_energy_consumed_wh - expected_energy) < 1e-4
    finally:
        node.destroy_node()


def test_power_publisher_on_missing_effort() -> None:
    from arena_robots.power_publisher import PowerPublisher

    node = PowerPublisher(
        parameter_overrides=[
            Parameter("static_power_w", value=35.0),
            Parameter("drivetrain_efficiency", value=0.70),
            Parameter("heating_coefficient_ch", value=1.25),
            Parameter("battery_capacity_wh", value=270.0),
        ]
    )

    try:
        # JointState with velocity but empty effort
        msg = JointState()
        msg.header.stamp.sec = 10
        msg.header.stamp.nanosec = 0
        msg.name = ["wheel_left", "wheel_right"]
        msg.velocity = [1.5, 1.5]
        msg.effort = []  # Missing effort

        node._on_joint_state(msg)
        # Should initialize _last_time and not crash
        assert node._total_energy_consumed_wh == 0.0
    finally:
        node.destroy_node()


def test_power_publisher_with_drivetrain_and_rolling_losses() -> None:
    from arena_robots.power_publisher import PowerPublisher

    node = PowerPublisher(
        parameter_overrides=[
            Parameter("static_power_w", value=10.0),
            Parameter("drivetrain_efficiency", value=0.80),
            Parameter("heating_coefficient_ch", value=1.0),
            Parameter("battery_capacity_wh", value=200.0),
            Parameter("filter_tau_s", value=0.0),  # disable EMA filter for exact step verification
            Parameter("drivetrain_damping", value=0.010),
            Parameter("drivetrain_friction", value=0.050),
            Parameter("rolling_resistance_crr", value=0.020),
            Parameter("robot_mass_kg", value=20.0),
            Parameter("wheel_radius_m", value=0.10),
            Parameter("num_wheels", value=2),
        ]
    )

    try:
        # tau_roll = (0.020 * 20.0 * 9.81 * 0.10) / 2 = 0.1962 Nm
        # At omega = 5.0 rad/s with raw effort = 0.0 (steady state cruise in Gazebo):
        # tau_parasitic = 0.050 + 0.010 * 5.0 + 0.1962 = 0.2962 Nm
        # total_effort = 0.0 + 0.2962 = 0.2962 Nm
        # P_mech per wheel = (0.2962 * 5.0) / 0.80 = 1.85125 W
        # Total P_mech (2 wheels) = 3.7025 W
        # P_therm per wheel = 1.0 * (0.2962^2) = 0.08773444 W
        # Total P_therm (2 wheels) = 0.17546888 W
        # Total power = 10.0 + 3.7025 + 0.17546888 = 13.87796888 W

        msg = JointState()
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.name = ["wheel_l", "wheel_r"]
        msg.velocity = [5.0, 5.0]
        msg.effort = [0.0, 0.0]

        node._on_joint_state(msg)

        msg2 = JointState()
        msg2.header.stamp.sec = 1
        msg2.header.stamp.nanosec = 0
        msg2.name = ["wheel_l", "wheel_r"]
        msg2.velocity = [5.0, 5.0]
        msg2.effort = [0.0, 0.0]

        node._on_joint_state(msg2)

        expected_energy = 13.87796888 / 3600.0
        assert abs(node._total_energy_consumed_wh - expected_energy) < 1e-4
    finally:
        node.destroy_node()


def _idle(sec: int) -> JointState:
    msg = JointState()
    msg.header.stamp.sec = sec
    msg.name = ["wheel_l", "wheel_r"]
    msg.velocity = [0.0, 0.0]
    msg.effort = [0.0, 0.0]
    return msg


def _idle_node() -> PowerPublisher:
    from arena_robots.power_publisher import PowerPublisher

    # 36 W draw on a 10 Wh battery: 100 s idle uses 1 Wh, 10 s at 360 W puts back 0.9 Wh net
    return PowerPublisher(
        parameter_overrides=[
            Parameter("static_power_w", value=36.0),
            Parameter("battery_capacity_wh", value=10.0),
            Parameter("filter_tau_s", value=0.0),
        ]
    )


def test_charging_refills_the_battery_and_leaves_the_consumed_energy_alone() -> None:
    charged = _idle_node()
    idle = _idle_node()
    try:
        for node in (charged, idle):
            node._on_joint_state(_idle(0))
            node._on_joint_state(_idle(100))
        assert charged._stored_energy_wh == pytest.approx(9.0)

        charged._on_charge_power(Float64(data=360.0))
        for node in (charged, idle):
            node._on_joint_state(_idle(110))
        assert charged._stored_energy_wh == pytest.approx(9.9)
        assert idle._stored_energy_wh == pytest.approx(8.9)
        assert charged._total_energy_consumed_wh == pytest.approx(1.1)
        assert charged._total_energy_consumed_wh == idle._total_energy_consumed_wh

        charged._on_charge_power(Float64(data=0.0))
        charged._on_joint_state(_idle(210))
        assert charged._stored_energy_wh == pytest.approx(8.9)
    finally:
        charged.destroy_node()
        idle.destroy_node()


def test_charging_stops_at_the_battery_capacity() -> None:
    node = _idle_node()
    try:
        node._on_joint_state(_idle(0))
        node._on_joint_state(_idle(100))
        node._on_charge_power(Float64(data=360.0))
        node._on_joint_state(_idle(1000))
        assert node._stored_energy_wh == 10.0
        assert node._total_energy_consumed_wh == pytest.approx(10.0)

        node._on_charge_power(Float64(data=0.0))
        node._on_joint_state(_idle(1100))
        assert node._stored_energy_wh == pytest.approx(9.0)
    finally:
        node.destroy_node()


def test_a_state_of_charge_setpoint_sets_the_battery_and_leaves_the_consumed_energy_alone() -> None:
    node = _idle_node()
    try:
        node._on_joint_state(_idle(0))
        node._on_joint_state(_idle(100))
        node._on_soc_percent(Float64(data=40.0))
        assert node._stored_energy_wh == pytest.approx(4.0)
        assert node._total_energy_consumed_wh == pytest.approx(1.0)

        node._on_joint_state(_idle(200))
        assert node._stored_energy_wh == pytest.approx(3.0)
        assert node._total_energy_consumed_wh == pytest.approx(2.0)

        node._on_soc_percent(Float64(data=150.0))
        assert node._stored_energy_wh == 10.0
    finally:
        node.destroy_node()


def _odom(sec: float, v: float = 0.0, w: float = 0.0, vy: float = 0.0) -> Odometry:
    msg = Odometry()
    msg.header.stamp.sec = int(sec)
    msg.header.stamp.nanosec = int(round((sec - int(sec)) * 1e9))
    msg.twist.twist.linear.x = v
    msg.twist.twist.linear.y = vy
    msg.twist.twist.angular.z = w
    return msg


def _flatsim_joints(sec: int) -> JointState:
    """A joint state like flatsim's: names and positions, no velocity, no effort."""
    msg = JointState()
    msg.header.stamp.sec = sec
    msg.name = ["front_left_wheel_joint", "front_right_wheel_joint"]
    msg.position = [0.0, 0.0]
    return msg


def _odometry_node(**params: float) -> PowerPublisher:
    from arena_robots.power_publisher import PowerPublisher

    # the jackal's coefficients on a 36 W standby draw
    defaults = {"static_power_w": 36.0, "battery_capacity_wh": 10.0, "filter_tau_s": 0.0, "odometry_linear_w_per_mps": 50.0, "odometry_angular_w_per_radps": 17.0}
    return PowerPublisher(parameter_overrides=[Parameter(name, value=value) for name, value in {**defaults, **params}.items()])


def _power_over(node: PowerPublisher, joints: Callable[[int], JointState], **twist: float) -> float:
    """Total power in W between two joint states 100 s apart, with odometry of `twist` before each."""
    consumed = node._total_energy_consumed_wh
    for sec in (0, 100):
        node._on_odom(_odom(sec, **twist))
        node._on_joint_state(joints(sec))
    return (node._total_energy_consumed_wh - consumed) * 3600.0 / 100.0


def test_a_robot_standing_still_draws_only_its_standby_power_when_estimated_from_odometry() -> None:
    node = _odometry_node()
    try:
        assert _power_over(node, _flatsim_joints) == pytest.approx(36.0)
        assert node._stored_energy_wh == pytest.approx(9.0)
    finally:
        node.destroy_node()


def _soc_then_capacity(node: PowerPublisher, soc: float, capacity: float) -> None:
    node._on_soc_percent(Float64(data=soc))
    node._on_battery_capacity_wh(Float64(data=capacity))


def _capacity_then_soc(node: PowerPublisher, soc: float, capacity: float) -> None:
    node._on_battery_capacity_wh(Float64(data=capacity))
    node._on_soc_percent(Float64(data=soc))


@pytest.mark.parametrize("order", [_soc_then_capacity, _capacity_then_soc], ids=["setpoint_first", "capacity_first"])
def test_a_capacity_override_keeps_the_charge_in_percent_whether_it_comes_before_or_after_the_setpoint(order) -> None:
    node = _idle_node()
    try:
        order(node, 40.0, 15.0)
        assert node._battery_capacity_wh == 15.0
        assert node._stored_energy_wh == pytest.approx(6.0)

        node._on_joint_state(_idle(0))
        node._on_joint_state(_idle(100))
        assert node._stored_energy_wh == pytest.approx(5.0)
        assert node._total_energy_consumed_wh == pytest.approx(1.0)
    finally:
        node.destroy_node()


@pytest.mark.parametrize(
    ("twist", "power"),
    [
        ({"v": 0.25}, 36.0 + 12.5),
        ({"v": 0.5}, 36.0 + 25.0),
        ({"v": -0.5}, 36.0 + 25.0),
        ({"v": 0.3, "vy": 0.4}, 36.0 + 25.0),
        ({"w": 1.5}, 36.0 + 25.5),
        ({"v": 0.5, "w": -1.0}, 36.0 + 25.0 + 17.0),
    ],
    ids=["slow", "nav2_cap", "reverse", "sideways", "turn_in_place", "drive_and_turn"],
)
def test_the_odometry_estimate_rises_with_speed_and_turn_rate(twist: dict[str, float], power: float) -> None:
    node = _odometry_node()
    try:
        assert _power_over(node, _flatsim_joints, **twist) == pytest.approx(power)
    finally:
        node.destroy_node()


def test_the_estimated_energy_integrates_into_consumption_and_charge() -> None:
    node = _odometry_node()
    try:
        node._on_odom(_odom(0, v=0.4))
        node._on_joint_state(_flatsim_joints(0))
        for sec in range(10, 101, 10):
            node._on_odom(_odom(sec, v=0.4))
            node._on_joint_state(_flatsim_joints(sec))
        # 36 W + 50 W per m/s * 0.4 m/s = 56 W for 100 s
        assert node._total_energy_consumed_wh == pytest.approx(56.0 * 100.0 / 3600.0)
        assert node._stored_energy_wh == pytest.approx(10.0 - 56.0 * 100.0 / 3600.0)
    finally:
        node.destroy_node()


def test_a_capacity_override_keeps_the_charge_in_percent_mid_episode_and_leaves_the_consumed_energy_alone() -> None:
    overridden = _idle_node()
    plain = _idle_node()
    try:
        for node in (overridden, plain):
            node._on_joint_state(_idle(0))
            node._on_joint_state(_idle(100))
        overridden._on_battery_capacity_wh(Float64(data=20.0))
        assert overridden._stored_energy_wh == pytest.approx(18.0)

        overridden._on_charge_power(Float64(data=360.0))
        for node in (overridden, plain):
            node._on_joint_state(_idle(200))
        assert overridden._stored_energy_wh == 20.0
        assert overridden._total_energy_consumed_wh == plain._total_energy_consumed_wh == pytest.approx(2.0)
    finally:
        overridden.destroy_node()
        plain.destroy_node()


@pytest.mark.parametrize("restore", [0.0, -5.0])
def test_a_capacity_of_0_or_less_restores_the_configured_one(restore: float) -> None:
    node = _idle_node()
    try:
        _capacity_then_soc(node, 40.0, 15.0)
        node._on_battery_capacity_wh(Float64(data=restore))
        assert node._battery_capacity_wh == 10.0
        assert node._stored_energy_wh == pytest.approx(4.0)
    finally:
        node.destroy_node()


def _gazebo_joints(sec: int) -> JointState:
    msg = _flatsim_joints(sec)
    msg.velocity = [5.0, 5.0]
    msg.effort = [0.5, 0.5]
    return msg


def _velocity_only_joints(sec: int) -> JointState:
    msg = _flatsim_joints(sec)
    msg.velocity = [5.0, 5.0]
    return msg


@pytest.mark.parametrize("joints", [_gazebo_joints, _velocity_only_joints], ids=["effort_and_velocity", "velocity_only"])
def test_joint_states_with_velocity_or_effort_never_get_the_odometry_estimate(joints: Callable[[int], JointState]) -> None:
    # a little drivetrain friction, so velocity alone has a drive power of its own
    estimating = _odometry_node(drivetrain_friction=0.1)
    plain = _odometry_node(drivetrain_friction=0.1, odometry_linear_w_per_mps=0.0, odometry_angular_w_per_radps=0.0)
    try:
        driving = _power_over(estimating, joints, v=0.5, w=1.0)
        assert driving == pytest.approx(_power_over(plain, joints, v=0.5, w=1.0))
        assert driving > 36.0
    finally:
        estimating.destroy_node()
        plain.destroy_node()


def test_without_coefficients_odometry_changes_nothing() -> None:
    node = _odometry_node(odometry_linear_w_per_mps=0.0, odometry_angular_w_per_radps=0.0)
    try:
        assert _power_over(node, _flatsim_joints, v=0.5, w=1.0) == pytest.approx(36.0)
    finally:
        node.destroy_node()


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_capacity_that_is_not_finite_is_ignored(bad: float) -> None:
    node = _idle_node()
    try:
        node._on_battery_capacity_wh(Float64(data=15.0))
        node._on_battery_capacity_wh(Float64(data=bad))
        assert node._battery_capacity_wh == 15.0
    finally:
        node.destroy_node()


def test_odometry_older_than_a_second_counts_as_standing_still() -> None:
    node = _odometry_node()
    try:
        node._on_odom(_odom(0, v=0.5))
        node._on_joint_state(_flatsim_joints(0))
        node._on_joint_state(_flatsim_joints(100))
        assert node._total_energy_consumed_wh == pytest.approx(36.0 * 100.0 / 3600.0)
        node._on_odom(_odom(199.5, v=0.5))
        node._on_joint_state(_flatsim_joints(200))
        assert node._total_energy_consumed_wh == pytest.approx((36.0 * 100.0 + 61.0 * 100.0) / 3600.0)
    finally:
        node.destroy_node()


def test_the_energy_message_reports_the_charge_against_the_overridden_capacity() -> None:
    from arena_robots_msgs.msg import Energy

    node = _idle_node()
    sent: list[Energy] = []
    node._energy_pub = SimpleNamespace(publish=lambda build: sent.append(build()))
    try:
        _capacity_then_soc(node, 50.0, 15.0)
        node._on_joint_state(_idle(0))
        node._on_joint_state(_idle(150))
        # 36 W for 150 s is 1.5 Wh off 7.5 Wh of 15 Wh
        assert sent[-1].battery_soc_percent == pytest.approx(40.0)
        assert sent[-1].total_energy_consumed_wh == pytest.approx(1.5)
    finally:
        node.destroy_node()


def test_the_estimate_reaches_the_power_topic_from_the_odom_and_joint_states_topics() -> None:
    import rclpy
    import rclpy.node
    from arena_robots_msgs.msg import Power
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    node = _odometry_node()
    sim = rclpy.node.Node("test_power_publisher_sim")
    odom = sim.create_publisher(Odometry, "/odom", qos)
    joints = sim.create_publisher(JointState, "/joint_states", qos)
    power: list[Power] = []
    sim.create_subscription(Power, "/power_publisher/power", power.append, qos)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(sim)

    def spin_until(pred: Callable[[], bool], what: str) -> None:
        deadline = time.monotonic() + 5.0
        while not pred():
            assert time.monotonic() < deadline, f"timed out waiting for {what}"
            executor.spin_once(timeout_sec=0.05)

    try:
        spin_until(lambda: odom.get_subscription_count() == 1 and joints.get_subscription_count() == 1 and node._power_pub.wanted, "the subscriptions")
        odom.publish(_odom(10, v=0.5, w=1.0))
        spin_until(lambda: node._odom is not None, "the odometry")
        joints.publish(_flatsim_joints(10))
        spin_until(lambda: len(power) == 1, "a power message")
        assert power[0].total_power_w == pytest.approx(36.0 + 25.0 + 17.0)
        assert power[0].static_power_w == pytest.approx(36.0)
    finally:
        executor.shutdown()
        node.destroy_node()
        sim.destroy_node()


def test_latched_capacity_and_setpoint_published_before_the_node_starts_reach_it_in_its_namespace() -> None:
    import rclpy
    import rclpy.node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    from arena_robots.power_publisher import PowerPublisher

    namespace = "/test_power_publisher/env_0/jackal"
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    peer = rclpy.node.Node("test_power_publisher_peer")
    peer.create_publisher(Float64, f"{namespace}/power_publisher/soc_percent", latched).publish(Float64(data=40.0))
    peer.create_publisher(Float64, f"{namespace}/power_publisher/battery_capacity_wh", latched).publish(Float64(data=15.0))
    node = PowerPublisher(namespace=namespace, parameter_overrides=[Parameter("static_power_w", value=36.0), Parameter("battery_capacity_wh", value=270.0)])
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        deadline = time.monotonic() + 5.0
        while (node._battery_capacity_wh, round(node._soc_fraction, 9)) != (15.0, 0.4) and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert node._battery_capacity_wh == 15.0
        assert node._stored_energy_wh == pytest.approx(6.0)
    finally:
        executor.shutdown()
        node.destroy_node()
        peer.destroy_node()
