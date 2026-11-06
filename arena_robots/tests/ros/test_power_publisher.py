"""Tests for PowerPublisher ROS 2 Node."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

if TYPE_CHECKING:
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
