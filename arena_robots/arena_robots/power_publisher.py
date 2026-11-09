"""Power publisher node"""

from __future__ import annotations

import math
import typing

import rclpy
from arena_rclpy_mixins.lazy import LazyPublisher
from arena_rclpy_mixins.spin import spin_node
from arena_robots_msgs.msg import Energy, Power
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


class PowerPublisher(Node):
    """Instantaneous power and integrated energy from joint effort and velocity, battery charge set by `~/soc_percent` and refilled by `~/charge_power` in W."""

    def __init__(self, *, parameter_overrides: list[Parameter] | None = None) -> None:
        super().__init__("power_publisher", parameter_overrides=parameter_overrides)

        self._static_power_w: float = float(typing.cast(float, self.declare_parameter("static_power_w", 0.0).value))
        self._efficiency: float = float(typing.cast(float, self.declare_parameter("drivetrain_efficiency", 1.0).value))
        self._heating_coeff: float = float(typing.cast(float, self.declare_parameter("heating_coefficient_ch", 0.0).value))
        self._battery_capacity_wh: float = float(typing.cast(float, self.declare_parameter("battery_capacity_wh", 0.0).value))
        self._max_torque_nm: float = float(typing.cast(float, self.declare_parameter("max_joint_torque_nm", 15.0).value))
        self._filter_tau_s: float = float(typing.cast(float, self.declare_parameter("filter_tau_s", 0.125).value))

        # Physical drivetrain & rolling resistance parameters
        self._drivetrain_damping: float = float(typing.cast(float, self.declare_parameter("drivetrain_damping", 0.0).value))
        self._drivetrain_friction: float = float(typing.cast(float, self.declare_parameter("drivetrain_friction", 0.0).value))
        self._rolling_resistance_crr: float = float(typing.cast(float, self.declare_parameter("rolling_resistance_crr", 0.0).value))
        self._robot_mass_kg: float = float(typing.cast(float, self.declare_parameter("robot_mass_kg", 0.0).value))
        self._wheel_radius_m: float = float(typing.cast(float, self.declare_parameter("wheel_radius_m", 0.0).value))
        self._num_wheels: int = int(typing.cast(int, self.declare_parameter("num_wheels", 1).value))

        if self._num_wheels > 0 and self._robot_mass_kg > 0.0 and self._wheel_radius_m > 0.0:
            self._tau_roll_per_wheel: float = (self._rolling_resistance_crr * self._robot_mass_kg * 9.81 * self._wheel_radius_m) / float(self._num_wheels)
        else:
            self._tau_roll_per_wheel = 0.0

        self._filtered_mech_w: float = 0.0
        self._filtered_therm_w: float = 0.0
        self._last_time: float | None = None
        self._total_energy_consumed_wh: float = 0.0
        self._stored_energy_wh: float = self._battery_capacity_wh
        self._charge_power_w: float = 0.0
        self._warned_empty_effort: bool = False

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._power_pub: LazyPublisher[Power] = LazyPublisher(self.create_publisher(Power, "~/power", qos))
        self._energy_pub: LazyPublisher[Energy] = LazyPublisher(self.create_publisher(Energy, "~/energy", qos))

        self.create_subscription(JointState, "joint_states", self._on_joint_state, qos)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Float64, "~/charge_power", self._on_charge_power, latched)
        self.create_subscription(Float64, "~/soc_percent", self._on_soc_percent, latched)

        self.get_logger().info(
            f"PowerPublisher ready: static={self._static_power_w:.1f} W, "
            f"efficiency={self._efficiency}, c_h={self._heating_coeff}, "
            f"tau_max={self._max_torque_nm:.1f} Nm, filter_tau={self._filter_tau_s:.3f} s, "
            f"battery={self._battery_capacity_wh} Wh, "
            f"losses(b={self._drivetrain_damping}, tau_f={self._drivetrain_friction}, tau_roll={self._tau_roll_per_wheel:.4f} Nm)"
        )

    def _on_charge_power(self, msg: Float64) -> None:
        self._charge_power_w = msg.data

    def _on_soc_percent(self, msg: Float64) -> None:
        self._stored_energy_wh = min(max(msg.data / 100.0 * self._battery_capacity_wh, 0.0), self._battery_capacity_wh)

    def _on_joint_state(self, msg: JointState) -> None:
        stamp = msg.header.stamp
        current_time = stamp.sec + stamp.nanosec * 1e-9

        n_joints = len(msg.name)
        has_effort = len(msg.effort) > 0
        has_velocity = len(msg.velocity) > 0

        if not has_effort and not self._warned_empty_effort:
            self.get_logger().warning("JointState carries no effort, mechanical and thermal power stay zero until it is published.")
            self._warned_empty_effort = True

        joint_names: list[str] = []
        joint_mech: list[float] = []
        joint_therm: list[float] = []
        joint_total: list[float] = []

        for i in range(n_joints):
            name = msg.name[i]  # pyright: ignore[reportIndexIssue]
            raw_effort = msg.effort[i] if i < len(msg.effort) else 0.0
            velocity = msg.velocity[i] if i < len(msg.velocity) else 0.0

            # Torque saturation clamping to eliminate unphysical Gazebo impulse spikes
            if self._max_torque_nm > 0.0:
                effort = max(-self._max_torque_nm, min(self._max_torque_nm, raw_effort))
            else:
                effort = raw_effort

            # Parasitic mechanical drag (coulomb friction + viscous damping + rolling resistance)
            if abs(velocity) > 1e-4:
                tau_parasitic = self._drivetrain_friction + self._drivetrain_damping * abs(velocity) + self._tau_roll_per_wheel
            else:
                tau_parasitic = 0.0

            total_effort = abs(effort) + tau_parasitic
            p_mech = (total_effort * abs(velocity)) / self._efficiency
            p_therm = self._heating_coeff * total_effort * total_effort

            joint_names.append(name)
            joint_mech.append(p_mech)
            joint_therm.append(p_therm)
            joint_total.append(p_mech + p_therm)

        total_mech_raw = math.fsum(joint_mech)
        total_therm_raw = math.fsum(joint_therm)

        # 125ms EMA low-pass filtering (conserves total energy while spreading transient impulses)
        if self._last_time is not None and self._filter_tau_s > 0.0:
            dt = max(current_time - self._last_time, 1e-6)
            alpha = 1.0 - math.exp(-dt / self._filter_tau_s)
            self._filtered_mech_w += alpha * (total_mech_raw - self._filtered_mech_w)
            self._filtered_therm_w += alpha * (total_therm_raw - self._filtered_therm_w)
        else:
            self._filtered_mech_w = total_mech_raw
            self._filtered_therm_w = total_therm_raw

        total_mech = self._filtered_mech_w
        total_therm = self._filtered_therm_w
        total_power = self._static_power_w + total_mech + total_therm

        if self._last_time is not None:
            dt = current_time - self._last_time
            if dt > 0.0:
                self._total_energy_consumed_wh += (total_power * dt) / 3600.0
                self._stored_energy_wh = min(max(self._stored_energy_wh + ((self._charge_power_w - total_power) * dt) / 3600.0, 0.0), self._battery_capacity_wh)
        self._last_time = current_time

        def power() -> Power:
            power_msg = Power()
            power_msg.header = msg.header
            power_msg.total_power_w = total_power
            power_msg.static_power_w = self._static_power_w
            power_msg.total_mechanical_power_w = total_mech
            power_msg.total_thermal_power_w = total_therm
            power_msg.joint_names = joint_names
            power_msg.joint_mechanical_power_w = joint_mech
            power_msg.joint_thermal_power_w = joint_therm
            power_msg.joint_total_power_w = joint_total
            return power_msg

        def energy() -> Energy:
            if self._battery_capacity_wh > 0.0:
                soc = self._stored_energy_wh / self._battery_capacity_wh * 100.0
            else:
                soc = 0.0
            energy_msg = Energy()
            energy_msg.header = msg.header
            energy_msg.total_energy_consumed_wh = self._total_energy_consumed_wh
            energy_msg.battery_soc_percent = soc
            return energy_msg

        self._power_pub.publish(power)
        self._energy_pub.publish(energy)


def main() -> None:
    rclpy.init()
    node = PowerPublisher()
    try:
        spin_node(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
