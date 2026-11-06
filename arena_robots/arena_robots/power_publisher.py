"""Power publisher node"""

from __future__ import annotations

import math

import rclpy
from arena_rclpy_mixins.lazy import LazyPublisher
from arena_rclpy_mixins.params import Param, Range, declare
from arena_rclpy_mixins.spin import spin_node
from arena_robots_msgs.msg import Energy, Power
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


class PowerPublisher(Node):
    """Instantaneous power and integrated energy from joint effort and velocity, battery charge refilled by `~/charge_power` in W."""

    def __init__(self, *, parameter_overrides: list[Parameter] | None = None) -> None:
        super().__init__("power_publisher", parameter_overrides=parameter_overrides)

        self._static_power_w: float = declare(self, Param[float]("static_power_w", 0.0, "Constant baseline power draw in W.", constraint=Range(lo=0.0))).value
        self._efficiency: float = declare(self, Param[float]("drivetrain_efficiency", 1.0, "Drivetrain efficiency dividing the mechanical power.", constraint=Range(lo=0.0, lo_open=True))).value
        self._heating_coeff: float = declare(self, Param[float]("heating_coefficient_ch", 0.0, "Thermal power per squared joint torque in W/Nm^2.", constraint=Range(lo=0.0))).value
        self._battery_capacity_wh: float = declare(self, Param[float]("battery_capacity_wh", 0.0, "Battery capacity in Wh (0 = unknown).", constraint=Range(lo=0.0))).value
        self._max_torque_nm: float = declare(self, Param[float]("max_joint_torque_nm", 15.0, "Joint torque clamp in Nm (0 = no clamp).", constraint=Range(lo=0.0))).value
        self._filter_tau_s: float = declare(self, Param[float]("filter_tau_s", 0.125, "Time constant of the power low-pass filter in s (0 = unfiltered).", constraint=Range(lo=0.0))).value

        # Physical drivetrain & rolling resistance parameters
        self._drivetrain_damping: float = declare(self, Param[float]("drivetrain_damping", 0.0, "Viscous drivetrain damping torque per joint velocity in Nm*s/rad.", constraint=Range(lo=0.0))).value
        self._drivetrain_friction: float = declare(self, Param[float]("drivetrain_friction", 0.0, "Coulomb drivetrain friction torque in Nm.", constraint=Range(lo=0.0))).value
        self._rolling_resistance_crr: float = declare(self, Param[float]("rolling_resistance_crr", 0.0, "Rolling resistance coefficient of the wheels.", constraint=Range(lo=0.0))).value
        self._robot_mass_kg: float = declare(self, Param[float]("robot_mass_kg", 0.0, "Robot mass in kg, used for rolling resistance (0 = none).", constraint=Range(lo=0.0))).value
        self._wheel_radius_m: float = declare(self, Param[float]("wheel_radius_m", 0.0, "Wheel radius in m, used for rolling resistance (0 = none).", constraint=Range(lo=0.0))).value
        self._num_wheels: int = declare(self, Param[int]("num_wheels", 1, "Number of wheels sharing the rolling resistance (0 = no rolling resistance).", constraint=Range(lo=0))).value

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
        self.create_subscription(Float64, "~/charge_power", self._on_charge_power, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.get_logger().info(
            f"PowerPublisher ready: static={self._static_power_w:.1f} W, "
            f"efficiency={self._efficiency}, c_h={self._heating_coeff}, "
            f"tau_max={self._max_torque_nm:.1f} Nm, filter_tau={self._filter_tau_s:.3f} s, "
            f"battery={self._battery_capacity_wh} Wh, "
            f"losses(b={self._drivetrain_damping}, tau_f={self._drivetrain_friction}, tau_roll={self._tau_roll_per_wheel:.4f} Nm)"
        )

    def _on_charge_power(self, msg: Float64) -> None:
        self._charge_power_w = msg.data

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
