"""Acoustics publisher node implementing M4 Ego-Noise Model"""

from __future__ import annotations

import math
from pathlib import Path

import rclpy
import yaml
from arena_rclpy_mixins.lazy import LazyPublisher
from arena_rclpy_mixins.params import Param, declare
from arena_rclpy_mixins.spin import spin_node
from arena_robots_msgs.msg import Acoustics, CollisionEvents
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


class AcousticsPublisher(Node):
    """ROS 2 node that computes acoustic ego-noise level from joint states and collisions."""

    COLLISION_IMPULSE_DBA: float = 100.0

    def __init__(self, *, parameter_overrides: list[Parameter] | None = None) -> None:
        super().__init__("acoustics_publisher", parameter_overrides=parameter_overrides)

        robot_name_param = declare(self, Param[str]("robot_name", "", "Robot model name the acoustic profile belongs to.")).value
        profile_path_param = declare(self, Param[str]("profile_path", "", "Path of the acoustic profile YAML of the robot (required).")).value

        profile_file = Path(profile_path_param)
        if not profile_path_param or not profile_file.is_file():
            self.get_logger().fatal(f"Acoustic profile file not found for robot '{robot_name_param}' (path: {profile_path_param})")
            raise SystemExit(1)

        with open(profile_file) as f:
            cfg = yaml.safe_load(f)

        self._L_base_0: float = float(cfg.get("L_base_0", 42.0))
        self._beta_0: float = float(cfg.get("beta_0", 45.0))
        self._beta_1: float = float(cfg.get("beta_1", 18.0))
        self._beta_2: float = float(cfg.get("beta_2", 5.0))
        self._omega_ref: float = float(cfg.get("omega_ref", 5.0))
        self._tau_ref: float = float(cfg.get("tau_ref", 10.0))
        self._max_torque_nm: float = float(cfg.get("max_joint_torque_nm", 15.0))
        self._omega_deadband: float = float(cfg.get("omega_deadband", 0.05))
        self._omega_active: float = float(cfg.get("omega_active", 0.20))
        self._sigma_base: float = float(cfg.get("sigma_base", 1.0))
        self._sigma_dynamic: float = float(cfg.get("sigma_dynamic", 0.8))
        self._sigma_no_effort: float = float(cfg.get("sigma_no_effort", 1.0))

        self._P_base: float = 10.0 ** (self._L_base_0 / 10.0)

        # The profile is a fitted parametric model, not a calibration against a sound level meter
        self._calibration_status: str = f"uncalibrated_parametric_model:{robot_name_param or profile_file.stem}"

        topic_param = declare(self, Param[str]("topic", "acoustics", "Topic the ego-noise level is published on.")).value

        # State for the Fast-weighting EMA (IEC 61672-1 Fast time constant = 125ms)
        self._last_time: float | None = None
        self._ema_p_drive: float = 0.0
        self._warned_empty_effort: bool = False

        # Collision positive-flank state tracking (trigger 100 dBA only on impact transition 0 -> 1)
        self._in_collision: bool = False
        self._collision_impulse_pending: bool = False

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._acoustics_pub: LazyPublisher[Acoustics] = LazyPublisher(self.create_publisher(Acoustics, topic_param, qos))

        self.create_subscription(JointState, "joint_states", self._on_joint_state, qos)
        self.create_subscription(CollisionEvents, "collision_events", self._on_collision_events, qos)

        self.get_logger().info(f"AcousticsPublisher ready: profile={profile_file}, topic={topic_param}, L_base_0={self._L_base_0} dBA")

    def _on_collision_events(self, msg: CollisionEvents) -> None:
        """Track collision state and trigger impulse on positive flank (impact onset)."""
        is_colliding = len(msg.events) > 0
        if is_colliding and not self._in_collision:
            self._collision_impulse_pending = True
        self._in_collision = is_colliding

    def _on_joint_state(self, msg: JointState) -> None:
        stamp = msg.header.stamp
        current_time = stamp.sec + stamp.nanosec * 1e-9

        n_joints = len(msg.velocity)
        has_velocity = n_joints > 0
        has_effort = len(msg.effort) > 0

        if not has_effort and not self._warned_empty_effort:
            self.get_logger().warning("JointState has no effort data. Acoustic model will set T_eq=0 and apply uncertainty penalty.")
            self._warned_empty_effort = True

        if has_velocity:
            omega_eq = math.sqrt(sum(w**2 for w in msg.velocity) / n_joints)
        else:
            omega_eq = 0.0

        if has_effort and len(msg.effort) > 0:
            efforts = [max(-self._max_torque_nm, min(self._max_torque_nm, tau)) if self._max_torque_nm > 0.0 else tau for tau in msg.effort]
            t_eq = sum(abs(tau) for tau in efforts) / len(efforts)
        else:
            t_eq = 0.0

        delta_active = self._omega_active - self._omega_deadband
        if delta_active > 0:
            lambda_omega = max(0.0, min(1.0, (omega_eq - self._omega_deadband) / delta_active))
        else:
            lambda_omega = 1.0 if omega_eq >= self._omega_active else 0.0

        dt = 0.0
        if self._last_time is not None:
            dt = current_time - self._last_time
        self._last_time = current_time

        # Pure torque and rotational speed formulation (physically captures dynamic load via torque)
        p_drive_raw = lambda_omega * (10.0 ** (self._beta_0 / 10.0)) * ((max(omega_eq, self._omega_active) / self._omega_ref) ** (self._beta_1 / 10.0)) * ((1.0 + t_eq / self._tau_ref) ** (self._beta_2 / 10.0))

        # IEC 61672-1 Fast time weighting, tau_F = 0.125 s
        TAU_FAST = 0.125
        if dt > 0.0:
            alpha = 1.0 - math.exp(-dt / TAU_FAST)
            self._ema_p_drive = (1.0 - alpha) * self._ema_p_drive + alpha * p_drive_raw
        else:
            self._ema_p_drive = p_drive_raw

        p_drive = self._ema_p_drive

        # Positive-flank collision acoustic impulse (fires on impact frame only)
        is_impact = self._collision_impulse_pending
        self._collision_impulse_pending = False

        def acoustics() -> Acoustics:
            p_total = self._P_base + p_drive
            l_1m = 10.0 * math.log10(p_total) if p_total > 0.0 else 0.0

            l_base = self._L_base_0
            l_drivetrain = 10.0 * math.log10(p_drive) if p_drive > 1e-12 else 0.0

            effort_unc = 0.0 if has_effort else (self._sigma_no_effort**2)
            sigma_dyn = self._sigma_dynamic * math.log(1.0 + (omega_eq / max(self._omega_ref, 1e-3)))
            sigma_total = min(2.5, math.sqrt(self._sigma_base**2 + sigma_dyn**2 + effort_unc))

            validity_flags = 0
            if not has_effort:
                validity_flags |= Acoustics.FLAG_NO_EFFORT
            if not has_velocity:
                validity_flags |= Acoustics.FLAG_NO_VELOCITY

            if is_impact:
                l_1m = max(l_1m, self.COLLISION_IMPULSE_DBA)
                operating_state = "collision"
            elif lambda_omega > 0.0:
                operating_state = "driving"
            else:
                operating_state = "idle"

            out_msg = Acoustics()
            out_msg.header = msg.header
            out_msg.total_level_af_dba = float(l_1m)
            out_msg.total_level_zf_db = float("nan")  # Broadband proxy only supports A-weighted dBA
            out_msg.baseline_level_dba = float(l_base)
            out_msg.drivetrain_level_dba = float(l_drivetrain)
            out_msg.uncertainty_1sigma_dba = float(sigma_total)
            out_msg.validity_flags = int(validity_flags)
            out_msg.operating_state = operating_state
            out_msg.calibration_status = self._calibration_status

            return out_msg

        self._acoustics_pub.publish(acoustics)


def main() -> None:
    rclpy.init()
    node = AcousticsPublisher()
    try:
        spin_node(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
