"""ROS node that exposes per-TaskKind action servers for a robot's configured bringups."""

import json

import rclpy
import tf2_ros
from arena_rclpy_mixins.params import Param, declare
from arena_rclpy_mixins.spin import spin_node
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.parameter import Parameter

from arena_robots.assembly import RequestPart
from arena_robots.bringup import BRINGUPS, check_caps
from arena_robots.Robot import RobotIdentifier
from arena_robots.task_kinds import action_type, endpoint


class TaskServerNode(Node):
    def __init__(self, *, parameter_overrides: list[Parameter] | None = None) -> None:
        super().__init__("task_server", parameter_overrides=parameter_overrides)

        robot_name = declare(self, Param[str]("robot_name", "", "Robot model name the task server serves (required).")).value
        bringup_caps = declare(self, Param[list[str]]("bringup_caps", [], "Capability of each configured bringup, parallel to bringup_kinds.", type_=Parameter.Type.STRING_ARRAY)).value
        bringup_kinds = declare(self, Param[list[str]]("bringup_kinds", [], "Bringup kind of each configured bringup, parallel to bringup_caps.", type_=Parameter.Type.STRING_ARRAY)).value
        frame = declare(self, Param[str]("frame", "", "Frame prefix of the robot.")).value
        parts_json = declare(self, Param[str]("parts_json", "{}", "JSON object mapping part type to the list of requested robot parts.")).value
        parts = {t: [RequestPart(variant=i["variant"], mount=i.get("mount")) for i in items] for t, items in json.loads(parts_json).items()}

        if not robot_name:
            raise RuntimeError("Parameter 'robot_name' is required")
        if len(bringup_caps) != len(bringup_kinds):
            raise RuntimeError(f"Parameter length mismatch: bringup_caps={len(bringup_caps)} bringup_kinds={len(bringup_kinds)}")
        if not bringup_caps:
            self.get_logger().warning(f"no bringups configured for robot {robot_name!r}; task_server idle")

        namespace = self.get_namespace()
        robot = RobotIdentifier(robot_name).resolve_sync()

        self._tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tf_buffer, self)

        self._bringups: list[object] = []
        self._servers: list[ActionServer] = []

        for cap, kind in zip(bringup_caps, bringup_kinds, strict=True):
            try:
                bringup_cls = BRINGUPS[cap].get(kind)
            except KeyError:
                self.get_logger().error(f"no bringup registered for ({cap!r}, {kind!r}); skipping")
                continue

            try:
                bringup = bringup_cls(robot, namespace, frame=frame, parts=parts)
                check_caps(bringup)
            except Exception as exc:
                self.get_logger().error(f"bringup ({cap!r}, {kind!r}) init failed: {exc}; skipping")
                continue

            self._bringups.append(bringup)

            for tk in bringup.accepts_task_kinds:
                loader = bringup.task_handlers.get(tk)
                if loader is None:
                    self.get_logger().warning(f"no handler for ({tk!r}, {kind!r}); skipping endpoint")
                    continue
                handler_cls = loader()
                try:
                    handler = handler_cls(bringup, tf_buffer=self._tf_buffer, node=self)
                    server = ActionServer(
                        self,
                        action_type(tk),
                        endpoint(namespace, tk),
                        execute_callback=handler.execute,
                        cancel_callback=handler.on_cancel,
                    )
                except Exception as exc:
                    self.get_logger().error(f"handler ({tk!r}, {kind!r}) init failed: {exc}; skipping endpoint")
                    continue
                self._servers.append(server)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    spin_node(TaskServerNode())


if __name__ == "__main__":
    main()
