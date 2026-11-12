"""Obstacle cloud node of the meshnav mobile backend."""

from __future__ import annotations

import array
import functools

import numpy as np
import rclpy
from arena_rclpy_mixins.params import Param, Range, declare
from arena_rclpy_mixins.spin import spin_node
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener

from arena_robots.meshnav_obstacles import ObstacleFilter, WalkingSurface

OBSTACLES_TOPIC = "meshnav/obstacles"
_FLOAT_KINDS = {PointField.FLOAT32: "f4", PointField.FLOAT64: "f8"}


def cloud_points(msg: PointCloud2) -> np.ndarray:
    """The x, y, z of every point of a cloud with float coordinates."""
    fields = {field.name: field for field in msg.fields}
    order = ">" if msg.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [f"{order}{_FLOAT_KINDS[fields[axis].datatype]}" for axis in "xyz"],
            "offsets": [fields[axis].offset for axis in "xyz"],
            "itemsize": msg.point_step,
        }
    )
    rows = np.asarray(msg.data, dtype=np.uint8).reshape(msg.height, msg.row_step)[:, : msg.width * msg.point_step]
    block = np.ascontiguousarray(rows).view(dtype).reshape(-1)
    return np.stack([block["x"], block["y"], block["z"]], axis=1).astype(np.float64)


def scan_points(msg: LaserScan) -> np.ndarray:
    """The returns of a scan within its valid range as points in its frame."""
    ranges = np.asarray(msg.ranges, dtype=np.float64)
    angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
    valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
    ranges, angles = ranges[valid], angles[valid]
    return np.stack([ranges * np.cos(angles), ranges * np.sin(angles), np.zeros(len(ranges))], axis=1)


def cloud_message(header: Header, points: np.ndarray) -> PointCloud2:
    """An unordered float32 x, y, z cloud of `points`."""
    fields = [PointField(name=axis, offset=4 * index, datatype=PointField.FLOAT32, count=1) for index, axis in enumerate("xyz")]
    data = array.array("B", np.ascontiguousarray(points, dtype="<f4").tobytes())
    return PointCloud2(header=header, height=1, width=len(points), fields=fields, is_bigendian=False, point_step=12, row_step=12 * len(points), data=data, is_dense=True)


def transform_matrix(transform: TransformStamped) -> np.ndarray:
    """The 4x4 homogeneous matrix of a transform."""
    q, t = transform.transform.rotation, transform.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


class MeshNavObstacles(Node):
    """Publishes the merged obstacle points of the robot's range sensors in the odom frame, ground the nav mesh explains dropped."""

    def __init__(self, *, parameter_overrides: list[Parameter] | None = None) -> None:
        super().__init__("meshnav_obstacles", parameter_overrides=parameter_overrides)

        self._odom_frame = declare(self, Param[str]("odom_frame", "odom", "Frame the obstacle points are published in.")).value
        self._base_frame = declare(self, Param[str]("base_frame", "base_link", "Base link frame of the robot.")).value
        mesh_file = declare(self, Param[str]("mesh_file", "", "Path of the PLY nav mesh the walkable surface is built from.")).value
        resolution = declare(self, Param[float]("resolution", 0.1, "Grid resolution of the walkable surface in m.", constraint=Range(lo=0.0, lo_open=True))).value
        self._filter = ObstacleFilter(
            surface=WalkingSurface.from_ply(mesh_file, resolution),
            min_height=declare(self, Param[float]("min_obstacle_height", 0.1, "Minimum height above the walkable surface for a point to count as obstacle in m.")).value,
            max_height=declare(self, Param[float]("max_obstacle_height", 0.7, "Maximum height above the walkable surface for a point to count as obstacle in m.")).value,
            max_step=declare(self, Param[float]("max_step", 0.1, "Largest height step the walkable surface explains away in m.", constraint=Range(lo=0.0))).value,
            obstacle_range=declare(self, Param[float]("obstacle_range", 9.0, "Maximum range of the sensor points kept in m.", constraint=Range(lo=0.0, lo_open=True))).value,
            self_radius=declare(self, Param[float]("self_radius", 0.3, "Radius around the robot whose points are dropped in m.", constraint=Range(lo=0.0))).value,
        )
        self._decay = Duration(seconds=declare(self, Param[float]("obstacle_decay", 1.0, "Seconds an obstacle observation is kept after its last update.", constraint=Range(lo=0.0))).value)
        scans = list(declare(self, Param[list[str]]("laserscan_topics", [], "Laserscan topics merged into the obstacle points.", type_=Parameter.Type.STRING_ARRAY)).value)
        clouds = list(declare(self, Param[list[str]]("pointcloud_topics", [], "Pointcloud topics merged into the obstacle points.", type_=Parameter.Type.STRING_ARRAY)).value)
        rate = declare(self, Param[float]("publish_rate", 10.0, "Publish rate of the merged obstacle points in Hz.", constraint=Range(lo=0.0, lo_open=True))).value

        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._pending: dict[str, LaserScan | PointCloud2] = {}
        self._latest: dict[str, tuple[Time, np.ndarray]] = {}
        self._published = True
        self._publisher = self.create_publisher(PointCloud2, OBSTACLES_TOPIC, 1)
        for topic in scans:
            self.create_subscription(LaserScan, topic, functools.partial(self.receive, topic), qos_profile_sensor_data)
        for topic in clouds:
            self.create_subscription(PointCloud2, topic, functools.partial(self.receive, topic), qos_profile_sensor_data)
        self.create_timer(1.0 / rate, self._tick)

    @property
    def tf_buffer(self) -> Buffer:
        """The transforms the scans are placed with."""
        return self._tf

    def receive(self, topic: str, msg: LaserScan | PointCloud2) -> None:
        """Classify a scan of `topic`, or hold it until its transforms arrive."""
        self._pending[topic] = msg
        self._process(topic)

    def _process(self, topic: str) -> None:
        msg = self._pending[topic]
        stamp = Time.from_msg(msg.header.stamp)
        try:
            sensor_in_base = transform_matrix(self._tf.lookup_transform(self._base_frame, msg.header.frame_id, stamp))
            base_in_odom = transform_matrix(self._tf.lookup_transform(self._odom_frame, self._base_frame, stamp))
        except TransformException:
            if self.get_clock().now() - stamp > self._decay:
                del self._pending[topic]
            return
        del self._pending[topic]
        points = scan_points(msg) if isinstance(msg, LaserScan) else cloud_points(msg)
        self._latest[topic] = (stamp, self._filter(points, sensor_in_base, base_in_odom))

    def merged(self) -> np.ndarray:
        """The obstacle points of every sensor's latest scan younger than the decay time."""
        now = self.get_clock().now()
        fresh = [points for stamp, points in self._latest.values() if now - stamp <= self._decay]
        return np.concatenate(fresh) if fresh else np.zeros((0, 3))

    def _tick(self) -> None:
        for topic in list(self._pending):
            self._process(topic)
        points = self.merged()
        if not len(points) and not self._published:
            return
        self._published = bool(len(points))
        self._publisher.publish(cloud_message(Header(stamp=self.get_clock().now().to_msg(), frame_id=self._odom_frame), points))


def main() -> None:
    rclpy.init()
    node = MeshNavObstacles()
    try:
        spin_node(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
