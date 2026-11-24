from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import rclpy
from arena_rclpy_mixins.Time import Time
from arena_simulation_setup.shared.navmesh import NavMesh
from geometry_msgs.msg import TransformStamped
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header


def _flat_mesh(tmp_path: Path) -> Path:
    n = 101
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    vertices = np.stack([ii.ravel() * 0.1 - 5.0, jj.ravel() * 0.1 - 5.0, np.zeros(n * n)], axis=1)
    index = np.arange(n * n).reshape(n, n)
    v00, v10, v11, v01 = (corner.ravel() for corner in (index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]))
    faces = np.concatenate([np.stack([v00, v10, v11], axis=1), np.stack([v00, v11, v01], axis=1)])
    return NavMesh(vertices, faces).write_ply(tmp_path / "flat.ply")


def _transform(parent: str, child: str, z: float, stamp) -> TransformStamped:
    transform = TransformStamped()
    transform.header.frame_id = parent
    transform.header.stamp = stamp
    transform.child_frame_id = child
    transform.transform.translation.z = z
    transform.transform.rotation.w = 1.0
    return transform


def _set_sim_time(node, seconds: float) -> None:
    publisher = node.create_publisher(Clock, "/clock", 10)
    target = Time.from_float(seconds)
    deadline = time.monotonic() + 10.0
    while node.sim_time != target:
        assert time.monotonic() < deadline, "sim time never reached the node"
        publisher.publish(Clock(clock=target.to_msg()))
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_publisher(publisher)


@pytest.fixture()
def node(tmp_path: Path):
    from arena_robots.meshnav_obstacles_node import MeshNavObstacles

    node = MeshNavObstacles(
        parameter_overrides=[
            Parameter("odom_frame", value="odom"),
            Parameter("base_frame", value="base"),
            Parameter("mesh_file", value=str(_flat_mesh(tmp_path))),
            Parameter("self_radius", value=0.3),
            Parameter("obstacle_decay", value=1.0),
            Parameter("laserscan_topics", value=["scan"]),
            Parameter("pointcloud_topics", value=["points"]),
        ]
    )
    _set_sim_time(node, 10.0)
    stamp = node.sim_time.to_msg()
    node.tf_buffer.set_transform_static(_transform("odom", "base", 0.0, stamp), "test")
    node.tf_buffer.set_transform_static(_transform("base", "lidar", 0.3, stamp), "test")
    try:
        yield node
    finally:
        node.destroy_node()


def _cloud(points: np.ndarray, stamp) -> PointCloud2:
    from arena_robots.meshnav_obstacles_node import cloud_message

    return cloud_message(Header(stamp=stamp, frame_id="lidar"), points)


def _scan(stamp) -> LaserScan:
    ranges = [np.inf] * 8
    ranges[0] = 2.0
    ranges[2] = 0.1
    return LaserScan(header=Header(stamp=stamp, frame_id="lidar"), angle_min=0.0, angle_increment=np.pi / 4, range_min=0.05, range_max=10.0, ranges=ranges)


def test_cloud_messages_round_trip_their_points():
    from arena_robots.meshnav_obstacles_node import cloud_points

    points = np.array([[1.0, 2.0, 3.0], [-0.5, 0.25, 0.0]])
    assert np.allclose(cloud_points(_cloud(points, Header().stamp)), points)


def test_scans_become_points_within_their_valid_range():
    from arena_robots.meshnav_obstacles_node import scan_points

    assert np.allclose(scan_points(_scan(Header().stamp)), [[2.0, 0.0, 0.0], [0.0, 0.1, 0.0]], atol=1e-9)


def test_scans_and_clouds_of_every_sensor_merge_into_odom_obstacles(node):
    stamp = node.sim_time.to_msg()
    node.receive("scan", _scan(stamp))
    node.receive("points", _cloud(np.array([[0.0, -3.0, 0.0], [0.0, 2.0, -0.3], [0.1, 0.0, 0.0]]), stamp))
    merged = node.merged()
    assert sorted(map(tuple, np.round(merged, 6))) == [(0.0, -3.0, 0.3), (2.0, 0.0, 0.3)]


def test_a_sensor_s_obstacles_decay_after_the_decay_time(node):
    now = node.sim_time
    node.receive("scan", _scan((now - Time.from_float(2.0)).to_msg()))
    node.receive("points", _cloud(np.array([[0.0, -3.0, 0.0]]), now.to_msg()))
    assert np.allclose(node.merged(), [[0.0, -3.0, 0.3]])


def test_a_scan_waits_for_its_transform(node):
    stamp = node.sim_time.to_msg()
    scan = _scan(stamp)
    scan.header.frame_id = "camera"
    node.receive("scan", scan)
    assert not len(node.merged())
    node.tf_buffer.set_transform_static(_transform("base", "camera", 0.3, stamp), "test")
    node._tick()
    assert np.allclose(node.merged(), [[2.0, 0.0, 0.3]])
