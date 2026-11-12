"""Tests for arena_robots.Sensor topic derivation (topic_elements, bridge_rows)."""

from __future__ import annotations

import pytest
from arena_robots.Sensor import BRIDGE_TYPES, SensorSpec, bridge_rows, topic_elements


def _spec(name: str, type_: str, topic: str, sensor: str | None) -> SensorSpec:
    return SensorSpec(name=name, type=type_, topic=topic, frame=f"{name}_frame", sensor=sensor)


LIDAR_SPECS = [
    _spec("lidar", "laserscan", "${namespace}/scan", "front_laser"),
    _spec("lidar_points", "pointcloud", "${namespace}/scan/points", "front_laser"),
]

RGBD_SPECS = [
    _spec("camera_image", "image", "${namespace}/camera/image", "camera_sensor"),
    _spec("camera_depth", "depth", "${namespace}/camera/depth_image", "camera_sensor"),
    _spec("camera_points", "pointcloud", "${namespace}/camera/points", "camera_sensor"),
    _spec("camera_info", "camera_info", "${namespace}/camera/camera_info", "camera_sensor"),
]

IMU_SPECS = [
    _spec("imu", "imu", "${namespace}/imu/data", "imu_sensor"),
]

CONTACT_SPECS = [_spec("bumper", "contact", "${namespace}/bumper", "bumper_sensor")]

RBTHERON_SPECS = [
    _spec("lidar", "laserscan", "${namespace}/scan", "front_laser"),
    _spec("lidar_points", "pointcloud", "${namespace}/scan/points", "front_laser"),
    _spec("lidar_rear", "laserscan", "${namespace}/scan/rear", "rear_laser"),
    _spec("lidar_rear_points", "pointcloud", "${namespace}/scan/rear/points", "rear_laser"),
    _spec("imu", "imu", "${namespace}/imu/data", "imu_sensor"),
    _spec("front_camera_image", "image", "${namespace}/front_camera/image", "front_camera_color"),
    _spec("front_camera_info", "camera_info", "${namespace}/front_camera/camera_info", "front_camera_color"),
    _spec("rear_camera_image", "image", "${namespace}/rear_camera/image", "rear_camera_color"),
    _spec("rear_camera_info", "camera_info", "${namespace}/rear_camera/camera_info", "rear_camera_color"),
]


class TestTopicElementsLidar:
    def test_relative_prefix(self):
        assert topic_elements(LIDAR_SPECS, "") == [("front_laser", "topic", "scan")]

    def test_absolute_prefix(self):
        assert topic_elements(LIDAR_SPECS, "/model/env_0/x") == [
            ("front_laser", "topic", "/model/env_0/x/scan"),
        ]


class TestTopicElementsRgbd:
    def test_relative_prefix(self):
        assert topic_elements(RGBD_SPECS, "") == [
            ("camera_sensor", "topic", "camera"),
            ("camera_sensor", "camera/camera_info_topic", "camera/camera_info"),
        ]

    def test_absolute_prefix(self):
        assert topic_elements(RGBD_SPECS, "/model/env_0/x") == [
            ("camera_sensor", "topic", "/model/env_0/x/camera"),
            ("camera_sensor", "camera/camera_info_topic", "/model/env_0/x/camera/camera_info"),
        ]


class TestTopicElementsImu:
    def test_relative_prefix(self):
        assert topic_elements(IMU_SPECS, "") == [("imu_sensor", "topic", "imu/data")]

    def test_absolute_prefix(self):
        assert topic_elements(IMU_SPECS, "/model/env_0/x") == [
            ("imu_sensor", "topic", "/model/env_0/x/imu/data"),
        ]


class TestTopicElementsContact:
    def test_relative_prefix(self):
        assert topic_elements(CONTACT_SPECS, "") == [
            ("bumper_sensor", "contact/topic", "bumper"),
        ]

    def test_absolute_prefix(self):
        assert topic_elements(CONTACT_SPECS, "/model/env_0/x") == [
            ("bumper_sensor", "contact/topic", "/model/env_0/x/bumper"),
        ]


class TestTopicElementsMismatch:
    def test_mismatched_pointcloud_topic_warns_but_still_patches(self, capsys: pytest.CaptureFixture[str]):
        specs = [
            _spec("lidar", "laserscan", "${namespace}/scan", "front_laser"),
            _spec("lidar_points", "pointcloud", "${namespace}/scan_points", "front_laser"),
        ]
        patches = topic_elements(specs, "")
        assert patches == [("front_laser", "topic", "scan")]
        err = capsys.readouterr().err
        assert "front_laser" in err
        assert "pointcloud" in err
        assert "scan_points" in err
        assert "scan/points" in err


class TestBridgeRows:
    def test_rbtheron_like_spec_set(self):
        rows = bridge_rows(RBTHERON_SPECS, "")
        assert rows == [
            ("scan", "scan", "sensor_msgs/msg/LaserScan", "gz.msgs.LaserScan"),
            ("scan/points", "scan/points", "sensor_msgs/msg/PointCloud2", "gz.msgs.PointCloudPacked"),
            ("scan/rear", "scan/rear", "sensor_msgs/msg/LaserScan", "gz.msgs.LaserScan"),
            ("scan/rear/points", "scan/rear/points", "sensor_msgs/msg/PointCloud2", "gz.msgs.PointCloudPacked"),
            ("imu/data", "imu/data", "sensor_msgs/msg/Imu", "gz.msgs.IMU"),
            ("front_camera/image", "front_camera/image", "sensor_msgs/msg/Image", "gz.msgs.Image"),
            ("front_camera/camera_info", "front_camera/camera_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"),
            ("rear_camera/image", "rear_camera/image", "sensor_msgs/msg/Image", "gz.msgs.Image"),
            ("rear_camera/camera_info", "rear_camera/camera_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"),
        ]

    def test_rbtheron_like_spec_set_absolute_prefix(self):
        rows = bridge_rows(RBTHERON_SPECS, "/model/env_0/x")
        gz_topics = [r[0] for r in rows]
        ros_topics = [r[1] for r in rows]
        assert gz_topics == [
            "/model/env_0/x/scan",
            "/model/env_0/x/scan/points",
            "/model/env_0/x/scan/rear",
            "/model/env_0/x/scan/rear/points",
            "/model/env_0/x/imu/data",
            "/model/env_0/x/front_camera/image",
            "/model/env_0/x/front_camera/camera_info",
            "/model/env_0/x/rear_camera/image",
            "/model/env_0/x/rear_camera/camera_info",
        ]
        assert ros_topics == [
            "scan",
            "scan/points",
            "scan/rear",
            "scan/rear/points",
            "imu/data",
            "front_camera/image",
            "front_camera/camera_info",
            "rear_camera/image",
            "rear_camera/camera_info",
        ]

    def test_sensorless_spec_produces_no_rows(self):
        specs = [_spec("orphan_image", "image", "${namespace}/orphan/image", None)]
        assert bridge_rows(specs, "") == []

    def test_contact_type_produces_a_bridged_row(self):
        specs = [_spec("bumper", "contact", "${namespace}/bumper", "bumper_sensor")]
        assert bridge_rows(specs, "") == [
            ("bumper", "bumper", "ros_gz_interfaces/msg/Contacts", "gz.msgs.Contacts"),
        ]

    def test_contact_type_produces_a_bridged_row_with_absolute_prefix(self):
        specs = [_spec("bumper", "contact", "${namespace}/bumper", "bumper_sensor")]
        assert bridge_rows(specs, "/model/env_0/x") == [
            ("/model/env_0/x/bumper", "bumper", "ros_gz_interfaces/msg/Contacts", "gz.msgs.Contacts"),
        ]

    def test_two_finger_contacts_are_two_rows(self):
        specs = [
            _spec("left_finger", "contact", "${namespace}/gripper_left_finger/contact", "left_contact"),
            _spec("right_finger", "contact", "${namespace}/gripper_right_finger/contact", "right_contact"),
        ]
        assert bridge_rows(specs, "") == [
            ("gripper_left_finger/contact", "gripper_left_finger/contact", "ros_gz_interfaces/msg/Contacts", "gz.msgs.Contacts"),
            ("gripper_right_finger/contact", "gripper_right_finger/contact", "ros_gz_interfaces/msg/Contacts", "gz.msgs.Contacts"),
        ]

    def test_all_bridge_kinds_are_in_bridge_types(self):
        rows = bridge_rows(RBTHERON_SPECS, "")
        for _gz_topic, _ros_topic, ros_type, gz_type in rows:
            assert (ros_type, gz_type) in BRIDGE_TYPES.values()
