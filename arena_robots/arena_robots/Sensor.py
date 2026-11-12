"""Arena-native sensor declarations shared by robots and adapters."""

from __future__ import annotations

import enum
import sys
from collections.abc import Sequence

import attrs

from arena_robots.sensors import output_topics


class SensorType(enum.StrEnum):
    """Canonical sensor-type vocabulary for the Arena-native sensors schema."""

    LASERSCAN = "laserscan"
    POINTCLOUD = "pointcloud"
    IMAGE = "image"
    DEPTH = "depth"
    CAMERA_INFO = "camera_info"
    IMU = "imu"
    CONTACT = "contact"
    AUDIO = "audio"


@attrs.define(frozen=True)
class SensorSpec:
    """Arena-native sensor declaration."""

    name: str
    type: SensorType | str
    topic: str
    frame: str
    sensor: str | None = None
    """Name of the backing URDF <sensor> element, used for topic injection at load time."""


SensorTypeOrStr = SensorType | str

_NAMESPACE_PLACEHOLDER = '${namespace}'

# ros/gz message types for the gz_ros_bridge, keyed by SensorSpec type. Direction is always gz->ros.
BRIDGE_TYPES: dict[str, tuple[str, str]] = {
    'laserscan': ('sensor_msgs/msg/LaserScan', 'gz.msgs.LaserScan'),
    'pointcloud': ('sensor_msgs/msg/PointCloud2', 'gz.msgs.PointCloudPacked'),
    'imu': ('sensor_msgs/msg/Imu', 'gz.msgs.IMU'),
    'image': ('sensor_msgs/msg/Image', 'gz.msgs.Image'),
    'depth': ('sensor_msgs/msg/Image', 'gz.msgs.Image'),
    'camera_info': ('sensor_msgs/msg/CameraInfo', 'gz.msgs.CameraInfo'),
    'contact': ('ros_gz_interfaces/msg/Contacts', 'gz.msgs.Contacts'),
}


def resolve_topic(spec: SensorSpec, namespace: str = '') -> str:
    """`spec.topic` with `${namespace}` substituted, or robot-relative when `namespace` is empty."""
    topic = spec.topic.replace(_NAMESPACE_PLACEHOLDER, namespace.rstrip('/'))
    return topic if namespace else topic.lstrip('/')


def _join(prefix: str, anchor: str) -> str:
    """Prefix-join a gz topic, staying relative when `prefix` is empty (Isaac's namespacing)."""
    return f'{prefix}/{anchor}' if prefix else anchor


def _grouped_sensors(
    sensors: Sequence[SensorSpec],
) -> list[tuple[str, str, str, dict[str, str], dict[str, str]]]:
    """Group specs by backing <sensor>, classify (sensor_type, anchor) and derive gz output topics.

    Returns `(sensor_name, sensor_type, anchor, outputs, derived)` per group, in sensors-list
    order. `outputs` is the group's declared `{spec type: relative topic}`, `derived` is
    `output_topics(sensor_type, anchor, ...)`. Skips specs with `sensor=None` and groups with
    no decidable anchor, printing the same stderr warnings topic_elements always has.
    """
    grouped: dict[str, dict[str, str]] = {}
    for spec in sensors:
        if spec.sensor is None:
            continue
        grouped.setdefault(spec.sensor, {})[str(spec.type)] = resolve_topic(spec)

    result: list[tuple[str, str, str, dict[str, str], dict[str, str]]] = []
    for sensor_name, outputs in grouped.items():
        if 'laserscan' in outputs:
            sensor_type, anchor = 'gpu_lidar', outputs['laserscan']
        elif 'image' in outputs and ('depth' in outputs or 'pointcloud' in outputs):
            sensor_type, anchor = 'rgbd_camera', outputs['image'].removesuffix('/image')
            if anchor == outputs['image']:
                print(f"[sensor_topics] {sensor_name!r}: rgbd image topic {outputs['image']!r} does not end in /image, skipping", file=sys.stderr)
                continue
        elif 'image' in outputs:
            sensor_type, anchor = 'camera', outputs['image']
        elif 'depth' in outputs:
            sensor_type, anchor = 'depth', outputs['depth']
        elif 'imu' in outputs:
            sensor_type, anchor = 'imu', outputs['imu']
        elif 'contact' in outputs:
            sensor_type, anchor = 'contact', outputs['contact']
        else:
            print(f"[sensor_topics] {sensor_name!r}: no anchor topic among {sorted(outputs)}", file=sys.stderr)
            continue

        derived = output_topics(sensor_type, anchor, outputs.get('camera_info'))
        assert derived is not None
        for kind, topic in outputs.items():
            if kind in derived and derived[kind] != topic:
                print(f"[sensor_topics] {sensor_name!r}: {kind} topic {topic!r} is not {derived[kind]!r}, gz publishes the latter", file=sys.stderr)

        result.append((sensor_name, sensor_type, anchor, outputs, derived))
    return result


def topic_elements(sensors: Sequence[SensorSpec], prefix: str) -> list[tuple[str, str, str]]:
    """Element patches `(sensor name, child element path, value)` for URDF topic injection.

    `prefix` is the per-instance gz topic scope, empty outside gazebo.
    """
    patches: list[tuple[str, str, str]] = []
    for sensor_name, sensor_type, anchor, outputs, _derived in _grouped_sensors(sensors):
        topic_path = 'contact/topic' if sensor_type == 'contact' else 'topic'
        patches.append((sensor_name, topic_path, _join(prefix, anchor)))
        if 'camera_info' in outputs and sensor_type in ('camera', 'depth', 'rgbd_camera'):
            patches.append((sensor_name, 'camera/camera_info_topic', _join(prefix, outputs['camera_info'])))
    return patches


def bridge_rows(sensors: Sequence[SensorSpec], prefix: str) -> list[tuple[str, str, str, str]]:
    """(gz_topic, ros_topic, ros_type, gz_type) for every derivable sensor output, direction gz->ros.

    gz_topic is the injected-<topic>-derived absolute gz topic (prefix-scoped);
    ros_topic is the declared topic with the ${namespace} placeholder stripped (relative,
    the per-robot bridge node's PushRosNamespace scopes it).
    """
    rows: list[tuple[str, str, str, str]] = []
    for _sensor_name, _sensor_type, _anchor, outputs, derived in _grouped_sensors(sensors):
        for kind, ros_topic in outputs.items():
            if kind not in BRIDGE_TYPES:
                continue
            ros_type, gz_type = BRIDGE_TYPES[kind]
            rows.append((_join(prefix, derived[kind]), ros_topic, ros_type, gz_type))
    return rows
