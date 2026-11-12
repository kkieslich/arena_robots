"""Canonical sensor-type vocabulary shared across arena_robots and arena_planners."""

from __future__ import annotations

SENSOR_TYPES: frozenset[str] = frozenset(
    {
        "laserscan",
        "pointcloud",
        "image",
        "depth",
        "imu",
        "contact",
        "camera_info",
        "audio",
    }
)


def output_topics(sensor_type: str, topic: str, camera_info_topic: str | None = None) -> dict[str, str] | None:
    """Topics a gazebo sensor of `sensor_type` publishes given an explicit <topic>.

    Keys are SENSOR_TYPES entries, None for types without a fixed layout.
    """
    parent = topic.rsplit('/', 1)[0] if '/' in topic else ''
    if sensor_type in ('gpu_lidar', 'ray'):
        return {'laserscan': topic, 'pointcloud': f'{topic}/points'}
    if sensor_type == 'imu':
        return {'imu': topic}
    if sensor_type == 'contact':
        return {'contact': topic}
    if sensor_type == 'camera':
        return {'image': topic, 'camera_info': camera_info_topic or f'{parent}/camera_info'}
    if sensor_type == 'depth':
        return {'depth': topic, 'pointcloud': f'{topic}/points', 'camera_info': camera_info_topic or f'{parent}/camera_info'}
    if sensor_type == 'rgbd_camera':
        return {
            'image': f'{topic}/image',
            'depth': f'{topic}/depth_image',
            'pointcloud': f'{topic}/points',
            'camera_info': camera_info_topic or f'{topic}/camera_info',
        }
    return None
