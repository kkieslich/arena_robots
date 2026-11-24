from __future__ import annotations

from pathlib import Path

import launch
import pytest
import yaml
from arena_rclpy_mixins.yaml_replace import YAMLReplacer
from arena_robots.nav2 import Nav2KinematicsDerivedYAML

PKG = Path(__file__).resolve().parents[2] / "arena_robots"
ROBOTS_DIR = PKG / "robots"
NAV2_YAML = PKG / "config" / "nav2" / "nav2.yaml"


def _smoother(robot: str) -> dict:
    derived = Nav2KinematicsDerivedYAML(str(ROBOTS_DIR / robot / "caps" / "mobile.yaml")).perform(launch.LaunchContext())
    variables = yaml.safe_load(Path(derived).read_text())
    block = yaml.safe_load(NAV2_YAML.read_text())["velocity_smoother"]["ros__parameters"]
    return YAMLReplacer(variables).replace(block)


def _all_floats(block: dict) -> None:
    for key in ("max_velocity", "min_velocity", "max_accel", "max_decel"):
        assert len(block[key]) == 3, key
        assert all(type(v) is float for v in block[key]), (key, block[key])


@pytest.mark.parametrize(
    ("robot", "max_velocity", "min_velocity", "max_accel", "max_decel"),
    [
        ("jackal", [2.0, 0.0, 4.0], [-2.0, 0.0, -4.0], [2.5, 0.0, 3.2], [-2.5, 0.0, -3.2]),
        ("mpo700", [0.8, 0.0, 1.0], [-0.8, 0.0, -1.0], [2.5, 0.0, 3.2], [-2.5, 0.0, -3.2]),
        ("husky", [1.0, 0.0, 2.0], [-1.0, 0.0, -2.0], [2.5, 0.0, 3.2], [-2.5, 0.0, -3.2]),
        ("rskomnidirectional", [0.5, 0.0, 1.5], [-0.5, 0.0, -1.5], [2.5, 0.0, 3.2], [-2.5, 0.0, -3.2]),
    ],
)
def test_velocity_smoother_follows_caps(robot, max_velocity, min_velocity, max_accel, max_decel):
    block = _smoother(robot)
    _all_floats(block)
    assert block["max_velocity"] == max_velocity
    assert block["min_velocity"] == min_velocity
    assert block["max_accel"] == max_accel
    assert block["max_decel"] == max_decel
