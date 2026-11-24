from __future__ import annotations

from pathlib import Path

import launch
import pytest
import yaml
from arena_rclpy_mixins.yaml_replace import YAMLReplacer
from arena_robots.nav2 import Nav2KinematicsDerivedYAML

PKG = Path(__file__).resolve().parents[2] / "arena_robots"
ROBOTS_DIR = PKG / "robots"
MPPI_YAML = PKG / "config" / "nav2" / "controllers" / "mppi" / "controller_config.yaml"
SMAC_HYBRID_YAML = PKG / "config" / "nav2" / "planners" / "smac_hybrid" / "planner_config.yaml"


def _derived(mobile_yaml: Path, config: Path) -> dict:
    derived = Nav2KinematicsDerivedYAML(str(mobile_yaml)).perform(launch.LaunchContext())
    return YAMLReplacer(yaml.safe_load(Path(derived).read_text())).replace(yaml.safe_load(config.read_text()))


def _mppi(mobile_yaml: Path) -> dict:
    return _derived(mobile_yaml, MPPI_YAML)["controller_plugins_dict"]["FollowPath"]


def _smac_hybrid(mobile_yaml: Path) -> dict:
    return _derived(mobile_yaml, SMAC_HYBRID_YAML)["planner_plugins_dict"]["GridBased"]


@pytest.mark.parametrize(
    ("robot", "motion_model", "mppi_radius", "smac_radius"),
    [
        ("jackal", "DiffDrive", 0.2, 0.4),
        ("mpo700", "Omni", 0.2, 0.4),
        ("f1tenth", "Ackermann", 0.7416, 0.7416),
    ],
)
def test_mppi_and_smac_hybrid_follow_the_caps_drive_kind(robot: str, motion_model: str, mppi_radius: float, smac_radius: float) -> None:
    mobile_yaml = ROBOTS_DIR / robot / "caps" / "mobile.yaml"
    mppi = _mppi(mobile_yaml)
    assert mppi["motion_model"] == motion_model
    assert mppi["AckermannConstraints"]["min_turning_r"] == mppi_radius
    assert _smac_hybrid(mobile_yaml)["minimum_turning_radius"] == smac_radius


def test_explicit_zero_turning_radius_keeps_the_config_defaults(tmp_path: Path) -> None:
    mobile_yaml = tmp_path / "mobile.yaml"
    mobile_yaml.write_text("is_holonomic: false\nmin_turning_radius: 0.0\n")
    assert _mppi(mobile_yaml)["motion_model"] == "DiffDrive"
    assert _mppi(mobile_yaml)["AckermannConstraints"]["min_turning_r"] == 0.2
    assert _smac_hybrid(mobile_yaml)["minimum_turning_radius"] == 0.4
