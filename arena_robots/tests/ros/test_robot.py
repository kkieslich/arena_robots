"""Tests for arena_robots.Robot (ModelParams, RobotView)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml


class TestModelParamsDefaults:
    def test_base_frame_explicit(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        mp = ModelParams.from_yaml(_write(tmp_path, {"base_frame": "my_base"}))
        assert mp.base_frame == "my_base"

    def test_base_frame_fallback_robot_base_frame(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        mp = ModelParams.from_yaml(_write(tmp_path, {"robot_base_frame": "rb_frame"}))
        assert mp.base_frame == "rb_frame"

    def test_odom_frame_from_model_params_key(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        mp = ModelParams.from_yaml(_write(tmp_path, {"robot_odom_frame": "my_odom"}))
        assert mp.odom_frame == "my_odom"

    def test_z_offset_explicit(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        mp = ModelParams.from_yaml(_write(tmp_path, {"z_offset": 0.15}))
        assert mp.z_offset == pytest.approx(0.15)


class TestModelParamsFromYaml:
    def test_non_dict_root_raises(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        path = tmp_path / "mp.yaml"
        path.write_text("- item\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            ModelParams.from_yaml(path)

    def test_missing_file_raises(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        with pytest.raises((FileNotFoundError, OSError)):
            ModelParams.from_yaml(tmp_path / "nonexistent.yaml")


class TestModelParamsLights:
    def test_lights_parse_on_robot_frames_and_are_intrinsic(self, tmp_path: Path) -> None:
        from arena_robots.Robot import ModelParams

        data = {"lights": [{"name": "headlight", "fixture": "spot", "frame": "base_link", "offset": {"x": 0.2, "y": 0.0, "z": 0.3, "pitch": -math.pi / 2}, "lumens": 1500}]}
        (light,) = ModelParams.from_yaml(_write(tmp_path, data)).lights
        assert (light.name, light.fixture, light.frame) == ("headlight", "spot", "base_link")
        assert light.intrinsic
        assert light.pose is None
        assert (light.offset.position.x, light.offset.position.z) == (0.2, 0.3)
        assert light.direction == pytest.approx((1.0, 0.0, 0.0))

    def test_no_lights_key_means_no_lights(self, tmp_path: Path) -> None:
        from arena_robots.Robot import ModelParams

        assert ModelParams.from_yaml(_write(tmp_path, {})).lights == []

    @pytest.mark.parametrize(
        ("lights", "match"),
        [
            ("headlight", "must be a list"),
            (["headlight"], "must be a mapping"),
            ([{"name": "headlight", "fixture": "spot", "pose": {"x": 1, "y": 0, "z": 0.5}, "lumens": 900}], "needs a frame of the robot"),
            ([{"name": "headlight", "fixture": "spot", "frame": "base_link", "pose": {"x": 1, "y": 0, "z": 0.5}, "lumens": 900}], "exactly one of pose, attach or frame"),
            ([{"name": "headlight", "fixture": "spot", "frame": "base_link", "lumens": 900}, {"name": "headlight", "fixture": "bulb", "frame": "top_link", "lumens": 400}], "declares 'headlight' twice"),
        ],
    )
    def test_malformed_lights_raise(self, tmp_path: Path, lights: object, match: str) -> None:
        from arena_robots.Robot import ModelParams

        with pytest.raises(ValueError, match=match):
            ModelParams.from_yaml(_write(tmp_path, {"lights": lights})).lights


class TestModelParamsSensors:
    def test_sensors_valid(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        data = {"sensors": [{"name": "laser", "type": "laserscan", "topic": "/scan", "frame": "laser_frame"}]}
        mp = ModelParams.from_yaml(_write(tmp_path, data))
        sensors = mp.sensors
        assert len(sensors) == 1
        assert sensors[0].name == "laser"
        assert sensors[0].topic == "/scan"
        assert sensors[0].frame == "laser_frame"

    def test_sensors_missing_keys_raises(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        data = {"sensors": [{"name": "laser"}]}
        mp = ModelParams.from_yaml(_write(tmp_path, data))
        with pytest.raises(ValueError, match="missing required keys"):
            _ = mp.sensors

    def test_sensors_non_list_raises(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        mp = ModelParams.from_yaml(_write(tmp_path, {"sensors": "not_a_list"}))
        with pytest.raises(ValueError, match="must be a list"):
            _ = mp.sensors

    def test_sensors_non_dict_entry_raises(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        mp = ModelParams.from_yaml(_write(tmp_path, {"sensors": ["not_a_dict"]}))
        with pytest.raises(ValueError, match="must be a mapping"):
            _ = mp.sensors

    def test_multiple_sensors(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        data = {
            "sensors": [
                {"name": "laser", "type": "laserscan", "topic": "/scan", "frame": "lf"},
                {"name": "cam", "type": "image", "topic": "/camera/image", "frame": "cf"},
            ]
        }
        mp = ModelParams.from_yaml(_write(tmp_path, data))
        sensors = mp.sensors
        assert len(sensors) == 2
        assert sensors[1].name == "cam"


class TestModelParamsCaps:
    def test_caps_with_mobile_yaml(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        robot_dir = tmp_path / "robot"
        robot_dir.mkdir()
        caps_dir = robot_dir / "caps"
        caps_dir.mkdir()
        (caps_dir / "mobile.yaml").write_text(yaml.dump({"odom_frame": "custom_odom"}))
        mp_path = robot_dir / "model_params.yaml"
        mp_path.write_text(yaml.dump({}))
        mp = ModelParams.from_yaml(mp_path)
        assert "mobile" in mp.caps.available
        assert mp.odom_frame == "custom_odom"

    def test_caps_no_caps_dir(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        mp = ModelParams.from_yaml(_write(tmp_path, {}))
        assert mp.caps.available == frozenset()

    def test_caps_property_cached(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        mp = ModelParams.from_yaml(_write(tmp_path, {}))
        assert mp.caps is mp.caps


class TestModelParamsCapabilities:
    def test_capabilities_valid(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        data = {"capabilities": [{"kind": "mobile"}, {"kind": "arm"}]}
        mp = ModelParams.from_yaml(_write(tmp_path, data))
        assert len(mp.capabilities) == 2
        assert mp.capabilities[0]["kind"] == "mobile"

    def test_capabilities_non_list_raises(self, tmp_path: Path):
        from arena_robots.Robot import ModelParams

        mp = ModelParams.from_yaml(_write(tmp_path, {"capabilities": "bad"}))
        with pytest.raises(ValueError, match="must be a list"):
            _ = mp.capabilities


class TestRobotView:
    def test_model_params_loaded(self, temp_robot_dir):
        from arena_robots.Robot import RobotView
        from arena_simulation_setup.tree import PathView

        rd = temp_robot_dir(model_params={"base_frame": "bl"}, control={"cmd_vel": "/cmd"})
        view = RobotView(rd)
        assert view.model_params.base_frame == "bl"

    def test_model_params_cached(self, temp_robot_dir):
        from arena_robots.Robot import RobotView

        rd = temp_robot_dir(model_params={}, control={})
        view = RobotView(rd)
        assert view.model_params is view.model_params

    def test_model_params_missing_raises(self, tmp_path: Path):
        from arena_robots.Robot import RobotView

        rd = tmp_path / "norobot"
        rd.mkdir()
        view = RobotView(rd)
        with pytest.raises(FileNotFoundError, match="model_params.yaml"):
            _ = view.model_params

    def test_control_loaded(self, temp_robot_dir):
        from arena_robots.Robot import RobotView

        rd = temp_robot_dir(model_params={}, control={"cmd_vel": "/cmd_vel"})
        view = RobotView(rd)
        assert view.control["cmd_vel"] == "/cmd_vel"

    def test_control_cached(self, temp_robot_dir):
        from arena_robots.Robot import RobotView

        rd = temp_robot_dir(model_params={}, control={})
        view = RobotView(rd)
        assert view.control is view.control

    def test_control_missing_raises(self, tmp_path: Path):
        from arena_robots.Robot import RobotView

        rd = tmp_path / "norobot2"
        rd.mkdir()
        (rd / "model_params.yaml").write_text("{}")
        view = RobotView(rd)
        with pytest.raises(FileNotFoundError, match="control.yaml"):
            _ = view.control

    def test_control_non_dict_raises(self, tmp_path: Path):
        from arena_robots.Robot import RobotView

        rd = tmp_path / "robot_bad_ctrl"
        rd.mkdir()
        (rd / "model_params.yaml").write_text("{}")
        (rd / "control.yaml").write_text("- list_item\n")
        view = RobotView(rd)
        with pytest.raises(ValueError, match="must contain a dictionary"):
            _ = view.control

    def test_mappings_path(self, temp_robot_dir):
        from arena_robots.Robot import RobotView

        rd = temp_robot_dir(model_params={}, control={})
        view = RobotView(rd)
        assert view.mappings == str(rd / "mappings.yaml")

    def test_caps_delegates_to_model_params(self, temp_robot_dir):
        from arena_robots.Robot import RobotView

        rd = temp_robot_dir(model_params={}, control={}, mobile_cap={"odom_frame": "odom"})
        view = RobotView(rd)
        assert "mobile" in view.caps.available

    def test_power_profile_with_assembly_and_telemetry_fallbacks(self, temp_robot_dir):
        from arena_robots.Robot import RobotView

        rd = temp_robot_dir(
            model_params={"mass": {"base_kg": 18.0}},
            control={},
        )
        (rd / "assembly.yaml").write_text(
            yaml.dump(
                {
                    "mounts": {},
                    "defaults": {},
                    "power": {
                        "compute_core_w": 30.0,
                        "idle_motors_w": 5.0,
                        "drivetrain_efficiency": 0.70,
                        "heating_coefficient_ch": 1.25,
                        "battery_capacity_wh": 270.0,
                    },
                }
            )
        )
        # Add telemetry/power.yaml with physical losses
        telem_dir = rd / "telemetry"
        telem_dir.mkdir()
        (telem_dir / "power.yaml").write_text(
            yaml.dump(
                {
                    "power_system": {
                        "drivetrain_damping": 0.006,
                        "drivetrain_friction": 0.050,
                        "rolling_resistance_crr": 0.025,
                        "robot_mass_kg": 17.0,
                        "wheel_radius_m": 0.098,
                        "num_wheels": 4,
                    }
                }
            )
        )

        view = RobotView(rd)
        profile = view.power_profile({})
        assert profile is not None
        assert profile["static_power_w"] == 35.0
        assert profile["drivetrain_efficiency"] == 0.70
        assert profile["heating_coefficient_ch"] == 1.25
        assert profile["battery_capacity_wh"] == 270.0
        assert profile["drivetrain_damping"] == 0.006
        assert profile["drivetrain_friction"] == 0.050
        assert profile["rolling_resistance_crr"] == 0.025
        assert profile["robot_mass_kg"] == 17.0
        assert profile["wheel_radius_m"] == 0.098
        assert profile["num_wheels"] == 4

    def test_power_profile_from_standalone_telemetry_yaml(self, temp_robot_dir):
        from arena_robots.Robot import RobotView

        rd = temp_robot_dir(
            model_params={"mass": {"base_kg": 17.0}},
            control={},
        )
        telem_dir = rd / "telemetry"
        telem_dir.mkdir()
        (telem_dir / "power.yaml").write_text(
            yaml.dump(
                {
                    "robot_model": "test_robot",
                    "power_system": {
                        "global_drivetrain_efficiency": 0.85,
                        "heating_coefficient_ch": 0.05,
                        "battery_capacity_wh": 270.0,
                        "drivetrain_damping": 0.15,
                        "drivetrain_friction": 0.65,
                        "rolling_resistance_crr": 0.035,
                        "robot_mass_kg": 17.0,
                        "wheel_radius_m": 0.098,
                        "num_wheels": 4,
                    },
                    "static_power_w": {
                        "compute_core": 20.0,
                        "idle_motors": 15.0,
                    },
                }
            )
        )

        view = RobotView(rd)
        profile = view.power_profile({})
        assert profile is not None
        assert profile["static_power_w"] == 35.0
        assert profile["drivetrain_efficiency"] == 0.85
        assert profile["heating_coefficient_ch"] == 0.05
        assert profile["battery_capacity_wh"] == 270.0
        assert profile["drivetrain_damping"] == 0.15
        assert profile["drivetrain_friction"] == 0.65
        assert profile["rolling_resistance_crr"] == 0.035
        assert profile["robot_mass_kg"] == 17.0
        assert profile["wheel_radius_m"] == 0.098
        assert profile["num_wheels"] == 4
        assert profile["odometry_linear_w_per_mps"] == 0.0
        assert profile["odometry_angular_w_per_radps"] == 0.0

    def test_power_profile_takes_the_odometry_drive_power_coefficients(self, temp_robot_dir):
        from arena_robots.Robot import RobotView

        rd = temp_robot_dir(model_params={}, control={})
        telem_dir = rd / "telemetry"
        telem_dir.mkdir()
        (telem_dir / "power.yaml").write_text(
            yaml.dump(
                {
                    "static_power_w": 35.0,
                    "odometry_drive_power": {"linear_w_per_mps": 50.0, "angular_w_per_radps": 17.0},
                }
            )
        )

        profile = RobotView(rd).power_profile({})
        assert profile is not None
        assert profile["static_power_w"] == 35.0
        assert profile["odometry_linear_w_per_mps"] == 50.0
        assert profile["odometry_angular_w_per_radps"] == 17.0

    def test_the_jackal_estimates_its_driving_power_from_odometry(self):
        from arena_robots.Robot import RobotView

        # this checkout's jackal, not the installed share
        profile = RobotView(Path(__file__).resolve().parents[2] / "robots" / "jackal").power_profile({})
        assert profile is not None
        assert profile["battery_capacity_wh"] == 270.0
        assert (profile["odometry_linear_w_per_mps"], profile["odometry_angular_w_per_radps"]) == (50.0, 17.0)


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "model_params.yaml"
    path.write_text(yaml.dump(data))
    return path
