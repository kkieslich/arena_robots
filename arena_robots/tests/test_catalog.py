"""Tests for arena_robots.catalog: component.yaml loading + sensor-template
rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_robots.assembly import Mount, Placement, ResolvedAssembly, apply_frame_overrides
from arena_robots.catalog import Catalog, ComponentSpec, render_control_joints, render_effective_control, render_effective_sensors, render_wrapper_xacro, resolve_mount_parent
from arena_robots.Robot import RobotView

LIDAR_COMPONENT = {
    "xacro": {
        "include": "sick_s300.xacro",
        "macro": "sick_s300",
        "attach": {"parent": "${prefix}${mount}_mount"},
        "args": {"min_angle": -2.269, "max_angle": 2.269},
    },
    "sensor": {
        "gz": [
            {"name": "${name:-lidar}", "type": "laserscan", "topic": "${topic:-scan}", "frame": "${prefix}${mount}_link", "sensor": "${mount}"},
            {"name": "${name:-lidar}_points", "type": "pointcloud", "topic": "${topic:-scan}/points", "frame": "${prefix}${mount}_link", "sensor": "${mount}"},
        ]
    },
}

CAMERA_COMPONENT = {
    "xacro": {
        "include": "d435.xacro",
        "macro": "d435_rgbd",
        "attach": {"parent": "${prefix}${mount}_mount"},
        "args": {},
    },
    "sensor": {
        "gz": [
            {
                "name": "${name:-${mount}}_camera_image",
                "type": "image",
                "topic": "${topic:-${mount}}_camera/image",
                "frame": "${prefix}${mount}_rgbd_camera_color_frame",
                "sensor": "${mount}_camera_color",
            },
            {
                "name": "${name:-${mount}}_camera_info",
                "type": "camera_info",
                "topic": "${topic:-${mount}}_camera/camera_info",
                "frame": "${prefix}${mount}_rgbd_camera_color_frame",
                "sensor": "${mount}_camera_color",
            },
        ]
    },
}


ARM_COMPONENT = {
    "xacro": {
        "include": "ur_macro.xacro",
        "macro": "ur_robot",
        "attach": {"parent": "${prefix}${mount}_parent"},
        "args": {},
    },
    "ros2_control": {
        "joints": [
            {"name": "${prefix}${mount}_shoulder_pan_joint", "command_interfaces": ["position"], "state_interfaces": ["position", "velocity"]},
            {"name": "${prefix}${mount}_shoulder_lift_joint", "command_interfaces": ["position"], "state_interfaces": ["position", "velocity"]},
        ]
    },
    "control": {
        "controller": "${mount}_controller",
        "type": "joint_trajectory_controller/JointTrajectoryController",
        "ros__parameters": {
            "joints": ["${prefix}${mount}_shoulder_pan_joint", "${prefix}${mount}_shoulder_lift_joint"],
            "command_interfaces": ["position"],
            "state_interfaces": ["position", "velocity"],
            "state_publish_rate": 50.0,
            "action_monitor_rate": 20.0,
        },
    },
    "caps": {
        "base_link": "${prefix}${mount}_base_link",
        "tip_link": "${prefix}${mount}_tool0",
    },
}


def _write_component(root: Path, type_: str, variant: str, data: dict) -> Path:
    d = root / type_ / variant
    d.mkdir(parents=True, exist_ok=True)
    path = d / "component.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def _mount(name: str, accepts: list[str], parent: str = "base_link") -> Mount:
    return Mount(name=name, parent=parent, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), accepts=frozenset(accepts))


LIFT_COMPONENT = {
    "xacro": {
        "include": "ewellix_lift_900mm.xacro",
        "macro": "ewellix_lift_900mm",
        "attach": {"prefix": "${prefix}${mount}_", "parent": "${prefix}${parent}"},
    },
    "frames": {"top": "${prefix}${mount}_ewellix_lift_top_link"},
}


def _write_robot_view(tmp_path: Path, name: str = "testbot") -> RobotView:
    """Minimal on-disk robot dir: just enough for ``render_wrapper_xacro`` (a chassis
    xacro with one ``xacro:arg``). ``RobotView`` takes a bare path
    (arena_simulation_setup.tree.PathView); no ament index / ROS environment involved."""
    robot_dir = tmp_path / "robots" / name
    urdf_dir = robot_dir / "urdf"
    urdf_dir.mkdir(parents=True)
    (urdf_dir / f"{name}.urdf.xacro").write_text(f'<?xml version="1.0"?>\n<robot name="{name}" xmlns:xacro="http://www.ros.org/wiki/xacro">\n  <xacro:arg name="prefix" default="robot_"/>\n</robot>\n')
    return RobotView(robot_dir)


class TestComponentSpecFromYaml:
    def test_parses_xacro_and_sensor_gz(self, tmp_path: Path):
        path = _write_component(tmp_path, "lidar", "sick_s300", LIDAR_COMPONENT)
        spec = ComponentSpec.from_yaml(path)
        assert spec.xacro_include == "sick_s300.xacro"
        assert spec.xacro_macro == "sick_s300"
        assert spec.args == {"min_angle": -2.269, "max_angle": 2.269}
        assert len(spec.sensor_gz) == 2
        assert spec.sensor_gz[0]["type"] == "laserscan"

    def test_rejects_missing_xacro_macro(self, tmp_path: Path):
        path = _write_component(tmp_path, "lidar", "broken", {"xacro": {"include": "x.xacro"}})
        with pytest.raises(ValueError, match="xacro"):
            ComponentSpec.from_yaml(path)

    def test_parses_ros2_control_control_caps(self, tmp_path: Path):
        path = _write_component(tmp_path, "arm", "ur5e", ARM_COMPONENT)
        spec = ComponentSpec.from_yaml(path)
        assert len(spec.ros2_control_joints) == 2
        assert spec.ros2_control_joints[0]["name"] == "${prefix}${mount}_shoulder_pan_joint"
        assert spec.control["controller"] == "${mount}_controller"
        assert spec.control["ros__parameters"]["state_publish_rate"] == 50.0
        assert spec.caps["tip_link"] == "${prefix}${mount}_tool0"

    def test_ros2_control_control_caps_default_empty(self, tmp_path: Path):
        path = _write_component(tmp_path, "lidar", "sick_s300", LIDAR_COMPONENT)
        spec = ComponentSpec.from_yaml(path)
        assert spec.ros2_control_joints == []
        assert spec.control == {}
        assert spec.caps == {}


class TestComponentSpecVariants:
    """A family component declares the variant names it serves via `variants:` (empty for
    ordinary one-dir-per-variant components)."""

    def test_variants_default_empty(self, tmp_path: Path):
        path = _write_component(tmp_path, "lidar", "sick_s300", LIDAR_COMPONENT)
        assert ComponentSpec.from_yaml(path).variants == []

    def test_variants_parsed(self, tmp_path: Path):
        path = _write_component(tmp_path, "arm", "ur", {**ARM_COMPONENT, "variants": ["ur5e", "ur10e"]})
        assert ComponentSpec.from_yaml(path).variants == ["ur5e", "ur10e"]

    def test_rejects_non_list_variants(self, tmp_path: Path):
        path = _write_component(tmp_path, "arm", "ur", {**ARM_COMPONENT, "variants": "ur5e"})
        with pytest.raises(ValueError, match="variants"):
            ComponentSpec.from_yaml(path)


class TestCatalogGet:
    def test_missing_variant_lists_available(self, tmp_path: Path):
        _write_component(tmp_path, "lidar", "sick_s300", LIDAR_COMPONENT)
        _write_component(tmp_path, "lidar", "vlp16", LIDAR_COMPONENT)
        catalog = Catalog(root=tmp_path)
        with pytest.raises(RuntimeError) as excinfo:
            catalog.get("lidar", "nope")
        msg = str(excinfo.value)
        assert "lidar/nope" in msg
        assert "sick_s300" in msg and "vlp16" in msg

    def test_get_caches_component_instance(self, tmp_path: Path):
        _write_component(tmp_path, "lidar", "sick_s300", LIDAR_COMPONENT)
        catalog = Catalog(root=tmp_path)
        assert catalog.get("lidar", "sick_s300") is catalog.get("lidar", "sick_s300")


class TestCatalogFamilyFallback:
    """A variant with no dedicated `<variant>/` dir resolves against a sibling family
    component whose `variants:` names it; a concrete dir still wins when present."""

    def test_variant_without_dir_resolves_to_family(self, tmp_path: Path):
        _write_component(tmp_path, "arm", "ur", {**ARM_COMPONENT, "variants": ["ur5e", "ur10e"]})
        spec = Catalog(root=tmp_path).get("arm", "ur5e")
        assert "ur5e" in spec.variants

    def test_exact_dir_wins_over_family(self, tmp_path: Path):
        _write_component(tmp_path, "arm", "ur", {**ARM_COMPONENT, "variants": ["ur5e"]})
        concrete = dict(ARM_COMPONENT)
        concrete["caps"] = {"tip_link": "CONCRETE"}
        _write_component(tmp_path, "arm", "ur5e", concrete)
        assert Catalog(root=tmp_path).get("arm", "ur5e").caps["tip_link"] == "CONCRETE"

    def test_unknown_variant_lists_family_variants(self, tmp_path: Path):
        _write_component(tmp_path, "arm", "ur", {**ARM_COMPONENT, "variants": ["ur5e", "ur10e"]})
        with pytest.raises(RuntimeError) as excinfo:
            Catalog(root=tmp_path).get("arm", "nope")
        msg = str(excinfo.value)
        assert "arm/nope" in msg
        assert "ur5e" in msg and "ur10e" in msg


class TestComponentSpecFrames:
    def test_frames_default_empty(self, tmp_path: Path):
        path = _write_component(tmp_path, "lidar", "sick_s300", LIDAR_COMPONENT)
        assert ComponentSpec.from_yaml(path).frames == {}

    def test_frames_parsed(self, tmp_path: Path):
        path = _write_component(tmp_path, "lift", "ewellix_900mm", LIFT_COMPONENT)
        spec = ComponentSpec.from_yaml(path)
        assert spec.frames == {"top": "${prefix}${mount}_ewellix_lift_top_link"}


class TestResolveMountParent:
    """``resolve_mount_parent`` is the shared chained-parent resolver
    consumed by both ``render_wrapper_xacro`` (xacro ``parent`` attr) and
    ``caps.py``'s ``_instances`` (``${parent}`` in a caps template, e.g. arm
    ``base_link``)."""

    @pytest.fixture
    def catalog(self, tmp_path: Path) -> Catalog:
        _write_component(tmp_path, "lift", "ewellix_900mm", LIFT_COMPONENT)
        return Catalog(root=tmp_path)

    def test_unchained_mount_passes_through_literal_parent(self, catalog: Catalog):
        mount = _mount("lift0", ["lift"], parent="base_link")
        resolved = ResolvedAssembly(placements=[Placement(type="lift", variant="ewellix_900mm", mount=mount)])
        assert resolve_mount_parent(resolved, catalog, mount) == "base_link"

    def test_chained_mount_resolves_bare_frame(self, catalog: Catalog):
        lift_mount = _mount("lift0", ["lift"], parent="base_link")
        arm_mount = _mount("arm0", ["arm"], parent="@lift0:top")
        resolved = ResolvedAssembly(
            placements=[
                Placement(type="lift", variant="ewellix_900mm", mount=lift_mount),
                Placement(type="arm", variant="ur10e", mount=arm_mount),
            ]
        )
        # bare (no prefix baked in): the caller re-applies its own ${prefix}${parent}
        assert resolve_mount_parent(resolved, catalog, arm_mount) == "lift0_ewellix_lift_top_link"

    def test_chained_mount_to_unplaced_mount_raises(self, catalog: Catalog):
        arm_mount = _mount("arm0", ["arm"], parent="@lift0:top")
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur10e", mount=arm_mount)])
        with pytest.raises(RuntimeError, match="unpopulated mount 'lift0'"):
            resolve_mount_parent(resolved, catalog, arm_mount)

    def test_chained_mount_to_undeclared_frame_raises(self, catalog: Catalog):
        lift_mount = _mount("lift0", ["lift"], parent="base_link")
        arm_mount = _mount("arm0", ["arm"], parent="@lift0:bogus")
        resolved = ResolvedAssembly(
            placements=[
                Placement(type="lift", variant="ewellix_900mm", mount=lift_mount),
                Placement(type="arm", variant="ur10e", mount=arm_mount),
            ]
        )
        with pytest.raises(RuntimeError, match="does not export frame 'bogus'"):
            resolve_mount_parent(resolved, catalog, arm_mount)


class TestRenderEffectiveSensors:
    """Reproduces rbtheron's lidar+camera SensorSpec ground truth (model_params.yaml)
    from a two-mount default (front) + overridden (rear) assembly, to prove the
    override vocabulary (name, topic) is sufficient for the fleet's asymmetric case."""

    @pytest.fixture
    def catalog(self, tmp_path: Path) -> Catalog:
        _write_component(tmp_path, "lidar", "sick_s300", LIDAR_COMPONENT)
        _write_component(tmp_path, "camera", "d435_rgbd", CAMERA_COMPONENT)
        return Catalog(root=tmp_path)

    def _resolved(self) -> ResolvedAssembly:
        front_laser = _mount("front_laser", ["lidar"])
        rear_laser = _mount("rear_laser", ["lidar"])
        front_cam = _mount("front", ["camera"])
        rear_cam = _mount("rear", ["camera"])
        return ResolvedAssembly(
            placements=[
                Placement(type="lidar", variant="sick_s300", mount=front_laser),
                Placement(type="lidar", variant="sick_s300", mount=rear_laser, overrides={"name": "lidar_rear", "topic": "scan/rear"}),
                Placement(type="camera", variant="d435_rgbd", mount=front_cam),
                Placement(type="camera", variant="d435_rgbd", mount=rear_cam),
            ]
        )

    def test_exact_sensorspec_tuples_match_rbtheron(self, catalog: Catalog):
        sensors = render_effective_sensors(self._resolved(), catalog)
        got = [(s.name, s.type, s.topic, s.frame, s.sensor) for s in sensors]
        assert got == [
            ("lidar", "laserscan", "${namespace}/scan", "robot_front_laser_link", "front_laser"),
            ("lidar_points", "pointcloud", "${namespace}/scan/points", "robot_front_laser_link", "front_laser"),
            ("lidar_rear", "laserscan", "${namespace}/scan/rear", "robot_rear_laser_link", "rear_laser"),
            ("lidar_rear_points", "pointcloud", "${namespace}/scan/rear/points", "robot_rear_laser_link", "rear_laser"),
            ("front_camera_image", "image", "${namespace}/front_camera/image", "robot_front_rgbd_camera_color_frame", "front_camera_color"),
            ("front_camera_info", "camera_info", "${namespace}/front_camera/camera_info", "robot_front_rgbd_camera_color_frame", "front_camera_color"),
            ("rear_camera_image", "image", "${namespace}/rear_camera/image", "robot_rear_rgbd_camera_color_frame", "rear_camera_color"),
            ("rear_camera_info", "camera_info", "${namespace}/rear_camera/camera_info", "robot_rear_rgbd_camera_color_frame", "rear_camera_color"),
        ]

    def test_topic_default_fallback_without_override(self, catalog: Catalog):
        front_laser = _mount("front_laser", ["lidar"])
        resolved = ResolvedAssembly(placements=[Placement(type="lidar", variant="sick_s300", mount=front_laser)])
        sensors = render_effective_sensors(resolved, catalog)
        assert sensors[0].topic == "${namespace}/scan"
        assert sensors[1].topic == "${namespace}/scan/points"

    def test_override_wins_over_default(self, catalog: Catalog):
        rear_laser = _mount("rear_laser", ["lidar"])
        resolved = ResolvedAssembly(placements=[Placement(type="lidar", variant="sick_s300", mount=rear_laser, overrides={"name": "lidar_rear", "topic": "scan/rear"})])
        sensors = render_effective_sensors(resolved, catalog)
        assert sensors[0].name == "lidar_rear"
        assert sensors[0].topic == "${namespace}/scan/rear"

    def test_frame_override_substitutes_into_rendered_frame(self, catalog: Catalog):
        resolved = apply_frame_overrides(self._resolved(), {"front_laser": "override_stem"})
        sensors = render_effective_sensors(resolved, catalog)
        assert sensors[0].frame == "robot_override_stem_link"
        assert "front_laser" not in sensors[0].frame

    def test_deepcopy_isolation_across_placements(self, catalog: Catalog):
        """Renders front-then-rear from the SAME cached ComponentSpec; if render_effective_sensors
        failed to deepcopy the shared template dicts (YAMLReplacer.replace mutates in place),
        the rear placement's overrides would bleed backward into the component's stored
        templates and corrupt a subsequent front render."""
        before = [dict(e) for e in catalog.get("lidar", "sick_s300").sensor_gz]

        render_effective_sensors(self._resolved(), catalog)

        after = catalog.get("lidar", "sick_s300").sensor_gz
        assert after == before

        front_laser = _mount("front_laser", ["lidar"])
        resolved_again = ResolvedAssembly(placements=[Placement(type="lidar", variant="sick_s300", mount=front_laser)])
        sensors = render_effective_sensors(resolved_again, catalog)
        assert sensors[0].name == "lidar"
        assert sensors[0].topic == "${namespace}/scan"


class TestRenderWrapperXacroNoControlSynthesis:
    """render_wrapper_xacro synthesizes no ``ros2_control`` tag of its own (regardless of
    whether a placement declares joints): the chassis and every placement's component each
    render their own xacro exactly as written, no gating required. Post-render merging is
    arena_simulation_setup.utils.models.urdf._inject_ros2_control_joints's job, fed by
    :func:`render_control_joints` (tested below)."""

    def test_no_joints_no_ros2_control_tag(self, tmp_path: Path):
        _write_component(tmp_path / "components", "lidar", "sick_s300", LIDAR_COMPONENT)
        catalog = Catalog(root=tmp_path / "components")
        view = _write_robot_view(tmp_path)
        front_laser = _mount("front_laser", ["lidar"])
        resolved = ResolvedAssembly(placements=[Placement(type="lidar", variant="sick_s300", mount=front_laser)])

        wrapper = render_wrapper_xacro(view, resolved, catalog=catalog)

        assert "<ros2_control" not in wrapper
        assert "base_hw_joints" not in wrapper

    def test_arm_joints_do_not_synthesize_a_tag(self, tmp_path: Path):
        """A placement declaring ``ros2_control.joints`` makes render_wrapper_xacro emit
        nothing extra for it: no ``<ros2_control>``, no ``<joint>``, no chassis-arg gating.
        The arm's own macro invocation renders normally."""
        _write_component(tmp_path / "components", "arm", "ur5e", ARM_COMPONENT)
        catalog = Catalog(root=tmp_path / "components")
        view = _write_robot_view(tmp_path, "testbot")
        arm_mount = _mount("arm0", ["arm"])
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur5e", mount=arm_mount)])

        wrapper = render_wrapper_xacro(view, resolved, catalog=catalog)

        assert "<ros2_control" not in wrapper
        assert "<joint " not in wrapper
        assert "generate_ros2_control_tag" not in wrapper
        assert "<xacro:ur_robot " in wrapper

    def test_chained_mount_parent_renders_referenced_frame(self, tmp_path: Path):
        """arm0 chained onto lift0's `top` frame renders the wrapper's
        `<xacro:ur_robot parent=...>` attr from the lift's rendered frame, not the
        literal `"@lift0:top"` chain string."""
        arm_component = {
            "xacro": {
                "include": "ur_macro.xacro",
                "macro": "ur_robot",
                "attach": {"parent": "${prefix}${parent}"},
                "args": {},
            }
        }
        _write_component(tmp_path / "components", "lift", "ewellix_900mm", LIFT_COMPONENT)
        _write_component(tmp_path / "components", "arm", "ur5e", arm_component)
        catalog = Catalog(root=tmp_path / "components")
        view = _write_robot_view(tmp_path, "testbot")
        lift_mount = _mount("lift0", ["lift"], parent="base_link")
        arm_mount = _mount("arm0", ["arm"], parent="@lift0:top")
        resolved = ResolvedAssembly(
            placements=[
                Placement(type="lift", variant="ewellix_900mm", mount=lift_mount),
                Placement(type="arm", variant="ur5e", mount=arm_mount),
            ]
        )

        wrapper = render_wrapper_xacro(view, resolved, catalog=catalog)

        assert '<xacro:ur_robot parent="$(arg prefix)lift0_ewellix_lift_top_link">' in wrapper
        assert "@lift0:top" not in wrapper

    def test_frame_override_flows_into_wrapper_attach(self, tmp_path: Path):
        """A deployment frame override baked onto resolved_assembly reaches
        render_wrapper_xacro via _frame_stem: the component's ${mount}-templated attach
        renders with the override stem, not the addressing mount name."""
        _write_component(tmp_path / "components", "arm", "ur5e", ARM_COMPONENT)
        catalog = Catalog(root=tmp_path / "components")
        view = _write_robot_view(tmp_path, "testbot")
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur5e", mount=_mount("arm0", ["arm"]))])
        overridden = apply_frame_overrides(resolved, {"arm0": "leftarm"})

        wrapper = render_wrapper_xacro(view, overridden, catalog=catalog)

        assert 'parent="$(arg prefix)leftarm_parent"' in wrapper
        assert "arm0_parent" not in wrapper


class TestRenderControlJoints:
    """Arm-on-any-chassis merge: ``render_control_joints`` computes the
    control-joint patch straight from ``resolved``/``catalog`` (no xacro involved),
    for arena_simulation_setup's urdf loader to inject post-render."""

    def test_no_placement_declares_joints_is_empty(self, tmp_path: Path):
        _write_component(tmp_path, "lidar", "sick_s300", LIDAR_COMPONENT)
        catalog = Catalog(root=tmp_path)
        front_laser = _mount("front_laser", ["lidar"])
        resolved = ResolvedAssembly(placements=[Placement(type="lidar", variant="sick_s300", mount=front_laser)])

        assert render_control_joints(resolved, catalog) == []

    def test_arm_placement_renders_joints_with_resolved_prefix(self, tmp_path: Path):
        _write_component(tmp_path, "arm", "ur5e", ARM_COMPONENT)
        catalog = Catalog(root=tmp_path)
        arm_mount = _mount("arm0", ["arm"])
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur5e", mount=arm_mount)])

        joints = render_control_joints(resolved, catalog, prefix="robot_")

        assert joints == [
            {"name": "robot_arm0_shoulder_pan_joint", "command_interfaces": ["position"], "state_interfaces": ["position", "velocity"]},
            {"name": "robot_arm0_shoulder_lift_joint", "command_interfaces": ["position"], "state_interfaces": ["position", "velocity"]},
        ]

    def test_multiple_placements_concatenate_in_order(self, tmp_path: Path):
        _write_component(tmp_path, "arm", "ur5e", ARM_COMPONENT)
        catalog = Catalog(root=tmp_path)
        resolved = ResolvedAssembly(
            placements=[
                Placement(type="arm", variant="ur5e", mount=_mount("arm0", ["arm"])),
                Placement(type="arm", variant="ur5e", mount=_mount("arm1", ["arm"])),
            ]
        )

        joints = render_control_joints(resolved, catalog, prefix="robot_")

        assert [j["name"] for j in joints] == [
            "robot_arm0_shoulder_pan_joint",
            "robot_arm0_shoulder_lift_joint",
            "robot_arm1_shoulder_pan_joint",
            "robot_arm1_shoulder_lift_joint",
        ]

    def test_frame_override_replaces_mount_stem_in_joint_names(self, tmp_path: Path):
        """A deployment frame override baked onto resolved_assembly flows through
        render_control_joints via _frame_stem: injected joint names carry the override
        stem, not the addressing mount name, so ros2_control binds the real driver's
        joints on a sim2real deployment."""
        _write_component(tmp_path, "arm", "ur5e", ARM_COMPONENT)
        catalog = Catalog(root=tmp_path)
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur5e", mount=_mount("arm0", ["arm"]))])
        overridden = apply_frame_overrides(resolved, {"arm0": "leftarm"})

        joints = render_control_joints(overridden, catalog, prefix="robot_")

        assert [j["name"] for j in joints] == ["robot_leftarm_shoulder_pan_joint", "robot_leftarm_shoulder_lift_joint"]


RBVOGUI_BASE_CONTROL = {
    "controller_manager": {
        "ros__parameters": {
            "use_sim_time": True,
            "update_rate": 50,
            "joint_state_broadcaster": {"type": "joint_state_broadcaster/JointStateBroadcaster"},
            "robotnik_base_controller": {"type": "arena_swerve_controller/SwerveController"},
        }
    },
    "joint_state_broadcaster": {"ros__parameters": {"use_sim_time": True, "publish_rate": 50}},
    "robotnik_base_controller": {"ros__parameters": {"use_sim_time": True}},
}


class TestRenderEffectiveControl:
    """rbvogui[arm=ur5e] control.yaml synthesis must reproduce rbvogui_plus's
    hand-authored control.yaml diff exactly: one controller_manager entry + one
    top-level controller section per arm placement."""

    def test_merges_arm_controller_into_rbvogui_shaped_base(self, tmp_path: Path):
        _write_component(tmp_path, "arm", "ur5e", ARM_COMPONENT)
        catalog = Catalog(root=tmp_path)
        arm_mount = _mount("arm", ["arm"])
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur5e", mount=arm_mount)])

        merged, extra = render_effective_control(resolved, RBVOGUI_BASE_CONTROL, catalog)

        assert extra == ["arm_controller"]
        assert merged["controller_manager"]["ros__parameters"]["arm_controller"] == {"type": "joint_trajectory_controller/JointTrajectoryController"}
        assert merged["arm_controller"]["ros__parameters"]["joints"] == [
            "robot_arm_shoulder_pan_joint",
            "robot_arm_shoulder_lift_joint",
        ]
        assert merged["arm_controller"]["ros__parameters"]["state_publish_rate"] == 50.0
        # existing controller_manager entries survive the merge
        assert merged["controller_manager"]["ros__parameters"]["robotnik_base_controller"] == {"type": "arena_swerve_controller/SwerveController"}
        # base_control is not mutated
        assert "arm_controller" not in RBVOGUI_BASE_CONTROL["controller_manager"]["ros__parameters"]
        assert "arm_controller" not in RBVOGUI_BASE_CONTROL

    def test_no_control_declared_is_a_no_op(self, tmp_path: Path):
        _write_component(tmp_path, "lidar", "sick_s300", LIDAR_COMPONENT)
        catalog = Catalog(root=tmp_path)
        front_laser = _mount("front_laser", ["lidar"])
        resolved = ResolvedAssembly(placements=[Placement(type="lidar", variant="sick_s300", mount=front_laser)])

        merged, extra = render_effective_control(resolved, RBVOGUI_BASE_CONTROL, catalog)

        assert extra == []
        assert merged == RBVOGUI_BASE_CONTROL

    def test_frame_override_flows_into_controller_and_joints(self, tmp_path: Path):
        """A deployment frame override baked onto resolved_assembly reaches
        render_effective_control via _frame_stem: both the synthesized controller name
        and its joints list carry the override stem."""
        _write_component(tmp_path, "arm", "ur5e", ARM_COMPONENT)
        catalog = Catalog(root=tmp_path)
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur5e", mount=_mount("arm", ["arm"]))])
        overridden = apply_frame_overrides(resolved, {"arm": "leftarm"})

        merged, extra = render_effective_control(overridden, RBVOGUI_BASE_CONTROL, catalog)

        assert extra == ["leftarm_controller"]
        assert merged["leftarm_controller"]["ros__parameters"]["joints"] == [
            "robot_leftarm_shoulder_pan_joint",
            "robot_leftarm_shoulder_lift_joint",
        ]

    def test_zero_prefix_chassis_renders_unprefixed_joints(self, tmp_path: Path):
        _write_component(tmp_path, "arm", "ur5e", ARM_COMPONENT)
        catalog = Catalog(root=tmp_path)
        arm_mount = _mount("arm", ["arm"])
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur5e", mount=arm_mount)])

        merged, extra = render_effective_control(resolved, RBVOGUI_BASE_CONTROL, catalog, prefix="")

        assert merged["arm_controller"]["ros__parameters"]["joints"] == ["arm_shoulder_pan_joint", "arm_shoulder_lift_joint"]


def _contact_component(sensor: str | None) -> dict:
    entry: dict[str, object] = {
        "name": "${mount}_finger",
        "type": "contact",
        "topic": "${mount}_finger/contact",
        "frame": "${prefix}${mount}_tip_link",
        "define": {"collision": "${prefix}${mount}_tip_link_collision", "update_rate": 100},
    }
    if sensor is not None:
        entry["sensor"] = sensor
    return {
        "xacro": {"include": "gripper.xacro", "macro": "gripper", "attach": {"parent": "${prefix}${parent}"}},
        "sensor": {"gz": [entry]},
    }


class TestRenderWrapperXacroContactDefine:
    def _render(self, tmp_path: Path, sensor: str | None) -> str:
        _write_component(tmp_path / "components", "gripper", "test", _contact_component(sensor))
        catalog = Catalog(root=tmp_path / "components")
        view = _write_robot_view(tmp_path)
        resolved = ResolvedAssembly(placements=[Placement(type="gripper", variant="test", mount=_mount("grip", ["gripper"]))])
        return render_wrapper_xacro(view, resolved, catalog=catalog)

    def test_sensor_element_takes_the_spec_sensor_name(self, tmp_path: Path):
        wrapper = self._render(tmp_path, "${prefix}${mount}_contact")

        assert '<gazebo reference="$(arg prefix)grip_tip_link">' in wrapper
        assert '<sensor name="$(arg prefix)grip_contact" type="contact">' in wrapper
        assert "<collision>$(arg prefix)grip_tip_link_collision</collision>" in wrapper
        assert "<update_rate>100</update_rate>" in wrapper

    def test_define_without_sensor_name_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="'sensor' is required"):
            self._render(tmp_path, None)
