"""Tests for components/gripper/robotiq_2f_85 (gripper-on-arm chaining): the
gripper mounts onto an arm's exported `tip` frame via a chained mount (jackal's
`top_tool`, parent "@top:tip"), its actuated knuckle joint merges into the chassis
GazeboSimSystem tag alongside the arm's (mimic joints ride along as bare param
entries), and its GripperActionController joins the synthesized control.yaml."""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from arena_robots.assembly import Mount, Placement, ResolvedAssembly, build_request, resolve
from arena_robots.catalog import Catalog, ComponentSpec, render_control_joints, render_effective_control, render_effective_sensors, render_wrapper_xacro
from arena_robots.Robot import RobotIdentifier
from arena_simulation_setup.utils.models.urdf import _inject_ros2_control_joints

COMPONENTS_ROOT = Path(__file__).resolve().parent.parent / "components"
_XACRO = shutil.which("xacro")
_DEPS = Path(__file__).resolve().parent.parent.parent / "deps"
_UR_MACRO = _DEPS / "ur_description" / "urdf" / "ur_macro.xacro"
_ROBOTIQ_MACRO = _DEPS / "robotiq_description" / "robotiq_description" / "urdf" / "robotiq_2f_85_macro.urdf.xacro"
KNUCKLE = "robotiq_85_left_knuckle_joint"
MIMIC_JOINTS = [
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]


class TestGripperComponentSpec:
    @pytest.fixture(scope="class")
    def spec(self) -> ComponentSpec:
        return ComponentSpec.from_yaml(COMPONENTS_ROOT / "gripper" / "robotiq_2f_85" / "component.yaml")

    def test_attach_keeps_native_ros2_control_off(self, spec: ComponentSpec) -> None:
        assert spec.xacro_macro == "robotiq_gripper"
        assert spec.attach["include_ros2_control"] == "false"
        assert spec.attach["prefix"] == "${prefix}${mount}_"
        assert spec.attach["parent"] == "${prefix}${parent}"

    def test_one_actuated_joint_five_state_only_mimics(self, spec: ComponentSpec) -> None:
        """Mimic joints ride along state-only with the explicit jazzy ros2_control mimic
        attribute (kinematics from the URDF mimic tags; humble-era param-style entries
        leave the joints NaN, bare listing leaves them limp)."""
        actuated = [j for j in spec.ros2_control_joints if j.get("command_interfaces")]
        mimics = [j for j in spec.ros2_control_joints if not j.get("command_interfaces")]
        assert [j["name"] for j in actuated] == [f"${{prefix}}${{mount}}_{KNUCKLE}"]
        assert not actuated[0].get("mimic")
        assert [j["name"] for j in mimics] == [f"${{prefix}}${{mount}}_{m}" for m in MIMIC_JOINTS]
        for j in mimics:
            assert j["state_interfaces"] == ["position", "velocity"]
            assert j["mimic"] is True

    def test_control_block_is_gripper_action_controller(self, spec: ComponentSpec) -> None:
        assert spec.control["type"] == "position_controllers/GripperActionController"
        assert spec.control["ros__parameters"]["joint"] == f"${{prefix}}${{mount}}_{KNUCKLE}"

    def test_caps_block_shape(self, spec: ComponentSpec) -> None:
        assert spec.caps["joint"] == f"${{prefix}}${{mount}}_{KNUCKLE}"
        assert spec.caps["controller"] == "${mount}_controller"
        assert spec.caps["moveit"]["srdf"]["path"] == "components/gripper/robotiq_2f_85/srdf/robotiq_2f_85.srdf.xacro"
        assert spec.caps["positions"] == {"open": 0.0, "close": 0.8}
        assert spec.caps["max_effort"] == 50.0


def _jackal_gripper_resolved() -> tuple[object, object]:
    view = RobotIdentifier("jackal").resolve_sync()
    assert view.assembly is not None
    request, cleared_sockets, cleared_types = build_request(view.assembly, {"top": ["arm/ur5e"], "top_tool": ["gripper/robotiq_2f_85"]})
    resolved = resolve(view.assembly, request, cleared_sockets=cleared_sockets, cleared_types=cleared_types)
    return view, resolved


class TestGripperChainsOntoArmTip:
    def test_wrapper_attaches_gripper_to_arm_tool0(self) -> None:
        view, resolved = _jackal_gripper_resolved()
        wrapper = render_wrapper_xacro(view, resolved, catalog=Catalog(root=COMPONENTS_ROOT))
        assert '<xacro:robotiq_gripper' in wrapper
        assert 'parent="$(arg prefix)top_tool0"' in wrapper
        assert "@top:tip" not in wrapper

    def test_gripper_without_arm_raises_at_resolve(self) -> None:
        from arena_robots.assembly import AssemblyError

        view = RobotIdentifier("jackal").resolve_sync()
        request, cleared_sockets, cleared_types = build_request(view.assembly, {"top_tool": ["gripper/robotiq_2f_85"]})
        with pytest.raises(AssemblyError, match="'top_tool' requires 'top'"):
            resolve(view.assembly, request, cleared_sockets=cleared_sockets, cleared_types=cleared_types)

    def test_control_joint_patch_lists_actuated_and_mimics(self) -> None:
        _, resolved = _jackal_gripper_resolved()
        joints = render_control_joints(resolved, Catalog(root=COMPONENTS_ROOT), prefix="")
        by_name = {str(j["name"]): j for j in joints}
        assert by_name[f"top_tool_{KNUCKLE}"]["command_interfaces"] == ["position"]
        for m in MIMIC_JOINTS:
            entry = by_name[f"top_tool_{m}"]
            assert not entry.get("command_interfaces")
            assert entry["state_interfaces"] == ["position", "velocity"]
            assert entry["mimic"] is True

    def test_effective_control_gains_both_controllers(self) -> None:
        _, resolved = _jackal_gripper_resolved()
        merged, extra = render_effective_control(resolved, {}, Catalog(root=COMPONENTS_ROOT), prefix="")
        assert {"top_controller", "top_tool_controller"} <= set(extra)
        assert merged["top_tool_controller"]["ros__parameters"]["joint"] == f"top_tool_{KNUCKLE}"
        assert merged["controller_manager"]["ros__parameters"]["top_tool_controller"] == {"type": "position_controllers/GripperActionController"}

    def test_gripper_caps_render_as_gripper_spec(self) -> None:
        from arena_robots.caps import RobotCaps

        _, resolved = _jackal_gripper_resolved()
        rc = RobotCaps(caps_dir=Path("/dev/null"), resolved=resolved, catalog=Catalog(root=COMPONENTS_ROOT), prefix="")
        grippers = rc.gripper
        assert grippers is not None and set(grippers) == {"top_tool"}
        assert grippers["top_tool"].joint == f"top_tool_{KNUCKLE}"
        assert grippers["top_tool"].controller == "top_tool_controller"
        assert grippers["top_tool"].positions == {"open": 0.0, "close": 0.8}
        assert grippers["top_tool"].max_effort == 50.0


@pytest.mark.parametrize(
    ("overrides", "topics"),
    [
        ({}, ["${namespace}/tool_left_finger/contact", "${namespace}/tool_right_finger/contact"]),
        ({"topic": "grip"}, ["${namespace}/grip_left_finger/contact", "${namespace}/grip_right_finger/contact"]),
    ],
)
def test_finger_contact_topics_stay_distinct(overrides: dict[str, str], topics: list[str]) -> None:
    mount = Mount(name="tool", parent="tool0", xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), accepts=frozenset({"gripper"}))
    resolved = ResolvedAssembly(placements=[Placement(type="gripper", variant="robotiq_2f_85", mount=mount, overrides=overrides)])
    assert [s.topic for s in render_effective_sensors(resolved, Catalog(root=COMPONENTS_ROOT), prefix="")] == topics


class TestRbkairosGripperMount:
    @pytest.fixture(scope="class")
    def placed(self) -> tuple[object, object]:
        view = RobotIdentifier("rbkairos").resolve_sync()
        assert view.assembly is not None
        request, cleared_sockets, cleared_types = build_request(view.assembly, {"arm_mount": ["arm/ur5e"], "gripper_mount": ["gripper/robotiq_2f_85"]})
        return view, resolve(view.assembly, request, cleared_sockets=cleared_sockets, cleared_types=cleared_types)

    def test_gripper_attaches_to_the_arm_tool0(self, placed: tuple[object, object]) -> None:
        view, resolved = placed
        wrapper = render_wrapper_xacro(view, resolved, catalog=Catalog(root=COMPONENTS_ROOT))
        assert '<xacro:robotiq_gripper' in wrapper
        assert 'parent="$(arg prefix)arm_tool0"' in wrapper

    def test_gripper_caps_carry_the_controller_and_positions(self, placed: tuple[object, object]) -> None:
        from arena_robots.caps import RobotCaps

        _, resolved = placed
        rc = RobotCaps(caps_dir=Path("/dev/null"), resolved=resolved, catalog=Catalog(root=COMPONENTS_ROOT), prefix="robot_")
        assert rc.gripper is not None and set(rc.gripper) == {"gripper_mount"}
        gripper = rc.gripper["gripper_mount"]
        assert gripper.joint == f"robot_gripper_{KNUCKLE}"
        assert gripper.controller == "gripper_controller"
        assert (gripper.positions, gripper.max_effort) == ({"open": 0.0, "close": 0.8}, 50.0)


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH; run under the Arena container (bash arena -c pytest)")
@pytest.mark.skipif(not _UR_MACRO.is_file(), reason="deps/ur_description not initialized; run `arena feature robots add arm/ur`")
@pytest.mark.skipif(not _ROBOTIQ_MACRO.is_file(), reason="deps/robotiq_description not initialized; run `arena feature robots add gripper/robotiq_2f_85`")
class TestJackalGripperEndToEnd:
    """Full compose->xacro->inject chain for ``jackal[top=arm/ur5e,top_tool=gripper/robotiq_2f_85]``,
    mirroring test_jackal_arm_e2e with the gripper chained on."""

    @pytest.fixture(scope="class")
    def merged(self, tmp_path_factory: pytest.TempPathFactory) -> ET.Element:
        view, resolved = _jackal_gripper_resolved()
        catalog = Catalog(root=COMPONENTS_ROOT)
        wrapper_path = tmp_path_factory.mktemp("jackal_gripper") / "jackal_gripper.urdf.xacro"
        wrapper_path.write_text(render_wrapper_xacro(view, resolved, catalog=catalog))
        rendered = ET.fromstring(subprocess.run([_XACRO, str(wrapper_path)], capture_output=True, text=True, check=True).stdout)
        _inject_ros2_control_joints(rendered, render_control_joints(resolved, catalog, prefix=view.assembly.prefix))
        return rendered

    def test_gripper_links_present(self, merged: ET.Element) -> None:
        links = {el.get("name", "") for el in merged.iter("link")}
        assert "top_tool_robotiq_85_base_link" in links
        assert "top_tool_robotiq_85_left_finger_tip_link" in links

    def test_single_merged_control_tag_with_arm_and_gripper(self, merged: ET.Element) -> None:
        tags = merged.findall("ros2_control")
        assert len(tags) == 1
        joint_names = {j.get("name", "") for j in tags[0].iter("joint")}
        assert f"top_tool_{KNUCKLE}" in joint_names
        assert "top_shoulder_pan_joint" in joint_names

    def test_mimic_joints_flagged_with_urdf_mimic_tags(self, merged: ET.Element) -> None:
        """The injected control entries stay state-only and carry the jazzy mimic
        attribute; the kinematics come from the macro's own URDF mimic tags, which must
        reference the actuated knuckle."""
        control = merged.find("ros2_control")
        mimic_entry = next(j for j in control.iter("joint") if j.get("name") == "top_tool_robotiq_85_right_knuckle_joint")
        assert mimic_entry.get("mimic") == "true"
        assert mimic_entry.find("command_interface") is None
        assert {s.get("name") for s in mimic_entry.findall("state_interface")} == {"position", "velocity"}
        urdf_mimics = {j.get("name"): j.find("mimic") for j in merged.iter("joint") if j.find("mimic") is not None}
        for m in MIMIC_JOINTS:
            tag = urdf_mimics[f"top_tool_{m}"]
            assert tag.get("joint") == f"top_tool_{KNUCKLE}"
