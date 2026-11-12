"""Build a MoveIt config dict for a robot, shared between move_group launch
and the RViz parameter-injection path."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_rclpy_mixins.yaml_replace import YAMLReplacer
from moveit_configs_utils import MoveItConfigsBuilder

import arena_robots.Robot
from arena_robots.caps import ArmSpec, _substitute_keys
from arena_robots.catalog import render_wrapper_xacro

if TYPE_CHECKING:
    from arena_robots.assembly import ResolvedAssembly


def _prefix_urdf_links(urdf_xml: str, tf_prefix: str) -> str:
    """Prepend ``tf_prefix`` to every link reference in ``urdf_xml``.

    Joint names stay untouched, so the controller / joint_states wiring
    keeps working; only link/frame names get namespaced to match the TF
    tree published by robot_state_publisher's own ``frame_prefix``.
    """
    if not tf_prefix:
        return urdf_xml
    root = ET.fromstring(urdf_xml)
    for link in root.iter("link"):
        name = link.get("name")
        if name:
            link.set("name", tf_prefix + name)
    for joint in root.iter("joint"):
        for endpoint in joint:
            if endpoint.tag in ("parent", "child"):
                link_attr = endpoint.get("link")
                if link_attr:
                    endpoint.set("link", tf_prefix + link_attr)
    for gz in root.iter("gazebo"):
        ref = gz.get("reference")
        if ref:
            gz.set("reference", tf_prefix + ref)
    return ET.tostring(root, encoding="unicode")


def _prefix_srdf_links(srdf_xml: str, tf_prefix: str) -> str:
    """Same prefixing rules as the URDF, applied to the SRDF link refs."""
    if not tf_prefix:
        return srdf_xml
    root = ET.fromstring(srdf_xml)
    for group in root.iter("group"):
        for child in group:
            if child.tag == "chain":
                for attr in ("base_link", "tip_link"):
                    v = child.get(attr)
                    if v:
                        child.set(attr, tf_prefix + v)
            elif child.tag == "link":
                v = child.get("name")
                if v:
                    child.set("name", tf_prefix + v)
    for dc in root.iter("disable_collisions"):
        for attr in ("link1", "link2"):
            v = dc.get(attr)
            if v:
                dc.set(attr, tf_prefix + v)
    for vj in root.iter("virtual_joint"):
        v = vj.get("child_link")
        if v:
            vj.set("child_link", tf_prefix + v)
    for ee in root.iter("end_effector"):
        v = ee.get("parent_link")
        if v:
            ee.set("parent_link", tf_prefix + v)
    return ET.tostring(root, encoding="unicode")


def _select_arm[T](robot_name: str, arms: dict[str, T], instance: str | None) -> tuple[str, T]:
    """Pick one ``(key, entry)`` pair out of ``arms`` (``RobotCaps.arm``-shaped).
    ``None`` selects the sole entry, or raises ``ValueError`` (listing available
    instances) when there are several; an unknown ``instance`` raises ``KeyError``.
    The key is the mount name for allocation-derived caps (author key for robots
    without assembly.yaml), needed to render the SRDF/joint_limits per placement."""
    if instance is None:
        if len(arms) != 1:
            raise ValueError(f"{robot_name}: multiple arm instances {sorted(arms)}; 'instance' is required")
        ((key, arm),) = arms.items()
        return key, arm
    if instance not in arms:
        raise KeyError(f"{robot_name}: arm instance {instance!r} not found; available: {sorted(arms)}")
    return instance, arms[instance]


def _compose_srdf(
    robot: arena_robots.Robot.RobotView,
    fragment_path: Path,
    context: dict[str, str],
    extras: list[tuple[Path, dict[str, str]]] | None = None,
) -> Path:
    """Xacro-render the placed arm component's SRDF fragment with ``context``
    (prefix/mount/parent), then merge it with the chassis's residual SRDF fragment
    (``robots/<robot>/srdf/<robot>.srdf.xacro``, if declared) and any ``extras``
    (per-placement ``(fragment, context)`` pairs, e.g. a gripper chained onto the arm)
    under one ``<robot>`` element. Returns the path to a temp file holding the
    composed document."""
    fragment_xml = xacro.process_file(str(fragment_path), mappings={k: str(v) for k, v in context.items()}).toxml()  # pyright: ignore[reportAttributeAccessIssue]
    merged = ET.Element("robot", {"name": robot.name})
    merged.extend(list(ET.fromstring(fragment_xml)))

    for extra_path, extra_context in extras or []:
        extra_xml = xacro.process_file(str(extra_path), mappings={k: str(v) for k, v in extra_context.items()}).toxml()  # pyright: ignore[reportAttributeAccessIssue]
        merged.extend(list(ET.fromstring(extra_xml)))

    base_srdf = robot.path / "srdf" / f"{robot.name}.srdf.xacro"
    if base_srdf.is_file():
        base_xml = xacro.process_file(str(base_srdf)).toxml()  # pyright: ignore[reportAttributeAccessIssue]
        merged.extend(list(ET.fromstring(base_xml)))

    with NamedTemporaryFile(mode="w", suffix=".srdf", delete=False) as f:
        f.write(ET.tostring(merged, encoding="unicode"))
        return Path(f.name)


def _gripper_srdf_extras(resolved: ResolvedAssembly, arm_mount: str, prefix: str) -> list[tuple[Path, dict[str, str]]]:
    """``(fragment, context)`` per gripper placement chained onto ``arm_mount`` whose
    component caps declare a ``moveit.srdf`` fragment. The context adds ``arm`` (the
    carrying arm's frame stem) beside the usual prefix/mount, for the fragment's
    end_effector parent link/group and the wrist adjacency disable."""
    from arena_robots.catalog import Catalog, _frame_stem

    arm_placement = next((p for p in resolved.placements if p.type == "arm" and p.mount.name == arm_mount), None)
    if arm_placement is None:
        return []
    catalog = Catalog()
    extras: list[tuple[Path, dict[str, str]]] = []
    for placement in resolved.placements:
        if placement.type != "gripper":
            continue
        chained = placement.mount.chained_parent
        if chained is None or chained[0] != arm_mount:
            continue
        mv = (catalog.get(placement.type, placement.variant).caps or {}).get("moveit")
        if not isinstance(mv, dict):
            continue
        srdf_ref = mv.get("srdf") or {}
        if "path" not in srdf_ref:
            continue
        fragment = Path(get_package_share_directory(srdf_ref.get("package", "arena_robots"))) / srdf_ref["path"]
        extras.append((fragment, {"prefix": prefix, "mount": _frame_stem(placement.mount), "arm": _frame_stem(arm_placement.mount)}))
    return extras


def _render_joint_limits(jl_path: Path, context: dict[str, str]) -> Path:
    """Re-key a component ``joint_limits.yaml``'s ``${prefix}${mount}_*`` joint
    names for one placement. Files with no ``${`` token pass through untouched."""
    text = jl_path.read_text()
    if "${" not in text:
        return jl_path
    data = yaml.safe_load(text)
    rendered = YAMLReplacer(context).replace(data)
    rendered = _substitute_keys(rendered, context)
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(rendered, f)
        return Path(f.name)


_SEGMENT_FRACTION = 0.002
_PILZ = "pilz_industrial_motion_planner"
_CARTESIAN_LIMITS = {"max_trans_vel": 0.5, "max_trans_acc": 1.0, "max_trans_dec": -1.0, "max_rot_vel": 1.57}
_KDL_SOLVER = {
    "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
    "kinematics_solver_search_resolution": 0.005,
    "kinematics_solver_timeout": 0.05,
}


def _write_yaml(data: dict) -> Path:
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(data, f)
        return Path(f.name)


def _controller_params(arm: ArmSpec) -> dict[str, object]:
    """MoveIt simple-controller-manager mapping naming the arm's FollowJointTrajectory controller."""
    return {
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": [arm.controller],
            arm.controller: {
                "action_ns": "follow_joint_trajectory",
                "type": "FollowJointTrajectory",
                "default": True,
                "joints": list(arm.chain),
            },
        },
        "trajectory_execution": {
            "allowed_execution_duration_scaling": 1.2,
            "allowed_goal_duration_margin": 0.5,
            "allowed_start_tolerance": 0.1,
        },
    }


def build_moveit_params(robot_name: str, tf_prefix: str = "", parts: dict[str, list] | None = None, instance: str | None = None) -> dict[str, object] | None:
    """Return a flat parameter dict (URDF, SRDF, kinematics, joint limits,
    arm controller mapping) for ``robot_name`` if it advertises ``arm`` with a
    ``caps.moveit`` block carrying a ``package`` or an ``srdf``; ``None`` otherwise.

    When ``tf_prefix`` is non-empty (e.g. ``"env_0/ridgeback_plus/"``) every
    link reference in the URDF/SRDF gets the prefix prepended, so MoveIt's
    planning-scene monitor looks up the same frame ids that
    robot_state_publisher publishes.

    ``parts`` is the robot's morphology request (allocation-derived
    caps); ``None``/``{}`` resolves defaults only. Both current callers
    (moveit.launch.py, rviz_config.py) run across a launch/process boundary
    from the env node that holds the real request and can't
    supply it yet, so they get defaults.

    ``instance`` selects which arm cap instance to build for (dict key from
    ``RobotCaps.arm``). ``None`` selects the sole instance when there is
    exactly one, and raises ``ValueError`` (listing available instances)
    when there are several; an unknown ``instance`` raises ``KeyError``.

    For allocation-derived robots (an ``assembly.yaml`` resolves ``parts``) the chassis
    xacro alone has no arm and no sensors, so ``robot_description`` is rendered through
    the full wrapper (:func:`catalog.render_wrapper_xacro`) instead of the bare chassis
    file, and ``robot_description_semantic``/joint_limits are composed/re-keyed per
    placement (mount/prefix/parent, from the placement's rendered ``caps.moveit.args``).
    Robots with no ``assembly.yaml`` (e.g. rbvogui_plus) take the static-path route
    unchanged: static chassis/SRDF/joint_limits paths.
    """
    robot = arena_robots.Robot.RobotIdentifier(robot_name).resolve_sync()
    caps = robot.effective_caps(parts or {})
    if "arm" not in caps.available:
        return None

    arms = caps.arm
    if arms is None:
        raise ValueError(f"{robot_name}: arm cap required but absent")
    arm_key, arm = _select_arm(robot_name, arms, instance)

    mv = arm.raw.get("moveit") or {}
    pkg = mv.get("package")
    if not pkg and not mv.get("srdf"):
        return None
    config_pkg = pkg or "arena_robots"
    builder_name = mv.get("robot_name") or (pkg.removesuffix("_moveit_config") if pkg else robot_name)

    args_dict = mv.get("args") or {}
    mappings = {k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in args_dict.items()}
    if "ur_type" in mappings:
        mappings.setdefault("name", mappings["ur_type"])

    resolved = robot._resolved(parts or {})

    if resolved is not None:
        wrapper_xml = render_wrapper_xacro(robot, resolved)
        with NamedTemporaryFile(mode="w", suffix=".urdf.xacro", delete=False) as wf:
            wf.write(wrapper_xml)
        urdf_abs = Path(wf.name)
    else:
        urdf_abs = Path(get_package_share_directory("arena_robots")) / "robots" / robot_name / "urdf" / f"{robot_name}.urdf.xacro"

    srdf_ref = mv.get("srdf") or {}
    srdf_pkg = srdf_ref.get("package", config_pkg)
    srdf_rel = srdf_ref.get("path", "srdf/ur.srdf.xacro")
    srdf_abs = Path(get_package_share_directory(srdf_pkg)) / srdf_rel

    jl_ref = mv.get("joint_limits") or {}
    jl_pkg = jl_ref.get("package", config_pkg)
    jl_rel = jl_ref.get("path", "config/joint_limits.yaml")
    jl_abs = Path(get_package_share_directory(jl_pkg)) / jl_rel

    if resolved is not None:
        placement_context = {k: mappings[k] for k in ("prefix", "mount", "parent") if k in mappings}
        extras = _gripper_srdf_extras(resolved, arm_key, placement_context.get("prefix", ""))
        srdf_abs = _compose_srdf(robot, srdf_abs, placement_context, extras)
        jl_abs = _render_joint_limits(jl_abs, placement_context)

    group = arm.planning_group or arm.name
    builder = MoveItConfigsBuilder(robot_name=builder_name, package_name=config_pkg).robot_description(file_path=str(urdf_abs), mappings=mappings).robot_description_semantic(str(srdf_abs), mappings=mappings).joint_limits(str(jl_abs)).trajectory_execution(file_path=str(_write_yaml(_controller_params(arm))))  # pyright: ignore[reportArgumentType]
    if not pkg:
        limits = {**_CARTESIAN_LIMITS, **(mv.get("cartesian_limits") or {})}
        builder = builder.robot_description_kinematics(str(_write_yaml({}))).planning_pipelines(pipelines=["ompl", _PILZ], default_planning_pipeline="ompl").pilz_cartesian_limits(str(_write_yaml({"cartesian_limits": limits})))
    params = builder.to_moveit_configs().to_dict()  # pyright: ignore[reportArgumentType]

    if not pkg:
        params.setdefault("ompl", {}).setdefault(group, {})["longest_valid_segment_fraction"] = _SEGMENT_FRACTION

    planning = params.get("robot_description_planning") or {}
    for limits in (planning.get("joint_limits") or {}).values():
        for key, value in limits.items():
            if isinstance(value, int) and not isinstance(value, bool):
                limits[key] = float(value)

    kinematics = params.get("robot_description_kinematics") or {}
    kinematics.setdefault(group, dict(_KDL_SOLVER))
    params["robot_description_kinematics"] = kinematics

    if tf_prefix:
        rd = params.get("robot_description")
        if isinstance(rd, str):
            params["robot_description"] = _prefix_urdf_links(rd, tf_prefix)
        rds = params.get("robot_description_semantic")
        if isinstance(rds, str):
            params["robot_description_semantic"] = _prefix_srdf_links(rds, tf_prefix)
    return params
