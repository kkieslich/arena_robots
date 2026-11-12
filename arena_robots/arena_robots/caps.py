"""Capability-file loader for robots/<name>/caps/.

Each YAML under caps/ declares one capability the robot advertises. File
presence is the advertisement: `caps/arm.yaml` means the robot has `arm`.

Shape convention:
    caps/mobile.yaml:  flat primitives + adapter sub-blocks (nav2, rl, ...)
    caps/arm.yaml:     dict of named instances (single-arm is one entry named "arm")
    caps/lift.yaml:    dict of named instances
    caps/gripper.yaml: dict of named instances, with per-entry `arm:` back-ref

Adapter-specific sub-blocks (moveit, drl_grasp, nav2, rl) are raw dicts inside
each entry / at top level, read only by their matching runtime-selected adapter.
"""

from __future__ import annotations

import ast
import copy
import math
import subprocess
import typing
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import attrs
import yaml
from arena_rclpy_mixins.yaml_replace import YAMLReplacer
from arena_simulation_setup.tree.Gesture import GestureSpec

from arena_robots.assembly import ResolvedAssembly
from arena_robots.catalog import Catalog, _frame_stem, resolve_mount_parent

_CAPS_PREFIX = 'robot_'
"""Frame-templating prefix for placement-rendered caps; matches
``catalog.render_effective_sensors``'s default so cap frame names (e.g. arm
base_link/tip_link) line up with the sensor frames on the same robot."""

_POLYGON_TYPES: frozenset[str] = frozenset({'polygon', 'circle'})
_ACTION_TYPES: frozenset[str] = frozenset({'stop', 'slowdown', 'approach', 'limit'})


@attrs.define(slots=False)
class PolygonSpec:
    """One entry from the top-level `polygons_dict:` in caps/mobile.yaml."""

    name: str
    type: Literal['polygon', 'circle']
    points: list[list[float]] | None
    radius: float | None
    action_type: str | None
    polygon_pub_topic: str | None
    min_points: int | None
    visualize: bool | None
    enabled: bool | None
    slowdown_ratio: float | None

    @classmethod
    def from_dict(cls, name: str, d: dict[str, typing.Any]) -> PolygonSpec:
        ptype = d.get('type')
        if ptype not in _POLYGON_TYPES:
            raise ValueError(f"polygon '{name}': type must be 'polygon' or 'circle'; got {ptype!r}")

        points: list[list[float]] | None = None
        radius: float | None = None

        if ptype == 'polygon':
            raw_pts = d.get('points')
            if raw_pts is None or len(raw_pts) < 3:
                raise ValueError(f"polygon '{name}': type=polygon requires points with at least 3 entries")
            points = [[float(c) for c in pt] for pt in raw_pts]
        else:
            raw_r = d.get('radius')
            if raw_r is None or float(raw_r) <= 0:
                raise ValueError(f"polygon '{name}': type=circle requires radius > 0")
            radius = float(raw_r)

        action_type = d.get('action_type')
        if action_type is not None and action_type not in _ACTION_TYPES:
            raise ValueError(f"polygon '{name}': action_type must be one of {sorted(_ACTION_TYPES)}; got {action_type!r}")

        enabled_raw = d.get('enabled')
        enabled = bool(enabled_raw) if enabled_raw is not None else None

        visualize_raw = d.get('visualize')
        visualize = bool(visualize_raw) if visualize_raw is not None else None

        min_points_raw = d.get('min_points')
        min_points = int(min_points_raw) if min_points_raw is not None else None

        slowdown_ratio_raw = d.get('slowdown_ratio')
        slowdown_ratio = float(slowdown_ratio_raw) if slowdown_ratio_raw is not None else None

        return cls(
            name=name,
            type=ptype,
            points=points,
            radius=radius,
            action_type=action_type,
            polygon_pub_topic=d.get('polygon_pub_topic'),
            min_points=min_points,
            visualize=visualize,
            enabled=enabled,
            slowdown_ratio=slowdown_ratio,
        )


def stringify_float_matrix(m: list[list[float]]) -> str:
    """Return the nav2-expected string form ``[[x0,y0],[x1,y1],...]``."""
    return '[' + ','.join('[' + ','.join(str(c) for c in pt) + ']' for pt in m) + ']'


@attrs.define(slots=False)
class Range:
    """[min, max] pair. Accepts either ``[lo, hi]`` list or ``{min, max}`` dict form."""

    min: float
    max: float

    @classmethod
    def from_value(cls, v: object, label: str = 'range') -> Range:
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"{label}: list form requires exactly 2 elements; got {list(v)!r}")
            return cls(min=float(v[0]), max=float(v[1]))
        if isinstance(v, dict):
            missing = {'min', 'max'} - set(v)
            if missing:
                raise ValueError(f"{label}: dict form requires 'min' and 'max' keys; missing {sorted(missing)}")
            return cls(min=float(v['min']), max=float(v['max']))
        raise ValueError(f"{label}: expected list [min, max] or dict {{min, max}}; got {type(v).__name__}")


@attrs.define(slots=False)
class VelocityLimits:
    """Operational velocity envelope (planner-facing, not the hardware envelope). Holonomic robots populate ``lateral``."""

    linear: Range
    angular: Range
    lateral: Range | None

    @classmethod
    def from_dict(cls, d: dict[str, typing.Any]) -> VelocityLimits:
        if 'linear' not in d or 'angular' not in d:
            raise ValueError(f"velocity_limits: requires 'linear' and 'angular' keys; got {sorted(d)}")
        lateral = Range.from_value(d['lateral'], 'velocity_limits.lateral') if 'lateral' in d else None
        return cls(
            linear=Range.from_value(d['linear'], 'velocity_limits.linear'),
            angular=Range.from_value(d['angular'], 'velocity_limits.angular'),
            lateral=lateral,
        )


@attrs.define(slots=False)
class AccelerationLimits:
    """Operational acceleration envelope, per-axis (symmetric magnitude). Not the hardware envelope."""

    linear: float
    angular: float
    lateral: float | None

    @classmethod
    def from_dict(cls, d: dict[str, typing.Any]) -> AccelerationLimits:
        if 'linear' not in d or 'angular' not in d:
            raise ValueError(f"acceleration_limits: requires 'linear' and 'angular' keys; got {sorted(d)}")
        lateral = float(d['lateral']) if 'lateral' in d else None
        return cls(
            linear=float(d['linear']),
            angular=float(d['angular']),
            lateral=lateral,
        )


@attrs.define(slots=False)
class DiscreteAction:
    """One entry from ``actions.discrete``. ``lateral`` is 0 unless the robot is holonomic."""

    name: str
    linear: float
    angular: float
    lateral: float

    @classmethod
    def from_dict(cls, d: dict[str, typing.Any]) -> DiscreteAction:
        if 'name' not in d:
            raise ValueError(f"actions.discrete entry missing 'name': {d!r}")
        return cls(
            name=str(d['name']),
            linear=float(d.get('linear', 0.0)),
            angular=float(d.get('angular', 0.0)),
            lateral=float(d.get('lateral', 0.0)),
        )


@attrs.define(slots=False)
class ContinuousActionLimits:
    """Continuous action envelope. Mirrors :class:`VelocityLimits` but may be
    authored independently when the RL action space is narrower than the
    physical velocity envelope."""

    linear: Range
    angular: Range
    lateral: Range | None

    @classmethod
    def from_dict(cls, d: dict[str, typing.Any]) -> ContinuousActionLimits:
        if 'linear' not in d or 'angular' not in d:
            raise ValueError(f"actions.continuous: requires 'linear' and 'angular' keys; got {sorted(d)}")
        lateral = Range.from_value(d['lateral'], 'actions.continuous.lateral') if 'lateral' in d else None
        return cls(
            linear=Range.from_value(d['linear'], 'actions.continuous.linear'),
            angular=Range.from_value(d['angular'], 'actions.continuous.angular'),
            lateral=lateral,
        )


@attrs.define(slots=False)
class ActionsSpec:
    """``actions.continuous`` envelope + ``actions.discrete`` enumeration."""

    continuous: ContinuousActionLimits
    discrete: list[DiscreteAction]

    @classmethod
    def from_dict(cls, d: dict[str, typing.Any]) -> ActionsSpec:
        if 'continuous' not in d:
            raise ValueError(f"actions: requires 'continuous' block; got {sorted(d)}")
        raw_discrete = d.get('discrete', [])
        if not isinstance(raw_discrete, list):
            raise ValueError(f"actions.discrete: must be a list; got {type(raw_discrete).__name__}")
        return cls(
            continuous=ContinuousActionLimits.from_dict(d['continuous']),
            discrete=[DiscreteAction.from_dict(entry) for entry in raw_discrete],
        )


@attrs.define(slots=False)
class LaserAngle:
    """Laser scanner angular extents."""

    min: float
    max: float
    increment: float


@attrs.define(slots=False)
class LaserSpec:
    """Laser scanner geometry, consumed by both nav2 AMCL and RL observation stacks."""

    angle: LaserAngle
    num_beams: int
    range: float
    update_rate: int

    @classmethod
    def from_dict(cls, d: dict[str, typing.Any]) -> LaserSpec:
        angle_raw = d.get('angle')
        if not isinstance(angle_raw, dict):
            raise ValueError(f"laser.angle: must be a mapping; got {type(angle_raw).__name__}")
        missing = {'min', 'max', 'increment'} - set(angle_raw)
        if missing:
            raise ValueError(f"laser.angle: missing required keys {sorted(missing)}")
        return cls(
            angle=LaserAngle(
                min=float(angle_raw['min']),
                max=float(angle_raw['max']),
                increment=float(angle_raw['increment']),
            ),
            num_beams=int(d['num_beams']),
            range=float(d['range']),
            update_rate=int(d['update_rate']),
        )


PLANAR_AXES = ('linear', 'lateral', 'angular')


@attrs.define(slots=False)
class PlanarSpec:
    """Body response of the base in a planar simulator, per axis. An axis left out is unbounded with unit gain."""

    velocity: dict[str, Range]
    acceleration: dict[str, float]
    gain: dict[str, float]
    lag: float
    traction: float | None
    pivot: float
    timeout: float | None

    @classmethod
    def from_dict(cls, d: dict[str, typing.Any]) -> PlanarSpec:
        def axes(key: str) -> dict[str, typing.Any]:
            raw = d.get(key, {})
            if not isinstance(raw, dict) or not set(raw) <= set(PLANAR_AXES):
                raise ValueError(f"planar.{key}: expected a mapping over {list(PLANAR_AXES)}; got {raw!r}")
            return raw

        return cls(
            velocity={axis: Range.from_value(v, f'planar.velocity.{axis}') for axis, v in axes('velocity').items()},
            acceleration={axis: float(v) for axis, v in axes('acceleration').items()},
            gain={axis: float(v) for axis, v in axes('gain').items()},
            lag=float(d.get('lag', 0.0)),
            traction=float(d['traction']) if 'traction' in d else None,
            pivot=float(d.get('pivot', 0.0)),
            timeout=float(d['timeout']) if 'timeout' in d else None,
        )


@attrs.define(slots=False)
class CapConfig:
    """Raw cap file content + the path it came from, for error messages and
    adapter-sub-block access. Typed subclasses front the fields adapters need
    directly."""

    path: Path
    raw: dict[str, typing.Any]

    def sub(self, adapter: str) -> dict[str, typing.Any]:
        """Return the adapter-specific sub-block or an empty dict."""
        v = self.raw.get(adapter, {})
        if not isinstance(v, dict):
            raise ValueError(f"{self.path}: '{adapter}' sub-block must be a mapping; got {type(v).__name__}")
        return v


INFLATION_MARGIN = 0.4
CLIMBABLE = ('stairs', 'escalator')


@attrs.define(slots=False)
class MobileSpec(CapConfig):
    """Primitives from caps/mobile.yaml (flat, single-instance)."""

    @property
    def odom_frame(self) -> str:
        return str(self.raw.get('odom_frame', 'odom'))

    @property
    def sensor_frame(self) -> str | None:
        v = self.raw.get('sensor_frame')
        return None if v is None else str(v)

    @property
    def radius(self) -> float | None:
        v = self.raw.get('radius')
        return None if v is None else float(v)

    @property
    def is_holonomic(self) -> bool:
        return bool(self.raw.get('is_holonomic', False))

    @property
    def climbs(self) -> frozenset[str]:
        """Stairs kinds the base can drive through, a subset of CLIMBABLE."""
        v = self.raw.get('climbs') or []
        if not isinstance(v, list) or not set(v) <= set(CLIMBABLE):
            raise ValueError(f"{self.path}: 'climbs' must be a list among {list(CLIMBABLE)}; got {v!r}")
        return frozenset(v)

    @property
    def polygons_dict(self) -> dict[str, PolygonSpec]:
        raw = self.raw.get('polygons_dict')
        if not raw:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"{self.path}: 'polygons_dict' must be a mapping; got {type(raw).__name__}")
        return {name: PolygonSpec.from_dict(name, entry) for name, entry in raw.items()}

    @property
    def footprint(self) -> list[list[float]] | None:
        v = self.raw.get('footprint')
        if v is None:
            return None
        if isinstance(v, str):
            v = ast.literal_eval(v)
        return [[float(c) for c in pt] for pt in v]

    @property
    def footprint_padding(self) -> float | None:
        v = self.raw.get('footprint_padding')
        return None if v is None else float(v)

    @property
    def inflation_radius(self) -> float | None:
        """Costmap inflation radius, the outer radius of the padded footprint or `radius` plus INFLATION_MARGIN unless set."""
        v = self.raw.get('inflation_radius')
        if v is not None:
            return float(v)
        footprint = self.footprint
        outer = [max(math.hypot(x, y) for x, y in footprint) + (self.footprint_padding or 0.0)] if footprint else []
        if self.radius is not None:
            outer.append(self.radius)
        return max(outer) + INFLATION_MARGIN if outer else None

    @property
    def velocity_limits(self) -> VelocityLimits | None:
        v = self.raw.get('velocity_limits')
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError(f"{self.path}: 'velocity_limits' must be a mapping; got {type(v).__name__}")
        return VelocityLimits.from_dict(v)

    @property
    def acceleration_limits(self) -> AccelerationLimits | None:
        v = self.raw.get('acceleration_limits')
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError(f"{self.path}: 'acceleration_limits' must be a mapping; got {type(v).__name__}")
        return AccelerationLimits.from_dict(v)

    @property
    def actions(self) -> ActionsSpec | None:
        v = self.raw.get('actions')
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError(f"{self.path}: 'actions' must be a mapping; got {type(v).__name__}")
        return ActionsSpec.from_dict(v)

    @property
    def planar(self) -> PlanarSpec | None:
        v = self.raw.get('planar')
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError(f"{self.path}: 'planar' must be a mapping; got {type(v).__name__}")
        return PlanarSpec.from_dict(v)

    @property
    def laser(self) -> LaserSpec | None:
        v = self.raw.get('laser')
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError(f"{self.path}: 'laser' must be a mapping; got {type(v).__name__}")
        return LaserSpec.from_dict(v)


@attrs.define(slots=False)
class InstanceSpec(CapConfig):
    """Per-instance spec for multi-instance caps (arm, lift, gripper).

    An instance's `path` points at the cap file; `name` is the dict key within
    that file. Adapter-specific sub-blocks are nested inside each instance.
    """

    name: str


@attrs.define(slots=False)
class ArmSpec(InstanceSpec):
    """Serial-chain arm primitives. Resolves via SRDF if `srdf:` is declared
    and the matching field isn't explicit; explicit always wins."""

    _srdf_cache: dict[str, typing.Any] | None = attrs.field(default=None, init=False)

    def _srdf(self) -> dict[str, typing.Any]:
        if self._srdf_cache is None:
            srdf_ref = self.raw.get('srdf')
            self._srdf_cache = _parse_srdf_group(srdf_ref, self.name) if srdf_ref else {}
        return self._srdf_cache

    @property
    def base_link(self) -> str:
        v = self.raw.get('base_link') or self._srdf().get('base_link')
        if v is None:
            raise ValueError(f"{self.path}: arm '{self.name}' has no base_link (not explicit, not derivable from srdf)")
        return str(v)

    @property
    def tip_link(self) -> str:
        v = self.raw.get('tip_link') or self._srdf().get('tip_link')
        if v is None:
            raise ValueError(f"{self.path}: arm '{self.name}' has no tip_link (not explicit, not derivable from srdf)")
        return str(v)

    @property
    def chain(self) -> list[str]:
        v = self.raw.get('chain')
        if v is None:
            v = self._srdf().get('chain')
        if not isinstance(v, list):
            raise ValueError(f"{self.path}: arm '{self.name}' has no chain (not explicit, not derivable from srdf)")
        return [str(j) for j in v]

    @property
    def controller(self) -> str:
        v = self.raw.get('controller')
        if v is None:
            raise ValueError(f"{self.path}: arm '{self.name}' missing 'controller' (controllers are not in SRDF; always author explicitly)")
        return str(v)

    @property
    def planning_group(self) -> str | None:
        mv = self.raw.get("moveit")
        if not isinstance(mv, dict):
            return None
        pg = mv.get("planning_group")
        return None if pg is None else str(pg)

    @property
    def workspace(self) -> dict[str, object] | None:
        """Raw workspace block (type / frame / min / max), or None if not declared."""
        ws = self.raw.get("workspace")
        return ws if isinstance(ws, dict) else None

    @property
    def stall_error(self) -> float:
        """Tracking error in rad above which the arm counts as blocked."""
        return float(self.raw.get("stall_error", 0.2))

    @property
    def stall_speed(self) -> float:
        """Joint speed in rad/s below which a joint with a stall error counts as stopped."""
        return float(self.raw.get("stall_speed", 0.05))

    @property
    def stall_time(self) -> float:
        """Seconds the tracking error must stay above `stall_error` before a reach fails as blocked."""
        return float(self.raw.get("stall_time", 0.5))

    @property
    def scene_radius(self) -> float:
        """Radius in m around the base within which episode objects enter the planning scene."""
        return float(self.raw.get("scene_radius", 1.5))

    @property
    def joint_limits(self) -> dict[str, dict[str, object]]:
        """``{joint_name: limits}`` from the moveit joint_limits file, empty when not declared."""
        mv = self.raw.get("moveit")
        if not isinstance(mv, dict):
            return {}
        jl = mv.get("joint_limits")
        if not isinstance(jl, dict):
            return {}
        full = _resolve_find_ref(f"$(find {jl.get('package', mv.get('package'))})/{jl.get('path', 'config/joint_limits.yaml')}")
        with open(full) as f:
            data = yaml.safe_load(f) or {}
        return data.get("joint_limits", {})

    @property
    def named_poses(self) -> dict[str, dict[str, float]]:
        """``{pose_name: {joint_name: radians}}``. Empty dict if not declared.

        Validates each entry's ``joints:`` block; raises ValueError on shape errors."""
        raw = self.raw.get("named_poses")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"{self.path}: arm '{self.name}' 'named_poses' must be a mapping; got {type(raw).__name__}")
        out: dict[str, dict[str, float]] = {}
        for name, entry in raw.items():
            if not isinstance(entry, dict):
                raise ValueError(f"{self.path}: named_pose '{name}' must be a mapping")
            joints = entry.get("joints")
            if not isinstance(joints, dict):
                raise ValueError(f"{self.path}: named_pose '{name}' missing 'joints' mapping")
            out[str(name)] = {str(j): float(v) for j, v in joints.items()}
        return out

    @property
    def gestures(self) -> dict[str, GestureSpec]:
        """Per-robot gesture overrides. Empty dict if absent."""
        raw = self.raw.get("gestures")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"{self.path}: arm '{self.name}' 'gestures' must be a mapping; got {type(raw).__name__}")
        return {str(k): GestureSpec.parse(v) for k, v in raw.items()}


@attrs.define(slots=False)
class LiftSpec(InstanceSpec):
    """Prismatic lift primitives."""

    @property
    def joint(self) -> str:
        v = self.raw.get('joint')
        if v is None:
            raise ValueError(f"{self.path}: lift '{self.name}' missing 'joint'")
        return str(v)

    @property
    def controller(self) -> str:
        v = self.raw.get('controller')
        if v is None:
            raise ValueError(f"{self.path}: lift '{self.name}' missing 'controller'")
        return str(v)


@attrs.define(slots=False)
class GripperSpec(InstanceSpec):
    """Gripper primitives with an optional back-reference to its arm."""

    @property
    def arm(self) -> str | None:
        v = self.raw.get('arm')
        return None if v is None else str(v)

    @property
    def joint(self) -> str:
        v = self.raw.get('joint')
        if v is None:
            raise ValueError(f"{self.path}: gripper '{self.name}' missing 'joint'")
        return str(v)

    @property
    def controller(self) -> str:
        v = self.raw.get('controller')
        if v is None:
            raise ValueError(f"{self.path}: gripper '{self.name}' missing 'controller'")
        return str(v)

    @property
    def positions(self) -> dict[str, float]:
        v = self.raw.get('positions')
        if v is None or not isinstance(v, dict):
            raise ValueError(f"{self.path}: gripper '{self.name}' missing 'positions'")
        return {str(k): float(val) for k, val in v.items()}

    @property
    def max_effort(self) -> float:
        v = self.raw.get('max_effort')
        if v is None:
            raise ValueError(f"{self.path}: gripper '{self.name}' missing 'max_effort'")
        return float(v)


_INSTANCE_CLASSES: dict[str, type[InstanceSpec]] = {
    'arm': ArmSpec,
    'lift': LiftSpec,
    'gripper': GripperSpec,
}


@attrs.define(slots=False)
class RobotCaps:
    """Lazy, typed view over a robot's caps/ directory.

    File presence → cap advertisement. Typed accessors front the cap files;
    the generic `raw(<cap>)` exposes adapter-sub-blocks authored on a cap
    Arena doesn't model as a first-class spec yet.

    ``resolved``/``catalog`` (optional, allocation-derived caps): when set,
    a placement whose component declares a `caps:` block advertises that cap even
    without a static `caps/<type>.yaml` file, and instances render one spec per
    placement instead of reading the static file. Leaving both `None` (the
    default) reproduces the file-only behavior byte-identically.
    """

    caps_dir: Path
    resolved: ResolvedAssembly | None = None
    catalog: Catalog | None = None
    # chassis assembly prefix (${prefix}); robots with no prefix convention (jackal) pass ""
    prefix: str = _CAPS_PREFIX

    _cached: dict[str, typing.Any] = attrs.field(factory=dict, init=False)

    @property
    def available(self) -> frozenset[str]:
        """Cap names present as `caps/<name>.yaml`, unioned with placement types
        whose component declares a `caps:` block when `resolved` is set."""
        file_stems = frozenset(p.stem for p in self.caps_dir.glob('*.yaml')) if self.caps_dir.is_dir() else frozenset()
        if self.resolved is None:
            return file_stems
        assert self.catalog is not None
        placed = frozenset(placement.type for placement in self.resolved.placements if self.catalog.get(placement.type, placement.variant).caps)
        return file_stems | placed

    def _load_cap_file(self, cap: str) -> dict[str, typing.Any]:
        if cap in self._cached:
            return self._cached[cap]
        path = self.caps_dir / f'{cap}.yaml'
        if not path.is_file():
            raise FileNotFoundError(f"cap '{cap}' not declared: {path} does not exist")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top-level structure must be a mapping; got {type(data).__name__}")
        self._cached[cap] = data
        return data

    @property
    def mobile(self) -> MobileSpec:
        data = self._load_cap_file('mobile')
        return MobileSpec(path=self.caps_dir / 'mobile.yaml', raw=data)

    @property
    def arm(self) -> dict[str, ArmSpec] | None:
        return self._instances('arm', ArmSpec)

    @property
    def lift(self) -> dict[str, LiftSpec] | None:
        return self._instances('lift', LiftSpec)

    @property
    def gripper(self) -> dict[str, GripperSpec] | None:
        return self._instances('gripper', GripperSpec)

    def _instances(
        self,
        cap: str,
        cls: type[InstanceSpec],
    ) -> dict[str, typing.Any] | None:
        placements = [p for p in self.resolved.placements if p.type == cap] if self.resolved is not None else []
        if placements:
            assert self.resolved is not None and self.catalog is not None
            path = self.caps_dir / f'{cap}.yaml'
            out: dict[str, typing.Any] = {}
            for placement in placements:
                component = self.catalog.get(placement.type, placement.variant)
                context: dict[str, typing.Any] = {
                    'mount': _frame_stem(placement.mount),
                    'variant': placement.variant,
                    'parent': resolve_mount_parent(self.resolved, self.catalog, placement.mount),
                    'prefix': self.prefix,
                    **placement.params,
                    **placement.overrides,
                }
                rendered = YAMLReplacer(context).replace(copy.deepcopy(component.caps))
                rendered = _substitute_keys(rendered, context)
                if not isinstance(rendered, dict):
                    raise ValueError(f"{placement.type} component {placement.variant!r}: 'caps' must render to a mapping; got {type(rendered).__name__}")
                out[placement.mount.name] = cls(path=path, raw=rendered, name=placement.mount.name)
            return out

        if cap not in self.available:
            return None
        data = self._load_cap_file(cap)
        path = self.caps_dir / f'{cap}.yaml'
        out = {}
        for name, entry in data.items():
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: '{name}' must be a mapping (dict-keyed instance); got {type(entry).__name__}. See robots/README.md on the uniform dict-keyed shape for multi-instance caps.")
            out[str(name)] = cls(path=path, raw=entry, name=str(name))
        return out


def _substitute_keys(obj: object, context: dict[str, typing.Any]) -> object:
    """`YAMLReplacer.replace` only substitutes dict VALUES (and `**spread` keys);
    a plain `${...}` dict key (e.g. caps `named_poses.<pose>.joints`, keyed by
    templated joint name) passes through unresolved. Walk an already-value-rendered
    structure and substitute those keys too, with the same context."""
    if isinstance(obj, dict):
        return {(YAMLReplacer(context).replace(k) if isinstance(k, str) else k): _substitute_keys(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_keys(v, context) for v in obj]
    return obj


def _parse_srdf_group(srdf_ref: str, group_name: str) -> dict[str, typing.Any]:
    """Resolve a `$(find …)/…srdf[.xacro]` ref and extract a <group>'s primitives.

    Returns a dict with `base_link`, `tip_link`, and, if the group enumerates
    joints via <joints> or the URDF walk succeeds, `chain`.
    """
    src_path = _resolve_find_ref(srdf_ref)
    if src_path.suffix == '.xacro':
        xml_text = subprocess.check_output(
            ['xacro', '--inorder', str(src_path)],
            text=True,
        )
        root = ET.fromstring(xml_text)
    else:
        root = ET.parse(str(src_path)).getroot()

    group = root.find(f"./group[@name='{group_name}']")
    if group is None:
        raise ValueError(f"{srdf_ref}: no <group name='{group_name}'> found")

    out: dict[str, typing.Any] = {}
    chain = group.find('./chain')
    if chain is not None:
        out['base_link'] = chain.attrib.get('base_link')
        out['tip_link'] = chain.attrib.get('tip_link')

    joints_nodes = group.findall('./joint')
    if joints_nodes:
        out['chain'] = [j.attrib['name'] for j in joints_nodes]

    return out


def _resolve_find_ref(ref: str) -> Path:
    """Resolve `$(find <pkg>)/<rest>` to an absolute Path. Non-$(find) inputs
    are treated as filesystem paths as-is."""
    ref = ref.strip()
    if ref.startswith('$(find '):
        end = ref.index(')')
        pkg = ref[len('$(find ') : end]
        rest = ref[end + 1 :].lstrip('/')
        from ament_index_python.packages import get_package_share_path

        return get_package_share_path(pkg) / rest
    return Path(ref)
