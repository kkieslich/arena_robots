"""Component catalog + sensor-template rendering. Launch-free and pure: loads
``components/<type>/<variant>/component.yaml`` and renders a robot's
:class:`arena_robots.assembly.ResolvedAssembly` into its effective gz ``SensorSpec``
list and the wrapper's own sensor XML. No xacro loading, no ROS: this module only
decides what a placement's sensor declarations *say*, not how they get instantiated.

``component.yaml`` shape::

    xacro:
      include: <macro file, robot-urdf-dir-relative>
      macro: <macro name>
      attach: {...}           # attach-interface template (parent/xyz/rpy keys, mount-templated)
      args: {...}             # declared overridable macro-arg surface with defaults
    sensor:
      gz:
        - {name: <SensorSpec.name>, type: <SensorSpec.type>, topic: <relative topic>,
           frame: <SensorSpec.frame>, sensor: <backing gz <sensor> name>,
           define: {collision: <collision name>, update_rate: <hz>}}
        # OPTIONAL define: emit the backing <sensor> element from the wrapper xacro
        # (currently only type: contact). Omit it when the component's own xacro
        # already emits the <sensor>.
        # one entry per gz output the component contributes (a component is a
        # LIST of outputs, e.g. gpu_lidar -> laserscan + pointcloud)
    ros2_control:      # OPTIONAL (arm components only); absent -> []
      joints:
        - {name: <joint name template>, command_interfaces: [...], state_interfaces: [...]}
    control:           # OPTIONAL; absent -> {}
      controller: <controller name template, e.g. ${mount}_controller>
      type: <controller plugin type>
      ros__parameters: {...}   # templated verbatim
    caps:              # OPTIONAL; absent -> {}
      ...              # full caps/arm.yaml raw shape, templated verbatim (caps.py ArmSpec)
    frames:            # OPTIONAL; absent -> {} (mount-on-part chaining)
      <name>: <template>   # e.g. {top: "${prefix}${mount}_ewellix_lift_top_link"}
    power:             # OPTIONAL; absent -> {}
      static_power_w: <W drawn while idle>
    mass:              # OPTIONAL; absent -> {} (no contribution)
      kg: <kg added on top of the robot's model_params.yaml mass.base_kg>

Every string value in a ``sensor.gz`` entry is a template rendered per-placement through
``arena_rclpy_mixins.yaml_replace.YAMLReplacer``. The substitution context is::

    {'mount': _frame_stem(placement.mount), 'prefix': prefix, **placement.params, **placement.overrides}

i.e. the mount name and frame prefix are always available, a placement's resolved macro
``params`` are spread in next, and ``placement.overrides`` (assembly.py, authoring-only,
never from a fleet-def request) are spread in *last* so they win.

Override-key vocabulary (the minimal set a component author needs to reproduce an
asymmetric multi-instance robot like rbtheron's dual lidar):

- ``name``  overrides a placement's ``SensorSpec.name`` base (``${name:-<component-default>}``
  in the template); the component's own per-output literal suffix (e.g. ``_points``) still
  appends after it, so overriding once fixes every output of that placement consistently
  (``name: lidar_rear`` -> ``lidar_rear`` / ``lidar_rear_points``).
- ``topic`` overrides a placement's base relative topic (``${topic:-<component-default>}``);
  again the component's own per-output suffix (e.g. ``/points``) still appends
  (``topic: scan/rear`` -> ``scan/rear`` / ``scan/rear/points``).

``frame`` and ``sensor`` are templated directly off ``${mount}``/``${prefix}`` with no
override indirection: mount naming alone is expressive enough (mount frames adopt the
chassis's own frame names), so a component rarely needs more than the two keys above.
Components are, of course, free to reference ``${name:-...}``/``${topic:-...}`` inside
their ``frame``/``sensor`` templates too if a real robot needs it.
"""

from __future__ import annotations

import copy
import typing
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import quoteattr

import attrs
import yaml
from ament_index_python.packages import get_package_share_path
from arena_rclpy_mixins.yaml_replace import YAMLReplacer

from arena_robots.assembly import Mount, ResolvedAssembly
from arena_robots.Sensor import SensorSpec

if typing.TYPE_CHECKING:
    from arena_robots.Robot import RobotView


@attrs.define
class ComponentSpec:
    """One parsed ``components/<type>/<variant>/component.yaml``."""

    xacro_include: str
    xacro_macro: str
    attach: dict[str, str]
    args: dict[str, object]
    sensor_gz: list[dict[str, object]]
    """Ordered, un-rendered ``sensor.gz`` output templates; render per-placement via
    :func:`render_effective_sensors`, never mutate this list in place (shared across
    every placement of this component). An entry may carry an optional ``define``
    block, which ``render_wrapper_xacro`` turns into a ``<gazebo><sensor>`` element."""
    ros2_control_joints: list[dict[str, object]] = attrs.field(factory=list)
    """Un-rendered ``ros2_control.joints`` entries; empty for
    components with no actuator axes (sensors). Render per-placement via
    :func:`render_control_joints`, then injected post-render into the chassis's
    own ``ros2_control`` tag (arena_simulation_setup's urdf loader,
    ``_inject_ros2_control_joints``, arm-on-any-chassis merge)."""
    control: dict[str, object] = attrs.field(factory=dict)
    """Un-rendered ``control`` block (``controller``/``type``/``ros__parameters``);
    empty for components with no controller. Render per-placement via
    :func:`render_effective_control`."""
    caps: dict[str, object] = attrs.field(factory=dict)
    """Un-rendered ``caps`` block, the full ``caps/arm.yaml`` raw shape; empty for
    components with no caps contribution. Rendered by ``RobotCaps``/``effective_caps``
    (caps.py), not this module."""
    frames: dict[str, str] = attrs.field(factory=dict)
    """Un-rendered ``frames`` templates: named frames this component
    exports, for another mount's ``Mount.parent`` to chain onto (e.g. a lift's
    ``top`` frame). Empty for components nothing else mounts onto. Resolved per
    placement by :func:`resolve_mount_parent`."""
    power: dict[str, object] = attrs.field(factory=dict)
    """Static power parameters of this component."""
    mass: dict[str, object] = attrs.field(factory=dict)
    """Mass contribution of this component, added on top of the robot's base mass."""
    variants: list[str] = attrs.field(factory=list)
    """Variant names this component serves when no dedicated ``<variant>/`` dir exists
    (family component, e.g. one ``arm/ur`` serving the whole UR family, parameterized
    per-placement off ``${variant}`` in attach name / ur_description config paths /
    ur_type). Empty for ordinary one-dir-per-variant components."""

    @classmethod
    def from_yaml(cls, path: Path) -> ComponentSpec:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"component.yaml at {path} must be a mapping; got {type(data).__name__}")

        xacro = data.get('xacro', {})
        if not isinstance(xacro, dict) or 'include' not in xacro or 'macro' not in xacro:
            raise ValueError(f"component.yaml at {path}: 'xacro' must declare 'include' and 'macro'")

        sensor = data.get('sensor', {})
        gz = sensor.get('gz', []) if isinstance(sensor, dict) else []
        if not isinstance(gz, list):
            raise ValueError(f"component.yaml at {path}: 'sensor.gz' must be a list; got {type(gz).__name__}")

        ros2_control = data.get('ros2_control', {})
        joints = ros2_control.get('joints', []) if isinstance(ros2_control, dict) else []
        if not isinstance(joints, list):
            raise ValueError(f"component.yaml at {path}: 'ros2_control.joints' must be a list; got {type(joints).__name__}")

        control = data.get('control', {})
        if not isinstance(control, dict):
            raise ValueError(f"component.yaml at {path}: 'control' must be a mapping; got {type(control).__name__}")

        caps = data.get('caps', {})
        if not isinstance(caps, dict):
            raise ValueError(f"component.yaml at {path}: 'caps' must be a mapping; got {type(caps).__name__}")

        frames = data.get('frames', {})
        if not isinstance(frames, dict):
            raise ValueError(f"component.yaml at {path}: 'frames' must be a mapping; got {type(frames).__name__}")

        variants = data.get('variants', [])
        if not isinstance(variants, list):
            raise ValueError(f"component.yaml at {path}: 'variants' must be a list; got {type(variants).__name__}")

        power = data.get('power', {})
        if not isinstance(power, dict):
            raise ValueError(f"component.yaml at {path}: 'power' must be a mapping; got {type(power).__name__}")

        mass = data.get('mass', {})
        if not isinstance(mass, dict):
            raise ValueError(f"component.yaml at {path}: 'mass' must be a mapping; got {type(mass).__name__}")

        return cls(
            xacro_include=str(xacro['include']),
            xacro_macro=str(xacro['macro']),
            attach=dict(xacro.get('attach', {})),
            args=dict(xacro.get('args', {})),
            sensor_gz=[dict(entry) for entry in gz],
            ros2_control_joints=[dict(joint) for joint in joints],
            control=dict(control),
            caps=dict(caps),
            frames={str(k): str(v) for k, v in frames.items()},
            power=dict(power),
            mass=dict(mass),
            variants=[str(v) for v in variants],
        )


class Catalog:
    """``components/<type>/<variant>/component.yaml`` loader, cached per (type, variant).

    ``root`` defaults lazily to ``<arena_robots share>/components``; pass an explicit
    ``root`` (e.g. a ``tmp_path`` fixture tree) to keep tests off the ament index.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root if root is not None else get_package_share_path('arena_robots') / 'components'
        self._cache: dict[tuple[str, str], ComponentSpec] = {}

    def get(self, type_: str, variant: str) -> ComponentSpec:
        key = (type_, variant)
        if key not in self._cache:
            path = self._root / type_ / variant / 'component.yaml'
            if path.is_file():
                self._cache[key] = ComponentSpec.from_yaml(path)
            else:
                self._cache[key] = self._resolve_family(type_, variant)
        return self._cache[key]

    def _resolve_family(self, type_: str, variant: str) -> ComponentSpec:
        """Resolve a variant with no dedicated ``<variant>/`` dir against a family
        component (a sibling whose ``variants:`` list names it, e.g. one ``arm/ur``
        serving the whole UR family). The shared spec is rendered per-placement off
        ``${variant}`` (catalog contexts thread in ``placement.variant``). Raises
        listing both concrete dirs and every family-declared variant."""
        type_dir = self._root / type_
        concrete = sorted(p.name for p in type_dir.iterdir() if p.is_dir()) if type_dir.is_dir() else []
        family_variants: list[str] = []
        for name in concrete:
            candidate = type_dir / name / 'component.yaml'
            if not candidate.is_file():
                continue
            spec = ComponentSpec.from_yaml(candidate)
            if variant in spec.variants:
                return spec
            family_variants.extend(spec.variants)
        available = sorted(set(concrete) | set(family_variants))
        raise RuntimeError(f"component '{type_}/{variant}' not found; available '{type_}' variants: {available}")


def _render_sensor_templates(component: ComponentSpec, context: dict[str, typing.Any]) -> list[dict[str, object]]:
    return [YAMLReplacer(context).replace(copy.deepcopy(template)) for template in component.sensor_gz]


def render_effective_sensors(resolved: ResolvedAssembly, catalog: Catalog, *, prefix: str = 'robot_') -> list[SensorSpec]:
    """Render every placement's ``sensor.gz`` templates into the robot's full effective
    ``SensorSpec`` list, in placement then per-component-output order. Each template dict
    is deep-copied before rendering (``YAMLReplacer.replace`` mutates its input in place;
    the source lives on the shared, cached :class:`ComponentSpec`)."""
    out: list[SensorSpec] = []
    for placement in resolved.placements:
        component = catalog.get(placement.type, placement.variant)
        context: dict[str, typing.Any] = {
            'mount': _frame_stem(placement.mount),
            'variant': placement.variant,
            'prefix': prefix,
            **placement.params,
            **placement.overrides,
        }
        for entry in _render_sensor_templates(component, context):
            out.append(
                SensorSpec(
                    name=str(entry['name']),
                    type=str(entry['type']),
                    topic='${namespace}/' + str(entry['topic']),
                    frame=str(entry['frame']),
                    sensor=str(entry['sensor']) if 'sensor' in entry else None,
                )
            )
    return out


def _frame_stem(mount: Mount) -> str:
    """Identity stem substituted for ``${mount}`` in component templates (frame, joint,
    sensor frame_id, controller name). Defaults to the addressing name; a mount's
    ``frame`` override decouples the sim2real TF/joint/sensor contract from a renamed
    addressing name."""
    return mount.frame or mount.name


def resolve_mount_parent(resolved: ResolvedAssembly, catalog: Catalog, mount: Mount) -> str:
    """Resolve a mount's ``parent`` for templating (chained mounts).

    An ordinary mount returns its literal parent frame name unchanged. A chained
    mount (``"@<mount>:<frame>"``) resolves to the referenced placement's rendered
    ``frames`` template, substituted with an EMPTY ``prefix`` so the caller's own
    ``${prefix}${parent}`` attach/caps template re-applies the real prefix exactly
    once (the same shape a literal parent already goes through)."""
    chained = mount.chained_parent
    if chained is None:
        return mount.parent
    ref_mount_name, frame_name = chained
    ref = next((p for p in resolved.placements if p.mount.name == ref_mount_name), None)
    if ref is None:
        raise RuntimeError(f"mount '{mount.name}' parent references unpopulated mount '{ref_mount_name}'")
    component = catalog.get(ref.type, ref.variant)
    if frame_name not in component.frames:
        raise RuntimeError(f"component '{ref.type}/{ref.variant}' (mount '{ref_mount_name}') does not export frame '{frame_name}'; declared frames: {sorted(component.frames)}")
    context: dict[str, typing.Any] = {'mount': _frame_stem(ref.mount), 'variant': ref.variant, 'prefix': '', **ref.params, **ref.overrides}
    return YAMLReplacer(context).replace(component.frames[frame_name])


_XACRO_NS = 'http://www.ros.org/wiki/xacro'


def _gazebo_sensor_lines(entry: dict[str, object]) -> list[str]:
    define = entry.get('define')
    if not define:
        return []
    if not isinstance(define, dict):
        raise ValueError(f"sensor {entry['name']!r}: 'define' must be a mapping")

    sensor_type = str(define.get('type') or entry['type'])
    if sensor_type != 'contact':
        raise ValueError(f"sensor {entry['name']!r}: only 'contact' define blocks are implemented, got {sensor_type!r}")

    if not entry.get('sensor'):
        raise ValueError(f"sensor {entry['name']!r}: 'sensor' is required with a 'define' block")
    link = str(entry['frame'])
    name = str(entry['sensor'])
    collisions = define.get('collision') or []
    if isinstance(collisions, str):
        collisions = [collisions]
    if not isinstance(collisions, (list, tuple)) or not collisions:
        raise ValueError(f"sensor {entry['name']!r}: 'define.collision' is required for a contact sensor")

    lines = [
        f'  <gazebo reference={quoteattr(link)}>',
        f'    <sensor name={quoteattr(name)} type="contact">',
        '      <always_on>true</always_on>',
    ]
    if 'update_rate' in define:
        lines.append(f'      <update_rate>{define["update_rate"]}</update_rate>')
    lines.append('      <contact>')
    lines.extend(f'        <collision>{str(collision)}</collision>' for collision in collisions)
    lines.extend(['      </contact>', '    </sensor>', '  </gazebo>'])
    return lines


def _chassis_args(chassis_path: Path) -> list[tuple[str, str]]:
    """``(name, default)`` for every top-level ``<xacro:arg>`` in the chassis xacro, in
    document order. The wrapper forwards exactly this surface.

    Matched by local tag name against direct children of ``<robot>`` only: chassis
    files across the fleet declare ``xmlns:xacro`` with at least four different URIs
    (``ros.org/wiki/xacro``, ``wiki.ros.org/xacro``, ``www.ros.org/wiki/xacro``, and a
    typo'd ``wiki.ros.org.xacro``), all functionally equivalent to the ``xacro`` CLI, so
    a fully-qualified-name match would miss most of the fleet. Direct-children-only
    (rather than a full subtree ``.iter()``) also skips non-top-level ``<xacro:arg>``
    elements some chassis files carry as inert children of an unrelated tag (e.g.
    turtlebot's ``<xacro:include>...<xacro:arg name="gazebo" .../></xacro:include>``,
    which xacro itself ignores with a warning)."""
    root = ET.parse(chassis_path).getroot()
    return [(el.get('name', ''), el.get('default', '')) for el in root if el.tag.rsplit('}', 1)[-1] == 'arg']


def render_wrapper_xacro(view: RobotView, resolved: ResolvedAssembly, *, catalog: Catalog | None = None) -> str:
    """Generate a complete wrapper ``.urdf.xacro`` document for ``view``: forwards the
    chassis's own ``xacro:arg`` surface, includes the chassis xacro, then includes +
    invokes each placement's component macro at its mount's parent/origin.
    Deterministic: placements render in ``resolved.placements`` order
    (assembly.yaml defaults order); a component's macro file is included once even if
    placed at multiple mounts.

    The attach-template substitution context mirrors :func:`render_effective_sensors`
    (``mount``/``prefix``/``params``/``overrides``) plus xacro-side-only keys
    (``parent``, ``namespace``, ``gazebo_classic``, ``gazebo_ignition``) whose values are
    literal ``$(arg ...)`` references, left for the actual xacro run to resolve.
    ``parent`` is resolved via :func:`resolve_mount_parent`: an ordinary
    mount's literal parent name passes through; a chained mount (``"@<mount>:<frame>"``)
    resolves to the referenced placement's rendered frame instead.

    No ``ros2_control`` synthesis happens here (arm-on-any-chassis merge): the
    chassis and every placement's component each render their own ``ros2_control`` tag
    (or none) exactly as their own xacro dictates, so the rendered document may carry
    MULTIPLE ``ros2_control`` tags. arena_simulation_setup's urdf loader
    (``_inject_ros2_control_joints``) merges them post-render, fed by
    :func:`render_control_joints`'s control-joint patch, computed independently from
    ``resolved`` (not by parsing this wrapper's xacro output).
    """
    catalog = catalog if catalog is not None else Catalog()
    chassis_path = view.path / 'urdf' / f'{view.name}.urdf.xacro'
    args = _chassis_args(chassis_path)

    body_lines: list[str] = []
    included: set[str] = set()
    for placement in resolved.placements:
        component = catalog.get(placement.type, placement.variant)
        # package-qualified ($(find ...)) and absolute includes pass through verbatim,
        # everything else resolves relative to the robot's urdf/ tree
        if component.xacro_include.startswith('$(') or Path(component.xacro_include).is_absolute():
            macro_ref = component.xacro_include
        else:
            macro_ref = str(view.path / 'urdf' / component.xacro_include)
        if macro_ref not in included:
            body_lines.append(f'  <xacro:include filename={quoteattr(macro_ref)}/>')
            included.add(macro_ref)

        context: dict[str, typing.Any] = {
            'mount': _frame_stem(placement.mount),
            'variant': placement.variant,
            'parent': resolve_mount_parent(resolved, catalog, placement.mount),
            'prefix': '$(arg prefix)',
            'namespace': '$(arg namespace)',
            'gazebo_classic': '$(arg gazebo_classic)',
            'gazebo_ignition': '$(arg gazebo_ignition)',
            **placement.params,
            **placement.overrides,
        }
        attach = YAMLReplacer(context).replace(copy.deepcopy(component.attach))
        attrs_str = ' '.join(f'{k}={quoteattr(str(v))}' for k, v in attach.items())
        x, y, z = placement.mount.xyz
        r, p, yw = placement.mount.rpy

        body_lines.append(f'  <xacro:{component.xacro_macro} {attrs_str}>')
        body_lines.append(f'    <origin xyz="{x} {y} {z}" rpy="{r} {p} {yw}"/>')
        body_lines.append(f'  </xacro:{component.xacro_macro}>')

        for entry in _render_sensor_templates(component, context):
            body_lines.extend(_gazebo_sensor_lines(entry))

    lines = ['<?xml version="1.0"?>', f'<robot name="{view.name}" xmlns:xacro="{_XACRO_NS}">']
    for name, default in args:
        lines.append(f'  <xacro:arg name={quoteattr(name)} default={quoteattr(default)}/>')
    lines.append(f'  <xacro:include filename={quoteattr(str(chassis_path))}/>')
    lines.extend(body_lines)
    lines.append('</robot>')
    return '\n'.join(lines) + '\n'


def render_control_joints(resolved: ResolvedAssembly, catalog: Catalog, *, prefix: str = 'robot_') -> list[dict[str, object]]:
    """Render every placement's ``ros2_control.joints`` templates into the robot's
    control-joint patch (arm-on-any-chassis merge): the ``<joint>``
    entries a joint-bearing part (e.g. an arm) contributes, for post-render injection
    into the chassis's own ``ros2_control`` tag (arena_simulation_setup's urdf loader,
    ``_inject_ros2_control_joints``). Unlike :func:`render_wrapper_xacro`'s xacro-side
    templating, ``prefix`` here is the real resolved value (not a ``$(arg prefix)``
    token): injection runs after xacro has already resolved everything. Same
    templating context as :func:`render_effective_sensors`. Empty when no placement
    declares joints."""
    out: list[dict[str, object]] = []
    for placement in resolved.placements:
        component = catalog.get(placement.type, placement.variant)
        if not component.ros2_control_joints:
            continue
        context: dict[str, typing.Any] = {
            'mount': _frame_stem(placement.mount),
            'variant': placement.variant,
            'prefix': prefix,
            **placement.params,
            **placement.overrides,
        }
        out.extend(YAMLReplacer(context).replace(copy.deepcopy(component.ros2_control_joints)))
    return out


def render_effective_control(resolved: ResolvedAssembly, base_control: dict, catalog: Catalog, *, prefix: str = 'robot_') -> tuple[dict, list[str]]:
    """Deep-merge every placement's rendered ``control`` block into a copy of the
    chassis's ``control.yaml`` (control synthesis): one
    ``controller_manager.ros__parameters.<controller>: {type: ...}`` entry plus one
    top-level ``<controller>: {ros__parameters: ...}`` section per placement that
    declares a ``control`` block. Returns ``(merged, extra_controller_names)``; ``base_control``
    is not mutated. Same templating context as :func:`render_effective_sensors`."""
    merged = copy.deepcopy(base_control)
    cm_params = merged.setdefault('controller_manager', {}).setdefault('ros__parameters', {})
    extra_controllers: list[str] = []
    for placement in resolved.placements:
        component = catalog.get(placement.type, placement.variant)
        if not component.control:
            continue
        context: dict[str, typing.Any] = {
            'mount': _frame_stem(placement.mount),
            'variant': placement.variant,
            'prefix': prefix,
            **placement.params,
            **placement.overrides,
        }
        rendered = YAMLReplacer(context).replace(copy.deepcopy(component.control))
        controller_name = str(rendered['controller'])
        cm_params[controller_name] = {'type': str(rendered['type'])}
        merged[controller_name] = {'ros__parameters': rendered.get('ros__parameters', {})}
        extra_controllers.append(controller_name)
    return merged, extra_controllers
