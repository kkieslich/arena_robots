import functools
import sys
import typing
from pathlib import Path

import attrs
import yaml
from ament_index_python.packages import get_package_share_path
from arena_simulation_setup.shared import Light
from arena_simulation_setup.tree import Identifier, PathView, SimplePathResolver
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.models import ModelWrapper
from arena_simulation_setup.utils.models.model_loader import (
    ModelProvider_URDF,
    ModelProvider_USD,
)

from arena_robots import assembly as assembly_mod
from arena_robots.caps import MobileSpec, RobotCaps
from arena_robots.catalog import Catalog, render_effective_sensors
from arena_robots.Sensor import SensorSpec

_CATALOG = Catalog()


@attrs.frozen
class ControlSpec:
    """ros2_control wiring declared in ``model_params.yaml`` under ``control:``.

    Presence in the YAML opts the robot into the ros2_control path: an
    in-process controller_manager (hosted by the URDF's gz_ros2_control
    plugin, or a ros2_control_node under Isaac) whose ``controllers`` the
    task generator loads and activates at robot bring-up. Absence means the
    gazebo_native path (PosePublisher + pose_to_tf + bridged cmd_vel) runs instead.
    """

    mode: str
    controllers: tuple[str, ...]
    config: str | None = None
    cmd_vel_topic: str = "cmd_vel"
    odom_topic: str = "odom"

    @classmethod
    def from_dict(cls, data: typing.Mapping[str, typing.Any]) -> "ControlSpec":
        mode = str(data.get("mode", "gazebo_native"))
        controllers_raw = data.get("controllers", [])
        if not isinstance(controllers_raw, list):
            raise ValueError(f"control.controllers must be a list; got {type(controllers_raw).__name__}")
        controllers = tuple(str(c) for c in controllers_raw)
        config = data.get("config")
        return cls(
            mode=mode,
            controllers=controllers,
            config=str(config) if config is not None else None,
            cmd_vel_topic=str(data.get("cmd_vel_topic", "cmd_vel")),
            odom_topic=str(data.get("odom_topic", "odom")),
        )

    @property
    def is_ros2_control(self) -> bool:
        return self.mode == "ros2_control"


class ModelParams(dict[str, typing.Any]):
    """Robot-wide identity (from ``model_params.yaml``) with typed accessors
    that delegate to the sibling ``caps/`` tree for mobile primitives.

    Constructed via :meth:`from_yaml` which records the file path; the
    ``caps`` cached property resolves ``<path>.parent / 'caps'`` so every
    ``ModelParams`` that came from disk can see its robot's cap tree without
    external wiring."""

    _path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ModelParams':
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Top-level structure in {path} must be a mapping")
        inst = cls(data)
        inst._path = path
        return inst

    @functools.cached_property
    def caps(self) -> RobotCaps:
        """Lazy view over ``<robot_dir>/caps/``. Empty if this ``ModelParams``
        wasn't loaded from disk or the caps dir doesn't exist."""
        caps_dir = self._path.parent / 'caps' if self._path is not None else Path('/dev/null')
        return RobotCaps(caps_dir=caps_dir)

    def _mobile(self) -> MobileSpec | None:
        if 'mobile' not in self.caps.available:
            return None
        return self.caps.mobile

    @property
    def base_frame(self) -> str:
        return str(self.get('base_frame', self.get('robot_base_frame', 'base_link')))

    @property
    def odom_frame(self) -> str:
        m = self._mobile()
        if m is not None:
            return m.odom_frame
        return self.get('robot_odom_frame', 'odom')

    @property
    def z_offset(self) -> float:
        return float(self.get('z_offset', 0.0))

    @property
    def control(self) -> ControlSpec | None:
        """Typed view of the ``control:`` block in model_params.yaml. ``None``
        when absent (gazebo_native pipeline)."""
        raw = self.get('control')
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"model_params 'control' must be a mapping; got {type(raw).__name__}")
        return ControlSpec.from_dict(raw)

    @property
    def sensors(self) -> list["SensorSpec"]:
        raw = self.get('sensors', [])
        if not isinstance(raw, list):
            raise ValueError(f"model_params 'sensors' must be a list; got {type(raw).__name__}")
        out: list[SensorSpec] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"model_params 'sensors[{i}]' must be a mapping; got {type(entry).__name__}")
            missing = {'name', 'type', 'topic', 'frame'} - set(entry)
            if missing:
                raise ValueError(f"model_params 'sensors[{i}]' missing required keys: {sorted(missing)}")
            out.append(
                SensorSpec(
                    name=str(entry['name']),
                    type=str(entry['type']),
                    topic=str(entry['topic']),
                    frame=str(entry['frame']),
                    sensor=str(entry['sensor']) if 'sensor' in entry else None,
                )
            )
        return out

    @property
    def lights(self) -> list[Light]:
        """Lights the robot carries, each on a frame of the robot and intrinsic, so the world keeps its default lighting."""
        raw = self.get('lights', [])
        if not isinstance(raw, list):
            raise ValueError(f"model_params 'lights' must be a list; got {type(raw).__name__}")
        out: list[Light] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"model_params 'lights[{i}]' must be a mapping; got {type(entry).__name__}")
            if 'frame' not in entry:
                raise ValueError(f"model_params 'lights[{i}]' needs a frame of the robot")
            light = converter.structure({**entry, 'intrinsic': True}, Light)
            if any(light.name == other.name for other in out):
                raise ValueError(f"model_params 'lights' declares {light.name!r} twice")
            out.append(light)
        return out

    @property
    def priority(self) -> float:
        """Rank for `robot: auto`, lowest wins. Undeclared sorts after every explicit value."""
        return float(self.get('priority', float('inf')))

    @property
    def capabilities(self) -> list[dict[str, typing.Any]]:
        """Structured multi-adapter declaration as a list of dicts."""
        raw = self.get('capabilities', [])
        if not isinstance(raw, list):
            raise ValueError(f"model_params 'capabilities' must be a list; got {type(raw).__name__}")
        return [dict(entry) for entry in raw]


class RobotView(PathView):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._cached_params: ModelParams | None = None
        self._cached_control: dict | None = None
        self._cached_assembly: assembly_mod.AssemblySpec | None = None
        self._assembly_loaded: bool = False

    @property
    def caps(self) -> RobotCaps:
        """Lazy view over robots/<name>/caps/, equivalent to
        ``self.model_params.caps``, exposed directly on ``RobotView`` for
        readability."""
        return self.model_params.caps

    @property
    def mobile(self) -> MobileSpec | None:
        """The robot's mobile cap, or ``None`` if it doesn't advertise ``mobile``.

        Shortcut for the common ``caps.mobile`` access that returns ``None``
        honestly instead of raising ``FileNotFoundError`` when the cap is absent.
        """
        if 'mobile' not in self.caps.available:
            return None
        return self.caps.mobile

    @property
    def model_params(self) -> ModelParams:
        if self._cached_params is None:
            path = self.path / 'model_params.yaml'
            if not path.is_file():
                raise FileNotFoundError(f"model_params.yaml not found for robot '{self.name}' at {path}")
            self._cached_params = ModelParams.from_yaml(path)
        return self._cached_params

    @property
    def mappings(self) -> str:
        return str(self.path / 'mappings.yaml')

    @property
    def control(self) -> dict:
        if self._cached_control is None:
            control_path = self.path / 'control.yaml'
            if not control_path.is_file():
                raise FileNotFoundError(f"control.yaml not found for robot '{self.name}' at {control_path}")
            with open(control_path) as f:
                mapping = yaml.safe_load(f)
                if not isinstance(mapping, dict):
                    raise ValueError(f"Control file {control_path} must contain a dictionary at the top level.")
                self._cached_control = mapping
        return self._cached_control

    @property
    def assembly(self) -> assembly_mod.AssemblySpec | None:
        """Lazy view over robots/<name>/assembly.yaml, or ``None`` for robots that don't
        declare one (they resolve directly from model_params.yaml instead)."""
        if not self._assembly_loaded:
            path = self.path / 'assembly.yaml'
            if path.is_file():
                with open(path) as f:
                    data = yaml.safe_load(f)
                self._cached_assembly = assembly_mod.AssemblySpec.parse(data if isinstance(data, dict) else {})
            else:
                self._cached_assembly = None
            self._assembly_loaded = True
        return self._cached_assembly

    def _resolved(self, parts: dict[str, list[assembly_mod.RequestPart]]) -> assembly_mod.ResolvedAssembly | None:
        """Shared resolve step for :meth:`effective_sensors`/:meth:`effective_caps`.
        ``None`` iff the robot has no assembly.yaml."""
        spec = self.assembly
        if spec is None:
            return None
        return assembly_mod.resolve(spec, parts)

    def effective_sensors(self, parts: dict[str, list[assembly_mod.RequestPart]], *, frames: dict[str, str] | None = None) -> list[SensorSpec]:
        """The robot's full effective ``SensorSpec`` list for a parts request (``{}``
        for pure defaults). Robots with no assembly.yaml (``assembly`` is ``None``)
        return ``model_params.sensors`` unchanged. Robots with an assembly.yaml
        resolve ``parts`` against the declared assembly and render it through the
        shared component catalog. No per-request caching: ``RobotView`` is a
        per-model singleton, but resolution is per-request.

        Consistency guard: warns on stderr, once per call, if ``model_params.sensors``
        declares a name absent from the assembly's own DEFAULTS render, i.e. the
        assembly's defaults must render every sensor model_params.yaml declares."""
        resolved = self._resolved(parts)
        if resolved is None:
            return self.model_params.sensors
        assert self.assembly is not None
        resolved = assembly_mod.apply_frame_overrides(resolved, frames or {})
        prefix = self.assembly.prefix
        rendered = render_effective_sensors(resolved, _CATALOG, prefix=prefix)
        defaults_rendered = rendered if not parts else render_effective_sensors(assembly_mod.resolve(self.assembly, {}), _CATALOG, prefix=prefix)
        missing = {s.name for s in self.model_params.sensors} - {s.name for s in defaults_rendered}
        if missing:
            print(
                f"assembly.yaml incomplete? model_params declares sensors absent from assembly defaults: {sorted(missing)}",
                file=sys.stderr,
            )
        return rendered

    def effective_caps(self, parts: dict[str, list[assembly_mod.RequestPart]], *, frames: dict[str, str] | None = None) -> RobotCaps:
        """The robot's effective :class:`RobotCaps` for a parts request (``{}`` for
        pure defaults), mirroring :meth:`effective_sensors`. Robots with no
        assembly.yaml return ``self.caps`` unchanged; robots with an assembly.yaml get
        a ``RobotCaps`` with ``resolved``/``catalog`` set so `.available`/instances see
        placement-derived caps."""
        resolved = self._resolved(parts)
        if resolved is None:
            return self.caps
        assert self.assembly is not None
        resolved = assembly_mod.apply_frame_overrides(resolved, frames or {})
        return RobotCaps(self.caps.caps_dir, resolved=resolved, catalog=_CATALOG, prefix=self.assembly.prefix)

    def power_profile(self, parts: dict[str, list[assembly_mod.RequestPart]]) -> dict[str, float | int] | None:
        """Power-publisher parameters, chassis block plus resolved component draw. ``None`` when unpowered."""
        telemetry_yaml = self.path / 'telemetry' / 'power.yaml'
        has_telemetry = telemetry_yaml.is_file()
        assembly_power = self.assembly.power if self.assembly is not None else None
        has_assembly_power = assembly_power is not None

        if not has_telemetry and not has_assembly_power:
            return None

        static_w = 0.0
        efficiency = 1.0
        heating_ch = 0.0
        battery_wh = 0.0
        damping = 0.0
        friction = 0.0
        crr = 0.0
        mass_kg = 0.0
        wheel_r = 0.0
        num_wheels = 1
        odom_linear_w = 0.0
        odom_angular_w = 0.0

        # 1. Primary telemetry configuration from telemetry/power.yaml
        if has_telemetry:
            try:
                with open(telemetry_yaml) as f:
                    t_cfg = yaml.safe_load(f)
                if isinstance(t_cfg, dict):
                    if "power_system" in t_cfg and isinstance(t_cfg["power_system"], dict):
                        ps = t_cfg["power_system"]
                        efficiency = float(ps.get("global_drivetrain_efficiency", ps.get("drivetrain_efficiency", 1.0)))
                        heating_ch = float(ps.get("heating_coefficient_ch", 0.0))
                        battery_wh = float(ps.get("battery_capacity_wh", 0.0))
                        damping = float(ps.get("drivetrain_damping", 0.0))
                        friction = float(ps.get("drivetrain_friction", 0.0))
                        crr = float(ps.get("rolling_resistance_crr", 0.0))
                        mass_kg = float(ps.get("robot_mass_kg", 0.0))
                        wheel_r = float(ps.get("wheel_radius_m", 0.0))
                        num_wheels = int(ps.get("num_wheels", 1))
                    if "static_power_w" in t_cfg:
                        sp = t_cfg["static_power_w"]
                        if isinstance(sp, dict):
                            static_w = float(sp.get("compute_core", 0.0)) + float(sp.get("idle_motors", 0.0))
                        elif isinstance(sp, (int, float)):
                            static_w = float(sp)
                    if "odometry_drive_power" in t_cfg and isinstance(t_cfg["odometry_drive_power"], dict):
                        od = t_cfg["odometry_drive_power"]
                        odom_linear_w = float(od.get("linear_w_per_mps", 0.0))
                        odom_angular_w = float(od.get("angular_w_per_radps", 0.0))
            except Exception:
                pass

        # 2. Backward-compatible fallback / override from assembly.yaml power:
        if assembly_power is not None:
            spec = assembly_power
            if static_w == 0.0:
                static_w = spec.compute_core_w + spec.idle_motors_w
            if efficiency == 1.0 and spec.drivetrain_efficiency != 1.0:
                efficiency = spec.drivetrain_efficiency
            if heating_ch == 0.0:
                heating_ch = spec.heating_coefficient_ch
            if battery_wh == 0.0:
                battery_wh = spec.battery_capacity_wh
            if damping == 0.0:
                damping = spec.drivetrain_damping
            if friction == 0.0:
                friction = spec.drivetrain_friction
            if crr == 0.0:
                crr = spec.rolling_resistance_crr
            if mass_kg == 0.0:
                mass_kg = spec.robot_mass_kg
            if wheel_r == 0.0:
                wheel_r = spec.wheel_radius_m
            if num_wheels <= 1 and spec.num_wheels > 1:
                num_wheels = spec.num_wheels

        # 3. Add dynamic component static power draw from catalog (e.g. mounted sensors)
        resolved = self._resolved(parts)
        if resolved is not None:
            for placement in resolved.placements:
                component = _CATALOG.get(placement.type, placement.variant)
                draw = component.power.get('static_power_w', 0.0)
                if not isinstance(draw, (int, float)):
                    raise ValueError(f"component {placement.type}/{placement.variant}: 'power.static_power_w' must be a number; got {type(draw).__name__}")
                static_w += float(draw)

        # 4. Fallback for mass from model_params
        if mass_kg == 0.0 and "mass" in self.model_params:
            mass_kg = float(self.model_params.get("mass", {}).get("base_kg", 0.0))

        return {
            'static_power_w': static_w,
            'drivetrain_efficiency': efficiency,
            'heating_coefficient_ch': heating_ch,
            'battery_capacity_wh': battery_wh,
            'drivetrain_damping': damping,
            'drivetrain_friction': friction,
            'rolling_resistance_crr': crr,
            'robot_mass_kg': mass_kg,
            'wheel_radius_m': wheel_r,
            'num_wheels': num_wheels,
            'odometry_linear_w_per_mps': odom_linear_w,
            'odometry_angular_w_per_radps': odom_angular_w,
        }

    @property
    def model(self) -> ModelWrapper:
        return ModelWrapper(
            self.name,
            {
                **ModelProvider_URDF.asdict(self.path, self.name),
                **ModelProvider_USD.asdict(self.path, self.name),
            },
        )


@attrs.define(eq=False, hash=False)
class RobotIdentifier(Identifier[RobotView]):
    def load(self, path: Path, /, **kwargs: object) -> RobotView:
        del kwargs  # unused
        return RobotView(path)


RobotIdentifier.use(SimplePathResolver(RobotIdentifier, get_package_share_path('arena_robots') / 'robots'))
