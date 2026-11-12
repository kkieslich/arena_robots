# Adding a robot

A robot is a directory under `arena_robots/arena_robots/robots/<name>/`. The
directory name is the robot's canonical identifier: task_generator, launch
files, and the `arena feature robots` CLI all look robots up by this name.

## Required files

```
<robot>/
├── model_params.yaml    # robot-wide identity (robot_model, base_frame, z_offset, sensors)
├── control.yaml         # ros2_control configuration
├── mappings.yaml        # sim ⇄ ROS2 topic bridge
└── caps/
    └── mobile.yaml      # required: every robot has a mobile base
```

### `model_params.yaml`

Robot-wide identity. Cap data does not live here except for fields that apply
to any robot regardless of cap: `base_frame` (every robot has a base link),
`z_offset` (height of the model root when the robot rests on the ground, defaults to 0.0), `sensors` (hardware
sensor declarations, defaults to `[]`) and `lights` (lights the robot carries, defaults to `[]`).

```yaml
robot_model: my_robot       # required; matches the directory name
base_frame: base_link       # required; TF frame of the robot's base link
z_offset: 0.0635            # optional; metres from the ground to the model root with the robot at rest, spawn and teleport place it there
sensors:                    # optional; declared sensors parsed into SensorSpec entries
  - {name: lidar, type: laserscan, topic: ${namespace}/scan, frame: base_scan}
lights:                     # optional; Light entries anchored on a frame of the robot
  - {name: headlight, fixture: spot, frame: base_link, offset: {x: 0.2, y: 0.0, z: 0.3, pitch: -1.42}, lumens: 1500}
```

Each `lights` entry takes the world `Light` schema with `frame` as its only anchor
([AUTHORING.md](../../../arena_simulation_setup/AUTHORING.md#lights)). The task generator
spawns it with the robot as `<robot>_<name>` on the robot's own frame, so it follows the
robot, is a `light` semantic entity (`arena env semantics <robot>_<name> lit=false`) and
leaves the world's default lighting on.

Parsed by [`arena_robots.Robot.ModelParams`](../arena_robots/Robot.py). Additional keys pass
through unchanged; nothing in-tree consumes them.

`sensors` also drives the Nav2 costmap sources ([`arena_robots.nav2`](../arena_robots/nav2.py)).
The local costmap raytraces planar lidars in a 2D obstacle layer and runs multi-ring lidars
and depth cameras through a spatio-temporal voxel layer, which clears inside a frustum fitted
to the sensor's field of view. Field of view and range are read from the gz `<sensor>` element
of the rendered URDF that the entry's `sensor:` key names, or its `name` when the key is absent
([`arena_robots.sensor_geometry`](../arena_robots/sensor_geometry.py)). A sensor without such
an element counts as planar when it only declares a `laserscan`. Otherwise it marks without
clearing and its obstacles expire after the layer's `voxel_decay`.

Every gz `<sensor>` sits on its `<gazebo reference>` link with an identity `<pose>`, and the
`frame` of each `sensors:` entry names that link. A camera's reference link also needs a REP-103
optical child link at zero offset, named `<reference>_optical_frame` with a trailing `_frame` of
the reference dropped. Isaac stamps image,
depth and point cloud messages in the optical frame, Gazebo and MuJoCo in the reference link,
and the voxel layer poses a camera's clearing frustum in the reference link.
`tests/test_sensor_geometry.py` checks this for every robot and camera component.

### `caps/`: capability declarations

Each `caps/<cap>.yaml` declares one capability the robot physically has. **File
presence IS the advertisement**: `caps/arm.yaml` existing means this robot
advertises `arm`. Derived at load time as `robot_view.caps.available`, a
`frozenset[str]` of cap stems.

Adapters declare `requires: frozenset[str]` via `@BringupMeta.attach(requires=frozenset({...}))` on the `Bringup` subclass; the broker gates binding on `adapter.requires ⊆ robot.caps.available`. A cap must have a documented vocabulary entry before any adapter cites it in `requires`.

#### Active cap vocabulary

| Cap | File | Shape | Meaning |
|---|---|---|---|
| `mobile` | `caps/mobile.yaml` | **flat** (singleton) | drivable chassis (wheeled / tracked / omni) |
| `arm` | `caps/arm.yaml` | **dict** (named instances) | serial manipulator with an IK-solvable tip |
| `lift` | `caps/lift.yaml` | **dict** (named instances) | prismatic vertical positioner (Ewellix column, torso lift, scissor), commanded by the `direct` lift adapter (`joint`, `controller`, optional `limits`) |
| `gripper` | `caps/gripper.yaml` | **dict** (named instances) | end effector with one actuated joint (`joint`, `controller`, `positions`, `max_effort`, optional `arm`), commanded by the `direct` gripper adapter |

Reserved for future: `ptu_head`, `dual_arm`. Continuum arms either
discretize into the serial-chain shape or use a future `kind: continuum`
variant of `caps/arm.yaml`.

#### `caps/mobile.yaml`: flat, singleton

A robot has exactly one mobile base by construction. All robot-physical
primitives live at top level; adapter sub-blocks are reserved for
adapter-specific wiring only. `actions` and `laser` are top-level fields. The
optional `nav2:` sub-block carries nav2-only wiring such as
`global_observation` and `local_observation` (the cloud sources' `min_obstacle_height` and `max_obstacle_height`, 0.1 and 0.8 m by default). The global planner's settings come from the selected
planner config.

```yaml
odom_frame: odom
sensor_frame: base_scan
radius: 0.5
is_holonomic: false
climbs: [stairs]              # optional, default []

footprint: [[0.5, 0.35], [0.5, -0.35], [-0.5, -0.35], [-0.5, 0.35]]
footprint_padding: 0.1        # optional
inflation_radius: 0.9         # optional, default: outer radius of the padded footprint or radius, + 0.4
polygons_dict:
  StopPolygon: {type: polygon, points: [...], action_type: stop}

velocity_limits:
  linear: {min: -2.0, max: 2.0}
  angular: {min: -4.0, max: 4.0}
acceleration_limits:
  linear: 3.0
  angular: 3.5

actions:
  continuous: {linear: {min: -2.0, max: 2.0}, angular: {min: -4.0, max: 4.0}}
  discrete: [{name: move_forward, linear: 0.3, angular: 0.0}, ...]

laser: {angle: {min: -2.35619, max: 2.35619}, num_beams: 720, range: 30.0, update_rate: 10}

planar:
  velocity: {linear: [-2.0, 2.0], lateral: [0.0, 0.0], angular: [-4.0, 4.0]}
  acceleration: {linear: 20.0, lateral: .inf, angular: 25.0}
  gain: {linear: 1.0, lateral: 1.0, angular: 1.36}
  lag: 0.0
  traction: 7.85
  pivot: 0.131
  timeout: 0.25

nav2:                      # optional, nav2-only wiring
  global_observation: {...}

meshnav:                   # optional, meshnav-only wiring
  max_step: 0.2            # default 0.2 when climbs lists stairs, else 0.1
  max_slope: 1.3           # rad, default 1.3 when climbs lists stairs, else 0.35
  inscribed_radius: 0.21   # default inscribed circle of the padded footprint, else radius
  inflation_radius: 0.71   # default inscribed_radius + 0.5
  obstacles: true          # default true when the robot has a laserscan or pointcloud sensor
  robot_height: 0.7        # m, returns higher above the walking surface are ignored
  min_obstacle_height: 0.1 # m, returns lower above the walking surface are ground
  obstacle_range: 11.0     # m from the sensor, default caps laser.range minus 1, else 9
  obstacle_decay: 1.0      # s a sensor's obstacles outlive its scan
  planner_frequency: 1.0   # Hz, MBF replans at it while a goal runs
  planner: dijkstra        # or cvp, MBF's fast marching planner
  resolution: 0.1          # grid spacing of the nav mesh
  params: {}               # merged over config/meshnav/meshnav.yaml
```

`velocity_limits` and `acceleration_limits` flow into nav2 controller configs
via `Nav2KinematicsDerivedYAML` substitution; controller YAMLs reference them
as `${max_linear_vel:-default}` etc. To override a specific controller without
touching `mobile.yaml`, replace the substitution with a literal value in that
controller YAML.

`planar` is the body response `sim:=flatsim` drives the robot with, the same keys
for every robot: per-axis velocity bounds, acceleration and gain, a first-order
`lag`, a `traction` cap on planar acceleration, the `pivot` x the body turns
about and the command `timeout`. It is the drive's envelope, not the planner's,
so it can exceed `velocity_limits`.

`climbs` lists the world stair kinds (`stairs`, `escalator`) the base can
drive through. Cross-level routing sends the robot through the boarding half of
a stair only when its kind is listed, and an explicit `via` naming a stair the
robot does not climb is an error. `go1` and `a1` climb stairs.

`meshnav:` sets the traversal limits of the meshnav adapter: the largest height
step and the steepest slope the base crosses, and the inflation around lethal
vertices. The obstacle layer reads `meshnav/obstacles`, published by the
`meshnav_obstacles` node from every range sensor (a sensor's point cloud when it
has one, else its scan) in the odom frame. A return is an obstacle when it stands
`min_obstacle_height` to `robot_height` above the highest mesh surface below it
and no mesh surface around it lies between `min_obstacle_height` under and
`max_step` over it, so floors, ramps, stair treads, risers and escalator belts are
ground on every level. Returns beyond `obstacle_range` or within the footprint's
outer radius are dropped. Velocity limits cap the mesh controller. Goals go to
MBF's Dijkstra mesh planner, `planner: cvp` picks the fast marching planner, whose
path back-tracking fails across some stair terraces.

#### `caps/arm.yaml`, `caps/lift.yaml`, `caps/gripper.yaml`: dict-keyed

Always a dict of named instances. Single-arm robots use a dict with one entry
(by convention named `arm`); dual-arm uses two entries (e.g. `left`, `right`).
No single-instance shorthand, uniform shape keeps the loader trivial.

```yaml
# caps/arm.yaml (single-arm)
arm:
  base_link: chassis_link
  tip_link: arm_tool0
  chain: [arm_shoulder_pan_joint, arm_shoulder_lift_joint, arm_elbow_joint,
          arm_wrist_1_joint, arm_wrist_2_joint, arm_wrist_3_joint]
  controller: arm_controller
  moveit:                  # adapter-specific; consumed by the moveit adapter
    package: ur_moveit_config
    args: {ur_type: ur5e, safety_limits: true}
```

The `moveit:` block declares what `moveit_factory.build_moveit_params` needs and assumes no
vendor. `planning_group` names the group the task handlers plan for; `srdf` and `joint_limits`
(each `{package, path}`) default into `package`; `args` are the xacro mappings of URDF and SRDF
(`prefix`/`mount`/`parent` also render the arm component's SRDF and joint limits per
placement); `robot_name` is the builder name (default: `package` minus `_moveit_config`, else the
robot). `package` is optional: a block with `srdf` and no `package` takes Arena-owned defaults
(OMPL with its stock planner configs). Every build, packaged or not, sets a KDL solver on the
planning group unless the package's kinematics file already names it, and maps the arm's
`controller` (FollowJointTrajectory over the `chain` joints) as the only MoveIt controller.
`tiago` declares its own `srdf/arm.srdf.xacro` and `srdf/joint_limits.yaml` this way; the `fr3`
component does the same under `components/arm/fr3`.

The pipelines of a block without `package` are OMPL (the default) and Pilz, which `motion: linear`
reaches use. `moveit.cartesian_limits` sets the Pilz limits per arm over the defaults
`max_trans_vel: 0.5`, `max_trans_acc: 1.0`, `max_trans_dec: -1.0`, `max_rot_vel: 1.57`. Four
top-level arm keys tune reaches and scene sync, each read with a default:

| Key | Default | Meaning |
|---|---|---|
| `stall_error` | 0.2 | Tracking error in rad above which the arm counts as blocked |
| `stall_speed` | 0.05 | Joint speed in rad/s below which that joint counts as stopped, so an arm that lags while moving is not blocked |
| `stall_time` | 0.5 | Seconds the error must stay above `stall_error` on a stopped joint before the reach fails with `arm blocked` |
| `scene_radius` | 1.5 | Radius in m around the base within which episode objects enter the planning scene |

For platforms that ship an upstream SRDF, structural primitives can be derived
from it; `controller:` and adapter sub-blocks stay authored:

```yaml
arm:
  srdf: $(find ur_moveit_config)/srdf/ur.srdf.xacro
  controller: arm_controller
  moveit: {package: ur_moveit_config, args: {ur_type: ur5e}}
```

Loader precedence per field: explicit > SRDF-derived > raise. No URDF-alone
derivation (structural-without-semantics is ambiguous).

`caps/lift.yaml`:

```yaml
lift:
  joint: robot_lift_ewellix_lift_top_joint
  controller: lift_controller
  limits: [0.0, 0.9]       # joint range the lift adapter clamps to, optional
```

`caps/gripper.yaml` (when a platform gets one) uses a back-reference to its
arm:

```yaml
gripper:
  arm: arm                 # key into caps/arm.yaml
  joint: arm_gripper_joint
  controller: arm_gripper_controller
  positions: {open: 0.0, close: 0.8}
  max_effort: 50.0
```

`positions` holds the joint values of `open` and `close`, `max_effort` the effort every gripper
command carries. Gripper components carry the same block under `caps:`.

#### Adapter sub-blocks are namespaced, not dispatch

`moveit:`, `nav2:`, `rl:`, `drl_grasp:`, ... inside a cap file are pure data
organization. **No in-file handler declaration**: adapters are selected at
runtime (CLI / setup YAML) and are handed the cap YAML, reading whichever
primitives + adapter-sub-block they know about.

### `control.yaml`

ROS 2 `controller_manager` configuration. Define `joint_state_broadcaster`
plus whichever controller drives the robot (e.g.
`diff_drive_controller/DiffDriveController` for wheeled, `JointTrajectoryController`
for an arm). See [`husky/control.yaml`](husky/control.yaml) for a diff-drive
example and [`rbkairos_plus/control.yaml`](rbkairos_plus/control.yaml) for an
arm-extended example.

An optional sibling `control.<sim>.yaml` (`sim` a `SimSimulator` value such as
`gazebo`, `isaac` or `mujoco`) overlays `control.yaml` key by key in that
simulator only, for a value the simulator's physics needs and the real robot
does not. [`jackal/control.gazebo.yaml`](jackal/control.gazebo.yaml) sets the
`wheel_separation_multiplier` Gazebo's skid steer turns at.

Every simulator spawns a joint at the `initial_value` of its ros2_control
position state interface, so a legged robot's `urdf/base_hw/<name>.ros2_control.urdf`
carries its standing pose there (see [`go1`](go1/urdf/base_hw/go1.ros2_control.urdf)).

### `mappings.yaml`

Simulator ⇄ ROS2 topic bridge declarations, as a JSON array. Each entry:

```yaml
{
  "gz_topic":  "/model/{robot_name}/cmd_vel",    # simulator-side topic
  "ros_topic": "cmd_vel",                         # ROS2-side topic
  "gz_type":   "gz.msgs.Twist",
  "ros_type":  "geometry_msgs/msg/Twist",
  "direction": "]",                               # "[" sim→ros, "]" ros→sim
}
```

`{robot_name}` and `{world}` are substituted at runtime. See
[`husky/mappings.yaml`](husky/mappings.yaml).

Hand-authored rows carry only what the catalog cannot derive: `pose`, ros2_control
plumbing (`odometry`/`cmd_vel`/`joint_states`), and `contact` sensors. Bridge rows for
catalog-derivable sensor types (laserscan, pointcloud, imu, image, depth, camera_info)
are generated at runtime from `effective_sensors` and deduped against this file, so
never duplicate them here.

### `assembly.yaml`: component-catalog robots only

Sibling of `model_params.yaml` (deliberately not inside it: the existing scalar
`priority:` field would collide). Its presence declares the robot's sensors are sourced
from the component catalog: sensors come from `components/` via mounts, and the
chassis xacro carries no sensor invocations.

```yaml
prefix: ""            # frame-prefix convention; omit for the Robotnik "robot_" default
mounts:
  front_laser:        # mount names adopt the chassis xacro's own frame names
    parent: chassis_link
    xyz: [0.53, 0.33, 0.1145]     # verbatim from the chassis xacro's sensor invocation
    rpy: [3.141592653589793, 0, 0.7853981633974483]
    accepts: [lidar]              # a set: a mount may accept more than one part type
defaults:
  lidar:
    - variant: sick_s300
      mount: front_laser
      overrides: {name: lidar_rear, topic: scan/rear}   # per-instance tuning lives HERE
```

`accepts` is a set, not a scalar: a versatile mounting point (e.g. a top plate) may
accept several part types. When more than one mount accepts the same type, allocation
preference among them follows mount DECLARATION ORDER in this file, first-declared
wins. There is no separate `priority:` field; reorder the `mounts:` block itself to
change preference.

Requests (`robot:=name[lidar=x,...]`) resolve against this file with replace-on-touch
semantics; see the grammar section in
`arena_runtime/arena_runtime/arena_runtime/env/README.md`.

A sensor that is physically present but deliberately left unbridged (e.g. a GPS/navsat
mast) stays inline in the robot's own xacro; it gets no mount or component entry here.

### Moving a robot's sensors to the component catalog

1. **Golden capture first**: render the robot's current URDF via xacro in-container and
   save to `tests/golden/<name>_default.urdf` (strip the container wrapper's non-XML
   stdout prefix; verify it parses).
2. Author/reuse `components/` entries (see [components/README.md](../components/README.md)).
3. Write `assembly.yaml` with mount origins copied verbatim from the xacro invocations
   (verify each sensor's actual parent link; robots differ).
4. Strip ONLY the sensor invocations + now-unused includes from the chassis xacro,
   leaving no comment behind; never delete macro files, other robots may include
   them. Ensure
   the wrapper-contract args exist (`namespace`, `prefix`, `gazebo_classic`,
   `gazebo_ignition`) with defaults preserving the chassis's existing values.
5. `tests/test_default_morphology.py` parametrizes over every `tests/golden/*_default.urdf`
   automatically, so the robot is now covered: effective-sensors consistency
   (`effective_sensors({}) == model_params.sensors`) and canonicalized URDF
   structural comparison against the golden from step 1.
6. The golden reproduces known drift byte-identically; this move is not a bug-fix
   opportunity. Dead (env-gated Gazebo-Classic) blocks stay untouched.

## Optional files

| Path | Purpose |
| --- | --- |
| `urdf/<name>.urdf.xacro` | robot description; mesh refs use `package://arena_robots/robots/<name>/meshes/…` or a fixed upstream package (`package://jackal_description/…`) |
| `meshes/` | STL/DAE/OBJ files, normally a git submodule under `github.com/arena-robots/<name>` (opt-in via `arena feature robots add <name>`) |
| `launch/` | robot-specific launch files (e.g. nav2 overrides) |
| `README.md` | free-form robot docs (upstream references, env vars, etc.) |

## Meshes

Every robot that needs per-robot geometry has a `meshes/` git submodule pinned
via `.gitmodules`, pointing at `github.com/arena-robots/<name>.git` with
`update = none`. Running `arena feature robots add <name>` clones it; config
edits you make under the robot dir stay in the main Arena repo. A robot whose
upstream repo is not a buildable ROS 2 package (p3at, tiago) instead points its
submodules under `meshes/` at that upstream repo with `sparse =` on the mesh
directory, and its URDF references `package://arena_robots/robots/<name>/meshes/<path in the upstream repo>`.
Robots that use upstream geometry (jackal, turtlebot) have no `meshes/` submodule,
their URDFs reference `package://jackal_description/…` etc., supplied by
`deps/jackal` or `deps/turtlebot4`.

## Checklist

1. `mkdir -p arena_robots/arena_robots/robots/<name>/caps`
2. Write `model_params.yaml` (minimal identity), `control.yaml`, `mappings.yaml`.
3. Write `caps/mobile.yaml` (every robot has a mobile base). Add `caps/arm.yaml`
   and/or `caps/lift.yaml` if the robot has those subsystems.
4. (Optional) add `urdf/<name>.urdf.xacro` and/or a `meshes/` submodule.
   For component-catalog sensors, add `assembly.yaml` + a golden (see "Moving a
   robot's sensors to the component catalog" above).
5. If the robot ships an upstream ROS package, add it as a submodule with
   `robot = <name>` (and `update = none`) in `.gitmodules`.
6. `arena feature robots add <name>` to fetch any submodules.
7. `arena feature robots check` to verify every `package://arena_robots/…`
   URI resolves on disk.
8. `arena launch … robot:=<name>` to bring it up.

## Adding a new cap

When a new subsystem kind (e.g. `gripper`, `ptu_head`) becomes needed:

1. Add an entry to the Active cap vocabulary table above with its file, shape,
   meaning, and list of robots that have it.
2. Define the primitive fields (what every adapter-agnostic consumer can count
   on) and document adapter sub-block conventions if any exist.
3. If adding a typed accessor is warranted, extend
   [`caps.py`](../arena_robots/caps.py) with a `<Cap>Spec` subclass. The loader dict-keyed
   pattern is uniform, see `ArmSpec`/`LiftSpec` for the template.
