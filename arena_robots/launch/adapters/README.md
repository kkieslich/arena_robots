# Bringup adapter launch files

This directory contains per-bringup-kind launch files. They are implementation
details of `arena_robots.bringup.*` classes, not public entry points.

The public entry point is [`launch/bringup.launch.py`](../bringup.launch.py).
Users invoke that file; it selects and includes the appropriate file here based
on the `bringup` argument.

## Files

### `nav2.launch.py`

Internals of `arena_robots.bringup.nav2.Nav2Bringup`. Instantiates a full nav2
stack (map server, AMCL, planner, controller, bt_navigator, lifecycle manager)
in the given namespace. Accepts launch arguments for planner selection
(`global_planner`, `local_planner`, `inter_planner`), sim time, costmap frame,
and an optional `arena_env_node` name for map topic remapping (empty
string = no remap, which is the standalone default). The companion
`env_namespace` arg (the env-root namespace, e.g. `/arena/env_0`) is
plumbed through identically and is used for env-intrinsic topics that do
not live under `arena_env_node`, such as `door_mask` for
`nav2_mask_overlay_layer`. Planner selection is
typically driven from
[cap-scoped overrides](../../../../arena_bringup/BRINGUP.md#cap-scoped-overrides)
at launch (`robot.mobile.local_planner:=teb`, etc.) which the task-generator
forwards to this launch file. `params_overlay` names a YAML file (a bare name
resolves to `config/nav2/overlays/<name>.yaml`) merged over `nav2.yaml`. Its
top-level `substitutions:` mapping is instead merged over the planner configs,
where `${...}` values such as `controller_plugins_dict` come from.

`mapping` picks the global costmap's map (`robot.mobile.mapping:=slam`).
`static` (default) reads the env's map server under `arena_env_node`. `slam` starts
`slam_toolbox` with [`config/nav2/slam.yaml`](../../config/nav2/slam.yaml) on the
robot's first laserscan sensor and reads its `<namespace>/map`, so the map grows
while the robot drives. It only maps: poses come from the simulator's `map -> base`
(so the map stays in the env's frame on every level of a stacked world), it matches
no scans and publishes no TF, and the simulator keeps `map -> odom`. Nav2's lifecycle
manager brings `slam_toolbox` up before the costmaps, and its absolute `/map` and
`/map_metadata` are remapped into the robot's namespace.

### `none.launch.py`

Internals of `arena_robots.bringup.none.NoneBringup`. Spins up no navigation
stack. The `task_server` will publish goals directly to a goal-pose topic.
Use this for robots driven by an external planner that subscribes to a goal
topic.

## Adding a new bringup kind

Create `<kind>.launch.py` here, then add a `Bringup` subclass in
`arena_robots/arena_robots/arena_robots/bringup/<cap>/<kind>.py` whose
`_launch_actions()` includes this file. Implement handlers under
`arena_robots/arena_robots/arena_robots/task_server_handlers/<task_kind>/`
and declare them on the `Bringup` subclass via a `task_handlers: ClassVar`
mapping `TaskKind` to a zero-arg loader function. The `task_server` reads
that mapping directly, no central registry to update.
