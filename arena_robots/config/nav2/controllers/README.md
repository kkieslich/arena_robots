# Nav2 controllers

One directory per selectable local planner. Pick it at launch with
`robot.mobile.local_planner:=<name>` (default comes from the robot's `caps/mobile.yaml`
`nav2:` block).

## Layout

```
controllers/<name>/
  controller_config.yaml   # required
  controller.launch.py     # optional side-car, see below
```

### `controller_config.yaml` (required)

Holds the `controller_plugins` / `controller_plugins_dict` block for this planner.
It is merged over `defaults/controller_config.yaml` by the nav2 mobile adapter
([nav2.launch.py](../../../launch/adapters/mobile/nav2.launch.py)), so only specify
what differs from the defaults. Two common shapes:

- **Native plugin** (`mppi`, `dwb`, `hateb`, ...): `plugin:` is a `nav2_core::Controller`
  loaded in-process by `controller_server`.
- **Bridge plugin** (`sicnav`, `crowdnav`, ...): `plugin:` is a `nav2py_*` shim that
  forwards to an external policy process.

Robot kinematics, frames, and namespace reach the config through `${...}` substitutions
the adapter emits (see `Nav2KinematicsDerivedYAML` / `Nav2CollisionDerivedYAML`):
velocity and acceleration limits, `${motion_model}` (`Omni`, `Ackermann` or `DiffDrive`)
and, for a car-like robot only, `${turning_radius}`.

### `controller.launch.py` (optional side-car)

If a controller needs extra nodes alongside `controller_server` (a message bridge, a
prediction node, ...), drop a `controller.launch.py` next to the config. The adapter
includes it automatically when the file exists, passing `namespace`, `env_namespace`,
`frame`, `base_frame`, `use_sim_time`, and `agent`. Controllers without one launch nothing extra.

Worked example: [hateb/controller.launch.py](hateb/controller.launch.py) starts the
`cohan_peds_bridge`, which republishes `arena_peds` as `cohan_msgs/TrackedAgents` on the
per-env frame the HATEB planner expects. [rosnav_rl/controller.launch.py](rosnav_rl/controller.launch.py)
starts the rosnav_rl `action_server.py` serving `get_command` for the agent named by
`robot.mobile.agent:=<name>` (or `rosnav_rl.agent` in `caps/mobile.yaml`).
