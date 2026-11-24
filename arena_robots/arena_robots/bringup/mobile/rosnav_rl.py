from __future__ import annotations

from typing import ClassVar

from launch import Action
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from arena_robots.bringup import Bringup, BringupMeta
from arena_robots.task_kinds import TaskKind


def _load_goto_pose_rosnav_rl() -> type:
    from arena_robots.task_server_handlers.goto_pose._passthrough import GotoPoseHandlerRosnavRl

    return GotoPoseHandlerRosnavRl


@BringupMeta.attach(requires={"mobile"}, cap="mobile")
class RosnavRlBringup(Bringup):
    kind = "rosnav_rl"
    task_handlers: ClassVar[dict] = {TaskKind.GOTO_POSE: _load_goto_pose_rosnav_rl}

    @property
    def _cfg(self) -> dict:
        return self.robot.caps.mobile.sub("rosnav_rl")

    @property
    def goal_topic(self) -> str:
        return self.namespace("goal_pose")

    @property
    def cmd_vel_topic(self) -> str:
        return self.namespace("cmd_vel")

    @property
    def inference_node_name(self) -> str:
        return "rosnav_rl_inference"

    def _launch_actions(
        self,
        *,
        use_sim_time: bool = True,
        frame: str = "",
        arena_env_node: str = "",
        train_mode: bool = False,
        agent: str = "",
        control_rate: float | None = None,
        min_lookahead_dist: float | None = None,
        max_lookahead_dist: float | None = None,
        lookahead_time: float | None = None,
        **_: object,
    ) -> list[Action]:
        cfg = self._cfg
        agent_name = agent or str(cfg.get("agent", ""))
        if not agent_name and not train_mode:
            raise ValueError(f"rosnav_rl bringup for '{self.robot.name}' missing required 'agent': set caps/mobile.yaml 'rosnav_rl.agent' or pass robot.mobile.agent:=<name>")
        rate = float(cfg.get("control_rate", 0.0)) if control_rate is None else float(control_rate)
        lo = float(cfg.get("min_lookahead_dist", 0.5)) if min_lookahead_dist is None else float(min_lookahead_dist)
        hi = float(cfg.get("max_lookahead_dist", 2.5)) if max_lookahead_dist is None else float(max_lookahead_dist)
        lt = float(cfg.get("lookahead_time", 1.5)) if lookahead_time is None else float(lookahead_time)

        launch_file = PathJoinSubstitution(
            [
                FindPackageShare("arena_robots"),
                "launch",
                "adapters",
                "mobile",
                "rosnav_rl.launch.py",
            ]
        )
        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(launch_file),
                launch_arguments={
                    "agent": agent_name,
                    "base_frame": self.robot.model_params.base_frame,
                    "control_rate": str(rate),
                    "frame": frame,
                    "lookahead_time": str(lt),
                    "max_lookahead_dist": str(hi),
                    "min_lookahead_dist": str(lo),
                    "namespace": self.namespace,
                    "node_name": self.inference_node_name,
                    "robot": self.robot.name,
                    "arena_env_node": arena_env_node,
                    "train_mode": str(train_mode).lower(),
                    "use_sim_time": str(use_sim_time).lower(),
                }.items(),
            )
        ]
