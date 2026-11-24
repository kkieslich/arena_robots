from __future__ import annotations

import asyncio
import json
import math
import threading
import time

import pytest
import yaml

pytest.importorskip("arena_mcp")
rclpy = pytest.importorskip("rclpy")

from arena_robots_msgs.action import GotoPose, PlayGesture, ReachPose
from arena_robots_msgs.msg import Energy, Power
from arena_runtime_msgs.msg import EnvRecord, EnvRegistry
from diagnostic_msgs.msg import KeyValue
from geometry_msgs.msg import PoseStamped, Twist
from mcp.server.mcpserver import MCPServer
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from arena_runtime_msgs.msg import EpisodeRecord, RobotCap, RobotDescriptor, RobotFleet, RobotState, SemanticEntityState, SemanticSnapshot
from arena_runtime_msgs.srv import SubmitTask

from arena_mcp.core import JOBS, ROS, ToolError, fleet
from arena_mcp.core.jobs import CANCELLED, FAILED, SUCCEEDED
from arena_mcp.core.ros import PEER_MATCH_S
from arena_robots import mcp
from arena_robots.Robot import RobotIdentifier
from arena_robots.Sensor import resolve_topic
from arena_simulation_setup.shared.task import GoToPhase, ReachPhase, TaskRequest

ENV = 7
TG = f"/arena/env_{ENV}/arena_env_node"
JACKAL = f"{TG}/jackal"
KAIROS = f"{TG}/kairos"
UPPER_LEVEL_Y = 50.0


def _sensor_topic(model: str, ns: str, sensor_type: str) -> str:
    view = RobotIdentifier(model).resolve_sync()
    return next(resolve_topic(spec, ns) for spec in view.effective_sensors({}) if spec.type == sensor_type)


LIDAR = _sensor_topic("jackal", JACKAL, "laserscan")
IMU = _sensor_topic("jackal", JACKAL, "imu")
CAMERA = _sensor_topic("rbkairos_plus", KAIROS, "image")


class Peer:
    """A real ROS node standing in for the task generator with its adapters, the task servers and the simulator of one env."""

    def __init__(self) -> None:
        self.node = rclpy.create_node("robots_mcp_peer")
        group = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.twists: list[Twist] = []
        self.kairos_twists: list[tuple[float, Twist]] = []
        self.cancelled: list[float] = []
        self.submissions: list[tuple[str, TaskRequest]] = []
        self.indices: list[list[int]] = []
        self.initialized = True
        self._lock = threading.Lock()
        self._tasks = {name: {"phase": "", "met": [], "failed": [], "dropped": []} for name in ("jackal", "kairos")}
        self._goals: dict[str, object] = {}
        self._next = 0
        self.node.create_publisher(EnvRegistry, "/arena/state/envs", latched).publish(EnvRegistry(envs=[EnvRecord(env_id=ENV, fqn=TG, ready=True)]))
        self.node.create_publisher(RobotFleet, f"{TG}/state/robots", latched).publish(self._fleet())
        trail = Path()
        trail.header.frame_id = "map"
        for x in (0.0, 1.0, 1.0):
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = 0.0 if len(trail.poses) < 2 else 2.0
            trail.poses.append(pose)
        self.node.create_publisher(Path, f"{JACKAL}/trail", latched).publish(trail)
        self.node.create_subscription(Twist, f"{JACKAL}/cmd_vel", self.twists.append, 10, callback_group=group)
        self.node.create_subscription(Twist, f"{KAIROS}/cmd_vel", lambda twist: self.kairos_twists.append((time.monotonic(), twist)), 10, callback_group=group)
        self._task_pose = self.node.create_publisher(PoseStamped, f"{JACKAL}/task_pose", 10)
        self._odom = self.node.create_publisher(Odometry, f"{JACKAL}/odom", 10)
        self._scan = self.node.create_publisher(LaserScan, LIDAR, 10)
        self._image = self.node.create_publisher(Image, CAMERA, 10)
        self._power = self.node.create_publisher(Power, f"{JACKAL}/power_publisher/power", 10)
        self._energy = self.node.create_publisher(Energy, f"{JACKAL}/power_publisher/energy", 10)
        self.node.create_timer(0.05, self._tick, callback_group=group)
        self._servers = [
            ActionServer(self.node, GotoPose, f"{JACKAL}/goto_pose", execute_callback=self._goto, cancel_callback=self._accept, callback_group=group),
            ActionServer(self.node, ReachPose, f"{KAIROS}/reach_pose", execute_callback=self._reach, cancel_callback=self._accept, callback_group=group),
            ActionServer(self.node, PlayGesture, f"{KAIROS}/play_gesture", execute_callback=self._gesture, cancel_callback=self._accept, callback_group=group),
        ]
        self._clients = {
            "goto": ActionClient(self.node, GotoPose, f"{JACKAL}/goto_pose", callback_group=group),
            "reach": ActionClient(self.node, ReachPose, f"{KAIROS}/reach_pose", callback_group=group),
            "gesture": ActionClient(self.node, PlayGesture, f"{KAIROS}/play_gesture", callback_group=group),
        }
        self._semantics = self.node.create_publisher(SemanticSnapshot, f"{TG}/state/semantics", latched)
        self._episode = self.node.create_publisher(EpisodeRecord, f"{TG}/state/episode", latched)
        self.start_episode(1)
        self.node.create_service(SubmitTask, f"{TG}/runtime/submit_task", self._submit, callback_group=group)
        self._publish_tasks()
        self.executor = MultiThreadedExecutor(num_threads=6)
        self.executor.add_node(self.node)
        self._thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._thread.start()

    @staticmethod
    def _accept(_handle: object) -> CancelResponse:
        return CancelResponse.ACCEPT

    @staticmethod
    def _fleet() -> RobotFleet:
        jackal = RobotState(
            descriptor=RobotDescriptor(name="jackal", model="jackal", ns=JACKAL, frame=f"env_{ENV}/jackal", base_frame=f"env_{ENV}/jackal/base_link"),
            caps=[RobotCap(cap="mobile", adapter="nav2")],
            params=[KeyValue(key="lidar@front", value="jackal_default"), KeyValue(key="imu@imu", value="jackal_default")],
        )
        kairos = RobotState(
            descriptor=RobotDescriptor(name="kairos", model="rbkairos_plus", ns=KAIROS, frame=f"env_{ENV}/kairos", base_frame=f"env_{ENV}/kairos/robot_base_footprint"),
            caps=[RobotCap(cap="arm", adapter="moveit", instance="arm"), RobotCap(cap="mobile", adapter="none")],
        )
        return RobotFleet(robots=[jackal, kairos])

    def _tick(self) -> None:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x, pose.pose.position.y = 1.5, -0.5
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(0.25), math.cos(0.25)
        self._task_pose.publish(pose)
        odom = Odometry()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = pose.pose
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = 0.3, 0.1
        self._odom.publish(odom)
        scan = LaserScan()
        scan.header.frame_id = "lidar_link"
        scan.angle_min, scan.angle_increment = -math.pi, 2 * math.pi / 360
        scan.angle_max = scan.angle_min + 359 * scan.angle_increment
        scan.range_min, scan.range_max = 0.1, 10.0
        ranges = [5.0] * 360
        ranges[300] = 0.7
        ranges[0] = math.inf
        scan.ranges = ranges
        self._scan.publish(scan)
        image = Image(height=4, width=6, encoding="rgb8", step=18, data=bytes([255, 0, 0] * 24))
        image.header.frame_id = "camera"
        self._image.publish(image)
        self._power.publish(Power(total_power_w=42.0, static_power_w=30.0))
        self._energy.publish(Energy(total_energy_consumed_wh=1.5, battery_soc_percent=88.0))

    def start_episode(self, episode_id: int) -> None:
        self._episode.publish(EpisodeRecord(episode_id=episode_id, world="map_empty", outcome_state=EpisodeRecord.RUNNING))

    def _publish_tasks(self) -> None:
        with self._lock:
            entities = [SemanticEntityState(entity=f"env_{ENV}/{name}", kind="robot", discrete_names=["phase", "met", "failed", "dropped", "violated"], discrete_values=[task["phase"], *(",".join(map(str, task[key])) for key in ("met", "failed", "dropped")), ""]) for name, task in self._tasks.items()]
        self._semantics.publish(SemanticSnapshot(env_id=ENV, world="map_empty", entities=entities))

    def _submit(self, request: object, response: object) -> object:
        response.success = False
        if not self.initialized:
            response.error_msg = "task generator not initialized"
            return response
        if request.robot not in self._tasks:
            response.error_msg = f"unknown robot {request.robot!r}, robots: {sorted(self._tasks)}"
            return response
        try:
            submitted = TaskRequest.parse(yaml.safe_load(request.task_yaml) or [])
        except Exception as exc:
            response.error_msg = str(exc)
            return response
        self.submissions.append((request.robot, submitted))
        indices: list[int] = []
        with self._lock:
            task = self._tasks[request.robot]
            if task["phase"]:
                task["dropped"].append(int(task["phase"]))
                task["phase"] = ""
            goal = self._goals.pop(request.robot, None)
            phase = submitted.phases[0] if submitted.phases else None
            if isinstance(phase, GoToPhase) and phase.pose is not None and phase.pose.position.y >= UPPER_LEVEL_Y:
                task["met"].append(self._next)
                indices.append(self._next)
                self._next += 1
            if phase is not None:
                indices.append(self._next)
                task["phase"] = str(self._next)
                self._next += 1
        self.indices.append(indices)
        if goal is not None:
            goal.cancel_goal_async()
        self._publish_tasks()
        if phase is not None:
            threading.Thread(target=self._dispatch, args=(request.robot, indices[-1], phase), daemon=True).start()
        response.phase_indices = indices
        response.success = True
        return response

    def _dispatch(self, robot: str, index: int, phase: object) -> None:
        if isinstance(phase, GoToPhase):
            client, goal = self._clients["goto"], GotoPose.Goal()
            goal.target.header.frame_id = "map"
            goal.target.pose.position.x = phase.pose.position.x
        elif isinstance(phase, ReachPhase):
            client, goal = self._clients["reach"], ReachPose.Goal(named_target=phase.named_target or "")
        else:
            client, goal = self._clients["gesture"], PlayGesture.Goal(gesture=phase.gesture)
        handle = _block(client.send_goal_async(goal))
        with self._lock:
            current = self._tasks[robot]["phase"] == str(index)
            if current:
                self._goals[robot] = handle
        if not current:
            handle.cancel_goal_async()
            return
        wrapped = _block(handle.get_result_async())
        with self._lock:
            task = self._tasks[robot]
            if task["phase"] != str(index):
                return
            task["phase"] = ""
            task["met" if wrapped.result.status == 0 else "failed"].append(index)
            self._goals.pop(robot, None)
        self._publish_tasks()

    def _goto(self, handle: object) -> GotoPose.Result:
        target = handle.request.target
        remaining = target.pose.position.x
        result = GotoPose.Result()
        while remaining > 0:
            if handle.is_cancel_requested:
                self.cancelled.append(remaining)
                handle.canceled()
                result.status = GotoPose.Result.STATUS_CANCELED
                return result
            remaining = max(0.0, remaining - 0.25)
            feedback = GotoPose.Feedback(distance_remaining=remaining, eta_seconds=remaining / 0.5)
            feedback.current_pose.header.frame_id = "map"
            feedback.current_pose.pose.position.x = target.pose.position.x - remaining
            handle.publish_feedback(feedback)
            time.sleep(0.05)
        handle.succeed()
        result.status = GotoPose.Result.STATUS_SUCCEEDED
        result.final_pose = target
        return result

    def _reach(self, handle: object) -> ReachPose.Result:
        for step in range(3):
            handle.publish_feedback(ReachPose.Feedback(state="EXECUTING", progress=(step + 1) / 3))
            time.sleep(0.05)
        handle.succeed()
        return ReachPose.Result(status=ReachPose.Result.STATUS_SUCCEEDED, reason=handle.request.named_target)

    def _gesture(self, handle: object) -> PlayGesture.Result:
        handle.abort()
        return PlayGesture.Result(status=PlayGesture.Result.STATUS_UNKNOWN_GESTURE, reason=f"no gesture {handle.request.gesture!r}")

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()


def _block(future: object, timeout: float = 30.0) -> object:
    deadline = time.monotonic() + timeout
    while not future.done():
        if time.monotonic() > deadline:
            raise TimeoutError("peer future did not complete")
        time.sleep(0.01)
    return future.result()


class Rig:
    """The test's event loop thread, where the shared ROS node and every tool run, plus the peer and an MCP server."""

    def __init__(self, loop: asyncio.AbstractEventLoop, peer: Peer) -> None:
        self.loop = loop
        self.peer = peer
        self.server = MCPServer(name="robots-test")
        mcp.domain.install(self.server)

    def __call__(self, coro: object, timeout: float = 60.0) -> object:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)


async def _until(check: object, what: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if await check():
                return
        except ToolError:
            pass
        await asyncio.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    patch.setenv("ARENA_DATA_DIR", str(tmp_path_factory.mktemp("data")))
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    rig = Rig(loop, Peer())

    async def ready() -> None:
        async def graph_ready() -> bool:
            actions = (await ROS.graph())["actions"]
            node = await ROS.node()
            return {f"{JACKAL}/goto_pose", f"{KAIROS}/reach_pose", f"{KAIROS}/play_gesture"} <= set(actions) and all(node.count_subscribers(f"{ns}/cmd_vel") > 0 for ns in (JACKAL, KAIROS))

        async def fleet_ready() -> bool:
            return len(await fleet(ENV)) == 2

        await _until(graph_ready, "the peer's actions in the graph")
        await _until(fleet_ready, "the env registry and fleet")

    rig(ready())
    yield rig
    rig.peer.close()
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    patch.undo()


def test_fleet_reports_served_actions_and_cmd_vel_subscribers(run):
    robots = {entry["name"]: entry for entry in run(mcp.robots_fleet(ENV))}
    jackal, kairos = robots["jackal"], robots["kairos"]
    assert jackal["model"] == "jackal" and jackal["params"] == {"lidar@front": "jackal_default", "imu@imu": "jackal_default"}
    assert jackal["actions"]["goto_pose"] == {"endpoint": f"{JACKAL}/goto_pose", "served": True, "cap": "mobile", "cap_present": True}
    assert jackal["actions"]["reach_pose"]["served"] is False and jackal["actions"]["reach_pose"]["cap_present"] is False
    assert kairos["actions"]["reach_pose"]["served"] and kairos["actions"]["play_gesture"]["served"]
    assert kairos["actions"]["goto_pose"]["served"] is False
    assert jackal["cmd_vel"]["topic"] == f"{JACKAL}/cmd_vel" and jackal["cmd_vel"]["subscribers"] >= 1


def test_robot_must_be_named_when_the_env_has_two(run):
    with pytest.raises(ToolError, match="2 robots"):
        run(mcp.robots_pose(ENV))


def _phases(submission: tuple[str, TaskRequest]) -> tuple[str, list[dict]]:
    robot, request = submission
    return robot, [phase.serialize() for phase in request.phases]


def _stop(run: Rig, robot: str) -> None:
    response = run(ROS.call(SubmitTask, f"{TG}/runtime/submit_task", SubmitTask.Request(robot=robot, task_yaml="[]")))
    assert response.success, response.error_msg


async def _moving(job_id: str) -> bool:
    return "distance_remaining" in JOBS.get(job_id).progress


def test_goto_submits_a_map_frame_phase_and_follows_it_until_met(run):
    job = run(mcp.robots_goto(1.0, 0.0, env_id=ENV, robot="jackal", pose_tolerance=0.3))
    assert job["kind"] == "goto" and job["state"] == "running"
    done = run(JOBS.wait(job["id"], 30))
    assert done.state == SUCCEEDED, done.error
    assert _phases(run.peer.submissions[-1]) == ("jackal", [{"goto": [1.0, 0.0, 0.0], "tolerance_radius": 0.3}])
    assert done.result["outcome"] == "met" and done.result["phase"] == done.progress["phase"]
    assert 0.0 <= done.progress["distance_remaining"] < 1.0 and "eta_seconds" in done.progress
    assert done.progress["frame"] == "map"


def test_goto_targets_off_the_map_are_submitted_unchanged(run):
    job = run(mcp.robots_goto(0.5, -500.0, env_id=ENV, robot="jackal"))
    assert run(JOBS.wait(job["id"], 30)).state == SUCCEEDED
    assert _phases(run.peer.submissions[-1]) == ("jackal", [{"goto": [0.5, -500.0, 0.0]}])


def test_cross_level_goto_follows_the_goto_phase_behind_the_boarding_phases(run):
    job = run(mcp.robots_goto(1.0, UPPER_LEVEL_Y + 10.0, env_id=ENV, robot="jackal"))
    done = run(JOBS.wait(job["id"], 30))
    assert done.state == SUCCEEDED, done.error
    boarding, goto = run.peer.indices[-1]
    assert done.result["phase"] == goto and goto != boarding


def test_a_new_goal_drops_the_running_one(run):
    first = run(mcp.robots_goto(100.0, 0.0, env_id=ENV, robot="jackal"))
    second = run(mcp.robots_goto(1.0, 0.0, env_id=ENV, robot="jackal"))
    dropped = run(JOBS.wait(first["id"], 30))
    assert dropped.state == FAILED and "dropped" in dropped.error
    assert run(JOBS.wait(second["id"], 30)).state == SUCCEEDED


def test_cancelling_a_goto_job_submits_the_empty_task_that_stops_the_robot(run):
    job = run(mcp.robots_goto(100.0, 0.0, env_id=ENV, robot="jackal"))
    run(_until(lambda: _moving(job["id"]), "goto feedback"))
    cancelled = run(JOBS.cancel(job["id"]))
    assert cancelled.state == CANCELLED
    assert _phases(run.peer.submissions[-1]) == ("jackal", [])
    assert run.peer.indices[-1] == []

    async def server_saw_cancel() -> bool:
        return bool(run.peer.cancelled)

    run(_until(server_saw_cancel, "the server to see the cancel", timeout=20))
    assert 0 < run.peer.cancelled[-1] < 100


def test_cancel_raises_when_the_task_generator_refuses_the_stop(run):
    job = run(mcp.robots_goto(100.0, 0.0, env_id=ENV, robot="jackal"))
    run(_until(lambda: _moving(job["id"]), "goto feedback"))
    run.peer.initialized = False
    try:
        with pytest.raises(ToolError, match="stopping jackal failed: task generator not initialized"):
            run(JOBS.cancel(job["id"]))
    finally:
        run.peer.initialized = True
    _stop(run, "jackal")


def test_goto_job_fails_when_the_episode_ends_before_its_phase(run):
    job = run(mcp.robots_goto(100.0, 0.0, env_id=ENV, robot="jackal"))
    run(_until(lambda: _moving(job["id"]), "goto feedback"))
    run.peer.start_episode(2)
    done = run(JOBS.wait(job["id"], 30))
    assert done.state == FAILED and "episode 1 ended before phase" in done.error
    _stop(run, "jackal")


def test_goto_on_a_robot_without_the_endpoint_fails_fast_with_what_it_serves(run):
    start = time.monotonic()
    with pytest.raises(ToolError) as error:
        run(mcp.robots_goto(1.0, 0.0, env_id=ENV, robot="kairos"))
    assert time.monotonic() - start < 30
    assert "goto_pose is not available" in error.value.message
    assert "reach_pose" in error.value.hint and "play_gesture" in error.value.hint


def test_reach_named_pose_job_succeeds_with_progress(run):
    job = run(mcp.robots_reach(ENV, "kairos", named="stow"))
    done = run(JOBS.wait(job["id"], 30))
    assert done.state == SUCCEEDED, done.error
    assert _phases(run.peer.submissions[-1]) == ("kairos", [{"reach": "stow"}])
    assert done.result["outcome"] == "met"
    assert done.progress["state"] == "EXECUTING" and 0 < done.progress["progress"] <= 1.0


def test_reach_cartesian_target_reaches_the_task_generator_in_full(run):
    job = run(mcp.robots_reach(ENV, "kairos", x=0.4, y=0.1, z=0.9, pitch=0.5, position_tolerance=0.02))
    assert run(JOBS.wait(job["id"], 30)).state == SUCCEEDED
    robot, request = run.peer.submissions[-1]
    phase = request.phases[0]
    assert robot == "kairos" and isinstance(phase, ReachPhase)
    assert phase.frame == f"env_{ENV}/kairos/robot_base_footprint" and phase.position_tolerance == 0.02
    assert list(phase.target.position) == pytest.approx([0.4, 0.1, 0.9])
    assert phase.target.orientation.to_euler() == pytest.approx((0.0, 0.5, 0.0))


def test_reach_without_a_target_is_refused(run):
    with pytest.raises(ToolError, match="named or all of x, y and z"):
        run(mcp.robots_reach(ENV, "kairos", x=0.5))


def test_gesture_job_fails_when_the_episode_fails_its_phase(run):
    job = run(mcp.robots_gesture("moonwalk", ENV, "kairos"))
    done = run(JOBS.wait(job["id"], 30))
    assert done.state == FAILED
    assert _phases(run.peer.submissions[-1]) == ("kairos", [{"gesture": "moonwalk"}])
    assert "play_gesture phase" in done.error and "failed" in done.error


def test_drive_repeats_the_command_within_its_rate_then_zero(run):
    run.peer.twists.clear()
    duration_s, rate_hz = 0.5, 20.0
    job = run(mcp.robots_drive(0.4, wz=0.2, duration_s=duration_s, rate_hz=rate_hz, env_id=ENV, robot="jackal"))
    done = run(JOBS.wait(job["id"], 30))
    assert done.state == SUCCEEDED, done.error
    assert done.result["type"] == "geometry_msgs/msg/Twist" and done.result["topic"] == f"{JACKAL}/cmd_vel"
    assert 2 <= done.result["messages"] <= duration_s * rate_hz + 1

    async def zeroed() -> bool:
        twists = run.peer.twists
        driven = any(twist.linear.x == pytest.approx(0.4) for twist in twists)
        return driven and (twists[-1].linear.x, twists[-1].linear.y, twists[-1].angular.z) == (0.0, 0.0, 0.0)

    run(_until(zeroed, "the zero twist after the drive"))
    moving = [twist for twist in run.peer.twists if twist.linear.x == pytest.approx(0.4)]
    assert len(moving) >= 2 and all(twist.angular.z == pytest.approx(0.2) for twist in moving)


def test_drive_commands_for_its_full_duration_after_a_slow_first_publish(run):
    duration_s, settled = 0.5, ROS.settle_discovery
    ROS.settle_discovery = True
    try:
        called = time.monotonic()
        job = run(mcp.robots_drive(0.2, duration_s=duration_s, env_id=ENV, robot="kairos"))
        assert run(JOBS.wait(job["id"], 30)).state == SUCCEEDED
    finally:
        ROS.settle_discovery = settled

    async def zeroed() -> bool:
        return any(twist.linear.x == 0.0 for _, twist in run.peer.kairos_twists)

    run(_until(zeroed, "the zero twist on kairos"))
    assert any(twist.linear.x == pytest.approx(0.2) for _, twist in run.peer.kairos_twists)
    first_zero = next(at for at, twist in run.peer.kairos_twists if twist.linear.x == 0.0)
    assert first_zero - called >= PEER_MATCH_S + duration_s


def test_stop_cancels_the_drive_job_and_publishes_zero(run):
    job = run(mcp.robots_drive(0.3, duration_s=60.0, env_id=ENV, robot="jackal"))

    async def driving() -> bool:
        return any(twist.linear.x == pytest.approx(0.3) for twist in run.peer.twists)

    run(_until(driving, "drive commands"))
    stopped = run(mcp.robots_stop(ENV, "jackal"))
    assert stopped["cancelled_drive"] == job["id"]
    assert JOBS.get(job["id"]).state == CANCELLED
    assert stopped["subscribers"] >= 1

    async def zeroed() -> bool:
        last = run.peer.twists[-1]
        return (last.linear.x, last.angular.z) == (0.0, 0.0)

    run(_until(zeroed, "the zero twist"))


def test_pose_reads_task_pose_odometry_and_trail(run):
    pose = run(mcp.robots_pose(ENV, "jackal"))
    assert pose["task_pose"]["x"] == 1.5 and pose["task_pose"]["frame"] == "map"
    assert pose["task_pose"]["yaw"] == pytest.approx(0.5)
    assert pose["odom"]["twist"] == {"vx": 0.3, "vy": 0.0, "wz": 0.1}
    assert pose["odom"]["child_frame"] == "base_link"
    assert pose["trail"] == {"points": 3, "length_m": 3.0}


def test_sensors_resolve_live_topics_and_probe_delivery(run):
    sensors = {sensor["name"]: sensor for sensor in run(mcp.robots_sensors(ENV, "jackal", probe_s=3.0))}
    lidar = next(sensor for sensor in sensors.values() if sensor["type"] == "laserscan")
    assert lidar["topic"] == LIDAR
    assert lidar["publishers"] >= 1 and lidar["ros_types"] == ["sensor_msgs/msg/LaserScan"] and lidar["received"] is True
    imu = next(sensor for sensor in sensors.values() if sensor["type"] == "imu")
    assert imu["topic"] == IMU and imu["publishers"] == 0 and imu["received"] is False


def test_scan_summary_and_image_through_the_server(run):
    summary = run(mcp.robots_scan(ENV, "jackal", sectors=4))
    assert summary["beams"] == 360 and summary["valid"] == 359
    assert summary["nearest"]["range"] == pytest.approx(0.7) and summary["nearest"]["bearing_deg"] == pytest.approx(120.0)
    assert [sector["min_range"] for sector in summary["sectors"]] == pytest.approx([5.0, 5.0, 5.0, 0.7])
    result = run(run.server.call_tool("robots_scan", {"env_id": ENV, "robot": "jackal", "image": True}))
    kinds = [block.type for block in result.content]
    assert kinds == ["text", "image"]
    assert json.loads(result.content[0].text)["sensor"] == summary["sensor"]
    assert result.content[1].mime_type == "image/png"


def test_camera_returns_the_frame_as_image_content_through_the_server(run):
    result = run(run.server.call_tool("robots_camera", {"env_id": ENV, "robot": "kairos"}))
    assert [block.type for block in result.content] == ["text", "image"]
    info = json.loads(result.content[0].text)
    assert info["topic"] == CAMERA and (info["width"], info["height"], info["encoding"]) == (6, 4, "rgb8")
    with pytest.raises(ToolError, match="declares no image or depth sensor"):
        run(mcp.robots_camera(ENV, "jackal"))


def test_power_reads_power_and_energy_and_refuses_unpowered_robots(run):
    power = run(mcp.robots_power(ENV, "jackal"))
    assert power["power"]["total_power_w"] == 42.0 and power["energy"]["battery_soc_percent"] == 88.0
    with pytest.raises(ToolError, match="no power publisher"):
        run(mcp.robots_power(ENV, "kairos"))
