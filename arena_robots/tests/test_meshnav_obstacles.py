from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from arena_robots.meshnav_obstacles import ObstacleFilter, WalkingSurface, read_ply
from arena_simulation_setup.shared.navmesh import NavMesh

_RES = 0.1
_CLIMB = 0.25
_RISER = 0.17
_TREAD = 0.3
_STAIR_START = 2.04
_STEPS = 10
_UPPER = 3.0


def _grid_mesh(heights: np.ndarray, x0: float, y0: float) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = heights.shape
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    vertices = np.stack([x0 + ii.ravel() * _RES, y0 + jj.ravel() * _RES, heights.ravel()], axis=1)
    index = np.arange(nx * ny).reshape(nx, ny)
    v00, v10, v11, v01 = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    quad = np.stack([heights[:-1, :-1], heights[1:, :-1], heights[1:, 1:], heights[:-1, 1:]])
    keep = (quad.max(axis=0) - quad.min(axis=0)) <= _CLIMB + 1e-9
    faces = np.concatenate([np.stack([v00[keep], v10[keep], v11[keep]], axis=1), np.stack([v00[keep], v11[keep], v01[keep]], axis=1)])
    return vertices, faces


def _stair_height(x: np.ndarray) -> np.ndarray:
    return _RISER * np.clip(np.floor((x - _STAIR_START) / _TREAD) + 1, 0, _STEPS)


def _write(tmp_path: Path, *meshes: tuple[np.ndarray, np.ndarray]) -> Path:
    vertices, faces, count = [], [], 0
    for v, f in meshes:
        vertices.append(v)
        faces.append(f + count)
        count += len(v)
    return NavMesh(np.concatenate(vertices), np.concatenate(faces)).write_ply(tmp_path / "mesh.ply")


@pytest.fixture()
def flat(tmp_path: Path) -> WalkingSurface:
    return WalkingSurface.from_ply(_write(tmp_path, _grid_mesh(np.zeros((101, 101)), 0.0, 0.0)), _RES)


@pytest.fixture()
def stairs(tmp_path: Path) -> WalkingSurface:
    x = np.arange(81) * _RES
    return WalkingSurface.from_ply(_write(tmp_path, _grid_mesh(np.repeat(_stair_height(x)[:, None], 21, axis=1), 0.0, 0.0)), _RES)


@pytest.fixture()
def stacked(tmp_path: Path) -> WalkingSurface:
    lower = _grid_mesh(np.zeros((101, 101)), 0.0, 0.0)
    upper = _grid_mesh(np.full((51, 101), _UPPER), 5.0, 0.0)
    return WalkingSurface.from_ply(_write(tmp_path, lower, upper), _RES)


def _obstacles(surface: WalkingSurface, points: np.ndarray, max_step: float = 0.2) -> np.ndarray:
    return surface.obstacles(np.asarray(points, dtype=float), min_height=0.1, max_height=0.7, max_step=max_step)


def _scatter(rng: np.random.Generator, n: int, x: tuple[float, float], y: tuple[float, float], z: tuple[float, float]) -> np.ndarray:
    return np.stack([rng.uniform(*x, n), rng.uniform(*y, n), rng.uniform(*z, n)], axis=1)


def test_read_ply_returns_the_vertices_and_faces_navmesh_writes(tmp_path: Path):
    vertices, faces = _grid_mesh(np.arange(12, dtype=float).reshape(3, 4) * 0.01, 1.0, -2.0)
    read_vertices, read_faces = read_ply(_write(tmp_path, (vertices, faces)))
    assert np.allclose(read_vertices, vertices, atol=1e-6)
    assert np.array_equal(read_faces, faces)


def test_floor_returns_are_not_obstacles(flat: WalkingSurface):
    points = _scatter(np.random.default_rng(0), 5000, (0.5, 9.5), (0.5, 9.5), (-0.03, 0.05))
    assert not _obstacles(flat, points).any()


def test_a_box_and_a_person_on_the_floor_are_obstacles(flat: WalkingSurface):
    rng = np.random.default_rng(1)
    box = _scatter(rng, 500, (3.0, 3.4), (3.0, 3.4), (0.12, 0.4))
    person = _scatter(rng, 500, (6.0, 6.3), (6.0, 6.3), (0.12, 0.7))
    assert _obstacles(flat, box).all()
    assert _obstacles(flat, person).all()


def test_returns_above_the_robot_height_are_dropped(flat: WalkingSurface):
    head = _scatter(np.random.default_rng(2), 500, (6.0, 6.3), (6.0, 6.3), (0.71, 1.8))
    assert not _obstacles(flat, head).any()


def test_returns_off_the_mesh_are_dropped(flat: WalkingSurface):
    beyond = _scatter(np.random.default_rng(3), 500, (11.0, 20.0), (0.0, 10.0), (0.2, 0.6))
    assert not _obstacles(flat, beyond).any()


@pytest.mark.parametrize("max_step", [0.1, 0.2])
def test_stair_treads_and_risers_are_not_obstacles(stairs: WalkingSurface, max_step: float):
    rng = np.random.default_rng(4)
    x = rng.uniform(0.2, 7.8, 5000)
    treads = np.stack([x, rng.uniform(0.2, 1.8, len(x)), _stair_height(x) + rng.uniform(-0.02, 0.02, len(x))], axis=1)
    edges = _STAIR_START + _TREAD * rng.integers(0, _STEPS, 5000)
    lows = _stair_height(edges - 0.01)
    risers = np.stack([edges + rng.uniform(-0.01, 0.01, len(edges)), rng.uniform(0.2, 1.8, len(edges)), lows + rng.uniform(0.0, _RISER, len(edges))], axis=1)
    assert not _obstacles(stairs, treads, max_step).any()
    assert not _obstacles(stairs, risers, max_step).any()


_BELT_SLOPE = 0.55


@pytest.fixture()
def belt(tmp_path: Path) -> WalkingSurface:
    x = np.arange(81) * _RES
    heights = _BELT_SLOPE * np.clip(x - 2.0, 0.0, 4.0)
    return WalkingSurface.from_ply(_write(tmp_path, _grid_mesh(np.repeat(heights[:, None], 21, axis=1), 0.0, 0.0)), _RES)


def test_a_moving_belt_and_its_cleats_are_not_obstacles_but_a_rider_is(belt: WalkingSurface):
    rng = np.random.default_rng(6)
    x = rng.uniform(2.2, 5.8, 5000)
    surface = np.stack([x, rng.uniform(0.2, 1.8, len(x)), _BELT_SLOPE * (x - 2.0) + rng.uniform(-0.02, 0.03, len(x))], axis=1)
    rider = np.stack([np.full(200, 4.0), np.full(200, 1.0), _BELT_SLOPE * 2.0 + np.linspace(0.3, 0.65, 200)], axis=1)
    assert not _obstacles(belt, surface, 0.1).any()
    assert _obstacles(belt, rider, 0.1).all()


@pytest.mark.parametrize("travel", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("max_step", [0.1, 0.2])
def test_escalator_steps_between_their_rest_positions_are_not_obstacles(stairs: WalkingSurface, travel: float, max_step: float):
    rng = np.random.default_rng(7)
    shift = travel * _TREAD
    lift = travel * _RISER
    x = rng.uniform(_STAIR_START + _TREAD, _STAIR_START + (_STEPS - 1) * _TREAD, 5000)
    treads = np.stack([x, rng.uniform(0.2, 1.8, len(x)), _stair_height(x - shift) + lift + rng.uniform(-0.02, 0.02, len(x))], axis=1)
    edges = _STAIR_START + shift + _TREAD * rng.integers(1, _STEPS - 1, 5000)
    lows = _stair_height(edges - shift - 0.01) + lift
    risers = np.stack([edges + rng.uniform(-0.01, 0.01, len(edges)), rng.uniform(0.2, 1.8, len(edges)), lows + rng.uniform(0.0, _RISER, len(edges))], axis=1)
    assert not _obstacles(stairs, treads, max_step).any()
    assert not _obstacles(stairs, risers, max_step).any()


def test_a_person_standing_on_the_stairs_is_an_obstacle(stairs: WalkingSurface):
    centers = _STAIR_START + _TREAD * (np.arange(1, _STEPS - 1) + 0.5)
    heights = np.linspace(0.3, 0.65, 8)
    x, z = np.meshgrid(centers, heights, indexing="ij")
    person = np.stack([x.ravel(), np.full(x.size, 1.0), _stair_height(x.ravel()) + z.ravel()], axis=1)
    assert _obstacles(stairs, person).all()


def test_returns_under_an_upper_level_are_judged_against_the_lower_floor(stacked: WalkingSurface):
    rng = np.random.default_rng(5)
    below_upper = _scatter(rng, 500, (5.5, 9.5), (0.5, 9.5), (0.12, 0.7))
    lower_floor = _scatter(rng, 500, (5.5, 9.5), (0.5, 9.5), (-0.02, 0.05))
    ceiling = _scatter(rng, 500, (5.5, 9.5), (0.5, 9.5), (1.0, _UPPER - 0.05))
    upper_floor = _scatter(rng, 500, (5.5, 9.5), (0.5, 9.5), (_UPPER - 0.02, _UPPER + 0.05))
    on_upper = _scatter(rng, 500, (5.5, 9.5), (0.5, 9.5), (_UPPER + 0.12, _UPPER + 0.7))
    assert _obstacles(stacked, below_upper).all()
    assert not _obstacles(stacked, lower_floor).any()
    assert not _obstacles(stacked, ceiling).any()
    assert not _obstacles(stacked, upper_floor).any()
    assert _obstacles(stacked, on_upper).all()


def _pose(yaw: float, x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    matrix[:3, 3] = [x, y, z]
    return matrix


def test_the_filter_drops_self_points_and_returns_beyond_range_and_places_the_rest_in_odom(flat: WalkingSurface):
    obstacle_filter = ObstacleFilter(flat, min_height=0.1, max_height=0.7, max_step=0.2, obstacle_range=5.0, self_radius=0.4)
    sensor_in_base = _pose(0.0, 0.1, 0.0, 0.3)
    base_in_odom = _pose(np.pi / 2, 5.0, 5.0, 0.0)
    points = np.array(
        [
            [0.1, 0.1, -0.1],
            [-0.3, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [3.0, 0.0, -0.3],
            [np.nan, 0.0, 0.0],
            [np.inf, 0.0, 0.0],
        ]
    )
    kept = obstacle_filter(points, sensor_in_base, base_in_odom)
    assert np.allclose(kept, [[5.0, 7.1, 0.3]])


def test_a_16_beam_lidar_scan_classifies_without_a_pathological_slowdown(tmp_path: Path):
    surface = WalkingSurface.from_ply(_write(tmp_path, _grid_mesh(np.zeros((501, 501)), 0.0, 0.0)), _RES)
    rings = np.radians(np.linspace(-15.0, 15.0, 16))
    azimuths = np.linspace(-np.pi, np.pi, 1800, endpoint=False)
    elevation, azimuth = np.meshgrid(rings, azimuths, indexing="ij")
    rays = np.stack([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)], axis=-1).reshape(-1, 3)
    points = np.array([25.0, 25.0, 0.4]) + rays * 8.0
    _obstacles(surface, points)
    start = time.thread_time()
    for _ in range(5):
        _obstacles(surface, points)
    assert (time.thread_time() - start) / 5 < 1.0
