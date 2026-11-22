import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.sweep_poses import pose_grid


def square(side_m, step=0.1):
    xs = np.arange(0, side_m, step)
    return np.array([[x, 0.0, z] for x in xs for z in xs])


def test_one_position_per_spacing_square_with_every_heading():
    poses = pose_grid(square(3.0), spacing_m=1.0, headings=4, cap=100, seed=0)
    assert len(poses) == 9 * 4
    assert sorted({round(yaw, 3) for _, yaw in poses}) == [0.0, 1.571, 3.142, 4.712]


def test_the_cap_limits_positions_and_not_headings():
    poses = pose_grid(square(5.0), spacing_m=1.0, headings=4, cap=6, seed=0)
    assert len({tuple(p) for p, _ in poses}) == 6 and len(poses) == 24


def test_the_same_seed_gives_the_same_poses():
    a = pose_grid(square(5.0), 1.0, 4, 6, seed=3)
    b = pose_grid(square(5.0), 1.0, 4, 6, seed=3)
    assert all(np.array_equal(p, q) and y == z for (p, y), (q, z) in zip(a, b))
