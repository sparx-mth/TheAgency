import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.buffer_rules import buffer_picks

TWO_PI = 2 * np.pi
FRAMES = {n: {"step": s, "yaw": y, "near": 0.1, "inside": 0.9, "floor_share": f}
          for n, s, y, f in (("a", 0, 0.0, 0.1), ("b", 5, 0.1, 0.9), ("c", 10, 3.1, 0.5), ("d", 15, 3.2, 0.3), ("e", 20, 0.2, 0.7))}
FRAMES["close"] = {"step": 7, "yaw": 1.0, "near": 0.9, "inside": 0.9, "floor_share": 1.0}


def rng():
    return np.random.default_rng(0)


def test_first_takes_the_earliest_usable_frames():
    assert buffer_picks(FRAMES, "first", 2, rng()) == ["a", "b"]


def test_even_time_spreads_the_picks_over_the_buffer():
    assert buffer_picks(FRAMES, "even_time", 3, rng()) == ["a", "c", "e"]


def test_floor_takes_the_most_floor_and_never_a_close_up():
    assert buffer_picks(FRAMES, "floor", 2, rng()) == ["b", "e"]


def test_heading_spread_adds_the_frame_looking_most_differently():
    picks = buffer_picks(FRAMES, "heading_spread", 2, rng())
    assert picks[0] == "b" and picks[1] in ("c", "d")


def test_random_picks_usable_frames_only():
    assert set(buffer_picks(FRAMES, "random", 3, rng())) <= {"a", "b", "c", "d", "e"}
