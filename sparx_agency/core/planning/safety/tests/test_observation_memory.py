"""Tests for the observation memory.

Each case is written as the flight situation it exists for, because the value of
this class is entirely in which situations it calls stale. A memory that brakes
the aircraft while it is flying forward into what it is looking at would be
worse than not having one.
"""

import math

import pytest

from sparx_agency.core.planning.safety.observation_memory import (
    ObservationMemory, ObservationMemoryConfig)


def test_nothing_is_observed_before_the_first_frame():
    mem = ObservationMemory()
    assert mem.age(0.0, 10.0) is None
    assert mem.is_stale(0.0, 10.0, max_age_s=1.0)


def test_the_direction_the_camera_faces_is_fresh():
    mem = ObservationMemory()
    mem.observe(yaw_rad=0.0, now_s=100.0)
    assert mem.age(0.0, 100.0) == pytest.approx(0.0)
    assert not mem.is_stale(0.0, 100.5, max_age_s=1.0)


def test_flying_forward_is_never_braked():
    """The whole point: what the camera looks at must stay fresh."""
    mem = ObservationMemory()
    now = 0.0
    for _ in range(50):
        mem.observe(yaw_rad=0.3, now_s=now)
        now += 0.1
    assert not mem.is_stale(0.3, now, max_age_s=1.0)


def test_behind_the_aircraft_is_stale_however_long_it_flies_forward():
    """A retreat backs into exactly this, which is why retreats hurt."""
    mem = ObservationMemory()
    now = 0.0
    for _ in range(50):
        mem.observe(yaw_rad=0.0, now_s=now)
        now += 0.1
    assert mem.is_stale(math.pi, now, max_age_s=1.0)


def test_sideways_is_stale_too():
    """An unstick that steps 90 degrees off the nose moves into unseen space."""
    mem = ObservationMemory()
    mem.observe(yaw_rad=0.0, now_s=100.0)
    assert mem.is_stale(0.5 * math.pi, 100.0, max_age_s=1.0)
    assert mem.is_stale(-0.5 * math.pi, 100.0, max_age_s=1.0)


def test_the_wedge_edges_are_marked():
    """A bearing just inside the field of view counts as looked at."""
    cfg = ObservationMemoryConfig()
    mem = ObservationMemory(cfg)
    mem.observe(yaw_rad=0.0, now_s=5.0)
    inside = cfg.half_fov_rad - cfg.margin_rad - 0.01
    assert not mem.is_stale(inside, 5.0, max_age_s=0.5)
    assert not mem.is_stale(-inside, 5.0, max_age_s=0.5)


def test_outside_the_wedge_is_not_marked():
    cfg = ObservationMemoryConfig()
    mem = ObservationMemory(cfg)
    mem.observe(yaw_rad=0.0, now_s=5.0)
    assert mem.is_stale(cfg.half_fov_rad + 0.4, 5.0, max_age_s=0.5)


def test_a_turn_on_the_spot_refreshes_the_whole_circle():
    """The recovery survey exists to do this; it must clear the memory."""
    mem = ObservationMemory()
    now = 0.0
    for i in range(72):
        mem.observe(yaw_rad=i * (2.0 * math.pi / 72.0), now_s=now)
        now += 0.05
    for probe in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        assert not mem.is_stale(probe, now, max_age_s=5.0)


def test_age_grows_with_time():
    mem = ObservationMemory()
    mem.observe(yaw_rad=0.0, now_s=10.0)
    assert mem.age(0.0, 12.5) == pytest.approx(2.5)


def test_bearings_wrap():
    mem = ObservationMemory()
    mem.observe(yaw_rad=0.0, now_s=1.0)
    assert mem.age(2.0 * math.pi, 1.0) == pytest.approx(0.0)
    assert mem.age(-2.0 * math.pi, 1.0) == pytest.approx(0.0)


def test_a_clock_reset_reads_as_fresh_not_as_ancient():
    """A simulation restart must not brake the aircraft for bookkeeping."""
    mem = ObservationMemory()
    mem.observe(yaw_rad=0.0, now_s=900.0)
    assert mem.age(0.0, 3.0) == pytest.approx(0.0)
    assert not mem.is_stale(0.0, 3.0, max_age_s=1.0)


def test_reset_forgets_everything():
    mem = ObservationMemory()
    mem.observe(yaw_rad=0.0, now_s=1.0)
    mem.reset()
    assert mem.age(0.0, 1.0) is None


def test_config_rejects_a_wedge_shrunk_to_nothing():
    with pytest.raises(ValueError):
        ObservationMemoryConfig(half_fov_rad=0.2, margin_rad=0.2)


def test_config_rejects_absurd_resolution():
    with pytest.raises(ValueError):
        ObservationMemoryConfig(sectors=2)


def test_no_sector_inside_the_wedge_is_skipped():
    """Stepping through the wedge must not leave holes between sectors."""
    cfg = ObservationMemoryConfig(sectors=36)
    mem = ObservationMemory(cfg)
    mem.observe(yaw_rad=1.0, now_s=7.0)
    reach = cfg.half_fov_rad - cfg.margin_rad
    probe = -reach
    while probe <= reach:
        assert not mem.is_stale(1.0 + probe, 7.0, max_age_s=0.5), probe
        probe += 0.02
