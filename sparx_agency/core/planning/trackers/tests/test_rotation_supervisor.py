"""Tests for the holonomic rotation-freeze + re-observation supervisor."""
import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.trackers.rotation_supervisor import (
    CRUISE,
    DISABLED,
    STOP_COAST,
    STOP_DWELL,
    TURNING,
    RotationReobserveSupervisor,
    RotationSupervisorParams,
)

DT = 0.1


def _drive(sup, cmd_wz_seq, *, updates_per_tick=1, dt=DT):
    """Feed a commanded-yaw-rate sequence through a simple unicycle plant.

    The plant rotates at the commanded rate when the supervisor is not holding,
    and stops instantly when held (so the coast is one tick). ``updates_per_tick``
    fresh voxel updates arrive each tick.
    """
    yaw, count, recs = 0.0, 0, []
    for cmd_wz in cmd_wz_seq:
        dec = sup.update(yaw, cmd_wz, dt, count)
        recs.append(dec)
        actual_wz = 0.0 if dec.hold else cmd_wz
        yaw = normalize_angle(yaw + actual_wz * dt)
        count += updates_per_tick
    return recs


def test_disabled_never_freezes_or_holds():
    sup = RotationReobserveSupervisor(RotationSupervisorParams(enabled=False))
    recs = _drive(sup, [0.7] * 40)
    assert all(d.state == DISABLED for d in recs)
    assert all(d.freeze is False and d.hold is False for d in recs)


def test_cruise_stays_live_below_threshold():
    sup = RotationReobserveSupervisor()      # wz_turn_on = 0.20
    recs = _drive(sup, [0.05] * 20)          # gentle path-tracking yaw
    assert all(d.state == CRUISE for d in recs)
    assert all(d.freeze is False and d.hold is False for d in recs)


def test_turn_freezes_from_onset():
    sup = RotationReobserveSupervisor()
    recs = _drive(sup, [0.7] * 5)            # a real turn
    # From the first turning tick the map is frozen.
    assert recs[0].freeze is True and recs[0].state == TURNING
    assert all(d.freeze is True for d in recs)


def test_midturn_stop_after_reobserve_interval():
    """A long continuous turn stops for a re-observation every reobserve_every_rad,
    coasting (frozen) then dwelling (live) until >=2 voxel updates land."""
    p = RotationSupervisorParams(reobserve_every_rad=math.radians(25.0),
                                 settle_dwell_s=0.3, settle_map_updates=2)
    sup = RotationReobserveSupervisor(p)
    recs = _drive(sup, [0.7] * 60, updates_per_tick=1)
    states = [d.state for d in recs]
    assert TURNING in states
    assert STOP_COAST in states              # it stopped mid-turn
    assert STOP_DWELL in states
    # During every STOP_DWELL tick the map is unfrozen (re-observing) and held.
    for d in recs:
        if d.state == STOP_DWELL:
            assert d.freeze is False and d.hold is True and d.reobserving is True
        if d.state == STOP_COAST:
            assert d.freeze is True and d.hold is True
    # It resumes turning after the checkpoint (TURNING appears again after a stop).
    first_stop = states.index(STOP_COAST)
    assert TURNING in states[first_stop:]


def test_reobserve_blocks_until_updates_land():
    """A stop will not resume until settle_map_updates fresh voxels arrive."""
    p = RotationSupervisorParams(reobserve_every_rad=math.radians(15.0),
                                 settle_dwell_s=0.2, settle_map_updates=2,
                                 map_wait_timeout_s=100.0)
    sup = RotationReobserveSupervisor(p)
    # Turn until the first stop, feeding NO voxel updates during the dwell.
    yaw, count = 0.0, 0
    reached_dwell = False
    for _ in range(200):
        dec = sup.update(yaw, 0.7, DT, count)
        yaw = normalize_angle(yaw + (0.0 if dec.hold else 0.7) * DT)
        if dec.state == STOP_DWELL:
            reached_dwell = True
            break
    assert reached_dwell
    # Now dwell for a long time with the voxel count FROZEN: must keep holding.
    for _ in range(60):
        dec = sup.update(yaw, 0.0, DT, count)     # count never advances
        assert dec.hold is True and dec.state == STOP_DWELL
    # Once >=2 fresh updates arrive, it releases and resumes turning.
    released = False
    for i in range(10):
        dec = sup.update(yaw, 0.0, DT, count + 2 + i)
        if not dec.hold:
            released = True
            break
    assert released


def test_reobserve_times_out_so_it_never_hangs():
    """A mapping stall (no updates ever) still releases after map_wait_timeout_s."""
    p = RotationSupervisorParams(reobserve_every_rad=math.radians(15.0),
                                 settle_dwell_s=0.2, settle_map_updates=2,
                                 map_wait_timeout_s=0.5)
    sup = RotationReobserveSupervisor(p)
    yaw, count = 0.0, 0
    released_after_stop = False
    saw_stop = False
    for _ in range(400):
        dec = sup.update(yaw, 0.7, DT, count)     # count frozen: never any update
        yaw = normalize_angle(yaw + (0.0 if dec.hold else 0.7) * DT)
        if dec.state in (STOP_COAST, STOP_DWELL):
            saw_stop = True
        if saw_stop and dec.state == TURNING:
            released_after_stop = True
            break
    assert saw_stop and released_after_stop      # timed out and resumed


def test_turn_end_does_final_stop_then_cruise():
    """When the commanded yaw drops, the turn ends with one stop + re-observe,
    then returns to CRUISE (live)."""
    sup = RotationReobserveSupervisor(
        RotationSupervisorParams(settle_dwell_s=0.2, settle_map_updates=1))
    seq = [0.7] * 8 + [0.0] * 40              # turn, then straighten out
    recs = _drive(sup, seq, updates_per_tick=1)
    states = [d.state for d in recs]
    assert states[-1] == CRUISE               # ended back in cruise
    assert recs[-1].freeze is False and recs[-1].hold is False
