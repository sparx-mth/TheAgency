"""Tests for the lost-localization recovery ladder."""
import math
import random

import pytest

from sparx_agency.core.planning.lost_localization import (
    BACK,
    CLIMB,
    DISABLED,
    GIVE_UP,
    HOLD,
    LADDER,
    NOMINAL,
    TURN,
    LostLocalizationParams,
    LostLocalizationRecovery,
    build_ladder,
)

DT = 0.1


def _drive(rec, tag_seq, *, dt=DT, yaw_rate=None, state=None):
    """Feed a tag-visibility sequence through a toy localization plant.

    The plant owns what the node measures: a pose age that resets to 0 when a
    message lands and grows by ``dt` otherwise, plus a monotonic count of the
    messages that have landed. ``tag_seq`` is a sequence of bools -- one entry
    per control tick, True meaning a tag was seen and a pose published.

    ``state`` carries (age, count, yaw) across calls so a test can drive the same
    recovery through several phases; pass the dict back in to continue.

    When ``yaw_rate`` is given the plant also integrates a heading from the
    commanded yaw rate and feeds it back, standing in for the platform's
    localization-independent bearing. Otherwise no heading is available and the
    sweep must run open loop.
    """
    s = state if state is not None else {"age": 0.0, "count": 0, "yaw": 0.0}
    recs = []
    for visible in tag_seq:
        if visible:
            s["age"] = 0.0
            s["count"] += 1
        else:
            s["age"] += dt
        dec = rec.update(s["age"], dt, s["count"],
                         yaw=None if yaw_rate is None else s["yaw"])
        recs.append(dec)
        if yaw_rate is not None and dec.command is not None:
            s["yaw"] += dec.command.yaw_rate * dt
    return recs


def _labels(recs):
    return [d.rung_label for d in recs]


def test_disabled_never_touches_the_drone():
    rec = LostLocalizationRecovery(LostLocalizationParams(enabled=False))
    recs = _drive(rec, [False] * 200)
    assert all(d.state == DISABLED for d in recs)
    assert all(d.active is False and d.command is None for d in recs)


def test_fresh_localization_is_passive():
    """While the pose is fresh the follower owns cmd_vel: publish NOTHING."""
    rec = LostLocalizationRecovery()
    recs = _drive(rec, [True] * 50)
    assert all(d.state == NOMINAL for d in recs)
    assert all(d.active is False and d.command is None for d in recs)


def test_never_bootstrapped_does_not_arm():
    """An infinite age means localization was never wired up, not lost.

    A mis-wired bridge (one that never bridges the pose topic at all) must not
    put the drone into a blind back-up on boot.
    """
    rec = LostLocalizationRecovery()
    for _ in range(200):
        dec = rec.update(float("inf"), DT, 0)
        assert dec.state == NOMINAL
        assert dec.active is False and dec.command is None


def test_stale_pose_stops_before_the_ladder():
    """0.3s cold => STOP and hold; the ladder waits for 1.0s."""
    p = LostLocalizationParams(stale_s=0.3, ladder_s=1.0)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 6)     # ages 0.0, 0.1 .. 0.6

    holds = [d for d in recs if d.state == HOLD]
    assert holds, "a cold pose must enter HOLD"
    for d in holds:
        assert d.active is True, "HOLD owns cmd_vel (the follower must go passive)"
        assert d.command.x == 0.0 and d.command.yaw_rate == 0.0
        assert d.command.z == 0.0
    assert all(d.state != LADDER for d in recs), "ladder must not start before ladder_s"


def test_hold_publishes_zeros_not_silence():
    """A stop must be an explicit zero command, never silence.

    The platform holds its last command until told otherwise, so publishing
    nothing would let the drone coast on the command it was already flying.
    """
    rec = LostLocalizationRecovery()
    recs = _drive(rec, [True] + [False] * 5)
    stopped = [d for d in recs if d.state == HOLD]
    assert stopped
    assert all(d.command is not None for d in stopped)


def test_ladder_runs_back_back_climb_climb_sweep_in_order():
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, back_duration_s=0.5,
                               dwell_s=0.5, climb_duration_s=0.5)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 200)

    seen = []
    for label in _labels(recs):
        if label and (not seen or seen[-1] != label):
            seen.append(label)
    assert seen == ["back#1", "settle-after-back#1",
                    "back#2", "settle-after-back#2",
                    "climb#1", "settle-after-climb#1",
                    "climb#2", "settle-after-climb#2",
                    "sweep360"]


def test_back_rung_drives_backwards_only():
    """'Fly back' is a negative x and nothing else.

    The bridge is a single-action hold protocol -- yaw beats x beats z -- so a
    rung that also carried a yaw or z would have that axis silently dropped.
    """
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, back_speed=0.25)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 12)
    backs = [d for d in recs if d.rung_label == "back#1"]
    assert backs
    for d in backs:
        assert d.command.x == pytest.approx(-0.25)
        assert d.command.y == 0.0 and d.command.z == 0.0
        assert d.command.yaw_rate == 0.0


def test_climb_rung_drives_up_only():
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, back_duration_s=0.2,
                               dwell_s=0.2, climb_speed=0.2)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 60)
    climbs = [d for d in recs if d.rung_label == "climb#1"]
    assert climbs
    for d in climbs:
        assert d.command.z == pytest.approx(0.2)
        assert d.command.x == 0.0 and d.command.y == 0.0
        assert d.command.yaw_rate == 0.0


def test_sweep_defaults_to_the_right():
    """RIGHT is a field-tested requirement, not a preference.

    A negative yaw rate is clockwise under REP-103, which the XTEND converter
    turns into "turn_right". A sign flip anywhere in that chain is silent -- the
    drone just sweeps the wrong way and fails to find the tag -- so pin it.
    """
    assert LostLocalizationParams().turn_dir == -1
    rec = LostLocalizationRecovery(LostLocalizationParams(
        stale_s=0.3, ladder_s=0.5, back_repeats=0, climb_enabled=False))
    sweeps = [d for d in _drive(rec, [True] + [False] * 20)
              if d.rung_label == "sweep360"]
    assert sweeps
    assert all(d.command.yaw_rate < 0.0 for d in sweeps), "must sweep RIGHT (CW)"


def test_sweep_turns_slowly_to_the_configured_side():
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, back_repeats=0,
                               climb_enabled=False, turn_rate=0.3, turn_dir=-1)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 20)
    sweeps = [d for d in recs if d.rung_label == "sweep360"]
    assert sweeps
    for d in sweeps:
        assert d.command.yaw_rate == pytest.approx(-0.3)
        assert d.command.x == 0.0 and d.command.z == 0.0


def test_sweep_closes_on_an_independent_heading():
    """With a bearing the sweep ends on angle, well before its timeout."""
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, back_repeats=0,
                               climb_enabled=False, turn_rate=0.5,
                               turn_timeout_s=60.0)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 400, yaw_rate=True)

    sweeps = [d for d in recs if d.rung_label == "sweep360"]
    # 2*pi at 0.5 rad/s = ~12.6s = ~126 ticks; the 60s timeout would be 600.
    assert 120 <= len(sweeps) <= 135, "sweep should end on angle, not on timeout"
    assert max(d.sweep_rad for d in sweeps) >= 2.0 * math.pi - 0.1
    assert any(d.state == GIVE_UP for d in recs)


def test_sweep_is_not_fooled_by_a_noisy_heading():
    """Compass noise must not accumulate into phantom rotation.

    Summing |delta| would: the deltas of a noisy heading are a random walk whose
    magnitudes only add up, so ~1 deg of noise at 20 Hz over a 21 s sweep invents
    most of a turn and ends the search after ~200 deg of real rotation -- pointed
    nowhere useful, which is the one thing the sweep exists to avoid.
    """
    rng = random.Random(7)
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, back_repeats=0,
                               climb_enabled=False, turn_rate=0.3,
                               turn_timeout_s=60.0)
    rec = LostLocalizationRecovery(p)

    true_yaw, sweeps = 0.0, []
    for i in range(1000):
        age = 0.0 if i == 0 else 10.0
        dec = rec.update(age, DT, 1, yaw=true_yaw + rng.gauss(0.0, 0.02))
        if dec.rung_label == "sweep360":
            sweeps.append((dec.sweep_rad, true_yaw))
            true_yaw += dec.command.yaw_rate * DT
        if dec.state == GIVE_UP:
            break

    swept_for_real = abs(sweeps[-1][1])        # magnitude: the sweep may go either way
    assert swept_for_real >= 2.0 * math.pi - 0.3, (
        "sweep ended after only %.0f deg of REAL rotation -- noise inflated it"
        % math.degrees(swept_for_real))


def test_sweep_measures_rotation_whichever_way_the_compass_counts():
    """The platform's heading sign convention must not matter.

    A compass that counts clockwise and one that counts counter-clockwise must
    both register a left turn as progress -- otherwise the sweep would only ever
    end on its timeout on half the platforms.
    """
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, back_repeats=0,
                               climb_enabled=False, turn_rate=0.5,
                               turn_timeout_s=60.0)
    lengths = []
    for convention in (+1.0, -1.0):
        rec = LostLocalizationRecovery(p)
        yaw, n = 0.0, 0
        for i in range(1000):
            dec = rec.update(0.0 if i == 0 else 10.0, DT, 1, yaw=yaw)
            if dec.rung_label == "sweep360":
                n += 1
                yaw += dec.command.yaw_rate * DT * convention
            if dec.state == GIVE_UP:
                break
        lengths.append(n)
    assert lengths[0] == lengths[1], \
        "sweep length depends on the compass sign convention: %r" % (lengths,)
    assert 120 <= lengths[0] <= 135, "and both must close on angle, not timeout"


def test_sweep_without_a_heading_ends_on_its_timeout():
    """No bearing => open loop; the timeout is the only thing that ends it."""
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, back_repeats=0,
                               climb_enabled=False, turn_rate=2.0,
                               turn_timeout_s=5.0)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 200)   # yaw_rate=None: no heading
    sweeps = [d for d in recs if d.rung_label == "sweep360"]
    assert all(d.sweep_rad == 0.0 for d in sweeps), "no heading => no angle closed"
    assert 45 <= len(sweeps) <= 55, "5.0s timeout at dt=0.1 is ~50 ticks"
    assert any(d.state == GIVE_UP for d in recs)


def test_fresh_pose_exits_recovery_and_hands_back():
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, exit_confirm_poses=2)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 10 + [True] * 5)
    assert recs[-1].state == NOMINAL
    assert recs[-1].active is False and recs[-1].command is None


def test_single_flicker_does_not_resume_flight():
    """One lone detection is not proof we relocalized: stay stopped.

    AprilTag detection flickers; resuming on a single frame would hand a
    half-recovered drone back to the follower on a pose that is about to vanish.
    """
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, exit_confirm_poses=3)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 10 + [True] + [False] * 5)
    flicker = recs[11]                       # the single visible tick
    assert flicker.active is True, "one frame must not release cmd_vel"
    assert flicker.command.x == 0.0, "and must not keep driving the rung"
    started = next(i for i, d in enumerate(recs) if d.active)
    assert all(d.state != NOMINAL for d in recs[started:]), \
        "once recovery owns the drone, one frame must not hand it back"


def _lose_then_return_at(rec, hz, seconds, *, dt=0.05, lost_s=2.0):
    """Lose localization, then bring it back at a STEADY ``hz``.

    The other harness delivers at most one pose per tick, i.e. only ever a
    control-rate stream; this one models the real thing -- localization
    publishing at its own rate, independent of the control loop -- which is the
    only way to exercise a stream that limps back rather than returning all at
    once. ``lost_s`` of silence first, so recovery is genuinely engaged before
    the stream returns and the states below are all post-takeover.

    Returns the states observed from the moment the stream starts returning.
    """
    t, last_rx, count = 0.0, 0.0, 1
    rec.update(0.0, dt, count)                  # bootstrap: one pose has landed
    for _ in range(int(lost_s / dt)):           # ...then nothing at all
        t += dt
        rec.update(t - last_rx, dt, count)

    states, next_pose = [], t
    for _ in range(int(seconds / dt)):
        t += dt
        if t >= next_pose:
            count += 1
            last_rx = t
            next_pose = t + 1.0 / hz
        states.append(rec.update(t - last_rx, dt, count).state)
        if states[-1] == GIVE_UP:
            break
    return states


@pytest.mark.parametrize("hz", [1.2, 1.5, 2.0, 3.0, 5.0, 7.0])
def test_localization_returning_slowly_still_hands_the_drone_back(hz):
    """A stream limping back at 1-7 Hz must end the recovery.

    The trap: voiding the hand-back credit on every stale tick makes the exit
    require a rate above 1/stale_s (3.3 Hz at the defaults) -- so an AprilTag seen
    obliquely or half-occluded, which is exactly what this recovery is for, would
    leave the drone stopped for ever with the follower passive and nothing to
    clear it but a node restart.

    Below ~3 Hz the pose genuinely IS stale between arrivals, so the drone still
    stops in the gaps; what must never happen is that it stops for good.
    """
    states = _lose_then_return_at(LostLocalizationRecovery(), hz, 60.0)
    assert NOMINAL in states, (
        "localization came back at %.1f Hz and the drone never flew again -- "
        "recovery held it in %s for the whole 60s" % (hz, sorted(set(states))))
    assert GIVE_UP not in states


def test_never_lands_while_poses_are_still_arriving():
    """Landing is for "totally lost", not "slow".

    Even at a rate too low to hand back cleanly, a pose that lands means we know
    where we are -- so the ladder must never run to exhaustion and commit the
    drone to a blind landing.
    """
    for hz in (0.8, 1.0):
        states = _lose_then_return_at(LostLocalizationRecovery(), hz, 120.0)
        assert GIVE_UP not in states, (
            "landed at %.1f Hz even though poses were still arriving" % hz)


def test_a_pose_mid_ladder_stops_the_blind_manoeuvre_at_once():
    """A pose means we are not lost any more: stop reversing immediately.

    And the next escalation must start from the TOP, at ladder_s -- not resume
    the rung it was on the moment the pose ages past stale_s.
    """
    p = LostLocalizationParams(stale_s=0.3, ladder_s=1.0, exit_confirm_poses=5,
                               back_duration_s=3.0)
    rec = LostLocalizationRecovery(p)
    plant = {"age": 0.0, "count": 0, "yaw": 0.0}
    recs = _drive(rec, [True] + [False] * 15, state=plant)
    assert recs[-1].rung_label == "back#1" and recs[-1].command.x < 0

    on_pose = _drive(rec, [True], state=plant)[0]
    assert on_pose.command.x == 0.0, "a pose must stop the blind reverse at once"
    assert on_pose.state == HOLD, "and drop out of the ladder"

    # It must now take a FULL ladder_s of silence to fly blind again, not stale_s.
    after = _drive(rec, [False] * 8, state=plant)      # 0.8s: past stale, not ladder
    assert all(d.command.x == 0.0 for d in after), \
        "a 0.3s dropout resumed a blind manoeuvre -- the two-tier rule is broken"


def test_give_up_survives_a_reset():
    """reset() must not un-commit a land already under way."""
    p = LostLocalizationParams(back_repeats=0, climb_enabled=False,
                               turn_enabled=False)
    rec = LostLocalizationRecovery(p)
    _drive(rec, [True] + [False] * 20)
    assert rec.state == GIVE_UP
    rec.reset()
    assert _drive(rec, [True] * 5)[-1].state == GIVE_UP


def test_recovery_restarts_from_the_top_after_a_relocalization():
    """A relocalization ends the episode; the next dropout is a new problem."""
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, exit_confirm_poses=2,
                               back_duration_s=0.5, dwell_s=0.5)
    rec = LostLocalizationRecovery(p)
    plant = {"age": 0.0, "count": 0, "yaw": 0.0}   # one plant across all phases
    _drive(rec, [True] + [False] * 20, state=plant)      # deep into the ladder
    assert rec.state == LADDER
    _drive(rec, [True] * 4, state=plant)                 # tag comes back
    assert rec.state == NOMINAL
    recs = _drive(rec, [False] * 12, state=plant)        # and dies again
    labels = [d.rung_label for d in recs if d.rung_label]
    assert labels[0] == "back#1", "a new dropout starts at the first rung"


def test_ladder_terminates_and_never_hangs():
    """The ladder is finite and every rung is time-capped, so a permanently
    dead localization always reaches GIVE_UP rather than flying forever."""
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, turn_timeout_s=20.0,
                               turn_rate=0.4)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 2000)
    assert recs[-1].state == GIVE_UP
    assert recs[-1].give_up is True
    assert recs[-1].command.x == 0.0 and recs[-1].command.yaw_rate == 0.0


def test_give_up_is_sticky_so_a_late_tag_cannot_abort_the_land():
    """Landing is irreversible: once committed, a re-acquired tag must not
    hand a half-landed drone back to the follower."""
    p = LostLocalizationParams(stale_s=0.3, ladder_s=0.5, back_repeats=0,
                               climb_enabled=False, turn_enabled=False)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 10)
    assert recs[-1].state == GIVE_UP, "an empty ladder gives up at once"
    after = _drive(rec, [True] * 20)         # tag returns
    assert all(d.state == GIVE_UP and d.give_up is True for d in after)


def test_empty_ladder_gives_up_rather_than_flying_blind():
    p = LostLocalizationParams(back_repeats=0, climb_enabled=False,
                               turn_enabled=False)
    assert build_ladder(p) == ()
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 20)
    assert recs[-1].state == GIVE_UP


def test_disabling_climb_removes_its_settle_too():
    """A disabled stage must not leave two settles back to back."""
    p = LostLocalizationParams(climb_enabled=False)
    kinds = [r.kind for r in build_ladder(p)]
    assert CLIMB not in kinds
    assert kinds == [BACK, "stop", BACK, "stop", TURN]


# ── params validation ───────────────────────────────────────────────
def test_ladder_threshold_must_exceed_the_stop_threshold():
    with pytest.raises(ValueError, match="must be > stale_s"):
        LostLocalizationParams(stale_s=1.0, ladder_s=0.5)


def test_turn_timeout_below_the_sweep_time_is_rejected():
    """A timeout under the nominal sweep time would silently truncate the 360."""
    with pytest.raises(ValueError, match="could never complete"):
        LostLocalizationParams(turn_rate=0.3, turn_timeout_s=5.0)


def test_turn_direction_must_be_a_side():
    with pytest.raises(ValueError, match="turn_dir"):
        LostLocalizationParams(turn_dir=0)
