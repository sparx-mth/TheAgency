"""Tests for the persist prelude -- finishing the move the dropout interrupted.

The ladder's job is to search from a standstill when we are thoroughly lost. The
persist stage's job is the opposite and comes first: in the instant the tag leaves
the frame we still know what the navigator was doing, and the commonest dropouts
are CAUSED by that move and undone by finishing or reversing it.

What is locked in here:

  * a mid-turn dropout keeps turning, the same way, at the navigator's own rate --
    not the sweep's rate and not the sweep's direction;
  * a mid-advance dropout does the opposite and gives the metres back, capped to a
    reverse speed that has been flown before, and never flies forwards;
  * both branches end stationary, because a still camera is what re-acquires a tag;
  * the context is read once, at the dropout, and cannot be rewritten mid-prelude;
  * no context (or the stage off) is EXACTLY the old behaviour -- straight to the
    stop -- so the prelude can never be what breaks a recovery;
  * the prelude delays the ladder but never replaces it.
"""
import pytest

from sparx_agency.core.planning.lost_localization import (
    BACK,
    FORWARD,
    HOLD,
    NOMINAL,
    PERSIST,
    STOP,
    TURN,
    TURNING,
    UNKNOWN,
    LostLocalizationParams,
    LostLocalizationRecovery,
    MotionContext,
    build_ladder,
    build_persist,
)

DT = 0.1


def _drive(rec, tag_seq, ctx=None, *, dt=DT, yaw_rate=None):
    """Feed a tag-visibility sequence through the same toy plant the ladder uses.

    ``ctx`` is either one :class:`MotionContext` used on every tick or a list of
    them, one per tick, so a test can change what the node reports mid-episode.
    ``yaw_rate`` non-None integrates a heading from the commanded yaw and feeds it
    back, standing in for the platform's localization-independent bearing.
    """
    s = {"age": 0.0, "count": 0, "yaw": 0.0}
    recs = []
    for i, visible in enumerate(tag_seq):
        if visible:
            s["age"] = 0.0
            s["count"] += 1
        else:
            s["age"] += dt
        dec = rec.update(s["age"], dt, s["count"],
                         yaw=None if yaw_rate is None else s["yaw"],
                         context=ctx[i] if isinstance(ctx, list) else ctx)
        recs.append(dec)
        if yaw_rate is not None and dec.command is not None:
            s["yaw"] += dec.command.yaw_rate * dt
    return recs


def _phases(recs):
    """The rung labels (or states) in order, with consecutive repeats collapsed."""
    out = []
    for d in recs:
        name = d.rung_label or d.state
        if not out or out[-1] != name:
            out.append(name)
    return out


def _labelled(recs, label):
    return [d for d in recs if d.rung_label == label]


# ── The table: what each context asks for ───────────────────────────
def test_unknown_context_has_no_prelude():
    """Nothing to finish => the plain stop, exactly as before this stage."""
    assert build_persist(LostLocalizationParams(), MotionContext.unknown()) == ()


def test_disabling_the_stage_removes_it_entirely():
    p = LostLocalizationParams(persist_enabled=False)
    assert build_persist(p, MotionContext.turning(0.7)) == ()
    assert build_persist(p, MotionContext.forward(0.3)) == ()


def test_a_turn_continues_at_the_navigators_own_rate():
    """Verbatim, not re-derived: the platform holds its last command, so
    re-sending the value already flying continues the rotation seamlessly."""
    p = LostLocalizationParams()
    rungs = build_persist(p, MotionContext.turning(0.7))
    assert rungs[0].kind == TURN
    assert rungs[0].rate == pytest.approx(0.7)
    assert rungs[0].rate != pytest.approx(p.turn_rate * p.turn_dir), \
        "the prelude must not inherit the SWEEP's rate -- it is a different move"


def test_a_turn_keeps_its_direction():
    for rate in (0.7, -0.7):
        rungs = build_persist(LostLocalizationParams(), MotionContext.turning(rate))
        assert rungs[0].rate == pytest.approx(rate)


def test_an_advance_retreats_then_looks():
    rungs = build_persist(LostLocalizationParams(), MotionContext.forward(0.25))
    assert [r.kind for r in rungs] == [BACK, STOP]
    assert rungs[0].rate == pytest.approx(0.25)


def test_the_retreat_is_capped_by_the_blind_retreat_speed():
    """Reversing goes into space nothing is looking at, so it is never faster
    than the speed the ladder's own blind retreat was tuned to."""
    p = LostLocalizationParams()
    rungs = build_persist(p, MotionContext.forward(10.0))
    assert rungs[0].rate == pytest.approx(p.back_speed)


def test_every_prelude_ends_stationary():
    """A still camera is what re-acquires a tag; without this a persist-turn
    would run straight into the ladder's first back-up having never looked."""
    p = LostLocalizationParams()
    for ctx in (MotionContext.turning(0.7), MotionContext.forward(0.3)):
        rungs = build_persist(p, ctx)
        assert rungs[-1].kind == STOP
        assert rungs[-1].duration_s == p.persist_settle_s


def test_only_the_sweep_ends_on_angle():
    """The 360 is a search and is done when it has looked everywhere. A persist
    turn is a bounded continuation of a move and ends on its clock."""
    p = LostLocalizationParams()
    assert build_persist(p, MotionContext.turning(0.7))[0].target_rad is None
    sweep = [r for r in build_ladder(p) if r.kind == TURN]
    assert sweep and sweep[0].target_rad == pytest.approx(p.turn_target_rad)


# ── The context itself ──────────────────────────────────────────────
def test_a_stop_is_not_an_intent_to_continue():
    for kind in (TURNING, FORWARD):
        with pytest.raises(ValueError):
            MotionContext(kind, 0.0)


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        MotionContext("drifting", 1.0)


def test_unknown_needs_no_rate():
    assert MotionContext.unknown().kind == UNKNOWN


# ── Running it: the state machine ───────────────────────────────────
def test_a_mid_turn_dropout_keeps_turning_the_same_way():
    """The headline case: the tag left the frame because we rotated it out, so
    the next one is very often already swinging in. Keep going."""
    rec = LostLocalizationRecovery()
    recs = _drive(rec, [True] + [False] * 8, MotionContext.turning(0.7))

    turning = _labelled(recs, "persist-turn")
    assert turning, "a mid-turn dropout must keep turning"
    assert all(d.state == PERSIST for d in turning)
    assert all(d.command.yaw_rate == pytest.approx(0.7) for d in turning)
    assert all(d.command.x == 0.0 for d in turning), "one axis at a time"


def test_a_mid_turn_dropout_does_not_stop_dead():
    """Stopping strands the camera pointed at the one heading we know is empty."""
    rec = LostLocalizationRecovery()
    recs = _drive(rec, [True] + [False] * 8, MotionContext.turning(0.7))
    first = next(d for d in recs if d.active)
    assert first.command.yaw_rate != 0.0


def test_a_mid_advance_dropout_gives_the_metres_back():
    """Losing a tag while advancing means we advanced INTO something."""
    rec = LostLocalizationRecovery()
    recs = _drive(rec, [True] + [False] * 8, MotionContext.forward(0.25))

    back = _labelled(recs, "persist-back")
    assert back, "a mid-advance dropout must retreat"
    assert all(d.command.x == pytest.approx(-0.25) for d in back)
    assert all(d.command.yaw_rate == 0.0 for d in back)


def test_the_retreat_never_flies_forwards():
    """Whatever sign the context carried, a retreat retreats."""
    rec = LostLocalizationRecovery()
    for rate in (0.25, -0.25):
        rec.reset()
        recs = _drive(rec, [True] + [False] * 8, MotionContext.forward(rate))
        assert all(d.command.x < 0.0 for d in _labelled(recs, "persist-back"))


def test_the_prelude_ends_still_and_then_the_ladder_runs():
    """Delayed, never replaced: if finishing the move did not help, escalate."""
    rec = LostLocalizationRecovery()
    recs = _drive(rec, [True] + [False] * 40, MotionContext.turning(0.7))
    phases = _phases(recs)

    assert phases[:3] == [NOMINAL, "persist-turn", "persist-look"]
    assert all(d.command.yaw_rate == 0.0 and d.command.x == 0.0
               for d in _labelled(recs, "persist-look"))
    assert "back#1" in phases, "the ladder must still run when the prelude fails"


def test_no_context_is_exactly_the_old_behaviour():
    """The regression guard: a caller that knows nothing gets the plain stop."""
    rec = LostLocalizationRecovery()
    recs = _drive(rec, [True] + [False] * 6)
    first = next(d for d in recs if d.active)
    assert first.state == HOLD
    assert first.command.x == 0.0 and first.command.yaw_rate == 0.0
    assert _phases(recs)[:2] == [NOMINAL, HOLD]


def test_a_tag_during_the_prelude_stops_the_turn_at_once():
    """The prelude exists to bring a tag back. The moment one lands it has
    succeeded, and holding still is what keeps it in frame while the exit
    debounces -- turning on would sweep it straight back out."""
    rec = LostLocalizationRecovery()
    recs = _drive(rec, [True] + [False] * 4 + [True] + [False] * 3,
                  MotionContext.turning(0.7))
    after_tag = recs[5:]
    assert all(d.command is None or d.command.yaw_rate == 0.0 for d in after_tag)
    assert not any(d.rung_label == "persist-turn" for d in after_tag)


def test_the_context_is_latched_at_the_dropout():
    """It describes the past. Once recovery owns cmd_vel the only command anyone
    can observe is recovery's own, so re-reading it would feed the stage its own
    output -- the back-up rung would look like 'we were flying backwards'."""
    rec = LostLocalizationRecovery()
    ctx = ([MotionContext.turning(0.7)] * 4
           + [MotionContext.forward(0.3)] * 10)   # a lie, arriving too late
    recs = _drive(rec, [True] + [False] * 13, ctx)

    assert _labelled(recs, "persist-turn"), "the turn latched at the dropout wins"
    assert not _labelled(recs, "persist-back")


def test_a_persist_turn_is_not_cut_short_by_the_sweep_target():
    """Feed back a heading that races past a tiny sweep target: the sweep would
    stop, the prelude must not -- it is a fixed continuation, not a search."""
    p = LostLocalizationParams(turn_target_rad=0.01, turn_timeout_s=40.0)
    rec = LostLocalizationRecovery(p)
    recs = _drive(rec, [True] + [False] * 8, MotionContext.turning(0.7),
                  yaw_rate=0.7)
    turning = _labelled(recs, "persist-turn")
    assert len(turning) >= int(p.persist_turn_s / DT) - 1


def test_the_prelude_never_delays_the_land_indefinitely():
    """Whatever runs in front of it, the ladder still terminates in GIVE_UP."""
    rec = LostLocalizationRecovery(LostLocalizationParams(
        back_repeats=1, climb_enabled=False, turn_enabled=False, dwell_s=0.2))
    recs = _drive(rec, [True] + [False] * 60, MotionContext.forward(0.25))
    assert recs[-1].give_up
