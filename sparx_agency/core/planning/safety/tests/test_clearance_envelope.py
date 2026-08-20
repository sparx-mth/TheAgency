"""The clearance envelope must transfer between worlds without being retuned.

The whole point of the class is that one configuration serves a 1.4 m warehouse
aisle and a 0.90 m hospital doorway. So the tests are written as those two
worlds, with the SAME envelope object, and the pass condition is that the
aircraft is left alone in both when it is flying what the planner gave it.
"""
import pytest

from sparx_agency.core.planning.safety.clearance_envelope import (
    ClearanceEnvelope, ClearanceEnvelopeConfig,
)

# One envelope, both worlds. Every test below constructs it from this.
WAREHOUSE_AISLE = 0.70      # half width of a 1.4 m aisle
HOSPITAL_DOOR = 0.45        # half width of a 0.90 m doorway


def _envelope(**overrides):
    return ClearanceEnvelope(ClearanceEnvelopeConfig(**overrides))


# ── the transfer property, which is the reason the class exists ──────────

def test_flying_the_plan_is_never_braked_in_either_world():
    """One config. The aircraft is exactly on its reference in both worlds."""
    envelope = _envelope()
    for clearance in (WAREHOUSE_AISLE, HOSPITAL_DOOR):
        budget = envelope.budget(clearance, clearance)
        assert budget.deficit_m == 0.0
        assert budget.speed_scale == 1.0
        assert not budget.breached
        assert not budget.hard_stop
        assert budget.reason == "on_clearance"


def test_a_constant_threshold_would_have_failed_the_doorway():
    """The failure this class replaces, stated as a test.

    A 0.55 m absolute veto -- the value this stack shipped -- vetoes a doorway
    pass the planner is entitled to make, because 0.45 m is all the doorway
    has. The envelope does not, and it does not because it is comparing against
    the plan rather than against a number chosen in a different building.
    """
    assert HOSPITAL_DOOR < 0.55                     # the constant vetoes it
    budget = _envelope().budget(HOSPITAL_DOOR, HOSPITAL_DOOR)
    assert budget.speed_scale == 1.0                # the envelope does not


def test_the_same_deficit_costs_the_same_in_both_worlds():
    """Being 0.15 m closer than planned is one failure, not two.

    This is the property that removes the per-world tuning: the response is a
    function of the margin spent, and the margin spent is measured in the same
    metres wherever the aircraft is.
    """
    envelope = _envelope()
    warehouse = envelope.budget(WAREHOUSE_AISLE, WAREHOUSE_AISLE - 0.15)
    hospital = envelope.budget(HOSPITAL_DOOR, HOSPITAL_DOOR - 0.15)
    assert warehouse.deficit_m == pytest.approx(hospital.deficit_m)
    assert warehouse.speed_scale == pytest.approx(hospital.speed_scale)


# ── the deficit response ─────────────────────────────────────────────────

def test_tracking_noise_inside_the_tolerance_does_not_bite():
    """A follower holding a curve to centimetres must not chatter the reflexes."""
    envelope = _envelope(tolerance_m=0.10)
    budget = envelope.budget(WAREHOUSE_AISLE, WAREHOUSE_AISLE - 0.08)
    assert budget.deficit_m == 0.0
    assert budget.speed_scale == 1.0


def test_speed_falls_with_the_margin_spent():
    envelope = _envelope(tolerance_m=0.10, deficit_span_m=0.25)
    scales = [envelope.budget(0.70, 0.70 - d).speed_scale
              for d in (0.10, 0.20, 0.30, 0.35)]
    assert scales == sorted(scales, reverse=True)
    assert scales[0] == pytest.approx(1.0)          # inside tolerance
    assert scales[-1] == pytest.approx(0.0, abs=1e-9)


def test_a_large_deficit_is_a_breach_and_not_merely_slow():
    envelope = _envelope(tolerance_m=0.10, breach_deficit_m=0.20)
    assert not envelope.budget(0.70, 0.55).breached      # deficit 0.05
    assert envelope.budget(0.70, 0.35).breached          # deficit 0.25


# ── the absolute floor, which is never relative to anything ──────────────

def test_the_hard_floor_stops_the_aircraft_whatever_the_plan_wanted():
    """A planner that routes inside the airframe does not get to be obeyed."""
    envelope = _envelope(hard_floor_m=0.30)
    budget = envelope.budget(0.25, 0.25)
    assert budget.hard_stop
    assert budget.speed_scale == 0.0
    assert budget.reason == "hard_floor"


def test_stopping_in_a_doorway_is_not_a_breach():
    """The loop this separation exists to end.

    In a 0.90 m opening the plan holds 0.45 m, so 0.15 m of drift reaches the
    floor. Reading that as a breach retreats the aircraft 1.3 m back out of the
    door it was passing, whereupon it replans the identical curve: 60 retreats
    in one measured hospital run. Stopping is right; reversing out is not.
    """
    envelope = _envelope(hard_floor_m=0.30, breach_deficit_m=0.20)
    budget = envelope.budget(HOSPITAL_DOOR, 0.29)
    assert budget.hard_stop
    assert not budget.breached


def test_reaching_the_floor_in_a_wide_aisle_IS_a_breach():
    """Same floor, same code, opposite verdict, because the plan is different.

    0.30 m of room where the reference had 0.70 m means the aircraft has thrown
    away 0.40 m of margin -- it is not passing anything, it has drifted into a
    wall, and backing out is exactly right.
    """
    envelope = _envelope(hard_floor_m=0.30, breach_deficit_m=0.20)
    budget = envelope.budget(WAREHOUSE_AISLE, 0.29)
    assert budget.hard_stop
    assert budget.breached


def test_a_plan_closer_than_the_floor_does_not_licence_flying_there():
    """The plan clearance is clamped UP, so the deficit is measured honestly.

    Without the clamp, a curve routed 0.20 m from a wall would make 0.20 m the
    reference and the aircraft would be told it has full speed while sitting
    inside its own radius.
    """
    envelope = _envelope(hard_floor_m=0.30)
    budget = envelope.budget(0.20, 0.32)
    assert budget.plan_clearance_m == pytest.approx(0.30)
    assert not budget.hard_stop
    assert budget.speed_scale == 1.0


# ── the unknown-clearance cases, which are the common ones in open space ─

def test_nothing_near_the_aircraft_is_never_a_reason_to_brake():
    envelope = _envelope()
    budget = envelope.budget(0.45, None)
    assert budget.speed_scale == 1.0
    assert budget.reason == "clear"
    assert not budget.breached


def test_an_unknown_plan_clearance_is_treated_as_open_space():
    """The dangerous default is the other one.

    Reading "unknown" as zero clearance would brake hardest exactly where the
    reference has so much room that nothing was found near it.
    """
    envelope = _envelope(open_clearance_m=0.90)
    budget = envelope.budget(None, 0.85)
    assert budget.plan_clearance_m == pytest.approx(0.90)
    assert budget.speed_scale == 1.0


def test_an_unknown_plan_clearance_still_brakes_a_genuinely_close_aircraft():
    envelope = _envelope(open_clearance_m=0.90, tolerance_m=0.10)
    budget = envelope.budget(None, 0.45)
    assert budget.deficit_m == pytest.approx(0.35)
    assert budget.speed_scale < 0.2


# ── the speed cap, which is a speed and not a fraction ───────────────────

def test_the_cap_never_scales_a_crawl_below_the_airframe_deadband():
    """An aircraft that cannot move cannot leave the situation being objected to."""
    envelope = _envelope(floor_speed=0.10)
    budget = envelope.budget(0.70, 0.40)            # deficit 0.20, scale 0.2
    assert envelope.speed_cap(budget, 0.12) == pytest.approx(0.10)


def test_the_cap_never_raises_the_planned_speed():
    envelope = _envelope()
    budget = envelope.budget(0.70, 0.70)
    assert envelope.speed_cap(budget, 0.25) == pytest.approx(0.25)


def test_a_hard_stop_caps_at_zero_regardless_of_the_floor_speed():
    envelope = _envelope(floor_speed=0.10, hard_floor_m=0.30)
    budget = envelope.budget(0.70, 0.28)
    assert envelope.speed_cap(budget, 0.25) == 0.0


def test_the_floor_speed_cannot_exceed_what_was_planned():
    """Slowing down must not become speeding up on a plan that asked for less."""
    envelope = _envelope(floor_speed=0.10)
    budget = envelope.budget(0.70, 0.35)
    assert envelope.speed_cap(budget, 0.05) <= 0.05
