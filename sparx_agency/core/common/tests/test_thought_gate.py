"""Tests for the producer-side thought gate.

The gate is what makes narrating from inside a 20 Hz control loop safe, so the
tests pin the contract the nodes rely on: an unchanged conclusion is silent, a
changed one always speaks, slots never interfere, and a reset lets a restarted
story narrate itself again.
"""
import pytest

from sparx_agency.core.common.thought_gate import ThoughtGate


def test_first_thought_on_a_slot_always_emits():
    assert ThoughtGate().should_emit("motion", "Stopping to turn", now=0.0)


def test_unchanged_thought_is_silent():
    gate = ThoughtGate()
    gate.should_emit("motion", "Stopping to turn", now=0.0)
    assert not gate.should_emit("motion", "Stopping to turn", now=0.05)
    assert not gate.should_emit("motion", "Stopping to turn", now=99.0)


def test_changed_thought_emits():
    gate = ThoughtGate()
    gate.should_emit("motion", "Stopping to turn", now=0.0)
    assert gate.should_emit("motion", "Flying forward to waypoint 3", now=0.05)


def test_a_20hz_loop_emits_once_per_state_not_once_per_tick():
    gate = ThoughtGate()
    emitted = [text
               for i in range(60)
               for text in ["Stopping to turn" if i < 30 else "Flying forward"]
               if gate.should_emit("motion", text, now=i * 0.05)]
    assert emitted == ["Stopping to turn", "Flying forward"]


def test_slots_are_gated_independently():
    gate = ThoughtGate()
    assert gate.should_emit("motion", "Stopping to turn", now=0.0)
    # A different slot saying something else must not be suppressed by, nor
    # suppress, the motion slot.
    assert gate.should_emit("sensor", "No localization", now=0.0)
    assert not gate.should_emit("motion", "Stopping to turn", now=0.1)
    assert not gate.should_emit("sensor", "No localization", now=0.1)


def test_returning_to_a_previous_text_emits_again():
    gate = ThoughtGate()
    gate.should_emit("motion", "Flying forward", now=0.0)
    gate.should_emit("motion", "Stopping to turn", now=1.0)
    assert gate.should_emit("motion", "Flying forward", now=2.0)


def test_repeat_after_re_narrates_an_unchanged_thought():
    gate = ThoughtGate(repeat_after_s=5.0)
    assert gate.should_emit("sensor", "No localization", now=0.0)
    assert not gate.should_emit("sensor", "No localization", now=4.9)
    assert gate.should_emit("sensor", "No localization", now=5.0)
    assert not gate.should_emit("sensor", "No localization", now=9.9)
    assert gate.should_emit("sensor", "No localization", now=10.0)


def test_repeat_after_can_be_overridden_per_call():
    gate = ThoughtGate()                       # default: change-only
    assert gate.should_emit("sensor", "No localization", now=0.0)
    assert not gate.should_emit("sensor", "No localization", now=3.0)
    assert gate.should_emit("sensor", "No localization", now=3.0, repeat_after_s=2.0)


def test_should_emit_records_its_decision():
    # Asking is acting: a caller that asks twice for one tick must not be told
    # to narrate twice.
    gate = ThoughtGate()
    assert gate.should_emit("motion", "Stopping to turn", now=0.0)
    assert not gate.should_emit("motion", "Stopping to turn", now=0.0)


def test_reset_lets_one_slot_narrate_again():
    gate = ThoughtGate()
    gate.should_emit("motion", "Aligning to waypoint 1", now=0.0)
    gate.should_emit("sensor", "No localization", now=0.0)
    gate.reset("motion")
    assert gate.should_emit("motion", "Aligning to waypoint 1", now=1.0)
    assert not gate.should_emit("sensor", "No localization", now=1.0)


def test_reset_all_lets_every_slot_narrate_again():
    gate = ThoughtGate()
    gate.should_emit("motion", "Aligning to waypoint 1", now=0.0)
    gate.should_emit("sensor", "No localization", now=0.0)
    gate.reset()
    assert gate.should_emit("motion", "Aligning to waypoint 1", now=1.0)
    assert gate.should_emit("sensor", "No localization", now=1.0)


def test_reset_of_an_unknown_slot_is_harmless():
    ThoughtGate().reset("never-used")


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_repeat_after_is_rejected(bad):
    with pytest.raises(ValueError, match="must be positive"):
        ThoughtGate(repeat_after_s=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_per_call_override_is_rejected_too(bad):
    # Otherwise it silently disables the gate for that slot, which looks like a
    # broken gate rather than a bad argument.
    gate = ThoughtGate()
    with pytest.raises(ValueError, match="must be positive"):
        gate.should_emit("motion", "Stopping to turn", now=0.0,
                         repeat_after_s=bad)
