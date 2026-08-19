"""Tests for the Amdahl gate.

These are arithmetic tests with hand-checkable numbers on purpose: every
expected value below can be recomputed on paper from ``s * (1 - 1/f)``, so a
failure points at the formula rather than at a fixture.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.common.trt_optimizer.amdahl import (
    max_speedup,
    rank,
    share,
    speedup_with,
    worth_converting,
)
from sparx_agency.tasks.common.trt_optimizer.spec import (
    Cadence,
    Component,
    Exportability,
    Plan,
)


def _component(name, latency_ms, calls=1.0, cadence=Cadence.PER_FRAME,
               exportability=Exportability.CLEAN, reason=""):
    """A profiled (or, with latency_ms=None, unprofiled) inventory entry."""
    return Component(name=name, latency_ms=latency_ms,
                     calls_per_decision=calls, cadence=cadence,
                     exportability=exportability, reason=reason)


def _plan(*components):
    """A plan whose decision is exactly the components handed to it."""
    return Plan(model="unit_test_model", components=list(components))


# --------------------------------------------------------------------------
# share
# --------------------------------------------------------------------------

def test_share_is_the_fraction_of_the_decision():
    vision = _component("vision", 25.0)
    plan = _plan(vision, _component("policy", 75.0))
    assert share(vision, plan) == pytest.approx(0.25)


def test_share_folds_in_calls_per_decision():
    denoiser = _component("denoiser", 3.0, calls=20.0,
                          cadence=Cadence.PER_STEP)
    plan = _plan(denoiser, _component("encoder", 40.0))
    # 60 ms of 100 ms, not 3 ms of 43 ms.
    assert share(denoiser, plan) == pytest.approx(0.6)


def test_shares_sum_to_one():
    parts = [_component("a", 10.0), _component("b", 30.0),
             _component("c", 60.0)]
    plan = _plan(*parts)
    assert sum(share(c, plan) for c in parts) == pytest.approx(1.0)


def test_share_raises_on_an_unprofiled_plan():
    vision = _component("vision", 25.0)
    plan = _plan(vision, _component("policy", None))
    with pytest.raises(ValueError) as excinfo:
        share(vision, plan)
    assert "policy" in str(excinfo.value)


def test_share_raises_for_the_unprofiled_component_itself():
    policy = _component("policy", None)
    plan = _plan(policy)
    with pytest.raises(ValueError):
        share(policy, plan)


def test_share_raises_for_a_component_outside_the_plan():
    plan = _plan(_component("vision", 25.0))
    with pytest.raises(ValueError) as excinfo:
        share(_component("stranger", 5.0), plan)
    assert "stranger" in str(excinfo.value)


def test_share_raises_on_a_zero_length_decision():
    idle = _component("idle", 0.0)
    plan = _plan(idle)
    with pytest.raises(ValueError):
        share(idle, plan)


# --------------------------------------------------------------------------
# max_speedup -- the ceiling
# --------------------------------------------------------------------------

def test_a_ten_percent_component_can_never_give_more_than_1_11x():
    small = _component("small", 10.0)
    plan = _plan(small, _component("rest", 90.0))
    ceiling = max_speedup(plan, ["small"])
    assert ceiling == pytest.approx(1.0 / 0.9)
    assert ceiling < 1.12


def test_ceiling_over_several_components_uses_their_summed_share():
    plan = _plan(_component("a", 30.0), _component("b", 20.0),
                 _component("c", 50.0))
    assert max_speedup(plan, ["a", "b"]) == pytest.approx(2.0)


def test_ceiling_ignores_duplicate_names():
    plan = _plan(_component("a", 30.0), _component("rest", 70.0))
    assert max_speedup(plan, ["a", "a", "a"]) == pytest.approx(1.0 / 0.7)


def test_ceiling_is_one_when_nothing_is_freed():
    plan = _plan(_component("a", 30.0), _component("rest", 70.0))
    assert max_speedup(plan, []) == pytest.approx(1.0)


def test_ceiling_is_infinite_when_the_whole_decision_is_freed():
    plan = _plan(_component("a", 30.0), _component("b", 70.0))
    assert max_speedup(plan, ["a", "b"]) == float("inf")


def test_ceiling_raises_on_an_unknown_component():
    plan = _plan(_component("a", 100.0))
    with pytest.raises(ValueError):
        max_speedup(plan, ["typo"])


def test_ceiling_raises_on_an_unprofiled_plan():
    plan = _plan(_component("a", 10.0), _component("b", None))
    with pytest.raises(ValueError):
        max_speedup(plan, ["a"])


# --------------------------------------------------------------------------
# speedup_with -- the realized number
# --------------------------------------------------------------------------

def test_speedup_with_on_a_multi_component_plan():
    plan = _plan(_component("vision", 60.0), _component("policy", 30.0),
                 _component("glue", 10.0))
    # 60/3 + 30/2 + 10 = 20 + 15 + 10 = 45 ms out of 100 ms.
    assert speedup_with(plan, {"vision": 3.0, "policy": 2.0}) == pytest.approx(
        100.0 / 45.0)


def test_unlisted_components_keep_factor_one():
    plan = _plan(_component("vision", 50.0), _component("policy", 50.0))
    assert speedup_with(plan, {}) == pytest.approx(1.0)
    assert speedup_with(plan, {"vision": 2.0}) == pytest.approx(100.0 / 75.0)


def test_speedup_with_approaches_the_amdahl_ceiling():
    plan = _plan(_component("vision", 40.0), _component("rest", 60.0))
    huge = speedup_with(plan, {"vision": 1e9})
    assert huge == pytest.approx(max_speedup(plan, ["vision"]), rel=1e-6)
    assert huge < max_speedup(plan, ["vision"]) + 1e-6


def test_speedup_with_reports_a_regression_below_one():
    plan = _plan(_component("vision", 50.0), _component("rest", 50.0))
    # A "conversion" that halved the speed of half the decision.
    assert speedup_with(plan, {"vision": 0.5}) == pytest.approx(100.0 / 150.0)


def test_speedup_with_raises_on_a_non_positive_factor():
    plan = _plan(_component("vision", 50.0), _component("rest", 50.0))
    with pytest.raises(ValueError):
        speedup_with(plan, {"vision": 0.0})


def test_speedup_with_raises_on_an_unknown_component():
    plan = _plan(_component("vision", 100.0))
    with pytest.raises(ValueError):
        speedup_with(plan, {"vison": 2.0})


def test_speedup_with_raises_on_an_unprofiled_plan():
    plan = _plan(_component("vision", 50.0), _component("policy", None))
    with pytest.raises(ValueError):
        speedup_with(plan, {"vision": 2.0})


# --------------------------------------------------------------------------
# worth_converting -- the five rules, each in isolation
# --------------------------------------------------------------------------

def test_rule_1_cold_cadence_is_rejected_however_slow_it_is():
    text = _component("text_encoder", 400.0, cadence=Cadence.ONCE_PER_EPISODE)
    plan = _plan(text, _component("policy", 10.0))
    ok, why = worth_converting(text, plan)
    assert ok is False
    assert "once_per_episode" in why
    assert "steady-state" in why


def test_rule_1_fires_even_when_the_plan_is_unprofiled():
    # The point of checking cadence first: a share threshold on an unprofiled
    # cold component has nothing to divide by.
    loader = _component("weight_loader", None,
                        cadence=Cadence.ONCE_PER_PROCESS)
    plan = _plan(loader, _component("policy", None))
    ok, why = worth_converting(loader, plan)
    assert ok is False
    assert "once_per_process" in why


def test_rule_2_hostile_export_quotes_the_recorded_blocker():
    blocker = "autoregressive decode loop with a KV cache"
    llm = _component("llm_decoder", 80.0, exportability=Exportability.HOSTILE,
                     reason=blocker)
    plan = _plan(llm, _component("rest", 20.0))
    ok, why = worth_converting(llm, plan)
    assert ok is False
    assert blocker in why


def test_rule_3_share_below_the_floor_states_the_measured_share():
    tiny = _component("layernorm_glue", 2.0)
    plan = _plan(tiny, _component("rest", 98.0))
    ok, why = worth_converting(tiny, plan)
    assert ok is False
    assert "2.0%" in why


def test_rule_4_gain_below_the_floor_states_the_computed_gain():
    # 10% share at an assumed 1.2x returns 0.10 * (1 - 1/1.2) = 1.67% < 2%.
    warm = _component("warm", 10.0)
    plan = _plan(warm, _component("rest", 90.0))
    ok, why = worth_converting(warm, plan, assumed_speedup=1.2)
    assert ok is False
    assert "1.7%" in why


def test_rule_5_accepts_a_dominant_component_and_quotes_the_projection():
    # 40% share at an assumed 3x returns 0.40 * (1 - 1/3) = 26.7% end to end.
    vision = _component("vision_tower", 40.0)
    plan = _plan(vision, _component("rest", 60.0))
    ok, why = worth_converting(vision, plan)
    assert ok is True
    assert "40.0%" in why
    assert "26.7%" in why


def test_rules_are_ordered_cadence_before_exportability():
    cold_and_hostile = _component("text_encoder", 400.0,
                                  cadence=Cadence.ONCE_PER_EPISODE,
                                  exportability=Exportability.HOSTILE,
                                  reason="sampling loop")
    plan = _plan(cold_and_hostile, _component("policy", 10.0))
    ok, why = worth_converting(cold_and_hostile, plan)
    assert ok is False
    assert "once_per_episode" in why
    assert "sampling loop" not in why


def test_rules_are_ordered_exportability_before_share():
    hostile_and_tiny = _component("sampler", 1.0,
                                  exportability=Exportability.HOSTILE,
                                  reason="multinomial sampling")
    plan = _plan(hostile_and_tiny, _component("rest", 99.0))
    ok, why = worth_converting(hostile_and_tiny, plan)
    assert ok is False
    assert "multinomial sampling" in why


def test_a_warm_component_in_an_unprofiled_plan_raises():
    vision = _component("vision", 40.0)
    plan = _plan(vision, _component("policy", None))
    with pytest.raises(ValueError):
        worth_converting(vision, plan)


def test_thresholds_are_tunable():
    small = _component("small", 3.0)
    plan = _plan(small, _component("rest", 97.0))
    assert worth_converting(small, plan)[0] is False
    ok, why = worth_converting(small, plan, min_share=0.01,
                               min_end_to_end_gain=0.001)
    assert ok is True
    assert "3.0%" in why


def test_a_non_positive_assumed_speedup_raises():
    vision = _component("vision", 40.0)
    plan = _plan(vision, _component("rest", 60.0))
    with pytest.raises(ValueError):
        worth_converting(vision, plan, assumed_speedup=0.0)


# --------------------------------------------------------------------------
# rank
# --------------------------------------------------------------------------

def test_rank_orders_by_decision_cost_descending():
    plan = _plan(_component("a", 10.0), _component("c", 50.0),
                 _component("b", 30.0))
    assert [c.name for c in rank(plan)] == ["c", "b", "a"]


def test_rank_puts_unprofiled_components_last():
    plan = _plan(_component("unknown", None), _component("small", 5.0),
                 _component("big", 50.0))
    assert [c.name for c in rank(plan)] == ["big", "small", "unknown"]


def test_rank_does_not_mutate_the_inventory():
    plan = _plan(_component("a", 10.0), _component("b", 50.0))
    before = [c.name for c in plan.components]
    rank(plan)
    assert [c.name for c in plan.components] == before


def test_a_cheap_per_step_component_outranks_an_expensive_per_frame_one():
    denoiser = _component("denoiser", 3.0, calls=20.0,
                          cadence=Cadence.PER_STEP)
    encoder = _component("encoder", 20.0, cadence=Cadence.PER_FRAME)
    plan = _plan(encoder, denoiser)
    # 3 ms x 20 calls = 60 ms per decision beats one 20 ms call.
    assert [c.name for c in rank(plan)] == ["denoiser", "encoder"]


def test_rank_of_an_empty_plan_is_empty():
    assert rank(Plan(model="empty")) == []


def test_a_sole_component_freed_outright_reports_an_infinite_overall():
    sole = _component("everything", 100.0)
    ok, why = worth_converting(sole, _plan(sole),
                               assumed_speedup=float("inf"))
    assert ok is True
    assert "inf" in why
