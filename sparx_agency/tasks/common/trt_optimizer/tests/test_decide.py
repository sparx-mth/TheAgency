"""The decision engine, exercised on the shape of a real dual-system VLA.

The fixture below is deliberately modelled on InternVLA-N1: a hostile
autoregressive System-2 VLM on a slow planning cadence, a small System-1 vision
encoder every frame, a denoiser looped many times inside one decision, a cheap
head, and an instruction encoder that runs once per episode. That last one is
the case the whole package exists to get right -- it is large and slow and must
still be left alone.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.common.trt_optimizer import decide
from sparx_agency.tasks.common.trt_optimizer.spec import (
    Cadence, Component, Exportability, GraphSpec, Plan,
)


class _FakeTarget(object):
    """Minimal stand-in for Target: decide() only reads these three things."""

    def __init__(self, precisions=("fp32", "fp16"), trt_version="11.1.0.106"):
        self._precisions = list(precisions)
        self.trt_version = trt_version
        self.strongly_typed = True

    def supported_precisions(self):
        return list(self._precisions)


def _plan():
    """A dual-system plan with every interesting cadence represented."""
    components = [
        Component(name="system2_vlm", params=8_292_000_000,
                  cadence=Cadence.PER_PLAN, calls_per_decision=0.125,
                  exportability=Exportability.HOSTILE,
                  reason="autoregressive generate() with a growing past_key_values "
                         "KV cache and a Python sampling loop",
                  latency_ms=700.0, dtype="bfloat16"),
        Component(name="instruction_encoder", params=120_000_000,
                  cadence=Cadence.ONCE_PER_EPISODE, calls_per_decision=0.0,
                  latency_ms=45.0, dtype="bfloat16"),
        Component(name="system1_encoder", params=44_000_000,
                  cadence=Cadence.PER_FRAME, calls_per_decision=1.0,
                  latency_ms=12.0, dtype="float32"),
        Component(name="system1_denoiser", params=28_000_000,
                  cadence=Cadence.PER_STEP, calls_per_decision=20.0,
                  latency_ms=1.5, dtype="float32"),
        Component(name="action_head", params=200_000,
                  cadence=Cadence.PER_FRAME, calls_per_decision=1.0,
                  latency_ms=0.20, dtype="float32"),
    ]
    graphs = [
        GraphSpec(key="system1_encoder", inputs={"rgb": (1, 3, 224, 224)},
                  outputs=["embed"], component="system1_encoder",
                  precision_sensitive=True),
        GraphSpec(key="system1_denoiser", inputs={"x": (32, 32, 3)},
                  outputs=["v"], component="system1_denoiser",
                  cadence=Cadence.PER_STEP, calls_per_decision=20.0),
    ]
    return Plan(model="dual_system", target_tag="test_sm120",
                components=components, graphs=graphs)


def _by_component(verdicts):
    out = {}
    for v in verdicts:
        out.setdefault(v.component, []).append(v)
    return out


def test_unprofiled_plan_is_refused_by_name():
    plan = _plan()
    plan.components[2].latency_ms = None
    with pytest.raises(ValueError) as excinfo:
        decide.decide(plan, _FakeTarget())
    message = str(excinfo.value)
    assert "system1_encoder" in message
    assert "Stage 2" in message or "baseline profile" in message


def test_once_per_episode_component_is_never_converted():
    """The headline rule: slow, large, and still not worth converting."""
    verdicts = _by_component(decide.decide(_plan(), _FakeTarget()))
    primary = verdicts["instruction_encoder"][0]
    assert primary.action == "cache_output"
    assert "once_per_episode" in primary.why
    assert primary.confidence == "high"


def test_autoregressive_component_routes_to_an_llm_runtime():
    verdicts = _by_component(decide.decide(_plan(), _FakeTarget()))
    actions = [v.action for v in verdicts["system2_vlm"]]
    assert "llm_runtime" in actions
    why = verdicts["system2_vlm"][0].why
    assert "KV cache" in why or "kv cache" in why.lower()
    assert "TensorRT-LLM" in why or "llama.cpp" in why


def test_hot_loop_gets_both_an_engine_and_a_step_count_lever():
    verdicts = _by_component(decide.decide(_plan(), _FakeTarget()))
    actions = [v.action for v in verdicts["system1_denoiser"]]
    assert "trt_fp16" in actions
    assert "reduce_calls" in actions
    lever = [v for v in verdicts["system1_denoiser"] if v.action == "reduce_calls"][0]
    assert "20" in lever.why
    assert "BEFORE the kernels" in lever.why


def test_precision_sensitive_graph_is_built_fp32_on_strong_typing():
    verdicts = _by_component(decide.decide(_plan(), _FakeTarget()))
    primary = verdicts["system1_encoder"][0]
    assert primary.action == "trt_fp32"
    assert "precision-sensitive" in primary.why
    assert "strongly typed" in primary.why


def test_a_dominated_component_falls_below_the_share_gate():
    """The Amdahl lesson, made concrete.

    Halve the System-1 encoder's cost and it drops under 5% of a decision that
    System 2 dominates -- at which point converting it is not worth the
    permanent numerical risk, however easy the export would be.
    """
    plan = _plan()
    plan.component("system1_encoder").latency_ms = 6.0
    verdicts = _by_component(decide.decide(plan, _FakeTarget()))
    primary = verdicts["system1_encoder"][0]
    assert primary.action == "leave_in_torch"
    assert "4." in primary.why           # its measured share, in percent


def test_tiny_component_is_left_alone_with_its_share_quoted():
    verdicts = _by_component(decide.decide(_plan(), _FakeTarget()))
    primary = verdicts["action_head"][0]
    assert primary.action == "leave_in_torch"
    assert "%" in primary.why


def test_convertible_lists_only_engine_builds():
    verdicts = decide.decide(_plan(), _FakeTarget())
    names = decide.convertible(verdicts)
    assert set(names) == {"system1_encoder", "system1_denoiser"}


def test_ceiling_is_bounded_by_amdahl_and_below_it():
    plan = _plan()
    verdicts = decide.decide(plan, _FakeTarget())
    projected, bound = decide.ceiling(plan, verdicts)
    assert 1.0 < projected <= bound
    # The System-2 VLM alone is ~87 ms of a ~123 ms decision, and it is not
    # convertible, so the whole exercise is capped well under 2x.
    assert bound < 2.5


def test_precision_ladder_reflects_the_toolchain_not_the_datasheet():
    assert decide.precision_ladder(_FakeTarget(("fp32", "fp16"))) == ["fp16", "fp32"]
    ladder = decide.precision_ladder(
        _FakeTarget(("fp32", "fp16", "int8", "fp8", "nvfp4")))
    assert ladder.index("int8") < ladder.index("fp8") < ladder.index("nvfp4")
    assert ladder[-1] == "fp32"


def test_ladder_falls_back_to_fp32_when_nothing_else_is_buildable():
    assert decide.precision_ladder(_FakeTarget(("fp32",))) == ["fp32"]


def test_fp32_is_always_the_last_rung_and_never_dropped():
    """A reduced-precision graph can be unreachable for a given network; an FP32
    engine still beats eager torch, so the race must never be left empty."""
    for supported in (("fp32", "fp16"), ("fp32", "fp16", "int8"),
                      ("fp32", "fp16", "int8", "fp8", "nvfp4")):
        ladder = decide.precision_ladder(_FakeTarget(supported))
        assert ladder[-1] == "fp32"
        assert ladder.count("fp32") == 1


def test_explain_groups_every_verdict():
    verdicts = decide.decide(_plan(), _FakeTarget())
    grouped = decide.explain(verdicts)
    assert sum(len(v) for v in grouped.values()) == len(verdicts)
    assert "llm_runtime" in grouped


# --------------------------------------------------------------------------
# Coverage: do the graphs match what the measurement asked for?
# --------------------------------------------------------------------------

def test_coverage_flags_a_convertible_component_with_no_graph():
    plan = _plan()
    plan.graphs = [g for g in plan.graphs if g.key != "system1_denoiser"]
    verdicts = decide.decide(plan, _FakeTarget())
    uncovered, unjustified = decide.coverage(plan, verdicts)
    assert "system1_denoiser" in uncovered
    assert unjustified == []
    notes = decide.coverage_notes(plan, verdicts)
    assert notes and "no exported graph" in notes[0]


def test_coverage_flags_a_graph_for_a_component_left_alone():
    plan = _plan()
    plan.graphs.append(GraphSpec(key="action_head", inputs={"t": (1, 3)},
                                 outputs=["a"], component="action_head"))
    verdicts = decide.decide(plan, _FakeTarget())
    uncovered, unjustified = decide.coverage(plan, verdicts)
    assert "action_head" in unjustified
    notes = decide.coverage_notes(plan, verdicts)
    assert any("leave alone" in n or "leave" in n for n in notes)


def test_coverage_is_silent_when_graphs_and_verdicts_agree():
    plan = _plan()
    verdicts = decide.decide(plan, _FakeTarget())
    assert decide.coverage(plan, verdicts) == ([], [])
    assert decide.coverage_notes(plan, verdicts) == []
