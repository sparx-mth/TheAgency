"""Tests for the plan vocabulary: components, graphs, cadence and the plan.

Pure standard library, no fakes needed -- :mod:`..spec` has no dependencies at
all, which is the property that lets it be imported on a Jetson's system
interpreter and inside the Noetic container.

Two behaviours here are load-bearing rather than incidental and are pinned
deliberately:

* :meth:`Plan.decision_ms` is **all-or-nothing**. A partial sum would silently
  shrink the denominator every share and every Amdahl bound is divided by, so a
  single unprofiled component must poison the whole answer.
* :meth:`GraphSpec.validate` rejects a dynamic dimension *by name*. The shared
  engine runtime refuses a dynamic engine, and the error a human reads has to
  say which tensor to go and fix.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.common.trt_optimizer.spec import (
    ShapeProfile,
    ACTIONS,
    Cadence,
    Component,
    Exportability,
    GraphSpec,
    Plan,
    Verdict,
)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def vit_graph(**kwargs):
    """A well-formed, fully static GraphSpec."""
    fields = dict(key="navdp_encoder",
                  inputs={"rgb": (1, 3, 224, 224), "goal": (1, 3)},
                  outputs=["memory"],
                  component="model.vision_tower")
    fields.update(kwargs)
    return GraphSpec(**fields)


def profiled_plan():
    """A three-component plan where every component has a measurement."""
    return Plan(
        model="navdp",
        components=[
            Component(name="encoder", latency_ms=10.0),
            Component(name="denoiser", latency_ms=2.0,
                      cadence=Cadence.PER_STEP, calls_per_decision=8.0),
            Component(name="head", latency_ms=0.5),
        ],
    )


# --------------------------------------------------------------------------
# Cadence / Exportability vocabulary
# --------------------------------------------------------------------------

def test_cold_is_exactly_the_two_cold_cadences():
    """COLD is the gate decide.py checks before any timing, so it is pinned."""
    assert Cadence.COLD == (Cadence.ONCE_PER_PROCESS, Cadence.ONCE_PER_EPISODE)
    assert len(Cadence.COLD) == 2


@pytest.mark.parametrize("cadence", [Cadence.PER_PLAN, Cadence.PER_FRAME,
                                     Cadence.PER_STEP, Cadence.ON_DEMAND])
def test_hot_cadences_are_not_cold(cadence):
    assert cadence not in Cadence.COLD


def test_all_cadences_are_distinct_and_contain_the_cold_ones():
    assert len(set(Cadence.ALL)) == len(Cadence.ALL) == 6
    for cadence in Cadence.COLD:
        assert cadence in Cadence.ALL


def test_exportability_vocabulary():
    assert Exportability.ALL == (Exportability.CLEAN, Exportability.NEEDS_PATCH,
                                 Exportability.HOSTILE)
    assert len(set(Exportability.ALL)) == 3


def test_actions_are_unique_and_include_every_engine_build():
    assert len(set(ACTIONS)) == len(ACTIONS)
    for action in ("trt_fp16", "trt_fp32", "trt_int8", "llm_runtime",
                   "leave_in_torch", "cache_output", "reduce_calls"):
        assert action in ACTIONS


# --------------------------------------------------------------------------
# Component.decision_ms
# --------------------------------------------------------------------------

def test_decision_ms_is_none_while_unprofiled():
    assert Component(name="encoder").decision_ms is None


def test_decision_ms_is_the_measurement_for_a_once_per_frame_component():
    assert Component(name="encoder", latency_ms=12.5).decision_ms == 12.5


def test_decision_ms_multiplies_a_denoise_loop_by_its_step_count():
    """A 3 ms kernel run 20 times costs 60 ms of one decision, not 3."""
    denoiser = Component(name="denoiser", latency_ms=3.0,
                         cadence=Cadence.PER_STEP, calls_per_decision=20.0)
    assert denoiser.decision_ms == pytest.approx(60.0)


def test_decision_ms_amortizes_a_system_two_backbone_over_its_replan_period():
    backbone = Component(name="system2", latency_ms=400.0,
                         cadence=Cadence.PER_PLAN, calls_per_decision=0.125)
    assert backbone.decision_ms == pytest.approx(50.0)


def test_decision_ms_of_a_zero_call_component_is_zero_not_none():
    """Zero calls is a measured answer; None means unmeasured. Not the same."""
    assert Component(name="idle", latency_ms=9.0,
                     calls_per_decision=0.0).decision_ms == 0.0


# --------------------------------------------------------------------------
# Component.weight_bytes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype,width", [
    ("float64", 8), ("float32", 4), ("float", 4), ("bfloat16", 2),
    ("float16", 2), ("half", 2), ("int8", 1), ("uint8", 1), ("fp8", 1),
    ("int4", 0.5), ("nvfp4", 0.5),
])
def test_weight_bytes_for_every_known_dtype(dtype, width):
    component = Component(name="c", params=1000, dtype=dtype)
    assert component.weight_bytes() == int(1000 * width)


def test_weight_bytes_is_an_int_even_at_half_byte_widths():
    """int4 is 0.5 B/param; the answer is still a byte count, not a float."""
    value = Component(name="c", params=1001, dtype="int4").weight_bytes()
    assert isinstance(value, int)
    assert value == 500


def test_weight_bytes_honours_an_explicit_width_over_the_dtype():
    """The override is how memory_budget prices a plan at a target precision."""
    component = Component(name="c", params=1000, dtype="float32")
    assert component.weight_bytes(bytes_per_param=2) == 2000
    assert component.weight_bytes(bytes_per_param=0.5) == 500


def test_weight_bytes_of_a_zero_parameter_component_is_zero():
    assert Component(name="wrapper").weight_bytes() == 0


def test_default_component_is_a_hot_clean_float32_row():
    component = Component(name="c")
    assert component.cadence == Cadence.PER_FRAME
    assert component.calls_per_decision == 1.0
    assert component.exportability == Exportability.CLEAN
    assert component.dtype == "float32"
    assert component.latency_ms is None


# --------------------------------------------------------------------------
# GraphSpec
# --------------------------------------------------------------------------

def test_input_names_keep_the_declared_order():
    """The order is the positional argument order of the export wrapper."""
    spec = vit_graph(inputs={"rgb": (1, 3, 224, 224), "goal": (1, 3),
                             "state": (1, 8)})
    assert spec.input_names() == ["rgb", "goal", "state"]


def test_volume_is_the_element_count_of_one_input():
    assert vit_graph().volume("rgb") == 1 * 3 * 224 * 224
    assert vit_graph().volume("goal") == 3


def test_volume_of_a_scalar_shaped_input_is_one():
    assert vit_graph(inputs={"t": (1,)}).volume("t") == 1


def test_volume_raises_for_a_name_that_is_not_an_input():
    with pytest.raises(KeyError):
        vit_graph().volume("memory")


def test_validate_passes_a_fully_static_spec():
    assert vit_graph().validate() is None


def test_validate_rejects_a_spec_with_no_inputs():
    with pytest.raises(ValueError) as excinfo:
        vit_graph(inputs={}).validate()
    assert "no inputs" in str(excinfo.value)
    assert "navdp_encoder" in str(excinfo.value)


def test_validate_rejects_a_spec_with_no_outputs():
    with pytest.raises(ValueError) as excinfo:
        vit_graph(outputs=[]).validate()
    assert "no outputs" in str(excinfo.value)
    assert "navdp_encoder" in str(excinfo.value)


def test_validate_rejects_an_empty_shape():
    with pytest.raises(ValueError) as excinfo:
        vit_graph(inputs={"rgb": ()}).validate()
    assert "empty shape" in str(excinfo.value)
    assert "rgb" in str(excinfo.value)


@pytest.mark.parametrize("dim", [-1, 0])
def test_validate_names_the_tensor_carrying_a_dynamic_dimension(dim):
    """-1 is how a traced dynamic batch arrives; the message must say where."""
    with pytest.raises(ValueError) as excinfo:
        vit_graph(inputs={"rgb": (dim, 3, 224, 224)}).validate()
    message = str(excinfo.value)
    assert "rgb" in message
    assert "navdp_encoder" in message
    assert "ShapeProfile" in message   # the message names the fix


def test_validate_checks_every_input_not_only_the_first():
    with pytest.raises(ValueError) as excinfo:
        vit_graph(inputs={"rgb": (1, 3, 224, 224), "goal": (-1, 3)}).validate()
    assert "goal" in str(excinfo.value)


def test_graph_defaults_are_a_hot_opset_17_graph():
    spec = GraphSpec(key="k")
    assert spec.opset == 17
    assert spec.cadence == Cadence.PER_FRAME
    assert spec.calls_per_decision == 1.0
    assert spec.precision_sensitive is False


def test_graph_default_containers_are_not_shared_between_instances():
    first, second = GraphSpec(key="a"), GraphSpec(key="b")
    first.inputs["rgb"] = (1, 3)
    first.outputs.append("memory")
    assert second.inputs == {}
    assert second.outputs == []


# --------------------------------------------------------------------------
# Plan lookups
# --------------------------------------------------------------------------

def test_component_lookup_returns_the_named_component():
    plan = profiled_plan()
    assert plan.component("denoiser").latency_ms == 2.0


def test_component_lookup_returns_none_for_a_miss():
    assert profiled_plan().component("vision_tower") is None


def test_component_lookup_on_an_empty_plan_is_none():
    assert Plan(model="empty").component("anything") is None


def test_graph_lookup_returns_the_named_graph():
    plan = Plan(model="navdp", graphs=[vit_graph(), vit_graph(key="navdp_head")])
    assert plan.graph("navdp_head").key == "navdp_head"


def test_graph_lookup_returns_none_for_a_miss():
    plan = Plan(model="navdp", graphs=[vit_graph()])
    assert plan.graph("navdp_head") is None


# --------------------------------------------------------------------------
# Plan.decision_ms -- the all-or-nothing denominator
# --------------------------------------------------------------------------

def test_decision_ms_of_an_empty_plan_is_none():
    """No inventory is not a zero-length decision; it is an unknown one."""
    assert Plan(model="empty").decision_ms() is None


def test_decision_ms_sums_every_component_including_call_counts():
    assert profiled_plan().decision_ms() == pytest.approx(10.0 + 16.0 + 0.5)


def test_decision_ms_is_none_when_any_single_component_is_unprofiled():
    plan = profiled_plan()
    plan.components.append(Component(name="tokenizer"))
    assert plan.decision_ms() is None


def test_decision_ms_is_none_even_when_the_unprofiled_component_is_first():
    plan = profiled_plan()
    plan.components.insert(0, Component(name="tokenizer"))
    assert plan.decision_ms() is None


def test_decision_ms_of_a_single_profiled_component():
    plan = Plan(model="one", components=[Component(name="c", latency_ms=4.0)])
    assert plan.decision_ms() == pytest.approx(4.0)


def test_plan_defaults_and_independent_containers():
    first, second = Plan(model="a"), Plan(model="b")
    first.components.append(Component(name="c"))
    first.notes.append("note")
    assert second.components == [] and second.notes == []
    assert second.target_tag == "unknown"
    assert second.baseline_hz is None


def test_verdict_defaults():
    verdict = Verdict(component="encoder", action="trt_fp16", why="62% share")
    assert verdict.expected_speedup == 1.0
    assert verdict.confidence == "medium"
    assert verdict.action in ACTIONS


# --------------------------------------------------------------------------
# ShapeProfile: dynamic inputs
# --------------------------------------------------------------------------

def test_a_static_graph_is_not_dynamic():
    spec = GraphSpec("g", {"x": (1, 3, 224, 224)}, ["y"])
    spec.validate()
    assert spec.is_dynamic is False
    assert spec.dynamic_axes() == {}


def test_a_profiled_dynamic_axis_validates():
    spec = GraphSpec("g", {"x": (-1, 3, 224, 224)}, ["y"],
                     profiles={"x": ShapeProfile(min=(1, 3, 224, 224),
                                                 opt=(4, 3, 224, 224),
                                                 max=(8, 3, 224, 224))})
    spec.validate()
    assert spec.is_dynamic is True
    assert spec.dynamic_axes() == {"x": [0]}


def test_volume_uses_the_opt_shape_for_a_dynamic_input():
    spec = GraphSpec("g", {"x": (-1, 4)}, ["y"],
                     profiles={"x": ShapeProfile((1, 4), (4, 4), (8, 4))})
    assert spec.volume("x") == 16


def test_profile_for_an_unknown_input_raises():
    spec = GraphSpec("g", {"x": (1, 4)}, ["y"],
                     profiles={"nope": ShapeProfile((1, 4), (1, 4), (2, 4))})
    with pytest.raises(ValueError, match="not.*one of its inputs"):
        spec.validate()


def test_an_unordered_profile_raises_naming_the_axis():
    spec = GraphSpec("g", {"x": (-1, 4)}, ["y"],
                     profiles={"x": ShapeProfile((4, 4), (2, 4), (8, 4))})
    with pytest.raises(ValueError, match="axis 0"):
        spec.validate()


def test_a_mismatched_rank_profile_raises():
    spec = GraphSpec("g", {"x": (-1, 4)}, ["y"],
                     profiles={"x": ShapeProfile((1, 4), (2, 4), (8, 4, 1))})
    with pytest.raises(ValueError, match="mismatched ranks"):
        spec.validate()


def test_a_profile_that_pins_a_dynamic_axis_is_rejected():
    """A "dynamic" axis whose min == max is a static axis with extra cost."""
    spec = GraphSpec("g", {"x": (-1, 4)}, ["y"],
                     profiles={"x": ShapeProfile((4, 4), (4, 4), (4, 4))})
    with pytest.raises(ValueError, match="pins it to 4"):
        spec.validate()


def test_dynamic_axes_reports_every_axis_that_varies():
    spec = GraphSpec("g", {"x": (-1, 3, -1, -1)}, ["y"],
                     profiles={"x": ShapeProfile((1, 3, 128, 128),
                                                 (2, 3, 224, 224),
                                                 (4, 3, 512, 512))})
    spec.validate()
    assert spec.dynamic_axes() == {"x": [0, 2, 3]}


def test_declaring_an_axis_dynamic_then_pinning_it_is_rejected():
    """A -1 the profile pins is a static axis paying a dynamic engine's cost."""
    spec = GraphSpec("g", {"x": (-1, 3, -1, -1)}, ["y"],
                     profiles={"x": ShapeProfile((1, 3, 224, 224),
                                                 (2, 3, 224, 224),
                                                 (4, 3, 224, 224))})
    with pytest.raises(ValueError, match="axis 2 is marked dynamic"):
        spec.validate()
