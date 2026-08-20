"""Regressions for four defects found while optimizing InternVLA-N1.

Each was silent: the pipeline kept running and produced a plausible artifact.
Three of them would have shipped a wrong engine.
"""
import json

import pytest

from sparx_agency.tasks.common.trt_optimizer import dissect, pipeline
from sparx_agency.tasks.common.trt_optimizer.spec import (
    Cadence, Component, Exportability, GraphSpec, Plan, ShapeProfile, Verdict,
)


# -- 1. synthetic ".other" buckets are not modules ------------------------

class _Leaf(object):
    def __init__(self):
        self._children = []

    def named_children(self):
        return iter(self._children)


def test_is_synthetic_matches_the_accounting_bucket():
    """``plan`` must not hand a ".other" row to the profiler as a module.

    ``dissect`` emits it to keep the parameter total exact; it is a sum over
    several places in the tree with no forward to hook, and
    ``profile_components`` rightly raises on a name it cannot resolve.
    """
    bucket = Component(
        name="rgb_model.other", params=1000,
        exportability=Exportability.HOSTILE,
        reason=("synthetic accounting bucket: parameters held directly on "
                "'rgb_model' plus subtrees below min_params; not a module, "
                "not exportable"))
    assert dissect.is_synthetic(bucket)


def test_a_real_submodule_called_other_is_not_synthetic():
    """The name alone is not the discriminator; the reason must match too."""
    real = Component(name="backbone.other", params=1000, reason="")
    assert not dissect.is_synthetic(real)


def test_a_synthetic_named_row_elsewhere_is_not_matched():
    assert not dissect.is_synthetic(Component(name="other", params=1, reason=""))


# -- 2. a plan can be read back for the staged bench path -----------------

def test_load_plan_round_trips_every_field(tmp_path):
    """Without this, staged ``bench`` loses the component table entirely."""
    original = Plan(
        model="demo", target_tag="tag",
        components=[Component(name="trunk", params=7, cadence=Cadence.PER_STEP,
                              calls_per_decision=10.0, latency_ms=1.5,
                              dtype="bfloat16")],
        graphs=[GraphSpec(key="g", inputs={"x": (1, 3)}, outputs=["y"],
                          component="trunk", precision_sensitive=True,
                          profiles={"x": ShapeProfile((1, 3), (2, 3), (4, 3))})],
        verdicts=[Verdict(component="trunk", action="trt_fp16", why="because")],
        baseline_hz=12.5, notes=["a note"])
    path = pipeline.save_plan(original, tmp_path / "plan.json")

    loaded = pipeline.load_plan(path)
    assert loaded.model == "demo" and loaded.baseline_hz == 12.5
    assert loaded.notes == ["a note"]
    component = loaded.components[0]
    assert component.latency_ms == 1.5 and component.calls_per_decision == 10.0
    assert component.dtype == "bfloat16"
    graph = loaded.graphs[0]
    assert graph.inputs == {"x": (1, 3)} and graph.precision_sensitive is True
    assert graph.profiles["x"].opt == (2, 3)
    assert loaded.verdicts[0].action == "trt_fp16"


# -- 3. precision_sensitive is honoured, not just recorded ----------------

def test_precision_sensitive_is_read_from_the_spec():
    spec = GraphSpec(key="g", inputs={"x": (1,)}, outputs=["y"],
                     precision_sensitive=True)
    assert pipeline._is_precision_sensitive(spec, {}, "g")


def test_precision_sensitive_falls_back_to_the_manifest():
    """A build driven from an ONNX directory alone must still honour the flag."""
    manifest = {"graphs": {"g": {"precision_sensitive": True}}}
    assert pipeline._is_precision_sensitive(None, manifest, "g")
    assert not pipeline._is_precision_sensitive(None, {"graphs": {"g": {}}}, "g")


# -- 4. a mixed-precision selection must name every engine ----------------

def _touch(directory, *names):
    for name in names:
        (directory / name).write_bytes(b"")


def test_engine_files_falls_back_to_fp32_for_a_pinned_graph(tmp_path):
    """A pinned graph is built FP32 even when the race blessed FP16.

    Globbing one suffix drops it, and a selection missing an engine points the
    runtime at a pipeline it cannot complete.
    """
    _touch(tmp_path, "a.fp16.engine", "b.fp32.engine", "c.fp16.engine")
    files = pipeline._engine_files(tmp_path, "fp16")
    assert files == {"a": "a.fp16.engine", "b": "b.fp32.engine",
                     "c": "c.fp16.engine"}


def test_engine_files_prefers_the_blessed_precision_when_both_exist(tmp_path):
    _touch(tmp_path, "a.fp16.engine", "a.fp32.engine")
    assert pipeline._engine_files(tmp_path, "fp16") == {"a": "a.fp16.engine"}


def test_engine_files_raises_when_a_required_key_has_no_engine(tmp_path):
    _touch(tmp_path, "a.fp16.engine")
    with pytest.raises(RuntimeError, match="no engine built for b"):
        pipeline._engine_files(tmp_path, "fp16", keys=["a", "b"])


def test_engine_files_restricts_to_the_requested_keys(tmp_path):
    _touch(tmp_path, "a.fp16.engine", "stale.fp16.engine")
    assert pipeline._engine_files(tmp_path, "fp16", keys=["a"]) == {
        "a": "a.fp16.engine"}


def test_fp32_selection_does_not_pick_up_fp16_engines(tmp_path):
    _touch(tmp_path, "a.fp16.engine", "a.fp32.engine")
    assert pipeline._engine_files(tmp_path, "fp32") == {"a": "a.fp32.engine"}
