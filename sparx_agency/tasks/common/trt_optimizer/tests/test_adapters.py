"""The shipped reference adapters, exercised as the pipeline exercises them.

These tests are the genericity claim made checkable. Two adapters from two task
families -- an image classifier and a semantic segmenter -- satisfy the same
:class:`..adapter.ModelAdapter` contract, pass the same static checks, and
produce completely different decision metrics from it.

The metric tests never build an engine. A perturbation applied straight to the
reference logits is a *better* probe than a real engine here: it is exact,
reproducible, and can be dialled from "no change at all" to "a different network
entirely", which is what proves a metric degrades in the direction it claims to.
The real engine is proved separately, in ``test_adapters_e2e.py``.

Run with the interpreter that owns torch::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
      ~/miniconda3/envs/navdp/bin/python -m pytest \\
      sparx_agency/tasks/common/trt_optimizer/tests/test_adapters.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the adapters build torch models")
pytest.importorskip("torchvision", reason="the reference adapters are torchvision")

from sparx_agency.tasks.common.trt_optimizer import (  # noqa: E402
    adapter as adapter_mod, dissect, pipeline)
from sparx_agency.tasks.common.trt_optimizer.adapters import (  # noqa: E402
    ImageClassifierAdapter, SemanticSegmentationAdapter)
from sparx_agency.tasks.common.trt_optimizer.export import op_gate  # noqa: E402
from sparx_agency.tasks.common.trt_optimizer.spec import Cadence  # noqa: E402

#: Perturbation ladder in logit units: none, negligible, real, catastrophic.
EPSILONS = (0.0, 0.02, 0.5, 8.0)


@pytest.fixture(scope="module")
def classifier():
    """The default resnet18 adapter -- no weights, so nothing is downloaded."""
    return ImageClassifierAdapter()


@pytest.fixture(scope="module")
def segmenter():
    """The default LR-ASPP adapter -- no weights and no backbone weights."""
    return SemanticSegmentationAdapter()


@pytest.fixture(scope="module")
def both(classifier, segmenter):
    """Both adapters, for the checks that must hold of any adapter at all."""
    return [classifier, segmenter]


def _logits(rows=48, classes=100, seed=0):
    """A batch of plausible classifier logits."""
    return np.random.default_rng(seed).standard_normal((rows, classes)) * 2.0


def _dense_logits(batch=2, classes=6, size=48, seed=0):
    """A batch of plausible dense segmentation logits, ``(N, C, H, W)``."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, classes, size, size)) * 2.0


def _perturbed(reference, epsilon, seed=17):
    """``reference`` plus scaled noise -- a stand-in for a worse engine."""
    rng = np.random.default_rng(seed)
    return reference + epsilon * rng.standard_normal(reference.shape)


# ------------------------------------------------------------- the contract

def test_both_adapters_pass_the_static_check(both):
    """``adapter.check`` accepts both: static shapes, unique keys, sane gates."""
    for candidate in both:
        assert adapter_mod.check(candidate) is candidate


def test_both_adapters_are_registered_and_creatable():
    """Importing the package registers them; the registry builds them back."""
    available = adapter_mod.available()
    assert "image_classifier" in available
    assert "semantic_segmentation" in available
    assert isinstance(adapter_mod.create("image_classifier"),
                      ImageClassifierAdapter)
    assert isinstance(adapter_mod.create("semantic_segmentation"),
                      SemanticSegmentationAdapter)


def test_graphs_are_static_and_carry_the_architecture(both):
    """One engine each, fully static, keyed so two backbones cannot collide."""
    for candidate in both:
        graphs = candidate.graphs()
        assert len(graphs) == 1
        graph = graphs[0]
        graph.validate()  # raises on any non-positive or missing dimension
        assert graph.inputs == {"image": (1, 3, 224, 224)}
        assert graph.cadence == Cadence.PER_FRAME
        assert candidate.arch in graph.key
        assert graph.volume("image") == 1 * 3 * 224 * 224


def test_gates_only_ever_name_decision_metrics(both):
    """Every gated name is emitted by ``decision_metrics``, and none is an error.

    A gate on a metric the adapter never emits fails the run for the wrong
    reason; a gate on a tensor error is the mistake this whole package exists
    to prevent.
    """
    for candidate in both:
        if isinstance(candidate, ImageClassifierAdapter):
            metrics = candidate.decision_metrics(_logits(), _logits())
        else:
            metrics = candidate.decision_metrics(_dense_logits(),
                                                 _dense_logits())
        for name in candidate.gates():
            assert name in metrics, name
            assert "error" not in name and "l2" not in name, name


def test_cadences_reach_every_component_through_the_dissector(classifier):
    """The empty-prefix declaration really does cover every inventory row."""
    model = classifier.load()
    components = dissect.inventory(model, max_depth=1,
                                   cadences=classifier.cadences())
    assert components
    assert all(c.cadence == Cadence.PER_FRAME for c in components)
    dissect.check_accounting(components, model)


# ------------------------------------------------------------ load and run

def test_load_is_reproducible_and_leaves_the_global_rng_alone(classifier):
    """Two untrained loads agree bit for bit, and the caller's RNG is untouched.

    Without this an ONNX exported by one process could not be compared against
    a torch reference loaded by another, and every parity number in the report
    would be noise.
    """
    before = torch.get_rng_state().clone()
    first = classifier.load()
    after = torch.get_rng_state()
    second = classifier.load()
    assert torch.equal(before, after)
    for (name, a), (_, b) in zip(first.state_dict().items(),
                                 second.state_dict().items()):
        assert torch.equal(a, b), name


def test_reference_run_matches_the_declared_graph_contract(both):
    """One scenario in, one output tensor out, of the shape the graph promises."""
    for candidate in both:
        model = candidate.patch(candidate.load())
        scenario = candidate.scenarios(1, seed=3)[0]
        out = candidate.run_reference(model, scenario)
        assert np.isfinite(out).all()
        if isinstance(candidate, ImageClassifierAdapter):
            assert out.shape == (1, 1000)
        else:
            assert out.shape[:2] == (1, 21)
            assert out.shape[2:] == candidate.image_shape[2:]


def test_segmentation_wrapper_returns_a_tensor_not_a_dict(segmenter):
    """The export wrapper is where a dict-returning forward becomes an engine IO."""
    model = segmenter.load()
    wrappers = segmenter.wrappers(model)
    assert list(wrappers) == [segmenter.graph_key]
    scenario = segmenter.scenarios(1, seed=5)[0]
    with torch.no_grad():
        out = wrappers[segmenter.graph_key](torch.from_numpy(scenario))
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape)[:2] == (1, 21)


def test_classifier_is_its_own_export_wrapper(classifier):
    """A single-tensor model needs no wrapper, and says so by returning itself."""
    model = classifier.load()
    assert classifier.wrappers(model)[classifier.graph_key] is model
    assert classifier.patch(model) is model


def test_segmentation_exports_a_clean_static_graph(segmenter, tmp_path):
    """The dict-unwrapping wrapper survives the exporter, Resize and all.

    ``Resize`` is a decoder's upsample here and a bug in a ViT export, where it
    would be a positional-embedding interpolation that should have been baked.
    The same graph therefore passes the default policy and fails
    ``op_gate.vit_policy()`` -- which is why the gate is a policy object and not
    a fixed list of forbidden operators.
    """
    pytest.importorskip("onnx", reason="the export stage needs onnx")
    manifest = pipeline.export(segmenter, checkpoint=None, out_dir=tmp_path,
                               device="cpu")
    entry = manifest["graphs"][segmenter.graph_key]
    assert entry["inputs"] == ["image"]
    assert entry["outputs"] == ["class_logits"]

    onnx_path = tmp_path / entry["onnx"]
    result = op_gate.gate(onnx_path, key=segmenter.graph_key)
    assert result.ok, result.messages
    assert result.dynamic_tensors == []
    assert result.op_counts.get("Resize", 0) >= 1
    assert not op_gate.gate(onnx_path, policy=op_gate.vit_policy(),
                            key=segmenter.graph_key).ok


def test_run_engines_names_the_missing_engine(both):
    """A missing runtime raises with the key to build, not a KeyError on a dict."""
    for candidate in both:
        scenario = candidate.scenarios(1)[0]
        with pytest.raises(KeyError) as excinfo:
            candidate.run_engines({}, scenario)
        assert candidate.graph_key in str(excinfo.value)


# -------------------------------------------------------------- scenarios

def test_scenarios_are_deterministic_for_a_fixed_seed(both):
    """Same seed, same batches; different seed, different batches."""
    for candidate in both:
        first = candidate.scenarios(4, seed=11)
        again = candidate.scenarios(4, seed=11)
        other = candidate.scenarios(4, seed=12)
        assert len(first) == 4
        for a, b in zip(first, again):
            assert a.dtype == np.float32
            assert a.shape == candidate.image_shape
            assert np.array_equal(a, b)
        assert not np.array_equal(first[0], other[0])
        assert not np.array_equal(first[0], first[1])


def test_scenarios_refuses_to_produce_nothing(both):
    """Zero scenarios means an empty comparison, which is not a soft condition."""
    for candidate in both:
        with pytest.raises(ValueError):
            candidate.scenarios(0)


# -------------------------------------------------- classifier decision metrics

def test_classifier_metrics_are_perfect_against_itself(classifier):
    """The candidate IS the reference: every agreement is 1.0, every error 0.0."""
    reference = _logits()
    metrics = classifier.decision_metrics(reference, reference)
    assert metrics["top1_agreement"] == 1.0
    assert metrics["top5_agreement"] == 1.0
    assert metrics["mean_abs_logit_error"] == 0.0
    assert metrics["max_softmax_delta"] == 0.0


def test_classifier_metrics_degrade_with_the_perturbation(classifier):
    """Agreements fall and tensor errors rise, monotonically, along the ladder."""
    reference = _logits()
    rows = [classifier.decision_metrics(reference, _perturbed(reference, eps))
            for eps in EPSILONS]
    top1 = [r["top1_agreement"] for r in rows]
    top5 = [r["top5_agreement"] for r in rows]
    errors = [r["mean_abs_logit_error"] for r in rows]
    assert top1 == sorted(top1, reverse=True), top1
    assert top5 == sorted(top5, reverse=True), top5
    assert errors == sorted(errors), errors
    assert top1[0] == 1.0 and top5[0] == 1.0
    assert top1[1] > 0.9                     # a negligible perturbation
    assert top1[-1] < 0.2 and top5[-1] < 0.5  # a different network entirely


def test_classifier_top5_is_the_more_forgiving_of_the_two(classifier):
    """Top-5 can never be worse than top-1, which is why it is gated harder."""
    reference = _logits(seed=4)
    for eps in EPSILONS:
        metrics = classifier.decision_metrics(reference,
                                              _perturbed(reference, eps))
        assert metrics["top5_agreement"] >= metrics["top1_agreement"]


def test_classifier_gates_pass_on_identity_and_fail_on_a_bad_engine(classifier):
    """The gate is what blocks a build, so it is asserted in both directions."""
    reference = _logits()
    passed, rows = adapter_mod.evaluate_gates(
        classifier, classifier.decision_metrics(reference, reference))
    assert passed and all(row[-1] for row in rows)

    bad = classifier.decision_metrics(reference, _perturbed(reference, 8.0))
    failed, rows = adapter_mod.evaluate_gates(classifier, bad)
    assert not failed
    assert [row[0] for row in rows] == ["top1_agreement", "top5_agreement"]
    assert not any(row[-1] for row in rows)


def test_classifier_metrics_reject_a_shape_mismatch(classifier):
    """Two differently shaped outputs is an adapter bug, not a low score."""
    with pytest.raises(ValueError):
        classifier.decision_metrics(_logits(), _logits(classes=90))


# ------------------------------------------------ segmentation decision metrics

def test_segmentation_metrics_are_perfect_against_itself(segmenter):
    """Identical maps: full pixel agreement, IoU 1.0, nothing changed."""
    reference = _dense_logits()
    metrics = segmenter.decision_metrics(reference, reference)
    assert metrics["pixel_agreement"] == 1.0
    assert metrics["mean_iou"] == 1.0
    assert metrics["changed_pixel_fraction"] == 0.0
    assert metrics["mean_abs_logit_error"] == 0.0


def test_the_two_adapters_answer_with_different_metrics(segmenter, classifier):
    """The same contract, two disjoint metric sets. This is the whole point."""
    seg = set(segmenter.decision_metrics(_dense_logits(), _dense_logits()))
    cls = set(classifier.decision_metrics(_logits(), _logits()))
    assert seg & cls == {"mean_abs_logit_error"}   # only the diagnostic is shared
    assert set(segmenter.gates()) & set(classifier.gates()) == set()


def test_segmentation_metrics_degrade_with_the_perturbation(segmenter):
    """Pixel agreement and IoU fall together, and the complement tracks."""
    reference = _dense_logits()
    rows = [segmenter.decision_metrics(reference, _perturbed(reference, eps))
            for eps in EPSILONS]
    agreement = [r["pixel_agreement"] for r in rows]
    iou = [r["mean_iou"] for r in rows]
    assert agreement == sorted(agreement, reverse=True), agreement
    assert iou == sorted(iou, reverse=True), iou
    assert agreement[0] == 1.0 and iou[0] == 1.0
    assert agreement[-1] < 0.4 and iou[-1] < 0.3
    for row in rows:
        assert row["changed_pixel_fraction"] == pytest.approx(
            1.0 - row["pixel_agreement"])


def test_mean_iou_punishes_a_lost_class_that_pixel_agreement_shrugs_off(segmenter):
    """Why both are gated: a small class vanishes under a high agreement."""
    reference = _dense_logits(batch=1, classes=4, size=64, seed=2)
    reference[0, 3] -= 20.0                 # class 3 exists only in one corner
    reference[0, 3, :6, :6] += 60.0
    candidate = reference.copy()
    candidate[0, 3, :6, :6] -= 60.0         # ... and the candidate loses it
    metrics = segmenter.decision_metrics(reference, candidate)
    assert metrics["pixel_agreement"] > 0.99
    assert metrics["mean_iou"] < 0.8
    assert not adapter_mod.evaluate_gates(segmenter, metrics)[0]


def test_segmentation_gates_pass_on_identity_and_fail_on_a_bad_engine(segmenter):
    """Both directions, as for the classifier."""
    reference = _dense_logits()
    passed, rows = adapter_mod.evaluate_gates(
        segmenter, segmenter.decision_metrics(reference, reference))
    assert passed and all(row[-1] for row in rows)

    bad = segmenter.decision_metrics(reference, _perturbed(reference, 8.0))
    failed, rows = adapter_mod.evaluate_gates(segmenter, bad)
    assert not failed
    assert [row[0] for row in rows] == ["mean_iou", "pixel_agreement"]
    assert not any(row[-1] for row in rows)


def test_segmentation_metrics_reject_a_flat_tensor(segmenter):
    """A 2D array is a classifier's output; asking for pixels is a bug."""
    with pytest.raises(ValueError):
        segmenter.decision_metrics(_logits(), _logits())
