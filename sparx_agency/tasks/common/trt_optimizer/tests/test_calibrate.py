"""Tests for the INT8 path, run on an interpreter with no TensorRT at all.

Everything here is pure logic plus stand-ins, and that is not a compromise --
it is the only way the interesting cases are reachable. This machine's TensorRT
11 has no ``IInt8EntropyCalibrator2``, so a real entropy calibrator cannot be
constructed here at any effort; and an Orin's TensorRT 10, where it can, is not
the machine running the suite. So ``tensorrt`` is injected as a namespace whose
only load-bearing property is whether ``BuilderFlag.FP16`` exists (the same
probe :func:`..engine.precision.is_strongly_typed` uses), and ``pycuda.driver``
is a fake that records allocations and copies.

That fake driver is what makes the one contract that actually matters testable:
``get_batch(names)`` must return DEVICE pointers **in the order TensorRT asked
for them**, not in the order the calibration mapping happens to iterate. An
implementation that returns them in mapping order passes every happy-path
smoke test and feeds each tensor another tensor's data.

The adapter side is a hand-written fake with two graphs, because
:func:`collect_calibration_arrays` reads nothing from an adapter except
``graphs()``.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from sparx_agency.tasks.common.trt_optimizer.engine import calibrate as C
from sparx_agency.tasks.common.trt_optimizer.engine import precision as P
from sparx_agency.tasks.common.trt_optimizer.spec import GraphSpec, ShapeProfile

try:  # neither interpreter here has it; the assertion below states which
    import modelopt  # noqa: F401
    HAS_MODELOPT = True
except ImportError:
    HAS_MODELOPT = False


# --------------------------------------------------------------------------
# stand-ins
# --------------------------------------------------------------------------

class _WeakFlags(object):
    """TensorRT <= 10 ``BuilderFlag``: the precision flags are still there."""

    FP16 = 0
    INT8 = 1
    TF32 = 2


class _StrongFlags(object):
    """TensorRT 11 ``BuilderFlag``: measured member list, precision gone."""

    DEBUG = 0
    GPU_FALLBACK = 1
    REFIT = 2
    TF32 = 3


class _CalibratorBase(object):
    """Stands in for ``trt.IInt8EntropyCalibrator2``, which is a base class."""

    def __init__(self):
        self.base_initialized = True


def _weak_trt(with_calibrator=True):
    """A TensorRT 10 module: weak typing, and (usually) a calibrator class."""
    ns = types.SimpleNamespace(BuilderFlag=_WeakFlags, __version__="10.3.0")
    if with_calibrator:
        ns.IInt8EntropyCalibrator2 = _CalibratorBase
    return ns


def _strong_trt():
    """A TensorRT 11 module: strong typing, no calibrator class anywhere."""
    return types.SimpleNamespace(BuilderFlag=_StrongFlags,
                                 __version__="11.1.0.106")


class _FakeCuda(object):
    """``pycuda.driver`` reduced to the two calls the calibrator makes."""

    def __init__(self):
        self.allocations = []
        self.copies = []

    def mem_alloc(self, nbytes):
        """Hand back a distinct fake device address per allocation."""
        self.allocations.append(int(nbytes))
        return 0x1000 + 0x100 * len(self.allocations)

    def memcpy_htod(self, dest, src):
        self.copies.append((int(dest), np.array(src, copy=True)))


class _FakeAdapter(object):
    """Only ``graphs()`` is read, so only ``graphs()`` exists."""

    name = "fake_net"

    def __init__(self, specs):
        self._specs = list(specs)

    def graphs(self):
        return list(self._specs)


IMG = (1, 3, 4, 4)
VEC = (1, 8)


def _adapter():
    """A two-graph classifier-shaped adapter: a trunk and a head."""
    return _FakeAdapter([
        GraphSpec(key="trunk", inputs={"image": IMG}, outputs=["features"]),
        GraphSpec(key="head", inputs={"features": VEC, "prompt": (1, 2)},
                  outputs=["logits"]),
    ])


def _trunk_capture(scenario):
    """One call of the trunk: a single sample at the declared shape."""
    return {"image": np.full(IMG, float(scenario), np.float32)}


# --------------------------------------------------------------------------
# collect_calibration_arrays -- shapes, counts, refusals
# --------------------------------------------------------------------------

def test_collect_stacks_one_sample_per_scenario():
    out = C.collect_calibration_arrays(_adapter(), "trunk", range(200),
                                       _trunk_capture, min_samples=128)
    assert set(out) == {"image"}
    assert out["image"].shape == (200,) + IMG
    assert out["image"].dtype == np.float32
    assert out["image"][7][0, 0, 0, 0] == pytest.approx(7.0)


def test_collect_accepts_a_whole_loop_stack_from_one_scenario():
    """Rule 3: an iterative graph returns every step, not only the first."""
    steps = 20

    def capture(scenario):
        return {"image": np.stack([np.full(IMG, float(k), np.float32)
                                   for k in range(steps)])}

    out = C.collect_calibration_arrays(_adapter(), "trunk", range(10), capture,
                                       min_samples=128)
    assert out["image"].shape == (200,) + IMG
    # The whole loop is present, not 10 copies of step 0.
    assert sorted(set(out["image"][:, 0, 0, 0, 0].tolist())) == \
        [float(k) for k in range(steps)]


def test_collect_stops_at_max_samples_without_draining_the_scenarios():
    seen = []

    def capture(scenario):
        seen.append(scenario)
        return _trunk_capture(scenario)

    out = C.collect_calibration_arrays(_adapter(), "trunk", range(10 ** 6),
                                       capture, max_samples=300)
    assert out["image"].shape == (300,) + IMG
    assert len(seen) == 300


def test_collect_keeps_every_input_aligned():
    def capture(scenario):
        return {"features": np.full(VEC, float(scenario), np.float32),
                "prompt": np.full((1, 2), float(scenario), np.float32)}

    out = C.collect_calibration_arrays(_adapter(), "head", range(128), capture)
    assert out["features"].shape == (128,) + VEC
    assert out["prompt"].shape == (128, 1, 2)


def test_collect_raises_on_a_shape_that_disagrees_with_the_graphspec():
    def capture(scenario):
        return {"image": np.zeros((1, 3, 8, 8), np.float32)}

    with pytest.raises(ValueError) as excinfo:
        C.collect_calibration_arrays(_adapter(), "trunk", range(200), capture)
    message = str(excinfo.value)
    assert "'image'" in message and "'trunk'" in message
    assert "(1, 3, 8, 8)" in message and "(1, 3, 4, 4)" in message


def test_collect_raises_when_too_few_samples_were_collected():
    with pytest.raises(ValueError) as excinfo:
        C.collect_calibration_arrays(_adapter(), "trunk", range(3),
                                     _trunk_capture)
    message = str(excinfo.value)
    assert "only 3 calibration sample" in message
    assert "128" in message          # the rule-2 minimum is quoted
    assert "rule 3" in message       # and the iterative-graph way out


def test_collect_min_samples_is_tunable_for_a_smoke_test():
    out = C.collect_calibration_arrays(_adapter(), "trunk", range(4),
                                       _trunk_capture, min_samples=4)
    assert out["image"].shape == (4,) + IMG


def test_collect_raises_on_a_missing_declared_input():
    def capture(scenario):
        return {"features": np.zeros(VEC, np.float32)}

    with pytest.raises(ValueError) as excinfo:
        C.collect_calibration_arrays(_adapter(), "head", range(200), capture)
    assert "'prompt'" in str(excinfo.value)


def test_collect_raises_on_an_undeclared_input():
    def capture(scenario):
        return {"image": np.zeros(IMG, np.float32),
                "temperature": np.zeros((1, 1), np.float32)}

    with pytest.raises(ValueError) as excinfo:
        C.collect_calibration_arrays(_adapter(), "trunk", range(200), capture)
    assert "'temperature'" in str(excinfo.value)


def test_collect_raises_on_a_ragged_capture():
    def capture(scenario):
        return {"features": np.zeros((4,) + VEC, np.float32),
                "prompt": np.zeros((3, 1, 2), np.float32)}

    with pytest.raises(ValueError) as excinfo:
        C.collect_calibration_arrays(_adapter(), "head", range(200), capture)
    assert "ragged" in str(excinfo.value)


def test_collect_raises_on_an_empty_scenario_set():
    with pytest.raises(ValueError) as excinfo:
        C.collect_calibration_arrays(_adapter(), "trunk", [], _trunk_capture)
    assert "only 0 calibration sample" in str(excinfo.value)
    assert "0 scenario" in str(excinfo.value)


def test_collect_rejects_an_undeclared_dynamic_dimension():
    """A -1 with no ShapeProfile is a broken spec, not a free axis."""
    broken = _FakeAdapter([GraphSpec(key="trunk", inputs={"image": (-1, 3)},
                                     outputs=["features"])])
    with pytest.raises(ValueError) as excinfo:
        C.collect_calibration_arrays(broken, "trunk", range(200),
                                     _trunk_capture)
    assert "ShapeProfile" in str(excinfo.value)


def _dynamic_adapter():
    """A detector-shaped graph whose batch axis genuinely varies at run time."""
    return _FakeAdapter([
        GraphSpec(key="boxes", inputs={"crops": (-1, 3, 4, 4)},
                  outputs=["scores"],
                  profiles={"crops": ShapeProfile(min=(1, 3, 4, 4),
                                                  opt=(4, 3, 4, 4),
                                                  max=(8, 3, 4, 4))}),
    ])


def test_collect_accepts_a_dynamic_axis_at_the_profiles_opt_shape():
    out = C.collect_calibration_arrays(
        _dynamic_adapter(), "boxes", range(200),
        lambda s: {"crops": np.full((4, 3, 4, 4), float(s), np.float32)})
    assert out["crops"].shape == (200, 4, 3, 4, 4)


def test_collect_refuses_a_calibration_set_captured_at_two_dynamic_sizes():
    def capture(scenario):
        batch = 4 if scenario < 50 else 8
        return {"crops": np.zeros((batch, 3, 4, 4), np.float32)}

    with pytest.raises(ValueError) as excinfo:
        C.collect_calibration_arrays(_dynamic_adapter(), "boxes", range(200),
                                     capture)
    message = str(excinfo.value)
    assert "opt shape" in message
    assert "(8, 3, 4, 4)" in message and "(4, 3, 4, 4)" in message


def test_collect_still_checks_the_fixed_axes_of_a_dynamic_input():
    with pytest.raises(ValueError) as excinfo:
        C.collect_calibration_arrays(
            _dynamic_adapter(), "boxes", range(200),
            lambda s: {"crops": np.zeros((4, 3, 9, 9), np.float32)})
    assert "dynamic and accepts any size" in str(excinfo.value)


def test_collect_raises_on_an_unknown_graph_key_naming_the_real_ones():
    with pytest.raises(KeyError) as excinfo:
        C.collect_calibration_arrays(_adapter(), "decoder", range(200),
                                     _trunk_capture)
    assert "head" in str(excinfo.value) and "trunk" in str(excinfo.value)


def test_collect_raises_when_capture_fn_returns_something_that_is_not_a_mapping():
    with pytest.raises(ValueError) as excinfo:
        C.collect_calibration_arrays(_adapter(), "trunk", range(200),
                                     lambda s: np.zeros(IMG, np.float32))
    assert "mapping" in str(excinfo.value)


# --------------------------------------------------------------------------
# sample_count_advice -- rule 2 in numbers
# --------------------------------------------------------------------------

def test_sample_count_advice_rejects_ten():
    ok, message = C.sample_count_advice(10)
    assert ok is False
    assert "128" in message and "512" in message


def test_sample_count_advice_accepts_the_band_ends():
    for n in (128, 512):
        ok, message = C.sample_count_advice(n)
        assert ok is True
        assert "128-512" in message


def test_sample_count_advice_calls_five_thousand_wasteful_not_wrong():
    ok, message = C.sample_count_advice(5000)
    assert ok is True, "an oversized set is wasted build time, not a bad engine"
    assert "1024" in message and "0.1%" in message
    assert "512" in message


# --------------------------------------------------------------------------
# route B -- Q/DQ, and why it is unreachable here
# --------------------------------------------------------------------------

@pytest.mark.skipif(HAS_MODELOPT, reason="modelopt got installed; see reason")
def test_qdq_unavailable_here_and_says_which_package_is_missing():
    available, reason = C.qdq_available()
    assert available is False
    assert "modelopt" in reason


def test_qdq_available_names_modelopt_either_way():
    available, reason = C.qdq_available()
    assert available is HAS_MODELOPT
    assert "modelopt" in reason


def test_qdq_instructions_are_actionable():
    text = C.qdq_instructions("int8")
    assert "modelopt" in text
    assert "pip install" in text and "nvidia-modelopt" in text
    assert "from modelopt.onnx.quantization import quantize" in text
    assert 'quantize_mode="int8"' in text
    assert "nodes_to_exclude" in text
    assert "*vision_tower*" in text          # the deny list is spelled out
    assert "onnx_precision" in text          # and how to check it worked


def test_qdq_instructions_carry_the_block_sizes_that_build():
    assert "block_size=128" in C.qdq_instructions("int4")
    assert "block_size=16" in C.qdq_instructions("nvfp4")
    assert "block_size" not in C.qdq_instructions("int8")


def test_qdq_instructions_refuse_a_precision_with_no_quantization_route():
    with pytest.raises(ValueError) as excinfo:
        C.qdq_instructions("fp16")
    assert "int8" in str(excinfo.value)


def test_deny_list_delegates_to_precision():
    assert C.deny_list() == P.quantization_deny_list()
    for pattern in C.deny_list():
        assert P.reason_for(pattern)


def test_calibration_guidance_states_every_rule_with_its_number():
    text = C.CALIBRATION_GUIDANCE
    for number in ("1.", "2.", "3.", "4.", "5."):
        assert number in text
    for fact in ("128", "512", "1024", "0.1%", "axis 0", "64", "16", "32"):
        assert fact in text
    assert "Autotuner: no tactics to implement operation" in text


# --------------------------------------------------------------------------
# route selection
# --------------------------------------------------------------------------

def test_int8_route_is_the_calibrator_on_a_weakly_typed_trt():
    route, reason = C.int8_route(_weak_trt())
    assert route == C.ROUTE_ENTROPY_CALIBRATOR
    assert "10.3.0" in reason


def test_int8_route_is_qdq_on_a_strongly_typed_trt():
    route, reason = C.int8_route(_strong_trt())
    assert route == C.ROUTE_QDQ
    assert "Q" in reason and "11.1.0.106" in reason


@pytest.mark.skipif(HAS_MODELOPT, reason="modelopt got installed; see reason")
def test_require_int8_buildable_refuses_here_with_the_full_recipe():
    with pytest.raises(RuntimeError) as excinfo:
        C.require_int8_buildable(_strong_trt())
    message = str(excinfo.value)
    assert "modelopt" in message
    assert "pip install" in message


def test_require_int8_buildable_reports_the_calibrator_route(monkeypatch):
    """On the Orin's stack the gate passes once pycuda is importable."""
    monkeypatch.setattr(C, "_import_pycuda", _FakeCuda)
    assert C.require_int8_buildable(_weak_trt()) == C.ROUTE_ENTROPY_CALIBRATOR


def test_require_int8_buildable_refuses_the_calibrator_route_without_pycuda(
        monkeypatch):
    def _no_pycuda():
        raise RuntimeError("pycuda is not importable (fake)")

    monkeypatch.setattr(C, "_import_pycuda", _no_pycuda)
    with pytest.raises(RuntimeError) as excinfo:
        C.require_int8_buildable(_weak_trt())
    assert "pycuda" in str(excinfo.value)


# --------------------------------------------------------------------------
# route A -- the entropy calibrator
# --------------------------------------------------------------------------

def _stacks(n=128):
    """Two aligned calibration stacks whose values encode (row, tensor)."""
    a = np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2)
    b = -np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    return {"alpha": a, "beta": b}


def _make(stacks, cuda, cache, **kwargs):
    return C.make_entropy_calibrator(stacks, cache, trt_module=_weak_trt(),
                                     cuda_module=cuda, **kwargs)


def test_entropy_calibrator_is_refused_on_tensorrt_11_naming_the_qdq_route():
    with pytest.raises(RuntimeError) as excinfo:
        C.make_entropy_calibrator(_stacks(), "/tmp/unused.cache",
                                  trt_module=_strong_trt(),
                                  cuda_module=_FakeCuda())
    message = str(excinfo.value)
    assert "IInt8EntropyCalibrator2" in message
    assert "11.1.0.106" in message
    assert "Q/DQ" in message
    assert "modelopt" in message and "pip install" in message


def test_entropy_calibrator_is_refused_when_the_class_is_simply_absent():
    """Weak typing but no calibrator class is still no calibrator."""
    with pytest.raises(RuntimeError) as excinfo:
        C.make_entropy_calibrator(_stacks(), "/tmp/unused.cache",
                                  trt_module=_weak_trt(with_calibrator=False),
                                  cuda_module=_FakeCuda())
    assert "Q/DQ" in str(excinfo.value)


def test_entropy_calibrator_subclasses_the_tensorrt_base(tmp_path):
    cal = _make(_stacks(), _FakeCuda(), tmp_path / "int8.cache")
    assert isinstance(cal, _CalibratorBase)
    assert cal.base_initialized is True
    assert cal.get_batch_size() == 1


def test_get_batch_returns_device_pointers_in_the_order_asked(tmp_path):
    cuda = _FakeCuda()
    cal = _make(_stacks(), cuda, tmp_path / "int8.cache")

    first = cal.get_batch(["beta", "alpha"])
    second = cal.get_batch(["alpha", "beta"])
    assert first == list(reversed(second)), \
        "the pointer order must follow TensorRT's names, not the mapping's"

    # ... and each pointer was filled with ITS OWN tensor's row.
    beta_ptr, alpha_ptr = first
    beta_copy = [c for c in cuda.copies if c[0] == beta_ptr][0][1]
    alpha_copy = [c for c in cuda.copies if c[0] == alpha_ptr][0][1]
    assert alpha_copy.ravel()[0] == pytest.approx(0.0)
    assert beta_copy.ravel()[0] == pytest.approx(0.0)
    assert alpha_copy.shape == (1, 2, 2) and beta_copy.shape == (1, 2)


def test_get_batch_walks_the_stack_and_then_returns_none(tmp_path):
    stacks = _stacks(n=130)
    cal = _make(stacks, _FakeCuda(), tmp_path / "int8.cache", batch_size=13)
    rows = 0
    while cal.get_batch(["alpha", "beta"]) is not None:
        rows += 13
    assert rows == 130
    assert cal.get_batch(["alpha", "beta"]) is None


def test_get_batch_raises_on_an_input_it_has_no_data_for(tmp_path):
    cal = _make(_stacks(), _FakeCuda(), tmp_path / "int8.cache")
    with pytest.raises(KeyError) as excinfo:
        cal.get_batch(["alpha", "gamma"])
    assert "gamma" in str(excinfo.value)


def test_calibration_cache_round_trips(tmp_path):
    cache = tmp_path / "nested" / "int8.cache"
    cal = _make(_stacks(), _FakeCuda(), cache)
    assert cal.read_calibration_cache() is None
    cal.write_calibration_cache(b"scales")
    assert cache.read_bytes() == b"scales"
    assert _make(_stacks(), _FakeCuda(),
                 cache).read_calibration_cache() == b"scales"


def test_device_buffers_are_allocated_per_input_and_scaled_by_batch(tmp_path):
    cuda = _FakeCuda()
    _make(_stacks(), cuda, tmp_path / "int8.cache", batch_size=4)
    # alpha rows are 2x2 float32 = 16 B, beta rows 2 float32 = 8 B.
    assert sorted(cuda.allocations) == [32, 64]


def test_undersized_calibration_is_refused_then_explicitly_allowed(tmp_path):
    small = _stacks(n=10)
    with pytest.raises(ValueError) as excinfo:
        _make(small, _FakeCuda(), tmp_path / "int8.cache")
    assert "128" in str(excinfo.value)
    assert "allow_undersized" in str(excinfo.value)
    cal = _make(small, _FakeCuda(), tmp_path / "int8.cache",
                allow_undersized=True)
    assert cal.get_batch(["alpha", "beta"]) is not None


def test_ragged_calibration_stacks_are_refused(tmp_path):
    stacks = {"alpha": np.zeros((128, 2), np.float32),
              "beta": np.zeros((64, 2), np.float32)}
    with pytest.raises(ValueError) as excinfo:
        _make(stacks, _FakeCuda(), tmp_path / "int8.cache")
    assert "collect_calibration_arrays" in str(excinfo.value)


def test_empty_calibration_set_is_refused(tmp_path):
    with pytest.raises(ValueError):
        _make({}, _FakeCuda(), tmp_path / "int8.cache")


def test_a_batch_larger_than_the_data_is_refused(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        _make(_stacks(n=128), _FakeCuda(), tmp_path / "int8.cache",
              batch_size=256)
    assert "one batch" in str(excinfo.value)


def test_collected_arrays_feed_the_calibrator_directly(tmp_path):
    """The two halves of route A meet: collect -> make, no reshaping between."""
    arrays = C.collect_calibration_arrays(_adapter(), "head", range(128),
                                          lambda s: {
                                              "features": np.full(VEC, float(s),
                                                                  np.float32),
                                              "prompt": np.zeros((1, 2),
                                                                 np.float32)})
    cal = _make(arrays, _FakeCuda(), tmp_path / "int8.cache")
    assert cal.get_batch(["prompt", "features"]) is not None
