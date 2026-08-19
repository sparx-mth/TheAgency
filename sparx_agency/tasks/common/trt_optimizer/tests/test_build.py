"""Tests for the non-GPU half of the engine build.

``build_engine`` itself needs a real TensorRT, a real GPU and a real ONNX, so
what is tested here is everything around it that decides whether the build is
even set up correctly: the parser error path, the network-creation flag that
picks the TensorRT generation, the re-exported options, and the CUDA context
that must NOT be created for an FP16 build.

All of it runs in ``.venv`` against stand-ins. ``tensorrt`` is a namespace with
a scripted ``OnnxParser``; ``pycuda`` is injected into ``sys.modules`` so the
context manager's two paths can be told apart by whether the import happened at
all -- which is the actual claim, since importing pycuda on a machine that has
no CUDA context to retain is itself the failure being avoided.
"""
from __future__ import annotations

import sys
import types

import pytest

from sparx_agency.tasks.common.trt_optimizer.engine import build as BLD
from sparx_agency.tasks.common.trt_optimizer.engine import builder_config


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class _FakeLogger(object):
    """``trt.Logger``: a severity constant and a constructor taking it."""

    WARNING = "Logger.WARNING"
    VERBOSE = "Logger.VERBOSE"

    def __init__(self, severity=None):
        self.severity = severity


class _FakeError(object):
    """One ``IParserError``, which the module stringifies."""

    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


class _FakeParser(object):
    """``trt.OnnxParser`` with a scripted verdict and error list."""

    def __init__(self, ok=True, errors=()):
        self.ok = ok
        self.errors = [_FakeError(e) for e in errors]
        self.payloads = []
        self.network = None
        self.logger = None

    @property
    def num_errors(self):
        return len(self.errors)

    def get_error(self, index):
        return self.errors[index]

    def parse(self, payload):
        self.payloads.append(payload)
        return self.ok


class _FakeNetwork(object):
    """What ``create_network`` hands back."""

    def __init__(self, flags):
        self.flags = flags


class _FakeBuilder(object):
    """``trt.Builder``: records the creation flags it was called with."""

    def __init__(self):
        self.network_flags = []

    def create_network(self, flags):
        self.network_flags.append(flags)
        return _FakeNetwork(flags)


def fake_trt(parser=None, strongly_typed_value=0, with_strongly_typed=True):
    """A ``tensorrt`` stand-in exposing only what the module touches."""
    ns = types.SimpleNamespace(Logger=_FakeLogger)
    if with_strongly_typed:
        ns.NetworkDefinitionCreationFlag = types.SimpleNamespace(
            STRONGLY_TYPED=strongly_typed_value)
    else:
        ns.NetworkDefinitionCreationFlag = types.SimpleNamespace(
            PREFER_AOT_PYTHON_PLUGINS=0)

    def _open_parser(network, logger):
        parser.network = network
        parser.logger = logger
        return parser

    ns.OnnxParser = _open_parser
    return ns


def onnx_file(tmp_path, payload=b"onnx-bytes"):
    path = tmp_path / "navdp_encoder.onnx"
    path.write_bytes(payload)
    return path


# --------------------------------------------------------------------------
# _parse
# --------------------------------------------------------------------------

def test_parse_hands_the_graph_bytes_to_the_parser(tmp_path):
    parser = _FakeParser(ok=True)
    network = _FakeNetwork(1)
    assert BLD._parse(network, fake_trt(parser), onnx_file(tmp_path)) is None
    assert parser.payloads == [b"onnx-bytes"]
    assert parser.network is network


def test_parse_opens_the_parser_with_a_warning_logger(tmp_path):
    parser = _FakeParser(ok=True)
    BLD._parse(_FakeNetwork(1), fake_trt(parser), onnx_file(tmp_path))
    assert isinstance(parser.logger, _FakeLogger)
    assert parser.logger.severity == _FakeLogger.WARNING


def test_parse_raises_with_every_parser_error(tmp_path):
    """The builder's own message is useless without the parser's error list."""
    errors = ("In node 3 (importResize): UNSUPPORTED_NODE",
              "In node 7 (importGather): axis out of range",
              "Network has no outputs")
    parser = _FakeParser(ok=False, errors=errors)
    path = onnx_file(tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        BLD._parse(_FakeNetwork(1), fake_trt(parser), path)
    message = str(excinfo.value)
    assert "ONNX parse failed" in message
    assert str(path) in message
    for error in errors:
        assert error in message
    assert message.count("\n  ") == len(errors)


def test_parse_raises_even_when_the_parser_reports_no_errors(tmp_path):
    """A False verdict with an empty error list is still a failed parse."""
    parser = _FakeParser(ok=False, errors=())
    with pytest.raises(RuntimeError):
        BLD._parse(_FakeNetwork(1), fake_trt(parser), onnx_file(tmp_path))


def test_parse_raises_when_the_onnx_is_missing(tmp_path):
    parser = _FakeParser(ok=True)
    with pytest.raises(FileNotFoundError):
        BLD._parse(_FakeNetwork(1), fake_trt(parser), tmp_path / "absent.onnx")
    assert parser.payloads == []


# --------------------------------------------------------------------------
# _create_network -- which TensorRT generation is being built for
# --------------------------------------------------------------------------

def test_create_network_sets_the_strongly_typed_bit():
    """TensorRT 11 numbers STRONGLY_TYPED 0, so the flag word is 1 << 0."""
    builder = _FakeBuilder()
    network = BLD._create_network(builder, fake_trt(strongly_typed_value=0), True)
    assert builder.network_flags == [1]
    assert network.flags == 1


@pytest.mark.parametrize("value,expected", [(0, 1), (1, 2), (3, 8)])
def test_create_network_shifts_whatever_value_the_enum_carries(value, expected):
    builder = _FakeBuilder()
    BLD._create_network(builder, fake_trt(strongly_typed_value=value), True)
    assert builder.network_flags == [expected]


def test_create_network_passes_zero_for_a_weakly_typed_build():
    """A TensorRT <= 10 network must stay weakly typed to take a precision flag."""
    builder = _FakeBuilder()
    BLD._create_network(builder, fake_trt(strongly_typed_value=0), False)
    assert builder.network_flags == [0]


def test_create_network_falls_back_to_zero_without_the_flag():
    builder = _FakeBuilder()
    BLD._create_network(builder, fake_trt(with_strongly_typed=False), True)
    assert builder.network_flags == [0]


# --------------------------------------------------------------------------
# the re-exported options
# --------------------------------------------------------------------------

def test_build_options_is_the_builder_config_class_itself():
    """One import for a caller configuring a build, and one class, not two."""
    assert BLD.BuildOptions is builder_config.BuildOptions


def test_build_options_defaults_are_the_real_time_loop_defaults():
    options = BLD.BuildOptions()
    assert options.precision == "fp16"
    assert options.optimization_level == 3
    assert options.max_aux_streams == 0
    assert options.tf32 is True
    assert options.monitor_memory is None
    assert options.detailed_profiling is True
    assert (options.strip_plan, options.refit) == (False, False)
    assert options.timing_cache is None


# --------------------------------------------------------------------------
# _cuda_context -- only the INT8 calibrator path may touch pycuda
# --------------------------------------------------------------------------

def _install_broken_pycuda(monkeypatch):
    """Make ``import pycuda.driver`` fail, so any import at all is visible."""
    monkeypatch.setitem(sys.modules, "pycuda", None)
    monkeypatch.setitem(sys.modules, "pycuda.driver", None)


def _install_fake_pycuda(monkeypatch, calls):
    """A pycuda whose context records the order of init/push/pop."""

    class _Context(object):
        def push(self):
            calls.append("push")

        def pop(self):
            calls.append("pop")

    class _Device(object):
        def __init__(self, index):
            calls.append("device%d" % index)

        def retain_primary_context(self):
            calls.append("retain")
            return _Context()

    driver = types.ModuleType("pycuda.driver")
    driver.init = lambda: calls.append("init")
    driver.Device = _Device
    package = types.ModuleType("pycuda")
    package.driver = driver
    monkeypatch.setitem(sys.modules, "pycuda", package)
    monkeypatch.setitem(sys.modules, "pycuda.driver", driver)


def test_cuda_context_is_a_no_op_when_not_needed(monkeypatch):
    """An FP16 build must not import pycuda -- the import itself is the cost."""
    _install_broken_pycuda(monkeypatch)
    entered = []
    with BLD._cuda_context(needed=False):
        entered.append("body")
    assert entered == ["body"]


def test_cuda_context_does_import_pycuda_when_it_is_needed(monkeypatch):
    """Proves the previous test passes by not importing, not by luck."""
    _install_broken_pycuda(monkeypatch)
    with pytest.raises(ImportError):
        with BLD._cuda_context(needed=True):
            pass


def test_cuda_context_retains_pushes_and_pops_the_primary_context(monkeypatch):
    calls = []
    _install_fake_pycuda(monkeypatch, calls)
    with BLD._cuda_context(needed=True):
        calls.append("build")
    assert calls == ["init", "device0", "retain", "push", "build", "pop"]


def test_cuda_context_pops_even_when_the_build_raises(monkeypatch):
    """A leaked context is a leaked GPU allocation for the whole process."""
    calls = []
    _install_fake_pycuda(monkeypatch, calls)
    with pytest.raises(RuntimeError):
        with BLD._cuda_context(needed=True):
            raise RuntimeError("build_serialized_network returned None")
    assert calls[-1] == "pop"


def test_cuda_context_yields_exactly_once(monkeypatch):
    _install_broken_pycuda(monkeypatch)
    bodies = 0
    for _ in BLD._cuda_context.__wrapped__(False):
        bodies += 1
    assert bodies == 1
