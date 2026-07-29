"""GPU-free guards for the shared VLA TensorRT engine runner.

The full name-based IO / resident-buffer / version-lock paths need TensorRT,
pycuda and a real ``.engine`` built on this machine, so those are exercised
on-target by each VLA's benchmark (``tasks/planning/vlas/<vla>/trt/benchmark``).
Here we assert only the fail-loud guards that run *before* any TensorRT import,
plus the property that makes one shared runner safe for several policies: it
raises each caller's own error type.
"""
from __future__ import annotations

import importlib.util
import json

import pytest

from sparx_agency.core.planning.vlas.common.errors import VlaError
from sparx_agency.core.planning.vlas.common.trt.engine_runner import TRTEngineRunner
from sparx_agency.core.planning.vlas.flownav.errors import FlowNavError
from sparx_agency.core.planning.vlas.navdp.errors import NavDPError

_HAS_TRT = (importlib.util.find_spec("tensorrt") is not None
            and importlib.util.find_spec("pycuda") is not None)


def test_missing_engine_raises_the_shared_base_by_default(tmp_path):
    # The missing-file check happens in __init__ before any TensorRT import, so
    # this runs even on a machine with no TensorRT.
    with pytest.raises(VlaError):
        TRTEngineRunner(tmp_path / "does_not_exist.engine")


@pytest.mark.parametrize("error_cls", [NavDPError, FlowNavError])
def test_missing_engine_raises_the_caller_s_error_class(tmp_path, error_cls):
    # Each policy passes its own error so downstream `except NavDPError` /
    # `except FlowNavError` keeps catching exactly what it caught when every
    # policy shipped its own copy of this runner.
    with pytest.raises(error_cls):
        TRTEngineRunner(tmp_path / "nope.engine", error_cls=error_cls)


@pytest.mark.parametrize("error_cls", [NavDPError, FlowNavError])
def test_every_policy_error_is_catchable_as_the_shared_base(tmp_path, error_cls):
    # An arbiter driving several policies (FALCON's hybrid/fallback planners)
    # needs one type to catch.
    with pytest.raises(VlaError):
        TRTEngineRunner(tmp_path / "nope.engine", error_cls=error_cls)


@pytest.mark.skipif(not _HAS_TRT, reason="tensorrt/pycuda not importable")
@pytest.mark.parametrize("error_cls", [NavDPError, FlowNavError])
def test_manifest_mismatch_raises(tmp_path, error_cls):
    # A present engine file whose manifest declares an impossible SM must fail
    # loud at construction, before deserialization would also fail.
    engine = tmp_path / "fake.engine"
    engine.write_bytes(b"not-a-real-engine")
    (tmp_path / "fake.engine.json").write_text(
        json.dumps({"sm": 1, "trt_version": "0.0.0"}))
    with pytest.raises(error_cls):
        TRTEngineRunner(engine, error_cls=error_cls)
