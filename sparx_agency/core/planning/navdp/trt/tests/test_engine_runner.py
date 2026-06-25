"""TRTEngineRunner behaviour that is checkable without a GPU/engine.

The full name-based IO / resident-buffer / version-lock paths need TensorRT,
pycuda and a real ``.engine`` built on this machine, so those are exercised by
the on-target benchmark (``tasks/planning/navdp/benchmark``). Here we only assert
the fail-loud guards that run before any TensorRT import.
"""
from __future__ import annotations

import importlib.util

import pytest

from sparx_agency.core.planning.navdp.trt.engine_runner import TRTEngineRunner
from sparx_agency.core.planning.navdp.trt.errors import NavDPError

_HAS_TRT = importlib.util.find_spec("tensorrt") is not None and \
    importlib.util.find_spec("pycuda") is not None


def test_missing_engine_raises_navdp_error(tmp_path):
    # Missing-file check happens in __init__ before any TensorRT import, so this
    # runs even on a machine with no TensorRT.
    with pytest.raises(NavDPError):
        TRTEngineRunner(tmp_path / "does_not_exist.engine")


@pytest.mark.skipif(not _HAS_TRT, reason="tensorrt/pycuda not installed")
def test_manifest_mismatch_raises(tmp_path):
    # A present engine file with a manifest declaring an impossible SM must fail
    # loud at construction (before deserialization would also fail).
    import json
    eng = tmp_path / "fake.engine"
    eng.write_bytes(b"not a real engine")
    (tmp_path / "fake.engine.json").write_text(json.dumps({"sm": 1, "trt_version": "0.0"}))
    with pytest.raises(NavDPError):
        TRTEngineRunner(eng)
