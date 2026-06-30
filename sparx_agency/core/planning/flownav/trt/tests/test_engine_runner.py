"""GPU-free guards for the FlowNav engine runner.

The full IO path (deserialize + ``execute_async_v3``) is exercised on-target by
the benchmark; here we only check the failures that do not need a GPU: a missing
engine raises before any TRT import, and a manifest version mismatch raises
(only when ``tensorrt``/``pycuda`` happen to be importable).
"""
import importlib.util

import pytest

from sparx_agency.core.planning.flownav.trt.engine_runner import TRTEngineRunner
from sparx_agency.core.planning.flownav.trt.errors import FlowNavError

_HAS_TRT = (importlib.util.find_spec("tensorrt") is not None
            and importlib.util.find_spec("pycuda") is not None)


def test_missing_engine_raises_flownav_error(tmp_path):
    with pytest.raises(FlowNavError):
        TRTEngineRunner(tmp_path / "does_not_exist.engine")


@pytest.mark.skipif(not _HAS_TRT, reason="tensorrt/pycuda not importable")
def test_manifest_mismatch_raises(tmp_path):
    engine = tmp_path / "fake.engine"
    engine.write_bytes(b"not-a-real-engine")
    (tmp_path / "fake.engine.json").write_text('{"sm": 1, "trt_version": "0.0.0"}')
    with pytest.raises(FlowNavError):
        TRTEngineRunner(engine)
