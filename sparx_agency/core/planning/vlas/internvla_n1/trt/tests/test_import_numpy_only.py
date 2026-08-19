"""The deployment runtime must import with numpy alone.

The build side needs torch, diffusers, onnx and TensorRT; the side that flies
must not. This is the test that keeps them apart -- it runs in the repo's
``.venv``, which has no torch at all, so an accidental top-level import fails
here rather than on the aircraft.
"""
import sys

import pytest


def test_policy_imports_without_torch_or_tensorrt():
    from sparx_agency.core.planning.vlas.internvla_n1.trt import policy  # noqa: F401

    for forbidden in ("torch", "tensorrt", "pycuda", "diffusers", "onnx",
                      "transformers"):
        assert forbidden not in sys.modules, (
            "importing the InternVLA-N1 System-1 runtime pulled in %r; it must "
            "stay numpy-only at import so it can be served without the build "
            "environment" % forbidden)


def test_submodules_import_without_torch():
    from sparx_agency.core.planning.vlas.internvla_n1.trt import (  # noqa: F401
        flow_matching, postprocess,
    )
    assert "torch" not in sys.modules


def test_engine_keys_are_the_export_contract():
    """The runner binds tensors by name, so these strings are the contract."""
    from sparx_agency.core.planning.vlas.internvla_n1.trt import policy

    assert policy.VISION_KEY == "internvla_n1_s1_vision"
    assert policy.CONDITION_KEY == "internvla_n1_s1_condition"
    assert policy.DENOISE_KEY == "internvla_n1_s1_denoise"
