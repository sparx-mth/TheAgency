"""Backwards-compatible shim: FP16 ONNX conversion now lives with the toolkit.

The implementation moved to
:mod:`sparx_agency.tasks.common.trt_optimizer.engine.fp16_graph` because nothing
about it is VLA-specific -- it is a property of ONNX and the TensorRT
generation -- and the network-agnostic optimizer needed it without depending on
a model package. The names are re-exported here unchanged so the NavDP and
FlowNav builders keep working untouched.

Prefer importing from ``...trt_optimizer.engine.fp16_graph`` in new code.
"""
from __future__ import annotations

from sparx_agency.tasks.common.trt_optimizer.engine.fp16_graph import (  # noqa: F401
    SENSITIVE_OPS, to_fp16_onnx,
)

__all__ = ["SENSITIVE_OPS", "to_fp16_onnx"]
