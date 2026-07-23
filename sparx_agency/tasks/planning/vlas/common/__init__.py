"""Tooling shared by every VLA: hardware detection, ONNX helpers, finetune base.

Only genuinely identical code lives here. Deliberately **not** shared, because
the policies really differ: ``trt/engine/build_engine.py`` (NavDP wires INT8,
FlowNav does not), ``trt/engine/inspect_onnx.py`` (different DLA/opt-level
defaults), ``trt/export/io_spec.py`` (different tensor names and shapes) and
``trt/benchmark/bench.py`` (precision race vs K-sweep drift gate).
"""
