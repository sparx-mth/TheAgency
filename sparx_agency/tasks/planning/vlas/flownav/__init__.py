"""FlowNav TensorRT builder + benchmark tooling (dev/host only).

Imports torch / onnx / tensorrt and the external FlowNav model, so it must NEVER
be imported by ``core`` (which stays torch-free and Python-3.8 compatible for the
FALCON Noetic adapter). The torch-free numpy runtime lives under
``sparx_agency/core/planning/vlas/flownav/trt``. See ``README.md`` for the two-stage
export -> build -> benchmark flow and the with/without-TRT comparison.
"""
