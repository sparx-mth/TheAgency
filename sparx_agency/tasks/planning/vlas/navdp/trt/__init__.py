"""NavDP TensorRT build tooling: ONNX export, engine build, benchmark.

Produces the artifacts in ``engines/<hardware_tag>/`` that the numpy runtime in
``core/planning/vlas/navdp/trt`` deserializes at serve time. Needs torch, onnx,
tensorrt and the external NavDP checkout; host/dev only.
"""
