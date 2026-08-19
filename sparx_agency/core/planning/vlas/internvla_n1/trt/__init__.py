"""TensorRT deployment runtime for InternVLA-N1 System 1.

Numpy-only at import; TensorRT and pycuda are lazy-imported by the shared engine
runner. The build tooling that produces the engines lives in
``sparx_agency.tasks.planning.vlas.internvla_n1.trt`` and is not imported here.
"""
