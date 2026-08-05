"""FlowNav navigation policy integration (core, ROS-free).

FlowNav (UTN-AIR, IROS 2025) is a NoMaD-derived image-goal navigation policy that
replaces NoMaD's DDPM diffusion with **flow matching**: the action trajectory is
produced by integrating a learned velocity field with an explicit Euler ODE
solver over only a few steps (the "K" / ``num_steps``), which is its core speed
advantage over diffusion policies.

This package currently holds the TensorRT *runtime* half (``trt/``); the builder
(ONNX export, engine build, benchmark) lives under
``sparx_agency/tasks/planning/vlas/flownav`` because it imports torch/onnx/tensorrt and
the external FlowNav model, and must never be imported by ``core``.
"""
