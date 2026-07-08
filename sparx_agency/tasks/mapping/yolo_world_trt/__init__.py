"""Hardware-dependent TensorRT optimization for the YOLO-World detector.

This task turns the ultralytics YOLO-World checkpoints (``s``/``m``/``l``/``x``)
into TensorRT engines tuned for the *specific* board they will run on -- in
particular the Jetson AGX Orin, where the two **DLA** cores are used to offload
the (baked, prompt-frozen) YOLO CNN backbone/neck off the GPU at a 15 W power cap.

Pipeline (see ``README.md`` for the full walkthrough)::

    weights (.pt) --export_onnx--> static-shape ONNX
                  --build_engine--> <target_tag>/<variant>.<prec>.engine (+ .json)
                  --benchmark----> per-variant latency / FPS / DLA-vs-GPU report

The build knobs (precision, DLA core, memory pools, workspace, optimization
level) are NOT left to TensorRT defaults: :mod:`hardware` detects the board and
:mod:`build_policy` derives an explicit policy from it plus what we know about the
YOLO-World graph. The core algorithm stays ROS-free in
``core.mapping.detection``; this task owns only the engine build + runtime + bench.
"""
