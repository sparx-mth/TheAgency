"""Open-set TensorRT optimization for the YOLO-World detector (DLA-first).

Turns the ultralytics YOLO-World checkpoints (``s``/``m``/``l``/``x``) into
TensorRT engines that stay **fully open-vocabulary** -- prompts are given at run
time, never baked -- while offloading the heavy convolutional work to the Jetson
AGX Orin **DLA**. See ``README.md`` for the full walkthrough.

The model is split into two engines with opposite hardware fits, plus a text
branch that runs only when the prompt list changes::

    prompts --TextEmbedder(CLIP)--> txt_feats            (only on re-prompt; cached)
    image   --backbone engine-----> feature maps         (static, text-free -> DLA)
    (feature maps, txt_feats) --head engine--> detections (dynamic N -> GPU)

Pipeline: ``export_onnx`` (splits the graph + parity-gates the cut) -> two ONNX ->
``build_engine`` (backbone static/DLA, head dynamic-N/GPU) -> ``benchmark`` (per
-variant latency / FPS / DLA-vs-GPU). The build knobs are set explicitly from
:mod:`hardware` + :mod:`build_policy`, never left to TensorRT defaults. The core
algorithm stays ROS-free in ``core.mapping.detection``; this task owns only the
engine build + runtime + benchmark.
"""
