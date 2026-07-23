"""Executable VLA tasks: optimize, fine-tune, serve and drive each policy.

The algorithmic half of every policy lives in
:mod:`sparx_agency.core.planning.vlas` (ROS-free, numpy-only at import,
Python-3.8 safe). This package holds everything that is *not* allowed in there:
torch, onnx, TensorRT, Flask, rclpy, external model checkouts and build output.

One directory per policy, each with the same four sub-areas (present only where
they exist -- we do not ship empty scaffolding)::

    vlas/
      common/            VLA-agnostic tooling shared by every policy
        hardware/        GPU/Jetson target detection -> engine dir tag
        engine/          ONNX helpers that are identical across policies
        finetune/        the model-agnostic training machinery (labels, ESDF
                         targets, augmentation, datasets, eval, verify tools)
      <vla>/
        trt/             ONNX export -> engine build -> benchmark -> engines/
        finetune/        the policy-specific loss / model / train loop
        serve/           the inference server the ROS nodes talk to
        ros2/            ROS2 node wrappers  (ROS1 nodes live with their task)
        upstream/        files we inject into a third-party repo (not importable)

Nothing here may be imported by ``core``; the dependency arrow is one-way.

Where the platform lives
------------------------
A policy never names a robot. Topic maps, camera intrinsics and actuation
encodings belong to :mod:`sparx_agency.robots` -- e.g. the Rooster R1 / Sphera
bindings are ``robots/ROBOTICAN/config/vla/*.yaml`` plus
``robots/ROBOTICAN/adapters/``. To run an existing policy on a new drone you add
a config (and, if its actuation differs, an adapter) under ``robots/``; you do
not touch ``core/planning/vlas`` or this package.
"""
