"""Vision-language-action / learned navigation policies (ROS-free runtime half).

A **VLA** here is a learned policy that consumes egocentric sensing plus a goal
and emits an *action or trajectory*. That is the boundary: perception backbones
that emit depth, boxes or masks (DepthAnything, YOLO-World, NanoOWL) are **not**
VLAs and stay under :mod:`sparx_agency.core.mapping`; classical planners
(A*, RRT*, pure pursuit) stay under their own ``core/planning`` packages.

Layout -- one directory per policy, all platform-agnostic::

    vlas/
      interfaces/   the NavigationPolicy contract + goal modality types
      registry.py   name -> policy factory (lazy; heavy deps never imported here)
      common/       shared, VLA-agnostic runtime: errors, image codec, HTTP base,
                    the single TensorRT engine runner, and plan_commit -- fly
                    one prediction as a route before asking for another
      navdp/        point-goal diffusion policy    (client + geometry + trt/)
      flownav/      image-goal flow-matching policy (client + preprocess + trt/)
      internvla_n1/ dual-system VLN policy          (client + types)
      omnivla/      multi-modal-goal VLA            (numpy goal/action codec)

Everything here is **numpy-only at import time**. The heavy halves live in
:mod:`sparx_agency.tasks.planning.vlas` -- TensorRT export/build (``<vla>/trt/``),
fine-tuning (``<vla>/finetune/``), inference servers (``<vla>/serve/``) and the
ROS nodes (``<vla>/ros2/``). Platform bindings (topics, intrinsics, actuation)
live under :mod:`sparx_agency.robots`.

Two hard rules, both enforced by ``core/planning/vlas/*/trt/tests/
test_import_numpy_only.py``:

1. **No eager heavy imports.** ``tensorrt`` / ``pycuda`` / ``torch`` / ``requests``
   / ``PIL`` are imported *inside methods*. The FALCON ROS1 adapter imports this
   package inside a Noetic container that has none of them.
2. **Python 3.8 syntax.** That same container runs 3.8: no PEP 604 ``X | Y``
   outside ``from __future__ import annotations``, no ``match``/``case``, no
   ``@dataclass(slots=True)``.

This module deliberately re-exports **nothing** from the per-policy packages --
importing :mod:`sparx_agency.core.planning.vlas` must stay free. Import the
policy you need directly, e.g.
``from sparx_agency.core.planning.vlas.navdp import NavDPPointgoalClient``.
"""
