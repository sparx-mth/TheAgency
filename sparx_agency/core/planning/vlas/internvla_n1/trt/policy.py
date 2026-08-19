"""TensorRT drop-in for InternVLA-N1's System-1 trajectory policy.

Replaces the three heavy forwards of ``InternVLAN1ForCausalLM.generate_traj``
(``system1 = "nextdit_async"``) with TensorRT engines, keeping every stochastic
and data-dependent part in numpy so a difference is attributable to the engines
and to nothing else::

    vision    : images (1,2,224,224,3)                     -> dino_feat (1,512,384)
    condition : (dino_feat, traj_latents (1,4,3584))       -> condition (1,36,768)
    denoise   : (latents (32,32,3), timestep (32,),
                 condition (32,36,768))                    -> velocity (32,32,3)   [10x]

Two departures from upstream, both deliberate and both measured:

* **The classifier-free-guidance branch is gone.** ``generate_traj`` runs the
  denoiser on ``cat([zeros_like(cond), cond])`` and combines with
  ``guidance_scale = 1.0``, where ``uncond + 1.0 * (cond - uncond) == cond``. The
  null half is computed and discarded, so the DiT runs at batch 64 to produce a
  batch-32 answer. Dropping it is algebraically exact.
* **The engines may be mixed precision.** ``selected.json`` names one engine per
  graph and each carries its own precision, because TensorRT 11 is strongly
  typed. The condition graph is pinned FP32; see the build package's README.

Numpy-only at import: TensorRT and pycuda are lazy-imported by the shared
:class:`TRTEngineRunner`, and nothing here needs torch. Python-3.8 clean, so it
survives the Noetic container.

This is single-robot only. The engines are built static at
``num_sample_trajs = 32``, so a second robot needs its own process.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sparx_agency.core.planning.vlas.common.trt.engine_runner import TRTEngineRunner
from sparx_agency.core.planning.vlas.internvla_n1.trt import flow_matching, postprocess

#: Engine keys. The ONNX export must use exactly these names -- the runner binds
#: tensors by name, so this is the contract, not a convention.
VISION_KEY = "internvla_n1_s1_vision"
CONDITION_KEY = "internvla_n1_s1_condition"
DENOISE_KEY = "internvla_n1_s1_denoise"

#: Input tensor names, per graph.
IN_IMAGES = "images"
IN_DINO_FEAT = "dino_feat"
IN_TRAJ_LATENTS = "traj_latents"
IN_LATENTS = "latents"
IN_TIMESTEP = "timestep"
IN_CONDITION = "condition"

#: Geometry the engines were built static at.
MEMORY_FRAMES = 2
IMAGE_SIZE = 224
VLM_DIM = 3584
N_QUERY = 4
PREDICT_STEPS = 32
DEFAULT_SAMPLES = 32
DEFAULT_STEPS = 10

#: ``sys1_depth_threshold`` in ``internvla_n1_agent``. Depth is not an engine
#: input for this System-1 variant -- ``nextdit_async`` conditions on RGB alone --
#: but the constant is kept here so a caller preparing observations has one place
#: to read it from.
DEPTH_CLIP_M = 5.0


class InternVLAN1System1Error(RuntimeError):
    """Raised when the engines, their selection, or an input is unusable."""


class InternVLAN1System1TRT:
    """Run InternVLA-N1 System 1 on TensorRT engines.

    Args:
        engine_dir: directory holding ``selected.json`` and the engines it names.
        samples: trajectory candidates per call. Must match the built engines.
        steps: flow-matching Euler steps. Free to change **without a rebuild** --
            the denoise engine is one step, not an unrolled loop -- and it is the
            cheapest behavioural lever this policy has.
        seed: RNG seed for the initial trajectory noise. Pass one for a
            reproducible comparison; leave None to draw fresh noise per call, as
            upstream does.
        device_id: CUDA device index.

    Raises:
        InternVLAN1System1Error: if the selection is missing or names an engine
            that is not there. A partially loaded pipeline would produce a
            confident, wrong trajectory.
    """

    def __init__(self, engine_dir, samples=DEFAULT_SAMPLES, steps=DEFAULT_STEPS,
                 seed=None, device_id=0):
        self.engine_dir = Path(engine_dir)
        self.samples = int(samples)
        self.steps = int(steps)
        self._rng = np.random.default_rng(seed)
        selection = self._read_selection()
        self.precisions = selection.get("precisions", {})
        self.runners = {}
        for key in (VISION_KEY, CONDITION_KEY, DENOISE_KEY):
            name = selection["engines"].get(key)
            if name is None:
                raise InternVLAN1System1Error(
                    "selected.json in %s names no engine for %r; the pipeline "
                    "cannot be completed without it" % (self.engine_dir, key))
            self.runners[key] = TRTEngineRunner(self.engine_dir / name,
                                                device_id=device_id)

    def _read_selection(self):
        """Load ``selected.json``, or explain why the engines may not be served.

        Its absence is the fail-safe working: the race withdraws the selection
        when no precision passed its quality gate, so "no selection" means
        "nothing was blessed", not "look for engines yourself".
        """
        path = self.engine_dir / "selected.json"
        if not path.is_file():
            raise InternVLAN1System1Error(
                "no selected.json in %s. Either the engines were never built, or "
                "the precision race withdrew the selection because no candidate "
                "passed its quality gate -- in which case there is deliberately "
                "nothing here to serve." % self.engine_dir)
        payload = json.loads(path.read_text())
        if not payload.get("engines"):
            raise InternVLAN1System1Error(
                "selected.json in %s lists no engines" % self.engine_dir)
        return payload

    # -- the three stages ------------------------------------------------
    def encode_vision(self, images):
        """RGB memory stack -> DINOv2 patch features.

        Args:
            images: ``(1, 2, 224, 224, 3)`` float32 in ``[0, 1]``, pixel-goal
                frame first and current frame second, exactly as
                ``internvla_n1_agent.step`` stacks them.

        Returns:
            ``(1, 512, 384)`` float32.

        Raises:
            InternVLAN1System1Error: on a shape the engine cannot accept.
        """
        images = _as_f32(images, (1, MEMORY_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3),
                         "images")
        return _one(self.runners[VISION_KEY].infer({IN_IMAGES: images}))

    def encode_condition(self, dino_feat, traj_latents):
        """Patch features + System-2 latents -> the cross-attention condition.

        Args:
            dino_feat: ``(1, 512, 384)`` from :meth:`encode_vision`.
            traj_latents: ``(1, 4, 3584)`` from System 2's ``generate_latents``.

        Returns:
            ``(1, 36, 768)`` float32.
        """
        traj_latents = _as_f32(traj_latents, (1, N_QUERY, VLM_DIM), "traj_latents")
        return _one(self.runners[CONDITION_KEY].infer({
            IN_DINO_FEAT: np.ascontiguousarray(dino_feat, dtype=np.float32),
            IN_TRAJ_LATENTS: traj_latents}))

    def denoise(self, condition, noise=None):
        """Run the flow-matching loop and return the candidate trajectories.

        The condition is uploaded once, broadcast to the candidate batch, and
        left resident: only ``latents`` and ``timestep`` change between steps.

        Args:
            condition: ``(1, 36, 768)`` from :meth:`encode_condition`.
            noise: optional ``(samples, 32, 3)`` initial sample. Pass one to
                compare two runtimes on identical noise; otherwise it is drawn.

        Returns:
            ``(samples, 32, 3)`` float32 candidate trajectory deltas.
        """
        broadcast = np.ascontiguousarray(
            np.repeat(np.asarray(condition, dtype=np.float32), self.samples, axis=0))
        if noise is None:
            sample = self._rng.standard_normal(
                (self.samples, PREDICT_STEPS, 3)).astype(np.float32)
        else:
            sample = _as_f32(noise, (self.samples, PREDICT_STEPS, 3), "noise")
        runner = self.runners[DENOISE_KEY]
        for sigma, sigma_next, timestep in flow_matching.schedule(self.steps):
            velocity = _one(runner.infer({
                IN_LATENTS: sample,
                IN_TIMESTEP: np.full((self.samples,), timestep, dtype=np.float32),
                IN_CONDITION: broadcast}))
            sample = flow_matching.euler_step(sample, velocity, sigma, sigma_next)
        return sample

    # -- the deployed entry point ---------------------------------------
    def predict_actions(self, images, traj_latents, noise=None):
        """One System-1 decision: observations in, executable action queue out.

        This is the whole of ``s1_step_latent`` with ``continuous_traj=True``.

        Args:
            images: ``(1, 2, 224, 224, 3)`` float32 in ``[0, 1]``.
            traj_latents: ``(1, 4, 3584)`` float32 from System 2.
            noise: optional initial trajectory noise, for reproducibility.

        Returns:
            Tuple of ``(actions, trajectories)`` -- at most four action indices
            (1 forward, 2 left, 3 right), and the ``(samples, 32, 3)`` candidate
            deltas they were derived from, kept for logging and visualisation.
        """
        dino_feat = self.encode_vision(images)
        condition = self.encode_condition(dino_feat, traj_latents)
        trajectories = self.denoise(condition, noise=noise)
        return postprocess.action_queue(trajectories), trajectories


def _one(outputs):
    """The single output array from a runner result of any container shape."""
    if isinstance(outputs, dict):
        return list(outputs.values())[0]
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def _as_f32(array, shape, name):
    """Contiguous float32 view of ``array``, with its shape checked.

    Raises:
        InternVLAN1System1Error: naming the tensor and both shapes. The engines
            are static, so a wrong shape is a caller bug and never something to
            reshape around.
    """
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.shape != tuple(shape):
        raise InternVLAN1System1Error(
            "%s has shape %s but the engine was built static at %s"
            % (name, array.shape, tuple(shape)))
    return array
