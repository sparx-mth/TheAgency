"""The :class:`ModelAdapter` for InternVLA-N1 System 1.

**System 2 is deliberately absent from every graph here**, for two independent
reasons either of which is sufficient: it is a Qwen2.5-VL-7B that generates
autoregressively behind a KV cache (``decide`` rule 2 -> ``llm_runtime``, never
an ONNX graph), and its 16.58 GB of bf16 weights do not fit on the 8 GB target
at all. See the report's "Deliberately not converted" section.

What is left is 91.4 M parameters that run **twice as often as System 2 does**
and hold 94% of a System-1 call in a ten-step denoise loop. That is the graph
worth converting, and this adapter describes it.

The gate is the one part of this file worth reading twice. ``s1_step_latent``
does not *select* among the 32 candidate trajectories -- ``traj_to_actions``
averages them, integrates the mean into a path, then walks that path into a
list of discrete actions, of which the agent executes the first. So "the same
answer" is not trajectory L2 and not an argmax flip: it is **the emitted action
sequence**, and above all its first element, because that is the command the
robot acts on.
"""
from __future__ import annotations

from sparx_agency.tasks.common.trt_optimizer import adapter as adapter_mod
from sparx_agency.tasks.common.trt_optimizer.spec import Cadence, GraphSpec
from sparx_agency.tasks.planning.vlas.internvla_n1.trt import model as model_mod
from sparx_agency.tasks.planning.vlas.internvla_n1.trt import wrappers as wrap_mod

#: Engine keys. Also the ONNX stems, the engine stems and the report rows.
VISION_KEY = "internvla_n1_s1_vision"
CONDITION_KEY = "internvla_n1_s1_condition"
DENOISE_KEY = "internvla_n1_s1_denoise"

#: One System-1 call yields at most 4 actions (``S1Output(idx=action_list[:4])``),
#: consumed one per control decision, so System 1 runs once every four of them.
S1_CALLS_PER_DECISION = 0.25

#: ``trajectory_to_discrete_actions_close_to_goal`` defaults, from
#: ``internnav/model/utils/vln_utils.py``. Reproduced rather than imported so
#: the gate does not silently change when upstream does.
STEP_SIZE_M = 0.25
TURN_ANGLE_DEG = 15
LOOKAHEAD = 4
GOAL_TOLERANCE_M = 0.2
#: ``traj_to_actions`` divides dx/dy by this before integrating.
DELTA_SCALE = 4.0
#: How many actions the agent keeps from one System-1 call.
ACTIONS_KEPT = 4


class InternVLAN1System1Adapter(adapter_mod.ModelAdapter):
    """Optimize InternVLA-N1's System-1 trajectory policy for TensorRT.

    Args:
        batch: trajectory candidates per call. Defaults to 32 -- upstream runs
            the DiT at 64 only because it builds a classifier-free-guidance
            branch it then cancels at ``guidance_scale = 1.0``.
        steps: flow-matching Euler steps per call.
        precision_sensitive_vision: build the DINOv2 trunk FP32. Left False so
            the precision race can answer it with a measurement; a deep ViT
            residual stream is the classic case for FP32, but this one is 12
            blocks at d=384, not 24 at d=1024.
    """

    name = "internvla_n1_s1"

    def __init__(self, batch=model_mod.NUM_SAMPLE_TRAJS,
                 steps=model_mod.NUM_INFERENCE_STEPS,
                 precision_sensitive_vision=False):
        self.batch = int(batch)
        self.steps = int(steps)
        self.precision_sensitive_vision = bool(precision_sensitive_vision)

    # -- Model -----------------------------------------------------------
    def load(self, checkpoint, device="cpu"):
        """Load System 1 from the released checkpoint, FP32, in eval mode.

        FP32 rather than the checkpoint's bf16: the export and both CPU parity
        tiers are FP32, and the engine's precision comes from the ONNX, not from
        the dtype the reference happened to be loaded in.

        Args:
            checkpoint: the ``InternVLA-N1-DualVLN`` directory.
            device: torch device string.

        Returns:
            A :class:`..model.System1`.
        """
        import torch
        return model_mod.load_system1(checkpoint, device=device, dtype=torch.float32)

    def cadences(self):
        """Declared cadences; the profiler counts real calls and overrides these."""
        return {
            "rgb_model": Cadence.PER_FRAME,
            "memory_encoder": Cadence.PER_FRAME,
            "rgb_resampler": Cadence.PER_FRAME,
            "cond_projector": Cadence.PER_FRAME,
            "action_encoder": Cadence.PER_STEP,
            "action_decoder": Cadence.PER_STEP,
            "traj_dit": Cadence.PER_STEP,
        }

    # -- Graphs ----------------------------------------------------------
    def graphs(self):
        """Three engines, split where the cadence changes.

        ``vision`` and ``condition`` are separate despite sharing a cadence:
        the DINOv2 trunk is the one graph whose FP16 numerics are in question,
        so keeping it its own engine lets the precision race answer that for it
        alone instead of for the whole per-frame stage.
        """
        return [
            GraphSpec(
                key=VISION_KEY,
                inputs={"images": (1, model_mod.MEMORY_FRAMES, 224, 224, 3)},
                outputs=["dino_feat"],
                component="rgb_model",
                cadence=Cadence.PER_FRAME,
                calls_per_decision=S1_CALLS_PER_DECISION,
                precision_sensitive=self.precision_sensitive_vision,
                notes="DepthAnythingV2 DINOv2-S encoder, both memory frames in "
                      "one batch; positional embedding baked at 224x224.",
            ),
            GraphSpec(
                key=CONDITION_KEY,
                inputs={
                    "dino_feat": (1, model_mod.MEMORY_FRAMES * model_mod.PATCH_TOKENS, 384),
                    "traj_latents": (1, model_mod.N_QUERY, model_mod.VLM_DIM),
                },
                outputs=["condition"],
                component="memory_encoder",
                cadence=Cadence.PER_FRAME,
                calls_per_decision=S1_CALLS_PER_DECISION,
                # Not a judgement about FP16 -- a measurement about TensorRT.
                # The identical FP16 ONNX runs at 3.1e-4 relative L2 under
                # onnxruntime and at 3.6e-1 as a TensorRT 11.1 engine on sm_120,
                # identically at builder optimization levels 0, 1 and 2. The
                # output keeps the reference's mean and standard deviation while
                # all 32 QFormer memory tokens are wrong by ~1.2 against a std
                # of 0.96; the 4 cond_projector tokens in the same graph are
                # correct to 8e-4. It costs 1.0% of the decision to pin it.
                precision_sensitive=True,
                notes="MemoryEncoder + QFormer resampler + cond_projector; the "
                      "only place a System-2 tensor enters System 1. Pinned "
                      "FP32: TensorRT miscompiles this graph in FP16 (see the "
                      "report's negative results).",
            ),
            GraphSpec(
                key=DENOISE_KEY,
                inputs={
                    "latents": (self.batch, model_mod.PREDICT_STEPS, 3),
                    "timestep": (self.batch,),
                    "condition": (self.batch, model_mod.N_QUERY + 32, model_mod.LATENT_EMB),
                },
                outputs=["velocity"],
                component="traj_dit",
                cadence=Cadence.PER_STEP,
                calls_per_decision=S1_CALLS_PER_DECISION * self.steps,
                notes="ONE Euler step, never %d unrolled -- unrolling freezes "
                      "the step count, which is the cheapest lever here."
                      % self.steps,
            ),
        ]

    def wrappers(self, model):
        """Engine key -> export wrapper."""
        return {
            VISION_KEY: wrap_mod.vision_wrapper(model),
            CONDITION_KEY: wrap_mod.condition_wrapper(model),
            DENOISE_KEY: wrap_mod.denoise_wrapper(model, batch=self.batch),
        }

    def patch(self, model):
        """Bake the DINOv2 positional embedding for the fixed 224x224 input.

        DINOv2 was pretrained at 518x518 with patch 14, so a 224x224 input hits
        the **bicubic** ``F.interpolate`` branch of ``interpolate_pos_encoding``
        on every forward. That traces to an ONNX ``Resize`` -- which the op gate
        rejects and TensorRT handles badly -- and its answer never changes,
        because the deployed input size never changes.

        Args:
            model: a :class:`..model.System1`, modified in place.

        Returns:
            The same model.
        """
        from sparx_agency.tasks.common.trt_optimizer.export import patches
        patches.bake_pos_embed(model.rgb_model, input_hw=(224, 224), patch_size=14,
                               num_register_tokens=0)
        return model

    # -- Execution -------------------------------------------------------
    def scenarios(self, count, seed=0):
        """Representative System-1 inputs.

        Args:
            count: how many scenarios to produce.
            seed: RNG seed; the same seed gives the same scenarios in every
                process, so a reference computed by one and an engine run by
                another are comparable.

        Returns:
            A list of dicts with ``images``, ``traj_latents`` and ``noise``.

        Note:
            These are synthetic. Real captures would be better -- uniform noise
            is out of distribution for a pretrained DINOv2 -- but this repo has
            no InternVLA-N1 rosbags yet, and a *shared* System-2 latent cannot
            be synthesised at all without running System 2. The report says so
            rather than implying otherwise.
        """
        import numpy as np

        rng = np.random.default_rng(int(seed))
        cases = []
        for _ in range(int(count)):
            cases.append({
                "images": rng.random(
                    (1, model_mod.MEMORY_FRAMES, 224, 224, 3), dtype=np.float32),
                "traj_latents": rng.standard_normal(
                    (1, model_mod.N_QUERY, model_mod.VLM_DIM)).astype(np.float32),
                # The initial trajectory noise is injected, not drawn inside the
                # loop, so reference and engine denoise the *same* sample and a
                # difference is attributable to the engine alone.
                "noise": rng.standard_normal(
                    (self.batch, model_mod.PREDICT_STEPS, 3)).astype(np.float32),
            })
        return cases

    def run_reference(self, model, scenario):
        """One complete System-1 decision in torch: vision, condition, 10 steps.

        Args:
            model: the loaded :class:`..model.System1`.
            scenario: one item from :meth:`scenarios`.

        Returns:
            ``(batch, 32, 3)`` numpy array of candidate trajectory deltas --
            exactly what ``generate_traj`` hands to ``traj_to_actions``.
        """
        import numpy as np
        import torch

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        images = torch.from_numpy(scenario["images"]).to(device, dtype)
        latents_in = torch.from_numpy(scenario["traj_latents"]).to(device, dtype)
        sample = torch.from_numpy(scenario["noise"]).to(device, dtype)

        with torch.no_grad():
            condition = model.condition(model.vision(images), latents_in)
            condition = condition.repeat_interleave(self.batch, dim=0)
            for sigma, sigma_next, timestep in euler_schedule(self.steps):
                step = torch.full((self.batch,), float(timestep), device=device, dtype=dtype)
                velocity = model.denoise_step(sample, step, condition)
                sample = (sample.float() + (sigma_next - sigma) * velocity.float()).to(dtype)
        return sample.float().cpu().numpy()

    def run_engines(self, runtimes, scenario):
        """The same decision with the built engines; the loop stays in numpy.

        Args:
            runtimes: engine key -> a loaded engine runner.
            scenario: one item from :meth:`scenarios`.

        Returns:
            ``(batch, 32, 3)`` numpy array, comparable element-for-element with
            :meth:`run_reference`.
        """
        import numpy as np

        dino = _first(runtimes[VISION_KEY].infer(
            {"images": scenario["images"]}))
        condition = _first(runtimes[CONDITION_KEY].infer(
            {"dino_feat": dino, "traj_latents": scenario["traj_latents"]}))
        condition = np.ascontiguousarray(
            np.repeat(condition, self.batch, axis=0), dtype=np.float32)

        sample = scenario["noise"].astype(np.float32)
        for sigma, sigma_next, timestep in euler_schedule(self.steps):
            step = np.full((self.batch,), float(timestep), dtype=np.float32)
            velocity = _first(runtimes[DENOISE_KEY].infer(
                {"latents": sample, "timestep": step, "condition": condition}))
            sample = sample + (sigma_next - sigma) * np.asarray(velocity, dtype=np.float32)
        return sample

    # -- Quality ---------------------------------------------------------
    def decision_metrics(self, reference, candidate):
        """Compare two System-1 decisions the way the agent consumes them.

        ``traj_to_actions`` averages the 32 candidates, integrates the mean into
        a path and walks it into discrete actions; the agent executes the first
        and queues the next three. So the metrics that matter are about that
        list, not about the tensors it came from.

        Args:
            reference: ``(batch, 32, 3)`` from :meth:`run_reference`.
            candidate: ``(batch, 32, 3)`` from :meth:`run_engines`.

        Returns:
            ``first_action_match`` and ``action_seq_match`` are the gated
            decision metrics; ``action_count_delta``, ``endpoint_err_m`` and
            ``traj_rel_l2`` are diagnostics and are never gated.
        """
        import numpy as np

        ref_path = mean_path(reference)
        cand_path = mean_path(candidate)
        ref_actions = discrete_actions(ref_path)[:ACTIONS_KEPT]
        cand_actions = discrete_actions(cand_path)[:ACTIONS_KEPT]

        first_match = float(
            bool(ref_actions) == bool(cand_actions)
            and (not ref_actions or ref_actions[0] == cand_actions[0]))
        denominator = np.linalg.norm(reference.astype(np.float64)) + 1e-12
        return {
            "first_action_match": first_match,
            "action_seq_match": float(ref_actions == cand_actions),
            "action_count_delta": float(len(cand_actions) - len(ref_actions)),
            "endpoint_err_m": float(np.linalg.norm(ref_path[-1] - cand_path[-1])),
            "traj_rel_l2": float(
                np.linalg.norm((candidate - reference).astype(np.float64)) / denominator),
        }

    def gates(self):
        """Thresholds an engine must clear before it may be served.

        ``first_action_match`` is 1.0 -- an exact requirement, not a rate --
        because a single scenario is one decision and the metric is already
        binary; the harness averages it across scenarios, so 1.0 means *every*
        scenario agreed on the command the robot executes. ``action_seq_match``
        is allowed to slip: the tail of the queue is re-derived on the next
        System-1 call anyway, and a 4th action differing is not a different
        immediate behaviour.

        A quantized format must clear a stricter bar than FP16; that ordering is
        applied by the precision race, which gates every candidate on this same
        mapping and then prefers the faster passing row.
        """
        return {
            "first_action_match": (">=", 1.0),
            "action_seq_match": (">=", 0.95),
        }


def euler_schedule(steps):
    """The ``FlowMatchEulerDiscreteScheduler`` schedule, as plain arithmetic.

    ``generate_traj`` builds the scheduler with
    ``sigmas = np.linspace(1.0, 1/steps, steps)`` and a terminal 0.0, and
    ``step`` reduces to ``sample + (sigma_next - sigma) * velocity``. Reproduced
    here so the numpy engine path and the torch reference share one schedule and
    neither depends on diffusers at run time.

    Args:
        steps: number of Euler steps.

    Yields:
        ``(sigma, sigma_next, timestep)`` triples, one per step.
    """
    sigmas = [1.0 - i * (1.0 - 1.0 / steps) / (steps - 1) for i in range(steps)]
    sigmas.append(0.0)
    for i in range(steps):
        yield sigmas[i], sigmas[i + 1], sigmas[i] * 1000.0


def mean_path(deltas):
    """Integrate candidate deltas into the single path the agent acts on.

    ``traj_to_actions`` unnormalises dx/dy by 4, cumulatively sums them per
    candidate, then averages the *paths* -- averaging after integration, not
    before, which is not the same thing.

    Args:
        deltas: ``(B, T, 3)`` predicted per-step deltas.

    Returns:
        ``(T + 1, 2)`` mean XY path starting at the origin.
    """
    import numpy as np

    xy = np.asarray(deltas, dtype=np.float64)[:, :, :2] / DELTA_SCALE
    paths = np.zeros((xy.shape[0], xy.shape[1] + 1, 2))
    paths[:, 1:] = np.cumsum(xy, axis=1)
    return paths.mean(axis=0)


def discrete_actions(path):
    """``trajectory_to_discrete_actions_close_to_goal``, reproduced exactly.

    Args:
        path: ``(T + 1, 2)`` XY path from :func:`mean_path`.

    Returns:
        List of action indices: 1 forward, 2 turn left, 3 turn right.
    """
    import numpy as np

    actions = []
    yaw = 0.0
    pos = path[0]
    goal = path[-1]
    turn = np.deg2rad(TURN_ANGLE_DEG)

    def normalize(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # Upstream's loop has no iteration cap; a degenerate path could spin here,
    # so the gate bounds it at the queue depth it can possibly consume.
    for _ in range(4 * len(path)):
        if np.linalg.norm(pos - goal) <= GOAL_TOLERANCE_M:
            break
        nearest = int(np.argmin(np.linalg.norm(path - pos, axis=1)))
        target = path[min(nearest + LOOKAHEAD, len(path) - 1)]
        direction = target - pos
        if np.linalg.norm(direction) < 1e-6:
            break
        delta_yaw = normalize(np.arctan2(direction[1], direction[0]) - yaw)
        turns = int(round(delta_yaw / turn))
        if turns > 0:
            actions += [2] * turns
        elif turns < 0:
            actions += [3] * (-turns)
        yaw = normalize(yaw + turns * turn)
        next_pos = pos + STEP_SIZE_M * np.array([np.cos(yaw), np.sin(yaw)])
        if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
            break
        actions.append(1)
        pos = next_pos
    return actions


def _first(outputs):
    """Return the single output tensor from a runner result of any shape."""
    if isinstance(outputs, dict):
        return list(outputs.values())[0]
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


adapter_mod.register("internvla_n1_s1", lambda: InternVLAN1System1Adapter())
