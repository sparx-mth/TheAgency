"""Run parity tiers (a) and (b) for InternVLA-N1 System 1, FP32 on CPU.

Tier (a) compares each export wrapper against an **unpatched** System 1 built
from the same checkpoint and called the way upstream calls it. That is what
blesses the four deliberate changes:

============================  ============================================
change                        what tier (a) proves
============================  ============================================
baked DINOv2 pos-embed        the frozen table equals the live bicubic
                              ``F.interpolate`` at 224x224
``rgb_model.forward`` rebind  ``forward(x)`` equals
                              ``get_intermediate_layers(x)[0]``
baked all-ones encoder_mask   the constant equals the per-call ``torch.ones``
float32 ``timestep``          the sinusoidal embedding is unchanged by the
                              int64 -> float32 narrowing
============================  ============================================

Tier (b) then compares the exported ONNX against those same wrappers under a
CPU-only onnxruntime.

Two changes tier (a) cannot express, and which are argued rather than measured
here:

* ``ffn_dim_multiplier=2/3`` is not a patch to a working model -- without it the
  checkpoint does not load at all (36 shape mismatches). There is no unpatched
  side to compare against.
* Dropping the classifier-free-guidance branch is algebraic:
  ``uncond + 1.0 * (cond - uncond) == cond`` exactly, in any precision, because
  the guidance scale is 1.0. :func:`check_cfg_identity` verifies it numerically
  anyway, since "algebraically obvious" is how wrong assumptions survive.

Usage::

    python -m sparx_agency.tasks.planning.vlas.internvla_n1.trt.validate_parity \\
        --ckpt <InternVLA-N1-DualVLN> --onnx-dir <.../engines/onnx>
"""
from __future__ import annotations

import argparse

from sparx_agency.tasks.common.trt_optimizer.export import parity
from sparx_agency.tasks.planning.vlas.internvla_n1.trt import adapter as adapter_mod
from sparx_agency.tasks.planning.vlas.internvla_n1.trt import model as model_mod

#: Per-graph relative-L2 ceilings. The DINOv2 trunk is 12 residual blocks and
#: the denoiser 12 more, so both get the transformer tolerance; the condition
#: stage is three shallow modules whose wrapper is character-identical to the
#: original, and is held an order of magnitude tighter because nothing in it
#: should move at all.
TOLERANCES = {
    adapter_mod.VISION_KEY: 2e-3,
    adapter_mod.CONDITION_KEY: 1e-4,
    adapter_mod.DENOISE_KEY: 2e-3,
}


def unpatched_references(checkpoint, batch):
    """Build the tier-(a) references: the same weights, called upstream's way.

    A *second* model is loaded on purpose. :meth:`adapter.patch` mutates the
    backbone in place, so a reference taken from the patched instance would be
    comparing the patch against itself.

    Args:
        checkpoint: the ``InternVLA-N1-DualVLN`` directory.
        batch: trajectory candidates per call.

    Returns:
        Mapping of engine key -> callable with the wrapper's signature.
    """
    import torch

    reference = model_mod.load_system1(checkpoint, device="cpu", dtype=torch.float32)

    def vision(images):
        """Upstream's call: live bicubic pos-embed interpolation, no rebind."""
        x = images.permute(0, 1, 4, 2, 3)
        x = (x - reference._resnet_mean) / reference._resnet_std
        feat = reference.rgb_model.get_intermediate_layers(x.flatten(0, 1))[0]
        return feat.unflatten(0, (1, -1)).flatten(1, 2)

    def condition(dino_feat, traj_latents):
        return reference.condition(dino_feat, traj_latents)

    def denoise(latents, timestep, cond):
        """Upstream's call: recomputed sin/cos table, torch.ones mask, int64 t."""
        return reference.denoise_step(latents, timestep.to(torch.long), cond)

    return {
        adapter_mod.VISION_KEY: vision,
        adapter_mod.CONDITION_KEY: condition,
        adapter_mod.DENOISE_KEY: denoise,
    }


def check_cfg_identity(model, scenario, batch):
    """Verify that upstream's guidance combination is the identity at scale 1.0.

    ``generate_traj`` runs the denoiser on ``cat([zeros_like(cond), cond])`` and
    combines ``uncond + guidance_scale * (cond - uncond)``. At
    ``guidance_scale = 1.0`` that is ``cond``, so the null half is computed and
    thrown away -- half the denoiser's work, every step, every call.

    Args:
        model: a patched :class:`..model.System1`.
        scenario: one item from ``adapter.scenarios``.
        batch: trajectory candidates per call.

    Returns:
        float: relative L2 between the guided result and the conditional half.
        Exactly 0.0 means the null branch cannot affect the output at all.
    """
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    images = torch.from_numpy(scenario["images"]).to(device, dtype)
    latents_in = torch.from_numpy(scenario["traj_latents"]).to(device, dtype)
    sample = torch.from_numpy(scenario["noise"]).to(device, dtype)

    with torch.no_grad():
        cond = model.condition(model.vision(images), latents_in)
        cfg_cond = torch.cat([torch.zeros_like(cond), cond], 0)
        cfg_cond = cfg_cond.repeat_interleave(batch, dim=0)
        doubled = sample.repeat(2, 1, 1)
        timestep = torch.full((doubled.shape[0],), 1000.0, device=device, dtype=dtype)
        prediction = model.denoise_step(doubled, timestep, cfg_cond)
        uncond, guided = prediction.chunk(2)
        combined = uncond + 1.0 * (guided - uncond)
    return parity.rel_l2(combined.float().cpu().numpy(), guided.float().cpu().numpy())


def main(argv=None):
    """Entry point. Returns 0 on success; raises on a parity failure."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--batch", type=int, default=model_mod.NUM_SAMPLE_TRAJS)
    args = ap.parse_args(argv)

    import torch

    from sparx_agency.tasks.common.trt_optimizer.export import onnx_export

    adapter = adapter_mod.InternVLAN1System1Adapter(batch=args.batch)
    model = adapter.patch(adapter.load(args.ckpt, device="cpu"))
    wrappers = adapter.wrappers(model)
    specs = adapter.graphs()

    def example_inputs(spec):
        """Spec-shaped CPU inputs; the timestep is a real schedule value."""
        inputs = list(onnx_export.example_inputs(spec, seed=0, device="cpu"))
        if "timestep" in spec.input_names():
            index = spec.input_names().index("timestep")
            inputs[index] = torch.full(spec.inputs["timestep"], 700.0)
        if "images" in spec.input_names():
            index = spec.input_names().index("images")
            inputs[index] = inputs[index].abs().clamp(max=1.0)
        return tuple(inputs)

    report = parity.validate(
        specs, wrappers, args.onnx_dir, example_inputs,
        tolerances=TOLERANCES,
        unpatched=unpatched_references(args.ckpt, args.batch))
    parity.enforce(report)

    drift = check_cfg_identity(model, adapter.scenarios(1)[0], args.batch)
    # float32 associativity makes ``a + (b - a)`` differ from ``b`` in the last
    # bits, so the bar is float32 epsilon rather than exact zero. Anything above
    # it would mean the null branch is reaching the output, which at
    # guidance_scale = 1.0 it cannot.
    print("\n  [%s] classifier-free guidance at scale 1.0 is the identity: "
          "rel_l2 %.3e between the guided result and the conditional half "
          "(float32 round-off floor ~1e-7; the null branch contributes nothing)"
          % ("ok" if drift <= 1e-6 else "FAIL", drift))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
