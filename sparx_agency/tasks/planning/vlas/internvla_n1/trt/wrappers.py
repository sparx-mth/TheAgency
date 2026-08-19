"""Export wrappers: one ``nn.Module`` per engine, positional inputs only.

These are not the original submodules. Each one flattens the keyword arguments
upstream passes, bakes the constants that would otherwise trace as runtime ops,
and presents the exact tensor list its :class:`GraphSpec` declares. Each also
holds **only the submodules its own graph runs** -- not the whole
:class:`..model.System1` -- so the parameter count the exporter records in the
manifest is this engine's, which is what the post-build precision verification
divides the engine's weight bytes by. Every
deliberate change here is blessed by parity tier (a) -- wrapper against the
original module, FP32 on CPU -- before anything is built.

The split follows **cadence**, not source structure:

* :class:`VisionWrapper` and :class:`ConditionWrapper` run **once per System-1
  call**;
* :class:`DenoiseWrapper` runs **ten times inside it**.

That boundary is why the cross-attention condition is an *input* to the denoiser
rather than something it recomputes: it is uploaded once and stays resident on
the device for all ten steps.
"""
from __future__ import annotations

from sparx_agency.tasks.planning.vlas.internvla_n1.trt import model as model_mod


def _module_base():
    """Return ``torch.nn.Module``; imported lazily so this module needs no torch."""
    import torch.nn as nn
    return nn.Module


def vision_wrapper(system1):
    """RGB memory stack -> DINOv2 patch features.

    The ImageNet normalisation and the ``(B,T,H,W,C) -> (B*T,C,H,W)`` reshape
    live inside the graph, so the runtime feeds raw ``[0, 1]`` RGB exactly as
    ``internvla_n1_agent.step`` prepares it and no host-side arithmetic sits
    between the camera and the engine.

    Args:
        system1: a :class:`..model.System1`.

    Returns:
        An ``nn.Module`` taking ``images (1,2,224,224,3)`` and returning
        ``dino_feat (1,512,384)``.
    """
    nn_module = _module_base()

    class VisionWrapper(nn_module):
        def __init__(self, net):
            super().__init__()
            self.rgb_model = net.rgb_model
            self.register_buffer("mean", net._resnet_mean, persistent=False)
            self.register_buffer("std", net._resnet_std, persistent=False)

        def forward(self, images):
            x = images.permute(0, 1, 4, 2, 3)
            x = (x - self.mean) / self.std
            feat = self.rgb_model(x.flatten(0, 1))
            return feat.unflatten(0, (1, -1)).flatten(1, 2)

    return VisionWrapper(system1).eval()


def condition_wrapper(system1):
    """Patch features + System-2 latents -> the cross-attention condition.

    Args:
        system1: a :class:`..model.System1`.

    Returns:
        An ``nn.Module`` taking ``dino_feat (1,512,384)`` and
        ``traj_latents (1,4,3584)``, returning ``condition (1,36,768)``.
    """
    import torch

    nn_module = _module_base()

    class ConditionWrapper(nn_module):
        def __init__(self, net):
            super().__init__()
            self.memory_encoder = net.memory_encoder
            self.rgb_resampler = net.rgb_resampler
            self.cond_projector = net.cond_projector

        def forward(self, dino_feat, traj_latents):
            memory = self.memory_encoder(dino_feat)
            tokens = self.rgb_resampler(torch.cat([dino_feat, memory], dim=-1))
            return torch.cat([tokens, self.cond_projector(traj_latents)], dim=1)

    return ConditionWrapper(system1).eval()


def denoise_wrapper(system1, batch=model_mod.NUM_SAMPLE_TRAJS):
    """One flow-matching Euler step, as a single static graph.

    Three deliberate changes, each blessed by the parity gate:

    * **The sinusoidal position table is baked.** It depends only on
      ``PREDICT_STEPS``, so upstream recomputes 32 sin/cos rows on every one of
      the ten steps of every call.
    * **The all-ones ``encoder_mask`` is baked.** Upstream materialises it with
      ``torch.ones`` per call, which traces to a ``ConstantOfShape``; the values
      cannot vary because the condition has no padding.
    * **``timestep`` arrives as float32, not int64.** ``Timesteps`` casts to
      float on its first line, so the embedding is bit-identical, and TensorRT
      would otherwise carry an int64 tensor it must narrow at the boundary.

    **One step, never ten unrolled.** Unrolling would multiply build time and
    engine size and freeze the step count into the engine -- and the step count
    is the cheapest behavioural lever this policy has.

    Args:
        system1: a :class:`..model.System1`.
        batch: trajectory candidates per call. ``NUM_SAMPLE_TRAJS`` (32), not
            64: upstream's classifier-free-guidance branch runs at
            ``guidance_scale = 1.0``, where
            ``uncond + 1.0 * (cond - uncond) == cond`` -- the null half is
            computed and discarded. See the report.

    Returns:
        An ``nn.Module`` taking ``latents (B,32,3)``, ``timestep (B,)`` and
        ``condition (B,36,768)``, returning ``velocity (B,32,3)``.
    """
    import torch

    nn_module = _module_base()

    class DenoiseWrapper(nn_module):
        def __init__(self, net, batch):
            super().__init__()
            self.action_encoder = net.action_encoder
            self.action_decoder = net.action_decoder
            self.traj_dit = net.traj_dit
            reference = next(net.parameters())
            positions = (torch.arange(model_mod.PREDICT_STEPS, device=reference.device)
                         .reshape(1, -1).expand(batch, -1))
            self.register_buffer(
                "pos_embed",
                model_mod.sinusoidal_encoding(positions, 384).to(reference.dtype),
                persistent=False)
            self.register_buffer(
                "encoder_mask",
                torch.ones(batch, model_mod.N_QUERY + 32, device=reference.device,
                           dtype=reference.dtype),
                persistent=False)

        def forward(self, latents, timestep, condition):
            features = self.action_encoder(latents) + self.pos_embed
            out = self.traj_dit(
                hidden_states=features,
                timestep=timestep,
                encoder_hidden_states=condition,
                encoder_mask=self.encoder_mask,
                image_rotary_emb=None,
                cross_attention_kwargs=dict(),
            ).sample
            return self.action_decoder(out)

    return DenoiseWrapper(system1, int(batch)).eval()
