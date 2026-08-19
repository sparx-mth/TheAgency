"""The InternVLA-N1 System-1 chain, standalone and faithful to the checkpoint.

System 1 is 91.4 M of the released checkpoint's 8383.5 M parameters (1.1%). The
other 98.9% is System 2 -- a Qwen2.5-VL-7B that generates autoregressively and
does not fit on an 8 GB card. This module builds *only* System 1, with no
System-2 weights loaded and no ``transformers`` import, so the part that can
actually be converted can be exported, built and benchmarked on an edge device.

The chain, per ``InternVLAN1ForCausalLM.generate_traj`` with
``system1 = "nextdit_async"``::

    images (1,2,224,224,3)  --DINOv2-S-->        (1,512,384)
                            --MemoryEncoder-->   (1,512,384)  concat -> (1,512,768)
                            --QFormer-->         (1,32,768)
    vlm latents (1,4,3584)  --cond_projector-->  (1,4,768)
                            concat            -> (1,36,768)   cross-attention condition
    latents (32,32,3)       --NextDiT x10-->     (32,32,3)     flow-matching Euler

**One deviation from upstream's own constructor, and it is required.** Upstream
pins ``diffusers==0.33.1``, whose ``LuminaFeedForward`` dropped the
``int(2 * inner_dim / 3)` step earlier versions applied. The pinned code
therefore builds FFN-1536 while the released weights are FFN-1024, and 36
tensors fail to load. ``ffn_dim_multiplier=2/3`` restores the trained width:
``int(2/3 * 1536) = 1024``. With it, all 600 System-1 tensors match the
checkpoint key-for-key and shape-for-shape.
"""
from __future__ import annotations

#: Qwen2.5-VL-7B hidden size -- the width of the latents System 2 hands over.
VLM_DIM = 3584
#: ``internvla_n1_arch.LatentEmbSize`` -- the cross-attention condition width.
LATENT_EMB = 768
#: ``config.n_query`` -- how many trajectory-latent tokens System 2 emits.
N_QUERY = 4
#: ``predict_step_nums`` -- waypoints per predicted trajectory.
PREDICT_STEPS = 32
#: ``num_sample_trajs`` -- trajectory candidates drawn per call.
NUM_SAMPLE_TRAJS = 32
#: ``num_inference_steps`` -- flow-matching Euler steps per call.
NUM_INFERENCE_STEPS = 10
#: DINOv2 patch tokens per 224x224 image at patch size 14.
PATCH_TOKENS = 256
#: Frames the memory stack carries: the pixel-goal frame and the current one.
MEMORY_FRAMES = 2

#: Checkpoint prefixes that belong to System 1.
S1_PREFIXES = (
    "model.rgb_model.", "model.memory_encoder.", "model.rgb_resampler.",
    "model.cond_projector.", "model.traj_dit.", "model.action_encoder",
    "model.action_decoder",
)

#: ImageNet statistics upstream registers as ``_resnet_mean`` / ``_resnet_std``.
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def build_dit():
    """Construct the NextDiT trajectory denoiser at the checkpoint's widths.

    Returns:
        A ``LuminaNextDiT2DModel``: 12 blocks, d=384, 6 heads, FFN 1024,
        cross-attending to a 768-wide condition.
    """
    from sparx_agency.tasks.planning.vlas.internvla_n1.trt import upstream
    upstream.install()
    from internnav.model.basemodel.internvla_n1.nextdit_traj import LuminaNextDiT2DModel

    return LuminaNextDiT2DModel(
        sample_size=8, patch_size=1, in_channels=384, hidden_size=384,
        num_layers=12, num_attention_heads=6, num_kv_heads=6, multiple_of=256,
        ffn_dim_multiplier=2 / 3,           # see module docstring
        norm_eps=1e-5, learn_sigma=False, qk_norm=True,
        cross_attention_dim=LATENT_EMB,
    )


def sinusoidal_encoding(positions, embedding_dim):
    """``internvla_n1_arch.SinusoidalPositionalEncoding``, as a free function.

    Args:
        positions: ``(B, T)`` tensor of positions.
        embedding_dim: output width; must be even.

    Returns:
        ``(B, T, embedding_dim)`` tensor of concatenated sin/cos features.
    """
    import torch

    positions = positions.float()
    half = embedding_dim // 2
    exponent = -torch.arange(half, dtype=torch.float, device=positions.device) * (
        torch.log(torch.tensor(10000.0)) / half)
    freqs = positions.unsqueeze(-1) * exponent.exp()
    return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


def build_system1():
    """Build the System-1 module tree with random weights, in eval mode.

    Returns:
        A :class:`System1`.
    """
    return _system1_class()()


def _system1_class():
    """Define :class:`System1` lazily so importing this module needs no torch."""
    import torch
    import torch.nn as nn

    from sparx_agency.tasks.planning.vlas.internvla_n1.trt import upstream
    upstream.install()
    from internnav.model.basemodel.internvla_n1.internvla_n1_arch import (
        MemoryEncoder, QFormer)
    from internnav.model.encoder.depth_anything.depth_anything_v2.dinov2 import DINOv2

    class System1(nn.Module):
        """InternVLA-N1 System 1 (``nextdit_async``), without System 2.

        Attribute names match the checkpoint under the ``model.`` prefix, so a
        state dict loads with a single prefix strip.
        """

        def __init__(self):
            super().__init__()
            self.rgb_model = _as_patch_extractor(DINOv2("vits"))
            self.memory_encoder = MemoryEncoder()
            self.rgb_resampler = QFormer()
            self.cond_projector = nn.Sequential(
                nn.Linear(VLM_DIM, LATENT_EMB),
                nn.GELU(approximate="tanh"),
                nn.Linear(LATENT_EMB, LATENT_EMB))
            self.action_encoder = nn.Linear(3, 384, bias=True)
            self.action_decoder = nn.Linear(384, 3, bias=True)
            self.traj_dit = build_dit()
            self.register_buffer("_resnet_mean",
                                 torch.tensor(RGB_MEAN).view(1, 1, 3, 1, 1))
            self.register_buffer("_resnet_std",
                                 torch.tensor(RGB_STD).view(1, 1, 3, 1, 1))

        def vision(self, images_dp):
            """RGB memory stack -> DINOv2 patch features.

            Args:
                images_dp: ``(1, 2, 224, 224, 3)`` in [0, 1], pixel-goal frame
                    first, current frame second.

            Returns:
                ``(1, 512, 384)`` -- both frames' patch tokens, concatenated.
            """
            x = images_dp.permute(0, 1, 4, 2, 3)
            x = (x - self._resnet_mean) / self._resnet_std
            feat = self.rgb_model(x.flatten(0, 1))
            return feat.unflatten(0, (1, -1)).flatten(1, 2)

        def condition(self, dino_feat, traj_latents):
            """Patch features + System-2 latents -> the cross-attention condition.

            Args:
                dino_feat: ``(1, 512, 384)`` from :meth:`vision`.
                traj_latents: ``(1, 4, 3584)`` from ``generate_latents``.

            Returns:
                ``(1, 36, 768)``: 32 resampled memory tokens then 4 projected
                trajectory latents.
            """
            import torch
            memory = self.memory_encoder(dino_feat)
            tokens = self.rgb_resampler(torch.cat([dino_feat, memory], dim=-1))
            return torch.cat([tokens, self.cond_projector(traj_latents)], dim=1)

        def denoise_step(self, latents, timestep, condition):
            """One flow-matching Euler step's network evaluation.

            Args:
                latents: ``(B, 32, 3)`` current trajectory estimate.
                timestep: ``(B,)`` integer timestep.
                condition: ``(B, 36, 768)`` from :meth:`condition`, broadcast.

            Returns:
                ``(B, 32, 3)`` predicted velocity.
            """
            import torch
            features = self.action_encoder(latents)
            positions = (torch.arange(features.shape[1], device=features.device)
                         .reshape(1, -1).expand(features.shape[0], -1))
            features = features + sinusoidal_encoding(positions, 384).to(features.dtype)
            out = self.traj_dit(
                hidden_states=features,
                timestep=timestep,
                encoder_hidden_states=condition,
                encoder_mask=torch.ones(condition.shape[0], condition.shape[1],
                                        device=condition.device),
                image_rotary_emb=None,
                cross_attention_kwargs=dict(),
            ).sample
            return self.action_decoder(out)

    return System1


def __getattr__(name):
    """Expose :class:`System1` without importing torch at module import time."""
    if name == "System1":
        return _system1_class()
    raise AttributeError(name)



def _as_patch_extractor(backbone):
    """Make the DINOv2 trunk's ``forward`` be what this policy actually runs.

    ``generate_traj`` calls ``rgb_model.get_intermediate_layers(x)[0]``, which
    bypasses ``nn.Module.__call__`` entirely: no forward hook fires, so the
    profiler measures the trunk as **zero** and the plan is silently decided
    without the one component that might have mattered. DINOv2's own ``forward``
    returns a dict of class and patch tokens that nothing here consumes.

    Rebinding ``forward`` fixes both: the profiler can time the trunk, and
    ``torch.onnx.export`` traces the same call the deployed policy makes.

    Args:
        backbone: a ``DinoVisionTransformer``.

    Returns:
        The same backbone, with ``forward`` returning ``(B, 256, 384)`` final
        normed patch tokens.
    """
    import types

    def forward(self, x):
        """Final-block normed patch tokens, class token dropped."""
        return self.get_intermediate_layers(x)[0]

    backbone.forward = types.MethodType(forward, backbone)
    return backbone


def _strip_prefix(key):
    """Rewrite a checkpoint key onto :class:`System1`'s module tree.

    Two levels come off. ``model.`` is the ``InternVLAN1Model`` wrapper System 1
    does not have. ``traj_dit.model.`` has a second one because upstream nests
    the ``LuminaNextDiT2DModel`` inside a thin ``NextDiTCrossAttn``
    ``PreTrainedModel`` that adds nothing the forward pass uses -- System 1
    holds the ``LuminaNextDiT2DModel`` directly.

    Args:
        key: a checkpoint tensor name.

    Returns:
        The corresponding :class:`System1` state-dict key.
    """
    key = key[len("model."):]
    if key.startswith("traj_dit.model."):
        key = "traj_dit." + key[len("traj_dit.model."):]
    return key


def load_system1(checkpoint, device="cpu", dtype=None):
    """Build System 1 and load its weights out of the released checkpoint.

    Reads only the System-1 tensors (183 MB of a 16.77 GB checkpoint), so no
    System-2 weight is ever materialised and the load fits any machine.

    Args:
        checkpoint: directory holding ``model.safetensors.index.json`` and its
            shards, or a single ``.safetensors`` file.
        device: torch device for the returned module.
        dtype: torch dtype; defaults to the checkpoint's own (bfloat16).

    Returns:
        A :class:`System1` in eval mode.

    Raises:
        FileNotFoundError: when the checkpoint path or its index is absent.
        RuntimeError: when a System-1 tensor is missing or mis-shaped. A
            partially loaded policy would fly on random weights, so this is
            never a warning.
    """
    import json
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    root = Path(checkpoint)
    if root.is_file():
        shards = {root.name: root.parent}
        wanted = None
    else:
        index = root / "model.safetensors.index.json"
        if not index.is_file():
            raise FileNotFoundError("no model.safetensors.index.json under %s" % root)
        weight_map = json.loads(index.read_text())["weight_map"]
        wanted = {k: v for k, v in weight_map.items() if k.startswith(S1_PREFIXES)}
        shards = {shard: root for shard in sorted(set(wanted.values()))}

    state = {}
    for shard, parent in shards.items():
        path = parent / shard
        if not path.is_file():
            raise FileNotFoundError("checkpoint shard %s is missing" % path)
        for key, tensor in load_file(str(path)).items():
            if key.startswith(S1_PREFIXES):
                state[_strip_prefix(key)] = tensor

    net = _system1_class()()
    missing, unexpected = net.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("_resnet_")]
    if missing or unexpected:
        raise RuntimeError(
            "System-1 state dict does not match the checkpoint: %d missing "
            "(%s), %d unexpected (%s)"
            % (len(missing), ", ".join(missing[:5]),
               len(unexpected), ", ".join(unexpected[:5])))
    net = net.to(device=device, dtype=dtype or torch.bfloat16)
    return net.eval()
