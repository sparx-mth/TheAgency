"""Export-friendly nn.Module wrappers around FlowNav's submodules.

The full FlowNav ``NoMaD`` model is not exportable as one graph (a string-routed
``forward``, the flow-matching ODE loop, numpy ingestion). We export only the
three heavy, deterministic forward passes as static-shape ONNX graphs and keep
the Euler integration + de-normalization in numpy (``core.planning.vlas.flownav.trt``):

  * :class:`EncoderWrapper`  -- the ``NoMaD_ViNT`` vision encoder: EfficientNet-B0
    (obs) + EfficientNet-B0 (obs+goal) + a frozen DINOv2 / DepthAnythingV2-ViT-S
    "depth prior" run on the current RGB frame + a 4-layer self-attention, mean
    pooled to a 256-d conditioning vector. Exported in **navigation mode** (goal
    used, ``input_goal_mask = 0`` baked in), so it takes no mask input.
  * :class:`VFieldWrapper` -- one velocity-field evaluation (``ConditionalUnet1D``)
    with the continuous flow-matching time embedded inside the graph.
  * :class:`DistWrapper`   -- the temporal-distance head (a small MLP).

Three export-time fixes baked in here:
  * **EfficientNet swish made exportable** (:func:`_disable_memory_efficient_swish`):
    ``efficientnet_pytorch`` defaults to ``MemoryEfficientSwish``, a custom
    autograd ``Function`` that does NOT trace to ONNX; we switch both encoders to
    the plain ``x * sigmoid(x)`` swish (identical numerics, exportable).
  * **DINOv2 positional-embedding pre-bake** (:func:`bake_pos_embed`): DINOv2
    bicubic-interpolates its 518px pos-embed down to the (padded) 98px grid every
    forward; we freeze that interpolation as a buffer so the exported graph
    contains no ``Resize`` (the single biggest parity / TRT risk).
  * **Baked navigation goal-mask**: ``input_goal_mask = 0`` (goal used) is created
    as a constant inside the wrapper, so the int64 ``Gather``/mask path folds away.

These modules import torch and the external FlowNav model and therefore live
under ``tasks/`` -- never importable from the Python-3.8 ROS-free ``core``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Depth branch geometry: the encoder pads the 96x96 current frame by 1px each
# side before the DINOv2 trunk (see NoMaD_ViNT.forward: F.pad(..., (1,1,1,1))).
DEPTH_INPUT_HW = (98, 98)


def _disable_memory_efficient_swish(effnet):
    """Switch an ``EfficientNet`` to plain (ONNX-exportable) swish."""
    if hasattr(effnet, "set_swish"):
        effnet.set_swish(memory_efficient=False)


def bake_pos_embed(vit, input_hw=DEPTH_INPUT_HW):
    """Freeze DINOv2's interpolated positional embedding as a constant buffer.

    DINOv2 sizes its positional embedding for 518px and bicubic-interpolates it to
    the actual token grid every forward (a ``Resize``). The interpolation depends
    only on the token-grid geometry, which is fixed here (the padded depth input is
    always ``input_hw``), so we compute it once in eager fp32 and override
    ``interpolate_pos_encoding`` on THIS instance to return the frozen result --
    the exported graph then has no ``Resize``.

    Args:
        vit: the DINOv2 trunk (``vision_encoder.depth_encoder``).
        input_hw: ``(H, W)`` the trunk actually receives (padded frame).
    """
    import types

    h, w = int(input_hw[0]), int(input_hw[1])
    dim = int(vit.pos_embed.shape[-1])
    patch = int(getattr(vit, "patch_size", 14))
    n_patch = (h // patch) * (w // patch)
    n_reg = int(getattr(vit, "num_register_tokens", 0))
    seq = 1 + n_patch + n_reg                       # cls + patches (+ registers)
    with torch.no_grad():
        dummy = torch.zeros(1, seq, dim, device=vit.pos_embed.device)
        baked = vit.interpolate_pos_encoding(dummy, w, h).detach().clone()
    vit.register_buffer("_baked_pos_embed", baked)

    def _baked(self, x, w, h):
        return self._baked_pos_embed

    vit.interpolate_pos_encoding = types.MethodType(_baked, vit)


def _adaptive_avg_pool1d_matrix(in_len, out_len):
    """Constant averaging matrix ``M (out_len, in_len)`` with ``x @ M.T ==
    F.adaptive_avg_pool1d(x, out_len)`` (matches PyTorch's window formula)."""
    M = torch.zeros(int(out_len), int(in_len))
    for i in range(int(out_len)):
        start = (i * in_len) // out_len
        end = -(-((i + 1) * in_len) // out_len)        # ceil((i+1)*in_len/out_len)
        M[i, start:end] = 1.0 / (end - start)
    return M


class _FixedAdaptiveAvgPool1d(nn.Module):
    """Static-size adaptive average pool over the last dim, as a constant matmul.

    torch.onnx (TorchScript) cannot export ``adaptive_avg_pool1d`` when the input
    length is not a graph constant. Here both the input length (the DINOv2 token
    count at the fixed padded geometry) and the output length are known, so the
    pool is an exact constant linear map -- exportable and numerically identical.
    """

    def __init__(self, in_len, out_len):
        super().__init__()
        self.register_buffer("M", _adaptive_avg_pool1d_matrix(in_len, out_len))

    def forward(self, x):                       # (B, C, in_len) -> (B, C, out_len)
        return torch.matmul(x, self.M.t())


def bake_depth_pool(vision_encoder, input_hw=DEPTH_INPUT_HW):
    """Replace the depth-head ``AdaptiveAvgPool1d`` with the exportable matmul.

    ``compress_depth_enc[0]`` pools the DINOv2 token axis (49 tokens at the padded
    98x98 geometry) to ``pool_dim``; we swap it for :class:`_FixedAdaptiveAvgPool1d`.
    """
    cd = getattr(vision_encoder, "compress_depth_enc", None)
    if not (isinstance(cd, nn.Sequential) and isinstance(cd[0], nn.AdaptiveAvgPool1d)):
        return
    patch = int(getattr(vision_encoder.depth_encoder, "patch_size", 14))
    in_len = (int(input_hw[0]) // patch) * (int(input_hw[1]) // patch)
    out = cd[0].output_size
    out_len = out[0] if isinstance(out, (tuple, list)) else out
    cd[0] = _FixedAdaptiveAvgPool1d(in_len, out_len)


class EncoderWrapper(nn.Module):
    """Vision encoder: ``(obs_img (1,Cobs,96,96), goal_img (1,3,96,96)) -> (1,256)``.

    Wraps the trained ``NoMaD_ViNT`` and calls its forward with a constant
    navigation mask (goal used). Applies the two export fixes (swish + pos-embed
    bake) to the held submodules in place; weights are shared with the loaded model.
    """

    def __init__(self, vision_encoder):
        super().__init__()
        self.encoder = vision_encoder
        _disable_memory_efficient_swish(self.encoder.obs_encoder)
        _disable_memory_efficient_swish(self.encoder.goal_encoder)
        bake_pos_embed(self.encoder.depth_encoder)
        bake_depth_pool(self.encoder)
        self.eval()

    def forward(self, obs_img, goal_img):
        b = obs_img.shape[0]
        # Navigation: input_goal_mask == 0 selects the "no_mask" row (goal used).
        mask = torch.zeros(b, dtype=torch.long, device=obs_img.device)
        return self.encoder(obs_img, goal_img, input_goal_mask=mask)


class VFieldWrapper(nn.Module):
    """One velocity-field evaluation.

    ``(sample (N,8,2), timestep (1,), global_cond (N,256)) -> vfield (N,8,2)``.
    The continuous flow-matching time is embedded (sinusoidal) inside the
    ``ConditionalUnet1D``; ``timestep`` is a rank-1 ``(1,)`` tensor broadcast to N.
    """

    def __init__(self, noise_pred_net):
        super().__init__()
        self.net = noise_pred_net
        self.eval()

    def forward(self, sample, timestep, global_cond):
        return self.net(sample, timestep, global_cond=global_cond)


class DistWrapper(nn.Module):
    """Temporal-distance head: ``(obsgoal_cond (1,256)) -> distance (1,1)``."""

    def __init__(self, dist_pred_net):
        super().__init__()
        self.net = dist_pred_net
        self.eval()

    def forward(self, obsgoal_cond):
        return self.net(obsgoal_cond)
