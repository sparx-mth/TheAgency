"""Torch-side monkey-patches that make a transformer trace cleanly to ONNX.

Every one of these exists because a real export failed without it. They are
collected here rather than buried in each network's exporter because they are
properties of *torch and its attention/pooling implementations*, not of any
particular model -- the same six patches keep reappearing across DINOv2,
EfficientNet, Qwen-style ViTs and diffusion heads.

Applied as a single context manager, :func:`export_context`, so the patches are
always undone even when the export raises. A patch that leaks past the export
would silently change the numerics of the very reference the parity gate is
about to compare against.

torch is imported lazily inside the functions so this module stays importable
in an environment without it.
"""
from __future__ import annotations

import types
from contextlib import contextmanager


@contextmanager
def sdpa_math():
    """Force ``scaled_dot_product_attention`` onto its MATH backend.

    Without this, SDPA dispatches to a fused flash/mem-efficient kernel that
    exports as a single opaque ``*Attention`` node (or fails outright). The MATH
    backend decomposes to MatMul/Softmax/Add, which TensorRT then re-fuses into
    its own MHA tactic -- the fusion you actually want, chosen by the builder
    that knows the target.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel
    try:
        ctx = sdpa_kernel(SDPBackend.MATH)
    except TypeError:                      # older torch takes a list
        ctx = sdpa_kernel([SDPBackend.MATH])
    with ctx:
        yield


@contextmanager
def mha_fastpath_disabled():
    """Disable ``nn.MultiheadAttention``'s fused fast path for the duration.

    The fast path is a single fused kernel with no ONNX symbolic. Older torch
    builds have no such knob, in which case this is a no-op -- documented rather
    than silent, because on those builds a fused node can still leak and the op
    gate is what will catch it.
    """
    import torch
    setter = getattr(getattr(torch.backends, "mha", None),
                     "set_fastpath_enabled", None)
    if setter is None:
        yield
        return
    try:
        previous = torch.backends.mha.get_fastpath_enabled()
    except Exception:  # noqa: BLE001  (getter absent on some builds)
        previous = True
    setter(False)
    try:
        yield
    finally:
        setter(previous)


@contextmanager
def eval_mode(module):
    """Put a module in eval mode and restore its previous training flag."""
    was_training = bool(getattr(module, "training", False))
    module.eval()
    try:
        yield module
    finally:
        if was_training:
            module.train()


@contextmanager
def gradient_checkpointing_disabled(module):
    """Turn off gradient checkpointing, which traces as re-entrant garbage.

    Several released checkpoints ship with ``_gradient_checkpointing=True`` in
    their config, so this fires more often than it looks like it should.
    """
    disable = getattr(module, "disable_gradient_checkpointing", None)
    enabled = bool(getattr(module, "is_gradient_checkpointing", False))
    if disable is None:
        yield module
        return
    disable()
    try:
        yield module
    finally:
        if enabled:
            enable = getattr(module, "enable_gradient_checkpointing", None)
            if enable is not None:
                enable()


def bake_pos_embed(backbone, input_hw, patch_size, num_register_tokens=0,
                   embed_dim=None):
    """Pre-compute a ViT's interpolated positional embedding at a fixed size.

    DINOv2-family backbones interpolate their positional embedding with a
    **bicubic** ``F.interpolate`` whenever the input is not the pretraining
    resolution. That traces to an ONNX ``Resize``, which the op gate rejects and
    which TensorRT handles poorly. Since the deployed input size is fixed, the
    interpolation has exactly one answer -- compute it once and replace the
    method with a constant lookup.

    Args:
        backbone: the ViT module exposing ``interpolate_pos_encoding``.
        input_hw: ``(H, W)`` of the deployed input.
        patch_size: ViT patch size in pixels.
        num_register_tokens: register tokens, if the variant has them.
        embed_dim: token width; read from ``backbone.pos_embed`` when omitted.

    Returns:
        The backbone, with ``interpolate_pos_encoding`` replaced.

    Raises:
        AttributeError: if the backbone has no ``interpolate_pos_encoding``.
    """
    import torch

    if not hasattr(backbone, "interpolate_pos_encoding"):
        raise AttributeError(
            "%s has no interpolate_pos_encoding; positional-embedding baking "
            "does not apply to it." % type(backbone).__name__)
    h, w = int(input_hw[0]), int(input_hw[1])
    n_patch = (h // int(patch_size)) * (w // int(patch_size))
    seq = 1 + n_patch + int(num_register_tokens)
    dim = int(embed_dim) if embed_dim is not None else int(backbone.pos_embed.shape[-1])
    with torch.no_grad():
        probe = torch.zeros(1, seq, dim, dtype=backbone.pos_embed.dtype,
                            device=backbone.pos_embed.device)
        baked = backbone.interpolate_pos_encoding(probe, w, h).detach().clone()
    backbone.register_buffer("_baked_pos_embed", baked, persistent=False)

    def _baked(self, x, w, h):
        """Return the frozen embedding; the size arguments are ignored.

        Bound as a real method with the ORIGINAL parameter names, because a
        backbone that calls ``interpolate_pos_encoding(x, w=..., h=...)`` by
        keyword must keep working -- a replacement that only accepts positional
        arguments turns the bake into a TypeError raised from inside
        ``torch.onnx.export``, which reads as "the pre-bake did not take".
        """
        return self._baked_pos_embed

    backbone.interpolate_pos_encoding = types.MethodType(_baked, backbone)
    return backbone


def reject_adaptive_avg_pool1d(module):
    """Fail loud on any ``AdaptiveAvgPool1d``, naming the fix.

    ``AdaptiveAvgPool1d`` derives its window boundaries from the input length at
    runtime, which traces into shape arithmetic TensorRT cannot fold. With a
    static input length the pooling is a constant matrix and becomes one
    ``MatMul`` -- but only the export wrapper knows that length, so this
    refuses rather than guessing.

    Args:
        module: the root module to scan, recursively.

    Returns:
        0, when the module tree is clean.

    Raises:
        ValueError: naming the offending submodule and pointing at
            :func:`pool_matrix`, which builds the replacement.
    """
    from torch import nn

    for name, child in list(module.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool1d):
            raise ValueError(
                "AdaptiveAvgPool1d at %r cannot be exported: its windows depend "
                "on the runtime input length. Build the constant with "
                "pool_matrix(in_len, out_len) and substitute a MatMul in the "
                "export wrapper." % name)
        reject_adaptive_avg_pool1d(child)
    return 0


def pool_matrix(in_len, out_len, dtype=None):
    """The constant matrix implementing ``AdaptiveAvgPool1d(out_len)``.

    Args:
        in_len: static input length.
        out_len: pooled output length.
        dtype: torch dtype for the matrix (float32 when omitted).

    Returns:
        A ``(in_len, out_len)`` tensor; ``x @ matrix`` equals the pooling.
    """
    import torch

    matrix = torch.zeros(int(in_len), int(out_len),
                         dtype=dtype or torch.float32)
    for j in range(int(out_len)):
        start = (j * int(in_len)) // int(out_len)
        end = -(-((j + 1) * int(in_len)) // int(out_len))
        matrix[start:end, j] = 1.0 / float(end - start)
    return matrix


def replace_memory_efficient_swish(module):
    """Replace EfficientNet's custom autograd Swish with plain ``SiLU``.

    ``MemoryEfficientSwish`` is a ``torch.autograd.Function``, which has no ONNX
    symbolic at all. It is numerically identical to ``SiLU``.

    Returns:
        The number of modules replaced.
    """
    from torch import nn

    replaced = 0
    for name, child in list(module.named_children()):
        if type(child).__name__ in ("MemoryEfficientSwish", "Swish"):
            setattr(module, name, nn.SiLU())
            replaced += 1
        else:
            replaced += replace_memory_efficient_swish(child)
    return replaced


@contextmanager
def export_context(module):
    """Apply every generic export patch around a block, then undo them all.

    This is what an exporter should wrap its ``torch.onnx.export`` call in.
    Model-specific patches (baking a positional embedding, deleting a
    classifier-free-guidance branch) are applied by the adapter *before*
    entering this context, because they change the module permanently.
    """
    import torch

    with torch.no_grad():
        with eval_mode(module):
            with gradient_checkpointing_disabled(module):
                with mha_fastpath_disabled():
                    with sdpa_math():
                        yield module
