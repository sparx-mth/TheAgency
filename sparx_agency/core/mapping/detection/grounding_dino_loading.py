"""Preserve the official Base checkpoint's shared decoder in Transformers 5.

Unlike LLMDet, Grounding DINO Base saves just model.decoder.bbox_embed.0.
Upstream chains two alias groups and can replace this tensor with an initialized
value while reporting no missing keys. Map every alias directly to the SAVED
head. Do not copy LLMDet's independent-head fix: that leaves other heads on meta.
"""
from __future__ import annotations


def shared_bbox_model_class(base_class, config, weights_path):
    """Use a local subclass; never patch installed Transformers or learned values."""
    from safetensors import safe_open

    if not config.decoder_bbox_embed_share or config.two_stage_bbox_embed_share:
        raise ValueError("Expected the official Base shared-decoder/separate-encoder checkpoint")
    canonical = "model.decoder.bbox_embed.0"
    expected = {canonical + ".layers.%d.%s" % (layer, suffix)
                for layer in range(3) for suffix in ("weight", "bias")}
    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        saved = {key for key in checkpoint.keys()
                 if key.startswith(("bbox_embed.", "model.decoder.bbox_embed."))}
    if saved != expected:
        raise ValueError("Checkpoint decoder layout differs from the verified Grounding DINO Base layout")
    aliases = {}
    for prefix in ("bbox_embed", "model.decoder.bbox_embed"):
        for layer in range(config.decoder_layers):
            alias = "%s.%d" % (prefix, layer)
            if alias != canonical:
                aliases[alias] = canonical

    class SharedCheckpointModel(base_class):
        _tied_weights_keys = aliases

    return SharedCheckpointModel


def require_materialized(model):
    """Never serve an unloaded/randomly initialized missing parameter as weights."""
    missing = [name for name, tensor in model.named_parameters() if tensor.is_meta]
    missing += [name for name, tensor in model.named_buffers() if tensor.is_meta]
    if missing:
        raise RuntimeError("Unmaterialized checkpoint tensors: %s" % missing)
