"""Preserve official LLMDet weights across Transformers' tied-weight loading.

Transformers 5.17.0 (also 5.10.0) declares *all* MM-GDINO decoder heads tied to
head zero. The released LLMDet checkpoint has distinct heads. Its safetensors
stores head zero under bbox_embed and heads 1..5 under model.decoder.bbox_embed;
the stock loader silently substitutes 35 tensors despite empty missing_keys.
Only alias bookkeeping is corrected here: architecture and forward are upstream.
"""
from __future__ import annotations


def checkpoint_model_class(base_class):
    """Keep decoder/output aliases, NEVER tie different refinement stages."""
    class CheckpointModel(base_class):
        _tied_weights_keys = {
            "model.decoder.bbox_embed": "bbox_embed",
            "model.decoder.class_embed": "class_embed",
        }

    return CheckpointModel


def verify_decoder_weights(model, weights_path):
    """Fail on silent tensor substitution, not just missing/unexpected key names."""
    import torch
    from safetensors import safe_open

    state = model.state_dict()
    checked = 0
    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            if name.startswith(("bbox_embed.", "class_embed.",
                                "model.decoder.bbox_embed.", "model.decoder.class_embed.")):
                if name not in state:
                    raise RuntimeError("Missing pretrained decoder tensor: " + name)
                loaded = state[name].detach().cpu()
                expected = checkpoint.get_tensor(name).to(dtype=loaded.dtype)
                if not torch.equal(loaded, expected):
                    raise RuntimeError("Pretrained decoder tensor was substituted: " + name)
                checked += 1
    if checked == 0:
        raise RuntimeError("Checkpoint contains no verifiable LLMDet decoder tensors")
    return checked
