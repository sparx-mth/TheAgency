"""Derive an explicit TensorRT build policy from an ONNX graph + hardware.

Nothing about the build is left to TensorRT defaults. This module inspects an
ONNX graph and the target :class:`HardwareProfile` and returns a
:class:`NetworkPolicy`: which layers to pin to higher precision under INT8
(LayerNorm, the action/critic heads, attention softmax, the encoder pos-embed
add -- the numerically sensitive places), the builder optimization level
(lowered on a 15 W Jetson), whether to obey precision constraints, and the
tactic sources to enable. The engine builder applies these explicitly.

DLA is deliberately never used for these graphs (ViT attention / LayerNorm are a
poor DLA fit), so no layer is marked DLA-eligible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Layer-name substrings whose outputs stay in float (never INT8): the loss-of-
# precision here flips the critic argmax / stop decision, not just MSE. These are
# the fallback defaults; configs/build_policy.json overrides them.
FP_KEEP_KEYWORDS = (
    "layernorm", "norm", "head", "softmax", "pos_embed", "former_pe",
    "former_query", "reducemean", "/norm", "instancenorm",
)

# configs/build_policy.json lives two levels up from engine/ (tasks/planning/navdp).
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "build_policy.json"


def load_build_policy(path=None):
    """Load the version-controlled build policy JSON (or {} if absent)."""
    p = Path(path) if path else _CONFIG_PATH
    return json.loads(p.read_text()) if p.exists() else {}


@dataclass
class NetworkPolicy:
    """Explicit TensorRT build knobs for one engine on one target."""
    engine_key: str
    fp_keep_keywords: List[str] = field(default_factory=lambda: list(FP_KEEP_KEYWORDS))
    builder_optimization_level: int = 5
    prefer_precision_constraints: bool = True
    use_fp16: bool = True
    use_int8: bool = False
    enable_all_tactics: bool = True
    node_count: int = 0
    # On the TRT>=11 strongly-typed path, build this engine FP32 instead of FP16
    # (the deep ViT encoder drifts in forced FP16; ignored on the TRT-10 path).
    force_fp32_strong: bool = False


def inspect(onnx_path, profile, precision="fp16"):
    """Build a :class:`NetworkPolicy` for ``onnx_path`` on ``profile``.

    Args:
        onnx_path: path to the exported ONNX graph.
        profile: target :class:`HardwareProfile`.
        precision: ``"fp16"`` or ``"int8"`` (int8 keeps the sensitive layers fp).

    Returns:
        A :class:`NetworkPolicy`.
    """
    engine_key = Path(onnx_path).stem
    node_count = _count_nodes(onnx_path)
    cfg = load_build_policy()
    # A 15 W Jetson benefits from a lower optimization level (shorter build, less
    # memory churn); a desktop dGPU can afford the max search.
    levels = cfg.get("builder_optimization_level", {})
    is_15w = profile.is_jetson and (profile.power_budget_w or 99) <= 15
    opt_level = levels.get("orin_15w", 3) if is_15w else levels.get("default", 5)
    keep = list(cfg.get("fp_keep_keywords", FP_KEEP_KEYWORDS))
    force_fp32 = engine_key in cfg.get("strongly_typed_fp32_engines", [])
    return NetworkPolicy(
        engine_key=engine_key,
        fp_keep_keywords=keep,
        builder_optimization_level=opt_level,
        use_fp16=True,                          # FP16 is always on (FP16-first)
        use_int8=(precision == "int8"),
        prefer_precision_constraints=(precision == "int8"),
        node_count=node_count,
        force_fp32_strong=force_fp32,
    )


def _count_nodes(onnx_path):
    """Count graph nodes (best-effort; 0 if onnx is unavailable)."""
    try:
        import onnx
        return len(onnx.load(str(onnx_path)).graph.node)
    except Exception:  # noqa: BLE001
        return 0
