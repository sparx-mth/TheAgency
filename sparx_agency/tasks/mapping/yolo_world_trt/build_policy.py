"""Derive an explicit TensorRT build policy per engine role (backbone vs head).

Open-set YOLO-World is split into two engines with opposite hardware fits, so the
policy is per **role**:

  * ``backbone`` -- text-free, image-only, fully static. Targets the **DLA** (with
    GPU fallback for the few unsupported ops) at FP16, which is the whole point of
    the split: the conv bulk runs off the GPU. Static shapes are what let DLA
    accept it at all.
  * ``head`` -- fuses the text embeddings; its class dimension is the runtime
    prompt count, so it is built with a **dynamic** ``N`` optimization profile and
    stays on the **GPU** (DLA cannot do dynamic shapes). FP16.

Nothing is left to TensorRT defaults: precision, device, DLA memory pools, the
workspace pool, and the (offline) builder optimization level all come from the
:class:`~...hardware.HardwareProfile` plus ``configs/build_policy.json``.

Pure standard library; importable anywhere (no torch / tensorrt needed).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "build_policy.json"

_DEFAULT_VARIANTS = {
    "s": "yolov8s-worldv2.pt",
    "m": "yolov8m-worldv2.pt",
    "l": "yolov8l-worldv2.pt",
    "x": "yolov8x-worldv2.pt",
}
_DEFAULT_IMGSZ = (288, 512)      # (H, W), stride-32, matches 504x294 landscape
ROLES = ("backbone", "head")


def load_config(path=None) -> dict:
    """Load the version-controlled build config JSON (or {} if absent)."""
    p = Path(path) if path else _CONFIG_PATH
    return json.loads(p.read_text()) if p.exists() else {}


def parse_imgsz(value) -> Tuple[int, int]:
    """Parse ``imgsz`` as ``(H, W)``; accepts ``"HxW"``, an int, or a 2-list.

    Both dims must be positive multiples of 32 (YOLOv8 max stride): the backbone
    engine has a fixed, stride-aligned input -- YOLO cannot run an arbitrary size.
    """
    if isinstance(value, (list, tuple)):
        h, w = int(value[0]), int(value[1])
    elif isinstance(value, int):
        h = w = int(value)
    else:
        s = str(value).lower().replace(" ", "")
        if "x" in s:
            hs, ws = s.split("x", 1)
            h, w = int(hs), int(ws)
        else:
            h = w = int(s)
    for name, d in (("height", h), ("width", w)):
        if d <= 0 or d % 32 != 0:
            raise ValueError(
                "imgsz %s must be a positive multiple of 32, got %d" % (name, d))
    return h, w


@dataclass
class BuildPolicy:
    """Explicit TensorRT build knobs for one engine (one role) on one target."""

    role: str                          # "backbone" | "head"
    variant: str                       # "s" | "m" | "l" | "x"
    precision: str = "fp16"           # "fp16" | "int8"
    use_dla: bool = False
    dla_core: int = 0
    gpu_fallback: bool = True
    dla_managed_sram_bytes: int = 1 << 20
    dla_local_dram_bytes: int = 1 << 30
    dla_global_dram_bytes: int = 1 << 29
    workspace_bytes: int = 1 << 30
    builder_optimization_level: int = 5

    @property
    def use_fp16(self) -> bool:
        """FP16 is always on: the floor for DLA and the default for the GPU head."""
        return True

    @property
    def use_int8(self) -> bool:
        return self.precision == "int8"

    @property
    def is_dynamic(self) -> bool:
        """Only the head carries a dynamic (prompt-count) optimization profile."""
        return self.role == "head"


def build_policy(role, variant, profile, config=None, precision=None,
                 dla=None) -> BuildPolicy:
    """Assemble the :class:`BuildPolicy` for ``role`` of ``variant`` on ``profile``.

    Args:
        role: ``"backbone"`` (static, DLA-preferred) or ``"head"`` (dynamic, GPU).
        variant: one of ``"s"``, ``"m"``, ``"l"``, ``"x"``.
        profile: the target :class:`HardwareProfile`.
        config: pre-loaded config dict (defaults to ``configs/build_policy.json``).
        precision: override ``"fp16"`` / ``"int8"`` (else the config value).
        dla: tri-state -- ``True`` force DLA, ``False`` forbid, ``None`` = config +
            whether the board has a DLA. Ignored for the head (always GPU: DLA
            cannot run its dynamic shapes).
    """
    if role not in ROLES:
        raise ValueError("role must be one of %s, got %r" % (ROLES, role))
    cfg = config if config is not None else load_config()
    dla_cfg = cfg.get("dla", {})

    prec = (precision or cfg.get("precision", "fp16")).lower()
    if prec not in ("fp16", "int8"):
        raise ValueError("precision must be 'fp16' or 'int8', got %r" % prec)

    if role == "head":
        use_dla = False                          # dynamic shapes -> GPU only
    else:
        want = dla_cfg.get("enable", True) if dla is None else bool(dla)
        use_dla = bool(want and profile.allow_dla)

    levels = cfg.get("builder_optimization_level", {})
    opt = levels.get("orin_15w", 5) if profile.is_15w else levels.get("default", 5)

    return BuildPolicy(
        role=role,
        variant=variant,
        precision=prec,
        use_dla=use_dla,
        dla_core=int(dla_cfg.get("core", 0)),
        gpu_fallback=bool(dla_cfg.get("gpu_fallback", True)),
        dla_managed_sram_bytes=int(dla_cfg.get("managed_sram_bytes", 1 << 20)),
        dla_local_dram_bytes=int(dla_cfg.get("local_dram_bytes", 1 << 30)),
        dla_global_dram_bytes=int(dla_cfg.get("global_dram_bytes", 1 << 29)),
        workspace_bytes=profile.recommended_workspace_bytes,
        builder_optimization_level=int(opt),
    )


def variant_weights(config=None) -> Dict[str, str]:
    """Return the ``{variant: default_weights_filename}`` map from the config."""
    cfg = config if config is not None else load_config()
    return dict(cfg.get("variants", _DEFAULT_VARIANTS))
