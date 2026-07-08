"""Derive an explicit TensorRT build policy from the hardware + YOLO-World facts.

Nothing about the build is left to TensorRT defaults. This module turns a
:class:`~sparx_agency.tasks.mapping.yolo_world_trt.hardware.HardwareProfile` plus
the version-controlled ``configs/build_policy.json`` into a :class:`BuildPolicy`
that :mod:`build_engine` applies literally: precision, whether to target the DLA
(and which core, with GPU fallback), the DLA memory pools, the workspace pool, and
the builder optimization level.

Why these choices, for a *prompt-baked* YOLO-World graph:
  * **DLA + FP16.** After :meth:`set_classes`, the text embeddings are frozen into
    the head, so the export is a pure conv/neck/head CNN -- the workload NVDLA is
    built for. DLA runs FP16 or INT8 only (never FP32), so FP16 is the floor.
  * **GPU fallback is mandatory.** A handful of YOLOv8 ops are not DLA-supported
    (the detection-head Reshape/Transpose/decode, some Slice/Concat, the
    max-sigmoid class step). Without ``GPU_FALLBACK`` the build fails; with it,
    the conv-heavy backbone/neck stay on DLA and only the tail runs on GPU.
  * **Optimization level is an offline knob.** It controls build-time tactic
    search, not runtime power -- nvpmodel clamps power regardless. So the 15 W
    target uses the same max search (5) as the desktop; it only costs a longer
    one-time build.
  * **INT8 is opt-in.** DLA is dramatically more efficient in INT8, but it needs a
    representative calibration set; default is FP16 and INT8 must still pass an
    on-target accuracy check before it is trusted.

Pure standard library; importable anywhere (no torch / tensorrt needed).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "build_policy.json"

# Fallback defaults if configs/build_policy.json is missing a key.
_DEFAULT_VARIANTS = {
    "s": "yolov8s-worldv2.pt",
    "m": "yolov8m-worldv2.pt",
    "l": "yolov8l-worldv2.pt",
    "x": "yolov8x-worldv2.pt",
}
_DEFAULT_IMGSZ = (288, 512)      # (H, W), stride-32, matches 504x294 landscape


def load_config(path=None) -> dict:
    """Load the version-controlled build config JSON (or {} if absent)."""
    p = Path(path) if path else _CONFIG_PATH
    return json.loads(p.read_text()) if p.exists() else {}


def parse_imgsz(value) -> Tuple[int, int]:
    """Parse ``imgsz`` as ``(H, W)``; accepts ``"HxW"``, an int, or a 2-list.

    Both dims must be positive multiples of 32 (YOLOv8 max stride); a fixed,
    stride-aligned engine input is required -- YOLO cannot run an arbitrary size.
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
    """Explicit TensorRT build knobs for one YOLO-World engine on one target."""

    variant: str                       # "s" | "m" | "l" | "x"
    imgsz: Tuple[int, int]             # (H, W)
    precision: str = "fp16"           # "fp16" | "int8"
    use_dla: bool = False
    dla_core: int = 0
    gpu_fallback: bool = True
    dla_managed_sram_bytes: int = 1 << 20
    dla_local_dram_bytes: int = 1 << 30
    dla_global_dram_bytes: int = 1 << 29
    workspace_bytes: int = 1 << 30
    builder_optimization_level: int = 5
    conf_thresh: float = 0.25
    iou_thresh: float = 0.5
    max_det: int = 100

    @property
    def use_fp16(self) -> bool:
        """FP16 is always enabled: it is the floor for DLA and the default core."""
        return True

    @property
    def use_int8(self) -> bool:
        return self.precision == "int8"


def build_policy(variant, profile, config=None, precision=None,
                 imgsz=None, dla=None) -> BuildPolicy:
    """Assemble the :class:`BuildPolicy` for ``variant`` on ``profile``.

    Args:
        variant: one of ``"s"``, ``"m"``, ``"l"``, ``"x"``.
        profile: the target :class:`HardwareProfile`.
        config: pre-loaded config dict (defaults to ``configs/build_policy.json``).
        precision: override ``"fp16"`` / ``"int8"`` (else the config value).
        imgsz: override ``(H, W)`` / ``"HxW"`` (else the config value).
        dla: tri-state override -- ``True`` forces DLA, ``False`` forbids it,
            ``None`` uses the config *and* whether the board actually has a DLA.
    """
    cfg = config if config is not None else load_config()
    dla_cfg = cfg.get("dla", {})

    prec = (precision or cfg.get("precision", "fp16")).lower()
    if prec not in ("fp16", "int8"):
        raise ValueError("precision must be 'fp16' or 'int8', got %r" % prec)

    hw = parse_imgsz(imgsz if imgsz is not None else cfg.get("imgsz", _DEFAULT_IMGSZ))

    want_dla = dla_cfg.get("enable", True) if dla is None else bool(dla)
    # DLA can only be *requested* where the board has one; the builder will still
    # error on x86 if forced. profile.allow_dla is False off-Jetson.
    use_dla = bool(want_dla and profile.allow_dla)

    levels = cfg.get("builder_optimization_level", {})
    opt = levels.get("orin_15w", 5) if profile.is_15w else levels.get("default", 5)

    return BuildPolicy(
        variant=variant,
        imgsz=hw,
        precision=prec,
        use_dla=use_dla,
        dla_core=int(dla_cfg.get("core", 0)),
        gpu_fallback=bool(dla_cfg.get("gpu_fallback", True)),
        dla_managed_sram_bytes=int(dla_cfg.get("managed_sram_bytes", 1 << 20)),
        dla_local_dram_bytes=int(dla_cfg.get("local_dram_bytes", 1 << 30)),
        dla_global_dram_bytes=int(dla_cfg.get("global_dram_bytes", 1 << 29)),
        workspace_bytes=profile.recommended_workspace_bytes,
        builder_optimization_level=int(opt),
        conf_thresh=float(cfg.get("conf_thresh", 0.25)),
        iou_thresh=float(cfg.get("iou_thresh", 0.5)),
        max_det=int(cfg.get("max_det", 100)),
    )


def variant_weights(config=None) -> Dict[str, str]:
    """Return the ``{variant: default_weights_filename}`` map from the config."""
    cfg = config if config is not None else load_config()
    return dict(cfg.get("variants", _DEFAULT_VARIANTS))
