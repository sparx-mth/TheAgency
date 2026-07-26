"""Pick a laptop detector backend for the webcam test rig.

``yoloworld`` (default) is the project's real **open-vocabulary YOLO-World**
detector (`core/mapping/detection/YoloWorldDetector`, ultralytics ``YOLOWorld`` +
``set_classes``) — the torch analog of the drone's TensorRT YOLO-World, so *any*
target prompt works, not a fixed class list. ``color`` is the zero-dependency
colour-blob mock for when you cannot / do not want to load the model.

Both expose ``set_prompts`` + ``detect(rgb) -> List[Detection2D]``, so the
target-lock pipeline neither knows nor cares which is in play.

First YOLO-World run: ultralytics auto-installs the CLIP text deps (``clip``,
``ftfy``) and downloads the YOLO-World weights + the CLIP text model (~340 MB) the
first time ``set_classes`` runs; it may print a one-time "rerun" notice.
"""
from __future__ import annotations

from typing import Sequence, Union

from sparx_agency.core.mapping.detection import YoloWorldConfig, YoloWorldDetector
from sparx_agency.tasks.planning.object_approach_webcam.color_detector import (
    ColorDetectorConfig,
    MockColorDetector,
)

YOLOWORLD = "yoloworld"
COLOR = "color"
DETECTORS = (YOLOWORLD, COLOR)
# Friendly aliases so `--detector yolo` still means open-vocab YOLO-World (never COCO).
_ALIASES = {"yolo": YOLOWORLD, "yolo-world": YOLOWORLD, "world": YOLOWORLD}

Detector = Union[MockColorDetector, YoloWorldDetector]


def _resolve_device(device: str) -> str:
    """A concrete torch device: an explicit one, else GPU if available, else CPU."""
    if device:
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:                          # torch absent -> CPU (color mock path)
        return "cpu"


def make_webcam_detector(kind: str, target: str,
                         distractors: Sequence[str] = (),
                         color: str = "red",
                         weights: str = "yolov8s-worldv2.pt",
                         conf: float = 0.1,
                         device: str = "",
                         imgsz: int = 640) -> Detector:
    """Build the detector for ``kind`` and prime it with the target/distractor prompts.

    Args:
        kind: One of :data:`DETECTORS` (``"yolo"`` is accepted as a YOLO-World alias).
        target: The mission target label (prompt[0]) — any word for YOLO-World.
        distractors: Extra labels the detector also scores (context only on the HUD).
        color: Blob colour for the ``color`` backend.
        weights: YOLO-World checkpoint for the ``yoloworld`` backend
            (``yolov8{n,s,m,l,x}-worldv2.pt``).
        conf / device / imgsz: YOLO-World inference settings (blank device = auto GPU).

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    k = _ALIASES.get(str(kind).strip().lower(), str(kind).strip().lower())
    prompts = [target] + list(distractors)
    if k == COLOR:
        det = MockColorDetector(ColorDetectorConfig(color=color))
        det.set_prompts(prompts)
        return det
    if k == YOLOWORLD:
        det = YoloWorldDetector(YoloWorldConfig(
            model_path=weights, device=_resolve_device(device),
            conf_thresh=float(conf), imgsz=int(imgsz)))
        det.set_prompts(prompts)               # open-vocab: prompts become the classes
        return det
    raise ValueError("detector must be one of %s (or the 'yolo' alias), got %r"
                     % (list(DETECTORS), kind))
