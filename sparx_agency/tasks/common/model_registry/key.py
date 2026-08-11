"""The identity of one model artifact: what it is and which device it targets.

Extends the naming convention already used by ``yolo_world_trt``/``navdp``/
``flownav`` (``engines/<target_tag>/<model>[.<role>].<precision>[.<H>x<W>].engine``)
with an explicit resolution, since DepthAnythingV3 needs one and those three
tasks' single-input models don't.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union


def parse_resolution(value: Union[None, str, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Parse a resolution as ``"HxW"``, a ``(H, W)`` pair, or ``None``."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    s = str(value).lower().replace(" ", "")
    if "x" not in s:
        raise ValueError(f"resolution must be 'HxW' or a (H, W) pair, got {value!r}")
    h_str, w_str = s.split("x", 1)
    return int(h_str), int(w_str)


@dataclass(frozen=True)
class ModelKey:
    """Identifies one engine: which model, at what precision/resolution/role,
    built for which device. ``target_tag`` is the GPU/SoC slug from
    :func:`sparx_agency.tasks.common.hardware.detect.detect` (e.g. ``orin_sm87``).
    """

    model_id: str
    precision: str = "fp16"
    height: Optional[int] = None
    width: Optional[int] = None
    role: Optional[str] = None
    device: str = "gpu"
    target_tag: Optional[str] = None

    def stem(self) -> str:
        """Filename stem, e.g. ``da3_metric_large.depth_only.fp16.546x364``."""
        parts = [self.model_id]
        if self.role:
            parts.append(self.role)
        s = ".".join(parts) + f".{self.precision}"
        if self.height and self.width:
            s += f".{self.height}x{self.width}"
        return s

    def filename(self) -> str:
        return self.stem() + ".engine"

    def relpath(self, target_tag: Optional[str] = None) -> Path:
        """Path relative to an engines root: ``engines/<target_tag>/<filename>``."""
        tag = target_tag or self.target_tag
        if not tag:
            raise ValueError(f"target_tag required to compute a path for {self.stem()}")
        return Path("engines") / tag / self.filename()
