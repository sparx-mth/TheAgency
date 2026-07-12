"""
Cage-bar removal for the ROBOTICAN Rooster drone — static mask only.

Applies TELEA inpainting to the permanent cage structure (side arcs, bottom arc)
before DA3 inference. The static mask is built once from calibration frames:
pixels dark in ≥PERSIST_FRAC of frames = permanent cage.

The moving horizontal bar is NOT handled here — use temporal logic (before/after
frames) at a higher level when bar presence is detected.

Usage:
    # One-time calibration:
    BarInpainter.build_and_save(sorted(glob("calib_dir/*.jpg")), "config/cage_static_mask.npy")

    # Per-frame at runtime:
    inpainter = BarInpainter("config/cage_static_mask.npy")
    bgr_clean = inpainter.process(bgr)
"""

from pathlib import Path

import cv2
import numpy as np

_DARK_THRESH  = 50    # pixel intensity below this = dark
_PERSIST_FRAC = 0.80  # fraction of calib frames a pixel must be dark → static cage
_DILATE_PX    = 4     # mask dilation radius to cover bar edges
_INPAINT_R    = 9     # TELEA inpainting radius (px)


class BarInpainter:
    """Inpaint permanent cage arcs out of a Rooster RGB frame before DA3 inference."""

    def __init__(self, static_mask_path: str | Path):
        path = Path(static_mask_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Static cage mask not found: {path}\n"
                "Run BarInpainter.build_and_save(calib_image_paths, path) first."
            )
        self._static_mask: np.ndarray = np.load(str(path))  # uint8 H×W, values 0/255

    def process(self, bgr: np.ndarray) -> np.ndarray:
        """Return a static-cage-free copy of bgr (same shape/dtype)."""
        if not self._static_mask.any():
            return bgr
        return cv2.inpaint(bgr, self._static_mask, _INPAINT_R, cv2.INPAINT_TELEA)

    @staticmethod
    def build_and_save(image_paths: list[str], out_path: str | Path) -> np.ndarray:
        """
        Compute the static cage mask from calibration frames and save it.

        Args:
            image_paths: paths to calibration JPEG/PNG frames (>=10 recommended)
            out_path:    where to write the .npy mask (uint8, 0/255)

        Returns:
            The computed mask array.
        """
        if not image_paths:
            raise ValueError("image_paths must not be empty")

        dark_count: np.ndarray | None = None
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"Cannot read calibration image: {path}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dark = (gray < _DARK_THRESH).astype(np.uint16)
            dark_count = dark if dark_count is None else dark_count + dark

        threshold = int(_PERSIST_FRAC * len(image_paths))
        static = (dark_count >= threshold).astype(np.uint8) * 255

        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_DILATE_PX * 2 + 1, _DILATE_PX * 2 + 1))
        static = cv2.dilate(static, ke)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), static)

        coverage = (static > 0).mean() * 100
        print(f"Static mask saved: {out_path}  coverage={coverage:.1f}%  "
              f"(built from {len(image_paths)} frames, persist>={_PERSIST_FRAC:.0%})")
        return static
