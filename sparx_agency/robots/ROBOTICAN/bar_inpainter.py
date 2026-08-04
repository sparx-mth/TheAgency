"""
Cage-bar removal for the ROBOTICAN Rooster drone.

Two independent cage artifacts, two different fixes:
  - Permanent side/bottom arcs: fixed shape, TELEA-inpainted using a static
    mask built once from calibration frames (pixels dark in >=PERSIST_FRAC of
    frames = permanent cage).
  - The moving horizontal crossbar: not a fixed shape (its row position drifts
    frame to frame -- confirmed live 2026-08-04, see LESSONS.md), so it is
    detected fresh every frame instead: a thin, near-black, low-variance band
    spanning the full frame width. Filled by copying real content down/up from
    just above and below the band (top half of the band <- rows just above,
    bottom half <- rows just below) rather than TELEA -- a plain vertical
    extension is simpler and more predictable for a thin horizontal occluder
    than a general-purpose PDE inpaint, which is tuned for irregular regions
    like the arcs.

Usage:
    # One-time calibration (static arcs only):
    BarInpainter.build_and_save(sorted(glob("calib_dir/*.jpg")), "config/cage_static_mask.npy")

    # Per-frame at runtime (handles both arcs + moving crossbar):
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

# ── Moving crossbar (dynamic, per-frame) ──────────────────────────────────
# Thresholds calibrated against a 2026-08-04 recording of the live camera feed
# (frame height ~345px in that capture): the bar measured mean row-intensity
# <15 (well under _BAR_DARK_MAX) and 11-17px thick. Kept generous since these
# haven't been checked against a native-resolution (540x360) capture yet --
# tune here first if the detector misses the bar or false-triggers on a
# genuinely dark, full-width scene feature (e.g. a shadowed doorway).
_BAR_DARK_MAX   = 30   # row mean intensity below this = candidate bar row
_BAR_ROWSTD_MAX = 25   # row std (across width) below this = uniform occluder, not scene texture
_BAR_MIN_PX     = 2    # thinnest plausible bar band
_BAR_MAX_PX     = 40   # thickest plausible bar band (bounds false-triggers on large dark regions)
_BAR_MARGIN_PX  = 2    # grow each detected band by this many rows (anti-aliased edges)
_BAR_REF_ROWS   = 3    # rows sampled above/below the band for the fill reference (median, robust to noise)


def _detect_dynamic_bar_rows(gray: np.ndarray) -> list[tuple[int, int]]:
    """Return [(row_start, row_end), ...] (inclusive) for each full-width dark
    band matching the crossbar's signature: low mean AND low variance across
    the whole row (a real scene is dark-but-textured; a physical occluder
    flush against the lens is dark-and-flat)."""
    row_mean = gray.mean(axis=1)
    row_std = gray.std(axis=1)
    is_bar_row = (row_mean < _BAR_DARK_MAX) & (row_std < _BAR_ROWSTD_MAX)

    raw_runs = []
    h = gray.shape[0]
    r = 0
    while r < h:
        if not is_bar_row[r]:
            r += 1
            continue
        start = r
        while r < h and is_bar_row[r]:
            r += 1
        raw_runs.append((start, r - 1))

    # Compression noise can briefly flip a row or two inside one real bar
    # (confirmed live: a single bar detected as 2-3 adjacent runs) -- merge
    # runs separated by a small gap before the thickness filter, so a noisy
    # middle row can't split one bar into pieces too thin to pass _BAR_MIN_PX.
    merged = []
    for start, end in raw_runs:
        if merged and start - merged[-1][1] - 1 <= _BAR_MARGIN_PX:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    bands = []
    for start, end in merged:
        if _BAR_MIN_PX <= (end - start + 1) <= _BAR_MAX_PX:
            bands.append((max(0, start - _BAR_MARGIN_PX), min(h - 1, end + _BAR_MARGIN_PX)))
    return bands


def _fill_band_vertical_median(bgr: np.ndarray, start: int, end: int) -> None:
    """In-place: fill rows [start, end] by copying the median of _BAR_REF_ROWS
    real rows from just above into the band's top half, and from just below
    into its bottom half. Falls back to whichever side exists if the band
    touches a frame edge."""
    h = bgr.shape[0]
    above_lo, above_hi = max(0, start - _BAR_REF_ROWS), start  # rows [above_lo, above_hi)
    below_lo, below_hi = end + 1, min(h, end + 1 + _BAR_REF_ROWS)  # rows [below_lo, below_hi)

    above_ref = np.median(bgr[above_lo:above_hi], axis=0) if above_hi > above_lo else None
    below_ref = np.median(bgr[below_lo:below_hi], axis=0) if below_hi > below_lo else None
    if above_ref is None and below_ref is None:
        return  # band is the whole frame -- nothing real to copy from
    if above_ref is None:
        above_ref = below_ref
    if below_ref is None:
        below_ref = above_ref

    mid = (start + end) // 2
    bgr[start:mid + 1] = above_ref.astype(bgr.dtype)
    bgr[mid + 1:end + 1] = below_ref.astype(bgr.dtype)


class BarInpainter:
    """Remove both cage artifacts from a Rooster RGB frame: TELEA-inpaint the
    permanent arcs (static mask), then vertical-median-fill the moving
    crossbar (detected fresh per frame)."""

    def __init__(self, static_mask_path: str | Path):
        path = Path(static_mask_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Static cage mask not found: {path}\n"
                "Run BarInpainter.build_and_save(calib_image_paths, path) first."
            )
        self._static_mask: np.ndarray = np.load(str(path))  # uint8 H×W, values 0/255

    def process(self, bgr: np.ndarray) -> np.ndarray:
        """Return a cage-free copy of bgr (same shape/dtype): static arcs
        inpainted, then the moving crossbar (if present this frame) filled."""
        out = cv2.inpaint(bgr, self._static_mask, _INPAINT_R, cv2.INPAINT_TELEA) \
            if self._static_mask.any() else bgr.copy()

        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        for start, end in _detect_dynamic_bar_rows(gray):
            _fill_band_vertical_median(out, start, end)
        return out

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
