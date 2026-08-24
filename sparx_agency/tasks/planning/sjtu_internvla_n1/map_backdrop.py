"""Load a nav2 occupancy map and cut world-aligned windows out of it to draw on.

The top-down panel of a recorded run is only worth watching if the route is
drawn *on the building*. A trail on graph paper says the drone moved; the same
trail on the hospital's floor plan says which corridor it moved down and which
rooms it never entered -- which is the whole question the exploration order asks.

This module owns the map side of that and nothing else: read the map once, and
hand out BGR images together with the exact world rectangle each one covers, so
the renderer's world->pixel transform is fixed by the map rather than fitted to
the trail. A fitted transform re-scales itself every frame, which makes a
recorded flight look like the building is breathing.

ROS-free and stateless after load, so the renderer stays unit-testable without a
simulator.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is present in the venv and ROS env
    cv2 = None

import yaml

# Panel palette. Free space stays dark so the yellow/orange/green route overlays
# read against it; walls are the light element, which is the inverse of a paper
# floor plan and the right way round for a dark video panel.
FREE_BGR = (46, 46, 48)
OCCUPIED_BGR = (168, 178, 188)
UNKNOWN_BGR = (26, 26, 28)


@dataclass(frozen=True)
class MapWindow:
    """A BGR image and the world rectangle it covers.

    Attributes:
        image: ``(h, w, 3)`` uint8 BGR.
        min_x: World x of the image's left edge, metres.
        min_y: World y of the image's *bottom* edge, metres.
        max_x: World x of the image's right edge, metres.
        max_y: World y of the image's top edge, metres.
    """

    image: np.ndarray
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def extent(self):
        # type: () -> Tuple[float, float, float, float]
        """``(min_x, min_y, max_x, max_y)``."""
        return (self.min_x, self.min_y, self.max_x, self.max_y)


class OccupancyMapImage:
    """A nav2 ``map_server`` map (PGM + YAML), ready to draw under a route.

    The one convention that has to be got right: a nav2 PGM stores **row 0 at
    maximum y**, while every world calculation here is y-up. The image is
    flipped once at load, so from then on ``rows[0]`` is minimum y and a single
    ``max_y - y`` gives the pixel row for any window.
    """

    def __init__(self, grid, resolution, origin_x, origin_y):
        # type: (np.ndarray, float, float, float) -> None
        """Build from an already y-up trinary grid.

        Args:
            grid: ``(h, w)`` uint8, row 0 = minimum y, holding the raw nav2
                greyscale values (0 occupied, 205 unknown, 254 free).
            resolution: Metres per cell.
            origin_x: World x of column 0's left edge.
            origin_y: World y of row 0's bottom edge.
        """
        if grid.ndim != 2:
            raise ValueError("map grid must be 2D, got shape %r" % (grid.shape,))
        self.grid = grid
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.height, self.width = grid.shape
        self._bgr = _colourise(grid)
        self._thick = None       # lazily dilated copy, see _thickened
        self._thick_factor = 0

    # -- construction ----------------------------------------------------

    @classmethod
    def from_yaml(cls, path):
        # type: (str) -> "OccupancyMapImage"
        """Load a nav2 map from its YAML sidecar.

        Args:
            path: Path to the ``.yaml``; ``image:`` is resolved relative to it.

        Returns:
            The loaded map, already flipped to y-up.

        Raises:
            FileNotFoundError: The YAML or the image it names is missing.
            ValueError: The YAML is missing ``resolution`` or ``origin``, or the
                image cannot be decoded. A map that loads wrong is worse than no
                map -- it would put the walls somewhere plausible and false.
        """
        with open(path, "r") as handle:
            meta = yaml.safe_load(handle) or {}
        if "resolution" not in meta or "origin" not in meta:
            raise ValueError("%s is not a nav2 map yaml: no resolution/origin" % path)
        image_name = meta.get("image")
        if not image_name:
            raise ValueError("%s names no image" % path)
        image_path = image_name
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(os.path.abspath(path)), image_name)
        if not os.path.isfile(image_path):
            raise FileNotFoundError("map image %s (named by %s)" % (image_path, path))
        raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError("could not decode map image %s" % image_path)
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        origin = meta["origin"]
        # nav2: row 0 is maximum y. Flip once, here, and never think about it again.
        return cls(np.flipud(raw).astype(np.uint8), float(meta["resolution"]),
                   float(origin[0]), float(origin[1]))

    # -- geometry --------------------------------------------------------

    @property
    def max_x(self):
        # type: () -> float
        """World x of the map's right edge."""
        return self.origin_x + self.width * self.resolution

    @property
    def max_y(self):
        # type: () -> float
        """World y of the map's top edge."""
        return self.origin_y + self.height * self.resolution

    def whole(self, size):
        # type: (Tuple[int, int]) -> MapWindow
        """The entire map, letterboxed into ``size`` without distorting it.

        Args:
            size: ``(width, height)`` of the panel to fill.

        Returns:
            A :class:`MapWindow` whose extent is *widened* past the map on
            whichever axis has the letterbox bars, so that the caller's
            world->pixel transform stays a single uniform scale.
        """
        w, h = int(size[0]), int(size[1])
        if w <= 0 or h <= 0:
            raise ValueError("window size must be positive, got %r" % (size,))
        span_x = self.max_x - self.origin_x
        span_y = self.max_y - self.origin_y
        # The scale is set by whichever axis runs out of panel first. Taking the
        # larger world span against the smaller panel side instead (the obvious
        # one-liner) zooms a tall building out until it is a stripe down the
        # middle of a mostly empty panel.
        metres_per_px = max(span_x / float(w), span_y / float(h))
        return self._window_at(0.5 * (self.origin_x + self.max_x),
                               0.5 * (self.origin_y + self.max_y),
                               metres_per_px, w, h)

    def window(self, centre_x, centre_y, span_m, size):
        # type: (float, float, float, Tuple[int, int]) -> MapWindow
        """A square-ish world window around a point, rendered at ``size``.

        The window is the smallest world rectangle with ``size``'s aspect ratio
        that contains a ``span_m`` square around the centre, so the returned
        extent maps to the image with one uniform scale in both axes.

        Args:
            centre_x: World x at the middle of the window.
            centre_y: World y at the middle of the window.
            span_m: The side of the square that must be visible, metres.
            size: ``(width, height)`` of the output image.

        Returns:
            A :class:`MapWindow`. World area outside the map reads as unknown.

        Raises:
            ValueError: ``size`` is not positive.
        """
        w, h = int(size[0]), int(size[1])
        if w <= 0 or h <= 0:
            raise ValueError("window size must be positive, got %r" % (size,))
        metres_per_px = max(float(span_m), 1e-3) / float(min(w, h))
        return self._window_at(centre_x, centre_y, metres_per_px, w, h)

    def _window_at(self, centre_x, centre_y, metres_per_px, w, h):
        # type: (float, float, float, int, int) -> MapWindow
        """Sample the map at a given scale about a centre.

        The row term of the transform is **negative** on purpose. This class
        stores its grid y-up (row 0 = minimum y) because that is the sane thing
        for its own API, while the images it returns are ordinary pictures with
        row 0 at the *top* -- maximum y -- because that is what every caller
        draws into. Getting the sign wrong yields a backdrop that is a perfect
        vertical mirror of the overlay drawn on it, which does not look broken:
        it looks like a plausible building the drone is flying through the wrong
        part of.
        """
        half_w = 0.5 * w * metres_per_px
        half_h = 0.5 * h * metres_per_px
        min_x, max_x = centre_x - half_w, centre_x + half_w
        min_y, max_y = centre_y - half_h, centre_y + half_h

        scale = metres_per_px / self.resolution
        # The +/-0.5 terms map a destination pixel's CENTRE to a source pixel's
        # centre. Without them the warp samples the source at cell edges, which
        # is a half-pixel offset from where the overlay's own transform puts the
        # same world point -- and leaves the top row of the window sampling one
        # row past the end of the grid, which renders as a strip of "unknown"
        # along an edge that is really floor.
        transform = np.array([
            [scale, 0.0, (min_x - self.origin_x) / self.resolution + 0.5 * (scale - 1.0)],
            [0.0, -scale, (max_y - self.origin_y) / self.resolution - 0.5 * (scale + 1.0)],
        ], dtype=np.float64)
        source = self._thickened(scale)
        image = cv2.warpAffine(
            source, transform, (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=UNKNOWN_BGR)
        return MapWindow(image=image, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

    def _thickened(self, scale):
        # type: (float) -> np.ndarray
        """The palette image with walls dilated to survive a downscale.

        ``INTER_NEAREST`` point-samples, so at the overview's ~2.5x reduction a
        one-cell partition falls between samples and simply is not drawn -- and
        the whole point of the overview is to show the rooms the drone did NOT
        enter. Dilating the occupied mask by the reduction factor first makes
        every wall that exists appear, at the cost of drawing it a cell thicker
        than it is. For a map that is read, not planned on, that is the right
        way round.
        """
        factor = int(round(scale))
        if factor <= 1:
            return self._bgr
        if self._thick_factor == factor:
            return self._thick
        occupied = np.all(self._bgr == np.array(OCCUPIED_BGR, dtype=np.uint8), axis=-1)
        grown = cv2.dilate(occupied.astype(np.uint8),
                           np.ones((factor, factor), np.uint8)) > 0
        out = self._bgr.copy()
        out[grown] = OCCUPIED_BGR
        self._thick, self._thick_factor = out, factor
        return out


def _colourise(grid):
    # type: (np.ndarray) -> np.ndarray
    """Turn nav2 trinary greyscale into the panel palette."""
    out = np.empty(grid.shape + (3,), dtype=np.uint8)
    out[:] = FREE_BGR
    out[grid <= 60] = OCCUPIED_BGR
    out[(grid > 60) & (grid < 250)] = UNKNOWN_BGR
    return out


def load_map_backdrop(path, logger=None):
    # type: (Optional[str], object) -> Optional[OccupancyMapImage]
    """Load a map if one is configured and present, else ``None``.

    Missing is a legitimate state -- the panel simply falls back to graph paper
    -- but a map that is configured and *broken* is not, and raises.

    Args:
        path: Path to the map YAML, or empty/None for "no backdrop".
        logger: Optional object with ``.info``/``.warn`` for one line of report.

    Returns:
        The loaded map, or ``None`` when no path was configured or the file is
        absent.
    """
    if not path:
        return None
    if not os.path.isfile(path):
        if logger is not None:
            logger.warn("map backdrop %s not found; the route panel falls back "
                        "to graph paper" % path)
        return None
    image = OccupancyMapImage.from_yaml(path)
    if logger is not None:
        logger.info("map backdrop %s: %dx%d cells at %.3f m, x[%.2f, %.2f] y[%.2f, %.2f]"
                    % (path, image.width, image.height, image.resolution,
                       image.origin_x, image.max_x, image.origin_y, image.max_y))
    return image
