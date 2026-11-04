"""The path the agent walked while tracking one candidate: what makes a close-range STOP trustworthy.

Two close-range situations look alike to the detector and are opposites to
the closing. A target the agent *tracked and approached over metres* -- it
locked the object from across the room, A* walked it in, the memory was
refined on the way -- is known: at the end of that path a cropped box, a
detector drop or a label that slips to the supporting surface is not a reason
to give it back. A target that *appears at arm's length with no history* --
the agent turned and there it was -- is a single viewpoint, and a single
viewpoint at 0.7 m is exactly where a counter front reads as a bed. The ledger
tells the two apart: it sums the planar displacement of every action since the
takeover started and tracks how much nearer the agent got.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


class ApproachHistory:
    """Planar path walked since the takeover began, and how much nearer to the target it brought the agent."""

    def __init__(self, xy: Tuple[float, float], range_m: float, step: int):
        self.start_xy: Tuple[float, float] = (float(xy[0]), float(xy[1]))
        self.start_range_m = float(range_m)
        self.min_range_m = float(range_m)
        self.start_step = int(step)
        self.path_m = 0.0
        self.actions = 0
        self._last_xy = self.start_xy
        self._last_step: Optional[int] = None

    def record(self, xy: Tuple[float, float], range_m: float, step: int) -> None:
        """Add this action's displacement; a repeated call for the same step is ignored."""
        step = int(step)
        if self._last_step is not None and step <= self._last_step:
            return
        here = (float(xy[0]), float(xy[1]))
        self.path_m += math.dist(self._last_xy, here)
        self._last_xy = here
        self._last_step = step
        self.actions += 1
        if math.isfinite(float(range_m)):
            self.min_range_m = min(self.min_range_m, float(range_m))

    @property
    def closed_m(self) -> float:
        """How much nearer the agent has got to the target than where it first saw it."""
        return max(0.0, self.start_range_m - self.min_range_m)

    def qualifies(self, path_m: float, closed_m: float) -> bool:
        """Whether this is an EXTENDED approach: at least ``path_m`` walked and ``closed_m`` of range closed."""
        return self.path_m + 1e-9 >= float(path_m) and self.closed_m + 1e-9 >= float(closed_m)

    def diagnostics(self) -> Dict[str, object]:
        return {"path_m": round(self.path_m, 3), "closed_m": round(self.closed_m, 3),
                "start_range_m": round(self.start_range_m, 3), "min_range_m": round(self.min_range_m, 3),
                "actions": self.actions, "start_step": self.start_step}
