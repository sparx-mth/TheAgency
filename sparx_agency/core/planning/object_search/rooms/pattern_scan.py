"""A look-around that turns only toward unknown floor, instead of a blind full circle."""
from __future__ import annotations

from dataclasses import dataclass
import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.exploration.view_gain import UnknownView, cone_from_camera
from sparx_agency.core.planning.objnav.types.command import NavigationCommand


@dataclass(frozen=True)
class PatternScanSettings:
    """``min_gain_m2``: unknown floor a heading must reveal to be worth turning to; ``max_turns``: 0 means one circle's turns."""

    min_gain_m2: float = 1.0
    max_turns: int = 0

    def __post_init__(self):
        if isinstance(self.min_gain_m2, bool) or not math.isfinite(self.min_gain_m2) or self.min_gain_m2 < 0:
            raise ValueError("min_gain_m2 must be finite and non-negative")
        if type(self.max_turns) is not int or self.max_turns < 0:
            raise ValueError("max_turns must be a non-negative integer")


class PatternScan:
    """Greedy look-around: the next heading is the one revealing the most unknown floor per turn, until little is left."""

    def __init__(self, policy, settings):
        self.policy = policy
        self.settings = settings
        self.stats = {"turns": 0, "scans": 0}
        self._turns = 0
        self._target = None

    def begin(self):
        self._turns = 0
        self._target = None
        self.stats["scans"] += 1

    def step(self, obs, world):
        """One in-place turn toward the chosen heading, or None when the look is done."""
        p = self.policy
        turn = p.episode.action_spec.turn_angle_rad
        budget = self.settings.max_turns or p.rotation_turns()
        if self._turns >= budget:
            return None
        xy = (obs.pose.x, obs.pose.y)
        view = self._view(world)
        if self._target is None or self._gain(view, xy, self._target) < self.settings.min_gain_m2:
            self._target = self._best_heading(view, xy, obs.pose.yaw, turn)
        if self._target is None:
            return None
        offset = normalize_angle(self._target - obs.pose.yaw)
        if abs(offset) < 0.5 * turn:
            self._target = None
            return self.step(obs, world) if self._turns < budget else None
        self._turns += 1
        self.stats["turns"] += 1
        direction = 1.0 if offset > 0 else -1.0
        return NavigationCommand.hold(
            final_yaw=normalize_angle(obs.pose.yaw + direction * turn),
            info={"kind": "pattern_scan", "pattern_turn": self._turns, "pattern_budget": budget,
                  "target_deg": round(math.degrees(self._target), 1)})

    def _view(self, world):
        p = self.policy
        camera = p.episode.camera
        k = camera.intrinsics
        cone = cone_from_camera(k.width, k.fx, float(camera.max_depth_m), p.glance_settings.blind_m)
        return UnknownView(world, cone, p.sight.resolved(world))

    def _gain(self, view, xy, heading):
        return float(view.unknown_ahead(xy, heading).sum()) * view.cell_area_m2

    def _best_heading(self, view, xy, yaw, turn):
        """The heading, a whole number of turns from ``yaw``, with the best gain per turn that clears the threshold."""
        best, best_score = None, 0.0
        circle = int(round(2.0 * math.pi / turn))
        for steps in range(1, circle // 2 + 1):
            for sign in (1, -1):
                heading = normalize_angle(yaw + sign * steps * turn)
                gain = self._gain(view, xy, heading)
                if gain >= self.settings.min_gain_m2 and gain / steps > best_score:
                    best, best_score = heading, gain / steps
        return best
