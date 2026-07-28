"""The XTEND, modelled honestly enough to judge a turn by.

A first-order velocity-tracking body, as in the controller's own closed-loop
tests, plus the one thing those tests leave out and that decides this particular
comparison: **the yaw this airframe delivers depends on what the translation
under it is doing.** From the flight logs the drift-PID controller was tuned on
(``core/planning/trackers/drift_pid/README.md``, nav_debug of 2026-07-21):

| translation under an active yaw | delivered rotation |
|---|---|
| forward (+0.08 m/s and up) | 30-68% of commanded, immediate bite |
| none (in place) | ~11% of commanded, coasts |
| backward (-0.10 m/s) | degrades, then **reverses** |

That table is the whole reason turn anticipation is worth having, so a rig that
leaves it out cannot measure the benefit: it would hand a stop-and-spin the same
free rotation a flying drone gets, and conclude the two manoeuvres are much
closer than they are. It is also the reason to keep the coupling switchable —
``yaw_bite=False`` gives the idealised airframe, and quoting both numbers is the
honest way to present the result.

What is deliberately NOT modelled, and why it matters when reading the output:
no obstacles or collision (the route is planned clear and the controller is not
being asked to avoid anything here), no localization noise or latency (both are
tested in the controller's own suite), and no roll-versus-yaw coupling — the
logs say YAW+ROLL is worse than YAW+forward but never say by how much, and
inventing a number would flatter or punish the crab on no evidence. So the crab
is, if anything, given a slightly easy ride here while the in-place spin is
modelled at its measured worst.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from sparx_agency.core.common.types import Pose2D


@dataclass(frozen=True)
class AirframeParams:
    """How the modelled drone answers a velocity command (SI, body frame FLU).

    Attributes:
        lag: Fraction of the gap to the commanded velocity closed each tick.
            0.45 at 10 Hz, matching the controller's own test plant.
        yaw_bite: Model the measured yaw-versus-translation coupling. False
            gives every commanded yaw in full, whatever the drone is doing.
        bite_static: Fraction of a commanded yaw delivered standing still.
        bite_flying: Fraction delivered with the forward speed at or above
            ``bite_full_vx``.
        bite_full_vx: Forward speed at which the yaw bites fully (m/s).
        bite_reverse_vx: Backward speed at which the delivered yaw has fully
            inverted (m/s, positive number).
        drift_vx: Constant world-frame push along x (m/s).
        drift_vy: Constant world-frame push along y (m/s). The disturbance the
            controller exists to cancel; -0.035 is the value its README quotes.
    """

    lag: float = 0.45
    yaw_bite: bool = True
    bite_static: float = 0.11
    bite_flying: float = 0.68
    bite_full_vx: float = 0.15
    bite_reverse_vx: float = 0.10
    drift_vx: float = 0.0
    drift_vy: float = -0.035


class Airframe:
    """A drone that tracks velocity commands with lag, drift and yaw coupling."""

    def __init__(self, params=None, pose=None):
        # type: (AirframeParams, Pose2D) -> None
        self.params = params or AirframeParams()
        self.pose = pose or Pose2D(0.0, 0.0, 0.0)
        self._v = [0.0, 0.0, 0.0]

    def yaw_delivery(self, vx):
        # type: (float) -> float
        """Fraction of a commanded yaw rate this translation actually delivers.

        Linear between the measured points, and allowed to go negative under a
        backward translation because that is what the drone did: a saturated
        right-yaw riding -0.10 m/s turned it LEFT.
        """
        p = self.params
        if not p.yaw_bite:
            return 1.0
        if vx >= 0.0:
            frac = min(1.0, vx / p.bite_full_vx) if p.bite_full_vx > 0 else 1.0
            return p.bite_static + (p.bite_flying - p.bite_static) * frac
        frac = min(1.0, -vx / p.bite_reverse_vx) if p.bite_reverse_vx > 0 else 1.0
        return p.bite_static - (p.bite_static + 1.0) * frac

    def step(self, vx, vy, wz, dt):
        # type: (float, float, float, float) -> Pose2D
        """Advance one tick under a body-frame velocity command."""
        p = self.params
        for i, target in enumerate((vx, vy, wz)):
            self._v[i] += p.lag * (target - self._v[i])
        bx, by, bwz = self._v
        yaw = self.pose.yaw
        wx = bx * cos(yaw) - by * sin(yaw) + p.drift_vx
        wy = bx * sin(yaw) + by * cos(yaw) + p.drift_vy
        self.pose = Pose2D(self.pose.x + wx * dt,
                           self.pose.y + wy * dt,
                           self.pose.yaw + bwz * self.yaw_delivery(bx) * dt)
        return self.pose
