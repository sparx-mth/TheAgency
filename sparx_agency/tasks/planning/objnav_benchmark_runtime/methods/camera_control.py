"""One owner for public LOOK commands; positive REP-103 pitch looks down.

The existing action converter still quantizes and bounds pitch. Inspections
are transactions, not requests indexed by the changing pitch/yaw. A committed
connector preempts inspection but never relinquishes the camera on a tread.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from sparx_agency.core.planning.objnav.types.command import NavigationCommand


@dataclass(frozen=True)
class CameraControlSettings:
    """Inspection shape and cadence. Counts are actions, angles degrees.

    Attributes:
        inspection_turns: In-place turns per look-down sweep.
        inspection_actions: Actions after which an unfinished sweep restores.
        inspection_cooldown_actions: Minimum gap between ANY two inspections.
        periodic_inspection_actions: Minimum gap between two *unprompted*
            inspections -- the ones a spent floor allowance requests with no
            stair detection and no exhausted frontier behind them. The
            Ranchester recording fired one every cooldown once its floor
            allowance ran out, ten sweeps in 472 actions, each also costing
            four turns to recover the heading: 24 % of the episode. This is
            deliberately several times the cooldown.
        normal_pitch_deg: Level viewing pitch.
        ascent_pitch_deg: Pitch while climbing a connector.
        inspection_pitch_deg: Pitch during a look-down sweep or descent.
    """

    inspection_turns: int = 4
    inspection_actions: int = 12
    inspection_cooldown_actions: int = 24
    periodic_inspection_actions: int = 100
    normal_pitch_deg: float = 0.0
    ascent_pitch_deg: float = 30.0
    inspection_pitch_deg: float = 60.0

    def __post_init__(self):
        for key in ("inspection_turns", "inspection_actions", "inspection_cooldown_actions",
                    "periodic_inspection_actions"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("%s must be a positive integer" % key)
        if self.periodic_inspection_actions < self.inspection_cooldown_actions:
            raise ValueError("periodic_inspection_actions cannot be shorter than the cooldown")
        for key in ("normal_pitch_deg", "ascent_pitch_deg", "inspection_pitch_deg"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 60:
                raise ValueError("Camera pitch must be finite and between 0 and 60 degrees")


class CameraController:
    def __init__(self, actions, settings=None):
        self.actions, self.settings = actions, settings or CameraControlSettings()
        self.owner = "SEARCH"
        self.inspection = None
        self.inspected = set()
        self.last_inspection = -self.settings.inspection_cooldown_actions
        self.terrain_pitch = None
        self.last = {}

    def normal_view(self, pose):
        return (self.inspection is None
                and abs(pose.camera_pitch - math.radians(self.settings.normal_pitch_deg)) < math.radians(10))

    def periodic_due(self, step):
        """Whether an unprompted sweep may start: none yet, or the long cadence has elapsed."""
        return not self.inspected or step - self.last_inspection >= self.settings.periodic_inspection_actions

    def begin_inspection(self, obs, floor_id):
        key = (floor_id, round(obs.pose.x), round(obs.pose.y))
        if (not self.actions.has_camera_tilt or self.inspection is not None
                or key in self.inspected or obs.step < 6
                or obs.step - self.last_inspection < self.settings.inspection_cooldown_actions):
            return False
        self.inspected.add(key)
        self.last_inspection = obs.step
        self.inspection = {"start": obs.step, "yaw": obs.pose.yaw, "turns": 0, "restoring": False}
        return True

    def inspection_command(self, obs):
        scan = self.inspection
        if scan is None:
            return None
        s = self.settings
        if obs.step - scan["start"] >= s.inspection_actions:
            scan["restoring"] = True
        pitch = self._bounded(s.inspection_pitch_deg)
        if not scan["restoring"] and abs(obs.pose.camera_pitch - pitch) < math.radians(10):
            error = math.atan2(math.sin(obs.pose.yaw - scan["yaw"]), math.cos(obs.pose.yaw - scan["yaw"]))
            if abs(error) < self.actions.turn_angle_rad / 2 + 1e-5:
                if scan["turns"] >= s.inspection_turns:
                    scan["restoring"] = True
                else:
                    scan["turns"] += 1
                    scan["yaw"] = obs.pose.yaw + self.actions.turn_angle_rad
        if scan["restoring"] and abs(obs.pose.camera_pitch - self._bounded(s.normal_pitch_deg)) < math.radians(10):
            self.inspection = None
            return None
        return NavigationCommand.hold(final_yaw=None if scan["restoring"] else scan["yaw"],
                                      info={"kind": "stair_inspection", "inspection_restore": scan["restoring"]})

    def apply(self, obs, command, owner="SEARCH", direction=0, close_support=False):
        """The only policy function that writes command.camera_pitch."""
        s = self.settings
        if owner in ("TRAVERSE", "CONFIRM_DESTINATION", "RETREAT", "SAFE_HALT"):
            self.inspection = None
            looking_down = direction < 0 or close_support or owner == "RETREAT"
            desired = s.inspection_pitch_deg if looking_down else s.ascent_pitch_deg
            self.terrain_pitch = max(desired, self.terrain_pitch or 0.0)
            desired = self.terrain_pitch
        elif owner == "APPROACH_STAIRS":
            # Retain an inspection view until the approach/traversal transaction
            # ends, rather than fighting it with an ordinary search reset.
            self.inspection = None
            desired = max(s.normal_pitch_deg, math.degrees(obs.pose.camera_pitch))
        else:
            self.terrain_pitch = None
            if self.inspection is not None:
                owner = "RESTORE" if self.inspection["restoring"] else "INSPECT"
                desired = s.normal_pitch_deg if self.inspection["restoring"] else s.inspection_pitch_deg
            else:
                desired = s.normal_pitch_deg
                if not self.normal_view(obs.pose):
                    owner = "RESTORE"
        self.owner = owner
        requested = self._bounded(desired) if self.actions.has_camera_tilt and not command.stop else None
        self.last = {"owner": owner, "observed_pitch_rad": obs.pose.camera_pitch,
                     "requested_pitch_rad": requested, "observation_step": obs.step,
                     "sign": "positive_down", "inspection_count": len(self.inspected)}
        return replace(command, camera_pitch=requested, info=dict(command.info, camera=dict(self.last)))

    def _bounded(self, degrees):
        value = math.radians(degrees)
        if self.actions.min_pitch_rad is not None:
            value = max(value, self.actions.min_pitch_rad)
        if self.actions.max_pitch_rad is not None:
            value = min(value, self.actions.max_pitch_rad)
        return value
