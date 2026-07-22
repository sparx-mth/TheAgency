"""Roll-assisted waypoint follower (waypoint navigation + cross-track ROLL).

This tracker keeps the deliberately "stupid" one-axis
:class:`~sparx_agency.core.planning.trackers.waypoint_follower.WaypointFollower`
in full charge of navigation — look at the next point, align to it, then advance,
with its discrete yaw pulse/settle loop, map-freeze gate and per-axis handshake
all unchanged — and layers a single extra behaviour on top: a lateral ("ROLL",
``+vy`` = left) velocity that continuously pulls the drone back onto its
trajectory whenever it drifts sideways. While turning or holding it may also add
a small forward/back nudge for along-track drift.

It is the answer to "keep the follower that works, just stop it drifting off the
line": every published command is the base follower's ``(vx, wz)`` plus the
corrector's ``(vy, +/- vx)``. The correction is scaled by what the base is doing
(full while advancing, weak while turning, small while holding) — see
:class:`CrossTrackRollCorrector`.

The public surface mirrors ``WaypointFollower`` / ``MultiAxisFollower`` (``params``
/ ``state`` / ``done`` / ``required_axis`` / ``settle_map_updates_required`` /
``set_path`` / ``step`` / ``reset``) so an adapter can swap it in, with the added
``vy`` channel carried on the returned :class:`FollowerCommand`.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sparx_agency.core.common.types import ControlCommand, Pose2D

from ..waypoint_follower.follower import WaypointFollower
from ..waypoint_follower.types import ControlAxis, FollowerCommand, FollowerState
from .corrector import CrossTrackRollCorrector
from .params import CrossTrackRollParams


class RollAssistFollower:
    """Waypoint follower plus a cross-track ROLL corrector on ``vy``."""

    name: str = "roll_assist_follower"

    def __init__(
        self,
        base: WaypointFollower,
        corrector: Optional[CrossTrackRollCorrector] = None,
    ) -> None:
        """Wrap an existing ``WaypointFollower`` with a ROLL corrector.

        Args:
            base: The one-axis follower that owns all navigation (unchanged).
            corrector: The cross-track corrector; a default one is built if omitted.
        """
        self._base = base
        self._corrector = corrector or CrossTrackRollCorrector()

    # ─── Delegated navigation surface ────────────────────────────
    @property
    def params(self):
        """The base follower's params (drives prediction / the banner)."""
        return self._base.params

    @property
    def roll_params(self) -> CrossTrackRollParams:
        """The cross-track corrector's params (for the adapter banner)."""
        return self._corrector.params

    @property
    def corrector(self) -> CrossTrackRollCorrector:
        return self._corrector

    @property
    def state(self) -> FollowerState:
        return self._base.state

    @property
    def active_path(self):
        """The base follower's re-anchored path; navigation is entirely its own.

        Indices line up with ``FollowerCommand.wp_idx``, so a caller can resolve
        the waypoint being pursued to a position.
        """
        return self._base.active_path

    @property
    def done(self) -> bool:
        return self._base.done

    @property
    def settle_map_updates_required(self) -> int:
        return self._base.settle_map_updates_required

    def required_axis(self) -> Optional[ControlAxis]:
        return self._base.required_axis()

    def reset(self) -> None:
        self._base.reset()
        self._corrector.reset()

    def set_path(self, waypoints: Sequence[Pose2D], pose: Optional[Pose2D]) -> None:
        self._base.set_path(waypoints, pose)
        self._corrector.reset()

    # ─── Step ────────────────────────────────────────────────────
    def step(
        self,
        pose: Pose2D,
        dt: float,
        *,
        axis_confirmed: bool = True,
        hold: bool = False,
        map_ready: bool = True,
    ) -> FollowerCommand:
        """Advance the base follower one tick and add the ROLL correction.

        The correction is suppressed (eased to zero) whenever the base is being
        gated — an external ``hold``, an unconfirmed required axis, or the DONE
        terminal — so we never inject motion while the platform is meant to be
        still. Otherwise the correction is scaled by the base's regime this tick:
        full lateral gain while advancing, weak while turning, small while holding.
        """
        # Mirror the base's own gating test (its pre-step state decides the axis),
        # so a held / unconfirmed tick adds no correction.
        axis_before = self._base.required_axis()
        gated = hold or (axis_before is not None and not axis_confirmed)

        base_cmd = self._base.step(pose, dt, axis_confirmed=axis_confirmed,
                                   hold=hold, map_ready=map_ready)

        if gated or base_cmd.done:
            vy, vx_extra = self._corrector.relax(dt)
        else:
            advancing = base_cmd.state == FollowerState.ADVANCE
            yaw_active = abs(base_cmd.wz) > self._corrector.params.yaw_active_eps
            vy, vx_extra = self._corrector.correct(
                pose, self._base.active_path, base_cmd.wp_idx,
                advancing=advancing, yaw_active=yaw_active, dt=dt)

        combined = ControlCommand.velocity(
            base_cmd.vx + vx_extra, vy, 0.0, base_cmd.wz, tracker=self.name)
        return FollowerCommand(
            command=combined,
            state=base_cmd.state,
            required_axis=base_cmd.required_axis,
            freeze=base_cmd.freeze,
            done=base_cmd.done,
            wp_idx=base_cmd.wp_idx,
            num_waypoints=base_cmd.num_waypoints,
        )
