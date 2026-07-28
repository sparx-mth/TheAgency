"""The expert: the safest route to the goal, encoded as a NavDP action.

This is the *teacher*, and it is deliberately not NavDP. The previous fine-tune
took NavDP's own trajectory and pushed it off the walls, which caps the student
at the teacher and cannot introduce behaviour the teacher never had. Here the
target is produced from the surveyed map, by machinery that already flies real
aircraft in this repo:

1. **Weighted A*** plans a route to the goal on the global map. Its soft
   clearance cost means the route already prefers the middle of a corridor to
   its edge, and because it plans on the *whole building* it turns toward a
   doorway several metres before the camera can see it -- which is precisely the
   behaviour a single-frame teacher can never demonstrate.
2. **The medial-axis corrector**
   (:class:`~...safety.trajectory_safety_corrector.TrajectorySafetyCorrector` in
   ``line_search`` mode) then samples clearance along the path normal and moves
   each waypoint to the local maximum -- the point equidistant from the walls on
   both sides. Its ``corner_swing`` widens that search at a turn, so the label
   swings wide around a corner instead of clipping its inside edge.
3. The first :attr:`ExpertConfig.horizon_m` metres are resampled to NavDP's 24
   steps and encoded as the action tensor NavDP's diffusion head predicts.

The horizon is arc-length, so the encoding does the arrival behaviour for free:
a goal 20 m away fills all 24 steps at cruise spacing, while a goal 2 m away
spreads 2 m over 24 steps and the per-step displacement shrinks toward zero --
which is exactly the "stop" signal NavDP's own post-processing looks for.

Finally the encoded label is *decoded again* and measured against the map. A
label whose own waypoints come closer than
:attr:`ExpertConfig.min_label_clearance_m` to geometry is thrown away rather
than taught: one poisoned target is worth more damage than one missing sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.safety.trajectory_safety_corrector import (
    TrajectorySafetyCorrector,
)
from sparx_agency.core.planning.safety.types import TrajectoryCorrectionParams
from sparx_agency.core.planning.vlas.navdp.geometry import point_to_pointgoal
from sparx_agency.tasks.planning.vlas.common.finetune.common.label_format import (
    to_navdp_label,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.polyline import (
    arclength, decode_action, resample, to_body, to_world, truncate,
    turn_magnitude_deg,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import Scene

# Reasons a candidate goal produced no label. Counted and reported, because the
# mixture of failures is how you tell a bad map from a bad sampler setting.
REJECT_NO_ROUTE = "no_route"
REJECT_SHORT = "route_too_short"
REJECT_UNSAFE = "label_hits_geometry"
REJECT_TURN = "label_turns_too_hard"
REJECT_GOAL_MOVED = "planner_moved_the_goal"


@dataclass(frozen=True)
class ExpertConfig:
    """How the expert route becomes a NavDP action label.

    Attributes:
        horizon_m: Arc length of route the label covers. NavDP's action is
            clamped at +-1 after a x4 scale, i.e. 0.25 m per step per axis, so 24
            steps can express at most 6 m. 4.8 m leaves headroom for the lateral
            component of a turn instead of saturating the clamp on every sample.
        horizon_steps: NavDP ``predict_size``. Do not change: it is baked into
            the pretrained decoder's output positional embedding.
        densify_m: Final spacing the route is resampled to before truncation.
        center: Run the medial-axis corrector. Off makes the label the raw A*
            route, which is useful for ablations.
        center_spacing_m: Waypoint spacing the corrector operates at. This is
            *not* ``densify_m`` and the difference matters: the line search moves
            each waypoint independently, so at 5 cm spacing a legitimate 10 cm
            sideways nudge is a 63-degree kink between neighbours. Centring at
            waypoint scale and densifying afterwards keeps the path smooth.
        center_max_shift_m: Cap on how far centring may move a waypoint.
        center_corner_swing: Extra lateral search range at a turn, as a fraction
            of the cap per 90 degrees of turn.
        center_step_m: Sampling spacing along the path normal.
        center_ramp_m: The correction is faded in over this distance from the
            aircraft. A drone cannot step sideways, so a label whose first
            waypoint is already displaced is not trackable however safe it looks;
            the ramp makes the label leave tangent to where the aircraft is.
        center_taper_m: Clearance at which centring stops. A waypoint that
            already has this much room is not moved. Without it the line search
            keeps hunting a distant clearance maximum across an open hall, which
            adds turning to a route that had no reason to turn -- the correction
            is for corridors, and this is what confines it to them.
        center_lookahead_m: Extra route centred beyond the horizon, so the last
            label waypoint is centred with its continuation in view rather than
            against a truncation.
        min_label_clearance_m: Reject the label if its own decoded waypoints come
            closer than this to geometry.
        goal_reach_tolerance_m: How far the planner's route may end from the
            goal it was given. The weighted A* *snaps* a blocked or unreachable
            goal onto a nearby free cell rather than failing, which would
            silently produce a sample whose goal token points at a wall while
            its label goes somewhere else entirely. Anything beyond half a cell
            of discretisation means the goal moved, and the sample is dropped.
        max_turn_deg: Reject the label if it deviates further than this from
            straight ahead inside the horizon. A near-reversal inside 4.8 m is
            not something a forward-facing aircraft at cruise can fly, and it
            saturates the action encoding; such goals are better dropped than
            taught.
        action_scale / action_clamp: NavDP's encoding constants.
    """

    horizon_m: float = 4.8
    horizon_steps: int = 24
    densify_m: float = 0.05
    center: bool = True
    center_spacing_m: float = 0.5
    center_max_shift_m: float = 0.9
    center_corner_swing: float = 0.0
    center_step_m: float = 0.05
    center_ramp_m: float = 1.0
    center_taper_m: float = 1.5
    center_lookahead_m: float = 2.0
    min_label_clearance_m: float = 0.30
    goal_reach_tolerance_m: float = 0.35
    max_turn_deg: float = 100.0
    action_scale: float = 4.0
    action_clamp: float = 1.0


@dataclass(frozen=True)
class ExpertLabel:
    """One training target and everything needed to audit it later."""

    action: np.ndarray             # (24, 3) float32 -- the diffusion target x0
    waypoints_body: np.ndarray     # (24, 2) float32 -- action decoded, body FLU
    goal_token: np.ndarray         # (2,)   float32 -- what the network is told
    goal_world: np.ndarray         # (2,)   float32 -- where the goal really is
    goal_kind: str
    goal_distance_m: float
    route_length_m: float
    horizon_used_m: float
    min_clearance_m: float
    mean_clearance_m: float
    turn_deg: float
    reaches_goal: bool


def make_corrector(config: ExpertConfig) -> TrajectorySafetyCorrector:
    """A medial-axis corrector configured for label generation.

    ``pin_first_k=1`` holds the aircraft's own position; the rest of the route is
    free to slide sideways onto the corridor centre-line.
    """
    return TrajectorySafetyCorrector(TrajectoryCorrectionParams(
        centering="line_search",
        center_step_m=config.center_step_m,
        corner_swing=config.center_corner_swing,
        max_total_shift_m=config.center_max_shift_m,
        max_step_m=config.center_max_shift_m,
        smoothing_passes=4,
        pin_first_k=1,
        pin_last=False,
        lateral_only=True,
    ))


def blend_correction(original: np.ndarray, corrected: np.ndarray, clearance: np.ndarray,
                     ramp_m: float, taper_m: float) -> np.ndarray:
    """Apply a centring correction only where and as fast as it makes sense.

    Two independent weights, multiplied:

    * an **arc-length ramp** from 0 at the aircraft to 1 at ``ramp_m``. The
      corrector pins only waypoint 0, so waypoint 1 may legally be displaced by
      the full shift cap -- a kink no aircraft can fly. The ramp makes the label
      leave tangent to the current motion and reach the centre-line smoothly.
    * an **openness taper** that reaches 0 once a waypoint already has
      ``taper_m`` of clearance. Centring is for corridors; in an open hall the
      nearest clearance maximum is metres away in an arbitrary direction, and
      chasing it adds turning to a route that had no reason to turn.

    Args:
        original: ``(N, 2)`` path as planned.
        corrected: ``(N, 2)`` path after centring, same length.
        clearance: ``(N,)`` signed clearance at the *original* waypoints, metres.
        ramp_m: Distance over which the correction reaches full strength.
        taper_m: Clearance at which the correction switches off. <= 0 disables.

    Returns:
        ``(N, 2)`` blended path.
    """
    weight = np.ones(original.shape[0], dtype=np.float64)
    if ramp_m > 0.0:
        weight *= np.clip(arclength(original) / float(ramp_m), 0.0, 1.0)
    if taper_m > 0.0:
        weight *= np.clip((float(taper_m) - clearance) / float(taper_m), 0.0, 1.0)
    return original + weight[:, None] * (corrected - original)


def build_label(scene: Scene, pose, goal_xy, goal_kind: str,
                config: Optional[ExpertConfig] = None,
                corrector: Optional[TrajectorySafetyCorrector] = None):
    """Turn one (frame pose, world goal) pair into a NavDP training target.

    Args:
        scene: The surveyed building, holding the map, the ESDF and the planner.
        pose: ``(x, y, yaw)`` world pose of the aircraft at this frame.
        goal_xy: ``(x, y)`` world goal, already known to be in the goal region.
        goal_kind: Which mixture component drew it (stored for diagnostics).
        config: Encoding parameters.
        corrector: Reuse one per scene -- it caches the field sampler.

    Returns:
        ``(label, None)`` on success, or ``(None, reason)`` with one of the
        ``REJECT_*`` constants so a campaign can report *why* samples were lost.
    """
    cfg = config or ExpertConfig()
    start = Pose2D(float(pose[0]), float(pose[1]), float(pose[2]))
    route = scene.plan_route(start, Pose2D(float(goal_xy[0]), float(goal_xy[1]), 0.0))
    if route is None:
        return None, REJECT_NO_ROUTE

    if float(np.hypot(route[-1, 0] - goal_xy[0],
                      route[-1, 1] - goal_xy[1])) > cfg.goal_reach_tolerance_m:
        return None, REJECT_GOAL_MOVED

    route_length = float(arclength(route)[-1])
    head, _ = truncate(resample(route, cfg.center_spacing_m),
                       cfg.horizon_m + cfg.center_lookahead_m)
    if head.shape[0] < 2:
        return None, REJECT_SHORT

    if cfg.center:
        corrector = corrector or make_corrector(cfg)
        u_rep, d_obs, origin_x, origin_y = scene.corrector_field()
        corrector.set_field(u_rep, scene.resolution, origin_x, origin_y, d_obs=d_obs)
        head = blend_correction(
            head, corrector.correct(head).waypoints.astype(np.float64),
            scene.clearance(head[:, 0], head[:, 1]),
            cfg.center_ramp_m, cfg.center_taper_m)

    horizon, used = truncate(resample(head, cfg.densify_m), cfg.horizon_m)
    if horizon.shape[0] < 2 or used < 1e-3:
        return None, REJECT_SHORT

    action = to_navdp_label(to_body(horizon, pose), horizon=cfg.horizon_steps,
                            scale=cfg.action_scale, clamp=cfg.action_clamp,
                            with_yaw=True)
    waypoints_body = decode_action(action, cfg.action_scale)
    turn = turn_magnitude_deg(waypoints_body)
    if turn > cfg.max_turn_deg:
        return None, REJECT_TURN
    clearance = scene.clearance(*to_world(waypoints_body, pose).T)
    if float(clearance.min()) < cfg.min_label_clearance_m:
        return None, REJECT_UNSAFE

    goal_body = to_body(np.asarray([goal_xy], dtype=np.float64), pose)[0]
    token = point_to_pointgoal(float(goal_body[0]), float(goal_body[1]))
    return ExpertLabel(
        action=action.astype(np.float32),
        waypoints_body=waypoints_body.astype(np.float32),
        goal_token=np.asarray(token, dtype=np.float32),
        goal_world=np.asarray(goal_xy, dtype=np.float32),
        goal_kind=goal_kind,
        goal_distance_m=float(np.hypot(goal_body[0], goal_body[1])),
        route_length_m=route_length,
        horizon_used_m=float(used),
        min_clearance_m=float(clearance.min()),
        mean_clearance_m=float(clearance.mean()),
        turn_deg=turn,
        reaches_goal=bool(route_length <= cfg.horizon_m + 1e-6),
    ), None
