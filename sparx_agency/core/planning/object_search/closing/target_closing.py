"""Episode-local target takeover: no transition back to global exploration."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.action_converter.action_choice import pitch_action, turn_action
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.object_search.closing.approach_history import ApproachHistory
from sparx_agency.core.control.camera.box_pitch_servo import (
    REGION_CENTRE, REGION_CUT_BOTTOM, REGION_CUT_TOP, REGION_LOWER, REGION_UPPER, vertical_fix)
from sparx_agency.core.mapping.objects.box_extent import clipped_box
from sparx_agency.core.planning.object_search.perception.projection import observed_objects
from sparx_agency.core.planning.object_search.perception.cycle import contradicted_by_map
from sparx_agency.core.planning.object_search.world_model.sightlines import unknown_around
from sparx_agency.core.mapping.objects.support_surface import (
    SUPPORT_SURFACE_CLASSES, is_support_surface, supports)
from sparx_agency.core.planning.object_search.closing.target_backoff import BackoffManeuver
from sparx_agency.core.planning.object_search.perception.target_context import suspect
from sparx_agency.core.mapping.tracking.target_memory_3d import TargetMemory3D
from sparx_agency.core.planning.object_search.closing.target_path import TargetApproachPath


@dataclass(frozen=True)
class TargetClosingSettings:
    confidence: float = 0.50
    # A detector's confidence in ONE object swings with the view. The Ranchester
    # couch, 3.8 m from the stair foot, read 0.68 at one heading, 0.42 one
    # 30-degree turn later with the same object centred in the frame, and 0.52
    # clipped at the next heading. The lock needs two CONSECUTIVE frames, and
    # the agent's own centring turn broke the run every time: twenty-four frames
    # with the couch in view, two twelve-step verifications released. A
    # takeover STARTS on ``confidence``; once a candidate is active, a box of
    # the target's class that projects onto its anchor is a re-sighting and
    # counts from this lower threshold -- the tracking convention: detect
    # high, associate low. A weak box that lands anywhere else is nothing.
    track_confidence: float = 0.30
    confirmation_frames: int = 2
    association_radius_m: float = 0.50
    # Within this range the camera pitch FOLLOWS THE TARGET'S ELEVATION -- the
    # close-range band of the dynamic pitch controller (``_pitch``, ``_elevation``,
    # since 2026-10-07): a target below the camera (a toilet 0.4 m high drops under a
    # level camera's lower edge from about 1.4 m in) is looked DOWN at, one above it
    # (a wall-mounted television, the top of a wardrobe) is looked UP at, by at least
    # one tilt step; beyond the band the approach is level. The name is historical:
    # until 2026-10-07 only the look-down existed, and a hardcoded LOOK_DOWN drove
    # high targets out of the frame.
    look_down_distance_m: float = 1.30
    # THE GEOMETRIC PITCH IS QUANTISED WITH HYSTERESIS (since 2026-10-14 evening). ``_pitch``
    # rounds the elevation it wants to the protocol's tilt step; a target whose depression
    # from the next step's range lands at half a step -- a 0.39 m toilet seen from 2.2 m by a
    # camera 0.9 m up is 15.0 degrees down, the step is 30 -- rounds to 0 on one frame and to
    # 30 on the next as the memory's height flickers by millimetres between frames taken at
    # different pitches, and the converter serves a pitch change before any move: Darden/
    # 000034 spent its last nine actions on LOOK_DOWN / LOOK_UP 2.19 m from a locked toilet
    # and never stepped. The step the camera is AT is kept while the wanted angle lies within
    # ``(0.5 + pitch_hysteresis)`` steps of it; the change needs that much more.
    pitch_hysteresis: float = 0.15
    # A target within this height of the camera is level with it: neither LOOK is
    # forced, and the verification frame falls back to whichever view keeps the box
    # inside the frame.
    elevation_band_m: float = 0.15
    terminal_distance_m: float = 1.00
    range_tolerance_m: float = 0.05
    refine_distance_m: float = 0.15
    max_verify_steps: int = 12
    # THE VERIFICATION WALKS TOWARD WHAT IT CAN SEE BUT CANNOT PLACE (since 2026-10-14). A box
    # of the target's class on the candidate's bearing WITHOUT a usable depth -- a toilet at
    # 5 m, beyond what the projection accepts -- is a sighting of the candidate, not its
    # absence: the consecutive-frame chain is carried across such a frame, the verification
    # faces the box and steps toward it, and ``max_verify_steps`` does not count the step
    # (at most ``max_verify_approach_steps`` such steps per candidate). The Markleeville
    # toilet (/000188, actions 20-31) was in view at 5 m on every other frame and the
    # verification turned in place at it for its whole budget, twice.
    max_verify_approach_steps: int = 24
    # THE TAKEOVER MAY START ON AN RGB DETECTION WITHOUT DEPTH (since 2026-10-21). The depth sensor
    # returns nothing past about 5 m, and the detector finds the target's class in the RGB frame well
    # beyond that; such a box has no 3-D projection, so nothing could start on it and the search walked
    # on. With ``bearing_start`` a confident, whole box of the target's class WITHOUT a usable depth
    # begins a bounded BEARING approach: the agent faces the box by its image-space bearing and steps
    # toward it, re-facing it on every frame it is seen, until a frame projects with depth -- then the
    # ordinary verification starts there, as if the box had been seen fresh from that spot (its clock
    # starts at that frame; the bearing steps are not verification actions). The approach gives up,
    # without a rejection memory (nothing was placed), after ``bearing_max_actions`` actions, after
    # ``bearing_lost_frames`` consecutive frames without the class in the frame at ``track_confidence``,
    # or when the steps stop moving the agent (something in the way); the spot it gave up from is
    # remembered, and no bearing approach starts again within ``bearing_retry_m`` of it on the same
    # storey -- the far flicker is given one walk, not one per frame. Two more ends (Corozal/000001 of
    # 2026-10-21: a television first seen at about 5.5 m was walked toward for the whole bound, 3.75 m, and
    # never projected -- a screen is a hole in the Gibson mesh, depthless at any range -- and the walk left
    # the agent beside another television it then STOPped on): ``bearing_max_walk_m`` ends the approach once
    # the agent has walked that far toward the box, since a box first seen within the depth cap plus that
    # distance is in range by then and one still depthless is unmeasurable or not where the bearing pointed;
    # ``bearing_near_box_frac`` ends it when the box fills that share of the frame's height without depth --
    # the object is within arm's reach and cannot be positioned, and no blind STOP is taken on it.
    bearing_start: bool = True
    bearing_max_actions: int = 16
    bearing_lost_frames: int = 3
    bearing_retry_m: float = 1.5
    bearing_max_walk_m: float = 3.0
    bearing_near_box_frac: float = 0.30
    # THE SWEEP IS ONE CIRCLE, NOT THE BUDGET (since 2026-10-14). An unlocked candidate that has
    # dropped out of view is swept for at the bearing and one turn to each side (``_scan``: the
    # headings 0, +30, 0, -30); the cycle takes four or five actions, and a candidate not in any
    # of those headings is not brought back by the same headings again -- the box went behind a
    # jamb when the agent stepped (Wiconisco/000014, actions 231-241: eleven turns in place at a
    # toilet one step had hidden). After this many consecutive sweep actions without a frame of
    # the candidate the verification is released as it would be at its budget; the rest of the
    # budget is the search's. 0 disables the cap (the budget alone ends the sweep).
    verify_sweep_max_actions: int = 6
    max_reacquire_steps: int = 24
    max_closing_steps: int = 160
    # THE INSPECTION STEPS WHEN THE RANGE IS A STEP OUT (since 2026-10-14). A fresh box
    # measured beyond the terminal range by less than one forward step, with the agent
    # facing within one turn of it, is one MOVE_FORWARD from a STOP; the inspection used
    # to turn in place at it instead (Wiconisco/000193, actions 476-499: the toilet's near
    # edge at 1.0505 m against a 1.05 m limit, twenty-four alternating turns at the step
    # limit). At most ``inspection_max_steps`` such steps per inspection.
    inspection_max_steps: int = 3
    # THE CLOSEST REACHABLE APPROACH IS TERMINAL (since 2026-10-15). The standoff ring stands 0.70 m
    # about the 3-D memory -- a point on the object's NEAR surface -- and a big object has no floor
    # there: Collierville/000009's 1.5 m television on its cabinet put every ring point inside the
    # cabinet, against it or behind it, and three correct locks were released from 3 m as "no safe
    # path from here" with the television in view. ``target_path`` now widens the ring out to the
    # terminal radius when no standoff point has a route (``widened``), and a goal there is reached
    # within the arrival tolerance -- up to one forward step outside the terminal range. From such a
    # standoff the inspection's step rule above closes the gap; when its steps are spent, or a step is
    # blocked (the cabinet), with the target FRESH in view within one step of the limit, the agent
    # stands as close as the floor allows and the closing STOPs. The alternative -- "the inspection saw
    # nothing", a 2 m rejection about a real target -- gives the episode up. False restores the release.
    closest_approach_stop: bool = True
    # A LOCKED TARGET IS WALKED TO THROUGH THE UNKNOWN BEFORE THE SPOT IS GIVEN UP (since
    # 2026-10-14). The planner blocks unknown cells, and a level camera 0.88 m up sees the
    # floor only from about 1.4 m out, so the floor between a corridor and the toilet seen
    # through a doorway is routinely unknown; the standoff ring then had no path, the lock
    # was released as boxed in and the spot refused a new takeover (Collierville/000023,
    # action 375, the toilet at 2.8 m in plain view). When no route to the ring exists on
    # the known floor, the ring is planned again with unknown cells passable at
    # ``through_unknown_cost`` times the cost of open floor -- the target was seen along
    # that line of sight -- and the committed route is re-checked against the map as it
    # fills in, like any other. ``through_unknown=False`` restores the strict planner alone.
    through_unknown: bool = True
    through_unknown_cost: float = 8.0
    # A box at or above this confidence starts a takeover past the rejection memory
    # of UNVERIFIED and map-contradicted releases and the release cooldown (since
    # 2026-10-07): the memory stops the same far flicker from restarting the
    # takeover, and a 0.95 toilet in plain view at 2.4 m is not a flicker -- the
    # Allensville toilet run refused nine such frames (actions 45-53) because a lock
    # the map had wrongly outvoted was remembered 2 m around it. A spot released by a
    # FAILED INSPECTION (the closing looked from close and saw nothing) or an approach
    # that stalled is not bypassed -- the detector's word from far does not beat a look
    # from near -- and neither is a boxed-in spot, where the agent itself is the
    # problem. Border clipping, incoherent depth and the map's contradiction stand.
    # 1.0 disables the override.
    override_confidence: float = 0.80
    # The semantic sanity check (``target_context.suspect``): a candidate in a room
    # whose STRONG type excludes the target, or whose measured surface height
    # contradicts its class (a counter top read as a bed), counts
    # ``context_penalty`` of its confidence toward the start threshold and needs
    # ``context_confirmation_frames`` consecutive frames from at least two
    # viewpoints ``context_baseline_m`` apart to lock. Allensville/2 and
    # Newfields/2 of the 5x3 benchmark STOPped on a kitchen counter at 0.55 m after
    # two frames from one spot. ``context_check=False`` is the former behaviour.
    context_check: bool = True
    context_penalty: float = 0.70
    context_confirmation_frames: int = 4
    context_baseline_m: float = 0.30
    # In VERIFY a fresh, aligned candidate too close for a step used to be held --
    # and a satisfied hold is idle, which the headless agent turns into a TURN that
    # moves the box out of the frame (Allensville toilet, action 77: centred at
    # 1.18 m, TURN_LEFT, lost). The verification frame is chosen to keep the box
    # whole: a LOOK_DOWN for a low target (the user's close-range policy), else the
    # one LOOK or TURN whose predicted shift leaves the box inside the frame.
    verify_keep_in_frame: bool = True
    # A box touching the image border is a partial view: its depth centroid is
    # unreliable and the clipped end of a bed reads as a sofa. Such a box may
    # not START a takeover; once locked the approach tolerates spill-over.
    border_margin_px: int = 8
    # An unlocked candidate is unverified evidence. Releasing it back to
    # exploration (remembering the spot) keeps the no-return contract for
    # locked targets only; False restores the historical episode-ending error.
    release_unverified: bool = True
    rejection_radius_m: float = 1.00
    # The rejection memory stops the SAME far flicker from restarting the takeover
    # from the same place; a candidate seen from within this range is new
    # evidence worth two frames whatever was released there before (Ranchester
    # attempt 3: the couch released at 3.8 m from the stair head was never
    # re-verified from 1 m, and the episode was lost on that).
    rejection_min_range_m: float = 2.00
    # A long object seen from far shows a different part of itself in every
    # frame, and its visible centroid moves more than half a metre between two
    # of them. The association radius grows by this much per metre of range
    # beyond two metres (3.8 m -> 0.77 m, 5 m -> 0.95 m); within two metres it
    # is the plain radius.
    association_range_gain: float = 0.15
    # THE VISIBLE PART MOVES ALONG THE LINE OF SIGHT (since 2026-10-11). The association
    # gate is a cylinder about the viewing ray, not a sphere about the anchor:
    # ``association_radius_m`` ACROSS the ray and in height, that plus
    # ``association_depth_m`` -- or the anchor frame's footprint diameter, whichever is
    # larger -- ALONG it. When the frame's lower edge cuts a low object the depth patch
    # lands on what is left in view, its upper and farther part: the Darden toilet's
    # centroid jumped from the bowl (0.38 m up) to the tank (0.59 m up, 0.47-0.49 m back
    # along the ray, 1.6-3.3 cm across it) the moment its box touched the bottom edge,
    # 0.50-0.53 m from the anchor against a 0.50 m sphere, and every 0.92-0.94 frame of
    # the toilet the agent was standing beside was discarded as another object until the
    # inspection released it as "saw nothing in any view" (the 1,000-episode validation
    # run, 2026-10-11: 63 % of the 148 toilet failures had locked on and released a
    # candidate). Across the ray the gate is as tight as before: two chairs side by side
    # stay two chairs.
    association_depth_m: float = 0.60
    # THE PART UNDER THE FRAME STANDS IN FRONT (since 2026-10-11). The terminal range of a
    # fresh box cut by the frame's lower edge -- of a target below the camera, with the
    # part in view now standing higher than the lowest part the lock has measured -- is the
    # smaller of the visible near edge and the planar range to the 3-D memory. The near
    # edge of what is in view is the near edge of the UPPER part of the object (the tank,
    # a backrest); the part the frame cut off (the bowl, the seat) stands in front of it
    # for the furniture of this benchmark, and the memory, fused from the whole-object
    # frames of the approach, is the centroid inside the object. It is also the point the
    # approach's standoff is measured from: without this the agent arrived by the memory
    # and was out of range by the measurement, in the same frame (the Darden tank's near
    # edge read 1.35 m with the agent 0.89 m from the bowl). The box is still fresh and
    # associated: this is no STOP from memory alone.
    floor_cut_terminal: bool = True
    floor_cut_min_hidden_m: float = 0.05
    # BELOW THE FIELD OF VIEW (since 2026-10-11). Without LOOK actions (the published Gibson
    # protocol) a low target at the terminal range drops under the frame -- the Wiconisco
    # toilet, 0.47 m high, shows under 40 px of its rim from 0.9 m, where the detector reads
    # 0.12-0.18 -- and no yaw sweep brings it back; the inspection released it as "saw
    # nothing in any view" with the agent inside the success region. When the geometry (the
    # camera's height, pitch and lower edge; the target's measured top and remembered
    # range) predicts fewer than ``blind_cone_min_fraction`` of the frame's rows of the
    # target in view -- the sliver threshold of ``perception.clipped_box``: a box this short
    # is not recognisable -- with its bearing inside the horizontal field, the target seen
    # fresh within ``blind_cone_max_missing`` actions and the memory within the terminal
    # range, the closing STOPs: the missing box is explained, and the one action that would
    # bring the object back, LOOK_DOWN, does not exist. The dynamic pitch controller is this
    # rule's counterpart where LOOK exists, so the rule is off whenever the action spec has
    # the tilt pair. ``blind_cone_stop=False`` restores the sweep and its release.
    blind_cone_stop: bool = True
    blind_cone_max_missing: int = 6
    blind_cone_min_fraction: float = 0.15
    # THE BOX IN THE FRAME DRIVES THE PITCH (since 2026-10-11; the image-space servo of
    # ``core/control/camera/box_pitch_servo``). Within ``bbox_pitch_distance_m`` of the target, with LOOK
    # actions and a FRESH box, the camera pitch is the servo's and not the geometry's: a box whose
    # centre lies in the lower ``bbox_pitch_region`` of the frame's rows -- or that the next step
    # toward the target is predicted to cut at the lower edge -- is looked DOWN at by the rounded
    # number of tilts (one at least); one in the upper region, or about to be cut by the top, is
    # looked UP at; one in the centre band leaves the camera where it is. No class, no height and
    # no 3-D memory enter: the frame-cut toilets of the 1,000-episode run showed the geometric
    # prediction biased exactly when it mattered (a cut frame lands its depth patch on the upper
    # part of the object), and a geometric tilt forced against a centred box would hunt against
    # this one, so when the box is fresh the servo alone has the camera. The geometric pitch
    # (``_pitch``) is the fallback for a frame WITHOUT a fresh box -- an occluded approach step --
    # and the whole approach beyond the distance. The verification frame (``_verification_view``)
    # asks the servo first too. The band is never narrower than one tilt's rounding dead band,
    # so the servo cannot hunt whatever the region says. ``bbox_pitch=False`` restores the
    # geometric controller alone.
    bbox_pitch: bool = True
    bbox_pitch_distance_m: float = 1.50
    bbox_pitch_region: float = 0.25
    # THE TERMINAL STOP IS TAKEN ON A WHOLE, CONFIDENT BOX (since 2026-10-11). Every terminal STOP
    # of the Gibson recordings fired on a box cut by the frame's lower edge -- at a metre a level
    # camera 0.88 m up cuts any floor-standing object -- whose depth patch is the upper part's.
    # With LOOK actions, a fresh in-range box the servo can re-frame (cut, or off-centre) is
    # centred first: ONE tilt, once per inspection (``stop_centring``), and the STOP is taken on
    # the next fresh frame, a whole box with a sound measurement; the Hanson toilet's twenty
    # actions of LOOK_UP / LOOK_DOWN after a fresh sighting are what the once-per-inspection bound
    # is for. Without LOOK actions the tilt-less rules above stand. And a fresh in-range box under
    # ``stop_confidence`` (the takeover's own start threshold by default: detect high, associate
    # low, STOP high -- a 0.30 re-sighting is not the evidence to end an episode on) does not STOP
    # on its own: the closing takes the one frame that keeps the box whole (a LOOK with the pair,
    # else a TURN) for at most ``weak_stop_max_holds`` consecutive frames, and STOPs if the detector
    # has not read it stronger by then, or at once when no such frame exists. The depth is never
    # NULL on this path by construction: a box without a coherent depth projection is never fresh.
    stop_centring: bool = True
    stop_confidence: float = 0.50
    weak_stop_max_holds: int = 2
    # THE STOP OVERSHOOT (since 2026-10-14). The terminal ranges are read short of the truth: the
    # near edge of a box with floor in it, and a centroid fused from the near surfaces, both stand
    # nearer than the object's footprint does, and the evaluator measures to the footprint.
    # Markleeville/000188 (``benchmark_toilet_10episodes_20261014T143259Z``) STOPped on a 0.84 m
    # near-edge reading 1.27 m from the toilet's footprint -- DTG 0.27, SR 0. (Reading a small
    # object's range from the fused memory instead was tried the same day and flipped CLOSE and
    # INSPECT on alternate frames; see ``_fresh_measured``.) So once the STOP is DECIDED, on
    # whichever evidence, the closing walks on toward the target and STOPs then -- no frame is
    # consulted, nothing is verified again, the decision stands: ``stop_overshoot_steps`` (one)
    # MOVE_FORWARD always, then more, up to ``stop_overshoot_max_steps`` in all, while the 3-D
    # memory still stands farther than ``stop_overshoot_within_m`` from the agent. The memory is
    # about the footprint's near face (000188: 1.25 m where the footprint's disc began at 1.27),
    # so a memory inside 0.90 m is a tenth inside the evaluator's radius; the near edge is not
    # trusted for this -- it is what decided the STOP, and in 000188 it undercut the memory by
    # 0.41 m where the eight STOPs that succeeded agreed with it within 0.14. Verified in turn
    # (2026-10-14 evening): two steps always, SR 0.90 but every other STOP 0.01-0.07 SPL dearer
    # (``…155701Z``); one step always, SR 0.80 with 000188 STOPped 3 cm outside the disc
    # (``…162832Z``); on those nine deciding frames this rule takes the second step twice.
    # A step is taken only while the memory would still stand ``stop_overshoot_min_range_m`` or
    # more beyond the agent after it, and the near edge the deciding frame measured (when it did)
    # at least the body radius; the agent must face within a quarter turn of the memory (farther
    # off the converter would turn first, and one over a turn off is not "a step forward"); a
    # step the converter reports blocked ends the overshoot at once, and
    # ``stop_overshoot_max_actions`` bounds the whole stage, turns included (the three turns a
    # quarter-turn error can take, plus the steps). ``stop_overshoot_max_steps`` 0 restores the
    # STOP on the deciding frame; 1 is one step and never a second.
    stop_overshoot_steps: int = 1
    stop_overshoot_max_steps: int = 2
    stop_overshoot_within_m: float = 0.90
    stop_overshoot_min_range_m: float = 0.45
    stop_overshoot_max_actions: int = 5
    # After a release, no new takeover may START on a far candidate for this
    # many actions: six twelve-step verifications from one spot three metres
    # from the couch (Ranchester attempt 4) were the same failure repeated,
    # and the search has to MOVE before it is worth two more frames. A close
    # view (within ``rejection_min_range_m``) is exempt, as always.
    release_cooldown_actions: int = 15
    # The terminal inspection can deadlock on a big object at close range: the
    # heading where the detector fires shows a box clipped at the image edge
    # whose centre says "turn", and after the turn a wall of upholstery shows
    # no box at all (Ranchester couch, actions 184-196: twelve such cycles).
    # When the inspection budget runs out but the LOCKED target was seen fresh
    # within the terminal range while the agent stood here, STOP rather than
    # end the episode. False restores the historical method error.
    stop_on_exhausted_inspection: bool = True
    # When the inspection budget runs out and the LOCKED target was NOT seen
    # fresh from here at all, the lock was wrong -- a sofa from four metres
    # that is nothing from one (Ranchester same-storey run, 2026-10-04: the
    # episode ended as an agent error with the real couch 2.85 m away). The
    # candidate is released like an unverified one, its anchor remembered with
    # twice the rejection radius, and the search resumes. False restores the
    # historical episode-ending error.
    release_on_failed_inspection: bool = True
    failed_inspection_radius_factor: float = 2.0
    # A LOCKED target with no safe path used to wait for A* for ever: the agent
    # that spawned beside a bed with the plant in view 4.4 m away (Hanson
    # 2026-10-05) spun 159 actions on "no safe target path" because the floor
    # under the camera's blind radius was unknown and unknown is impassable.
    # After ``footing_after_steps`` such actions the closing asks the camera
    # for a FOOTING sweep (one LOOK_DOWN, a circle, one LOOK_UP: the floor
    # around the feet becomes known, and a path either appears or is shown
    # not to exist from here); if ``release_after_footing_steps`` more
    # pathless actions follow, the lock is released WITHOUT a rejection -- the
    # target was never disproved, the spot was -- and the search moves on,
    # refusing a new takeover on a far candidate until the agent has left the
    # spot (``boxed_in_radius_m``). ``footing_sweep=False`` drops the sweep
    # alone (a protocol without LOOK actions never has it); the release after
    # the bound stands either way.
    footing_sweep: bool = True
    footing_after_steps: int = 2
    release_after_footing_steps: int = 6
    boxed_in_radius_m: float = 1.0
    # The sweep is asked for only where the floor around the feet is still unknown:
    # at least this share of the cells within ``footing_radius_m`` of the agent.
    footing_radius_m: float = 1.2
    footing_unknown_fraction: float = 0.25
    # An approach that goes nowhere -- the agent's position has not left a circle of
    # ``approach_stall_m`` in ``approach_stall_actions`` consecutive CLOSE actions --
    # releases the lock with a rejection. A* finds a path every action; the follower
    # pushes into something the map does not show (Leonardo 2026-10-05: 160 actions
    # of MOVE_FORWARD against an unseen obstacle 4.6 m from a real sofa, until the
    # closing bound raised an agent error). 0 disables the clock.
    approach_stall_actions: int = 20
    approach_stall_m: float = 0.20
    # THE 3-D MEMORY (``target_memory.TargetMemory3D``, since 2026-10-08). The target's
    # XYZ centroid is saved the moment a takeover starts and fused, precision-weighted,
    # with every frame that projects onto it: an observation from ``r`` metres enters
    # with sigma ``memory_noise_floor_m + memory_noise_per_m * r`` (depth noise and the
    # visible-part shift of a big object both grow with range), so the near frames of
    # the approach dominate and the estimate's standard error shrinks frame by frame.
    # Everything below reads that memory: the elevation the pitch follows, the range the
    # terminal test and the history STOP measure, the spot a support surface must sit at.
    memory_noise_floor_m: float = 0.03
    memory_noise_per_m: float = 0.03
    # EXTENDED APPROACH HISTORY (``approach_history.ApproachHistory``). A target the agent
    # tracked and walked toward over at least ``approach_history_m`` of path, closing at
    # least ``approach_history_closed_m`` of range, is known: inside the terminal range
    # a cropped box, a detector drop or a budget run down is not a reason to release it.
    # A fresh in-range sighting STOPs whatever its alignment, and after
    # ``history_stop_after_missing`` consecutive terminal frames without a fresh box the
    # closing completes on the 3-D memory and STOPs -- the one deliberate exception to
    # "no STOP from remembered range alone", earned by the path. 0 path disables it.
    approach_history_m: float = 3.0
    approach_history_closed_m: float = 1.0
    history_stop_after_missing: int = 3
    # SUDDEN CLOSE PROXIMITY (``target_backoff.BackoffManeuver``). A candidate first seen
    # within ``sudden_proximity_m`` with no approach history -- the agent turned and there
    # it was -- is not closed on from that single viewpoint. Before the lock the agent
    # backs ``backoff_steps`` forward steps into space it knows to be clear (its own
    # trail, the observed grid, the floor guard, A*, the NavMesh when bound), turns back to
    # face the remembered coordinates and re-verifies from there; the whole manoeuvre is
    # bounded by ``backoff_max_actions`` (a retreat straight back is six 30-degree turns,
    # three steps and six turns back). With no clear space known it is skipped and
    # recorded, and verification proceeds in place as before. ``backoff=False`` disables it.
    backoff: bool = True
    sudden_proximity_m: float = 1.30
    backoff_steps: int = 3
    backoff_max_actions: int = 20
    # A candidate the wider perspective does not confirm is DISPROVED -- unless the detector
    # read it at ``backoff_trust_confidence`` or better from where it was first seen. From
    # the retreat spot a toilet 1.5 m inside its doorway is behind the jamb (Collierville/000036,
    # 2026-10-13: 0.93-0.95 in two frames, backed off, never re-seen, released as disproved
    # with a 2x rejection radius, and the toilet landmark on the map 2 m away was never a node
    # again). Such a candidate is released WITHOUT a rejection: the place the agent stood was
    # not the place to verify it from, and the next frame of it re-takes it. A weaker box --
    # the table read as a bed at 0.80-0.90 from 0.9 m (Markleeville 2026-10-08) -- is disproved
    # as before. A candidate FIRST SEEN at that confidence is not backed off from at all
    # (since 2026-10-14): the manoeuvre and the verification that follows cost 23-25 actions
    # in both Collierville toilet episodes of the 2026-10-13 run and lost the target once;
    # a 0.92 box is the detector's word, and the lock is taken where it was seen.
    backoff_trust_confidence: float = 0.92
    # NOR IS A CANDIDATE THE DETECTOR HAD BEEN FIRING ON (since 2026-10-14). "Sudden" means the
    # agent turned and there it was; a toilet whose box stood in the frame on
    # ``backoff_persistence_frames`` of the ``backoff_persistence_window`` actions before the
    # takeover -- clipped by the frame's edge, so no takeover could start on those frames -- is
    # not sudden, it is persistent, and the manoeuvre that re-verifies it from farther back is
    # not owed (Collierville/000023, 2026-10-14: 0.95, 0.79, 0.79, 0.76 on four border-cut frames,
    # the takeover on the fifth at 0.84, a 25-action retreat to a spot the jamb hid it from, and a
    # 2 m rejection that refused eleven more frames of it at 0.86-0.91).
    backoff_persistence_frames: int = 2
    backoff_persistence_window: int = 6
    # SUPPORT-SURFACE RESILIENCE (``support_surface``). At close range a mounted or
    # supported object leaves the frame before its support does (a television on a
    # dresser becomes the dresser). A box of a ``support_surface_classes`` label whose
    # centroid lies under the memory's footprint and at or below its height is spatial
    # evidence that the tracked thing is still there: the inspection is not released as
    # "saw nothing", the map's vote for the surface is not a contradiction, and once the
    # inspection budget is spent with that evidence fresh and the memory inside the
    # terminal range, the closing STOPs. False restores the former releases.
    support_surface_resilience: bool = True
    support_surface_classes: tuple = SUPPORT_SURFACE_CLASSES

    def __post_init__(self):
        if isinstance(self.confidence, bool) or not math.isfinite(self.confidence) or not 0 < self.confidence <= 1:
            raise ValueError("Target confidence must be in (0, 1]")
        if (isinstance(self.track_confidence, bool) or not math.isfinite(self.track_confidence)
                or not 0 < self.track_confidence <= self.confidence):
            raise ValueError("track_confidence must be in (0, confidence]")
        if type(self.bearing_start) is not bool:
            raise ValueError("bearing_start must be a bool")
        for key in ("bearing_max_actions", "bearing_lost_frames"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("%s must be a positive int" % key)
        if isinstance(self.bearing_retry_m, bool) or not math.isfinite(self.bearing_retry_m) or self.bearing_retry_m < 0:
            raise ValueError("bearing_retry_m must be finite and non-negative")
        if isinstance(self.bearing_max_walk_m, bool) or not math.isfinite(self.bearing_max_walk_m) or self.bearing_max_walk_m <= 0:
            raise ValueError("bearing_max_walk_m must be positive and finite")
        if (isinstance(self.bearing_near_box_frac, bool) or not math.isfinite(self.bearing_near_box_frac)
                or not 0 < self.bearing_near_box_frac <= 1):
            raise ValueError("bearing_near_box_frac must lie in (0, 1]")
        for key in ("association_radius_m", "look_down_distance_m", "terminal_distance_m", "range_tolerance_m",
                    "refine_distance_m", "rejection_radius_m", "rejection_min_range_m", "failed_inspection_radius_factor",
                    "elevation_band_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        if isinstance(self.association_range_gain, bool) or not math.isfinite(self.association_range_gain) or self.association_range_gain < 0:
            raise ValueError("association_range_gain must be finite and non-negative")
        for key in ("association_depth_m", "floor_cut_min_hidden_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and non-negative" % key)
        for key in ("floor_cut_terminal", "blind_cone_stop"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("%s must be a bool" % key)
        if type(self.blind_cone_max_missing) is not int or self.blind_cone_max_missing < 1:
            raise ValueError("blind_cone_max_missing must be a positive int")
        if (isinstance(self.blind_cone_min_fraction, bool) or not math.isfinite(self.blind_cone_min_fraction)
                or not 0 < self.blind_cone_min_fraction < 1):
            raise ValueError("blind_cone_min_fraction must lie in (0, 1)")
        for key in ("bbox_pitch", "stop_centring"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("%s must be a bool" % key)
        if (isinstance(self.bbox_pitch_distance_m, bool) or not math.isfinite(self.bbox_pitch_distance_m)
                or self.bbox_pitch_distance_m <= 0):
            raise ValueError("bbox_pitch_distance_m must be positive and finite")
        if (isinstance(self.bbox_pitch_region, bool) or not math.isfinite(self.bbox_pitch_region)
                or not 0 < self.bbox_pitch_region < 0.5):
            raise ValueError("bbox_pitch_region must lie in (0, 0.5)")
        if (isinstance(self.pitch_hysteresis, bool) or not math.isfinite(self.pitch_hysteresis)
                or not 0 <= self.pitch_hysteresis < 0.5):
            raise ValueError("pitch_hysteresis must lie in [0, 0.5)")
        if (isinstance(self.stop_confidence, bool) or not math.isfinite(self.stop_confidence)
                or not self.track_confidence <= self.stop_confidence <= 1):
            raise ValueError("stop_confidence must lie in [track_confidence, 1]")
        if type(self.weak_stop_max_holds) is not int or self.weak_stop_max_holds < 0:
            raise ValueError("weak_stop_max_holds must be a non-negative int")
        if type(self.stop_overshoot_steps) is not int or self.stop_overshoot_steps < 0:
            raise ValueError("stop_overshoot_steps must be a non-negative int")
        if type(self.stop_overshoot_max_steps) is not int or self.stop_overshoot_max_steps < self.stop_overshoot_steps:
            raise ValueError("stop_overshoot_max_steps must be an int of at least stop_overshoot_steps")
        for key in ("stop_overshoot_within_m", "stop_overshoot_min_range_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        if type(self.stop_overshoot_max_actions) is not int or self.stop_overshoot_max_actions < self.stop_overshoot_max_steps:
            raise ValueError("stop_overshoot_max_actions must be an int of at least stop_overshoot_max_steps")
        if type(self.release_cooldown_actions) is not int or self.release_cooldown_actions < 0:
            raise ValueError("release_cooldown_actions must be a non-negative int")
        if self.terminal_distance_m + self.range_tolerance_m > self.look_down_distance_m:
            raise ValueError("Terminal inspection must be within the look-down distance")
        for key in ("confirmation_frames", "max_verify_steps", "max_reacquire_steps", "max_closing_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (2 if key == "confirmation_frames" else 1):
                raise ValueError("Invalid target-closing action/frame bound: %s" % key)
        if type(self.border_margin_px) is not int or self.border_margin_px < 0:
            raise ValueError("border_margin_px must be a non-negative int")
        if (isinstance(self.backoff_trust_confidence, bool) or not math.isfinite(self.backoff_trust_confidence)
                or not 0 < self.backoff_trust_confidence <= 1):
            raise ValueError("backoff_trust_confidence must lie in (0, 1]")
        for key in ("release_unverified", "stop_on_exhausted_inspection", "release_on_failed_inspection", "footing_sweep"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("%s must be a bool" % key)
        for key in ("footing_after_steps", "release_after_footing_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("%s must be a positive int" % key)
        if type(self.approach_stall_actions) is not int or self.approach_stall_actions < 0:
            raise ValueError("approach_stall_actions must be a non-negative int")
        if isinstance(self.approach_stall_m, bool) or not math.isfinite(self.approach_stall_m) or self.approach_stall_m <= 0:
            raise ValueError("approach_stall_m must be positive and finite")
        for key in ("boxed_in_radius_m", "footing_radius_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        if (isinstance(self.footing_unknown_fraction, bool) or not math.isfinite(self.footing_unknown_fraction)
                or not 0 <= self.footing_unknown_fraction <= 1):
            raise ValueError("footing_unknown_fraction must lie in [0, 1]")
        if (isinstance(self.override_confidence, bool) or not math.isfinite(self.override_confidence)
                or not self.confidence <= self.override_confidence <= 1):
            raise ValueError("override_confidence must lie in [confidence, 1]")
        for key in ("context_check", "verify_keep_in_frame"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("%s must be a bool" % key)
        if (isinstance(self.context_penalty, bool) or not math.isfinite(self.context_penalty)
                or not 0 < self.context_penalty <= 1):
            raise ValueError("context_penalty must lie in (0, 1]")
        if type(self.context_confirmation_frames) is not int or self.context_confirmation_frames < self.confirmation_frames:
            raise ValueError("context_confirmation_frames must be an int of at least confirmation_frames")
        if isinstance(self.context_baseline_m, bool) or not math.isfinite(self.context_baseline_m) or self.context_baseline_m < 0:
            raise ValueError("context_baseline_m must be finite and non-negative")
        for key in ("memory_noise_floor_m", "sudden_proximity_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        for key in ("memory_noise_per_m", "approach_history_m", "approach_history_closed_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and non-negative" % key)
        for key in ("history_stop_after_missing", "backoff_steps", "backoff_max_actions"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("%s must be a positive int" % key)
        for key in ("backoff", "support_surface_resilience", "through_unknown", "closest_approach_stop"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("%s must be a bool" % key)
        for key in ("max_verify_approach_steps", "inspection_max_steps", "verify_sweep_max_actions"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError("%s must be a non-negative int" % key)
        for key in ("backoff_persistence_frames", "backoff_persistence_window"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("%s must be a positive int" % key)
        if (isinstance(self.through_unknown_cost, bool) or not math.isfinite(self.through_unknown_cost)
                or self.through_unknown_cost < 1.0):
            raise ValueError("through_unknown_cost must be finite and at least 1 (the cost of open floor)")
        if (not isinstance(self.support_surface_classes, (tuple, list)) or not self.support_surface_classes
                or not all(isinstance(name, str) and name.strip() for name in self.support_surface_classes)):
            raise ValueError("support_surface_classes must be a non-empty sequence of labels")


class TargetClosing:
    """Own detection verification, path, yaw and pitch until STOP or episode error.

    Only consecutive, coherent 3D observations of the same object count. Alias
    duplicates were removed by PerceptionCycle. No evaluator goal/DTG is read.
    The first confident candidate WITH a depth projection suspends exploration
    before any lock; an unverified candidate is released back to exploration
    (``release_unverified``), and so is a locked one whose terminal inspection
    never saw it from close (``release_on_failed_inspection``); a locked target
    seen fresh at terminal range is STOPped on, never given back.

    A frame shows a PART of the object, and the part changes as the frame's
    edges cut it (since 2026-10-11, the toilets of the 1,000-episode run):
    a re-sighting is associated in a cylinder along the line of sight
    (:meth:`_associated`); the terminal range of a box the lower edge cut
    reads the 3-D memory beside the visible near edge (:meth:`_floor_cut`);
    and where the protocol has no LOOK_DOWN, a target the geometry puts
    under the frame is STOPped on from the memory it was seen fresh at a
    few actions before (:meth:`_below_the_frame`) -- the tilt-less
    counterpart of the dynamic pitch controller (:meth:`_pitch`).

    Where the protocol HAS the LOOK pair, the pitch within
    ``bbox_pitch_distance_m`` is the image-space servo's (:meth:`_servo`,
    ``core/control/camera/box_pitch_servo``): the fresh box's place in the frame -- lower
    region or about to be cut below, LOOK_DOWN; upper region or cut above,
    LOOK_UP; centre band, hold -- with the geometry as the fallback when no
    box is fresh. The terminal STOP is then taken on a whole box (one
    centring tilt per inspection) at or above ``stop_confidence``.
    """

    def __init__(self, policy):
        self.policy, self.settings = policy, policy.settings.target_closing
        self.path = TargetApproachPath(policy)
        self.rejected = []                 # (anchor xyz, floor id, action, radius m, why) per released candidate
        self.releases = 0
        self.inspection_releases = 0
        self.inspection_resumptions = 0    # inspections ended to step closer: seen fresh, measured out of range
        self.stall_releases = 0            # locks released because the approach went nowhere
        self.map_releases = 0              # locks the map outvoted after the fact
        self.last_release = None           # why the last candidate was released
        self.released_step = None          # the action of the last release; the cooldown counts from it
        self.border_rejections = 0
        self.map_rejections = 0
        self.weak_resightings = 0          # frames that counted on ``track_confidence`` alone
        self.footing_sweeps = 0            # footing sweeps asked for by a lock with no safe path
        self.boxed_releases = 0            # locks released because no path existed from where the agent stood
        self.boxed_in = []                 # (robot xy, floor id, action) per boxed-in release
        self.overrides = 0                 # takeovers started past the memory by ``override_confidence``
        self.context_rejections = 0        # confident boxes the context penalty kept under the start threshold
        self.suspect_locks_held = 0        # frames a suspect candidate had the frames but not the viewpoints
        self.same_spot_rejections = 0      # near candidates refused because they were released from this very spot
        self.blind_frames = 0              # frames the detector was in back-off: no evidence, clocks paused
        self.backoffs = 0                  # backing manoeuvres made for a candidate seen suddenly at close range
        self.backoff_skips = 0             # ... and those skipped for want of clear space behind the agent
        self.backoff_repeats = 0           # close candidates on an anchor already backed off from: verified in place
        self.backed_off = []               # (anchor xyz, floor id) per manoeuvre made
        self.wider_releases = 0            # candidates the wider perspective did not confirm
        self.trusted_releases = 0          # ... of which, read at backoff_trust_confidence or better: released without a rejection
        self.repeat_releases = 0           # second unverified releases of one anchor, escalated to non-overridable
        self.disproved_refusals = 0        # legacy-pursuit landmarks refused on a release that disproved them from close
        self.no_step_holds = 0             # verify steps withheld because the step would push the box out of the frame
        self.support_sightings = 0         # frames a support-surface box stood consistent with the 3-D memory
        self.support_holds = 0             # inspection releases withheld on that evidence
        self.support_stops = 0             # STOPs on consistent support-surface tracking at terminal range
        self.history_holds = 0             # releases withheld inside the terminal range by the approach history
        self.history_stops = 0             # STOPs the approach history completed without a fresh aligned box
        self.depth_associations = 0        # re-sightings the along-ray tolerance kept that a 0.50 m sphere would have dropped
        self.floor_cut_frames = 0          # fresh frames the lower edge cut: sightings that did not move the 3-D memory
        self.floor_cut_measurements = 0    # terminal ranges read from the 3-D memory because the frame's lower edge cut the box
        self.blind_cone_stops = 0          # STOPs on a target predicted below the frame, in a protocol without LOOK
        self.bbox_pitch_corrections = 0    # tilts the image-space servo asked for on the approach and in verification
        self.stop_centring_tilts = 0       # terminal STOPs deferred one tilt so they are taken on a whole box
        self.weak_stop_holds = 0           # terminal frames held because the fresh box was under stop_confidence
        self.resighting_exemptions = 0     # re-sightings of the active candidate the rejection memory would have dropped
        self.glimpse_steps = 0             # verification steps toward a box in view without a usable depth
        self.verify_pauses = 0             # verification actions the clock did not count: steps toward a visible candidate
        self.backoff_trusted_skips = 0     # manoeuvres skipped because the first frame read at backoff_trust_confidence
        self.backoff_persistence_skips = 0  # ... or because the class had been in the frame on the actions before
        self.strong_box_steps = []         # the last actions a box of the class at or above ``confidence`` was in the frame
        self.inspection_steps = 0          # inspection frames answered with a step: the fresh range a step beyond the limit
        self.dead_band_alignments = 0      # inspection frames read as aligned because a turn back would undo the last turn
        self.through_unknown_routes = 0    # standoff routes planned through unknown cells after the known floor had none
        self.widened_standoffs = 0         # standoffs found beyond the standoff ring, after no point of it had a route
        self.stale_standoffs_kept = 0      # standoffs a refinement made stale, kept because no ring point had a route
        self.closest_approach_stops = 0    # STOPs from a widened standoff with the fresh range a step outside the limit
        self.occluder_reads = 0            # candidate frames whose box was read around a surface standing in front of the target
        self.sweep_releases = 0            # verifications released after a fruitless sweep cycle, before the budget
        self.overshoot_commands = 0        # MOVE_FORWARDs asked for between a decided STOP and the STOP (``stop_overshoot_steps``)
        self.bearing_starts = 0            # bearing approaches begun on an RGB box without a usable depth
        self.bearing_resolved = 0          # ... of which a frame with depth turned into a verification
        self.bearing_abandoned = 0         # ... and those given up: lost, out of actions, or not moving
        self.bearing_refusals = 0          # depthless boxes refused a bearing approach: cooldown, boxed in, or a spot already tried
        self.bearing_failures = []         # (robot xy, floor id, action) per bearing approach given up
        self._depthless = []               # this frame's strong, whole boxes of the class with no usable depth
        self._in_frame = []                # this frame's whole boxes of the class at the threshold in force
        self._clear()

    def _clear(self):
        """Per-candidate state; rejection memory and counters survive a release."""
        self.active = self.locked = False
        self.phase = "SEARCH"
        self.xyz = self.anchor = self.box = None
        self.label = None
        self.count = 0
        self.best_conf = 0.0               # the strongest fresh reading of this candidate
        self.started = self.last_seen = self.processed = -1
        self.last_sighting = -1            # the last action the candidate was in view: measured (``last_seen``) or glimpsed
        self.floor_id = None
        self.failure = None
        self.path = TargetApproachPath(self.policy)
        self.observed_xyz = None
        self.observed_near_m = None        # near-edge range of the fresh observation, when measured
        self.inspection_started = None
        self.inspection_index = 0
        self.inspection_sighting = None    # the last action of the inspection that saw the target in terminal range
        self.occluded_path_steps = 0
        self.verification_yaw = 0.0
        self.no_path_steps = 0             # consecutive actions the LOCKED target had no safe path
        self.footing_done = False          # a footing sweep was asked for by this lock
        self.pathless_after_footing = 0    # pathless actions since the sweep ended
        self.close_trail = []              # (x, y) per consecutive CLOSE action, for the stall clock
        self.suspect = None                # why the context check doubts this candidate (``target_context``)
        self.sightings = []                # (x, y) of the agent per counted consecutive frame
        self.memory = None                 # TargetMemory3D: the fused XYZ centroid; ``xyz`` mirrors it
        self.history = None                # ApproachHistory: the path walked since the takeover began
        self.backoff = None                # BackoffManeuver for a candidate seen suddenly at close range
        self.support_seen = -1             # last step a support-surface box stood consistent with the memory
        self.support_xyz = None
        self.terminal_missing = 0          # consecutive terminal-range frames without a fresh target box
        self.visible_floor_m = None        # the lowest height of the fresh frame's supported points
        self.lowest_m = None               # the lowest height the target's supported points reached over this lock
        self.top_m = None                  # ... and the highest: the object's measured top, for the below-the-frame test
        self.footprint_m = None            # the anchor frame's footprint half-extent (``perception.footprint_radius_m``)
        self._floor_cut_step = -1          # the last action whose terminal range the memory supplied (counted once)
        self.conf = None                   # the fresh box's detector confidence
        self.box_drift_rows = None         # the fresh box centre's row change since the previous consecutive frame (+ down)
        self.last_fix = None               # the image-space servo's last verdict (``box_pitch_servo.VerticalFix``)
        self.centring_done = False         # this inspection has spent its one centring tilt
        self.centring_step = -1            # ... and the action it was spent on (the next frame STOPs even without a box)
        self.weak_holds = 0                # consecutive terminal frames held for a box under stop_confidence
        self.glimpse_box = None            # a box of the target's class in view WITHOUT a usable depth, on the anchor's bearing
        self.glimpse_step = -1             # ... and the action it was seen on
        self.verify_paused = 0             # verification actions this candidate's clock did not count (bounded)
        self.inspection_stepped = 0        # steps this inspection took for a fresh range one step beyond the limit
        self.inspection_step_pending = False   # the last command was such a step: the hooks say whether it moved
        self.inspection_blocked = False    # ... and one of them did not: the floor ends here
        self.inspection_turn = None        # (step, sign) of the inspection's last centring turn on a fresh box
        self.sweep_actions = 0             # consecutive verification actions spent sweeping for a candidate out of view
        self.overshoot = None              # the decided STOP's forward steps in progress (``_terminal``), else None
        self.bearing = None                # the BEARING approach in force (an RGB box without depth): its box, yaw, counts

    def _clipped(self, box, intrinsics):
        return clipped_box(box, intrinsics, self.settings.border_margin_px)

    #: Releases whose spot a box at ``override_confidence`` may start a takeover inside of.
    OVERRIDABLE_RELEASES = ("unverified", "contradicted by the map")
    #: A candidate backed off from and never re-seen from the wider perspective: not overridable, and refused
    #: from twice ``boxed_in_radius_m`` of the spot it was released at (the agent must really go elsewhere).
    WIDER_RELEASE = "unverified from a wider perspective"
    #: A confident candidate the wider perspective did not confirm: the retreat spot is blamed, not the candidate.
    TRUSTED_RELEASE = "unverified from a wider perspective; trusted, not disproved"
    #: The second ``unverified`` release of the same anchor: the same evidence is not new evidence (Markleeville
    #: 2026-10-08, 35 takeovers of one table in 450 actions, every one past the ``unverified`` memory at 0.80).
    REPEAT_RELEASE = "unverified again"
    #: Releases that disproved the candidate from close: the legacy pursuit does not walk after them at any range.
    DISPROVED_RELEASES = (WIDER_RELEASE, REPEAT_RELEASE, "inspection saw nothing", "inspection saw nothing in any view")

    def _rejected_nearby(self, xyz, floor_id, override=False, here=None):
        """Whether a released candidate's anchor covers ``xyz`` on ``floor_id``.

        With ``here`` (the agent's position) only the releases made from
        within ``boxed_in_radius_m`` of it count: the test for a NEAR
        candidate, which is new evidence from a new spot but the same
        evidence from the spot it was already released at (the review of
        2026-10-07 reproduced twelve takeovers of one chair from one spot).
        """
        for entry in self.rejected:
            spot, floor, _, radius = entry[:4]
            why = entry[4] if len(entry) > 4 else "unverified"     # a 4-tuple is a legacy (unverified) entry
            if floor != floor_id or math.dist(spot[:2], xyz[:2]) > radius:
                continue
            if here is not None:
                stood = entry[5] if len(entry) > 5 else None
                reach = self.settings.boxed_in_radius_m * (2.0 if why in (self.WIDER_RELEASE, self.REPEAT_RELEASE) else 1.0)
                if stood is None or math.dist(stood, here) > reach:
                    continue
            if override and why in self.OVERRIDABLE_RELEASES:
                continue
            return True
        return False

    def rejected_near(self, xy, floor_id):
        """Whether a released candidate's anchor lies within ``rejection_radius_m`` of ``xy`` on ``floor_id``.

        For the search loop: a confirmed landmark the takeover already
        verified and rejected at is not worth another look.
        """
        return self._rejected_nearby((float(xy[0]), float(xy[1]), 0.0), floor_id)

    def landmark_refused(self, xy, floor_id):
        """Whether a confirmed landmark of the target's class at ``xy`` is NOT worth offering as a node.

        Only a release that DISPROVED its candidate from close
        (``DISPROVED_RELEASES``) refuses the landmark: the closing looked
        from near and saw nothing. A plain ``unverified`` release -- two
        consecutive frames not had from far -- does not: the landmark map
        fused the object from several frames, which is more evidence than
        a far verification that failed, and the node's peek brings the
        agent to within ``landmark_standoff_m`` of it, where a close view is
        new evidence whatever the memory holds (since 2026-10-14; until then
        every release within range hid the seen toilet from the HUD and
        the order alike).
        """
        floor = self.policy.mapping.floor_id
        if floor_id != floor:
            return False
        return self._disproved_near(xy)

    def _candidates(self, obs):
        """``(kept, projected, glimpses)``: confident in-frame boxes, those with a coherent 3-D position, and the depthless boxes on the candidate's bearing.

        Before a lock the box must also clear the image border and lie away
        from every spot already released as unverified on this floor --
        unless it reaches ``override_confidence`` (the unverified and
        map-contradicted memory, the cooldown and a boxed-in spot are
        bypassed; a failed inspection's spot is not). A box the context
        check doubts (``target_context.suspect``) counts
        ``context_penalty`` of its confidence toward the start threshold.
        While a candidate is active, a box of the target's class down to
        ``track_confidence`` is projected too, and counts if -- and only if
        -- it lands on the candidate's anchor: a re-sighting, not a new
        object -- and the memory, the cooldown and the boxed-in spot, which
        judge whether a takeover may START, do not drop it (since 2026-10-14).
        A box of the class WITHOUT a usable depth that lies on the active
        candidate's bearing is a ``glimpse``: in view, not placeable from
        here (:meth:`_on_anchor_bearing`).
        """
        p, s = self.policy, self.settings
        k = obs.camera.intrinsics
        tracking = self.active and (self.anchor is not None or self.bearing is not None)
        threshold = s.track_confidence if tracking else s.confidence
        boxes = [d for d in p.perception.detections if p.target.accepts(d.cls)
                 and d.conf >= threshold
                 and 0 <= d.xyxy[0] < d.xyxy[2] <= k.width
                 and 0 <= d.xyxy[1] < d.xyxy[3] <= k.height]
        if any(d.conf >= s.confidence for d in boxes) and (not self.strong_box_steps or self.strong_box_steps[-1] != obs.step):
            self.strong_box_steps.append(int(obs.step))          # clipped or not: the class was in the frame
            del self.strong_box_steps[:-2 * s.backoff_persistence_window]
        if not self.locked:
            kept = [d for d in boxes if not self._clipped(d.xyxy, k)]
            self.border_rejections += sum(1 for d in boxes if d.conf >= s.confidence) - sum(
                1 for d in kept if d.conf >= s.confidence)
            boxes = kept
        projected, kept, glimpses = [], [], []
        self._depthless = []
        self._in_frame = list(boxes)                                   # whole boxes of the class at the threshold
        for detection in boxes:
            rows = []
            strong = detection.conf >= s.confidence
            values = list(observed_objects(obs, (detection,), threshold, rows, masks=p.perception.masks))
            if not values:
                # No usable depth. A strong box is kept for the record -- and may begin a BEARING
                # approach (``bearing_start``); one that lies on the active candidate's bearing is a
                # GLIMPSE of it -- the verification faces and approaches it (:meth:`_scan`), since
                # the depth that failed from here is had from closer.
                if tracking and not self.locked and self._on_anchor_bearing(obs, detection.xyxy):
                    glimpses.append(detection)
                if strong:
                    kept.append(detection)
                    self._depthless.append(detection)
                continue
            label, xyz = values[0]
            range_m = math.dist((obs.pose.x, obs.pose.y), xyz[:2])
            radius = s.association_radius_m + s.association_range_gain * max(0.0, range_m - 2.0)
            associated = (self.anchor is not None and p.mapping.floor_id == self.floor_id
                          and self._associated(obs, xyz, radius))
            if not strong and not associated:
                continue                                       # a weak box is only ever a re-sighting
            override = s.override_confidence < 1.0 and detection.conf >= s.override_confidence
            if not self.locked and associated:
                # A re-sighting of the ACTIVE candidate: the rejection memory, the cooldown and
                # the boxed-in spot judge whether a takeover may START, not whether the one in
                # force may count its next frame (since 2026-10-14; Markleeville/000188, actions
                # 34-45: a takeover begun on a 0.84 override past an "unverified" memory had every
                # 0.60-0.79 re-sighting of the same toilet dropped by that memory, and the
                # consecutive count never reached two).
                if range_m > s.rejection_min_range_m and self._rejected_nearby(xyz, p.mapping.floor_id, override=override):
                    self.resighting_exemptions += 1
            elif not self.locked and range_m > s.rejection_min_range_m:
                cooling = (not self.active and self.released_step is not None
                           and obs.step - self.released_step < s.release_cooldown_actions)
                if self._rejected_nearby(xyz, p.mapping.floor_id, override=override):
                    continue
                if cooling and not override:
                    continue
                if self._boxed_in_here(obs):
                    continue                                   # no path from HERE: the spot, not the box, is the problem
                if override and not associated and not self.active and (cooling or self._rejected_nearby(xyz, p.mapping.floor_id)):
                    self.overrides += 1
            elif not self.locked and not self.active:
                # A close view is new evidence -- from a NEW spot. From the spot the same
                # candidate was already released at, or given up at for want of a path, it
                # is the same evidence, and the takeover/release cycle it bought was unbounded
                # (twelve takeovers of one chair in eighty actions, review of 2026-10-07).
                here = (obs.pose.x, obs.pose.y)
                if self._rejected_nearby(xyz, p.mapping.floor_id, override=override, here=here) or self._boxed_in_here(obs):
                    self.same_spot_rejections += 1
                    continue
            if not self.locked and contradicted_by_map(p, label, xyz, rows[-1].get("radius_m")) is not None:
                self.map_rejections += 1                       # a confirmed non-target object stands here
                continue
            doubt = self._suspect(obs, label, xyz) if (s.context_check and not self.locked) else None
            if doubt is not None and not associated and detection.conf * s.context_penalty < s.confidence:
                # Confident enough on its own, not where it stands: no takeover on this box.
                self.context_rejections += 1
                kept.append(detection)
                continue
            kept.append(detection)
            if not obs.pose.z - 0.3 <= xyz[2] <= obs.pose.z + 2.2:
                continue
            if self.anchor is not None and not associated:
                continue
            if not strong:
                self.weak_resightings += 1
            if rows[-1].get("occluder_depth_m") is not None:
                self.occluder_reads += 1                       # the target's box read around a surface in front of it
            projected.append((detection, tuple(float(v) for v in xyz), rows[-1].get("range_near_m"), doubt, rows[-1]))
        return kept, projected, glimpses

    def _associated(self, obs, xyz, radius):
        """Whether a projected centroid is a re-sighting of the anchor.

        A cylinder about the line of sight from the agent to the anchor:
        within ``radius`` across the ray and in height, within ``radius``
        plus the object's depth along it -- ``association_depth_m`` or the
        anchor frame's footprint diameter, whichever is larger. The part of
        an object a frame shows moves along the ray as the frame's edges cut
        it (a toilet's bowl from 2 m, its tank from 1.3 m); across the ray
        it moves no further than the object is wide. ``depth_associations``
        counts the re-sightings a sphere of ``radius`` would have dropped.
        """
        ax, ay, az = (float(v) for v in self.anchor)
        dx, dy, dz = float(xyz[0]) - ax, float(xyz[1]) - ay, float(xyz[2]) - az
        rx, ry = ax - float(obs.pose.x), ay - float(obs.pose.y)
        length = math.hypot(rx, ry)
        if length < 1e-6:
            return math.dist(self.anchor, xyz) <= radius
        along = (dx * rx + dy * ry) / length
        across = abs(dx * ry - dy * rx) / length
        depth = max(self.settings.association_depth_m, 2.0 * float(self.footprint_m or 0.0))
        inside = math.hypot(across, dz) <= radius and abs(along) <= radius + depth
        if inside and math.dist(self.anchor, xyz) > radius:
            self.depth_associations += 1
        return inside

    def _on_anchor_bearing(self, obs, box):
        """Whether ``box`` (a frame's ``xyxy``) lies within one turn of the bearing to the active candidate's anchor.

        The test for a box of the target's class WITHOUT a usable depth
        projection: it cannot be associated in 3-D, but on the anchor's
        bearing it is the candidate in view -- a toilet at 5 m, beyond the
        depth the projection accepts -- and the verification faces and
        approaches it (:meth:`_scan`, ``glimpse``) rather than sweeping as
        if nothing were there.
        """
        if self.anchor is None or self.policy.mapping.floor_id != self.floor_id:
            return False
        turn = self.policy.episode.action_spec.turn_angle_rad
        bearing = math.atan2(self.anchor[1] - obs.pose.y, self.anchor[0] - obs.pose.x)
        seen = normalize_angle(obs.pose.yaw + self._bbox_error(obs, box))
        return abs(normalize_angle(seen - bearing)) <= turn + 1e-6

    def _suspect(self, obs, label, xyz):
        """Why the context check doubts a box of ``label`` at ``xyz`` (``target_context.suspect``), or None."""
        p = self.policy
        info, objects = {}, ()
        graph = getattr(p, "graph", None)
        world = getattr(p, "last_world", None)
        if graph is not None and world is not None and getattr(graph, "labels", None) is not None:
            pid = graph.room_near(world, (float(xyz[0]), float(xyz[1])), 0.6)
            if pid is not None:
                info = graph.label_info(pid) or {}
                objects = graph.objects_in(pid)
        return suspect(p.target, label, xyz, obs.pose.z, label=info.get("label"),
                       strength=info.get("strength"), objects=objects)

    def blind(self):
        """Whether this frame carried no detector evidence (the service failed or is in back-off)."""
        evidence = getattr(getattr(self.policy, "perception", None), "detector_evidence", None) or {}
        return bool(evidence.get("skipped") or evidence.get("failed"))

    def observe(self, obs):
        if obs.step == self.processed:
            return
        self.processed = obs.step
        self.inspection_step_pending = False      # ``note_blocked`` for the last step has spoken before this frame
        if self.overshoot is not None:
            # The STOP is decided: the frames between it and the STOP are not evidence
            # (``_terminal``). Nothing is associated, released or re-verified on them.
            return
        p, s = self.policy, self.settings
        if self.active and self.blind():
            # No evidence either way: the consecutive-frame chain is carried across the
            # frame, nothing is counted and no clock advances (``plan`` pauses them). Before
            # 2026-10-07 a detector back-off read as "target not seen": a correct lock was
            # released as "the inspection saw nothing" with a 2 m rejection around it.
            self.blind_frames += 1
            if self.last_seen == obs.step - 1:
                self.last_seen = obs.step
            if self.last_sighting == obs.step - 1:
                self.last_sighting = obs.step
            return
        boxes, projected, glimpses = self._candidates(obs)
        if self.bearing is not None:
            if projected:
                # The step closer gave the box its depth: the verification begins HERE, on this
                # frame, with a clock of its own (the bearing steps were not verification actions).
                self.bearing_resolved += 1
                self.bearing = None
                self.active = False
            else:
                self._track_bearing(obs)
                return
        # Only a candidate with a coherent 3-D projection starts a takeover: a confident
        # box without depth support has nothing the next frame can be checked against,
        # and the twelve-step verifications it bought (Ranchester attempts 4 and 5:
        # "waiting for valid target depth" from 3.8 m) released every time. Since
        # 2026-10-21 such a box begins a BEARING approach instead (``bearing_start``): the
        # agent walks toward it along its image-space bearing until a frame has the depth.
        if not self.active and projected:
            self.active, self.started, self.phase = True, obs.step, "VERIFY"
            self.verification_yaw = obs.pose.yaw
            p.route_memory.clear("target_takeover")
            p._route = p._goal = None
            p.peek.cancel(obs, "target_takeover", restore=False)
            p.camera_control.inspection = None
            glances = getattr(p, "glances", None)
            if glances is not None and glances.active is not None:
                # A cue glance toward the box is often what un-clipped it; left in
                # force it resumed, stale, after the release (turns toward a yaw the
                # route no longer has).
                glances.abort(obs, "target_takeover")
        elif not self.active and self._depthless and s.bearing_start:
            self._start_bearing(obs)
            return
        if not self.active:
            return
        here = (float(obs.pose.x), float(obs.pose.y))
        if projected:
            detection, xyz, near, doubt, row = max(projected, key=lambda item: item[0].conf)
            # Consecutive means no frame WITHOUT the candidate in between: a glimpse -- the box in
            # view on the anchor's bearing without a usable depth -- carries the chain (``last_sighting``).
            chained = obs.step == self.last_sighting + 1
            self.count = self.count + 1 if chained else 1
            self.sightings = (self.sightings if chained else []) + [here]
            self.last_sighting = obs.step
            self.observed_xyz, self.observed_near_m = xyz, (None if near is None else float(near))
            heights = row.get("height_range_m")
            self.visible_floor_m = None if not heights else float(heights[0])
            cut = self._floor_cut_box(obs, detection.xyxy, self.visible_floor_m)
            if heights:
                self.lowest_m = float(heights[0]) if self.lowest_m is None else min(self.lowest_m, float(heights[0]))
                self.top_m = float(heights[1]) if self.top_m is None else max(self.top_m, float(heights[1]))
            range_m = math.dist(here, xyz[:2])
            if self.anchor is None:
                # The 3-D memory and the approach ledger start with the takeover; a candidate
                # that appears at arm's length with no history is re-verified from farther back.
                self.anchor, self.floor_id = xyz, p.mapping.floor_id
                self.footprint_m = None if row.get("radius_m") is None else float(row["radius_m"])
                self.memory = TargetMemory3D(xyz, range_m, obs.step, s.memory_noise_floor_m, s.memory_noise_per_m)
                self.history = ApproachHistory(here, range_m, obs.step)
                if s.backoff and range_m < s.sudden_proximity_m:
                    before = sum(1 for step in self.strong_box_steps
                                 if obs.step - s.backoff_persistence_window <= step < obs.step)
                    if detection.conf >= s.backoff_trust_confidence:
                        # The detector's word at this level is not re-verified from farther back:
                        # the retreat cost 23-25 actions per candidate and once hid the toilet
                        # behind its door jamb (the Collierville episodes of 2026-10-13).
                        self.backoff_trusted_skips += 1
                    elif before >= s.backoff_persistence_frames:
                        # Not sudden: the class had been in the frame on the actions before (border-cut
                        # boxes that could not start the takeover). Verified where it stands.
                        self.backoff_persistence_skips += 1
                    elif self._backed_off_near(xyz):
                        self.backoff_repeats += 1          # one manoeuvre per object: verify this one in place
                    else:
                        self.backoff = BackoffManeuver(p, s.backoff_steps, s.backoff_max_actions)
                        self.backed_off.append((xyz, p.mapping.floor_id))
            elif cut:
                # The frame shows the UPPER part of the object (the lower edge cut the rest): its
                # centroid is that part's, a biased reading of the object's, and fusing it walked
                # the memory -- and the standoff measured from it -- back toward the tank frame by
                # frame. A sighting, not a position (since 2026-10-11).
                self.floor_cut_frames += 1
            else:
                self.memory.update(xyz, range_m, obs.step)
            self.xyz = self.memory.xyz
            if self.box is not None and obs.step == self.last_seen + 1:
                # The box's vertical motion between consecutive fresh frames (+ down): recorded for the
                # diagnostics; the servo predicts the next frame from the geometry rather than extrapolating this.
                self.box_drift_rows = 0.5 * (detection.xyxy[1] + detection.xyxy[3]) - 0.5 * (self.box[1] + self.box[3])
            else:
                self.box_drift_rows = None
            self.box, self.label, self.last_seen = detection.xyxy, detection.cls, obs.step
            self.conf = float(detection.conf)
            self.best_conf = max(self.best_conf, float(detection.conf))
            if doubt is not None and not self.locked:
                self.suspect = doubt
            if self.backoff is None or not (self.backoff.state == "pending" or self.backoff.active):
                self.locked |= self._confirmed()
            p._target_xy, p._target_step, p._target_floor_id = self.xyz[:2], obs.step, self.floor_id
        else:
            glimpsed = False
            if not self.locked:
                # A box of the class on the anchor's bearing, without a usable depth: the candidate
                # is in view and cannot be placed from here. Remembered for the verification's step;
                # the consecutive chain is carried across the frame, not broken by it.
                if glimpses:
                    best = max(glimpses, key=lambda d: d.conf)
                    self.glimpse_box, self.glimpse_step = best.xyxy, obs.step
                    glimpsed = True
                if glimpsed and obs.step == self.last_sighting + 1:
                    self.last_sighting = obs.step
                else:
                    self.count = 0
                    self.sightings = []
            self._support_resighting(obs)
        if self.history is not None and self.xyz is not None:
            self.history.record(here, math.dist(here, self.xyz[:2]), obs.step)
        if self.backoff is not None and (self.backoff.state == "pending" or self.backoff.active):
            self.started += 1                  # the manoeuvre's actions are not verification actions: the clock waits
        swept_out = (s.verify_sweep_max_actions > 0 and self.sweep_actions >= s.verify_sweep_max_actions
                     and self.last_sighting < obs.step)
        if swept_out:
            self.sweep_releases += 1
        if (not self.locked and s.release_unverified
                and (obs.step - self.started >= s.max_verify_steps or swept_out)):
            manoeuvre = self.backoff
            if (manoeuvre is not None and manoeuvre.goal is not None and manoeuvre.finished is not None
                    and self.last_seen < manoeuvre.finished):
                # Backed off, turned to face it, swept -- and the detector never fired again: the
                # close-range box the wider perspective does not confirm is disproved, like a
                # failed inspection (Markleeville 2026-10-08: eight manoeuvres on one table read
                # as a bed from 0.9 m, released as "unverified" and re-taken at 0.80 each time) --
                # unless it was read at ``backoff_trust_confidence`` or better where it was first
                # seen: then the retreat spot, not the candidate, is what failed (the toilet
                # behind its door jamb), and it is given back without a rejection to be re-taken.
                self.wider_releases += 1
                if self.best_conf >= s.backoff_trust_confidence:
                    self.trusted_releases += 1
                    self._release(obs, why=self.TRUSTED_RELEASE, remember=False)
                else:
                    self._release(obs, radius_factor=s.failed_inspection_radius_factor, why=self.WIDER_RELEASE)
            elif self.anchor is not None and self._released_unverified_near(self.anchor):
                # The same candidate was already given back as unverified once: a third look from
                # the same evidence is not owed (the override is for far flickers, not for this).
                self.repeat_releases += 1
                self._release(obs, radius_factor=s.failed_inspection_radius_factor, why=self.REPEAT_RELEASE)
            else:
                self._release(obs)
            return
        if self.locked and s.release_on_failed_inspection and self.anchor is not None:
            landmark = contradicted_by_map(p, self.label or "", self.anchor)
            if landmark is not None and self._is_support(landmark.class_name):
                landmark = None                # the surface the target stands on: the same spot, not another object
            if landmark is not None:
                # The map outvoted the lock: the object at the anchor has been confirmed as
                # something else since (the sofa from four metres is the bed from two).
                self.map_releases += 1
                self._release(obs, radius_factor=s.failed_inspection_radius_factor, why="contradicted by the map")

    # -- the BEARING approach: an RGB detection without depth, walked toward until a frame has the depth ----------
    def _start_bearing(self, obs):
        """Begin a bearing approach on this frame's strongest depthless box, or refuse it (counted).

        Refused during the release cooldown, from a boxed-in spot, within
        ``bearing_retry_m`` of a spot a bearing approach was already given
        up from on this storey, and unless the class was in the frame on
        the previous action too (``strong_box_steps``): one flicker of a
        far false positive is not walked after.
        """
        p, s = self.policy, self.settings
        here = (float(obs.pose.x), float(obs.pose.y))
        best = max(self._depthless, key=lambda d: d.conf)
        cooling = self.released_step is not None and obs.step - self.released_step < s.release_cooldown_actions
        tried = any(floor == p.mapping.floor_id and math.dist(here, spot) <= s.bearing_retry_m
                    for spot, floor, _ in self.bearing_failures)
        persistent = (int(obs.step) - 1) in self.strong_box_steps
        if cooling or tried or not persistent or self._boxed_in_here(obs):
            self.bearing_refusals += 1
            return
        error = self._bbox_error(obs, best.xyxy)
        self.active, self.started, self.phase = True, int(obs.step), "BEARING"
        self.verification_yaw = float(obs.pose.yaw)
        self.floor_id = p.mapping.floor_id
        self.label = best.cls
        self.bearing = dict(started=int(obs.step), actions=0, last_seen=int(obs.step), box=tuple(best.xyxy),
                            conf=float(best.conf), yaw=float(normalize_angle(obs.pose.yaw + error)), lost=0,
                            moves=[], spot=here)
        self.bearing_starts += 1
        p.route_memory.clear("target_bearing")
        p._route = p._goal = None
        p.peek.cancel(obs, "target_bearing", restore=False)
        p.camera_control.inspection = None
        glances = getattr(p, "glances", None)
        if glances is not None and glances.active is not None:
            glances.abort(obs, "target_bearing")

    def _track_bearing(self, obs):
        """Keep the bearing approach on this frame's box of the class, or give it up: lost, out of actions, not moving."""
        b, s, p = self.bearing, self.settings, self.policy
        if p.mapping.floor_id != self.floor_id:
            return self._end_bearing(obs, "storey changed")
        if self._in_frame:
            best = max(self._in_frame, key=lambda d: d.conf)
            b.update(last_seen=int(obs.step), box=tuple(best.xyxy), conf=float(best.conf), lost=0,
                     yaw=float(normalize_angle(obs.pose.yaw + self._bbox_error(obs, best.xyxy))))
        else:
            b["lost"] += 1
            if b["lost"] >= s.bearing_lost_frames:
                return self._end_bearing(obs, "lost: %d frames without the class in view" % b["lost"])
        if b["actions"] >= s.bearing_max_actions:
            return self._end_bearing(obs, "out of actions (%d)" % b["actions"])
        here = (float(obs.pose.x), float(obs.pose.y))
        walked = math.dist(b["spot"], here)
        if walked >= s.bearing_max_walk_m:
            return self._end_bearing(obs, "walked %.1f m toward it without the depth" % walked)
        k = obs.camera.intrinsics
        if b["last_seen"] == obs.step and (b["box"][3] - b["box"][1]) >= s.bearing_near_box_frac * float(k.height):
            return self._end_bearing(obs, "near and unmeasurable: the box fills %.0f%% of the frame without depth"
                                     % (100.0 * (b["box"][3] - b["box"][1]) / float(k.height)))
        moves = b["moves"]
        step = p.episode.action_spec.forward_step_m
        if len(moves) >= 3 and math.dist(moves[-3], here) < 0.5 * step:
            return self._end_bearing(obs, "not moving: three steps toward it went nowhere")

    def _end_bearing(self, obs, why):
        """Give the bearing approach up: no rejection memory (nothing was placed), the spot remembered, the search resumes."""
        b = self.bearing
        self.bearing_abandoned += 1
        self.bearing_failures.append((tuple(b["spot"]), self.policy.mapping.floor_id, int(obs.step)))
        self.last_release = "bearing approach given up: %s" % why
        self._clear()

    def _bearing_command(self, obs, world):
        """Face the depthless box by its image-space bearing, then step toward it; the search's clocks do not run.

        The frame's own box gives the bearing when the class is in view;
        otherwise the yaw it was last seen at. Faced within one turn, the
        agent takes one MOVE_FORWARD (a two-step waypoint, as the
        verification's step does), and the next frame is nearer -- the
        depth that failed from here is had from closer.
        """
        b, p = self.bearing, self.policy
        actions = p.episode.action_spec
        turn = actions.turn_angle_rad
        b["actions"] += 1
        here = (float(obs.pose.x), float(obs.pose.y))
        seen = b["last_seen"] == obs.step
        error = self._bbox_error(obs, b["box"]) if seen else normalize_angle(b["yaw"] - obs.pose.yaw)
        info = {"kind": "target_closing", "phase": "BEARING", "target_visible": seen, "target_measured": False,
                "reason": "RGB detection of the target without depth: approaching along its bearing",
                "bearing_action": b["actions"], "bearing_conf": round(float(b["conf"]), 2)}
        if abs(error) > turn + 1e-6:
            return NavigationCommand.hold(final_yaw=normalize_angle(obs.pose.yaw + error),
                                          info=dict(info, bearing_step="face the detection"))
        b["moves"].append(here)
        reach = 2.0 * actions.forward_step_m
        ahead = (obs.pose.x + reach * math.cos(obs.pose.yaw), obs.pose.y + reach * math.sin(obs.pose.yaw))
        return NavigationCommand.follow([here, ahead], info=dict(info, bearing_step="toward the detection"))

    def _confirmed(self):
        """Whether the consecutive frames so far lock the candidate.

        ``confirmation_frames`` frames as before; a candidate the context
        check doubts (``suspect``) needs ``context_confirmation_frames`` of
        them, taken from at least two spots ``context_baseline_m`` apart --
        a turn in place is not a second viewpoint, and the Allensville
        counter was locked on two frames from one spot.
        """
        s = self.settings
        if self.suspect is None or not s.context_check:
            return self.count >= s.confirmation_frames
        if self.count < s.context_confirmation_frames:
            return False
        spots = self.sightings[-self.count:]
        baseline = max((math.dist(a, b) for a in spots for b in spots), default=0.0)
        if baseline + 1e-9 < s.context_baseline_m:
            self.suspect_locks_held += 1
            return False
        return True

    def _is_support(self, label):
        return self.settings.support_surface_resilience and is_support_surface(label, self.settings.support_surface_classes)

    def _support_resighting(self, obs):
        """Record a support-surface box standing where the memory holds the target, on a frame without the target's own class.

        The television on the dresser becomes the dresser from one metre: a
        box of a supporting class, projected to a coherent centroid under
        the memory's footprint and at or below its height, is spatial
        evidence that the tracked thing is still there (``support_surface``).
        It never counts toward the lock or the terminal test on its own.
        """
        p, s = self.policy, self.settings
        if not s.support_surface_resilience or not self.active or self.memory is None:
            return
        k = obs.camera.intrinsics
        boxes = [d for d in p.perception.detections if self._is_support(d.cls) and d.conf >= s.track_confidence
                 and 0 <= d.xyxy[0] < d.xyxy[2] <= k.width and 0 <= d.xyxy[1] < d.xyxy[3] <= k.height]
        if not boxes:
            return
        here = (obs.pose.x, obs.pose.y)
        for label, xyz in observed_objects(obs, boxes, s.track_confidence, masks=p.perception.masks):
            range_m = math.dist(here, self.xyz[:2])
            radius = s.association_radius_m + s.association_range_gain * max(0.0, range_m - 2.0)
            if supports(self.xyz, xyz, radius):
                self.support_sightings += 1
                self.support_seen, self.support_xyz = obs.step, tuple(float(v) for v in xyz)
                return

    def _disproved_near(self, xy):
        """Whether a release that disproved its candidate from close covers ``xy`` on this floor."""
        floor = self.policy.mapping.floor_id
        for entry in self.rejected:
            why = entry[4] if len(entry) > 4 else "unverified"
            if entry[1] == floor and why in self.DISPROVED_RELEASES and math.dist(entry[0][:2], (float(xy[0]), float(xy[1]))) <= entry[3]:
                return True
        return False

    def _released_unverified_near(self, xyz):
        """Whether an anchor within the association radius of ``xyz`` on this floor was already released as unverified."""
        s = self.settings
        floor = self.policy.mapping.floor_id
        return any(entry[1] == floor and math.dist(entry[0][:2], xyz[:2]) <= s.association_radius_m
                   and (entry[4] if len(entry) > 4 else "unverified") == "unverified" for entry in self.rejected)

    def _backed_off_near(self, xyz):
        """Whether a manoeuvre was already made for a candidate anchored within the association radius of ``xyz``."""
        s = self.settings
        floor = self.policy.mapping.floor_id
        return any(f == floor and math.dist(anchor[:2], xyz[:2]) <= s.association_radius_m for anchor, f in self.backed_off)

    def _history_qualifies(self):
        """Whether this candidate was tracked and approached over an EXTENDED path (``approach_history_m``)."""
        s = self.settings
        return (self.history is not None and s.approach_history_m > 0
                and self.history.qualifies(s.approach_history_m, s.approach_history_closed_m))

    def _boxed_in_here(self, obs):
        """Whether the agent still stands where a lock was released for want of a path (``boxed_in_radius_m``)."""
        here = (obs.pose.x, obs.pose.y)
        floor = self.policy.mapping.floor_id
        return any(f == floor and math.dist(xy, here) <= self.settings.boxed_in_radius_m for xy, f, _ in self.boxed_in)

    def _release(self, obs, radius_factor=1.0, why="unverified", remember=True):
        """Give a candidate back to exploration and remember the spot (``radius_factor`` times the rejection radius).

        ``remember=False`` leaves the anchor out of the rejection memory: the
        candidate was not disproved, the place the agent stood was.
        """
        p = self.policy
        if self.anchor is not None and remember:
            self.rejected.append((self.anchor, self.floor_id, obs.step,
                                  float(radius_factor) * self.settings.rejection_radius_m, str(why),
                                  (float(obs.pose.x), float(obs.pose.y))))
        self.releases += 1
        self.released_step = int(obs.step)
        self.last_release = why
        p._target_xy = p._target_id = p._target_floor_id = None
        p._route = p._goal = None
        p.route_memory.clear("target_released")
        self._clear()
        self.processed = obs.step
        self.phase = "RELEASED"

    def fail(self, reason):
        self.phase, self.failure = "FAILED", reason
        # Expected method failure, not a converter/harness invariant violation:
        # the evaluator records it as failure, with any forced STOP distinguished.
        raise RuntimeError("Target closing failed (exploration remains locked): " + reason)

    def _elevation(self, obs, distance=None):
        """Where the target stands against the camera: ``+1`` above, ``-1`` below, ``0`` level.

        From the measured 3-D centroid when there is one (its height against
        the camera's, within ``elevation_band_m`` is level); else from the box
        centre's bearing above the horizon -- the optical ray's elevation
        corrected by the camera's own pitch -- read at ``distance`` when
        known, otherwise against half a tilt step. The one input of the
        dynamic pitch controller: LOOK_DOWN below, LOOK_UP above, never a
        fixed direction (a hardcoded look-down drove a wall-mounted
        television out of the frame).
        """
        s = self.settings
        camera_z = obs.pose.z + obs.camera.height_m
        if self.xyz is not None:
            dz = float(self.xyz[2]) - camera_z
            return 0 if abs(dz) <= s.elevation_band_m else (1 if dz > 0 else -1)
        if self.box is None:
            return 0
        k = obs.camera.intrinsics
        v = (self.box[1] + self.box[3]) / 2
        angle = math.atan2(k.cy - v, k.fy) - float(obs.pose.camera_pitch)      # elevation above the horizon
        if distance is not None and math.isfinite(distance) and distance > 0:
            dz = math.tan(angle) * distance
            return 0 if abs(dz) <= s.elevation_band_m else (1 if dz > 0 else -1)
        band = 0.5 * float(self.policy.episode.action_spec.tilt_angle_rad or 0.0)
        return 0 if abs(angle) <= band else (1 if angle > 0 else -1)

    def _pitch(self, obs, distance):
        actions = self.policy.episode.action_spec
        if not actions.has_camera_tilt:
            return None
        # The pitch is the target's measured height's, never its label's: "a potted
        # plant is low" sent the camera 30 degrees down at a plant standing in a metre-
        # tall planter, where its foliage never projected, for 24 actions (Hanson
        # 2026-10-05); a plant on the floor is low by this geometry anyway. Positive
        # is down: a target below the camera gives a positive angle, one above it a
        # negative one -- LOOK_UP. Inside ``look_down_distance_m`` the elevation
        # commits the camera to at least one tilt in its direction, so the terminal
        # frames hold a low toilet or a high television whole.
        camera_z = obs.pose.z + obs.camera.height_m
        desired = math.atan2(camera_z - self.xyz[2], max(distance, 1e-6))
        if distance < self.settings.look_down_distance_m:
            elevation = self._elevation(obs, distance)
            if elevation < 0:
                desired = max(actions.tilt_angle_rad, desired)
            elif elevation > 0:
                desired = min(-actions.tilt_angle_rad, desired)
        desired = self._quantise_pitch(obs, desired)
        if actions.min_pitch_rad is not None:
            desired = max(actions.min_pitch_rad, desired)
        if actions.max_pitch_rad is not None:
            desired = min(actions.max_pitch_rad, desired)
        return float(desired)

    def _quantise_pitch(self, obs, desired):
        """``desired`` on the protocol's tilt steps -- the step the camera is at while ``desired`` lies within the hysteresis band of it.

        Plain rounding chatters: a wanted angle at half a step rounds up on
        one frame and down on the next as the memory moves by millimetres,
        and every change of pitch is an action that is not a step toward
        the target (Darden/000034, 2026-10-14: nine LOOK_DOWN / LOOK_UP
        actions 2.19 m from a locked toilet, the wanted angle 15.0 degrees
        against a 30-degree step). The present step is kept while the wanted
        angle is within ``(0.5 + pitch_hysteresis)`` steps of it; a change
        has to be wanted by that much more.
        """
        tilt = float(self.policy.episode.action_spec.tilt_angle_rad)
        current = round(float(obs.pose.camera_pitch) / tilt) * tilt
        if abs(desired - current) <= (0.5 + self.settings.pitch_hysteresis) * tilt:
            return current
        return round(desired / tilt) * tilt

    def _servo(self, obs, advance_m=0.0):
        """The image-space servo's verdict on the fresh box, or None when it has no say.

        None without the LOOK pair, with ``bbox_pitch`` off, or on a frame
        without a fresh box (the geometry decides then). ``advance_m`` is
        the next action's step toward the target, so the verdict is read on
        the frame that step will produce (:func:`bbox_pitch.vertical_fix`);
        the range it is predicted at is the visible near edge of the fresh
        surface, else the centroid's planar range.
        """
        s, actions = self.settings, self.policy.episode.action_spec
        if not s.bbox_pitch or not actions.has_camera_tilt or self.last_seen != obs.step or self.box is None:
            return None
        k = obs.camera.intrinsics
        range_m = self.observed_near_m
        if range_m is None and self.observed_xyz is not None:
            range_m = math.dist((obs.pose.x, obs.pose.y), self.observed_xyz[:2])
        fix = vertical_fix(self.box, k.width, k.height, k.cy, k.fy, float(obs.pose.camera_pitch), actions.tilt_angle_rad,
                           s.bbox_pitch_region, s.border_margin_px, actions.min_pitch_rad, actions.max_pitch_rad,
                           range_m=range_m, advance_m=advance_m)
        self.last_fix = fix
        return fix

    def _servo_pitch(self, obs, fix):
        """The camera pitch the servo's verdict asks for, from the pitch the frame was taken at."""
        return float(obs.pose.camera_pitch) + fix.steps * self.policy.episode.action_spec.tilt_angle_rad

    def _approach_pitch(self, obs, distance):
        """The pitch for a step toward the target: where the target will be AFTER the step.

        The proactive half of the dynamic pitch controller (since 2026-10-08):
        a low target that is in the lower half of the frame now leaves it
        through the bottom edge on the next step if the camera stays level
        (Wiconisco toilet, action 430: bottom-cut box at 1.32 m, one
        MOVE_FORWARD, lost; four times, then released). Within
        ``bbox_pitch_distance_m``, with a fresh box, the verdict is the
        image-space servo's (since 2026-10-11): the box predicted after the
        step, in the lower region or cut below -- LOOK_DOWN; upper or cut
        above -- LOOK_UP; centred -- the camera stays where it is, whatever
        the geometry would have forced. Otherwise the pitch is :meth:`_pitch`
        at the range one step closer -- the elevation the memory predicts
        there, quantised to the tilt step, a full tilt at least inside the
        close band -- for every approach step; a level target stays level.
        None without LOOK actions.
        """
        actions = self.policy.episode.action_spec
        if not actions.has_camera_tilt or self.xyz is None:
            return None
        if distance < self.settings.bbox_pitch_distance_m:
            fix = self._servo(obs, advance_m=actions.forward_step_m)
            if fix is not None:
                if fix.wanted:
                    self.bbox_pitch_corrections += 1
                    return self._servo_pitch(obs, fix)
                return float(obs.pose.camera_pitch)
        ahead = max(0.3, distance - actions.forward_step_m)
        return self._pitch(obs, ahead)

    def _step_keeps_box(self, obs, distance):
        """Whether a step toward the target leaves its box inside the frame, with the camera as it is.

        For a protocol without LOOK actions (the published Gibson one): a box
        already cut by the bottom edge, or whose centre the memory's geometry
        puts below the frame after the step, is lost by stepping -- the
        verification frame is taken from here instead (``no_step_holds``).
        """
        if self.box is None or self.xyz is None:
            return True
        k = obs.camera.intrinsics
        margin = self.settings.border_margin_px
        x1, y1, x2, y2 = self.box
        if y2 > k.height - margin or y1 < margin:
            return False
        step = self.policy.episode.action_spec.forward_step_m
        ahead = max(0.3, distance - step)
        camera_z = obs.pose.z + obs.camera.height_m
        angle = math.atan2(camera_z - float(self.xyz[2]), ahead) - float(obs.pose.camera_pitch)   # below the axis
        predicted_v = k.cy + k.fy * math.tan(angle)
        half = 0.5 * (y2 - y1) * distance / ahead                                                   # the box grows
        return margin <= predicted_v - half and predicted_v + half <= k.height - margin

    def _scan(self, obs, pitch, reason):
        """Face the candidate while it is in view -- and step towards it -- or sweep when it is not.

        The lock needs two CONSECUTIVE frames of the same object. The former
        pattern stepped through (0, +30, 0, -30 deg) around the bearing on
        every action, so a candidate three metres away sat at the frame's
        edge on every other frame, was refused as clipped, and the count
        reset -- six twelve-step verifications from one spot in Ranchester
        attempt 4. A fresh box is faced (its own centre, as the inspection
        does); once faced, and farther than the terminal range, the agent
        takes one step towards it -- the next frame shows it larger and its
        depth more coherent, and a verification that cannot be had from here
        is had from closer. The sweep is for a candidate that has dropped
        out of view.
        """
        self.phase = "REACQUIRE" if self.locked else "VERIFY"
        actions = self.policy.episode.action_spec
        offset = actions.turn_angle_rad
        bearing = (math.atan2(self.xyz[1] - obs.pose.y, self.xyz[0] - obs.pose.x)
                   if self.xyz is not None else self.verification_yaw)
        fresh = self.last_seen == obs.step and self.box is not None
        if fresh or self.glimpse_step == obs.step:
            self.sweep_actions = 0
        if fresh:
            self.inspection_index = 0
            error = self._bbox_error(obs)
            info = {"kind": "target_closing", "reason": reason, "target_visible": True}
            if self.suspect is not None:
                info["suspect"] = self.suspect
            distance = math.dist((obs.pose.x, obs.pose.y), self.xyz[:2]) if self.xyz is not None else float("inf")
            aligned = abs(error) <= offset + 1e-6
            # A suspect candidate needs two viewpoints ``context_baseline_m`` apart, and a
            # turn in place is not one: let it step inside the terminal range (never onto
            # the object) rather than hold at 1.2 m where it could never lock (review of
            # 2026-10-07: 52 frames held, five releases, no move).
            nearest = self.settings.terminal_distance_m if self.suspect is None else 0.6 * self.settings.terminal_distance_m
            if (not self.locked and aligned
                    and distance > nearest + actions.forward_step_m
                    and (actions.has_camera_tilt or self._step_keeps_box(obs, distance))):
                # Two steps ahead: a one-step waypoint sits inside the converter's arrival
                # tolerance and yields no action at all. One MOVE_FORWARD results either way.
                # Within one turn of the centre the box stays in the frame after a step,
                # and a step is what makes the next frame a consecutive one; the centring
                # turn moved the box across the image and the detector dropped it on the
                # other side -- four times in a row at a chair 3.5 m off (Hanson 2026-10-05).
                # Never for a LOCKED target without a safe path: that one waits for A*.
                # Without LOOK actions, never a step that pushes a low box out through the
                # bottom edge (``_step_keeps_box``); with them, the step carries the pitch
                # the target will need after it (``_approach_pitch``).
                reach = 2.0 * actions.forward_step_m
                ahead = (obs.pose.x + reach * math.cos(obs.pose.yaw), obs.pose.y + reach * math.sin(obs.pose.yaw))
                if pitch is None:
                    pitch = self._approach_pitch(obs, distance)
                return NavigationCommand.follow([(obs.pose.x, obs.pose.y), ahead], camera_pitch=pitch,
                                                info=dict(info, verify_step="towards the candidate",
                                                          elevation=self._elevation(obs, distance)))
            if (not self.locked and aligned and not actions.has_camera_tilt
                    and distance > nearest + actions.forward_step_m):
                self.no_step_holds += 1
                info["verify_step_withheld"] = "a step would push the box out of the frame"
            if self.settings.verify_keep_in_frame and turn_action(error, offset) is None:
                # The hold below would be satisfied as it stands -- idle -- and an idle
                # result is executed as a turn. Choose the frame instead. For a LOCKED target
                # without a path too: the idle turn and the centring turn back alternated
                # for the whole pathless wait (233 reversals in one Darden episode).
                view = self._verification_view(obs, distance)
                if view is not None:
                    return view
            return NavigationCommand.hold(camera_pitch=pitch, final_yaw=normalize_angle(obs.pose.yaw + error), info=info)
        if not self.locked and self.glimpse_step == obs.step and self.glimpse_box is not None:
            # In view without a usable depth (a toilet at 5 m, beyond what the projection
            # accepts): the candidate is placed from closer, not swept for. Face the box, and
            # once faced step toward it; the clock does not count the step (since 2026-10-14;
            # Markleeville/000188 turned in place at such a box for two whole verifications).
            error = self._bbox_error(obs, self.glimpse_box)
            info = {"kind": "target_closing", "reason": reason, "target_visible": True, "target_measured": False}
            distance = math.dist((obs.pose.x, obs.pose.y), self.xyz[:2]) if self.xyz is not None else float("inf")
            if abs(error) <= offset + 1e-6 and distance > self.settings.terminal_distance_m + actions.forward_step_m:
                reach = 2.0 * actions.forward_step_m
                ahead = (obs.pose.x + reach * math.cos(obs.pose.yaw), obs.pose.y + reach * math.sin(obs.pose.yaw))
                self.glimpse_steps += 1
                self._pause_verify_clock()
                return NavigationCommand.follow([(obs.pose.x, obs.pose.y), ahead], camera_pitch=pitch,
                                                info=dict(info, verify_step="towards the glimpsed candidate"))
            return NavigationCommand.hold(camera_pitch=pitch, final_yaw=normalize_angle(obs.pose.yaw + error),
                                          info=dict(info, verify_step="face the glimpsed candidate"))
        angles = (0, 1, 0, -1)
        yaw = normalize_angle(bearing + angles[self.inspection_index % 4] * offset)
        if turn_action(normalize_angle(yaw - obs.pose.yaw), offset) is None:
            self.inspection_index += 1
            yaw = normalize_angle(bearing + angles[self.inspection_index % 4] * offset)
        if not self.locked:
            self.sweep_actions += 1
        return NavigationCommand.hold(camera_pitch=pitch,
                                      final_yaw=yaw,
                                      info={"kind": "target_closing", "reason": reason, "target_visible": False,
                                            "sweep_action": self.sweep_actions})

    def _pause_verify_clock(self):
        """One verification action spent stepping toward a visible candidate: the clock does not count it (bounded)."""
        if self.verify_paused < self.settings.max_verify_approach_steps:
            self.verify_paused += 1
            self.verify_pauses += 1
            self.started += 1

    def _backoff_command(self, obs, world, distance):
        """The backing manoeuvre's action for a candidate seen suddenly at close range, or None.

        Begun on the first action after the takeover (the retreat spot needs
        the world), it owns the action until the agent stands back and faces
        the remembered target; then the consecutive-frame count is reset, so
        the lock needs fresh frames from the wider perspective. A manoeuvre
        with no clear space is skipped and recorded; verification proceeds in
        place as before.
        """
        manoeuvre = self.backoff
        if manoeuvre is None or manoeuvre.state == "done":
            return None
        p, s = self.policy, self.settings
        if manoeuvre.state == "pending":
            if manoeuvre.begin(obs, world, self.xyz):
                self.backoffs += 1
            else:
                self.backoff_skips += 1
                return None
        pitch = None
        if p.episode.action_spec.has_camera_tilt:
            pitch = self._pitch(obs, distance) if distance <= s.look_down_distance_m else 0.0
        command = manoeuvre.command(obs, world, self.xyz, pitch)
        if command is not None:
            self.phase = "BACK_OFF"
            return replace(command, info=dict(command.info, backoff=manoeuvre.diagnostics(),
                                              target_visible=self.last_seen == obs.step))
        self.count, self.sightings = 0, []          # re-verify from here: fresh consecutive frames
        return None

    def _verification_view(self, obs, distance):
        """The next frame for a fresh, aligned candidate too close to step toward: one that keeps its box whole.

        A satisfied hold is idle, and the headless agent's idle action is a
        turn, which moved the centred Allensville toilet out of the frame
        at 1.18 m. **The pitch follows the target's elevation**
        (:meth:`_elevation`): a target below the camera is looked DOWN at,
        one above it -- a wall-mounted television, the top of a wardrobe --
        is looked UP at; a box cut by the bottom or the top edge says the
        same thing; never a fixed direction. Level with the camera, the LOOK
        whose predicted shift (``fy * tan(tilt)`` pixels) leaves the box
        inside the frame margins, down before up (the floor is nearer than
        the ceiling); failing both, the TURN whose shift (``fx * tan(turn)``)
        keeps it, toward the box's side before away. None when no view keeps
        it.
        """
        actions, s = self.policy.episode.action_spec, self.settings
        k = obs.camera.intrinsics
        x1, y1, x2, y2 = self.box
        margin = s.border_margin_px
        info = {"kind": "target_closing", "reason": "verification frame that keeps the box whole",
                "target_visible": True}
        if self.suspect is not None:
            info["suspect"] = self.suspect
        pitch_now = float(obs.pose.camera_pitch)
        tilt = float(actions.tilt_angle_rad) if actions.has_camera_tilt else None
        top_cut, bottom_cut = y1 < margin, y2 > k.height - margin

        def pitch_ok(value):
            if actions.min_pitch_rad is not None and value < actions.min_pitch_rad - 1e-9:
                return False
            return actions.max_pitch_rad is None or value <= actions.max_pitch_rad + 1e-9

        def fits(du, dv):
            return (x1 + du >= margin and x2 + du <= k.width - margin
                    and y1 + dv >= margin and y2 + dv <= k.height - margin)

        def look(sign, step):
            """One tilt step, ``+1`` down / ``-1`` up: when the limits allow it and the box stays in (or is cut on that side)."""
            value = pitch_now + sign * tilt
            dv = -sign * k.fy * math.tan(tilt)                 # LOOK_DOWN moves the image content UP
            cut = bottom_cut if sign > 0 else top_cut
            if pitch_ok(value) and (cut or fits(0.0, dv)):
                return NavigationCommand.hold(camera_pitch=value, final_yaw=obs.pose.yaw,
                                              info=dict(info, verify_step=step, elevation=elevation))
            return None

        elevation = self._elevation(obs, distance) if tilt is not None else 0
        if tilt is not None:
            fix = self._servo(obs)
            if fix is not None and fix.wanted and pitch_ok(self._servo_pitch(obs, fix)):
                # The box's own place in the frame says where to look (since 2026-10-11): below the band
                # or cut by the lower edge, down; above or cut by the top, up -- the same words the
                # elevation used, now read off the frame rather than the memory.
                self.bbox_pitch_corrections += 1
                step = "look down at a low target" if fix.steps > 0 else "look up at a high target"
                return NavigationCommand.hold(camera_pitch=self._servo_pitch(obs, fix), final_yaw=obs.pose.yaw,
                                              info=dict(info, verify_step=step, elevation=elevation,
                                                        bbox_region=fix.region, servo=fix.reason))
            geometry_decides = fix is None or fix.region != REGION_CENTRE
            if geometry_decides and (elevation < 0 or (elevation == 0 and bottom_cut)):
                view = look(+1, "look down at a low target")
                if view is not None:
                    return view
            elif geometry_decides and (elevation > 0 or (elevation == 0 and top_cut)):
                view = look(-1, "look up at a high target")
                if view is not None:
                    return view
            for sign, step in ((+1, "look down"), (-1, "look up")):
                view = look(sign, step)
                if view is not None:
                    return view
        du = k.fx * math.tan(actions.turn_angle_rad)
        toward_left = (x1 + x2) / 2 < k.cx                  # the box is left of centre: a LEFT turn moves it right
        # A turn moves a low or high box vertically too: a point below the camera at bearing
        # b sits at v - cy = fy tan(e) / cos(b), so turning it off-axis stretches the offset
        # by cos(b) / cos(b') -- the Wiconisco toilet's bottom-cut box left through the floor
        # on a "turn that keeps it in frame" (2026-10-08).
        bearing = math.atan2(((x1 + x2) / 2 - k.cx), k.fx)

        def rows_after(turn_left):
            after = bearing + (actions.turn_angle_rad if turn_left else -actions.turn_angle_rad)
            scale = math.cos(bearing) / max(math.cos(after), 1e-6)
            return (y1 - k.cy) * scale, (y2 - k.cy) * scale
        for turn_left in ((True, False) if toward_left else (False, True)):
            top, bottom = rows_after(turn_left)
            if fits(du if turn_left else -du, 0.0) and k.cy + top >= margin and k.cy + bottom <= k.height - margin:
                yaw = normalize_angle(obs.pose.yaw + (actions.turn_angle_rad if turn_left else -actions.turn_angle_rad))
                return NavigationCommand.hold(camera_pitch=None if tilt is None else pitch_now, final_yaw=yaw,
                                              info=dict(info, verify_step="turn %s" % ("left" if turn_left else "right")))
        return None

    def plan(self, obs, world):
        p, s = self.policy, self.settings
        if self.failure:
            self.fail(self.failure)
        if self.overshoot is not None:
            return self._overshoot_step(obs)
        if self.blind():
            return self._blind_frame(obs, world)
        if self.bearing is not None:
            return self._bearing_command(obs, world)
        if obs.step - self.started >= s.max_closing_steps:
            self.fail("closing action bound reached")
        if not self.locked and not s.release_unverified and obs.step - self.started >= s.max_verify_steps:
            self.fail("candidate never received consecutive depth-consistent confirmations")
        if self.xyz is None:
            return self._scan(obs, None, "waiting for valid target depth")
        if p.mapping.floor_id != self.floor_id:
            self.fail("target floor changed")
        distance = math.dist((obs.pose.x, obs.pose.y), self.xyz[:2])
        if not self.locked:
            command = self._backoff_command(obs, world, distance)
            if command is not None:
                return command
            return self._scan(obs, None, "consecutive depth-consistent frames required")
        command = self._continue_footing(obs, world, distance)
        if command is not None:
            return command
        measured = self._fresh_measured(obs)
        limit = s.terminal_distance_m + s.range_tolerance_m
        out_of_range = measured is not None and measured > limit

        def nearly():
            # An ARRIVED standoff found only beyond the standoff ring (``target_path.widened``: no closer point
            # had a route) with the fresh range within one forward step of the limit: as close as the planner
            # gets. The inspection's step rule, then the closest-approach STOP, own it (since 2026-10-15).
            return (s.closest_approach_stop and self.path.arrived and self.path.widened and measured is not None
                    and measured <= limit + p.episode.action_spec.forward_step_m)

        if (self.inspection_started is not None and out_of_range and not nearly()
                and (self.inspection_sighting is None or self._centred(obs))):
            # Seen fresh from here but the measured surface lies beyond the terminal range:
            # the filtered estimate entered the inspection early (an older, nearer reading
            # averaged in), or an earlier borderline reading registered a sighting the
            # centred view does not confirm. Standing still would spend the budget on a
            # target in plain view and release it as "saw nothing" -- or STOP on it from
            # too far when the budget runs out; step closer.
            self.inspection_started = None
            self.inspection_resumptions += 1
            if self.path.arrived:
                # The inspection was entered from an arrived standoff: walking its route again
                # would arrive again. Give it up, and the next selection is a closer ring.
                self.path.drop()
        # A fresh sighting whose NEAR EDGE is already inside the terminal radius is terminal
        # evidence wherever the centroid is (since 2026-10-08): the benchmark measures to the
        # object, and a low object without LOOK actions is still in the frame from here where
        # it may not be from the standoff (the Wiconisco toilet at 0.65 m is below the FOV).
        in_range = measured is not None and measured <= limit
        if self.inspection_started is not None or in_range or (distance <= s.terminal_distance_m and not out_of_range):
            return self._inspect_or_release(obs, distance)
        # During transit A* owns heading. A bbox outside the FOV must not replace
        # a collision-qualified path with a centering hold or a reacquisition spin.
        command = self.path.command(self, obs, world)
        if command is None:
            return self._no_path(obs, world, distance)
        if self.path.arrived and (not out_of_range or nearly()):
            # The route to the standoff is walked (since 2026-10-11): the executor would report
            # ``arrived`` and the headless agent would spend the idle result on a TURN -- twenty of
            # them at a kept standoff 1.15 m from the memory, inside the success region, until the
            # stall clock released the lock as "went nowhere" (Collierville/000194, /000043). The
            # standoff is where the inspection happens: begin it here.
            self.no_path_steps = 0
            self.pathless_after_footing = 0
            return self._inspect_or_release(obs, distance)
        self.no_path_steps = 0
        self.pathless_after_footing = 0
        if self._approach_stalled(obs):
            self.stall_releases += 1
            self._release(obs, why="approach stalled: no progress in %d actions" % s.approach_stall_actions)
            turn = p.episode.action_spec.turn_angle_rad
            return NavigationCommand.hold(
                final_yaw=normalize_angle(obs.pose.yaw + turn),
                info={"kind": "target_released", "phase": "RELEASED",
                      "reason": "the approach from %.1f m went nowhere for %d actions; the search resumes"
                      % (distance, s.approach_stall_actions)})
        visible = self.last_seen == obs.step
        self.phase = "CLOSE" if visible else "CLOSE_OCCLUDED"
        self.occluded_path_steps += int(not visible)
        # Level during transit; inside ``look_down_distance_m`` the pitch follows the
        # target's elevation (``_pitch``: down at a toilet, up at a wall-mounted
        # television) so the terminal frames hold it whole -- the close-range policy,
        # in both directions (since 2026-10-07; before, the approach forced level and
        # the inspection alone tilted).
        pitch = self._approach_pitch(obs, distance)
        return replace(command, camera_pitch=pitch,
                       info=dict(command.info, target_confirmed=True, persistent_lock=True,
                                 target_visible=visible, range_m=distance, phase=self.phase))

    def _inspect_or_release(self, obs, distance):
        """The inspection's action -- or, when it released the lock, the one turn that hands the action back."""
        command = self._inspect(obs, distance)
        if command is None:
            # The lock was wrong: released, the spot remembered. This action is one
            # turn in place -- the first frame of the resumed search -- and the next
            # action is the search's own; the closing never explores.
            turn = self.policy.episode.action_spec.turn_angle_rad
            return NavigationCommand.hold(
                final_yaw=normalize_angle(obs.pose.yaw + turn),
                info={"kind": "target_released", "phase": "RELEASED",
                      "reason": "the inspection from %.1f m saw nothing; the search resumes" % distance})
        return command

    def _blind_frame(self, obs, world):
        """The action for a frame without detector evidence: the clocks stand still, the approach may go on.

        Every bound that counts frames moves one action later; a locked
        target with a path is walked toward (A* needs no detections), an
        unlocked candidate is faced where it was last seen.
        """
        self.started += 1
        if self.inspection_started is not None:
            self.inspection_started += 1
        info = {"kind": "target_closing", "reason": "detector frame missing: no evidence, clocks paused",
                "target_visible": False, "blind": True}
        if self.locked and self.xyz is not None:
            command = self.path.command(self, obs, world)
            if command is not None:
                self.phase = "CLOSE_OCCLUDED"
                return replace(command, info=dict(command.info, **info, persistent_lock=True))
        bearing = (math.atan2(self.xyz[1] - obs.pose.y, self.xyz[0] - obs.pose.x)
                   if self.xyz is not None else self.verification_yaw)
        return NavigationCommand.hold(final_yaw=normalize_angle(bearing), info=info)

    def _no_path(self, obs, world, distance):
        """A LOCKED target A* cannot reach from here: look at the floor around, then give the spot up.

        First the footing sweep (``footing_sweep``; the camera controller
        owns it and its turns come back here one per action, tagged
        ``FOOTING``): the blind disk under the camera is the usual reason a
        path does not exist from where the agent has not moved, so the
        sweep is asked for only while ``footing_unknown_fraction`` of the
        cells within ``footing_radius_m`` are still unknown. Then, if the
        sweep bought no path either -- or there was nothing to sweep -- the
        lock is released without a rejection after
        ``release_after_footing_steps`` pathless actions, the spot is
        remembered as boxed in, and the search -- which can walk -- carries
        on; a new takeover from within ``boxed_in_radius_m`` of here is
        refused. Between those, the centring hold as before: the target
        stays in view.
        """
        p, s = self.policy, self.settings
        camera = p.camera_control
        self.no_path_steps += 1
        if (s.footing_sweep and not self.footing_done and p.episode.action_spec.has_camera_tilt
                and self.no_path_steps >= s.footing_after_steps):
            self.footing_done = True
            blind = unknown_around(world, (obs.pose.x, obs.pose.y), s.footing_radius_m) >= s.footing_unknown_fraction
            if blind and camera.begin_inspection(obs, p.mapping.floor_id, reason="footing"):
                self.footing_sweeps += 1
                sweep = camera.inspection_command(obs)
                if sweep is not None:
                    return self._footing_command(obs, sweep)
        if self.footing_done or not s.footing_sweep or not p.episode.action_spec.has_camera_tilt:
            self.pathless_after_footing += 1
            if self.pathless_after_footing >= s.release_after_footing_steps:
                self.boxed_releases += 1
                self.boxed_in.append(((float(obs.pose.x), float(obs.pose.y)), p.mapping.floor_id, int(obs.step)))
                self._release(obs, why="no safe path from here", remember=False)
                turn = p.episode.action_spec.turn_angle_rad
                return NavigationCommand.hold(
                    final_yaw=normalize_angle(obs.pose.yaw + turn),
                    info={"kind": "target_released", "phase": "RELEASED",
                          "reason": "no safe path to the target %.1f m away from here; the search moves on" % distance})
        return self._scan(obs, None, "no safe target path; exploration remains suspended")

    def _continue_footing(self, obs, world, distance):
        """The footing sweep in force, one action at a time -- cut short the moment a path exists.

        The sweep was asked for because A* had no path from here; its turns
        come back through here (``FOOTING``), and once the floor it mapped
        connects the agent to the target's standoff the rest of the circle
        is not worth its actions: the camera is told to restore (one
        LOOK_UP) and the approach begins level. Without this the sweep's
        pitch stood for the whole approach and the first inspection (Hanson
        2026-10-05, actions 5-63: 30 degrees down, the plant's foliage never
        projected, the lock released as "saw nothing").
        """
        camera = self.policy.camera_control
        if not camera.footing:
            return None
        sweep = camera.inspection_command(obs)
        if sweep is None:
            return None
        if (not camera.inspection["restoring"] and self.inspection_started is None
                and distance > self.settings.terminal_distance_m
                and self.path.command(self, obs, world) is not None):
            camera.inspection["restoring"] = True
            sweep = camera.inspection_command(obs)
            if sweep is None:
                return None
        return self._footing_command(obs, sweep)

    def refuses_far_candidate(self, obs, xy):
        """Whether a target seen at ``xy`` is one the takeover would not start on from here.

        Far candidates only (beyond ``rejection_min_range_m``; a close view is
        new evidence): one within the rejection memory, one seen during the
        cooldown after a release, or one seen from a spot the lock was given
        up at for want of a path. The legacy target evidence asks the same
        question, so the search does not walk after what the takeover has
        just refused (Hanson 2026-10-05, actions 18-26: nine actions toward
        a released chair, and the warm-up begun again where they ended).
        **At any range** a landmark on an anchor a release *disproved from
        close* (``DISPROVED_RELEASES``: the wider perspective, the second
        unverified look, an inspection that saw nothing) is refused too --
        Markleeville/000000, 2026-10-08: the closing gave the table back from
        the wider perspective at action 41 and the legacy pursuit STOPped on
        it at action 50.
        """
        s = self.settings
        if self.locked:
            return False
        if self._disproved_near(xy):
            self.disproved_refusals += 1
            return True
        range_m = math.dist((obs.pose.x, obs.pose.y), (float(xy[0]), float(xy[1])))
        if range_m <= s.rejection_min_range_m:
            return False
        cooling = (not self.active and self.released_step is not None
                   and obs.step - self.released_step < s.release_cooldown_actions)
        return (cooling or self._rejected_nearby((float(xy[0]), float(xy[1]), 0.0), self.policy.mapping.floor_id)
                or self._boxed_in_here(obs))

    def _footing_command(self, obs, sweep):
        self.phase = "FOOTING"
        return replace(sweep, info=dict(sweep.info, kind="target_closing", phase=self.phase, persistent_lock=True,
                                        target_visible=self.last_seen == obs.step,
                                        reason="no safe target path; mapping the floor around the feet"))

    def _centred(self, obs):
        """Whether the fresh box is aligned: its centre within half a turn, or spanning the image's centre column."""
        if self.last_seen != obs.step or self.box is None:
            return False
        k = obs.camera.intrinsics
        if self.box[0] <= k.cx <= self.box[2]:
            return True
        return turn_action(self._bbox_error(obs), self.policy.episode.action_spec.turn_angle_rad) is None

    def _bbox_error(self, obs, box=None):
        # Optical x points right whereas positive ENU yaw turns left. At a
        # pitched camera the vertical box coordinate also affects body forward.
        k = obs.camera.intrinsics
        box = self.box if box is None else box
        u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        forward = math.cos(obs.pose.camera_pitch) - (v - k.cy) / k.fy * math.sin(obs.pose.camera_pitch)
        error = math.atan2(-(u - k.cx) / k.fx, forward)
        return error

    def _inspect(self, obs, distance):
        """Stationary pitch/yaw inspection; never STOP from remembered range alone.

        Three facts about a big object at terminal range shape this:

        * its box is clipped at an image edge, so the box centre is not the
          object's centre -- a fresh box that SPANS the image centre column
          is aligned, whatever its centre says (centring it turned the
          Ranchester agent off the couch twelve times in a row);
        * its centroid is not its range -- the terminal test reads the near
          edge of the freshly measured surface beside the centroid, which is
          what the benchmark's success radius is measured to;
        * the pitch its height predicts is where to look for it when it is
          NOT in view; a fresh, aligned sighting within range STOPs at
          whatever pitch it came at;
        * when the inspection budget runs out after the target WAS seen
          fresh in terminal range from this very spot, a STOP is the right
          verdict and an error is not (``stop_on_exhausted_inspection``);
          when it runs out and the target was NOT seen from here at all, the
          lock was wrong and the candidate is released
          (``release_on_failed_inspection``; None is returned and the
          caller hands the action to the search).
        """
        actions, s = self.policy.episode.action_spec, self.settings
        k = obs.camera.intrinsics
        if self.inspection_started is None:
            self.inspection_started, self.inspection_index = obs.step, 0
            self.inspection_sighting = None
            self.centring_done, self.centring_step, self.weak_holds = False, -1, 0
            self.inspection_stepped, self.inspection_turn = 0, None
        self.phase = "INSPECT"
        pitch = self._pitch(obs, distance)
        bearing = math.atan2(self.xyz[1] - obs.pose.y, self.xyz[0] - obs.pose.x)
        fresh = self.last_seen == obs.step
        fix = self._servo(obs) if fresh else None
        if fix is not None:
            # A fresh box: its place in the frame, not the memory's geometry, says where the camera goes.
            pitch = self._servo_pitch(obs, fix) if fix.wanted else float(obs.pose.camera_pitch)
        spans_centre = fresh and self.box is not None and self.box[0] <= k.cx <= self.box[2]
        error = 0.0 if spans_centre else self._bbox_error(obs) if fresh else normalize_angle(bearing - obs.pose.yaw)
        yaw = normalize_angle(obs.pose.yaw + error)
        turning = not spans_centre and turn_action(error, actions.turn_angle_rad) is not None
        if fresh and turning and self._turn_back_undoes_the_last(obs, error):
            # The last centring turn carried the box past the centre by less than a turn, and the
            # turn back would recreate the frame before it: the box is as centred as the turn
            # quantum allows (since 2026-10-14; Wiconisco/000193 alternated TURN_LEFT/TURN_RIGHT
            # at a toilet 16-17 degrees either side of the centre for twenty-four actions).
            self.dead_band_alignments += 1
            turning, error, yaw = False, 0.0, obs.pose.yaw
        tilting = pitch is not None and pitch_action(obs.pose.camera_pitch, pitch, actions.tilt_angle_rad,
                                                     actions.min_pitch_rad, actions.max_pitch_rad) is not None
        limit = s.terminal_distance_m + s.range_tolerance_m
        # The memory STOPs below measure the planar range to the memory; from an ARRIVED standoff
        # (since 2026-10-11) the arrival tolerance is part of where the agent stands -- a kept
        # standoff ends within the terminal radius of the memory, the executor stops within
        # ``goal_tolerance_m`` of it -- so the trajectory is complete that much farther out.
        reach = limit + (self.policy.converter_params.goal_tolerance_m if self.path.arrived else 0.0)
        measured = self._fresh_measured(obs)
        if measured is not None and measured <= limit:
            self.inspection_sighting = obs.step
        history = self._history_qualifies()
        support = s.support_surface_resilience and self.inspection_started <= self.support_seen
        self.terminal_missing = 0 if fresh else self.terminal_missing + 1
        info = {"kind": "target_closing", "target_confirmed": True, "persistent_lock": True,
                "target_visible": fresh, "range_m": distance, "measured_m": measured, "bbox_yaw_error_rad": error,
                "box_spans_centre": spans_centre, "phase": self.phase, "approach_history": history,
                "support_surface_fresh": self.support_seen == obs.step,
                "floor_cut": fresh and self._floor_cut(obs),
                "confidence": self.conf if fresh else None,
                "bbox_region": None if fix is None else fix.region,
                "memory_sigma_m": None if self.memory is None else self.memory.sigma_m}
        a_step_out = (fresh and measured is not None and limit < measured <= limit + actions.forward_step_m)
        if (a_step_out and s.closest_approach_stop and self.path.arrived and self.path.widened
                and (self.inspection_stepped >= s.inspection_max_steps or self.inspection_blocked)):
            # The standoff was found only beyond the standoff ring -- no closer point had a route -- the
            # agent has walked it and stepped the inspection's steps, or was blocked stepping, and the
            # target is fresh in view within one forward step of the terminal range: as close as the floor
            # allows. STOP here; the release that would follow gives up a target in plain view (since
            # 2026-10-15; Collierville/000009: a 1.5 m television on its cabinet, the 0.70 m ring inside
            # the cabinet or against it, three correct locks released as "no safe path" from 3 m).
            self.closest_approach_stops += 1
            return self._terminal(obs, distance, measured, dict(
                info, phase="STOP", depth_valid=True,
                reason="closest reachable approach: the fresh range %.2f m is within one step of the %.2f m limit, no "
                       "closer standoff has a route and %s" % (measured, limit, "the last step was blocked"
                                                                if self.inspection_blocked else "the inspection's steps are spent")))
        if a_step_out and abs(error) <= actions.turn_angle_rad + 1e-6 and self.inspection_stepped < s.inspection_max_steps:
            # The fresh range is a step beyond the limit and the agent faces the target within a
            # turn: one MOVE_FORWARD puts the next frame inside the terminal range. Turning in
            # place here only moved the box from one side of the centre to the other.
            self.inspection_stepped += 1
            self.inspection_steps += 1
            self.inspection_step_pending = True
            reach_m = 2.0 * actions.forward_step_m
            ahead = (obs.pose.x + reach_m * math.cos(obs.pose.yaw), obs.pose.y + reach_m * math.sin(obs.pose.yaw))
            return NavigationCommand.follow([(obs.pose.x, obs.pose.y), ahead],
                                            camera_pitch=None if pitch is None else float(obs.pose.camera_pitch),
                                            info=dict(info, inspect_step="one step: the fresh range is %.2f m against a "
                                                                         "%.2f m limit" % (measured, limit)))
        if fresh and measured <= limit and (not turning or history):
            # The terminal evidence is in this frame. Before the STOP is taken on it: a whole box,
            # then a confident one (since 2026-10-11; both bounded, see ``_before_stop``).
            deferred = self._before_stop(obs, distance, fix, turning, info)
            if deferred is not None:
                return deferred
        # The pitch the geometry predicts is where to LOOK for the target, not a
        # condition on having seen it: the Hanson toilet projected only at 60 degrees
        # down where the prediction said 30, and twenty actions of LOOK_UP/LOOK_DOWN
        # followed a fresh, centred, in-range sighting before the budget STOPped.
        if fresh and not turning and measured <= limit:
            return self._terminal(obs, distance, measured, dict(info, phase="STOP", depth_valid=True,
                                                         reason="fresh terminal target confirmation"))
        if history and fresh and measured <= limit:
            # Tracked and approached over metres: a cropped box whose centre says "turn" is
            # still the object, and the benchmark measures range, not heading.
            self.history_stops += 1
            return self._terminal(obs, distance, measured, dict(
                info, phase="STOP", reason="fresh in-range sighting after a %.1f m approach; alignment not required"
                % self.history.path_m))
        if (history and not fresh and distance <= reach
                and self.terminal_missing >= s.history_stop_after_missing):
            # The trajectory is complete and the detector has dropped the box for a few
            # frames: the 3-D memory refined over the approach is what we STOP on.
            self.history_stops += 1
            return self._terminal(obs, distance, measured, dict(
                info, phase="STOP", standoff_reached=self.path.arrived,
                reason="approach history: target tracked over %.1f m, the 3-D memory stands %.2f m "
                "away (sigma %.2f m) and %d frames passed without a fresh box%s"
                % (self.history.path_m, distance, self.memory.sigma_m, self.terminal_missing,
                   "; the standoff is reached" if self.path.arrived else "")))
        if (not fresh and self.centring_step == obs.step - 1 and self.inspection_sighting == obs.step - 1
                and distance <= limit):
            # The previous frame met the STOP's own conditions -- fresh, in range, confident -- and the
            # one centring tilt deferred it for a whole box; the detector did not show one. That tilt
            # may cost an action, never the STOP: it is taken on the evidence of the frame before.
            return self._terminal(obs, distance, measured, dict(
                info, phase="STOP", depth_valid=True,
                reason="fresh terminal target confirmation one action ago; the centring tilt lost the box, the "
                       "sighting stands (memory %.2f m away, sigma %.2f m)" % (distance, self.memory.sigma_m)))
        below = None if fresh else self._below_the_frame(obs, distance)
        if below is not None and distance <= limit and obs.step - self.last_seen <= s.blind_cone_max_missing:
            # The camera cannot show the target from here -- it is under the frame's lower
            # edge, and this protocol has no LOOK_DOWN -- so a missing box is not evidence of
            # absence, and a yaw sweep cannot find what no yaw brings into view. Seen fresh
            # a few actions ago, the memory inside the terminal range: STOP (since 2026-10-11;
            # before, the Wiconisco toilet was released as "saw nothing in any view" with the
            # agent standing inside the success region).
            self.blind_cone_stops += 1
            return self._terminal(obs, distance, measured, dict(
                info, phase="STOP", below_frame_px=round(below, 1),
                reason="target below the camera's field of view from here (%d px of it would show; no LOOK action): "
                       "seen fresh %d action(s) ago, the 3-D memory stands %.2f m away (sigma %.2f m)"
                % (max(0, int(below)), obs.step - self.last_seen, distance, self.memory.sigma_m)))
        views = self._inspection_views(pitch)
        blank = (s.release_on_failed_inspection and self.inspection_sighting is None
                 and self.last_seen < self.inspection_started and self.inspection_index >= len(views))
        exhausted = obs.step - self.inspection_started >= s.max_reacquire_steps
        if (blank or exhausted) and self.inspection_sighting is None and (history or support):
            if distance <= reach and (exhausted or history):
                # Not "saw nothing": the approach brought us here, or the surface the target
                # stands on is still measured at the memory's spot. Complete and STOP.
                if history:
                    self.history_stops += 1
                else:
                    self.support_stops += 1
                return self._terminal(obs, distance, measured, dict(
                    info, phase="STOP", standoff_reached=self.path.arrived,
                    reason=("approach history: %.1f m walked to this spot; the 3-D memory stands"
                            % self.history.path_m) if history else
                    "support surface consistent with the 3-D memory at terminal range; inspection budget spent"))
            if history and distance > reach:
                # Too far for a memory STOP: step closer along the trajectory instead of releasing.
                self.history_holds += 1
                self.inspection_started = None
                self.inspection_resumptions += 1
                if self.path.arrived:
                    self.path.drop()                 # walking the arrived route again would arrive again
                return NavigationCommand.hold(camera_pitch=None, final_yaw=yaw, info=dict(
                    info, reason="approach history: resuming the approach from %.2f m instead of releasing" % distance))
            if support and not exhausted:
                self.support_holds += 1                      # keep looking: the dynamic pitch brings the object back
                blank = False
        if exhausted or blank:
            if s.stop_on_exhausted_inspection and self.inspection_sighting is not None:
                return self._terminal(obs, distance, measured, dict(
                    info, phase="STOP", reason="inspection exhausted; target seen fresh in terminal range at action %d"
                    % self.inspection_sighting))
            if s.release_on_failed_inspection:
                # Every view tried once and nothing seen (``blank``), or the budget spent
                # without an in-range sighting: the lock was wrong.
                self.inspection_releases += 1
                self._release(obs, radius_factor=s.failed_inspection_radius_factor,
                              why="inspection saw nothing" + (" in any view" if blank else ""))
                return None
            self.fail("terminal visual confirmation unavailable")
        if not fresh or (not turning and not tilting):
            yaw, pitch = self._inspection_view(obs, bearing, pitch)
        elif turning:
            # The converter tilts before it faces: asking for the predicted pitch here
            # would LOOK away from a target in view and never turn (LOOK_DOWN/LOOK_UP
            # for the whole budget). Centre it at the pitch it was seen at.
            pitch = None
            self.inspection_turn = (int(obs.step), 1 if error > 0 else -1)
        return NavigationCommand.hold(camera_pitch=pitch, final_yaw=yaw, info=info)

    def _turn_back_undoes_the_last(self, obs, error):
        """Whether a centring turn on this fresh box would undo the inspection's turn of the previous action.

        The dead band of :func:`turn_action` is half a turn; a box the
        detector draws a few degrees differently at two headings can sit
        just outside it on both sides, so the inspection turns left, then
        right, then left, for ever. A turn opposite to the last one, one
        action later, with the error under a whole turn, is that cycle.
        """
        last = self.inspection_turn
        if last is None or last[0] != obs.step - 1:
            return False
        sign = 1 if error > 0 else -1
        return sign == -last[1] and abs(error) < self.policy.episode.action_spec.turn_angle_rad

    def _before_stop(self, obs, distance, fix, turning, info):
        """What to do instead of STOPping on this fresh in-range frame -- or None: STOP.

        Two bounded steps between the terminal evidence and the STOP (since
        2026-10-11):

        * **a confident box.** A fresh box under ``stop_confidence`` is held
          for the one frame that keeps it whole (:meth:`_verification_view`:
          the servo's tilt with the LOOK pair, else a LOOK or TURN that keeps
          the box in), at most ``weak_stop_max_holds`` times in a row; when
          no such frame exists, or the holds are spent, the STOP goes ahead
          on the evidence there is. Not on the approach-history path, which
          STOPs on the trajectory, not the box.
        * **a whole box.** With the LOOK pair, when the servo can re-frame the
          box (cut by an edge, or off the centre band), ONE centring tilt --
          once per inspection (``stop_centring``, ``centring_done``) -- and
          the STOP is taken on the next frame: on the whole box if the
          detector shows it, and on this frame's evidence if it does not
          (:meth:`_inspect`, ``centring_step``) -- the tilt may cost one
          action, never the STOP. Not while the box still needs a yaw turn:
          the converter tilts before it faces.
        """
        s = self.settings
        weak = self.conf is not None and self.conf < s.stop_confidence
        if weak and not turning and self.weak_holds < s.weak_stop_max_holds:
            view = self._verification_view(obs, distance)
            if view is not None:
                self.weak_holds += 1
                self.weak_stop_holds += 1
                return replace(view, info=dict(info, verify_step=view.info.get("verify_step"),
                                               bbox_region=view.info.get("bbox_region", info.get("bbox_region")),
                                               reason="fresh in-range box at %.2f, under stop_confidence %.2f: another frame "
                                                      "(%d of %d) before the STOP"
                                                      % (self.conf, s.stop_confidence, self.weak_holds, s.weak_stop_max_holds)))
        if s.stop_centring and not self.centring_done and not turning and fix is not None and fix.wanted:
            self.centring_done, self.centring_step = True, obs.step
            self.stop_centring_tilts += 1
            return NavigationCommand.hold(camera_pitch=self._servo_pitch(obs, fix), final_yaw=obs.pose.yaw, info=dict(
                info, servo=fix.reason,
                reason="fresh in-range box in the %s of the frame: one centring tilt before the terminal STOP"
                       % fix.region.replace("_", " ")))
        return None

    def _terminal(self, obs, distance, measured, info):
        """The decided STOP -- taken now, or after the forward steps of ``stop_overshoot_steps``.

        The decision is final here: every STOP of :meth:`_inspect` comes
        through, and once the overshoot has begun no frame is consulted
        again (:meth:`observe` and :meth:`plan` short-circuit on it) -- the
        steps are walked blind toward the 3-D memory, and the STOP follows
        whatever the detector shows meanwhile. ``stop_overshoot_steps``
        steps always; more, up to ``stop_overshoot_max_steps``, while the
        memory still stands beyond ``stop_overshoot_within_m``. See the
        settings for why.

        Args:
            obs: The deciding frame.
            distance: Planar range to the 3-D memory from it, metres.
            measured: Its fresh near-edge range, or None without a fresh box.
            info: The STOP's diagnostics, ``reason`` included.
        """
        self.phase = "STOP"
        if self.settings.stop_overshoot_max_steps < 1:
            return NavigationCommand.stop_here(info=info)
        self.overshoot = {"info": dict(info), "decided": int(obs.step), "step": -1,
                          "from_xy": (float(obs.pose.x), float(obs.pose.y)),
                          "near_m": None if measured is None else float(measured), "memory_m": float(distance),
                          "steps": 0, "actions": 0, "blocked": False, "ended": None}
        return self._overshoot_step(obs)

    def _overshoot_step(self, obs):
        """One action of the overshoot: a forward step toward the memory, or the STOP that ends it."""
        s, actions, o = self.settings, self.policy.episode.action_spec, self.overshoot
        if o["step"] != obs.step:
            o["actions"] += int(o["step"] >= 0)
            o["step"] = int(obs.step)
        step_m = actions.forward_step_m
        here = (float(obs.pose.x), float(obs.pose.y))
        walked = math.dist(here, o["from_xy"])
        distance = math.dist(here, self.xyz[:2])
        near = None if o["near_m"] is None else o["near_m"] - walked
        bearing = math.atan2(self.xyz[1] - obs.pose.y, self.xyz[0] - obs.pose.x)
        error = normalize_angle(bearing - obs.pose.yaw)
        ended = None
        if o["blocked"]:
            ended = "a step was blocked"
        elif o["steps"] >= s.stop_overshoot_max_steps:
            ended = "%d step(s) taken" % o["steps"]
        elif o["steps"] >= s.stop_overshoot_steps and distance <= s.stop_overshoot_within_m:
            ended = ("%d step(s) taken and the memory stands %.2f m away, within %.2f m"
                     % (o["steps"], distance, s.stop_overshoot_within_m))
        elif o["actions"] >= s.stop_overshoot_max_actions:
            ended = "%d actions spent" % o["actions"]
        elif distance - step_m < s.stop_overshoot_min_range_m:
            ended = "the memory would stand %.2f m away after another step" % (distance - step_m)
        elif near is not None and near - step_m < self.policy.settings.body_radius_m:
            ended = "the measured near edge would stand %.2f m away after another step" % (near - step_m)
        elif abs(error) > math.pi / 2:
            ended = "the memory is %.0f degrees off the heading" % math.degrees(abs(error))
        stage = {"decided_step": o["decided"], "steps": o["steps"], "actions": o["actions"], "blocked": o["blocked"],
                 "walked_m": round(walked, 3), "memory_m": round(distance, 3),
                 "near_m": None if near is None else round(near, 3), "ended": ended}
        if ended is not None:
            self.phase = "STOP"
            o["ended"] = ended
            return NavigationCommand.stop_here(info=dict(o["info"], phase="STOP", range_m=distance, stop_overshoot=stage))
        self.phase = "ADVANCE"
        self.overshoot_commands += 1
        # Within a turn of the memory the heading that put the object in the frame is the better
        # aim; farther off the converter turns toward the memory first (turns cost no path).
        aim = float(obs.pose.yaw) if abs(error) <= actions.turn_angle_rad else bearing
        reach_m = 2.0 * step_m                   # the converter arrives within a step of a goal: aim past it
        ahead = (here[0] + reach_m * math.cos(aim), here[1] + reach_m * math.sin(aim))
        return NavigationCommand.follow(
            [here, ahead], camera_pitch=None,          # the camera stays where it is (``camera_control`` holds it)
            info=dict(o["info"], phase=self.phase, target_visible=self.last_seen == obs.step, range_m=distance,
                      stop_overshoot=stage,
                      reason="STOP decided at action %d (%s): step %d of at most %d toward the target before it%s"
                             % (o["decided"], o["info"].get("reason"), o["steps"] + 1, s.stop_overshoot_max_steps,
                                "" if o["steps"] < s.stop_overshoot_steps else
                                "; the memory still stands %.2f m away, beyond %.2f m" % (distance, s.stop_overshoot_within_m))))

    def note_action(self, action):
        """The action the agent took on the closing's command (the policy's ``notify_action``)."""
        forward = getattr(action, "name", action) == "MOVE_FORWARD"
        if self.overshoot is not None and forward:
            self.overshoot["steps"] += 1
        if self.inspection_step_pending and not forward:
            self.inspection_step_pending = False       # the converter turned instead: no step was tried

    def note_blocked(self):
        """The converter reports the last MOVE_FORWARD did not move the agent (the policy's ``notify_blocked``)."""
        if self.overshoot is not None:
            self.overshoot["blocked"] = True
        if self.inspection_step_pending:
            self.inspection_blocked = True
            self.inspection_step_pending = False

    def _approach_stalled(self, obs):
        """Whether the last ``approach_stall_actions`` CLOSE actions left the agent inside a circle of ``approach_stall_m``."""
        s = self.settings
        if s.approach_stall_actions <= 0:
            return False
        self.close_trail.append((float(obs.pose.x), float(obs.pose.y)))
        if len(self.close_trail) <= s.approach_stall_actions:
            return False
        del self.close_trail[:-s.approach_stall_actions - 1]
        x0, y0 = self.close_trail[0]
        return all(math.dist((x, y), (x0, y0)) <= s.approach_stall_m for x, y in self.close_trail[1:])

    def _fresh_measured(self, obs):
        """The freshly measured range to the target this action, or None when not seen.

        The near edge of the surface in view, or the centroid, whichever is
        nearer -- and, for a box the frame's lower edge cut
        (:meth:`_floor_cut`), the planar range to the 3-D memory when that
        is nearer still: the part in view is then the upper, farther part
        of the object, and the memory is its centroid. (A 2026-10-14
        experiment read a small object's range from the fused memory
        instead of its near edge, after Markleeville/000188 STOPped 1.25 m
        from a toilet on a 0.84 m near edge; it flipped CLOSE and INSPECT
        on alternate frames -- TURN_LEFT / TURN_RIGHT for 25 actions -- and
        stopped 3 cm outside the evaluator's disk where it did STOP. The
        near edge stands.)
        """
        if self.last_seen != obs.step or self.observed_xyz is None:
            return None
        measured = math.dist((obs.pose.x, obs.pose.y), self.observed_xyz[:2])
        if self.observed_near_m is not None:
            measured = min(measured, self.observed_near_m)
        if self._floor_cut(obs):
            memory = math.dist((obs.pose.x, obs.pose.y), self.xyz[:2])
            if memory < measured:
                if self._floor_cut_step != obs.step:
                    self._floor_cut_step = obs.step
                    self.floor_cut_measurements += 1
                measured = memory
        return measured

    def _floor_cut(self, obs):
        """Whether the fresh box is cut by the frame's lower edge with a lower part of the target hidden under it."""
        return self.last_seen == obs.step and self._floor_cut_box(obs, self.box, self.visible_floor_m)

    def _floor_cut_box(self, obs, box, visible_floor_m):
        """Whether ``box`` is cut by the frame's lower edge with a lower part of the target hidden under it.

        Three facts, all measured: the box reaches the lower margin; the
        target is below the camera (its memory); and the lowest supported
        point in view this frame (``visible_floor_m``) stands at least
        ``floor_cut_min_hidden_m`` higher than the lowest point the lock has
        measured so far -- part of what was seen before is now under the
        frame. A box that touches the bottom edge while showing the object
        down to the same height as always (a pot standing on the floor,
        seen from far) is not cut in this sense.
        """
        s = self.settings
        if (not s.floor_cut_terminal or box is None or self.xyz is None
                or visible_floor_m is None or self.lowest_m is None):
            return False
        k = obs.camera.intrinsics
        if box[3] < k.height - s.border_margin_px:
            return False
        camera_z = float(obs.pose.z) + float(obs.camera.height_m)
        return float(self.xyz[2]) < camera_z and visible_floor_m - self.lowest_m >= s.floor_cut_min_hidden_m

    def _below_the_frame(self, obs, distance):
        """The rows of the target the camera would show from here, when too few to recognise it; else None.

        Only without LOOK actions (``blind_cone_stop``; with them the pitch
        controller tilts instead), with the target's bearing inside the
        horizontal field of view -- a target to the side is a yaw matter the
        sweep handles -- and its measured top below the camera. The count is
        the rows from the top's predicted image row down to the frame's
        lower margin; fewer than ``blind_cone_min_fraction`` of the frame is
        the sliver ``perception.clipped_box`` will not accept, and the
        detector has nothing to recognise.
        """
        s, actions = self.settings, self.policy.episode.action_spec
        if not s.blind_cone_stop or actions.has_camera_tilt or self.xyz is None or self.top_m is None:
            return None
        k = obs.camera.intrinsics
        margin = s.border_margin_px
        bearing = math.atan2(self.xyz[1] - obs.pose.y, self.xyz[0] - obs.pose.x)
        if abs(normalize_angle(bearing - obs.pose.yaw)) > math.atan2(0.5 * k.width - margin, k.fx):
            return None
        camera_z = float(obs.pose.z) + float(obs.camera.height_m)
        if self.top_m >= camera_z:
            return None
        depression = math.atan2(camera_z - self.top_m, max(float(distance), 1e-6)) - float(obs.pose.camera_pitch)
        visible = (k.height - margin) - (k.cy + k.fy * math.tan(depression))
        return float(visible) if visible < s.blind_cone_min_fraction * k.height else None

    def _inspection_views(self, pitch):
        """The bounded yaw/pitch views an inspection cycles through, centred on the stored bearing."""
        actions = self.policy.episode.action_spec
        pitches = [pitch]
        if pitch is not None:
            for value in (pitch - actions.tilt_angle_rad, pitch + actions.tilt_angle_rad):
                if (actions.min_pitch_rad is None or value >= actions.min_pitch_rad) and (actions.max_pitch_rad is None or value <= actions.max_pitch_rad):
                    pitches.append(value)
        return [(offset, value) for offset in (0, 1, -1) for value in pitches]

    def _inspection_view(self, obs, bearing, pitch):
        """Try bounded up/down views centred on the stored target bearing."""
        actions = self.policy.episode.action_spec
        views = [(normalize_angle(bearing + offset * actions.turn_angle_rad), value)
                 for offset, value in self._inspection_views(pitch)]
        yaw, value = views[self.inspection_index % len(views)]
        aligned = turn_action(normalize_angle(yaw - obs.pose.yaw), actions.turn_angle_rad) is None
        pitched = value is None or pitch_action(obs.pose.camera_pitch, value, actions.tilt_angle_rad,
                                                actions.min_pitch_rad, actions.max_pitch_rad) is None
        if aligned and pitched:
            self.inspection_index += 1
            yaw, value = views[self.inspection_index % len(views)]
        return yaw, value

    def diagnostics(self):
        return {"active": self.active, "locked": self.locked, "phase": self.phase,
                "bearing": None if self.bearing is None else {k: v for k, v in self.bearing.items() if k != "moves"},
                "bearing_starts": self.bearing_starts, "bearing_resolved": self.bearing_resolved,
                "bearing_abandoned": self.bearing_abandoned, "bearing_refusals": self.bearing_refusals,
                "persistent_lock": self.locked, "visible": self.last_seen == self.processed,
                "navmesh_goal_xyz": self.path.goal_xyz, "goal_refinements": self.path.refinements,
                "occluded_path_steps": self.occluded_path_steps, "inspection_started_step": self.inspection_started,
                "inspection_sighting_step": self.inspection_sighting, "observed_near_m": self.observed_near_m,
                "consecutive_frames": self.count, "started_step": self.started,
                "last_seen_step": self.last_seen, "xyz": self.xyz, "bbox": self.box,
                "floor_id": self.floor_id, "label": self.label, "failure": self.failure,
                "releases": self.releases, "inspection_releases": self.inspection_releases,
                "inspection_resumptions": self.inspection_resumptions, "map_releases": self.map_releases,
                "stall_releases": self.stall_releases,
                "last_release": self.last_release, "border_rejections": self.border_rejections,
                "map_rejections": self.map_rejections, "weak_resightings": self.weak_resightings,
                "no_path_steps": self.no_path_steps, "footing_sweeps": self.footing_sweeps,
                "boxed_releases": self.boxed_releases,
                "overrides": self.overrides, "context_rejections": self.context_rejections,
                "suspect": self.suspect, "suspect_locks_held": self.suspect_locks_held,
                "same_spot_rejections": self.same_spot_rejections, "blind_frames": self.blind_frames,
                "memory": None if self.memory is None else self.memory.diagnostics(),
                "approach": None if self.history is None else dict(self.history.diagnostics(),
                                                                   qualifies=self._history_qualifies()),
                "backoff": None if self.backoff is None else self.backoff.diagnostics(),
                "backoffs": self.backoffs, "backoff_skips": self.backoff_skips, "backoff_repeats": self.backoff_repeats,
                "wider_releases": self.wider_releases, "trusted_releases": self.trusted_releases,
                "repeat_releases": self.repeat_releases,
                "disproved_refusals": self.disproved_refusals, "no_step_holds": self.no_step_holds,
                "support_sightings": self.support_sightings, "support_seen_step": self.support_seen,
                "support_xyz": self.support_xyz, "support_holds": self.support_holds, "support_stops": self.support_stops,
                "history_holds": self.history_holds, "history_stops": self.history_stops,
                "terminal_missing": self.terminal_missing,
                "visible_floor_m": self.visible_floor_m, "lowest_m": self.lowest_m, "top_m": self.top_m,
                "footprint_m": self.footprint_m, "depth_associations": self.depth_associations,
                "floor_cut_frames": self.floor_cut_frames, "floor_cut_measurements": self.floor_cut_measurements,
                "blind_cone_stops": self.blind_cone_stops,
                "confidence": self.conf, "box_drift_rows": self.box_drift_rows,
                "bbox_region": None if self.last_fix is None else self.last_fix.region,
                "bbox_pitch_steps": None if self.last_fix is None else self.last_fix.steps,
                "bbox_pitch_corrections": self.bbox_pitch_corrections, "stop_centring_tilts": self.stop_centring_tilts,
                "weak_stop_holds": self.weak_stop_holds, "goal_kept": self.path.kept,
                "resighting_exemptions": self.resighting_exemptions, "glimpse_steps": self.glimpse_steps,
                "verify_pauses": self.verify_pauses, "backoff_trusted_skips": self.backoff_trusted_skips,
                "backoff_persistence_skips": self.backoff_persistence_skips,
                "inspection_steps": self.inspection_steps, "dead_band_alignments": self.dead_band_alignments,
                "through_unknown_routes": self.through_unknown_routes, "sweep_releases": self.sweep_releases,
                "widened_standoffs": self.widened_standoffs, "stale_standoffs_kept": self.stale_standoffs_kept,
                "closest_approach_stops": self.closest_approach_stops, "inspection_blocked": self.inspection_blocked,
                "occluder_reads": self.occluder_reads,
                "overshoot_commands": self.overshoot_commands,
                "stop_overshoot": None if self.overshoot is None else {
                    key: self.overshoot[key] for key in ("decided", "steps", "actions", "blocked", "ended")},
                "standoff_arrived": self.path.arrived, "standoff_arrivals": self.path.arrivals,
                "standoff_dropped": self.path.dropped, "standoff_widened": self.path.widened,
                "boxed_in": [{"xy": [round(v, 2) for v in xy], "floor_id": floor, "step": step}
                             for xy, floor, step in self.boxed_in],
                "rejected": [{"xyz": list(entry[0]), "floor_id": entry[1], "step": entry[2], "radius_m": entry[3],
                              "why": entry[4] if len(entry) > 4 else "unverified",
                              "from_xy": None if len(entry) < 6 else [round(v, 2) for v in entry[5]]}
                             for entry in self.rejected]}
