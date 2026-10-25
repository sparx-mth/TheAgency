# ObjectNav benchmark runtime: the adapters, the services, the operator tooling

**The method itself lives in `core/planning/object_search/`** (its README is
the architecture reference for the algorithm); this tree is what applies it
to a benchmark -- the Habitat bridge, the Gibson and HM3D dataset/protocol/scorer
adapters, the HTTP service clients, the recording and the dashboards -- and the
regression rig that exercises the method on the benchmark's camera. See "Where
the code is" below for the map and the recipe for a new dataset.

Observed RGB-D mapping → semantic room graphs → room LLM → RPT* room order →
local exploration and A*/WA* paths → existing discrete actions. Both explorers
retain persistent floor contexts; stairs are what the robot's own RGB-D
measures by default (`RPTSettings.multifloor.stair_source: "observed"` since
2026-10-21 -- a reachable run of treads rising 0.40 m within 5 m in a
flight's profile: three or more risers of 0.10-0.24 m with no level between
them walked for more than 0.55 m, `world_model/stair_terrain.py`
`stair_profile` since 2026-10-22, when a coffee table and a sofa had passed
for one; no simulator geometry is read. The navmesh
connectors are the `"ground_truth"` ablation, see
[MULTISTORY.md](gibson/MULTISTORY.md): their centreline begins 0.45 m before
Darden's first riser, on a sloped threshold, and drawn on the grid it closed
the junction below the flight). **Taking them is off by default**
(`RPTSettings.allow_stair_traversal: false`): the standard ZSON benchmarks
score an episode on its spawn storey, so the search is confined to it --
seen staircases and observed drops are impassable (a drop is a surface
`RPTSettings.step_down_m`, 0.30 m, below the storey plane -- a knob since
2026-10-14, and not lower: the plane is the pose height and the floor the
camera sees lies up to 0.3 m under it in some scenes, so 0.15 m masked
Corozal/000142's own storey; the three-tread flight Darden/000034 walked
down, inside the tolerance, is the open item), no goal off
the spawn plane is accepted, and no staircase is a node, a fallback or a
climb (`core/planning/object_search/multifloor/spawn_floor_guard.py`; the multi-storey development
entrypoints lift the flag explicitly). Habitat target closing additionally uses
declared local NavMesh projection (`target_navmesh_projection` in run configuration).
This is an experimental multi-story algorithm, **not a claim of solved navigation**.
## Layout: one navigation core, one task per dataset
| Layer | Where | Owns |
|---|---|---|
| **Shared navigation core** | `methods/` (RPT*, room-search loop, frontier sweep, exploration fallback, perception with YOLO-World default), `habitat/` (RGB-D bridge, navmesh stair connectors), `evaluation.py`, `provenance.py`, `recording.py`, `visualization.py`, `dashboard.py`, `replay.py` | The algorithm, its settings, and the dataset-neutral execution/recording path. **No dataset loader, episode list, category table, scoring rule or benchmark profile lives here**, and `tests/test_runtime_boundaries.py` fails if a core module imports `gibson/`, `hm3d/` or a label table. |
| **GibsonTask** | [`gibson/`](gibson/README.md) | SemExp v1.1 episodes, FMM scoring maps, planar path accounting, sliding, success without STOP, its CLI and campaigns. |
| **HM3DTask** | [`hm3d/`](hm3d/README.md) | HM3D ObjectNav v1/v2 episodes and scenes, habitat-lab's view-point geodesic metrics, STOP-required success, published-geodesic agreement preflight, its CLI and campaign runner. |
Both tasks construct the same `RPTSearchPolicy` from the same `RPTSettings`,
feed it the same `HabitatRGBDSimulator`, and hand it to the same
`run_evaluation`/`run_benchmark`; what differs is only the `ObjNavEnv` adapter,
the label mapper and the `EvaluationSettings` each protocol declares. Fix the
algorithm once, in `methods/`, and both benchmarks pick it up.
**To set up and run Gibson end to end, start with [QUICKSTART.md](QUICKSTART.md)**;
for HM3D, with [hm3d/README.md](hm3d/README.md). This file is the architecture
reference for the shared core.

**To run the published Gibson val benchmark end to end, start with
[gibson/BENCHMARK.md](gibson/BENCHMARK.md)** (one script: services, preflight, the 1,000
episodes with lean records and live progress, the summary). [QUICKSTART.md](QUICKSTART.md)
is the step-by-step manual and the guide to the multi-storey development campaigns. This
file is the architecture reference.

## Exploration selection

### High-priority target closing

Both explorers yield **before any room/LLM/RPT* decision** when a target-category
YOLO detection reaches `target_closing.confidence` (default **0.50**) **and projects
to a coherent 3-D position** -- a confident box the depth sensor cannot place has
nothing a second frame can be checked against, and the twelve-step verifications
such boxes bought ("waiting for valid target depth" from 3.8 m, Ranchester attempts
4 and 5) released every time. The first such candidate latches ownership in
`VERIFY`; at least **two consecutive distinct frames** with coherent depth and the
same fixed 3D association anchor (0.50 m tolerance) are required to lock `CLOSE`.
**A confident box WITHOUT a usable depth begins a bounded BEARING approach
instead** (`bearing_start`, since 2026-10-21): the depth sensor returns nothing
past about 5 m and the detector finds the class well beyond that. Once the class
has been in the frame on two consecutive actions, the agent faces the box by its
image-space bearing and steps toward it, re-facing it on every frame it is seen
(`phase=target_bearing`, `target_measured: false` in the step record); the first
frame that projects with depth starts the ordinary `VERIFY` there, with its own
clock. The walk gives up -- without a rejection memory, nothing having been
placed -- after `bearing_max_actions` (16), after `bearing_lost_frames` (3)
frames without the class, when three steps move the agent nowhere, after
`bearing_max_walk_m` (3.0 m) walked toward the box (a box first seen within the
depth cap plus that distance is in range by then; one still depthless is
unmeasurable -- a television screen is a hole in the Gibson mesh -- or not where
the bearing pointed), or when the still-depthless box is `bearing_near_box_frac`
(0.30) of the frame's height tall, within reach and unpositionable, so that no
blind STOP is taken on it; and no bearing approach starts again within
`bearing_retry_m` (1.5 m) of the spot the walk began
(`bearing_starts` / `bearing_resolved` / `bearing_abandoned` / `bearing_refusals`
in the closing's diagnostics).
Duplicate boxes and repeated calls for the same frame do not count. Context
detections below this threshold cannot enter legacy target pursuit.

**Detect high, associate low.** A detector's confidence in one object swings with
the view: the Ranchester couch read 0.68 from the stair foot at one heading, 0.42
one 30-degree turn later with the same object centred, 0.52 clipped at the next --
and the agent's own centring turn broke the consecutive run on every cycle
(twenty-four frames with the couch in view, two releases). Once a candidate is
active, a box of the target's class that projects onto its anchor counts from
`track_confidence` (**0.30**); a weak box anywhere else, or before any takeover,
counts for nothing (`weak_resightings` in the diagnostics).

**What may start a takeover.** A box within `border_margin_px` (**8 px**) of any
image edge is a partial view — its depth centroid is unreliable, and the clipped
end of a bed reads as a sofa (the Ranchester couch-downstairs run of 2026-10-04
ended on exactly such an 84 px sliver) — unless it fills 30 % of the frame in
both dimensions, the big object at terminal range that no frame holds whole
(`clipped_box`). **A box cut only by the bottom edge is not a sliver** (since
2026-10-07): a level camera 0.88 m up sees the floor from about 1.4 m out, so a
LOW object at 1-2 m -- a toilet, a chair seat -- is cut by the bottom edge on
every frame while standing whole laterally, and its near edge and centroid are
on the object. The Allensville toilet run refused twenty-three such frames of a
0.97 toilet at 1.2-1.7 m, never counted two in a row and released the candidate
twice; a box touching only the bottom edge, with its top in the lower half of
the frame and at least 15 % of the frame in both dimensions, counts. Clipped
boxes cannot start or count
towards a lock; once locked, the approach tolerates spill-over. A box projecting
within `rejection_radius_m` (**1.0 m**) of a spot already released as unverified
on the same floor is ignored before a lock -- **unless it is seen from within
`rejection_min_range_m` (2.0 m)**: the memory stops the same far flicker from
restarting the takeover from the same place, and a close view is new evidence
(the Ranchester couch released at 3.8 m from the stair head was never
re-verified from 1 m, twice, until this) -- **or it reaches `override_confidence`
(0.80)** (since 2026-10-07): the detector's word at that level is not a flicker,
and the same toilet run refused nine frames of a 0.95-0.97 toilet in plain view
at 2.4 m because a lock the map had wrongly outvoted was remembered 2 m around
it. The override passes the memory of `unverified` and `contradicted by the map`
releases and the release cooldown; a spot released by a **failed inspection**
(the closing looked from close and saw nothing), a stalled approach or a
boxed-in spot still holds, and border clipping, incoherent depth and the map's
own contradiction are not overridden (`overrides` in the diagnostics; the
rejection memory records `why`). The 3-D association radius for the
consecutive-frame lock grows with range (`association_range_gain`, 0.15 m per
metre beyond 2 m: 0.77 m at 3.8 m), because the visible centroid of a long
object seen from far moves more than half a metre between two frames.

**The part of the object a frame shows moves along the line of sight** (since
2026-10-11; the toilets of the 1,000-episode validation run, 63 % of whose 148
failures had locked on and released a candidate). From 2 m a toilet's box holds
the whole object and the depth patch lands on the bowl (0.38 m up); from 1.3 m
the frame's lower edge cuts the box, the patch lands on what is left in view --
the tank -- and the centroid jumps 0.47-0.49 m BACK along the viewing ray (and
0.2 m up) with 2-3 cm of lateral error. Against a 0.50 m sphere about the anchor
that is 0.50-0.53 m: every 0.92-0.94 frame of the toilet the agent was standing
beside was discarded as another object, the inspection released it as "saw
nothing in any view" with the agent inside the success region, and a 2 m
rejection around the toilet followed. Three rules of `TargetClosing` answer it:

- **Association is a cylinder, not a sphere** (`_associated`,
  `association_depth_m` 0.60 m): `association_radius_m` across the ray and in
  height, that plus the object's depth -- 0.60 m or the anchor frame's footprint
  diameter, whichever is larger -- along it. Two chairs side by side stay two
  chairs (`depth_associations` counts what the sphere would have dropped).
- **A floor-cut frame is a sighting, not a position** (`_floor_cut_box`,
  `floor_cut_terminal`, `floor_cut_min_hidden_m` 0.05 m): when the box reaches
  the lower margin, the target is below the camera and the lowest point in view
  stands higher than the lowest point the lock has measured, the frame counts as
  fresh but is NOT fused into the 3-D memory (its centroid is the upper part's,
  a biased reading of the object's; fused, it walked the memory and the standoff
  measured from it back toward the tank frame by frame), and the terminal range
  is the smaller of the visible near edge and the planar range to the memory:
  the part under the frame -- bowl, seat -- stands in front of the part in view
  for the furniture of this benchmark, and the memory is the centroid the
  approach's standoff was measured from (`floor_cut_frames`,
  `floor_cut_measurements`; `floor_cut` in the STOP's info).
- **Below the field of view, without LOOK actions, the memory completes the
  STOP** (`_below_the_frame`, `blind_cone_stop`, `blind_cone_max_missing` 6,
  `blind_cone_min_fraction` 0.15): the Wiconisco toilet, 0.47 m high, showed
  under 40 px of its rim from 0.9 m, the detector read 0.12-0.18 and no yaw
  sweep brings back what no yaw puts in the frame. When the geometry (the
  camera's height, pitch and lower edge; the target's measured top and
  remembered range) predicts fewer than 15 % of the frame's rows of the target
  in view -- the sliver `clipped_box` will not accept -- with its bearing inside
  the horizontal field, the target seen fresh within six actions and the memory
  within the terminal range, the closing STOPs: the missing box is explained,
  and the one action that would bring the object back does not exist. The rule
  is off whenever the action spec has the tilt pair -- there the dynamic pitch
  controller (below) is its counterpart (`blind_cone_stops`).

**The context check** (`core/planning/object_search/perception/target_context.py`, `context_check`, since
2026-10-07). A detector's confidence is a score over the crop; it knows nothing
about the room the crop was taken in or how high the surface it fired on is.
Allensville/2 and Newfields/2 of the 5x3 benchmark STOPped on a kitchen counter
front read as `bed` from 0.55 m -- its supported points 0.67-0.70 m above the
agent's base where every real bed of the recording measured 0.11-0.52 m, inside
a room the map had made a STRONG kitchen from its refrigerator and oven -- after
two frames from one spot. A candidate is **suspect** when its footprint lies in
a room whose strong type is one the target is not searched for in
(`room_priors.IMPLAUSIBLE_ROOMS`, `room:kitchen`) or when its measured centroid
height contradicts its class (`CLASS_HEIGHT_BANDS`, calibrated on the Allensville
recordings: a bed's or a toilet's surface under 0.60 m above the base,
`height:0.69`). A suspect box counts `context_penalty` (0.70) of its confidence
toward the start threshold (0.60 x 0.70 is no takeover; 0.75 is), and its lock
needs `context_confirmation_frames` (4) consecutive frames from at least two
viewpoints `context_baseline_m` (0.30 m) apart -- a turn in place is not a
second viewpoint (`suspect`, `context_rejections`, `suspect_locks_held` in the
diagnostics and the HUD). It is a penalty, not a refusal: a bed in a room the
partition merged with a kitchen is still locked, from two places.

**Before the lock, ownership is provisional.** A fresh candidate within one
turn of the image centre is *stepped toward*, not centred (since 2026-10-05):
the box stays in the frame after a step and the next frame can be the
consecutive one, where the centring turn moved the box across the image and
the detector dropped it on the other side, four cycles running, at a chair
3.5 m off; a candidate farther off-centre is faced first. **A candidate too
close to step toward gets a frame that keeps its box whole**
(`verify_keep_in_frame`, since 2026-10-07): the hold that used to follow was
already satisfied -- idle -- and the headless agent executes an idle result as
a TURN, which moved the centred Allensville toilet out of the frame at 1.18 m
(action 77) and reset the count. **The pitch follows the target's elevation**
(`TargetClosing._elevation`, `elevation_band_m` 0.15 m): a target whose
measured 3-D centroid stands below the camera -- or, without one, whose box
centre bears below the horizon once the camera's own pitch is taken out -- is
looked DOWN at, one above the camera (a wall-mounted television, the top of a
wardrobe) is looked UP at, and a box cut by the bottom or the top edge says
the same thing; never a fixed direction (a hardcoded LOOK_DOWN drove high
targets out of the frame). Level with the camera, the LOOK whose predicted
shift (`fy * tan(tilt)` pixels) leaves the box inside the frame margins, down
before up; failing both, the TURN whose shift (`fx * tan(turn)`) keeps it,
toward the box's side before away (`verify_step` and `elevation` in the command
info). If `max_verify_steps` (12) pass
without the lock, `release_unverified` (default **true**) hands control back to
exploration in phase `RELEASED`, records the anchor in the rejection memory and
clears the legacy target hint; the room/stair search resumes where it was
interrupted. **What the takeover refuses, the legacy pursuit does not walk
after** (`TargetClosing.refuses_far_candidate`, read by `PerceptionCycle.fuse`):
a far landmark within the rejection memory, one seen during the release
cooldown, or one seen from a spot given up for want of a path, sets no legacy
target hint (`refused_by_takeover` on the projection row) -- the second Hanson
fly walked nine actions toward a chair the takeover had released a frame
earlier, and the move restarted the warm-up where it ended. `release_unverified: false` restores the historical behaviour, where
the exhausted verification raises the recordable method error. **After the lock**
neither a timeout nor a rejected route releases ownership; the four releases a
locked target does have are the ones below -- an inspection that saw nothing from
close, a map whose class vote at the anchor outvotes the lock, a spot with no
path out of it, and an approach that goes nowhere (`approach_stall_actions` CLOSE
actions inside `approach_stall_m`: A* finds a path, the follower pushes into
something the map does not show) -- and episode reset. Room reasoning, stair decisions, doorway peeks,
glances, FALCON masks and exploration fallback remain suspended while the takeover
owns the action (a glance in force is aborted the action it starts); the landmark
map keeps voting on the frames the takeover sees, so the map release can fire,
while the legacy target evidence stands down.

The closing sub-policy reuses coherent RGB-D backprojection with the actual camera
pitch. It samples standoff footpoints around the observed target, projects them
onto Habitat's local NavMesh when bound, and uses existing collision-qualified
observed-grid A* to approach. **The selected 3D footpoint and route persist through
occlusion**: missing detections neither cancel the path nor invoke a scan. Fresh
associated observations refine the stored target with a 50/50 update; an XY shift
over `refine_distance_m` (0.15 m) since the goal was selected triggers a refinement
-- and a refinement **keeps the standoff goal when it still ends inside
`terminal_distance_m` of the refined memory** (since 2026-10-11, `TargetApproachPath.kept`;
`goal_kept` in the diagnostics): re-selecting a bearing-relative ring about the new
position is a different set of points, each needing a navmesh projection and a path,
and Collierville/000194 traded a reachable goal for a ring with none and released the
real television as "no safe path from here" the moment the along-ray association made
its memory 0.11 m more accurate. A kept goal that a slow drift carries past the terminal
radius is re-selected the action it falls out. Small refinements do not move the endpoint
every frame. **The ring widens before the spot is given up, and a stale goal is kept over
no goal** (since 2026-10-15, `target_path.widened` / `stale_kept`; `widened_standoffs`,
`stale_standoffs_kept` in the diagnostics): the standoff ring stands `goal_tolerance_m`
inside the terminal radius -- 0.70 m about the 3-D memory, a point on the object's NEAR
surface -- and a big object has no floor there. Collierville/000009's 1.5 m television on
its cabinet put every 0.70 m point inside the cabinet, against its front or behind it, in
both planners, from 3 m away, three times in one episode, and each correct lock was
released as "no safe path from here" with the television in view; the same episode also
dropped a standoff 1.01 m from the refined memory -- with a seventeen-waypoint route to
it -- as stale, for a ring that had no route at all. Now, when neither planner reaches
the standoff ring, wider rings are tried in the front arc out to the terminal radius; a
goal there is reached within the arrival tolerance up to one forward step outside the
terminal range, which the inspection's step rule closes; and a goal a refinement makes
stale is dropped only for a ring point that has a route, and kept while it lies within
the terminal radius plus the tolerance when none has. **From such a standoff the closest
reachable approach is terminal** (`closest_approach_stop`, `closest_approach_stops`):
with the target fresh in view within one step of the limit, once the inspection's steps
are spent or a step is blocked (the cabinet), the agent stands as close as the floor
allows and STOPs -- the release that followed before gave up a target in plain view.
Collision, blockage and
progress validation remain active; persistence is not permission to cross obstacles.
Remote or other-floor snaps and unreachable paths
are rejected. The optional simulator callback contains no target annotations,
success flag or evaluator DTG; NavMesh geometry is nevertheless privileged and
is declared in the frozen configuration. Non-Habitat callers use observed-grid A*
without NavMesh projection, explicitly reported as such.

A* owns heading during transit, including doorway/corner turns that put the target
outside the FOV. Bbox yaw servoing is confined to terminal inspection so it cannot
fight the route. The camera is level in transit; **inside `look_down_distance_m`
(1.30 m) the approach carries the target's own pitch** (since 2026-10-07, the same
`_pitch` the inspection uses): one tilt step at least in the direction of the
target's elevation -- down at a toilet, up at a wall-mounted television -- so the
terminal frames hold the object whole. **Since 2026-10-11 that geometric controller is
the fallback**: within `bbox_pitch_distance_m` (1.50 m) with a fresh box, the camera
pitch is the image-space servo's (next section), and the geometry decides only on a
frame without a fresh box. At `terminal_distance_m` (**1.0 m**),
forward motion stops and inspection remains stationary. The inspection pitch is
the one the target's **measured height** predicts -- a toilet 0.45 m up is low, a
plant in a metre-tall planter is at eye level and gets no look-down (until
2026-10-05 "a potted plant is low" by label sent the camera 30 degrees down at
exactly such a plant for 24 actions), a television above the camera gets LOOK_UP;
missed detections
trigger bounded pitch-up/down and yaw views referenced to the stored target bearing,
not accumulating turns from the current yaw. The predicted pitch is where to
*look* for a target not in view, never a condition on having seen it: a fresh,
aligned sighting within range STOPs at whatever pitch it came at (the Hanson
toilet projected at 60 degrees where its height predicted 30, and twenty actions
of LOOK_UP / LOOK_DOWN followed the sighting before the budget STOPped). The camera owner is `TARGET_CLOSING`,
so level-view restoration cannot override inspection. The original no-tilt Gibson
protocol cannot emit LOOK_DOWN; multi-story development has tilt-enabled actions.
Turns retain the protocol's fixed increments (30 degrees in Habitat) and its
half-turn dead band, not arbitrary micro-turns.

STOP requires a **fresh**, associated detection, completed verification, an aligned
box (its centre within half a turn, or a box that spans the image's centre column)
and a **freshly measured** horizontal range -- the near edge of the detected
surface -- within the terminal radius plus `range_tolerance_m` (**0.05 m**, for
pitch-dependent projection noise), at whatever pitch the sighting came. No STOP
from remembered distance alone is allowed, with three earned exceptions: the
approach history (below), the support surface (below), and -- without LOOK
actions -- a target the geometry puts under the frame, seen fresh a few actions
before (`_below_the_frame`, above). For a box the frame's lower edge cut, the
"freshly measured" range is the smaller of the visible near edge and the range
to the 3-D memory (`_floor_cut`, above): the box is fresh, the range is not
read from the part of the object that happens to be in view. The pitch the target's height predicts
is where the inspection LOOKS for it when it is not in view; a fresh off-centre
sighting is centred at the pitch it was seen at. An inspection the filtered
estimate entered while every fresh frame measures the surface beyond the limit is
ended and the approach resumed (`inspection_resumptions`). This terminal radius
replaces legacy `stop_distance_m` for the closing sub-policy. Verification is
bounded at 12 actions (`max_verify_steps`, released as above), terminal inspection
at 24 (`max_reacquire_steps`), and total closing at 160 by default.
Occlusion during transit has no separate timeout. **The terminal inspection on a
big object at close range** (the Ranchester couch, 2026-10-04, actions 172-196):
a fresh box that *spans the image centre column* counts as aligned whatever its
centre says (a clipped box's centre is not the object's centre, and centring it
turned the agent off the couch twelve times); the terminal range is the
**near edge** of the freshly measured surface (`range_near_m`, the 10th
percentile horizontal distance of the supported depth points) or the centroid,
whichever is nearer -- the benchmark's success radius is measured to the
object, not to its middle. **The box pipeline reads around a surface standing in
front of the object inside its box** (since 2026-10-15, `core/mapping/objects/box_occluder.py`;
`occluder_depth_m` / `occluder_px` in the projection rows). The box's range is a low
percentile of the depth inside the shrunken box -- the near side of the object,
robust to the background a box overhangs onto -- and a surface IN FRONT of the object
wins that percentile the moment it covers a third of the box's centre: in
Collierville/000018 the top rail of a seat back at camera height stood 1.3 m from the
agent between it and a television 4.1 m away, the television's box held 79 %
television and 20 % rail, the rail was read as the range, the closing walked to 0.8 m
of the rail and STOPped 3.3 m from the television -- the same five Collierville tv
episodes, at the same actions, in two 1,000-episode runs. The test asks the whole box
four measured things: a second surface stands at least 0.5 m (or 30 % of the near
range) behind the chosen one; that far surface is one connected surface spanning at
least 80 % of the box's height AND width (what a detector's box outlines when it
outlines a flat object -- a couch's backrest, a toilet's tank, a bed's headboard span
the width and not the height); it covers more of the box than the near surface; and
the near surface leaves the top quarter of the box to it (a floor-standing chair or
plant in front of a wall reaches the top of its own box, and the wall is never read as
the object). When all four hold, the near surface's pixels are set aside and the box is
read again without them. Depth only, no class; the mask pipeline (`segmentation`)
answers the same question from the object's own pixels and is untouched. And when the inspection budget runs out after the
locked target *was* seen fresh within terminal range from this spot, the
closing **STOPs** (`stop_on_exhausted_inspection`, default true) instead of
raising. Exhaustion of a **locked** target that was *never* seen from the
inspection spot means the lock was wrong -- a sofa from four metres that is
nothing from one (the Ranchester same-storey run of 2026-10-04 ended as an agent
error on exactly that, with the real couch 2.85 m away) -- so since 2026-10-04 the
candidate is **released** (`release_on_failed_inspection`, default true): its
anchor goes into the rejection memory with twice the radius
(`failed_inspection_radius_factor`), the release action is one turn in place,
and the search resumes on the next action with the spot excluded from the
takeover and from the target-landmark nodes. The release comes as soon as every
inspection view (three yaws, up to three pitches) has been tried once with
nothing seen -- not after the 24-action budget -- and, under the library's
class-voting map only, *at once* when the map's class vote at the anchor outvotes
the lock (`map_releases`: the sofa from four metres that the map has since
confirmed as a bed from two; this runtime's map never merges classes since
2026-10-07, so the vote and the release read zero here). `release_on_failed_inspection:
false` restores the historical recordable method error (**not**
`ObjNavInternalError`: the harness finalizes failed metrics; its forced STOP is
not a successful policy STOP and appears as `termination=agent_error`).
**The STOP overshoot** (since 2026-10-14; `stop_overshoot_steps` 1,
`stop_overshoot_max_steps` 2, `stop_overshoot_within_m` 0.90,
`stop_overshoot_min_range_m` 0.45, `stop_overshoot_max_actions` 5;
`TargetClosing._terminal`). Every terminal reading is biased short of the
object's footprint -- the near edge of a box with floor in it, a centroid fused
from the near surfaces -- and the evaluator measures to the footprint:
Markleeville/000188 STOPped on a 0.84 m near edge 1.27 m from the toilet (DTG
0.27, SR 0). Reading a small object's range from the fused memory instead was
tried and flipped CLOSE and INSPECT on alternate frames for 25 actions (see
`_fresh_measured`), so the remedy is after the decision, not in it: once a STOP
is decided, on whichever of the seven paths of `_inspect`, the closing walks
one more MOVE_FORWARD toward the 3-D memory -- and a second only while, after
the first, the memory still stands farther than 0.90 m from the agent -- and
STOPs then (phase `ADVANCE`). No frame is consulted meanwhile -- `observe`
short-circuits, the memory stands, nothing is re-verified -- and the STOP
follows whatever the detector shows. The memory, not the near edge, judges
the second step: the memory is about the footprint's near face (000188: 1.25
m where the footprint's disc began at 1.27), so a memory inside 0.90 m is a
tenth inside the evaluator's radius, while the near edge is what decided the
STOP and in 000188 undercut the memory by 0.41 m where the eight STOPs that
succeeded agreed with it within 0.14. A step is taken only while the memory would still stand
0.45 m beyond the agent after it and the deciding frame's near edge (when it
measured one) at least the body radius; the agent must face within a quarter
turn of the memory, and beyond one turn of it the converter turns toward the
memory first (turns cost no path); a step the converter reports blocked ends
the stage at once (`note_blocked`, from the policy's `notify_blocked`;
`note_action` counts the steps), and the action bound ends it without the
hooks. `stop_overshoot` in the command info and the diagnostics says how it
ended; `overshoot_commands` counts the steps asked for. Verified in turn on
the same ten episodes (round i, `…143259Z`: SR 0.80 / SPL 0.366): two steps
always (`…155701Z`) SR 0.90 / SPL 0.450, Markleeville/000188 from DTG 0.27 to
SPL 1.0 but every other STOP 0.01-0.07 SPL dearer; one step always
(`…162832Z`) SR 0.80 / SPL 0.357, 000188 STOPped 3 cm outside the disc. On
those nine deciding frames the rule above takes the second step twice (000188
and Collierville/000023, memories at 1.25 and 1.19 m) and nowhere else --
verified in `…170224Z`: SR 0.90 / SPL 0.456 / DTG 0.59 m, the best of the
four.
`stop_overshoot_max_steps` 0 restores the STOP on the deciding frame, which is
how the shared test fixture runs the decision tests; 1 is one step and never
a second.
**A locked target with no safe path** (since 2026-10-05). Hanson/000002 spawned
beside a bed with the potted plant in view 4.4 m away and spent 159 actions
turning on "no safe target path": the floor under the camera's blind radius
(a level camera 0.88 m up sees the floor from about 1.4 m out) was unknown,
unknown is impassable, and the lock waited for A* until the 160-action
closing bound ended the episode as an agent error. After
`footing_after_steps` (2) pathless actions, while at least
`footing_unknown_fraction` (0.25) of the cells within `footing_radius_m`
(1.2 m) of the agent are unknown, the closing asks the camera
controller for a **footing sweep** -- one LOOK_DOWN to `footing_pitch_deg`
(30: the image's lower edge on the floor 0.47 m out, its upper edge just
above the horizon), a full circle, one LOOK_UP; phase `FOOTING`, the
takeover keeps the camera, the target stays in the record -- which maps
the disk and lets the path exist or show that it does not. The sweep is
cut short the action a path exists (one LOOK_UP, then the approach, level):
the second Hanson fly of 2026-10-05 found its path at the third turn and
kept the sweep's 30 degrees for the whole approach and the first
inspection, where the plant's foliage never projected. If
`release_after_footing_steps` (6) more pathless actions follow, the lock is
**released without a rejection** (`boxed_releases`, `last_release="no safe
path from here"`): the target was never disproved, the spot was. The spot
is remembered as boxed in, and no far candidate may start a takeover from
within `boxed_in_radius_m` (1.0 m) of it, so the search -- which can walk --
carries on and the same plant is locked again from somewhere a path exists.
`footing_sweep: false` drops the sweep alone (a protocol without LOOK
actions never has it); the release after the bound stands either way.
**Before any of that, the ring is planned through the unknown** (since
2026-10-14, `through_unknown`, `through_unknown_cost` 8): the floor between a
corridor and the toilet seen through its doorway is routinely unknown -- a
level camera sees the floor from 1.4 m out -- and Collierville/000023
(action 375) released a lock on a toilet in plain view at 2.8 m as boxed in,
then refused its 0.82-0.88 re-sightings from the spot. When no route to the
standoff ring exists on the known floor, the same ring is planned with unknown
cells passable at eight times the cost of open floor (the policy's
`optimistic_planner`, the strict planner's twin); the committed route is
re-checked against the map as it fills in like any other, an obstruction
re-plans, a bump marks the cell. `through_unknown_routes` counts them and the
command carries `through_unknown: true`; the footing sweep and the boxed-in
release are what happens when even that finds nothing.
All thresholds are configurable under `RPTSettings.target_closing`; the phase,
persistent lock, visibility, target xyz, projected goal, refinements, occluded path
steps, bbox, failures, release counts (`releases`, `inspection_releases`,
`boxed_releases`), footing sweeps, boxed-in spots, border
rejections and the rejected spots with their radii
are included in episode/frame diagnostics and the HUD.

#### The seen toilet that was not locked, and the reached toilet that was not STOPped on (since 2026-10-14)

Four rules from the per-step traces of the toilet campaign of 2026-10-13
(`runs/benchmark_toilet_15episodes_20261013T233505Z`, SR 0.80), where
toilets in plain view were released and a toilet at the terminal range was
turned at for twenty-four actions:

- **The rejection memory judges a START, not a re-sighting.** `_candidates`
  applied the unverified/cooldown/boxed-in memory to every frame before a
  lock, the active candidate's included: Markleeville/000188 (actions 34-45)
  began a takeover on a 0.84 override past an `unverified` memory and had
  every 0.60-0.79 re-sighting of the same toilet dropped by that memory, so
  the consecutive count never reached two. A box ASSOCIATED with the active
  candidate's anchor counts whatever the memory holds
  (`resighting_exemptions`).
- **A box in view without a usable depth is a glimpse, not an absence.** At
  5 m the projection has no coherent depth, the frame was read as "nothing
  seen", the count reset and the sweep turned AWAY from the centred box
  (Markleeville/000188, actions 20-31: two whole verifications turning in
  place at a toilet in view every other frame). A box of the target's class
  within one turn of the anchor's bearing (`_on_anchor_bearing`) carries the
  consecutive chain across the frame, is faced, and once faced is stepped
  toward (`verify_step="towards the glimpsed candidate"`); the step does not
  run the verify clock, for at most `max_verify_approach_steps` (24) steps
  per candidate (`glimpse_steps`, `verify_pauses`).
- **The inspection steps when the range is a step out, and does not oscillate.**
  Wiconisco/000193 (actions 476-499): the near edge at 1.0505 m against the
  1.05 m limit, the box 16-17 degrees off the centre on alternating sides --
  just outside `turn_action`'s half-turn dead band at both headings -- and
  TURN_LEFT/TURN_RIGHT to the step limit. A fresh range beyond the limit by
  less than one forward step, faced within a turn, is one MOVE_FORWARD
  (`inspect_step`, `inspection_max_steps` 3 per inspection); and a centring
  turn that would undo the inspection's turn of the previous action, with
  the error under a whole turn, is read as aligned (`dead_band_alignments`).
- **A candidate first read at `backoff_trust_confidence` is not backed off
  from, nor one the detector had been firing on.** The manoeuvre and the
  verification after it cost 23-25 actions in both Collierville toilet
  episodes and hid one toilet behind its jamb; a 0.92 box is the detector's
  word and is locked where it was seen (`backoff_trusted_skips`). And
  "sudden" means the agent turned and there it was: a class that stood in
  the frame on `backoff_persistence_frames` (2) of the
  `backoff_persistence_window` (6) actions before the takeover -- border-cut
  boxes that could not start it (Collierville/000023 again: 0.95, 0.79,
  0.79, 0.76 on four cut frames, the takeover on the fifth at 0.84, a
  25-action retreat and a 2 m rejection) -- is persistent, not sudden, and
  is verified where it stands (`backoff_persistence_skips`). A weaker,
  truly sudden first frame still makes the manoeuvre, and a candidate that
  reaches the trust confidence during it is still given back without a
  rejection when the wider perspective misses it (above).

And the map's own evidence is not hidden by a far flicker: a confirmed
landmark of the target's class is offered as a node (`T<n>` on the HUD)
unless a release **disproved** it from close (`TargetClosing.landmark_refused`;
`DISPROVED_RELEASES`), where until now any release within range -- a
twelve-step verification that failed from 4 m -- removed the toilet from the
order and the display alike ("no `O`, as if the object were not there").
**A target landmark seen ONCE at `landmark_sighting_confidence` (0.50, the
closing's start threshold) is a node too** (`opening_nodes.target_landmarks`;
the perception cycle keeps each landmark's best confidence in
`_landmark_best_conf`): the landmark map confirms an object on its second
observation, which a far glimpse through a doorway does not get --
Wiconisco/000014 saw its toilet once at 0.54 from 3.2 m, the verification
lost it after one step, and with no node on the map the search went
elsewhere for 65 actions. Nor
by the floor-support guard: `PerceptionCycle._floor_support` refused every
projection whose lowest sampled point stood more than 0.18 m above the
storey's floor level -- a toilet seen from a metre, its base under the
frame's lower edge and the floor in front inside the camera's blind zone,
sampled its bowl at 0.2-0.45 m and never became a landmark. A storey is 2.5 m
and more away; the band is 0.60 m now. And **the sweep for a candidate out
of view is one circle, not the budget** (`verify_sweep_max_actions` 6,
`sweep_releases`): the headings 0, +30, 0, -30 take four or five actions,
and a candidate none of them shows -- the toilet one step had put behind a
jamb, Wiconisco/000014 actions 231-241 -- is not brought back by the same
headings again; the verification is released as at its budget and the rest
of the budget is the search's (which now has the `T` node to walk to).

#### Close range: the 3-D memory, the approach history and the sudden-proximity manoeuvre (since 2026-10-08)

**The 3-D memory** (`core/mapping/tracking/target_memory_3d.py`). The target's XYZ centroid is saved the
moment a takeover starts and fused, precision-weighted, with every frame that projects
onto it: a frame from `r` metres enters with sigma `0.03 + 0.03 r` (depth noise and the
visible-part shift of a big object both grow with range), so the near frames of the
approach dominate and the estimate's standard error shrinks frame by frame (`memory` in
the diagnostics: `xyz`, `frames`, `sigma_m`, `spread_m`). The first centroid stays the
association anchor. Everything else in the closing reads the memory -- the elevation the
pitch follows, the range the approach and the terminal test measure, the spot a support
surface must sit at -- which replaced the former 50/50 update.

**Two close-range situations look alike to the detector and are opposites to the
closing** (`core/planning/object_search/closing/approach_history.py`, the ledger of the planar path walked since the
takeover began and the range it closed):

- *Extended approach history* -- at least `approach_history_m` (3 m) walked and
  `approach_history_closed_m` (1 m) closed: the target was tracked and walked toward
  over metres, and inside the terminal range a cropped box, a detector drop or a budget
  run down is **not** a reason to release it. A fresh in-range sighting STOPs whatever
  its alignment (the benchmark measures range, not heading), and after
  `history_stop_after_missing` (3) consecutive terminal frames without a fresh box the
  closing completes on the 3-D memory and STOPs -- the one deliberate exception to "no
  STOP from remembered range alone", earned by the path; an inspection that would
  release from beyond the terminal range steps closer instead (`history_holds`,
  `history_stops`).
- *Sudden close proximity* -- a candidate first seen inside `sudden_proximity_m`
  (1.3 m) with no history: the agent turned and there it was, a single viewpoint, and a
  single viewpoint at 0.7 m is exactly where a counter front read as a bed. Before the
  lock the **backing manoeuvre** (`core/planning/object_search/closing/target_backoff.py`, phase `BACK_OFF`) takes
  the agent `backoff_steps` (3) forward steps back into space it knows to be clear --
  its own trail on this storey, the observed grid, the floor guard, collision-qualified
  A*, the NavMesh when bound -- **and that the observed grid shows the target from**
  (`BackoffManeuver._sees`, since 2026-10-14: a Bresenham ray on the known occupancy from
  the spot to the target, stopped 0.45 m short of the target's own cells; a spot the door
  jamb hides the toilet from is not a wider perspective -- Collierville/000036 backed off
  0.5 m from the goal viewpoint to one, saw nothing, disproved the candidate and lost 380
  actions) -- turns it to face the remembered coordinates and resets
  the consecutive-frame count, so the lock needs fresh frames from the wider
  perspective; then the approach and the terminal inspection follow as usual. The
  manoeuvre is bounded (`backoff_max_actions`, 20), its actions do not count against the
  verification budget, and with no clear space known it is skipped and recorded
  (`backoffs`, `backoff_skips`, `backoff.skipped`): verification proceeds in place as
  before. `backoff: false` disables it. Two rules keep it from becoming a loop
  (Markleeville 2026-10-08: eight manoeuvres on one table read as a bed from 0.9 m): the
  manoeuvre is made **once per object** -- a later close sighting on an anchor already
  backed off from is verified in place (`backoff_repeats`) -- and a candidate the wider
  perspective never re-saw is released as `unverified from a wider perspective`
  (`wider_releases`): the failed-inspection radius, no confidence override, and refused
  again from twice `boxed_in_radius_m` of the spot -- **unless the detector read it at
  `backoff_trust_confidence` (0.92) or better** where it was first seen (since
  2026-10-13): then the retreat spot, not the candidate, is what failed, and it is given
  back without a rejection (`trusted_releases`, `unverified from a wider perspective;
  trusted, not disproved`) so the next frame of it re-takes the closing. Collierville/000036:
  a toilet at 0.93-0.95 in two frames, 1.5 m inside its doorway, backed off from, hidden
  by the jamb from the retreat spot, released as disproved with twice the radius -- and the
  toilet landmark on the map 2 m away was never a node again; the table read as a bed at
  0.80-0.90 (Markleeville) is disproved as before. The retreat arrives by the converter's
  own tolerance (plus half a step, or three idle actions while facing the goal), so it
  does not spin on the spot. Two more rules from the re-fly: the **second `unverified`
  release of one anchor escalates** the same way (`unverified again`, `repeat_releases`:
  the override exists for far flickers, and 35 takeovers of one surface in 450 actions
  were the same evidence 35 times), and every release that **disproved its candidate
  from close** (`DISPROVED_RELEASES`: the wider perspective, the second look, an
  inspection that saw nothing) binds the legacy target pursuit at any range
  (`refuses_far_candidate`, `disproved_refusals`) -- Markleeville/000000 re-flown: the
  closing gave the table back at action 41 and the legacy pursuit STOPped on it at 50.

**Proactive pitch, and no step that loses the box** (Wiconisco toilet, 2026-10-08:
bottom-cut box at 1.32 m, one MOVE_FORWARD, gone -- four times). Every approach step and
every verification step toward the candidate carries `_approach_pitch`: the pitch the
memory predicts for *after* the step (atan of the height difference over the range one
step closer, quantised to the tilt step, a full tilt at least inside the close band), so
a low target is stepped toward looking DOWN from the moment the predicted angle is
nearer a tilt step than level, and an elevated one looking UP, all the way in. The
published Gibson protocol has no LOOK actions at all; there `_step_keeps_box` predicts
the box's rows after the step (and `_verification_view` the vertical stretch a TURN
gives a low box) and a verification step that would push a bottom-cut box out of the
frame is withheld (`no_step_holds`) -- the lock comes from the next frame where the
agent stands. And since the benchmark measures to the object, a **fresh sighting whose
near edge is inside the terminal radius is terminal evidence wherever the centroid is**:
the inspection starts, and STOPs, from there, where a low object is still in the frame;
a fresh *centred* reading beyond the limit resumes the approach rather than spinning
through the budget.

**Support-surface resilience** (`core/mapping/objects/support_surface.py`). At close range a mounted
or supported object leaves the frame before its support does -- the television on a
dresser becomes the dresser. A box of a `support_surface_classes` label (cabinet, desk,
dining table, shelf, ...) whose coherent centroid lies under the memory's footprint and
at or below its height is spatial evidence that the tracked thing is still there
(`support_sightings`): the inspection is not released as "saw nothing" while that
evidence is fresh (the dynamic pitch brings the object back into the frame), the map's
vote for the surface is not a contradiction, and once the inspection budget is spent
with the evidence fresh and the memory inside the terminal range the closing STOPs
(`support_stops`). It never counts toward the lock or the terminal test on its own.

**Dynamic pitch during close verification.** The verification step toward a candidate
inside `look_down_distance_m` carries the memory's elevation (`_pitch`): a low target is
stepped toward looking DOWN, an elevated one looking UP, as the approach and the
inspection already did; the face turn that ends the backing manoeuvre carries it too.

#### The box in the frame drives the pitch, and the STOP is taken on a whole, confident box (since 2026-10-11)

**The servo needs the LOOK pair, and the Gibson protocol has it since 2026-10-13**
(`gibson/protocol.py`, `camera_tilt`, protocol `/2`): under the published four-action set
of `/1` none of the rules below ever fired -- the fifteen toilet episodes of the campaign
recorded that day show `LOOK_UP 0 / LOOK_DOWN 0` and `bbox_pitch_corrections 0` in every
one, nine of them locked approaches -- and the tilt-less substitutes (`_step_keeps_box`,
the blind-cone STOP) stood in. A LOOK is an action against the same budget and moves
nothing; see `gibson/README.md` for the deviation as recorded in the frozen configuration.

The geometric controller above reads the target's *memory* -- its fused 3-D height
against the camera's -- and the memory is biased exactly when the pitch matters most: a
frame cut by the lower edge lands its depth patch on the upper part of the object (the
toilets of the 1,000-episode run). Every terminal STOP of the nine inspection recordings
fired on a box cut by the lower edge: at a metre, a level camera 0.88 m up cuts any
floor-standing object. `core/control/camera/box_pitch_servo.py` is the **image-space servo** that reads
the box itself, with no class, no height and no memory involved (`TargetClosing._servo`,
`bbox_pitch`, `bbox_pitch_distance_m` 1.50 m, `bbox_pitch_region` 0.25):

- **Which frame.** On the approach the verdict is read on the frame the next step will
  produce (`box_pitch_servo.box_after_advance`: every edge row is a ray at a depression below
  the horizon, met at the fresh surface's near-edge range, and re-projected from one step
  closer), so a box about to leave through the bottom is looked down at *before* the step;
  in verification and inspection it is read on the frame as it is.
- **The verdict** (`box_pitch_servo.vertical_fix`). Cut by BOTH the top and the bottom margins:
  the object fills the frame, nothing helps. Cut by the bottom margin alone: LOOK_DOWN by
  the rounded steps (one at least) provided the top edge stays inside afterwards -- a bottom
  cut traded for a top cut is no gain; symmetrically LOOK_UP for a top cut. Whole, with its
  centre in the lower `bbox_pitch_region` of the rows: LOOK_DOWN provided the whole box stays
  inside and its centre does not land in the upper region (fewer steps are tried first);
  symmetrically for the upper region. Otherwise -- the centre band -- the camera stays. The
  band is never narrower than one tilt's rounding dead band (`fy * tan(tilt / 2)` rows), so
  the servo cannot hunt; a tilt past the pitch limits is cut back to what they allow.
- **Who has the camera.** With a fresh box within `bbox_pitch_distance_m` the servo alone:
  a geometric tilt forced against a centred box would hunt against it (the forced one-tilt
  rule of `_pitch` inside `look_down_distance_m` says LOOK_DOWN at a toilet whose box is
  already centred). Without a fresh box -- an occluded approach step -- the geometry decides
  as before; beyond the distance the approach is level as before; without the LOOK pair
  (the published Gibson protocol) the servo does not exist and the tilt-less rules above
  stand (`_step_keeps_box`, `_floor_cut`, `_below_the_frame`). **The geometry's own angle is
  quantised with hysteresis** (`pitch_hysteresis` 0.15, `_quantise_pitch`, since 2026-10-14):
  the tilt step the camera is at is kept while the wanted angle lies within 0.65 steps of it.
  Plain rounding chattered -- a 0.39 m toilet 2.2 m from a camera 0.9 m up wants 15.0 degrees
  against the 30-degree step, the memory's height moves by millimetres between frames taken
  at the two pitches, and Darden/000034 spent its last nine actions on LOOK_DOWN / LOOK_UP
  without a step (the converter serves a pitch change before any move). The verification frame
  (`_verification_view`) asks the servo first and keeps the `look down at a low target` /
  `look up at a high target` wording, now read off the frame rather than the memory; the
  elevation-driven looks remain for a frame the servo has no verdict on.
- **The terminal STOP, with the LOOK pair.** Before the STOP is taken on a fresh in-range
  frame (`_before_stop`): a box under `stop_confidence` (0.50, the takeover's own start
  threshold -- detect high, associate low, STOP high) is held for the one frame that keeps
  it whole, at most `weak_stop_max_holds` (2) times in a row, then the STOP goes ahead on the
  evidence there is -- or at once when no such frame exists (a box filling the frame); then
  a box the servo can re-frame earns ONE centring tilt per inspection (`stop_centring`), and
  the STOP is taken on the next frame: on the whole box if the detector shows it, and on the
  previous frame's evidence if it does not (`centring_step`) -- the tilt may cost an action,
  never the STOP (the Hanson toilet's twenty actions of LOOK_UP / LOOK_DOWN after a fresh
  sighting are what the bounds are for). Not while the box still needs a yaw turn: the
  converter tilts before it faces. The STOP's info carries `confidence`, `bbox_region` and
  `depth_valid` (never NULL on this path: a box without a coherent depth projection is never
  fresh). Counters: `bbox_pitch_corrections`, `stop_centring_tilts`, `weak_stop_holds`;
  `bbox_region`, `bbox_pitch_steps` and `box_drift_rows` (the fresh box centre's row change
  since the previous consecutive frame) in the diagnostics.

Verified end to end through the real converter (`tests/test_target_closing.py`: the
approach, the verification frame, the centring tilt and its safety net, the confidence
hold, the Gibson path unchanged) and on the pure maths (`core/control/camera/tests/test_box_pitch_servo.py`).

- `--explorer frontier`: the existing observed-frontier baseline (default).
- `--explorer falcon`: bounded FALCON 2D/2.5D adaptation, not the unchanged ROS
  aerial binary. Local bursts, target interrupts and the episode ledger remain
  distinct; all emitted actions count toward the public 500-action cap.

See [FALCON component mapping](gibson/BOUNDED_FALCON.md),
[historical FALCON results](gibson/FALCON_RESULTS.md),
[distinct-building comparison](gibson/DISTINCT_BUILDINGS.md),
[Gibson protocol](gibson/PROTOCOL_AND_TUNING.md), and
[multi-story protocol](gibson/MULTISTORY.md). Earlier campaigns retain their
own frozen configurations; changed code does not relabel their outcomes.

## Where the code is

The method and the benchmark are two trees, and the line between them is the
line between an algorithm and its application:

```text
core/planning/objnav/            the ObjectNav CONTRACT: types, ObjNavEnv / SearchPolicy / LabelMapper, the action converter
core/planning/object_search/     the METHOD: a SearchPolicy -- mapping, rooms, LLM, RPT*, exploration, closing (README there)
core/mapping/{objects,tracking,topology}/   the method's policy-free pieces, shared with the scene-graph runtime
core/control/camera/             the camera-tilt controller and the box-driven pitch servo (where to LOOK, beside where to move)
core/localization/floor_atlas.py the storey localiser the per-storey maps are keyed by
tasks/planning/objnav_benchmark/           the HARNESS: run loop, scoring, records, statistics (every benchmark, one scorer)
tasks/planning/objnav_benchmark_runtime/   THIS TREE: the adapters and the operator tooling
    policy_factory.py  build the agent under test from the method options -- the same for every dataset
    detector_options.py, gpu_plan.py, provenance.py, progress.py   the shared CLI options, GPU placement, run provenance, live progress + lean records
    evaluation.py      the shared run loop the HM3D / MP3D / RoboTHOR adapters call (Gibson's run.py keeps its own, see below)
    services/          the HTTP clients the method is given: detector, segmenter, room LLM
    habitat/           the habitat-sim RGB-D bridge (navmesh="published"|"agent") and the navmesh storey/stair reader (any Habitat dataset)
    benchmark_services.sh   the services every launcher brings up the same way: GPU plan, Ollama, the detector with the
                       benchmark's vocabulary, the segmenter, FFmpeg, teardown (sourced by gibson/ and hm3d/ run_benchmark.sh)
    gibson/            the Gibson v1.1 adapter: protocol, dataset, SemExp scorer, ObjNavEnv, run CLI, launcher, reports
    hm3d/              the HM3D v1/v2 adapter: protocol, dataset, habitat-lab's view-point metrics, ObjNavEnv, run CLI,
                       the scene-per-process campaign and its launcher, reports (README there)
    mp3d/              the next Habitat dataset, on its own branch (feat/objnav-habitat-mp3d-*); same shape as hm3d/
    recording.py, visualization.py, floor_panels.py, search_panel.py, dashboard.py   the step log, the map journal, video, HUD, live page
    review_recording.py   read a recording by step from the log (summary, timeline, step N, map N, events)
    tests/             the method's regression rig (fake services, the benchmark camera, synthetic maps) and the adapter tests
```

The method imports nothing from this tree (enforced by
`core/planning/object_search/tests/test_package_boundaries.py`); this tree
imports the method by module path and hands it services behind the protocols
of `core/planning/object_search/interfaces.py`.

### Adding a dataset adapter

A new dataset -- HM3D, MP3D, another simulator -- touches only this tree and
the label tables, never the method:

1. **Protocol.** A frozen dataclass like `gibson/protocol.py`: camera
   (`intrinsics_from_hfov`), action set, step budget, success radius, sliding,
   path-length convention, kinematic tolerances. Changing a field is a
   different experiment; the frozen configuration records it.
2. **Dataset.** A validated local reader like `gibson/dataset.py`: published
   episode rows (scene-qualified ids, start pose, goal category) and whatever
   privileged maps the scorer needs. Nothing from it reaches the policy.
3. **Scorer and `ObjNavEnv`.** The benchmark's own distance-to-goal and
   success (`gibson/distance.py`) behind the shared environment contract
   (`gibson/env.py`): `reset` returns the episode and its first observation,
   `step` the next, `measure` the privileged `EpisodeMeasurement`. A Habitat
   dataset reuses `habitat/simulator.py` as is -- frames, pose, depth clips,
   collisions and `scene_structure()` are already in the public frame.
4. **Label table.** `core/planning/objnav/labels/datasets/<dataset>.py`, a
   `TableLabelMapper` registered in `labels/registry.py`: the dataset's goal
   names, the detector prompts in a fixed order, and the labels that count.
5. **CLI.** A `run.py` like Gibson's: `--agent` from `policy_factory.AGENT_CHOICES`,
   the detector options of `detector_options.add_detector_options`, the policy
   from `policy_factory.build_policy(args, <label mapper>)`, the loop from
   `objnav_benchmark.runner.run_benchmark` with `MetricsLogger`, the episode
   selection and the frozen configuration from `provenance`
   (`select_episodes`, `episodes_of_category`, `source_fingerprint`, `gpu_plan_record`).
6. **Launcher.** A `run_benchmark.sh` that sources `benchmark_services.sh` for
   the services (`objnav_services_defaults` before its command line,
   `objnav_services_up` after it, with `RUN_MODULE` set to its `run.py`) and
   owns only what is the dataset's: where its episodes and scenes are, how the
   split is partitioned into processes, what the output is named. A launcher
   that runs several processes against the one set of services (Gibson's
   `--parallel N`) tells the library so before `objnav_services_up` --
   `HABITAT_RENDERERS=N` for the GPU plan -- and defines
   `objnav_launcher_cleanup` so the teardown stops its processes before the
   detector they call.

The method needs no change: it sees a camera, an action spec, RGB-D frames
and a pose, and answers in world coordinates.

### The pieces

**The method** (`core/planning/object_search/`; its README is the authority):

| Component | Responsibility |
|---|---|
| `policy.py`, `settings.py` | `RPTSearchPolicy`: detector/mapper composition, RPT* and baseline selection; `RPTSettings` |
| `interfaces.py` | `Detection`, the `Detector` / `Segmenter` / `RoomReasoner` protocols and the service failure contract |
| `baselines/falcon_yolo.py` | The exploration baseline: goal-free planar FALCON (bounded transactions in a disc about the agent, A* relocation to the nearest frontier, a look-around and a STOP on exhaustion) hosting the shared perception cycle, target closing, camera controller and spawn-floor confinement under the attribute names the policy gives them; no LLM |
| `closing/target_closing.py` | Episode-local target takeover, consecutive-frame verification, bbox/depth servo, standoff A* and explicit STOP |
| `closing/target_path.py` | Persistent NavMesh standoff goal, meaningful target refinement and continuous collision-qualified A* execution through occlusion |
| `closing/approach_history.py` | The path walked and the range closed since the takeover began: what separates an extended approach (complete and STOP) from a sudden close sighting (back off and re-verify) |
| `closing/target_backoff.py` | The backing manoeuvre for a target seen suddenly at arm's length: a verified clear spot two or three steps back (trail, grid, floor guard, A*, NavMesh) that the observed grid shows the target from (`_sees`), the turn back, the hand-off to re-verification |
| `perception/cycle.py`, `perception/projection.py` | Fresh raw predictions, coherent pixel projection with a footprint radius, floor-qualified fusion, plan-view association within a class with a height check |
| `perception/doors.py` | Confirmed doorway landmarks from the detector's door boxes |
| `perception/target_context.py` | The semantic sanity check on a candidate: a STRONG room type the target is not searched for in, or a measured height its class is never seen at (a counter top read as a bed) -- a penalty and a multi-viewpoint lock, not a refusal |
| `rooms/room_search_loop.py` | The seven-step room-search loop: one scan visit per room (vantage point, full rotation, finished for the episode) or the bounded room-confined sweep; re-classify → re-estimate → re-order at each loop point over the rooms that are still nodes; transit to the chosen room's vantage point / nearest frontier or to the foot of the chosen stairs; a room re-identified by a new kind of object ends its turn for a fresh solve |
| `rooms/scene_graph.py` | Observed rooms and doors, the room label image, the nodes handed to the oracle |
| `rooms/room_scans.py` | The scan ledger: where every completed look-around stood, on every floor; a room is finished when a scan stood in it or saw more than half of it through observed free space, or -- with no live frontier left -- when the camera looked into it or walked it through (`seen_through`), or when it is a doorless fragment under 3 m2 -- sticky, id-independent |
| `rooms/room_vantage.py` | Where to stand in a room to see it: the reachable interior cell of greatest clearance (distance transform of the room mask) |
| `rooms/stair_nodes.py` | Staircases as nodes of the loop's RPT* instance: ids above every room pid, the facts the oracle values them by, the climb as a leaf charged on every arc |
| `rooms/opening_nodes.py` | Doorways and gaps at the edge of the mapped floor -- a room's exits, its attributes to the oracle and its visit's to walk (`room_exits`); the peek that visits a target landmark, and the exits as nodes under the deprecated `exits_as_nodes` |
| `rooms/house_context.py`, `rooms/oracle_retry.py` | The HOUSE line of the oracle's prompt; one bounded schema repair around its scoring |
| `rooms/floor_context.py`, `rooms/discovery.py` | Per-storey policy state with shared ledgers; episode start, semantic interrupts and the resumption of a local task |
| `world_model/observed_map.py` | Independent per-storey occupancy from RGB-D on the core log-odds integrator; a blocked forward step marks the cell ahead occupied outright (`LogOddsGridCostmap.mark_occupied`), and a second bump within `repeat_bump_m` (0.5 m) of an earlier one leaves a sticky bar one body wide across the heading, forced occupied in every export for the rest of the episode (`_touched`) |
| `world_model/sightlines.py` | The sight ledger: unknown looked through without a depth return (within 2.5 m, from two poses) and enclosed unknown pockets under 3 m2 are settled -- written occupied for the frontier logic alone -- and every camera pose per storey |
| `world_model/peek_stairs.py` | Floor-local seen-connector footprint: the flight within the departure band but off the floor run and off the agent's own trail, its capsule margin (`stair_margin_m`, 0.45 m) bounded per flight so it never splits a body-passable corridor (`adaptive_capsule`), its storey end cut square so no half-disc of floor in front of the first riser is drawn (`_square_foot`, Darden 2026-10-21); excluded from the room partition, from peek viewpoints and from a peek-only A* copy |
| `world_model/stair_terrain.py`, `world_model/stair_grid.py` | RGB-D support surfaces for the observed stair source |
| `exploration/exploration_fallback.py` | Where every failed plan, model or decision lands: the best floor-wide exit (object shadows and frontiers of rooms the target cannot be in wait behind it), the stairs by the explicit fallback rule, the demoted frontiers, a retired frontier, a relocation, a footing sweep, an escape probe -- a move, never an idle spin; failure records and service back-off |
| `exploration/escape_probe.py` | The fallback's rung before its idle turn: boxed in on the inflated map, the grid read without inflation along every turn heading, one turn or one forward step toward the thinnest occupied band with the most floor beyond it, or (since 2026-10-25) along the longest free run of at least two steps whatever closes it; the simulator decides, a blocked heading is remembered per cell |
| `exploration/frontier_sweep.py` | Frontier goal generation for a room or the floor, committed-goal lifetime, optional look-around (the `sweep` ablation) |
| `exploration/path_glances.py`, `exploration/doorway_peek.py`, `exploration/doorway_candidates.py` | Informative glances along a route; the one-shot doorway inspection and its candidates |
| `exploration/exploration_metrics.py` | Observed-area proxy, actions, revisits, stagnation and latency |
| `multifloor/spawn_floor_guard.py` | Spawn-floor confinement under the default `allow_stair_traversal: false`: every seen staircase footprint and every cell observed more than `multifloor.floor_match_m` below the storey plane written occupied, and `_navigate` refusing a goal on those cells or on a storey further than `floor_plane_bound_m` (0.5 m) from the spawn height; never the cells the agent stands on or has stood on (`walked_exemption`); inert when traversal is allowed |
| `multifloor/multifloor_policy.py` | The building coordinator: portals per floor, the approach and climb of the staircase the loop's order chose, arrival bookkeeping, both stair sources; in ground-truth mode it decides nothing by a clock. With traversal forbidden, a height departure the atlas reports (a flight walked down by a committed route, three treads before the atlas calls it one) is walked back up the storey trail (`_retreat_to_storey`, `retreat_started` / `retreat_ended` events, bounded by `multifloor.retreat_actions`) -- until 2026-10-25 it was an `AttributeError` on every remaining action of the episode |
| `multifloor/floor_departure.py` | The optional (`doorway_peek.gate_floor_departure`, off by default) room-peek coverage gate before a floor choice, with its approach revalidation, movement guard and safe recovery from accidental early stair entry |
| `multifloor/stair_ground_truth.py`, `multifloor/floor_decision.py`, `multifloor/ground_truth_traversal.py`, `multifloor/stair_sightings.py` | Ground-truth stairs (the `stair_source: "ground_truth"` ablation): the navmesh connectors as the policy reads them, the fallback rule's up/down choice, the committed polyline traversal, the perfect stair detector |
| `world_model/stair_terrain.py`, `world_model/stair_grid.py`, `multifloor/stair_traversal.py` | Observed stairs (the default since 2026-10-21): measured support surfaces, step-connected candidate paths rising `stair_min_rise_m` in a flight's profile (`stair_profile`: `stair_min_risers` risers of `stair_riser_min_m`-`max_step_m`, treads at most `stair_max_tread_m`) -- the portals the capsule masks under the benchmark's confinement -- and, with traversal allowed, native-heading safety checks and the committed transition lifecycle |
| `falcon/policy.py`, `falcon/regions.py` | Bounded hierarchy and persistent local region history |
| `falcon/motion.py`, `falcon/routes.py`, `route_memory.py` | Safe motion and interruption-aware committed routes |

**The camera's control process** (`core/control/camera/`; the method asks it for the pitch, the
action converter serialises the LOOK before the body action):

| Component | Responsibility |
|---|---|
| `pitch_controller.py` | Sole pitch owner; bounded stair inspection, the footing sweep (a circle at 30 degrees down that maps the blind radius, for a pathless lock or a boxed-in fallback), long unprompted cadence and safe restoration |
| `box_pitch_servo.py` | The image-space vertical servo: where a box lands after a tilt or a step, and the tilt that keeps it whole and centred (pure functions) |

**Policy-free pieces in `core/mapping/`** (shared with the scene-graph runtime):

| Component | Responsibility |
|---|---|
| `objects/box_extent.py`, `objects/box_occluder.py` | A box cut by the frame's edge (sliver vs. floor cut) and its footprint radius; a surface standing in front of the object inside its box |
| `objects/detection_dedup.py`, `objects/target_evidence.py` | Alias/frame deduplication; separated-view target evidence and bounded rejection |
| `objects/support_surface.py` | Whether a box of another class is the surface the target stands on: exact class membership and strict geometry under the memory's footprint |
| `objects/landmarks.py` | The landmark map: per-class association (a tight dedupe radius or footprint-disc IoU; classes never merge), confirmation by observation count; the class vote of 2026-10-04 kept behind `class_votes=True` |
| `tracking/target_memory_3d.py` | The closing's 3-D memory: the fused, precision-weighted XYZ centroid (near frames dominate), its standard error and spread, the fixed association anchor |
| `topology/room_priors.py`, `topology/revisable_room_labels.py` | Where a target cannot be (exclusions, not permissions; `unknown` never excluded); one-object (weak) and two-class (strong) room labels revised the action a new kind of object lands |
| `topology/search_node_oracle.py` | The node oracle: one call per loop point, to the LLM client's REASONING model, over every room still a node and every staircase -- P(going there next finds the target) per node plus "elsewhere" |

**The adapters and the operator tooling** (this tree):

| Component | Responsibility |
|---|---|
| `policy_factory.py` | `build_policy`: the method and its frozen description from the CLI's method options, for every dataset adapter; the `stop` plumbing diagnostic |
| `detector_options.py`, `gpu_plan.py`, `provenance.py`, `progress.py` | The perception-service CLI options, the GPU placement plan, the run provenance (source hash, episode selection, GPU plan record) and the live progress / lean records every adapter shares; `gibson/detector_options.py`, `gibson/gpu_plan.py` and `gibson/progress.py` are Gibson's import paths (and CLIs) for them |
| `evaluation.py` | `run_evaluation`: the shared prepared-environment run loop (frozen-lock check, optional recording, the harness runner) the sibling adapters call. Gibson's `run.py` drives the harness runner itself because it carries the operator features of the 1,000-episode campaign -- resume with accepted config changes, the progress tracker, lean records, the evaluator sidecar, the report; folding those into `run_evaluation` is the one harness unification still open, to be done when the adapter branches merge |
| `services/http_detector.py`, `services/http_segmenter.py`, `services/llm.py` | The detector and segmenter clients (pinned identity, one POST per frame, the contract's two failure families) and the verified LLM client |
| `habitat/simulator.py` | Synchronous registered RGB-D, native extrinsic checks and ENU pose; any Habitat dataset |
| `habitat/stair_connectors.py` | Storeys and stair connectors read once per scene from the navmesh, attached to the episode metadata |
| `gibson/protocol.py`, `gibson/dataset.py`, `gibson/distance.py`, `gibson/env.py` | The SemExp v1.1 protocol, the published episodes and evaluator maps, the FMM scorer, the `ObjNavEnv` |
| `gibson/run.py`, `gibson/frozen_eval.py`, `gibson/run_development.py` | Protocol execution and frozen source/model/data checks |
| `gibson/gpu_plan.py` | Who gets the one GPU -- LLM, then YOLO-World, then Habitat -- from the free VRAM and explicit footprints; the rest on the CPU; the plan recorded in every run |
| `recording.py`, `floor_panels.py`, `search_panel.py`, `visualization.py` | Exact commands/poses, persistent UNKNOWN panels with the rooms and visit order drawn on the active floor, the search column, grids and video; `floor_panels.py` also keeps the per-action **map journal** (`map_journal.npz`) |
| `review_recording.py` | The log-side review of a recording: `summary`, `timeline`, `step N` (everything the log holds about one action, and the map of that action by sector round the agent), `map N` (the map of that action as text), `events`. A review pointed at a step reads this, never a video frame |
| `gibson/run.py`, `hm3d/run.py`, their `frozen_eval.py`, `run_development.py` | Per-task protocol execution and frozen source/model/data checks |
| `tests/trace_room_search_loop.py` | Development probe: the loop on the production map size with fake services and the real action converter on a point agent; not a test |

The storey localiser (`core/localization/floor_atlas.py`) is ROS-free and standard-library only.
Host-side numpy/scipy terrain processing does not enter the Noetic exploration
facade. No robot's flight controller is changed.

## Episode discovery and doorway inspection

Both explorers start with **one full look-around in place** before normal room
selection, taken where the agent stands: as many turns as the camera's field
of view needs to have seen every bearing once -- **ten** at the benchmark's
30-degree turn with its 79-degree camera (`RPTSearchPolicy.rotation_turns`;
`RPTSettings.warmup_steps=12` is the upper bound, since 2026-10-13 no longer
the length). RGB-D, YOLO, object
confirmation and scene-graph geometry remain synchronous on every frame, so the
rotation gives the map its first rooms and the detector its first objects; when
it ends the agent's position is recorded in the **scan ledger**
(`core/planning/object_search/rooms/room_scans.py`), which finishes the spawn room before the first room
is ever chosen. Target pursuit and committed stair safety take priority; a
warm-up interrupted by one and resumed more than 0.5 m away starts its circle
again, so the ledger's record lands where the whole circle was turned. The
warm-up is repeated on every storey the agent first sets foot on (the stair
head is scanned before any room there is chosen). The first global room
decision waits for its completion.

**Doorway peeks are OFF by default since 2026-10-04.** Under the scan visit a
peek -- approach, a half turn from the threshold, the walk back -- costs about
as much as a room's own scan and finishes nothing (its half turn is not a
ledger scan), and the four peeks of the last Ranchester recording were all
cancelled. `"doorway_peek": {"enabled": true}` in the JSON passed through
`--policy-config` restores them with the behaviour below. **The floor-exit
gate they fed is OFF too** (`doorway_peek.gate_floor_departure`, default
false): a floor change is the RPT* order's or the fallback rule's to take
whenever it is chosen, with the 40-action arrival-return guard the only hold.
The Ranchester recording of 2026-10-04 held its descent fourteen times while
re-partitioning kept producing new "pending" rooms, and never reached the
storey the couch was on. `gate_floor_departure: true` restores the former
behaviour as an ablation. Both settings are included in the frozen method
configuration (`room_visit`, `floor_exit_gate`, `room_partition`).

When enabled: for a room with an entrance within **3 m of reachable travel**, the policy plans
**a further metre inside** (`doorway_peek.inset_m=1.0`), beyond the safe threshold
and allowing for the converter's early stopping tolerance. Door proximity
constrains the entrance, not the deeper viewpoint. Small rooms use the deepest
safe observed viewpoint available. A room-contained sightline and the normal
clearance-qualified planner reject viewpoints through walls or unknown space.
The robot aligns to one side and sweeps **at least 180 degrees of measured yaw**.
Duplicate plan calls and back-and-forth yaw jitter do not complete a scan.
The spawn room and a room merely walked through can receive their first peek
only when still unclassified and never previously peeked.

**At most ONE peek attempt per room per episode.** `room_peeked=True` is latched
when the attempt starts, not only after a successful scan. Cancellation, a failed
entry, a scan budget or target/stair preemption never refunds it. A room already
scanned, initially classified (including a single-object weak/provisional label,
with no extra confidence threshold), or previously peeked is exempt. Available
confirmed-object clues are classified before choosing a peek; a label obtained
during a peek cancels the rest of that inspection. `unknown` alone is not a label
exemption. `done=True` still means an actual measured scan completed, not simply
that the one-time allowance was consumed.

With `gate_floor_departure` on, the floor-exit gate asks whether any **eligible
unclassified non-stair rooms** still lack their one-time attempt, not whether
exempt rooms completed a scan. This is an eligibility policy, not a full
semantic-coverage guarantee. Zero known rooms remains incomplete. A genuinely
new unknown room can revoke a stair approach; classified/scanned/attempted
rooms cannot force forbidden repeat peeks. A vetoed ordinary room transit is
released as unreachable rather than repeatedly replanned through the same
gated tread. Native collision checks and committed-traversal safety remain in
force whether the gate is on or off.

Peeks cannot start or continue during a committed stair approach, traversal, atlas
transition, or target takeover. Seen connector geometry near the current elevation
is excluded from peek viewpoints and from a peek-only A* occupancy copy. A path
crossing the stair corridor is refused even if its endpoint lies in a room beyond
it; overhead/other-floor connector segments do not erase ordinary room space.
Stair-only regions are exempt from the peek gate. No scene geometry is added to
the normal map and unseen connectors do not create exclusions.

The previous route is isolated, not destroyed. Its local budget and timeout clocks
pause during inspection and any necessary return transit. After the scan, room
classification, node probabilities and RPT* run synchronously. If the old room
stays first, its **remaining** exploration budget resumes; if another room wins,
normal selection/arrival starts its local turn. New off-route rooms/object kinds
also trigger synchronous reconsideration with peeks disabled. A retained task
does not get a new ten-action allowance merely because evidence changed.

Peeks are non-nesting and remembered per floor and geometric region. Consumed
allowances follow overlapping persistent IDs and substantially overlapping
renumbered/split/expanded regions (60% of the smaller region), but never disjoint
rooms or another floor. This conservatively avoids repeating inspections after
segmentation changes; it does not certify each split child was physically scanned. Defaults:
approach budget 36 actions (extended by travel distance for mandatory distant
visits); return allowance 60 actions; scan budget at least 24 actions (scaled for small turn increments);
at most one attempt per region (`max_attempts` cannot exceed 1; legacy retry
timestamps remain diagnostic only). Invalidated
rooms, blocked entries and stalled scans resume the search rather than spin.
Target approach cancels inspection immediately; committed stair safety is never
abandoned. `doorway_peek.coverage`, `floor_change_held`, completed scan positions
and measured angles, warm-up, phase and event diagnostics are recorded.

ObjectNav's optional FALCON local burst also defaults to 10 actions; an explicit
`falcon.burst_actions` override remains supported. Its specialized local planner
and bounded region history remain distinct from the frontier explorer.

**Accessible frontiers are one shared inventory**: passable known free space,
robot clearance, connected reachability from the current pose, and safe frontier
goal association. No distance horizon is mistaken for inaccessibility. The room
LLM, room facts, entry selection, floor-wide discovery/fallback and dashboard use
it. Unassigned frontiers remain exploration choices and their representatives
remain in the recorded search snapshot, not cluttering the minimalist map. Retired/reached goals are a separate
action-selection filter, not falsely declared geometrically inaccessible. Zero
accessible frontiers does not prove complete semantic observation of a room.

CPU regressions in `tests/test_discovery.py` and `tests/test_doorway_actions.py`
cover startup, real action-converter entry/scan/return, budget preservation,
target preemption, reachability, map changes and synchronous reassessment.
`tests/test_one_time_peeks.py` covers exemptions, no-refund cancellations and stair
isolation. ObjectNav enables a 1-degree path-only angular hysteresis margin: after
forward progress or an opposite turn on the same path, prefer a distance-reducing
forward step within half a turn plus this margin. The generic converter defaults
to zero; STOP, pitch, final facing, blocked recovery and safety vetoes retain priority.
They are not measurements of detector/LLM accuracy or Habitat navigation success.

## The room-search loop (frontier explorer)

`--explorer frontier` runs one explicit loop, with the scene graph maintained
synchronously on every action (`RPTSettings.graph_period_steps`, default 1):
room geometry, object→room association, confirmed doors, cumulative search
time and remaining frontier clusters per room. No LLM call is part of that
refresh.

1. **One visit, bounded to the room in force** (`LoopSettings.visit`,
   default `explore` since 2026-10-13; `scan` was the default from
   2026-10-04). **A visit is a frontier walk**: enter the room at the
   frontier that would reveal the most unknown floor per metre walked, and
   when the camera has resolved it walk to the next such frontier, until the
   room has no live frontier left -- then the room is **finished for the
   episode**. The frontiers are the inventory's clusters credited to the
   room, through the sweep's admissibility filter, re-scored by
   `core/planning/exploration/frontier_ranking.rank_by_view_gain`: the
   unknown floor the camera's full circle would reach from the frontier's
   cell (rays through the unknown, blocked by walls and by what the sight
   ledger has settled) over the geodesic distance -- "the frontier that
   reveals the most unknown area", not the longest boundary and not the
   nearest cell. **Only the significant ones are walked** (since
   2026-10-13, `explore_min_gain_m2`, 1 m2): a frontier revealing less --
   the strip behind a bed, the sliver under a window, a long boundary onto
   a hand's breadth of unknown -- is not a goal, does not keep the room
   open and is not counted for the oracle's `frontier=` (`_live_frontier`
   reads the visit's own population), so a room whose doorways were looked
   through and whose large unknowns were walked is finished: the search is
   for a large object (the Wiconisco recording `de02bc7475e7` walked four
   frontiers of one living room for 61 actions). A doorway onto an unseen
   room clears the bar by the room behind it; a gap onto a pocket the
   walls were seen round does not, and that pocket is given up. The route
   follower faces each goal as it walks, so the
   frame at arrival shows what lay behind the boundary; the sweep keeps a
   committed goal while it is passable, still near the room and still has
   unknown beside it, and the walk itself is where the detector sees the
   room. No vantage point and no fixed circle: the Darden recording
   `ec8d17aeaaaa` (2026-10-13) walked to the middle of a bedroom it had
   already walked across and turned twelve times (actions 255-279), and
   before that turned a full glance on the way into another room, walked on
   to its vantage point and turned the same circle again (61-106) -- 33
   actions for nothing. The visit is bounded hard (`explore_visit_steps`,
   60 -- released `budget_spent` beyond it, not cooled) and charged to RPT*
   by its frontiers: a peek's worth (`openings.service_steps`, 4) plus
   `explore_frontier_steps` (4) per live frontier, so the one-frontier
   sliver of a room glimpsed through its door is priced like the peek it
   replaces and a room with five frontiers like the walk it will take
   (`explore_service_steps`, 12, when the count is unknown; a second-pass
   revisit is priced as the scan it is). **A full rotation completed inside the
   room since the visit began -- a glance taken on the way -- finishes it
   at once** (`finished_by_rotation`): no circle is turned twice.

   Finishing is the **scan ledger's** verdict (`core/planning/object_search/rooms/room_scans.py`): it
   remembers *where* every rotation was completed, not which room id it was
   credited to -- the watershed renumbers, splits and merges rooms on every
   update, and the Ranchester recordings re-entered the same room under a
   new number three times in a row. A room is finished when a completed
   rotation stood inside its mask (`scan_point_inside`: the warm-up, a room
   scan, or a **full glance** -- recorded in the ledger since 2026-10-13),
   when a completed rotation *saw* more than `scan_seen_fraction` (0.5) of
   its cells within the camera's depth range through observed free space
   (`seen_from_scan`: the small room fully visible from the corridor, the
   half of a split room the scan stood in; unknown cells end the sightline,
   so nothing unresolved is credited -- and, since 2026-10-13, a room with
   a **live frontier left** is not credited either: the unknown beyond that
   boundary was never in the scan's view. Corozal/000076's warm-up saw half
   of the known part of a room the agent never entered, finished it with
   three frontier clusters open, and the bathroom behind one of them -- the
   toilet's -- was never offered; `scan_point_inside` stands regardless,
   the circle was turned inside the room and what is left is an opening or
   a sliver the openings offer), or -- under `explore` -- when it has
   **no live frontier and the agent has stood in it** (`explored`), at least
   `explored_inset_m` (0.5 m) in from its edge: the doorway cell is the
   threshold, not the room; a room too narrow for the inset accepts any pose
   inside it. Every floor cell of such a room is known, so every part of it
   has been in the camera's frame; no circle from its middle is owed. The
   live frontier count behind `explored` (and the `seen_through` and
   `fragment` verdicts below) is the admissible one -- a goal under the
   agent's feet or beside one already retired is not a frontier the visit
   can walk to, so it is not one that keeps the room open; the visit and
   the verdict read the same population. The verdict is sticky per
   `(floor, pid)`. **A clue mid-visit does not switch rooms**
   (`reconsiders_deferred`): the Ranchester recording switched at 57, 73
   and 121 and came back to each; the clue is kept for the loop point the
   visit ends on, where every node is re-valued anyway.

   **Every rotation is the camera's, not the body's** (since 2026-10-13):
   the warm-up, a room scan, a full glance and a footing sweep end when the
   field of view has covered the circle -- `turns * 30 + 79 >= 360`, ten
   turns at the benchmark's 30 degrees, not twelve
   (`view_gain.ViewCone.rotation_turns`, `RPTSearchPolicy.rotation_turns`);
   the two turns that bring the body back to its start heading show what the
   first two frames already showed. `RPTSettings.warmup_steps` (12) is now
   an upper bound on the warm-up, not its length.

   `visit="scan"` is the former default, kept as a mode: walk to the room's
   **vantage point** -- the interior cell of greatest clearance the observed
   map can reach (`core/planning/object_search/rooms/room_vantage.py`: the Euclidean distance transform
   of the room mask, so the middle of the open floor, not the doorway and not
   the far frontier) -- turn a **full circle**, and leave; the room is then
   finished whatever frontier its mask still shows, because the camera's ray
   ends at the nearest object and a 360-degree scan from the open floor shows
   what a second viewpoint would. The visit is bounded hard
   (`scan_visit_steps`, 36 -- released `budget_spent` beyond it) and the walk
   to the vantage point separately (`scan_approach_steps`, 20; an approach
   that runs out, or that A* refuses, scans from where the agent stands if
   that is inside the room -- `scans_in_place` -- and otherwise releases the
   room `unreachable`). The rotation ends on measured yaw (the camera's
   circle swept, or two turns past that count), the scan is recorded, and
   the turn ends `exhausted`: productive, cooled, never repeated by the
   escape hatch. Under `explore` a **second-pass revisit** of a finished
   room is exactly this scan, from a spot `revisit_standoff_m` away from the
   room's earlier scan points.

   **LLM-first (since 2026-10-07).** The node oracle
   (`core/mapping/topology/search_node_oracle.py`, the 14B reasoning model)
   is shown **every room of the storey** -- finished and type-excluded ones
   included, each with a `status=` of `never_entered`, `entered` or
   `scanned(<how>, <ago>)` -- together with a **HOUSE line**
   (`core/planning/object_search/rooms/house_context.py`): every room type found so far with its
   status, the rooms still unidentified, the openings not yet looked into,
   whether unexplored frontier is still reachable, and the budget
   (`actions used about 150 of 500`, in 25-action steps so an unchanged map
   keeps an unchanged prompt). The prompt asks the model to reason as a
   person who knows homes does, in three written steps before any number:
   **home** (where the target lives: a toilet in a bathroom, separate or an
   en-suite reached through a bedroom; a television in a living room, then
   a bedroom, rarely a kitchen, never a bathroom), **house** (a home has one
   kitchen and one living room, so a living room found makes a sofa
   unlikely in the unidentified rooms; a bathroom still missing makes the
   small unidentified rooms and the unlooked doorways where the toilet must
   be -- exploration first), and **stage** (early: the unexplored first,
   scanned rooms low; late, with the home type found and the target not
   seen: the detector may have MISSED it in a scanned room of the right
   type, and the chance that an unseen room holds it falls with every place
   looked at). It writes a structured **`pass` verdict** -- `first` while an
   unexplored place is still worth looking at before any scanned room is
   re-checked, `second` once the home is covered -- recorded and shown
   (`oracle_pass`), but **since 2026-10-13 the geometry decides whether
   finished rooms are offered as nodes again** (`LoopSettings.second_pass`):
   the storey must be *covered* -- no unfinished room, no exit left to peek,
   no reachable frontier worth a walk (`explore_min_gain_m2`, below) -- and
   a `second` from the model while openings still stand does nothing
   (`qwen2.5:3b` wrote "no unexplored places or frontier left" with six
   openings on its own HOUSE line, Corozal/000076 action 159, and the agent
   walked back across the storey to re-scan living rooms while the
   bathroom's door stood unpeeked). A revisit is a scan from a spot at least
   `revisit_standoff_m` (1.5 m) from the room's earlier scan points; a
   storey with no reachable frontier at all is the forced form
   (`second_pass_forced`, `exhausted` beside `covered` in the record).
   **Since 2026-10-14 the second pass is off by default** (`second_pass:
   false`): one look at a room is the look it gets. The re-scans a covered
   storey bought in the 2026-10-13 toilet run found nothing the first scan
   had not (Collierville/000023: room 0 scanned at the warm-up, again at
   237-263 and again at 403-420, each from its vantage point, with the
   toilet's WC unentered 3 m away), and the user's rule is that a room
   scanned once is finished. With the storey covered the exploration
   fallback carries the search to the unknown the first pass skipped -- the
   slivers and shadows, now ranked by what a look from each reveals (below)
   -- which is where an unseen target is. **Finished rooms are left off the
   oracle's node list too** (`scanned_to_oracle: false`): they are no node,
   so their line bought the model nothing but a longer prompt and, for
   `qwen2.5:3b`, the `pass: second` confusion; the HOUSE line still names
   every room type found, scanned ones included, and the hidden rooms read
   0.0 downstream. An ENTERED room the agent LINGERED in -- searched for
   `lingered_steps` (20) actions or more with at most one significant
   frontier left -- keeps `lingered_discount` (0.2) of the oracle's number
   in place of `entered_discount` (0.5): as good as scanned, whatever the
   model writes (the prompt now says so too: "searched 20 s or longer with
   frontier 0 or 1 is as good as scanned: 0-5"). The
   three arithmetic floors below (`unexplored_floor`, `unexplored_elsewhere`,
   `home_floor`) and the hard type exclusion (`type_prior`) are **off by
   default** -- they were the 3B era's defaults, and the model is given the
   facts they encoded instead; each stays a knob. The storey/stairs
   reasoning of the multi-storey development protocol is a supplement to
   the prompt shown only when a staircase is a node. With nothing to offer
   -- the spawn room finished by the warm-up, no doorway seen yet -- the
   reasoning model is not spent (`oracle_calls_skipped`); the fallback
   carries the search until a node appears.
   `tests/probe_node_oracle.py` runs five single-storey scenarios against
   the live model and checks the judgement (all five pass on
   `qwen2.5:14b-instruct`, ~30 s per call on a 32-thread CPU).

   **What the geometry adds to the model's numbers (since 2026-10-13).**
   The oracle's probability is scaled before the supervisor filters on it
   and RPT* is handed it (`_scaled_values`; `factor` per node in the
   estimates, `value_factors` in the record): an **exit keeps
   `opening_value_factor`** of its number -- `clamp(gain_m2 / gain_full_m2
   (4 m2), gain_min_factor (0.25), 1)`, where `gain_m2` is the unknown
   floor the camera's cone would reach from the threshold along the
   opening's heading (rays through the unknown, blocked by walls and the
   settled unknown) -- a `doorway` (a confirmed door at the threshold) in
   full whatever the rays count; and a **room the agent has stood in but
   the ledger has not finished keeps `entered_discount` (0.5)** of its
   number. The model values "an opening with nothing glimpsed" the same
   whether a wing or a strip lies behind it (Corozal/000076, action 126: a
   10-cell gap 4.6 m away with a square metre behind it was walked to
   before a 148-cell opening onto the unseen half of the storey), and it
   valued a bathroom walked through at 0.6 -- "known to contain a toilet"
   -- over an unknown bathroom-sized room never entered at 0.2
   (Darden/000034, action 139, 11 m back). The prompt tells the model both
   rules (an entered room at most half of an unexplored one of its type;
   prefer places the robot has not been near); the factors hold whatever
   it writes. **Two more factors since 2026-10-21** (the user's
   recalibration for the unvisited): a room the agent has NEVER stood in,
   not finished from outside, is raised by `unvisited_boost` (1.5x) and
   read at no less than `unvisited_floor` (0.15) when the model gave it
   anything -- the models read `frontier=0` or `1` on a never-entered room
   as little to find, when the count measures the map and not the room,
   and one boundary cleared opens the whole of it (the prompt now says
   "never lower a never-entered room for a low frontier count"); and the
   entered factor **decays with the time spent inside**:
   `entered_discount * 0.5 ** (steps inside / dwell_half_life_steps)`, 20
   actions a half-life, floored at `dwell_floor` (0.05), the lingered rule
   still capping it -- a room walked through in five actions keeps most of
   its half, one searched for forty a quarter of it. `value_factors` in
   the loop's diagnostics and the HUD carry every factor.

   **What is not a node (the first pass).** At every loop point the loop
   withholds finished rooms from the solver and the transit until the
   storey is covered, and records why (`excluded` in the events,
   estimates and HUD); with `LoopSettings.type_prior` switched on it also
   withholds a room whose identified type **cannot hold the target**
   (`core/mapping/topology/room_priors.py`: a sofa is not
   searched for in a bedroom, a bathroom or a kitchen; a toilet not in a
   bedroom; the table is exclusions, not permissions, so a hallway stays
   searchable and `unknown` is never excluded). Since 2026-10-05 the type
   rule has three guards (`room_priors.ruled_out`): it applies to a
   **strong** label only (two or more kinds of object agree -- a weak
   label is a guess the oracle sees as `type=bedroom?` and values itself,
   `weak_type_kept` events); never when an object of the target's own
   class has been confirmed in the room; and never when a **home object**
   of the target stands in it (`HOME_OBJECTS`: a sink or a shower where a
   toilet lives -- the Hanson recording merged a bathroom's sink into a
   "living room" of sofas and ruled it out for the toilet a metre from
   the sink; `home_object_kept` events). **One signature object makes a
   label strong** (`room_priors.SIGNATURE_OBJECTS`,
   `RoomLabelSettings.signature_objects`, since 2026-10-05): a bed is a
   bedroom, a toilet, a shower or a bathtub a bathroom, an oven, a stove or
   a refrigerator a kitchen, with no second kind of object needed, when the
   classifier's label agrees -- the Hanson re-fly of 2026-10-05 saw a bed
   and a television through a doorway at action 270, kept the "bedroom?"
   weak and walked in to scan it for a toilet (actions 270-289). A chair,
   a cabinet, a desk or a plant names nothing on its own and stays weak;
   the home-object guard still holds (a bed in a room with a sink keeps
   the room a node for a toilet). The two kinds that make a label strong
   must include one that is **distinctive**
   (`RoomLabelSettings.distinctive_required`, `room_priors.GENERIC_OBJECTS`:
   a cabinet, a plant, a book, a vase, a clock, a cup, a bottle describe no
   room): the Ranchester couch search had a cabinet and a potted plant make
   an upstairs room a strong "living_room" at 0.95, which put a living room
   on the storey summary the node oracle reads and wobbled its verdict on
   where living rooms are. **A home object is a prior, not only a guard**
   (`LoopSettings.home_floor`, off by default since the LLM-first change;
   0.60 restores it): a room holding a
   confirmed home object of the target is handed to the oracle with
   `SearchNode.home` set and, with the floor on, is never read below it,
   whatever the model wrote and whatever its storey verdict says -- a
   bathtub on this storey IS the toilet's home type, found -- and such a
   room is **never finished by sight from outside it** (this guard stands
   regardless): the scan ledger's `seen_from_scan`
   and `seen_through` verdicts (half its floor seen through its door) do not
   exclude it, only a scan the agent stood in does (`home_object_kept` events
   with `scanned`). The Allensville toilet run's warm-up spin at the spawn
   point saw more than half of the bathroom's floor through its door,
   finished it with the bathtub inside and the toilet behind the jamb, valued
   it 0 and peeked twenty openings before coming back to it at action 283.
   **The labels are read before the nodes
   are chosen** (`ObservedSceneGraph.refresh_labels` at the loop point):
   until 2026-10-05 the re-classification ran inside the oracle call,
   after the exclusions, so a room that became a strong bedroom on that
   very action was shown to the oracle, valued at 0.10 ("bedroom, fully
   seen, no toilet") and kept in the order for one more loop point (Hanson
   action 150). Withheld rooms keep the oracle's own probability for the
   record and the HUD but are not offered; when nothing is left to offer
   and no staircase is a node, no model call is made and the exploration
   fallback carries the search. `LoopSettings.min_prob` (0.05) keeps the
   oracle's "1" for a bedroom in a search for a couch out of the order.

   **Two more ways a room is finished** (`core/planning/object_search/rooms/room_scans.py`, since
   2026-10-05), both for a room with **no live frontier** -- no accessible
   unexplored boundary the sight ledger has not settled (below): it is
   `seen_through` when the camera **looked into it** (one recorded pose,
   not only a scan point, had `scan_seen_fraction` of its cells inside the
   camera's cone with a clear line of sight: the balcony seen whole from
   its threshold, the closet from its door) or **walked through it** (a
   pose lies inside it and it is narrow -- its widest point under
   `LoopSettings.walkthrough_clearance_m` (0.9 m) of clearance, a corridor
   or a balcony the cone spans as the agent walks; a wide room merely
   stepped into is not finished so, the walls beside its door being behind
   the camera); and it is a `fragment` when it is under
   `LoopSettings.fragment_max_m2` (3 m2) with no confirmed door on it -- the
   strip behind a bed the watershed carved into a room of its own (Hanson
   R13, action 150: valued 0.25 as "a bathroom perhaps"). A bathroom is
   small too, but a bathroom has a door or an unseen boundary. The Hanson
   re-fly put its balcony (R5) back in the order at 0.25, "never entered",
   at action 206 -- 158 actions after the agent had stood at its far end.
   A room finished by what the walk to it showed is released `exhausted`
   before it is entered (`finished_in_transit` events).

   **The uncertainty floor** (`LoopSettings.unexplored_floor`, applied in
   `search_node_oracle.SearchNodeOracle`, since 2026-10-05; **0 -- off --
   by default since 2026-10-07**, when the rule moved into the prompt and
   the 14B model was found to apply it; 0.25 restores the arithmetic). A
   room never entered and still `unknown`, or an opening nothing was
   glimpsed through, is a place the search knows *nothing* about, and the
   least it owes such a place is a look: its probability is never read
   below the floor, whatever the model wrote (`floored` in the estimate
   events). The Hanson recording of 2026-10-04 had the 3B model write
   "small unknown room, no toilet fits" on two never-entered rooms and
   "no toilet glimpsed through gap" on six doorways -- the words of the
   prompt's own example, applied to a toilet -- `min_prob` dropped them
   all, the order was `[stairs]`, and the agent left the storey at action
   27 with every door on it unlooked into. With the floor the unexplored
   nodes stay in RPT*'s hands, where distance decides: two unknown rooms
   3 m away at 0.25 come before a staircase 21 m away at 0.6, and the
   user's "peek into the rooms before you descend" is a probability, not
   a rule. A room with a *weak* type, or an opening with a glimpse, has a
   preliminary classification and is the model's to value; the room line
   now says `entered=no` instead of `ago=never`, and the prompt's rules
   2b and 2c say what that means (an unexplored place owes a look, never
   "too small"; a home-type object in a room of another type is the map's
   merge showing). **The floor has one condition** (`LoopSettings.
   unexplored_elsewhere`, since the Ranchester couch search of
   2026-10-05; off by default since 2026-10-07, 0.10 restores it; only the
   multi-storey supplement ever produces the verdict): the model's own
   storey step, asked for as a structured verdict
   `home_here` -- `found` (a room of the home type is on this storey),
   `missing` (none yet, but it belongs here) or `elsewhere` (it lives on
   another storey). The 3B model answers that reliably ("living rooms are
   downstairs" at every loop point) and then writes the worked example's 25
   on every upstairs gap all the same, so thirteen peeks (~25 actions each)
   outranked the staircase it stood 0.9 m from at 0.60, and the 500 actions
   ended upstairs. With `elsewhere` an unexplored place is read at 0.10
   exactly (`capped` / `floored` in the estimate events, the verdict on the
   HUD): rule 2b's "less", applied by the code because the model does not
   apply it; the stairs come first and a door beside the route is still a
   cheap peek. `found` and `missing` keep the 0.25 floor, and a reply
   without the field (an older model's) is read as before.

   **The clue rule holds in both modes**: one confirmed object names the
   room (`RoomLabelSettings.min_objects = 1`; a single class is a *weak*
   label, shown to the oracle as `type=kitchen?`, two or more confident
   classes a *strong* one), the classifier is re-asked for the room in
   force or in transit **the action a new kind of object lands in it**
   (`relabel` events; a second chair is not a clue, and a set of kinds
   already judged is never bought twice), and a changed name ends the
   room's turn as `reclassified` -- neutral: not cooled, no attempt charged
   -- so that steps 2-4 run over the new fact; the sink first taken for a
   kitchen is re-valued the action the toilet shows.

   `visit="sweep"` is the former behaviour, kept as an ablation: goals are
   the frontier clusters the room label image credits to the room, ranked
   by `core/planning/exploration/frontier_ranking.py`, each route planned
   on a **copy of the map in which every other room's cells are written
   unknown** (`LoopSettings.confine_routes`; lifted while the agent's own
   cell is outside the room); the burst ends after `local_steps` actions
   (10), when nothing reachable is left, or on the clue rule, and the same
   room may be chosen straight back.
2. **Global re-classification.** At the loop point `graph.reason` re-labels
   every known room from every confirmed object observed so far. **Stairs are
   never a room**: the footprint of every seen staircase near the storey's
   level is removed from the free mask before the watershed runs
   (`ObservedSceneGraph.update(..., exclude=policy.room_exclusion(world))`),
   so no "room" is carved on a landing, labelled, valued or waited on; a
   staircase enters the search only as a STAIRS node (below).
3. **Probability per node** -- the same call, to
   `core/mapping/topology/search_node_oracle.py`. The model is shown every
   ROOM of the floor **that is still a node** (not finished, not ruled out
   by type) -- type or `unknown`, size, frontier clusters left, its
   **exits** (`exits=2 (1 doorway)`: the doorways and gaps at its edge to
   space not yet seen and the objects glimpsed through them -- the room's
   attributes since 2026-10-21, see *Exits are the room's attributes*
   below), time
   searched and how long ago it was last inside, objects seen, whether the
   robot stands in it -- and every STAIRCASE off the floor -- up or
   down, whether the other storey was visited and what was found and
   searched there, whether the robot arrived by it -- plus
   one line on this storey and each other storey known (under the
   deprecated `openings.exits_as_nodes` every **OPENING** not folded into
   an offered room is a node of its own as well). It answers, for
   each node, the probability in percent that **going there and searching
   finds the target**.
   These are independent search-success estimates, **not mutually exclusive
   location shares**, and are not restricted to the ten-action burst. Each is
   converted to a fraction and clamped below 1, without cross-node normalisation.
   RPT* uses multiplicative survival; the legacy `elsewhere` diagnostic now means
   the product of failure probabilities, and `p_present` its complement. This is
   an explicit independence approximation, not proof that real rooms are independent.
   The current room may remain the best choice for successive bursts. Everything the search
   needs weighed is the model's to weigh, and the system prompt says how:
   a type that rules the object out has low probability; a
   room searched long and recently reads low; a never-entered room of the
   right type is the best bet; an `unknown` room is an **exploration node**
   valued by its size, its frontier and the room types the target needs
   that are still missing on this storey; a hallway with exits is valued
   by what the home still lacks behind them; a room's exits raise it only
   where its type fits the target (a bathroom's unlooked door for a toilet,
   never a living room's gaps), and a toilet glimpsed through one of them
   names the room behind it;
   a staircase is the whole set of rooms the other
   storey may hold, read from what this storey turned out to be (a kitchen
   and a living room make a ground floor, so the bedrooms are upstairs),
   discounted when that storey was searched, and not taken straight back
   when the robot has just come from it. It is told, twice, **never to
   reason about distance**: travel is the planner's. This is the one
   judgement per loop point the search cannot afford to get wrong, so the
   runtime routes it to the LLM client's reasoning model
   (`LLM_REASONING_MODEL`, default `qwen2.5:14b-instruct`, with its own
   timeout and context window; the 14B follows the rules and answers in
   ~25 s on the laptop's CPU once loaded, and is the model for the
   benchmark runs. Development runs since 2026-10-04 set it to
   `qwen2.5:3b-instruct` for turnaround -- the type prior and the finished
   ledger now take the exclusions the 3B used to get wrong out of its
   hands, so what it is left to weigh is the unknown rooms against the
   stairs) and
   waits for it. The oracle
   reuses the model's last reply when the prompt it would show is
   byte-identical (effort numbers are shown in coarse steps for exactly
   that), so a loop point over an unchanged map costs no call
   (`oracle_reuses`). The per-node record in `room_search_loop.estimates`
   carries the probability, the model's reason and the distance charged.
4. **Visit order** -- RPT* over the surviving nodes, rooms and staircases
   together (`core/planning/object_search/rooms/stair_nodes.py`). A room enters the instance at its
   entry point; a staircase at the foot of the flight on this floor's map,
   with a **leaf** as long as the flight plus `MultiFloorParams
   .floor_change_cost_m` (8 m, about thirty actions) charged on every arc
   into and out of it -- so going upstairs puts every room down here that
   much further away, which is exactly what the expected-time-to-find
   objective must see. The expected visit is folded in as per-node service
   time (`LoopSettings.service_steps()`: half the approach bound plus the
   rotation under `scan`, `local_steps` under `sweep`); the order is
   re-solved at every loop point
   (`ObjectSearchParams.resolve_on_release`) and on semantic discovery events,
   not merely on a timer. **A floor change is never gated on room coverage**
   (unless `doorway_peek.gate_floor_departure` is turned on): the staircase
   is a node like the rooms, taken when the order puts it first; the
   existing 40-action arrival-return guard and the fallback rule still apply.
   **All of this presumes `allow_stair_traversal: true`.** Under the benchmark
   default (false) no staircase is offered to the oracle or the solver, `commit`
   and the fallback's explicit rule refuse, an unplanned height departure starts
   no traversal, and the spawn-floor guard keeps every goal on the spawn plane.
   **The way back is the one exception**: under `scan` the staircase the
   agent arrived by is withheld while this storey still has a room that is
   neither finished nor ruled out (`way_back_held` events) -- the storey is
   looked at before "not here" means anything. The 3B oracle sent the
   Ranchester agent straight back up twice with "living room found; no
   couch here", three unscanned rooms on its list each time; once every
   room is finished the way back is offered, marked `arrived_by`.
   When the order puts a staircase first the loop commits it to the
   building coordinator, which approaches the foot of the flight and
   climbs; the node's turn ends `traversed` the action the climb begins,
   and the floor's search resumes from a fresh estimate when the robot
   comes back down -- with the way back up a node like any other, marked
   `arrived_by`. The order and its head are kept on the loop (`order`,
   `order_index`, `next_room`) for the recording and the dashboard.

   **Exits are the room's attributes** (room-level GAPs,
   `OpeningSettings.exits_as_nodes=False`, the default since 2026-10-21).
   A standalone LLM number per doorway or gap let the order chase open
   passes for their own sake -- every gap of a living room, in a search for
   a toilet -- in place of the room-level prior. So under `explore` an exit
   is never a node: **every** exit read off the live map is folded into the
   room it opens from (`_fold_openings`, `opening_folded` events; the HUD
   still lists it, flagged `~`), the room's node line carries its exits as
   attributes (`exits=2 (1 doorway)  glimpsed through its exits: toilet`,
   `room_exits` -> `SearchNode.exits` / `doorways` / `glimpsed`; the HOUSE
   line counts every exit of the storey), and the prompt's EXITS rule ties
   them to the room's type: exits raise a room only where its type fits the
   target's home (a bathroom's unlooked door for a toilet, a bedroom's
   possible en-suite), a hallway's or an unidentified room's exits are
   valued by what the home still lacks behind them, and a living room's
   gaps do not raise it for a toilet however many there are. The **global
   decision is over rooms** (and stairs); the **look through a doorway is a
   step of the chosen room's local exploration** -- the explore visit's
   frontier walk, which ends when the room's GAPs are exhausted or
   `explore_visit_steps` runs out, and whose release is the loop point that
   re-selects globally. A room the ledger has finished but with GAPs still
   left to walk (significant live frontier on a room with an exit:
   `_gaps_left`) is **reopened** for them (`room_reopened` events,
   `rooms_reopened`, `reopened_for_gaps` in the diagnostics): shown to the
   oracle as `status=scanned(...)` with its `exits=`, entered at its best
   frontier, priced by its frontiers, discounted as an entered room, walked
   and not re-scanned -- the warm-up spun a circle in the spawn room whose
   doorways were never looked through, and before this its doorways were
   the opening nodes that carried the search on. A glance that completes a
   rotation mid-visit finishes the room in the ledger but no longer ends
   the visit while GAPs are left, and a room in transit with GAPs left is
   not released `finished_in_transit`. A new exit on an offered room buys
   the oracle a call at most every `revalue_actions`, as a new opening node
   did. The `scan` and `sweep` visits have no local walk to fold an exit
   into and keep the exits as peek nodes; target landmarks are nodes in
   every mode. Regressions: `tests/test_room_level_gaps.py`.

   **Openings as nodes** (`core/planning/object_search/rooms/opening_nodes.py`,
   `LoopSettings.openings`, since 2026-10-04; **deprecated since 2026-10-21**
   -- the `openings.exits_as_nodes=True` mode, described here as it ran).
   The room partition is a
   watershed over *observed* free space, so a room the agent has only
   glimpsed through its door was not a room until 2026-10-21: its few seen
   cells hung off the hallway's region, and whatever object stood in the
   glimpse named the hallway. **The doorway partition** (since 2026-10-21,
   `core/mapping/topology/room_watershed.py`, `rooms/scene_graph.py`): the
   doorway an opening is seen through is read off the line of sight from
   the viewing pose to the opening -- the floor narrows between observed
   jambs (`sightline_thresholds`, the unknown passable so an open room's
   frontier is no doorway), the dip is snapped to the doorway's choke and
   kept as a sticky threshold of the floor (`ObservedSceneGraph.note_openings`,
   `threshold_events` with `source: sightline`); the carve there is a
   two-cell **line across the passage** (the disk of old swallowed a thin
   sliver or left two crescents of it), ranked by clearance to observed
   obstacles with a short far side allowed when it leads into the unknown;
   and the floor beyond the cut that no clearance peak seeded is a **room
   of its own** from `stub_min_cells` (4) cells, exempt from the room-size
   floor, with the unknown beyond it as its frontier -- a node the global
   loop can choose, which keeps its number as it grows (the registry
   matches by containment) and is joined to the room it was seen from by a
   **threshold edge** through the doorway coordinate (`graph.doors`, kind
   `threshold`, index `t<k>`; a detected door's edge is kind `door`). The
   same carve now parts the sliver behind a detected door. Regressions:
   `tests/test_doorway_partition.py` and the watershed's own
   (`core/mapping/topology/tests/test_room_watershed.py`). In the Ranchester recording of 2026-10-04 (attempt 7, action 75)
   the agent stood in a hallway facing three doors; the toilet seen through
   one of them made the whole west half of the storey a "bathroom", every
   door off the hallway was demoted with it, and nothing was left to value
   but the stairs it happened to turn towards. The decision to descend was
   right; the reasoning that produced it was not, and in a search for
   something on *this* storey it would have been fatal. So every genuine
   **exit** of the mapped floor -- a frontier that is not an object's
   shadow, not the floor under the agent's feet, at least `min_cells` (5,
   0.5 m) wide after merging within `merge_m` (1.5 m), not at the foot of a
   seen staircase nor within `stair_exclusion_m` (1.5 m) of a cell the
   spawn-floor guard has masked (a staircase's footprint, an observed drop
   -- Corozal/000076 walked an 87-action loop round a staircase's capsule
   to peek at the gap at its foot; a peek whose threshold the stairs are
   placed under mid-walk is retired at once, `peek_abandoned` "at a
   staircase placed since"; since 2026-10-14 the test is *directional*:
   only masked cells and portal points on the frontier's unknown side
   count, `near_stair_mask(..., heading)` / `beyond` -- Darden/000034's
   bathroom door, 1.3-1.5 m beside the flight with its room opening the
   other way, was struck off the openings for a whole episode by the
   omnidirectional window), not toward unknown the **sight ledger** has
   settled (an enclosed pocket, or unknown looked through without a depth
   return; see below) -- is an **opening**: a node with a sticky building-wide id
   (`O<n>` on the HUD, `OpeningRegistry`), described to the oracle by the
   room it opens from, whether a confirmed door frame stands at it
   (`doorway` / `gap`) and the objects glimpsed through it so far
   (confirmed landmarks 0.3-3 m beyond the threshold *inside a 35-degree
   cone* around its heading; the heading is the direction from the known
   floor around the frontier cell toward the unknown -- the mean of the
   unknown cells alone pointed the stair passage back into the hallway in
   attempt 8, and a half-plane glimpse then credited a hallway bed to it,
   so the passage read as a bedroom door for sixty actions), and priced
   to RPT* at a **peek's** few actions (`service_steps`, 4; `build_instance
   (service_s=...)`), not a room's scan. Its visit is the peek
   (`entry_peek`, `peek_look` / `peek_complete` events): walk to the
   threshold (bounded by `approach_bound`: the distance's worth of forward
   steps times 1.6 plus six turns, at least `approach_steps` 20; a walk
   that runs out looks from within `merge_m` or retires the opening; a
   refused threshold is re-aimed once at a passable cell on the agent's
   side), face the unknown, and sweep `2 * look_turns + 1` headings a turn
   apart from the side nearer the agent's yaw, one in-place turn per
   action; then the opening is retired for the storey and the node's turn
   ends `exhausted`. **The look is taken from the threshold itself**
   (`step_through`, since 2026-10-14): the supervisor declares a transit
   arrived within `arrival_tol_m` (0.6 m) of its goal -- a room centre's
   tolerance, generous on purpose -- and Collierville/000023 took its
   three-heading look 0.58 m behind the doorway, through the wedge the
   jambs leave, and saw nothing of the room. A peek that arrived more
   than the loop's arrival tolerance short walks the remainder first
   (`through_steps` 8 at most, `peek_step_through`, `peeks_stepped_through`)
   and looks `through_look_turns` (2) to each side of the way it came
   through -- five headings, left and right, standing in the door
   (`peek_look` with `through: 1`); a threshold cell that still borders
   unknown is walked to after the first look instead, once that look has
   mapped the floor beyond it; a walk whose budget the transit already
   spent (`peek_from_here`) is not extended. An opening the camera resolves **on the way** -- gone
   from the live inventory for `vanish_actions` (3) actions running while
   the threshold is still more than `merge_m` away -- ends the peek there
   (`peek_resolved`, released `exhausted`, nothing retired: a frontier that
   re-snaps elsewhere is offered again). The HUD and the per-step record
   list **every opening read off the live map this action** (since
   2026-10-13): a folded one flagged `~` ("folded into R<n>'s walk"), never
   hidden, and the one a peek is walking to flagged `>` and kept on the
   list even when the inventory has lost it, with its `gain_m2`.
   What the look saw is floor in the partition now: the room behind
   separates, is named by its objects, and is excluded by type or scanned
   like any other. **A weak type label does not rule out a room with more
   than `weak_type_max_openings` (1) openings** (`weak_type_kept` events):
   the one object that named it may belong to a room behind one of them.
   Openings count as "rooms left" for the way-back hold, so the agent does
   not go back upstairs with doors on this storey unlooked-into. A new
   opening appearing while nothing is in force buys the oracle a call at
   most every `revalue_actions` (6); a new room or staircase is valued at
   once. `openings.enabled=False` is the ablation in which the exits are the
   exploration fallback's alone.

   **The openings are read off the live map every action** (since
   2026-10-13, `RoomSearchLoop.plan` -> `_opening_options`, cached per
   action): the HUD and the per-step record show the openings as they stand
   now, and an opening the camera has just resolved -- looked straight
   through from its threshold -- is gone the action it is resolved, not at
   the next loop point. The Darden recording `ec8d17aeaaaa` kept `O3` on the
   HUD for fifty actions after the agent stood on its threshold looking
   along its heading. **An opening off a room that is itself a node is
   folded into that room's walk** under `explore` (`_fold_openings`,
   `opening_folded` events, `openings_folded` in the HUD and diagnostics;
   under room-level GAPs every exit is folded, whatever its room's standing):
   it is one of the room's frontiers, and the room's visit walks exactly
   those, so offering both is the same place twice -- that recording's
   order read `R3 -> O3`, a room and the gap at its own far end, and the
   room's scan resolved the gap before the peek was ever due. Openings off
   finished rooms, off rooms the oracle valued below `min_prob` (a kitchen
   at 0.0 whose doorways lead on) and target landmarks stay nodes of their
   own: they are how the next room is found. (Keeping an opening with a
   room's worth of unknown behind it out of the fold as well was tried on
   2026-10-14 after Corozal/000019 folded the toilet's door into a
   never-entered living room whose visit entered 11 m away; it did not
   rescue the episode and was reverted with the rest of that round -- see
   `LESSONS.md`.) **An exit the oracle has not
   valued is never read as 0.0** (since 2026-10-14): the oracle's numbers
   are replaced wholesale at every call and the fold is read against the
   numbers *before* it, so an exit folded into its room at the call --
   not shown to the model -- and unfolded by the reply that dropped the
   room under `min_prob` comes out with no number at all. Corozal/000142
   lost the toilet's own doorway that way (12 m2 unseen behind it, 6 m
   away; the room it opened from went "office, entered" 1 %) and walked
   12 m to a gap the model gave 10 %, with no SELECT in the 54 actions
   left. Such an exit is handed to RPT* at the oracle's own *last* number
   for it when an earlier call gave one, else at
   `OpeningSettings.unvalued_prob` (0.10, what the model gives an unseen
   doorway it is shown; the floor, 0.05, was tried on 2026-10-14 after six
   such exits read as outranking every room in Wiconisco/000014 and
   Darden/000034, and cost Markleeville/000158 -- the 3B oracle's numbers
   do not repeat, and the run with 0.10 is the one the user chose);
   `RoomSearchLoop._exit_prob` says which (`prob_source`:
   `oracle` / `remembered` / `provisional`, in the estimate and the HUD
   record; event `opening_provisional`, stat `openings_provisional`). The
   node without a number still buys the next call, so the prior lasts one
   SELECT. 0 disables.

   **A confirmed landmark of the target's class is a node too**
   (`kind="landmark"`, `T<n>` on the HUD, `landmarks_enabled`). Ranchester
   attempt 9's downstairs warm-up saw the couch three times at 0.41-0.45 --
   under the takeover's 0.50, the far frames without depth -- put a `sofa`
   landmark on the map 3.8 m away, and the search then walked *west for
   forty actions* chasing frontiers while the one object it was looking for
   stood confirmed on its own map. Such a landmark -- not yet looked at
   from close on this storey, not within the takeover's rejection memory --
   is offered to RPT* at a fixed `landmark_prob` (0.85; the detector's word
   is the evidence, the oracle is not asked) at a standoff point
   `landmark_standoff_m` (1.5 m) from it on the agent's side, and visited by
   the same peek: walk there facing it, one look to each side -- a few
   close frames for the detector, and the takeover does the rest, or the
   landmark is marked inspected (`landmarks_inspected`).
5. **A\*** -- the weighted planner behind `_navigate`.
6. **Direct transit to the chosen room's most revealing frontier** under
   `explore` (`entry_frontier`; the gain-ranked goal of `_explore_goals`,
   re-aimed at the room's next one when the camera resolves it en route; a
   room with no frontier left -- to be confirmed on foot -- enters at its
   **doorstep**, the room's passable cell nearest the agent along the
   floor (`entry_doorstep`, `_nearest_entry`, since 2026-10-14: until then
   it was the vantage point, the middle of the room, and the walk across a
   room whose frontiers were all walked showed nothing its doorway did not
   -- the user's rule: no walk to the centre without a reason; the explore
   visit then finds no goal and the ledger finishes the room `explored`,
   stood in); a second-pass revisit enters at the vantage point,
   `entry_vantage`); **to
   the chosen room's vantage point** under `scan` (`entry_vantage`; re-aimed
   once at the vantage point's new position, then the centroid, if the mask
   re-segments under the route); or **to the closest frontier inside the
   chosen room** under `sweep` (`LoopSettings.entry_frontier`; a room with
   no frontier left keeps a centroid entry; a committed entry frontier the
   camera resolves en route is re-aimed at the room's next one, and a room
   with none left is released without the visit -- `frontier_exhausted` in
   transit); or **to the threshold of the chosen opening**, where the peek
   begins. A room the walk finishes before the agent is in it -- its last
   frontier resolved from the corridor, or a full glance turned inside it
   -- is released `exhausted` in transit (`finished_in_transit`). **A plan
   the planner refuses is re-aimed before the room is released** (since
   2026-10-13, `route_reaimed`): the room's next three frontiers by view
   gain under `explore`, then its doorstep, then its vantage point, then
   its centroid, the
   first the planner accepts becoming the entry in force; only when every
   one is refused is the room released `unreachable` (`plan_failures`).
   One refused plan is not an unreachable room -- the committed entry is a
   frontier cell the map re-snaps under as the agent closes in. Darden/000034
   (`4ae716bacfab`) released its bathroom `unreachable` twice on one refused
   plan each, the second time 2.4 m from it after a 65-action walk, with
   the toilet 1.3 m away at the step limit. The route memory's
   `no_progress` refusal -- the follower stuck on the route -- still
   releases the room as before. **Nor is a room missing from the registry
   for an action an unreachable room** (`room_missing_grace` 3, since
   2026-10-14): the watershed re-segments every action, and Collierville/
   000023's room 4 was gone for exactly one action at 170 -- the transit
   was released `unreachable` and the order re-solved without it, the room
   back at 171. The walk to the committed entry goes on (`room_missing_held`)
   and the room is given up only if it stays missing.
7. **Reset** -- arrival is the agent's own cell inside the room's mask
   (`arrived`), which restarts the visit counter.

**Room numbers are unique across the building** (since 2026-10-04): each
storey keeps its own registry, and a storey first entered starts numbering
after the highest pid any storey has handed out
(`RoomRegistry(first_pid=...)`, `FloorContextBank.new`), so `R0` names one
room in a recording, not one per floor. **A room that grows keeps its
number** (since 2026-10-05): the registry matched fresh masks to the last
tick's by IoU alone, and a room seen through its door is a sliver that
grows tenfold as the agent walks in -- the Hanson re-fly renumbered one
bedroom R11 -> R14 -> R16 while standing in it, and with the number went
the record of having stood in it (`entered=no` to the oracle). A pair
under the IoU threshold now matches when `containment_threshold` (0.6) of
the smaller mask lies inside the larger; IoU matches are consumed first,
so a split's larger half keeps the number and a merge's survivor is the
old room with the larger overlap. **A room keeps its number by where it
was born** (since 2026-10-07, `RoomRegistry(anchors=True)`): each room
carries its birth anchor -- the cell it was first instantiated at -- and
a fresh mask holding a previous room's anchor is matched to it before any
IoU score, oldest pid first. The IoU rule alone handed a split room's
number to the LARGER half, which is whichever side the agent has mapped
more of: the Allensville couch run's spawn room was R0 at step 0, became
R1 when a door 0.5 m away cut it off at step 8, merged back as R0 at step
10 and re-split as R2 at step 44 while the hallway the agent had just
walked into carried R0 -- and with it the spawn room's record of having
been stood in. The registry's memory of vanished rooms is the episode's
(`REGISTRY_MEMORY_TICKS`, bounded by `REGISTRY_MEMORY_ROOMS` masks), not
ten ticks, so the hallway comes back as R1 after a 34-tick merge instead of
as R2.

**Where a door cuts, and whose room the furniture is** (since 2026-10-05).
The partition is a clearance watershed over observed free space with a
confirmed door as an absolute boundary
(`core/mapping/topology/room_watershed.py`). A door *detected* from RGB-D
sits where the box's depth put it, and the Hanson recording had one 0.6 m
off its doorway, on the bedroom's own floor at a junction: the 0.75 m disk
carved there severed the bedroom around its bed into R0 and R6 and cut
nothing it should have. So the ObjectNav segmentation
(`scene_graph.DEFAULT_SEGMENTATION`, `door_snap_reach_m=0.9`) **snaps each
door's cut to the nearest choke** -- the medial-axis cell of least
clearance within reach whose disk parts the skeleton into two substantial
pieces (a dead-end notch is narrow too, but cutting it severs nothing) --
and sizes the disk to the passage; a door with no choke in reach (walls not
yet observed) keeps the plain disk. The merge then never joins basins that
lie on *different sides* of the carved mask and may still join two basins
of one room beside the cut (`merge_basins_by_dynamics(basin_sides=)`),
where before any border through the disk was unmergeable. Replayed on the
fifteen recorded floors of 2026-10-04 the change fixed the one it was
written for (bedroom whole, R7 and the corridor apart) and moved the
others by at most a room; it is a heuristic over the observed geometry --
a furniture passage narrower than the doorway within reach of the door
would be taken for it. **A threshold the agent walked through is a cut too**
(since 2026-10-07, `room_watershed.trail_thresholds`,
`threshold_snap_reach_m=0.5`): the scene graph keeps the agent's recent
trail and reads the clearance along it; a dip to 0.55 m or less with 0.30 m
more clearance within 1.5 m of travel on BOTH sides is a doorway, carved at
the severing choke within reach exactly as a snapped door is -- never a plain
disk, since no detection vouches for it -- so the room beyond separates the
tick the agent is through, not when its floor has grown a clearance peak.
A uniformly narrow corridor never dips and is never cut (the Allensville
hallway: 0.30-0.50 m for forty actions). Thresholds are sticky for the
storey (`ObservedSceneGraph.thresholds`, `threshold_events`). Objects are
credited to rooms by
`ObservedSceneGraph.object_room`: the landmark's cell when it is room
floor, else the nearest room floor within its footprint radius plus 0.6 m
-- furniture stands on cells the map reads OCCUPIED, which the watershed
never labels, so the bed, the desks and the wardrobe of that bedroom were
`room: null` and the room was a "living room?" from its one chair.

**Distances are measured the way A\* flies them** (since 2026-10-05).
The planner plans at the preferred standoff (`preferred_clearance_m`,
0.30 m) and relaxes to the floor rung (`RPTSettings.min_clearance_m`,
0.25 m since 2026-10-25; the body radius, 0.18 m, before) only when nothing
else gets through; the frontier inventory and the RPT* instance used to
measure every distance on the body-radius graph, so a threshold 3.5 m
away through a 0.5 m squeeze read as 3.5 m while the route around the
squeeze was 12.5 m (Hanson, actions 166-196: a 30-action approach bound
spent 5.8 m short on a route still being followed). Now
`accessible_frontiers(preferred_cost=)` and `build_instance(preferred_cost=)`
measure at the preferred standoff where it reaches and fall back to the
body radius where only a squeeze connects (`RPTSearchPolicy.preferred_cost`),
and a peek's approach bound is raised to what the route the planner
actually adopted needs (`peek_approach_extended` events). **The planner
squeezes rather than detours** (since 2026-10-13, `RPTSettings.detour_factor`
1.5 -> `WeightedAStarParams.detour_factor`): a route found at the
preferred standoff that is more than 1.5 times the route the body radius
allows is a detour, and the squeeze is returned instead
(`artifacts["detour_avoided_m"]`; one extra search at the floor rung). "The
safest feasible route first" is right for an aircraft; a ground agent that
walks 12.7 m round a room because a 0.5 m pinch on the 7.5 m way read as
0.25 m short of its standoff (Corozal/000076, actions 394-486: a committed
route through unvisited space replaced by one back through visited rooms)
is not safer, only late. **The floor rung is 0.25 m, not the body radius**
(since 2026-10-25, `RPTSettings.min_clearance_m`, also the "passable" map
every count, entry and frontier is measured on): on the 0.1 m grid the body
radius admitted a 0.4 m gap, and HM3D moves the agent without sliding
(`allow_sliding: false`, unlike Gibson) -- a 0.36 m body with a 30-degree
heading grid cannot thread one without a lateral touch, and a touch refuses
the whole step. 00824/000066 walked a route between a bed and a dresser
4-5 cells apart and spent 460 actions wedged there, fencing itself in with
its own bump marks. A 0.5 m gap (five free cells) is still a passage; the
cells the body has stood on this storey are free for planning whatever a
frame or a bump wrote there (`RPTSearchPolicy._walked_passable`).

Steps 2–6 run on the action a room's turn ends, so a released room is
replaced by a transit at once -- never by a throwaway floor-wide route. A
room whose budget ran out is **not** put on the visit cooldown (the fresh
estimate may rightly send the agent straight back), nor is a staircase
whose climb began or a room released `reclassified`; a room the live map
says is finished (exhausted or mapped) is never repeated by the
every-room-cooling escape hatch: the exploration fallback carries the
search instead. A turn the planner or the follower failed (`unreachable`,
`blocked`, `transit_timeout`) cools the room for `failure_cooldown_steps`
(30) only, not the supervisor's 120-action visit cooldown (since
2026-10-13): the map a room was unreachable on is not the map a few
actions on, and the 120 kept the toilet's room out of the order for the
rest of Darden/000034 (released on one refused plan at action 112, still
cooling at 217 with the agent 4 m from it). `room_search_loop.events` in the episode record lists every
transit, arrival, relabel, release and `stairs_taken` with its verdict and
local step count; `estimate_events` lists every oracle round with the nodes
shown, `p_present`, `elsewhere` and whether the reply was reused.
Runtime is not the constraint here, the judgement is: the oracle is asked
once per loop point and the search waits for the answer. The cheaper calls
stay bounded by construction: one classifier call per **new kind** of
object per room, with a verdict for a set of kinds reused across the
watershed's re-partitions rather than bought again (the trace probe went
from 22 classifier calls in 60 actions to 5 with identical decisions).
## The exploration fallback: a failure is a move, not a spin

Every way the decision pipeline can fail lands in one place,
`core/planning/object_search/exploration/exploration_fallback.py`, and comes out as a command that moves:

1. the **highest-utility reachable EXIT anywhere on the floor** -- not confined to
   the room in force, so a room whose routes all fail is left rather than
   spun in. Two kinds of frontier are not exits and wait for rung 3
   (`exits_first`, default **true**): an **object's shadow** -- the camera's
   ray ends at the nearest object, so behind every bed and cabinet a scan
   leaves a strip of unknown whose frontier sits beside the object
   (`shadow_margin_m` 0.30 beyond its footprint radius) and is no longer than
   the object could cast (`shadow_length_factor` 3 halo radii) -- and a
   frontier of a **room the target cannot be in** (the loop's `type:`
   exclusion). The 2026-10-04 Ranchester recording spent actions 61-102
   walking from one strip between the beds to the next (ten goal changes, a
   150-degree spin, back within 6 cm of where it started) while the passage
   to the stairs stood two metres away as a frontier the whole time,
   outranked because the strips were nearer. **What casts a shadow, and
   what is too much unknown to be one** (since 2026-10-14): a landmark of a
   `SHADOWLESS_CLASSES` class (clock, vase, book, bottle, cup, toy, stuffed
   animal, television) or with a measured radius under
   `shadow_min_radius_m` (0.25) casts none -- Collierville/000023 lost its
   doorway `O2` as "the shadow of clock 24" 0.5 m from it -- and a frontier
   with `shadow_exempt_m2` (2 m2) of unknown within `shadow_exempt_range_m`
   (1.5 m) of it, through the walls' blocking (`shadow_exemption`,
   `UnknownView.unknown_in_sector(max_range_m=...)`), is nobody's shadow
   whatever stands beside it: the strip behind a bed has the wall a few
   cells behind it, a doorway has the room's floor from the threshold on.
   The same recording dropped the 2.9 m frontier onto its toilet's room,
   16 m2 of unknown through it, as "the shadow of refrigerator 0" (halo
   1.0 m, three halos 3.0 m) and re-explored the house for 180 actions.
   `detect_openings` applies the same rule, so an opening is never lost to
   a neighbour's halo while a room lies behind it. A finished room's opening is
   NOT demoted: it is how the next room is found. A frontier inside the
   camera's **blind radius** (`near_blind_m`, 1.2 m -- a level camera 0.88 m
   up sees the floor from about there) is demoted too: it appears under the
   agent wherever it stops, and no step toward it resolves it. **The exits
   that remain are ranked by what a look from each reveals** (since
   2026-10-14, `_by_view_gain`, the loop's own ray caster): the unknown
   floor within the camera's range from the goal's cell over the walk to
   it, as the explore visit ranks a room's frontiers -- not the length of
   the boundary -- and an exit that reveals less than `min_gain_m2` (the
   loop's `explore_min_gain_m2`, 1 m2, by default) is a **sliver** that
   waits with the shadows (`slivers_demoted`, "reveals under 1.0 m2").
   Collierville/000036 (actions 99-113) walked from one sliver to the next,
   each resolved by the step toward it and the next one behind the agent:
   six turns left, six right, 180 degrees and back. Within a rung
   the fallback **commits**: the goal in force is driven until it is gone or
   refused unless another goal is worth `goal_switch_gain` (2x) its current
   utility -- the greedy per-action order, re-ranked by the agent's own
   turning, once produced a 180-degree turn toward one goal and a 210-degree
   turn back (`goal_kept` / `goal_switched` / `goal_outranked` in the stats)
   -- and the commitment survives a glance or a peek's look in between
   (`goal_memory_actions`, 12: until 2026-10-14 the goal was forgotten after
   ONE action the fallback did not own, and every glance re-ranked the
   frontier greedily);
2. the **stairs** by the explicit fallback rule (`floor_decision.py`:
   eligible, reachable now, storey worth visiting, worth per metre of
   approach, flight and fixed change cost), when the floor is exhausted.
   The normal way to the stairs is not this rung: the loop offers every
   staircase to the oracle and to RPT* as a node beside the rooms and climbs
   when the order says so. This is for the floor where that machinery has
   nothing left -- the room LLM is away, or no node is worth anything and
   the frontier is gone -- and a climb it makes is recorded
   `rule: fallback`;
3. the **demoted frontiers** of rung 1 (`frontier_demoted`): a shadow is still
   unknown space once every exit is spent, and so is the bathroom;
4. a **retired frontier** -- a goal dropped for a transient plan failure is
   still unknown space (a goal under the agent's feet is not retired but
   unreachable by construction: the converter has no action for a waypoint
   inside its arrival tolerance, and would spend an idle turn on it);
4b. a **squeeze** (since 2026-10-25, `ExplorationFallback._squeeze`): the
   frontier goals reachable at the body radius alone, routed by the policy's
   `squeeze_planner` (the planner with the body radius as its floor rung, 0.18 m
   under the 0.25 m everything above is measured at). A doorway the depth map
   draws 0.4 m wide -- walked on the way in -- closes a whole wing at the rung
   (00835/000064: five frontier goals a squeeze away, 458 relocations instead);
   the target closing's standoff rings try the same planner before the one
   through the unknown;
5. a **relocation** to the farthest reachable known cell, for a vantage point
   the map may show a frontier from -- never one within `relocation_revisit_m`
   (1.0 m) of a vantage point already walked to during the current run of
   fallback actions (since 2026-10-25: a closed pocket's two ends were
   relocated between 458 times in 00835/000064 while the rungs below never
   got a turn);
6. a **footing sweep** (since 2026-10-05) when nothing on the observed map is
   reachable and at least `footing_unknown_fraction` (0.25) of the cells
   within `footing_radius_m` (1.2 m) of the agent are unknown: an agent that
   has not moved stands on a disk of unknown under the camera's blind
   radius, and unknown is impassable; one LOOK_DOWN, a circle and a LOOK_UP
   map it (`CameraController.begin_inspection(reason="footing")`, once per
   spot; a protocol without LOOK actions skips it);
7. an **escape probe** (since 2026-10-13, `core/planning/object_search/exploration/escape_probe.py`) when the
   inflated map has no way out of the pocket the agent stands in -- the strip
   behind a sofa, the ring a counter's top draws round the agent -- while the
   body may well have one. The grid is read **without inflation** along every
   turn heading (`EscapeSettings.range_m` 3.0 m): the free or unknown pocket,
   the first occupied band, and the free or unknown floor beyond it. A heading
   whose band is at most `max_barrier_m` (0.4 m) thick with at least
   `min_beyond_m` (0.5 m) of floor beyond it is a candidate, scored by the
   floor beyond minus twice the band minus the turns to face it -- and so is
   (since 2026-10-25) a heading with at least `min_pocket_m` (0.5 m, two
   steps) of free floor BEFORE anything occupied, whatever lies beyond,
   scored by that run: a wedge whose open side ends at a bed too thick to
   probe had no candidate at all (00824/000066, 450 actions); the best gets
   one turn toward it or one forward step along it, and **the simulator
   decides**: a step that moves re-maps the pocket and the ordinary rungs take
   over, a step the simulator blocks is remembered for that cell and heading
   for `memory_actions` (80) and the next probe takes the next heading. The
   planner's map is never rewritten (LESSONS 2026-10-08: opening "genuine"
   obstacles around the start is the one way A* threads a real wall).
   Collierville/000036 (`6bc7ace42dcf`) spawned behind a sofa and spent 488
   actions on the hold below, twice; with the probe it was out in 12.
   `escape_probe` in the episode record carries the settings, the counts
   (`probes`, `turns`, `steps`, `blocked`) and the last probe;
8. a single hold only when even that has been done here, where the idle
   turn is the one action that changes anything.

It is reached from the loop whenever no room is in force, and from
`RPTSearchPolicy.plan` whenever the decision itself raises -- an A* with no
path, an RPT* instance that cannot be built, a bug. A **room LLM** failure
(timeout, bad JSON, refused uniform prior) no longer ends the episode: the
loop records it, arms a doubling back-off (25 → 50 → 100 → 200 actions,
`FallbackSettings`) during which no model is asked, and the fallback carries
every action until the model answers again, at which point the loop resumes
its steps. A **detector** failure is handled the same way in
`perception/cycle.py`: the frame carries no detections, is counted as
skipped, and the service is left alone for the back-off. Every failure is
counted, kept with its type, message and origin under
`exploration_fallback.failures` in the episode record, and logged once per
type. Before this, an empty hold cost one idle `TURN_LEFT` per action until
the step budget ran out -- the "camera spinning in place" of the Ranchester
recordings -- and a model failure was an agent error.

## Glances: where along a route a look to the side pays

`core/planning/object_search/exploration/path_glances.py` (`GlanceSettings` on the policy, since 2026-10-05).
The camera's cone is narrow and the map is what it saw: an agent walking a
corridor sees the corridor, and a door a step to the side passes through
the edge of the frame or not at all (Hanson, action 35). Turning in place
costs no path length -- nothing under SPL -- but under the 500-action
budget a full circle every few steps is unaffordable, so the question is
*where* along the route a look pays, and which look. The scheduler scores
the route in force every `revalue_actions` (2) and whenever it changes:
candidate points one every `stride_m` (0.5 m) up to `horizon_m` (6 m),
each with the route's heading there; for each, the unknown floor a glance
would sweep -- left (from the heading's cone out to a right angle), right,
or the full circle -- by optimistic rays through the unknown that stop at
walls (`core/planning/exploration/view_gain.py`), **minus what the walk
reveals anyway** (the forward cones along the whole route ahead), so a
doorway's room counts once, at the point it shows best: the user's "one
step forward and all three areas show; two and the first is hidden
again". The best gain per action wins -- a side glance costs the turns
out and back (six), a circle the turns the camera's field of view needs to
see it (ten at 30 degrees with the 79-degree camera, not twelve) -- if
it clears `min_gain_m2` (3) and `min_gain_per_action_m2` (0.5): six
forward steps into unknown space sweep the cone over 1.5 m of new floor,
around 10 m2 in a room, so walking is the better buy wherever the unknown
is ahead; a doorway beside the route shows the walk nothing and a glance
5-15 m2 (the Hanson routes replayed on their final maps: 14.8 m2 at
action 33, 7.6 m2 at 164). **A bird in the hand** (`lookahead_discount`,
0.5, since 2026-10-13): a candidate's rate is discounted by how far along
the route it lies, down to half at the horizon, because the route and the
map change before the agent gets there -- the Darden recording
`ec8d17aeaaaa` scheduled 32 glances and took 4 of the geometric ones, the
rest re-planned away as the route moved; a look the agent can take now is
worth more than one of equal rate planned six metres on. **The plan is
committed** (since 2026-10-21): the discount alone did not stop the churn
(the fifteen episodes of 2026-10-21: 27 scheduled and 4 geometric glances
taken in Collierville/000000, 27 and none in Markleeville/000000), so the
plan in force is re-scored as the map grows and kept while its gain clears
the floors; only a candidate at or nearer than its point may replace it,
never a farther one. **A circle pays for its extra turns with its rear
sector alone** -- the circle's extra over the better side glance must
clear the rate floor over the four turns it adds -- and **no circle is
taken inside a room the search has finished, that has no frontier left, or
whose frontier walk is in force** (a side glance still looks OUT of such a
room, through its door): the recordings promised 5.5 and 18.5 m2 for
circles that revealed 0.8 and 1.9. When the agent
reaches the point the look is performed, one in-place turn per action, as
a suspended phase like the warm-up: the loop is not ticked and not charged,
the supervisor's deadlines pause -- and so, since 2026-10-13, does the
**blocked clock** behind the supervisor's `blocked` verdict
(`blocked_abandon_s`, 6 s): a look turned in place after a blocked step is
not the follower still blocked. Corozal/000142 (`51eff3fa1637`) took one
blocked step, a ten-turn glance, and had its next four transits released
`blocked` one action in. What the pause must NOT move is the mark a plan is
measured against (`_plan_s_at_goal`, "a plan newer than this was made for
the goal in force"): until 2026-10-13 it did, so a committed route planned
once and followed without re-planning read as "no route" after a glance,
and every transit longer than `plan_grace_s` (5 s) was one glance away from
an `unreachable` release (Darden/000034 at action 493, 2.4 m from its
bathroom after a 65-action walk) -- and the committed route's watchdogs are told
the pause (`glance_scheduled` / `glance_started` / `glance_complete` /
`glance_aborted` events in `episode_info()["glances"]`; the planned and
active glance in the per-step `search` record and on the HUD, and -- since
2026-10-14 -- `glance.here`: the left, right and full gains at the agent's
own position from the last evaluation, so a recording can say *why* no
look was taken at a given action. Darden/000190 (`ec8d17aeaaaa`, 2026-10-14)
walked three metres north with a circle 2.3 m ahead valued at 32 m2 and a
right look at its own position valued lower, took a left look at 5.02 and
the circle at 5.27; the user's eye wanted the right look at 4.54 -- the
rule is the one the user describes, the judgement is the ray model's, and
the record now shows the numbers it weighed). **A
completed full glance is a completed look-around** and is recorded in the
scan ledger (`source="glance"`, `rotations_recorded`, since 2026-10-13):
the room the agent stood in is finished by it like a room scan finishes a
room, and the loop's visit in force ends the same action
(`finished_by_rotation`) -- the Darden recording turned a full glance
inside the room it was walking into at action 61, then walked on to that
room's vantage point and turned the same circle again at 94. A spot within
`scanned_m` (1.5 m) of a completed rotation (the warm-up, a room scan, a
full glance -- the scan ledger) is never glanced from -- whatever is
unknown from there is beyond range or behind furniture, and the 4.4 m2
"right" glance that recording took 1.0 m from its own full glance (actions
82-85, three turns back after) revealed nothing -- and a target sighting
or a stair traversal aborts a glance; the warm-up, a peek's look, a scan's
rotation and the takeover never start one. After a glance no new one
starts for `cooldown_actions` (4; the former 8 swallowed a 15 m2 right
glance whose point the agent walked past at action 56 of that recording,
three actions after a two-turn cue glance), and a cue glance no longer
discards the geometric plan in force. **A look is not repeated from beside
where it was taken** (`repeat_m`, 1.0 m, since 2026-10-13): the scheduler
remembers every completed side and cue glance -- where it stood and the
bearings its frames covered, the field of view included -- and neither a
side glance nor a cue whose sector a look from within a metre already
swept is taken again (the re-run of that recording looked left through the
same doorway at the same dining room at actions 60 and 68, 0.25 m apart:
the unknown beyond a doorway stays unknown after the look, beyond range or
a hole in the mesh, and the optimistic scorer values it again every time).
**A cue of an ordinary class clears the geometric rate floor** over its
turns out and back (`max(cue_min_gain_m2, min_gain_per_action_m2 * 2 *
turns)`: 3 m2 for three turns, 2 m2 for two) -- that re-run took nine cue
glances of 1.7-2.6 m2 at six actions each, a chair or a cabinet at the
edge of a room already mapped; the target's own class and its home objects
still need no gain. `enabled=False` is the ablation.

**Wide glances** (since 2026-10-21, `wide_rad`, 120 degrees): a right angle
with the 79-degree camera shows the floor to about 130 degrees off the
route; a pocket beside and a little behind the agent -- the alcove just
passed, the room whose door opens back toward it -- lies in the 130-160
band only a wider look reaches. Every candidate point is scored for the
120-degree look as well (`wide_left_m2` / `wide_right_m2` in `glance.here`),
and it is taken only when the extra band alone pays for the two extra
turns each way at `min_gain_per_action_m2`; otherwise the look stays a
right angle. `turn_deg` on the scheduled/started events, `wide: true` on
the plan and the action's `info`, `wide_scheduled` / `wide_started` in the
stats; `wide_rad` at or under `side_rad` disables it.

**Cue glances** (since 2026-10-05, `cue_*` settings): the scorer values
unknown floor, and a small room beside the route shows it little, but
the detector sees the room's furniture at the edge of the frame. A box
of at least `cue_confidence` (0.4), spanning at least `cue_min_box_frac`
(30 %) of the frame's height, cut off by the frame's left or right
edge (`cue_border_px`) is a cue to turn that way -- enough turns to centre
the object and one more, at most `cue_max_turns` (3), not a right angle.
The target's own class and its home objects are cues outright (the
takeover refuses a border box, so turning is how it starts); any other
class only while that side still holds `cue_min_gain_m2` (1.5) of unknown
floor; a class already glanced at on that side within `cue_repeat_m` (2 m)
is not a cue again (`cues`, `cues_repeated`, `cues_without_gain` in the
stats; `cue` on the glance's events and HUD line). Hanson 2026-10-05,
action 101: a bathroom vanity read as `cabinet 0.84` on the left edge of
the frame, the scheduler glanced right twice toward a larger unknown, and
the toilet beside the vanity was reached 250 actions later. The unknown
the scorer counts excludes what the sight ledger has settled: a window is
no longer worth a look for the garden behind it.

## The sight ledger: unknown that is not worth a step

`core/planning/object_search/world_model/sightlines.py` (`SightSettings` on the policy, `sight` in the
configuration and the episode record, since 2026-10-05). The frontier --
free beside unknown -- is where exploration goes, and two kinds of unknown
are not worth a step:

- **Looked-through unknown.** Every action, after the map has taken the
  frame, the camera's cone is cast over it: a ray runs through known free
  cells and ends at the first occupied or unknown one; an unknown cell it
  ends at inside the floor's visible band -- beyond the blind radius under
  the camera (from the pitch and the vertical field of view) plus
  `near_margin_m`, within `far_m` (2.5 m: past that the depth image's floor
  samples thin out and an unknown cell between two seen rows is a sampling
  hole) -- is a cell the depth should have resolved and did not: a balcony
  railing with the garden below, a window, a glass door, a hole in the
  mesh. Looked through from `min_looks` (2) distinct poses (binned by
  `pose_bin_m` / `pose_bin_deg`) it is **settled**; about a cell and a half
  behind it (`depth_cells`, 3 half-cell samples) with it, so the free cell on
  the boundary stops being a frontier cell. The Hanson re-fly walked to both ends of a balcony it
  had seen whole from its threshold (actions 25-51, two openings, two
  peeks).
- **Pockets.** A connected component of unknown of at most `pocket_max_m2`
  (3 m2) that does not reach the edge of the map is enclosed by known
  cells and leads nowhere: the strip behind a bed against an observed
  wall, the island between the spawn point and the bed in front of it
  (the blind radius leaves the floor under the camera's own feet unseen).
  The same recording spent actions 51-82 walking back to such an island
  (O4, valued 0.01, visited because it was near) and 150-192 on the strip
  behind a bed (O9). A toilet, a bed or a sofa does not fit in 3 m2 behind
  a wardrobe; a cup would, and that is the trade the bound makes. **But a
  pocket is settled only when nothing could stand in it** (since
  2026-10-13, `pocket_max_width_m` 0.6): its inscribed width -- twice the
  largest distance from one of its cells to its edge -- is under 0.6 m, or
  the agent has itself stood on it (the spawn island). The Darden
  recording `ec8d17aeaaaa` and both of its re-runs saw a bathroom's walls
  through its door, which enclosed the unseen half of the bathroom -- 0.9
  m2, 0.8 m wide, with the toilet in it -- and the area bound alone settled
  it: no frontier, no opening, no visit, three times, with the agent
  standing a metre away inside the success region at the door. A strip
  behind a bed is 0.3 m wide; a toilet needs 0.6.

Settled unknown is written OCCUPIED on the map the **frontier logic**
reads (`ObservedSceneGraph.frontier_world`, through `resolved_provider`):
the frontier inventory, the per-room `frontier=` counts the oracle sees,
the openings, the exploration fallback's exits and the glance scorer's
rays. The planner keeps the real map (unknown is never inflated, occupied
is; a settled railing must not shrink the balcony the agent may still
stand on), and so does the display, which draws settled unknown a darker
grey (`settled_cells` in the panel metadata). A settled cell that later
turns out to be floor is known from then on and the mark is void by
itself. The ledger also keeps every pose the camera stood at on each
storey, which is what the scan ledger's `seen_through` verdict reads.
`enabled=False` is the ablation, in which every unknown cell is an exit
again.

## Frontier sweep and the action economy

Every action counts against the public 500-action cap, so the frontier explorer
is built to spend them moving toward unmapped space (the Ranchester recording
`e7c4f2ad5402` that motivated this spent 48 % of its actions on idle turns,
stair inspections and heading recovery):

- A boundary with no known-free path is never proposed. Up to
  `SweepSettings.plan_attempts` goals are tried per action, so a refused A*
  costs no idle action; a refused *transit* route ends the room through the
  supervisor's `route_failed` exit and the next room is chosen on the same
  action.
- A frontier cluster's goal is its **member cell nearest the centroid**, on
  the boundary. The centroid itself lies inside seen floor for the arc-shaped
  clusters a depth range produces, which made every goal "no longer
  informative" on the action it was adopted.
- A committed frontier goal is kept while it is passable, still has unknown
  space near it and still lies near the room being swept. A re-segmented room
  mask alone does not drop it; a boundary the camera has resolved does.
- The optional extra look-around after a swept room is off by default
  (`SweepSettings.room_scan_turns = 0`): the loop's termination rule is *N
  steps or nothing left*, and a swept room is released on the action its last
  goal resolves. This is separate from the mandatory first-entry interior peek
  and its measured 180-degree scan. The supervisor's 30 s stall and 90 s budget clocks remain as
  backstops behind the caller-driven exits.
- Every adopted route records `route_replaced`: why the route before it was
  dropped (`goal_changed`, `route_obstructed`, `frontier_completed_or_invalid`,
  `room_completed` ...). `route_commitment` stats count clears per reason.

## Camera, perception and transitions

`SEARCH → APPROACH_STAIRS → TRAVERSE → CONFIRM_DESTINATION → SEARCH` retains
connector, direction and progress across turns. Recovery uses the measured
reverse trail including the source-platform approach. Unconfirmed recovery
ends visibly in `SAFE_HALT`, not silent room search on a tread. Height
hysteresis alone cannot finish either traversal or retreat.

Positive camera pitch looks down. One controller arbitrates all pitch: a
location-deduplicated bounded inspection, a latched stair view, or ordinary
level viewing. A look-down inspection starts for one of three named reasons,
recorded in the building events: the floor has no frontier left
(`floor_exhausted`), a stair detection is pending (`stair_detection`), or the
floor allowance is spent and `periodic_inspection_actions` have passed since
the last sweep (`periodic`). The last two wait until no route is committed --
an inspection mid-route costs its own actions plus the turns to recover the
heading it leaves behind. STOP carries no camera action. Raw detection runs on
every observation; committed transitions and inspection views are quarantined
from global room/object fusion. Target closing has its own pitch-aware evidence
and takes priority even over committed stair tasks. Coherent depth pixels are
projected at their actual image coordinates, with frame/floor/association
rejection reasons recorded. No bed/sofa proximity blacklist is used.

**Object identity is per class, and same-class instances are told apart by
size** (since 2026-10-07; `core/mapping/objects/landmarks.py` with
`class_votes=False`, wired in `core/planning/object_search/rooms/floor_context.py` from
`RPTSettings.landmark_dedupe_radius_m` / `landmark_footprint_iou`, and
`core/planning/object_search/perception/cycle.py`). Every projected detection carries a
**footprint**: a disc around its world centroid, half the box's width at its
depth (`perception.footprint_radius_m`, clamped to 0.15-1.5 m). **Classes
never merge**: a detection associates only with a landmark of its own class,
whatever the proximity or the footprint overlap -- a cup on a table and the
table are two instances in memory, a toilet and the bathtub beside it are
two, a vase on a cabinet is its own landmark. Every cross-class rule the map
ever had collapsed distinct objects resting on or beside one another into
one and cost a target lock: the two vases on the cabinet in the first frame
of the Allensville couch run (dropped as `same_frame_association` on every
frame by the 0.70 m centroid radius), and the toilet and bathtub 0.5 m apart
in the toilet run, voted into one landmark that flipped to `bathtub` and had
the toilet's lock released as contradicted by the map. **Two observations of
one class are one instance** when their centroids lie within
`landmark_dedupe_radius_m` (**0.35 m**, re-observation jitter -- the ported
0.70 m merged two dining chairs) or when both carry a measured footprint and
the discs overlap by IoU ≥ `landmark_footprint_iou` (**0.25**, `landmarks.
disc_iou`, the lens formula: about one radius apart for equal discs, so a
bed re-seen from its other side with the centroids 0.8 m apart is one bed and
two chairs 0.3 m apart are two chairs), and when the height is within 0.35 m
of the landmark's first measured height (two of a kind stacked in plan view
on different levels are two; the band stays at 0.35 m because one object's
measured centroid height swings that much with viewing range -- the
Allensville toilet read 0.33 m from 5 m and 0.64 m from 1 m). A second box
of the same object in one frame is `same_frame_association`; a box of
another class over the same pixels is its own landmark. The per-frame alias
filter (`object_evidence.deduplicate_detections`: a box of another class
over the **same pixels**, IoU ≥ 0.9 -- one crop, two prompt names) is the
one cross-class rule kept, and it acts on the detector's output of a frame,
not on the map. The detection floor (`RPTSettings.detection_confidence`) is
0.30, the takeover's own tracking threshold -- a landmark needs two
observations to be confirmed, so a weak box costs nothing alone, and the
former 0.35 dropped a cup at 0.34 on that table. A landmark is **confirmed**
-- shown to the room classifier, the node oracle and the HUD, and allowed to
start target evidence -- once it has `min_observations` (2) observations; a
lone detector flicker of a wrong class is a one-observation landmark the
search never acts on. Target evidence is judged on the landmark's class --
the box's own now -- and the takeover verifies a candidate on two consecutive
frames of its own and the context check (above), not on the map's word.
**The class vote** of 2026-10-04 -- co-located detections of any class
folded into one landmark as votes, the plurality naming it, a `sofa` box on a
confirmed `bed` outvoted (`contradicted_by_map`, `map_rejections`,
`map_releases`, `outvoted_detections`) -- stays in the library behind
`ObjectLandmarkMap(class_votes=True)` for its other users and reads zero in
this runtime's diagnostics.

**Confirmation is no longer one-way** (since 2026-10-21, `landmark_max_misses`
4): a confirmed landmark that stands in a CLEAR view -- its anchor inside
the frame by `landmark_miss_margin_px` (40), between
`landmark_miss_min_range_m` (0.8) and `landmark_miss_max_range_m` (3.5) away,
the depth at its pixel reaching at least as far as it (a shorter reading is
an occluder, no reading is no view) -- without a detection of its class is
**missed** (`perception/cycle.py` `_count_misses`, normal uncommitted views
only), and a miss counts only from a viewpoint distinct from the last
counted one (`landmark_miss_viewpoint_m` 0.5 m of travel or 30 degrees of
turn: a bad angle held for four frames is one miss). After as many misses
as `max(landmark_max_misses, its observation count)` it is **retracted** --
the negative evidence must outweigh the positive: the twice-seen table after
four, a sofa seen twelve times after twelve; a flat four retracted 29 of
Markleeville's 51 confirmed objects in the first run, most of them real --
out of `confirmed()` (the room's evidence, the target landmark nodes) but
kept on the map, and the next detection that folds into it reinstates it
with its misses cleared.
Darden/000000 (2026-10-21): YOLO called the kitchen's dining table `bed` at
0.52 twice, and the kitchen was a "bedroom" for the rest of the episode;
now the room's evidence loses the table at the next scene-graph update,
the evidence gate takes the label back to `unknown` (or the model is re-asked
on what remains) and the label history records it. `landmark_misses`,
`landmark_views_occluded`, `landmark_retractions` in `perception.counts`,
the retractions themselves in `perception.retractions`; 0 restores the
one-way confirmation.

**The room classifier's taxonomy is residential** (since 2026-10-21,
`core/mapping/topology/room_classifier.py` `HOME_LABEL_SET`): kitchen,
bedroom, bathroom, living room, dining room, office / study, hallway /
corridor, foyer / entryway, closet / pantry / storage, laundry / utility,
balcony / patio and garage are offered first (the clinical labels the
classifier's other users keep follow), synonyms and slashed forms are read
as the set's labels (`LABEL_ALIASES`), and the prompt names the detector's
confusions. The classifier is told each room's **geometry** -- floor area,
shape (the ratio of the mask's principal extents) and the doorways off it
(`scene_graph._room_geometry`) -- beside the objects, and a room with
nothing in it that reads as a passage (long and narrow, or two or more
doorways, 2 m2 or more; `RoomLabelSettings.shape_only_rooms`) is asked
about on its shape alone, once per geometry bucket: a hallway has no
furniture to name it, and `unknown` left the oracle to value every corridor
as a possible bedroom. A shape-only label is WEAK and provisional -- it
rules nothing out -- and the first object that appears takes over.

### The target's depth on its own pixels: the `segmentation` setting (2026-10-12, off by default)

A detection box overhangs its object and admits whatever stands **in front**
of it. The box pipeline reads a detection's range as a low percentile of the
depth in the box's central quarter (`core/mapping/objects/geometry.py::
robust_bbox_depth`, then the connected near-surface component in
`perception._supported_centroid`), so when a nearer object overlaps the
target's box -- a chair before a sofa -- the percentile is the chair's, the
centroid and the near edge (`range_near_m`) land on the chair, and the
terminal test of the closing (`target_closing._fresh_measured`, the smaller
of the two) reads "arrived" a metre short of the sofa. Frames of exactly this
kind STOPped wrong in the Gibson recordings.

`RPTSettings.segmentation` (**False** by default; `--segmentation` on the
`gibson.run` / `five_scene` command line, `SEGMENTATION=1` for
`run_benchmark.sh`) adds one step: every frame's **target-class boxes** at or
above `detection_confidence` go, in one call, to the box-prompted Segment
Anything service (`tasks/mapping/scene_graph/serve/segmentation_server.py`,
MobileSAM by default, `tasks/planning/objnav_benchmark_runtime/services/http_segmenter.HttpSegmenter` the client,
`--segmenter-url`), and `perception.observed_objects` reads each of them on
the returned mask: the depth is `robust_mask_depth` over the mask within the
box, the near-surface component is sought over the whole box on the mask's
pixels only, so neither the centroid nor `range_near_m` can land on an
occluder the mask leaves out. The coherence gate (35 % of the patch within
the ±6 % band) is not applied on the mask path -- the mask has answered "one
object?", and a deep object seen whole (a bed from its foot, a sofa at an
angle) legitimately spans more than the band; `supported_fraction` is still
recorded. Everything downstream -- association, the 3-D memory, the closing's
phases -- is unchanged, so the two pipelines differ in exactly one thing:
**which pixels count as the object.**

Boxes of other classes are read on their box as always. A target box the
service could not mask this frame -- a transport failure, the back-off it arms
(`exploration_fallback.SEGMENTER`, the detector's doubling rule) -- is **not**
read the old way: its row says `segmentation_unavailable`, the frame is
counted (`segmenter_failures`, `segmenter_skipped`), and the comparison stays
clean. Identity drift of the service (another checkpoint answering) ends the
run like the detector's. The policy refuses to start with the setting on and
no segmenter; `run.json` records the segmenter's checkpoint hash, device,
configuration and URL under `method.segmenter`, `configuration()` says
`perception_fusion: .../target-masked`, and the HUD tints each target mask
magenta on the camera panel. The per-frame cost is one SAM call on frames with
a target box (0.3 s on four CPU threads with MobileSAM; `steps.jsonl` carries
it as `segmenter_http`). Tests:
`tests/test_segmented_projection.py` (the occluder scene with and without the
mask, the failure families, the HUD), `core/mapping/segmentation/tests`, and
`tasks/mapping/scene_graph/tests/test_segmentation_service.py`.

## Models and vocabulary

YOLO-World X-v2 remains the default; pretrained LLMDet Swin-L and explicit S/L
checkpoints remain selectable. The Gibson context vocabulary includes `stairs`
and `staircase`, neither a goal category nor a navigation oracle. Semantics can
request inspection; observed step geometry must independently support a portal.
Since 2026-10-05 it also carries **distractor prompts** -- `toy`, `rocking
horse`, `stuffed animal` -- and `bathtub` (a home object of the toilet the
vocabulary lacked): an open-vocabulary detector scores every prompt it was
given against the crop and lands on the nearest, and Hanson/000001 stopped at
action 14 on a child's ride-on horse read as `chair` at 0.9, 11.7 m from the
nearest chair. A distractor is never a goal, never a home object and names no
room; it only gives the wrong match somewhere else to go. **A detector
service started before this change must be restarted** with the new
`--print-vocabulary` output; the health check pins the class list.

Restart a dedicated detector with the exact `gibson.run --print-vocabulary`
(or `hm3d.run --print-vocabulary`) output. A former 24-prompt service is intentionally rejected. HTTP checks pin
classes, checkpoint, configuration, runtime and prompt-feature provenance on
health and inference responses. See the [CPU detector runbook](../../mapping/scene_graph/serve/README.md).
Do not reconfigure another mission's service or place inference on Habitat's GPU.

## Persistent display and evaluation boundary

The recorder reserves **one occupancy map for each of K configured storeys**,
not K layers within every storey. All slots start UNKNOWN, gray; discovery IDs
bind slots without surveyed floor numbering. Each retains its full grid, room
partition, objects and trail. A shared viewport grows with the observed house
footprint in metre-sized increments; all floors use the same metric scale. Maps
stack or sit side-by-side to fit the house aspect ratio. Full occupancy arrays
are saved in `floor_maps.npz`, room labels in `room_partitions.npz`, without
pickled state. Display counts remain recorder-only, never in policy
settings or episode knowledge. Unvisited slots have no inferred elevation.

The dashboard frame (`visualization.render_dashboard`) reads, left to right:
the camera with the detector's boxes and the depth; minimalist floor plans with
thin logical room outlines on **every discovered floor**, short `R<n>` identifiers,
the room in force ringed amber and the next node green, a compact square for a
seen staircase, the robot, a subdued trail, the committed route and a scale bar.
Object dots, semantic names, probabilities, frontier diamonds and the visit-order
chain are deliberately absent from the maps. Those details remain in the **search column**
(`search_panel.py`): the target and the oracle's split between the listed
nodes and elsewhere, the RPT* visit order with its head (`S0` is a
staircase), the room in force with its local step count or the node in
transit, every room's type (a `?` marks a weak, one-object label), oracle
probability and the model's reason, frontier left, time spent and when it
was last inside, the objects seen in it, every staircase offered with its
probability, direction, the climb's cost in metres and the storey beyond,
the confirmed objects by room, and the loop's last events. All of it comes
from `method_snapshot(policy)["search"]`, which is also written to every
`steps.jsonl` row and, in short, to `live.json`, so the video and the trace
cannot disagree.

GT goals, topology, occupied cells, heights, distances and navmesh queries are
evaluator-only. The policy gets RGB, metric depth, pose and public task/action
information. Gibson reference-floor annotations are incomplete for full-building
ObjectNav; a real unannotated instance cannot be verified by its own detector.
The development-v2 native vertical validator remains **0.60 m**, distinct from
the policy's unchanged **0.24 m** maximum observed tread step.

Coverage is ever-observed occupancy area, not true coverage; it excludes initial
gain and transient terrain, and deduplicates revisits by stable floor ID.
Recordings and interpolated replays never feed evaluator data back to decisions.
Passing tests, encoded videos and completed episodes are not navigation success.

## Adapter boundary — keep benchmark rules out of shared code

Each simulator branch supplies:

1. An `ObjNavEnv` implementation: published episode loading, reset/step, episode
   termination and evaluator-only measurements. `measure()` is not called early.
2. Exact scene/split identities and a `TableLabelMapper` with that dataset's
   category spellings, detector prompts and accepted aliases. The infrastructure
   registry does not claim any dataset is implemented.
3. Explicit camera intrinsics, registered metric depth, body dimensions, action
   lengths/angles, sliding policy, step budget and protocol/version metadata.
4. Its published success/STOP/distance/path-accounting rules. Shared defaults
   remain **STOP required, 3D observed path length, zero initial accumulator**.
   Override `EvaluationSettings` only when the adapter's protocol requires it.
5. Data/model provisioning and checks, including GPU ownership. No weights,
   licensed scenes, containers or remote services are downloaded/launched by the
   shared evaluation entrypoint.
6. A manifest identifying the actual dataset/assets, model checkpoint/revision,
   package versions, seed and protocol. Keep credentials out of this metadata;
   the shared service URL validator rejects embedded credentials/query strings.

Gibson and both HM3D versions reuse the Habitat bridge. The HM3D versions
share one adapter -- [`hm3d/`](hm3d/README.md) -- with two versioned protocol
objects, because only their episodes and scene release differ. RoboTHOR needs an AI2-THOR environment bridge but reuses
the policy, action conversion, provenance and recording; neither it nor MP3D is
implemented here.

The Habitat bridge takes an explicit `navmesh` policy, because which navmesh an
episode runs on decides what is reachable and therefore sets `l` and every SPL.
`"published"` uses the shipped mesh verbatim; `"agent"` is habitat-sim's own
load-then-recompute-at-the-agent's-dimensions behaviour, which is what
habitat-lab's ObjectNav actually runs. Whichever an adapter picks,
`navmesh_provenance()` records what was in force.

### Explicit embodiment and method settings

`RPTSettings` (`core/planning/object_search/settings.py`) carries `body_height_m`,
`body_radius_m`, `preferred_clearance_m`, `min_clearance_m` and `stop_distance_m`;
each task's `run.py` sets the embodiment from its protocol's agent, never from a
benchmark name. Physical inflation is never relaxed below `min_clearance_m`
(0.25 m), itself never below the supplied body radius. The stop distance
is a method choice, not an override of the evaluator's success definition.

`RPTSearchPolicy` records its planner, loop, fallback, sweep, door, room-label
and route settings in `configuration()`. `run_evaluation`
uses the same converter settings in the actual headless agent, so route
arrival/progress checks agree with the executor.

`HabitatRGBDSimulator` separately requires body height and radius; camera mount
height is not body height. Under `navmesh="published"` it loads the supplied
navmesh verbatim and never recomputes it; under `navmesh="agent"` it uses
habitat-sim's own recomputation and refuses a navmesh path that is not the
scene's sibling. It never moves a published start's (x, z); under `"agent"` it
fixes the start height once, from the recomputed mesh, so the first forward step
does not carry a height jump into the episode. Only the explicitly synthetic
renderer smoke constructs its own fixture navmesh.

## Shared execution and locks

Construct the environment, policy, label mapper and adapter metadata, then call
`run_evaluation` with an explicit output directory. It uses the existing
`run_benchmark`/`MetricsLogger`, not a second scoring loop, and closes the
environment even on recording failures. Extra `evaluation_diagnostics()` is an
optional recorder-only hook; environments without it remain recordable.

`progress=` takes a `(index, total, record)` callback, so an adapter's CLI can
report each finished episode — a thousand-episode run that prints nothing until
it ends is unusable. It runs *after* the recorder's own hook, which is what
finishes writing an episode's artifacts, so a caller can never displace it.

For frozen evaluation:

1. Use `evaluation_configuration(config, episode_ids, settings, policy=policy)`
   to capture actual Python sources, the policy, converter, selected episode
   order and execution settings alongside adapter metadata.
2. Call `freeze_configuration` with a new lock path and an explicit
   `development_note` describing prior tuning/exposure. Existing locks are not
   overwritten.
3. Pass that lock to `run_evaluation(frozen_lock=...)`. It checks the actual
   execution configuration before output creation or reset, not merely an
   earlier preflight. To claim a complete published split, also supply the
   publisher's exact ordered `expected_episode_ids`.

There is no hardcoded scene count or episode count. A lock is a reproducibility
commitment, **not proof that validation was unseen**. It never creates a
held-out or SOTA claim. Full benchmark equivalence, tuning disclosure and
reference-version caveats remain the adapter's responsibility. Dashboards only
summarize recorded rows and do not label arbitrary subsets as complete splits.

## Recording and visual-only replay

Recording is optional and loads plotting/video dependencies only when requested.
It keeps images/maps out of per-step JSON; the JSON records compact observed
state, actions, predictions, paths and reasoning. Model service metadata pins
checkpoint bytes and vocabulary atomically with inference. Missing or changed
services raise rather than silently changing the evaluated method.

**The log is the record; the video is for a person watching.** A recording
directory holds, per episode:

| file | holds |
|---|---|
| `steps.jsonl` | one row per decision: pose, action and the follower's status, the command (route reason, kind, waypoints), the method's snapshot -- state, the search column (rooms with probability / type / frontier / entry / distance, the RPT* order, the node in force and the next, the transit's committed room and entry point, the openings, the **frontier inventory in full** -- size, walk, facing, utility, room -- the explore visit's ranked goals, the loop's events, the oracle's reading, the exclusions, the places failed), detections, target closing, the control counters (A\*, classifier, room LLM, RPT\* calls so far) |
| `map_journal.npz` | the cells of the occupancy grid, the settled-unknown mask and the room partition that changed on every action, as deltas (since 2026-10-22): the map of ANY action is replayed from it |
| `floor_maps.npz`, `room_partitions.npz`, `floor_panels.json` | the final maps and the display panels' metadata |
| `metrics.json` | the outcome and the policy's diagnostics, including the latency ledger (where the wall time went) |
| `trajectory.csv`, `episode.json` | the poses per action; the episode's identity and the review command |
| `video.mp4`, `final.jpg`, `frames/` | the dashboard, for watching |

A review that is pointed at an action reads the log through
`review_recording.py`, never a frame:

```bash
R=runs/<campaign>/<scene>/recordings/<key>
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording $R summary         # outcome, latency ledger, release verdicts
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording $R timeline --start 120 --end 140
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording $R step 261         # the action, and its map by sector round the agent
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording $R map 261 --radius 6 --cell 0.2
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording $R events --name release
```

`step N` ends with the map of that action summarised in the agent's frame --
ahead / left / behind / right, within the radius: free, occupied, unknown and
settled area, raw frontier cells, and the frontier goals and openings that
fall in each sector -- so "there is unknown to the robot's left and no
frontier there" is a line of numbers with its reason beside it (settled by
the sight ledger; behind occupied cells; under the significance floor), not
an impression from a picture. Recordings made before the journal existed
fall back to the final map and say so.

`render_replay(recording_dir, render_pose, ...)` takes a callback receiving an
`AgentPose` and returning uint8 RGB. The adapter owns scene loading, GPU safety
and camera-pitch handling. Replay interpolates translation/pitch and the shortest
yaw arc, includes the final pose, preserves aspect ratio, and writes only a
separate `smooth_rgb.mp4`. It does not alter original scores or observations.

## Dependencies and validation

`requirements.txt` lists the shared CPU/scientific extras. Core contracts and the
lightweight harness remain free of simulator/model imports. Install simulator
bindings in their dedicated environments; do not install torch into the
lightweight test environment. Recording requires a working FFmpeg/libx264;
`IMAGEIO_FFMPEG_EXE` may select an already installed executable.

From the checkout root with the test interpreter selected:

```bash
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests \
  sparx_agency/core/mapping/objects/tests sparx_agency/core/mapping/topology/tests \
  sparx_agency/core/planning/objnav sparx_agency/tasks/planning/objnav_benchmark \
  sparx_agency/core/planning/planners/common/tests \
  sparx_agency/core/planning/planners/astar/tests -q
```

Tests use explicit synthetic geometry and the existing fake/corridor fixtures,
not a real dataset's constants. They cover route lifecycle, physical clearance,
object/room evidence, optional imports, explicit protocol accounting, locks,
resumption, metadata drift, recording without GT telemetry and replay integrity.

For a provisioned Habitat environment and an idle rendering GPU:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.smoke
```

This creates a temporary license-free room and checks real RGB-D, ENU frame
conversion, left/right turns, forward movement and STOP. Its result is explicitly
synthetic, not a published benchmark score.
