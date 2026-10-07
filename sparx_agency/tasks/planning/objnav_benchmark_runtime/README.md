# ObjectNav runtime and Gibson adapter

Observed RGB-D mapping → semantic room graphs → room LLM → RPT* room order →
local exploration and A*/WA* paths → existing discrete actions. Both explorers
retain persistent floor contexts; stair connectors come from the simulator's
navmesh by default (`RPTSettings.multifloor.stair_source`, see
[MULTISTORY.md](gibson/MULTISTORY.md)). Habitat target closing additionally uses
declared local NavMesh projection (`target_navmesh_projection` in run configuration).
This is an experimental multi-story algorithm, **not a claim of solved navigation**.

**To set up and run it end to end, start with [QUICKSTART.md](QUICKSTART.md)**: environments,
dataset and weight downloads, then the services, the frozen episodes and the campaign in
order. This file is the architecture reference.

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

**The context check** (`methods/target_context.py`, `context_check`, since
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
after** (`TargetClosing.refuses_far_candidate`, read by `perception_cycle.fuse`):
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
over `refine_distance_m` (0.15 m) since the goal was selected triggers replanning.
Small refinements do not move the endpoint every frame. Collision, blockage and
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
terminal frames hold the object whole. At `terminal_distance_m` (**1.0 m**),
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
from remembered distance alone is allowed. The pitch the target's height predicts
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
object, not to its middle; and when the inspection budget runs out after the
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
All thresholds are configurable under `RPTSettings.target_closing`; the phase,
persistent lock, visibility, target xyz, projected goal, refinements, occluded path
steps, bbox, failures, release counts (`releases`, `inspection_releases`,
`boxed_releases`), footing sweeps, boxed-in spots, border
rejections and the rejected spots with their radii
are included in episode/frame diagnostics and the HUD.

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

## Reusable pieces

| Component | Responsibility |
|---|---|
| `methods/rpt_policy.py`, `rpt_settings.py` | Detector/mapper composition, RPT* and baseline selection |
| `methods/target_closing.py` | Episode-local target takeover, consecutive-frame verification, bbox/depth servo, standoff A* and explicit STOP |
| `methods/target_context.py` | The semantic sanity check on a candidate: a STRONG room type the target is not searched for in, or a measured height its class is never seen at (a counter top read as a bed) -- a penalty and a multi-viewpoint lock, not a refusal |
| `methods/target_path.py` | Persistent NavMesh standoff goal, meaningful target refinement and continuous collision-qualified A* execution through occlusion |
| `methods/room_search_loop.py` | The seven-step room-search loop: one scan visit per room (vantage point, full rotation, finished for the episode) or the bounded room-confined sweep; re-classify → re-estimate → re-order at each loop point over the rooms that are still nodes; transit to the chosen room's vantage point / nearest frontier or to the foot of the chosen stairs; a room re-identified by a new kind of object ends its turn for a fresh solve |
| `methods/room_scans.py` | The scan ledger: where every completed look-around stood, on every floor; a room is finished when a scan stood in it or saw more than half of it through observed free space, or -- with no live frontier left -- when the camera looked into it or walked it through (`seen_through`), or when it is a doorless fragment under 3 m2 -- sticky, id-independent |
| `methods/sightlines.py` | The sight ledger: unknown looked through without a depth return (within 2.5 m, from two poses) and enclosed unknown pockets under 3 m2 are settled -- written occupied for the frontier logic alone -- and every camera pose per storey |
| `methods/room_vantage.py` | Where to stand in a room to see it: the reachable interior cell of greatest clearance (distance transform of the room mask) |
| `methods/room_priors.py` | Where a target cannot be: the room types a search need not enter for it (exclusions, not permissions; `unknown` never excluded) |
| `core/mapping/topology/search_node_oracle.py` | The node oracle: one call per loop point, to the LLM client's REASONING model, over every room still a node and every staircase -- P(going there next finds the target) per node plus "elsewhere" |
| `methods/stair_nodes.py` | Staircases as nodes of the loop's RPT* instance: ids above every room pid, the facts the oracle values them by, the climb as a leaf charged on every arc |
| `methods/exploration_fallback.py` | Where every failed plan, model or decision lands: the best floor-wide exit (object shadows and frontiers of rooms the target cannot be in wait behind it), the stairs by the explicit fallback rule, the demoted frontiers, a retired frontier, a relocation, a footing sweep -- a move, never an idle spin; failure records and service back-off |
| `methods/frontier_sweep.py` | Frontier goal generation for a room or the floor, committed-goal lifetime, optional look-around (the `sweep` ablation) |
| `methods/peek_stairs.py` | Floor-local seen-connector footprint: excluded from the room partition (stairs are never a room), from peek viewpoints and from a peek-only A* copy; ordinary stair navigation is unchanged |
| `methods/camera_control.py` | Sole pitch owner; bounded stair inspection, the footing sweep (a circle at 30 degrees down that maps the blind radius, for a pathless lock or a boxed-in fallback), long unprompted cadence and safe restoration |
| `methods/perception.py`, `perception_cycle.py` | Fresh raw predictions, coherent pixel projection with a footprint radius, floor-qualified fusion, plan-view association within a class with a height check |
| `core/mapping/objects/landmarks.py` | The landmark map: per-class association (a tight dedupe radius or footprint-disc IoU; classes never merge), confirmation by observation count; the class vote of 2026-10-04 kept behind `class_votes=True` |
| `methods/observed_map.py`, `floor_context.py` | Independent occupancy, rooms, objects, association anchors and paused floor clocks |
| `methods/multifloor_policy.py` | The building coordinator: portals per floor, the approach and climb of the staircase the loop's order chose, arrival bookkeeping, both stair sources; in ground-truth mode it decides nothing by a clock |
| `methods/floor_departure.py` | The optional (`doorway_peek.gate_floor_departure`, off by default) room-peek coverage gate before a floor choice, with its approach revalidation, movement guard and safe recovery from accidental early stair entry |
| `methods/stair_ground_truth.py`, `floor_decision.py`, `ground_truth_traversal.py` | Ground-truth stairs (default): the navmesh connectors as the policy reads them, the fallback rule's up/down choice (worth per metre of approach, flight and fixed change cost), and the committed polyline traversal |
| `methods/stair_traversal.py`, `stair_terrain.py` | Observed stairs (`stair_source: "observed"`): RGB-D support surfaces, step-connected paths, native-heading safety checks and the committed transition lifecycle |
| `gibson/stair_connectors.py` | Evaluator side: storeys and stair connectors read once per scene from the navmesh, attached to the episode metadata |
| `methods/falcon_policy.py`, `falcon_regions.py` | Bounded hierarchy and persistent local region history |
| `methods/falcon_motion.py`, `falcon_routes.py`, `route_memory.py` | Safe motion and interruption-aware committed routes |
| `methods/object_evidence.py` | Alias/frame deduplication, separated-view target evidence and bounded rejection |
| `methods/scene_graph.py`, `room_labels.py`, `doors.py` | Observed rooms/doors, the room label image, one-object (weak) and two-class (strong) room labels revised the action a new kind of object lands, the nodes handed to the oracle |
| `methods/exploration_metrics.py` | Observed-area proxy, actions, revisits, stagnation and latency |
| `habitat/simulator.py` | Synchronous registered RGB-D, native extrinsic checks and ENU pose |
| `recording.py`, `floor_panels.py`, `search_panel.py`, `visualization.py` | Exact commands/poses, persistent UNKNOWN panels with the rooms and visit order drawn on the active floor, the search column, grids and video |
| `gibson/run.py`, `frozen_eval.py`, `run_development.py` | Protocol execution and frozen source/model/data checks |
| `tests/trace_room_search_loop.py` | Development probe: the loop on the production map size with fake services and the real action converter on a point agent; not a test |

The floor tracker remains ROS-free in `core/planning/exploration/floor_atlas.py`.
Host-side numpy/scipy terrain processing does not enter the Noetic exploration
facade. No robot's flight controller is changed.

## Episode discovery and doorway inspection

Both explorers start with **`RPTSettings.warmup_steps=12`** discovery actions
before normal room selection -- **one full rotation in place** at the
benchmark's 30-degree turn, taken where the agent stands. RGB-D, YOLO, object
confirmation and scene-graph geometry remain synchronous on every frame, so the
rotation gives the map its first rooms and the detector its first objects; when
it ends the agent's position is recorded in the **scan ledger**
(`methods/room_scans.py`), which finishes the spawn room before the first room
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
   default `scan`, specified 2026-10-04). **A visit is a look-around**: walk
   to the room's **vantage point** -- the interior cell of greatest clearance
   the observed map can reach (`methods/room_vantage.py`: the Euclidean
   distance transform of the room mask, so the middle of the open floor, not
   the doorway and not the far frontier) -- turn a **full circle**, and
   leave. The camera's ray ends at the nearest object, so a 360-degree scan
   from the open floor shows what a second viewpoint or a walk to the far
   frontier would; the room is then **finished for the episode**, whatever
   frontier its mask still shows. Finishing is the **scan ledger's**
   verdict (`methods/room_scans.py`): it remembers *where* every rotation was
   completed, not which room id it was credited to -- the watershed
   renumbers, splits and merges rooms on every update, and the Ranchester
   recordings re-entered the same room under a new number three times in a
   row. A room is finished when a completed scan point lies inside its mask
   (`scan_point_inside`) **or** when a completed scan *saw* more than
   `scan_seen_fraction` (0.5) of its cells within the camera's depth range
   through observed free space (`seen_from_scan`: the small room fully
   visible from the corridor, the half of a split room the scan stood in;
   unknown cells end the sightline, so nothing unresolved is credited). The
   verdict is sticky per `(floor, pid)`. The visit is bounded hard
   (`scan_visit_steps`, 36 -- released `budget_spent` beyond it) and the
   walk to the vantage point separately (`scan_approach_steps`, 20; an
   approach that runs out, or that A* refuses, scans from where the agent
   stands if that is inside the room -- `scans_in_place` -- and otherwise
   releases the room `unreachable`). The rotation ends on measured yaw
   (a full circle swept, or two turns past a circle's worth), the scan is
   recorded, and the turn ends `exhausted`: productive, cooled, never
   repeated by the escape hatch. **A clue mid-visit does not switch rooms**
   (`reconsiders_deferred`): the Ranchester recording switched at 57, 73
   and 121 and came back to each; the clue is kept for the loop point the
   visit ends on, where every node is re-valued anyway.

   **What is not a node.** At every loop point the loop withholds two kinds
   of room from the oracle, the solver and the transit, and records why
   (`excluded` in the events, estimates and HUD): a room the ledger has
   **finished**, and a room whose identified type **cannot hold the target**
   (`methods/room_priors.py`, `LoopSettings.type_prior`: a sofa is not
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
   (since 2026-10-07, `LoopSettings.home_floor`, 0.60): a room holding a
   confirmed home object of the target is handed to the oracle with
   `SearchNode.home` set and is never read below the floor, whatever the
   model wrote and whatever its storey verdict says -- a bathtub on this
   storey IS the toilet's home type, found -- and such a room is **never
   finished by sight from outside it**: the scan ledger's `seen_from_scan`
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
   action 150). Excluded rooms carry probability
   0 so nothing reads them as unvalued; when nothing is left to value and
   no staircase is offered, no model call is made and the exploration
   fallback carries the search. `LoopSettings.min_prob` (0.05) keeps the
   oracle's "1" for a bedroom in a search for a couch out of the order.

   **Two more ways a room is finished** (`methods/room_scans.py`, since
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

   **The uncertainty floor** (`LoopSettings.unexplored_floor`, 0.25,
   applied in `search_node_oracle.SearchNodeOracle`, since 2026-10-05). A
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
   unexplored_elsewhere`, 0.10, since the Ranchester couch search of
   2026-10-05): the model's own STEP 2, asked for as a structured verdict
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
   by type) -- type or `unknown`, size, frontier clusters left, time
   searched and how long ago it was last inside, objects seen, whether the
   robot stands in it -- and every STAIRCASE off the floor -- up or
   down, whether the other storey was visited and what was found and
   searched there, whether the robot arrived by it -- and every **OPENING**
   of the floor (below) -- a doorway or gap to space not yet seen, the
   mapped room it opens from and the objects glimpsed through it -- plus
   one line on this storey and each other storey known. It answers, for
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
   that are still missing on this storey; a hallway with frontier is valued
   by what it may lead to; an opening is valued by the room likely **behind**
   it -- a toilet glimpsed through it makes it a bathroom's door, nothing
   glimpsed makes it the unknown room this storey may still owe the target;
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
   together (`methods/stair_nodes.py`). A room enters the instance at its
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

   **Openings are nodes too** (`methods/opening_nodes.py`,
   `LoopSettings.openings`, since 2026-10-04). The room partition is a
   watershed over *observed* free space, so a room the agent has only
   glimpsed through its door is not a room yet: its few seen cells hang off
   the hallway's region, and whatever object stood in the glimpse names the
   hallway. In the Ranchester recording of 2026-10-04 (attempt 7, action 75)
   the agent stood in a hallway facing three doors; the toilet seen through
   one of them made the whole west half of the storey a "bathroom", every
   door off the hallway was demoted with it, and nothing was left to value
   but the stairs it happened to turn towards. The decision to descend was
   right; the reasoning that produced it was not, and in a search for
   something on *this* storey it would have been fatal. So every genuine
   **exit** of the mapped floor -- a frontier that is not an object's
   shadow, not the floor under the agent's feet, at least `min_cells` (8,
   0.8 m) wide after merging within `merge_m` (1.5 m), not at the foot of a
   seen staircase, not toward unknown the **sight ledger** has settled (an
   enclosed pocket, or unknown looked through without a depth return; see
   below) -- is an **opening**: a node with a sticky building-wide id
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
   ends `exhausted`.
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
6. **Direct transit to the chosen room's vantage point** under `scan`
   (`entry_vantage`; re-aimed once at the vantage point's new position, then
   the centroid, if the mask re-segments under the route), or **to the
   closest frontier inside the chosen room** under `sweep`
   (`LoopSettings.entry_frontier`; a room with no frontier left keeps a
   centroid entry; a committed entry frontier the camera resolves en route
   is re-aimed at the room's next one, and a room with none left is
   released without the visit -- `frontier_exhausted` in transit); or **to
   the threshold of the chosen opening**, where the peek begins.
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
0.30 m) and relaxes to the body radius (0.18 m) only when nothing else
gets through; the frontier inventory and the RPT* instance used to
measure every distance on the body-radius graph, so a threshold 3.5 m
away through a 0.5 m squeeze read as 3.5 m while the route around the
squeeze was 12.5 m (Hanson, actions 166-196: a 30-action approach bound
spent 5.8 m short on a route still being followed). Now
`accessible_frontiers(preferred_cost=)` and `build_instance(preferred_cost=)`
measure at the preferred standoff where it reaches and fall back to the
body radius where only a squeeze connects (`RPTSearchPolicy.preferred_cost`),
and a peek's approach bound is raised to what the route the planner
actually adopted needs (`peek_approach_extended` events).

Steps 2–6 run on the action a room's turn ends, so a released room is
replaced by a transit at once -- never by a throwaway floor-wide route. A
room whose budget ran out is **not** put on the visit cooldown (the fresh
estimate may rightly send the agent straight back), nor is a staircase
whose climb began or a room released `reclassified`; a room the live map
says is finished (exhausted or mapped) is never repeated by the
every-room-cooling escape hatch: the exploration fallback carries the
search instead. `room_search_loop.events` in the episode record lists every
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
`methods/exploration_fallback.py`, and comes out as a command that moves:

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
   outranked because the strips were nearer. A finished room's opening is
   NOT demoted: it is how the next room is found. A frontier inside the
   camera's **blind radius** (`near_blind_m`, 1.2 m -- a level camera 0.88 m
   up sees the floor from about there) is demoted too: it appears under the
   agent wherever it stops, and no step toward it resolves it. Within a rung
   the fallback **commits**: the goal in force is driven until it is gone or
   refused unless another goal is worth `goal_switch_gain` (2x) its current
   utility -- the greedy per-action order, re-ranked by the agent's own
   turning, once produced a 180-degree turn toward one goal and a 210-degree
   turn back (`goal_kept` / `goal_switched` / `goal_outranked` in the stats);
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
5. a **relocation** to the farthest reachable known cell, for a vantage point
   the map may show a frontier from;
6. a **footing sweep** (since 2026-10-05) when nothing on the observed map is
   reachable and at least `footing_unknown_fraction` (0.25) of the cells
   within `footing_radius_m` (1.2 m) of the agent are unknown: an agent that
   has not moved stands on a disk of unknown under the camera's blind
   radius, and unknown is impassable; one LOOK_DOWN, a circle and a LOOK_UP
   map it (`CameraController.begin_inspection(reason="footing")`, once per
   spot; a protocol without LOOK actions skips it);
7. a single hold only when even that has been done here, where the idle
   turn is the one action that changes anything.

It is reached from the loop whenever no room is in force, and from
`RPTSearchPolicy.plan` whenever the decision itself raises -- an A* with no
path, an RPT* instance that cannot be built, a bug. A **room LLM** failure
(timeout, bad JSON, refused uniform prior) no longer ends the episode: the
loop records it, arms a doubling back-off (25 → 50 → 100 → 200 actions,
`FallbackSettings`) during which no model is asked, and the fallback carries
every action until the model answers again, at which point the loop resumes
its steps. A **detector** failure is handled the same way in
`perception_cycle.py`: the frame carries no detections, is counted as
skipped, and the service is left alone for the back-off. Every failure is
counted, kept with its type, message and origin under
`exploration_fallback.failures` in the episode record, and logged once per
type. Before this, an empty hold cost one idle `TURN_LEFT` per action until
the step budget ran out -- the "camera spinning in place" of the Ranchester
recordings -- and a model failure was an agent error.

## Glances: where along a route a look to the side pays

`methods/path_glances.py` (`GlanceSettings` on the policy, since 2026-10-05).
The camera's cone is narrow and the map is what it saw: an agent walking a
corridor sees the corridor, and a door a step to the side passes through
the edge of the frame or not at all (Hanson, action 35). Turning in place
costs no path length -- nothing under SPL -- but under the 500-action
budget a full circle every few steps is unaffordable, so the question is
*where* along the route a look pays, and which look. The scheduler scores
the route in force every `revalue_actions` (3) and whenever it changes:
candidate points one every `stride_m` (0.5 m) up to `horizon_m` (6 m),
each with the route's heading there; for each, the unknown floor a glance
would sweep -- left (from the heading's cone out to a right angle), right,
or the full circle -- by optimistic rays through the unknown that stop at
walls (`core/planning/exploration/view_gain.py`), **minus what the walk
reveals anyway** (the forward cones along the whole route ahead), so a
doorway's room counts once, at the point it shows best: the user's "one
step forward and all three areas show; two and the first is hidden
again". The best gain per action wins -- a side glance costs the turns
out (the turns back are the follower's), a circle the full count -- if
it clears `min_gain_m2` (3) and `min_gain_per_action_m2` (0.5): six
forward steps into unknown space sweep the cone over 1.5 m of new floor,
around 10 m2 in a room, so walking is the better buy wherever the unknown
is ahead; a doorway beside the route shows the walk nothing and a glance
5-15 m2 (the Hanson routes replayed on their final maps: 14.8 m2 at
action 33, 7.6 m2 at 164). When the agent reaches the point the look is
performed, one in-place turn per action, as a suspended phase like the
warm-up: the loop is not ticked and not charged, the supervisor's clocks
pause, and the committed route's watchdogs are told the pause
(`glance_scheduled` / `glance_started` / `glance_complete` /
`glance_aborted` events in `episode_info()["glances"]`; the planned and
active glance in the per-step `search` record and on the HUD). A spot
where a full rotation already stood (the scan ledger) is never glanced
from -- whatever is unknown from there is beyond range or behind
furniture -- and a target sighting or a stair traversal aborts a glance;
the warm-up, a peek's look, a scan's rotation and the takeover never
start one. `enabled=False` is the ablation.

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

`methods/sightlines.py` (`SightSettings` on the policy, `sight` in the
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
  a wardrobe; a cup would, and that is the trade the bound makes.

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
`class_votes=False`, wired in `methods/floor_context.py` from
`RPTSettings.landmark_dedupe_radius_m` / `landmark_footprint_iou`, and
`methods/perception_cycle.py`). Every projected detection carries a
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
output. A former 24-prompt service is intentionally rejected. HTTP checks pin
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
