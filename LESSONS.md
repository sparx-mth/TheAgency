# Lessons

Errors, gotchas, and failure modes that cost real debugging time — kept here so they're
never re-derived from scratch. Check here before debugging anything that feels familiar.

Format per entry:
- **Symptom** — what you actually observed
- **Root cause** — what was really going on (often different from the first suspect)
- **Fix / workaround** — what resolved it, or the current best mitigation if unresolved
- **Don't** — anything that looked like a fix but wasn't, or made it worse

---

## 2026-10-14 — The seen toilet that was not locked, the reached toilet that was not STOPped on, and the search that re-scanned, walked to room centres and spun between slivers

**Symptom:** The user read the recordings of `runs/benchmark_toilet_15episodes_20261013T233505Z` (SR 0.800 / SPL
0.433 / DTG 0.62 m) and listed nine behaviours: the robot not looking down at a toilet it had reached; toilets spotted
but not locked; returns to rooms already scanned; no `O`/`T` marker for a seen toilet on the HUD; spinning in place
without a planned path; walks to the centre of rooms instead of to a frontier; rooms scanned a second and third time;
and routes changed mid-way. The per-step traces (`steps.jsonl` -> `method.target_closing`, `method.search`,
`method.fallback`; `metrics.json` -> `room_search_loop.events`) gave one mechanism per complaint:

1. **Markleeville/000188, actions 34-45** -- a takeover begun on a 0.84 override past an `unverified` memory had every
   0.60-0.79 re-sighting of the same toilet dropped BY that memory in `_candidates` (the memory ran on every unlocked
   frame, the active candidate's included); `consecutive_frames` never reached 2.
2. **Markleeville/000188, actions 20-31** -- the toilet at 5 m read 0.62 with depth at the frame's edge and 0.49
   WITHOUT depth once centred (beyond what the projection accepts); the depthless frame counted as "nothing seen", the
   count reset, and the sweep turned AWAY from the centred box every other action, twice through the 12-step budget.
3. **Wiconisco/000193, actions 476-499** -- locked, standing 1.0505 m from the toilet's near edge against a 1.05 m
   limit, the box 16-17 degrees off the centre on alternating sides (both just outside `turn_action`'s half-turn dead
   band, the detector drawing the box a few degrees differently at the two headings): TURN_LEFT/TURN_RIGHT to the
   step limit, SR 1 by the distance alone.
4. **Collierville/000023, action 375** -- the toilet locked at 2.8 m through its WC doorway; the floor between (a
   level camera sees the floor from 1.4 m out) was unknown, the planner blocks unknown, the standoff ring had no
   route, a footing sweep (11 actions) mapped only the feet, and the lock was released as boxed in -- which then
   refused the 0.82-0.88 re-sightings from the spot (actions 376-392).
5. **Collierville/000023 (180-202) and /000036 (241-265)** -- a 0.88-0.95 toilet seen at 1.5 m backed off, 10-12
   actions of retreat and 13 of verification from a spot the jamb hid it from, then a trusted release; 23-25 actions
   per candidate, and one toilet lost.
6. **Collierville/000023, actions 237-420** -- the storey "covered", the second pass re-offered the scanned rooms and
   the agent re-scanned room 0 from its vantage point twice (237-263, 403-420) and room 1 once, with the toilet's WC
   unentered 3 m away; the 3B oracle was also shown every scanned room as a node on every call (48 calls in
   Darden/000190).
7. **Collierville/000036, actions 99-113** -- `order=[]`, the fallback's exits in the inventory's size order: a 1.2 m
   sliver first, resolved by the step toward it, the next sliver behind the agent: six turns left, six right. And the
   goal in force was forgotten after ONE action the fallback did not own, so every glance re-ranked greedily.
8. **Darden/000190 (room 8) and the "vantage" entries** -- a room with no significant frontier was entered at its
   vantage point (the middle of the room); the walk across showed nothing the doorway did not, and an unreachable
   vantage point released the room `unreachable` on every loop point.
9. The landmark node (`T<n>`) of a seen toilet was hidden whenever ANY release lay within range of it
   (`landmark_openings` asked `rejected_near`), a 12-step verification failed from 4 m included.

**Fix:** (1) a box ASSOCIATED with the active candidate's anchor is a re-sighting: the memory, cooldown and boxed-in
spot judge a START only (`resighting_exemptions`). (2) A box of the class within one turn of the anchor's bearing
without a usable depth is a `glimpse`: it carries the consecutive chain, is faced, then stepped toward, and the step
does not run the verify clock (`max_verify_approach_steps` 24; `glimpse_steps`, `verify_pauses`). (3) A fresh range
beyond the limit by less than a forward step, faced within a turn, is one MOVE_FORWARD (`inspection_max_steps` 3);
a centring turn that would undo the previous action's turn with the error under a whole turn is read as aligned
(`dead_band_alignments`). (4) `TargetApproachPath` plans the ring again with unknown passable at 8x the cost of open
floor when the known floor has no route (`through_unknown`, the policy's `optimistic_planner`; `_navigate(planner=)`).
(5) A candidate FIRST READ at `backoff_trust_confidence` (0.92) skips the manoeuvre (`backoff_trusted_skips`).
(6) `LoopSettings.second_pass=False` and `scanned_to_oracle=False`: finished rooms are no node and are left off the
oracle's list (the HOUSE line still names their types); `lingered_discount` 0.2 for an entered room searched 20+
actions with <=1 significant frontier; the prompt says "searched 20s or longer with frontier 0 or 1 is as good as
scanned: 0-5". (7) The fallback's exits are ranked by view gain and a sliver under `explore_min_gain_m2` is demoted
(`slivers_demoted`); the goal in force survives `goal_memory_actions` (12) without the fallback. (8) A room with no
frontier left is entered at its DOORSTEP, the passable cell nearest the agent along the floor (`_nearest_entry`,
`entry_doorstep`), in `_explore_entry`, `_reaim_after_refusal` and the entry-lost re-aim. (9) `landmark_refused`:
only a release that DISPROVED the candidate from close hides the landmark node.
Tests: `test_target_closing.py` (the four closing rules, from "the toilet re-run of 2026-10-13"),
`test_target_backoff.py` (the trusted skip), `test_opening_nodes.py` (the landmark), `test_search_economy.py`
(lingered, the oracle's list), `test_exploration_fallback.py` (slivers, goal memory), `test_explore_visit.py`
(the doorstep).

**Result (round 1):** the same 10 episodes (2 per scene, seed 17; `runs/benchmark_toilet_10episodes_20261014T085618Z`)
went from SR 0.700 / SPL 0.425 / DTG 0.94 m (their scores in the 15-episode run of 2026-10-13) to **SR 0.900 /
SPL 0.489 / DTG 0.23 m**, mean 230 actions (302). Corozal/000019 and Darden/000190, never sighted before, succeeded;
Collierville/000036 149 actions (277), Corozal/000142 251 (408), Wiconisco/000193 179 (500). Two went the other way,
and their traces gave round 2:

10. **Collierville/000023** (the one failure, DTG 2.3 m): the toilet at 0.95 on four border-cut frames (no takeover
    can start on a cut box), the takeover on the fifth at 0.84 -- "sudden" by the range alone -- the 25-action
    retreat to a spot the jamb hid it from, and a 2 m `wider perspective` rejection that refused eleven more frames
    at 0.86-0.91. And every one of its frames read `unsupported_floor_association`: seen from a metre with the floor
    in front inside the camera's blind zone and the base under the frame's lower edge, the lowest sample near the
    toilet was its bowl at 0.2-0.45 m, outside the 0.18 m band -- never a landmark, never a `T` node.
11. **Wiconisco/000014** (129 -> 320 actions): the toilet seen once at 0.54 from 3.2 m; the one step toward it put
    it behind a jamb and eleven in-place turns at the same three headings never brought it back; the landmark had
    one observation, which the map does not confirm, so no `T` node drew the search back for 65 actions.
12. **Darden/000034 (round 2), actions 112-124** -- a wedged peek walk (two blocked MOVE_FORWARDs, slides under the
    0.15 m progress threshold) released its opening `blocked` at 119, correctly; then the NEXT FIVE nodes were
    released `blocked` on their first action each (120, 121, 122, 123, 124): `policy._blocked_since` is the agent's
    motion clock, reset only by a forward step's worth of displacement, and the supervisor reads it per transit --
    so every node selected after the release inherited "blocked for 7 s" and four openings were retired for good.

**Fix (round 2):** (10) `backoff_persistence_frames` 2 / `backoff_persistence_window` 6: a class in the frame on two
of the six actions before the takeover is persistent, not sudden, and is verified where it stands
(`backoff_persistence_skips`); the floor-support band is 0.60 m (a storey is 2.5 m away). (11)
`OpeningSettings.landmark_sighting_confidence` 0.50: a target landmark seen once at the closing's threshold is a `T`
node (`target_landmarks`, `_landmark_best_conf`); `verify_sweep_max_actions` 6: a fruitless sweep cycle releases the
verification before its budget (`sweep_releases`). (12) `RoomSearchLoop._tick` resets `_blocked_since` on every
release: the clock was the released node's. Tests: `test_target_backoff.py` (persistence),
`test_perception_association.py` (the band), `test_opening_nodes.py` (the single sighting; the clean clock after a
blocked release), `test_target_closing.py` (the sweep cap).

**Result (round 2, the same 10 episodes, `runs/benchmark_toilet_10episodes_20261014T100437Z`):** SR 0.800 / SPL
0.388 / DTG 1.45 m, mean 262 actions. Collierville/000023 succeeded (223 actions; the persistence skip fired once and
the toilet was locked where it was seen), Wiconisco/000014 came back to 183 actions (320), and the two Darden
episodes went the other way -- and neither on this entry's rules:
- Darden/000190 ran bit-identical to round 1 for 193 actions, up to the same `exhausted` release; at that loop point
  the SAME prompt (same HOUSE line, same four nodes, same two openings) drew 0.01 for both openings in one run and
  0.10 in the other from `qwen2.5:3b`, so one run peeked and the other fell to the fallback. It ended 1.24 m from
  the goal at the cap. This is the 3B oracle's non-reproducibility at temperature 0 (the LESSONS entry below says
  not to read two campaigns' means as a code difference); the 14B model is the office's.
- Darden/000034 (55 actions in both earlier runs) was sent by the order into the house's sunken level: `transit/10`
  walked the agent from z 0.02 to -0.45 m, the spawn-floor guard adopted the height as a half level after 45
  actions (`half_level_adopted`, a fresh floor context, a fresh map, a new warm-up), a far fallback frontier then
  took it down again to -0.78 m (a second adoption at 368), and the last 130 actions were footing sweeps (48) and
  escape probes (86) in a map that knew nothing but the cells around the agent's feet. DTG 13.2 m. **Open item:**
  a half-level adoption on a split-level storey should keep the map and the search (the storey is the same), not
  start the search over; and a fallback frontier 12 m away across a level change should not be an exit while the
  spawn plane has frontier left.
Across the three runs of these ten episodes the code's own effect is read per episode, not from the means:
Collierville/000023 (0 -> 0 -> 1), Corozal/000019 (0 -> 1 -> 1), Wiconisco/000193 (500 -> 179 -> 179 actions),
Collierville/000036 (277 -> 149 -> 143) are the fixes; Darden is the model and the split level.

**Don't:**
- Don't run a start-of-takeover gate (memory, cooldown, boxed-in spot) on the active candidate's own frames; the
  Markleeville toilet was in view and associated on eleven of twelve frames and never locked.
- Don't read a depthless box on the candidate's bearing as absence, and don't sweep away from a centred box.
- Don't trust a half-turn dead band against a detector that draws the box a few degrees differently per heading:
  remember the last centring turn.
- Don't let the standoff ring fail on unknown floor the target was seen across: plan through it and let the map
  re-check the route.
- Don't re-scan (`second_pass`) while unknown slivers remain; the fallback's ranked exits are the second pass.
- Don't enter a frontier-less room at its vantage point; its doorstep finishes it (`explored`) for the same evidence.
- Don't call a candidate "sudden" by its range alone: four border-cut frames of the class before the takeover are
  not a surprise, and the retreat that re-verifies a surprise costs 25 actions and the line of sight.
- Don't sweep the same three headings for twelve actions: a candidate none of them shows after one cycle is
  behind something, and only a position change -- the `T` node's standoff walk -- brings it back.
- Don't require two observations of the TARGET class before its landmark is a node; the second frame of a toilet
  glimpsed through a doorway is the one the search has to go and get.
- Don't let a motion clock outlive the node it condemned: the supervisor's `blocked` verdict is per transit, and a
  clock that is not reset at the release judges every later node by a walk it never made. (Also: never edit a
  `.py` under `sparx_agency/` while `five_scene` runs -- it hashes every one and aborts the campaign.)

---

## 2026-10-13 (night) — Backtracking in the toilet re-run: the model's "second pass" with doors unpeeked, exits valued blind, a walked room valued over an unknown one, a 120-action cooldown on the toilet's room, and a planner that walks round the house rather than squeeze

**Symptom:** The user read six recordings of `runs/benchmark_toilet_15episodes_20261013T184541Z` and listed eleven
places where the agent went back to rooms it had already walked, chose a gap beside a scanned area over openings onto
unseen space, abandoned a route through unvisited rooms for one back through visited ones, or stood with no `O` on the
HUD while openings were in its order. Per-episode traces (`steps.jsonl` → `method.search`, `metrics.json` →
`room_search_loop.events` / `estimate_events` / `solver_records`) gave one mechanism per complaint:

1. **Corozal/000076 action 159, Darden/000034 action 321, Wiconisco/000014 action 238** — the oracle (`qwen2.5:3b`)
   wrote `pass: second` ("no unexplored places or frontier left") with six openings on its own HOUSE line, and the loop
   re-offered every scanned room; RPT* sent the agent back across the storey to re-scan living rooms while the
   bathroom's door stood unpeeked.
2. **Corozal/000076 action 126** — O3 (10 cells, 1 m2 behind it, 4.6 m away) chosen over O13 (148 cells, the unseen
   half of the storey, 6.4 m): both "nothing glimpsed", both 0.10 from the model; RPT* takes the nearer.
3. **Darden/000034 action 139** — R7, a "bathroom" walked THROUGH at action 103, valued 0.6 ("known to contain a
   toilet") over R5, unknown, bathroom-sized, never entered, 0.2 and 6 m away; the agent walked 11 m back to R7.
4. **Darden/000034 action 217** — R5 (the toilet's room) released `unreachable` on one refused plan at 112 was still
   on the supervisor's 120-action visit cooldown at 217, when the agent stood 4 m from it; the order went north.
5. **Corozal/000076 action 394** — the committed route to O13 (7.5 m, through unvisited space) was `route_obstructed`
   by a cell that appeared 0.25 m from it, and the re-plan at the preferred 0.30 m standoff was a 12.7 m loop back
   through visited rooms, because the planner relaxes to the body radius only when NO route exists at the standoff.
6. **Corozal/000076 actions 168-255** — O23 was offered before the staircase beside it was placed (action 203); the
   capsule then stood between the agent and the threshold 1.8 m away, and the peek walked an 87-action loop round it.
   `detect_openings` excluded exits near the portal's points only, and a peek in force was never re-checked.
7. **Corozal/000142 action 69** — two openings in the RPT* order vanished from the HUD: the fold was computed BEFORE
   the oracle call with the old probabilities (room 0 under `min_prob`, its openings kept as nodes), and the HUD's
   fold the next action read the new ones (room 0 a node, its openings folded -- hidden). The targeted opening
   vanished too whenever the inventory lost it mid-walk.
8. **Wiconisco/000014 actions 177-238** — a living room's explore visit walked four frontiers for 61 actions; the
   user's rule is "scan the doors and the large areas, skip the slivers, leave and do not come back".
9. **Collierville/000036 (run C) action 213** — the toilet at 0.93-0.95, 1.5 m inside its doorway; the sudden-proximity
   back-off retreated to a spot the jamb hid it from, the detector never fired again, and the candidate was released
   as DISPROVED with twice the rejection radius -- the toilet landmark on the map was never a node again.

**Fix:** (1) `RoomSearchLoop.second_pass` is gated on coverage -- no unfinished room, no exit, no reachable frontier
worth a walk -- and the model's verdict is recorded (`oracle_pass`) but decides nothing. (2) `Opening.gain_m2` (the
camera's cone from the threshold along the heading, through the unknown) and `opening_value_factor` scale the model's
number, `clamp(gain / 4 m2, 0.25, 1)`, a `doorway` in full. (3) `LoopSettings.entered_discount` 0.5 on a room stood in
but not finished, and the prompt's rules rewritten (an entered room at most half of an unexplored one of its type;
"never write known to contain"; prefer places not been near). (4) `ObjectSearchParams.failure_cooldown_s` (the loop:
30 actions) for `unreachable` / `blocked` / `transit_timeout`. (5) `WeightedAStarParams.detour_factor` (the policy:
1.5): a standoff route more than that times the floor route is a detour and the squeeze is returned. (6) exits within
`stair_exclusion_m` of the guard's mask are not openings, and a peek whose threshold the mask reaches is retired at
once. (7) the fold is recomputed after the oracle call; the HUD lists every live opening with `folded` / `targeted` /
`live` flags and the peek's opening is kept on it. (8) `explore_min_gain_m2` 1 m2: only frontiers revealing that much
are goals, keep a room open or count for `frontier=`; a peek is ended when its opening is resolved on the way
(`vanish_actions`). (9) `backoff_trust_confidence` 0.92: a candidate read that well is released without a rejection
when the wider perspective misses it. Tests: `tests/test_search_economy.py`, `test_room_scans.py` (the covered storey),
`test_target_backoff.py` (the trusted release).

**Result:** the same 15 episodes (`runs/benchmark_toilet_15episodes_20261013T233505Z`, all scored, no native crash)
went from SR 0.667 / SPL 0.293 / DTG 2.19 m (run C) to **SR 0.800 / SPL 0.433 / DTG 0.62 m**, mean 252 actions
(C: 302). Six episodes improved, two of them to SPL 1.0 (Darden/000034 in 55 actions after 251; Darden/000072 in 49
after 91); Collierville/000036 and Wiconisco/000193, failed in every earlier run, succeeded. Two regressed:
Collierville/000023 (C: 131 actions) ended 0.58 m from the goal at the step limit, and Corozal/000019 (C: 477) ended
2.4 m away. The /000023 trace is a target-closing failure, not a search one, and is the open item:
- action 180: the toilet seen at 1.5 m through its WC doorway; the wider-perspective back-off retreated along the
  trail to a spot the jamb hid it from (no line-of-sight test on the retreat goal), the verification oscillated
  TURN_LEFT/TURN_RIGHT at that spot for 13 actions without a sighting, and the trusted release followed. The toilet
  never entered the landmark registry (its boxes failed the depth association), so nothing drew the search back.
- action 357: seen again from 2.8 m; the two "towards the candidate" viewpoint steps wedged the agent against the
  corridor wall, and the standoff ring had no path -- the floor in front of the WC was still unknown (a level camera
  sees the floor from ~1.4 m out, and the agent had only ever stood 0.6-0.9 m from those cells) and the planner
  blocks the unknown. `_no_path` released the lock as boxed in, which refuses a new takeover within 1 m of the spot,
  so the full glance there a step later -- through the heading the toilet had just been seen on -- re-engaged nothing.
  Replaying the ring on the FINAL map from the same pose finds a route (offset 30, 0.60 m from the target).
  Candidate fixes: a line-of-sight test on the back-off goal (the sight ledger has the rays); return to the sighting
  pose instead of hunting when the retreat lost the target; plan the standoff ring through the unknown, or walk to the
  reachable cell nearest the target, before calling the spot boxed in.
The two never-sighted failures (Corozal/000019, Darden/000190) spent 141 / 92 actions on peeks and 73 / 83 on
glances; the toilets sit in small WCs the search never looked into. Glance cost is the number to watch next.

**Don't:**
- Don't let the model open the second pass. It cannot count the openings on the line it was just shown.
- Don't value an exit by its frontier's cell count: a long boundary onto a strip is many cells; a doorway onto a room
  is few. Rays through the unknown from the threshold are the measure; a confirmed door frame is exempt.
- Don't keep `keep_size_cells` as a significance exemption for in-room frontiers: a strip behind a cabinet is 20 cells
  wide and reveals 0.2 m2. The exemption was tried and removed the same evening.
- Don't widen the planner's preferred standoff to fix a detour, and don't let the planner relax by default -- the
  factor is the rule: a squeeze only when the standoff route is a multiple of it.
- Don't trust `seen_from_scan` or `pass: second` on a storey with an unpeeked door. The door is where the target is.

---

## 2026-10-13 — The 15-episode toilet re-run: five step sinks the per-episode trace exposed (a blocked clock that ran through a glance, a bump the map forgot, a room "seen" with its door unexplored, a spawn behind a sofa, a pause that un-planned committed routes)

**Symptom:** The same 15 published toilet episodes (`--five-scene 3 --category toilet --shuffle`, seed 17,
`qwen2.5:3b-instruct` for both models) run before and after the day's efficiency work went from SR 0.600 /
SPL 0.258 / DTG 2.02 m to SR 0.533 / SPL 0.233 / DTG 3.01 m, with five episodes flipping each way. The
per-step records (`steps.jsonl`, `metrics.json` → `room_search_loop.events`) of the four worst showed four
distinct mechanisms, none of them the day's changes and none of them the LLM:

1. **Corozal/000142** (`51eff3fa1637`): one blocked `MOVE_FORWARD` at action 274, a ten-turn glance in place,
   and the next four transits were released `blocked` one action after they began (285→286, 316→327,
   327→328, 328→330). Then, at 327-336, ten `MOVE_FORWARD`s into the same wall at (7.91, -0.72), heading 134°,
   `forward_blocked` every time, and again at 349-350. The toilet was first seen at action 494 of 500.
2. **Corozal/000076** (`c7a4c81d4dee`, a success in 88 actions before): the room holding the bathroom door
   (`office`, centroid (5.7, 2.7), `last_inside_step: None`, `frontier_clusters: 3`) was excluded at action
   168 as `scanned:seen_from_scan` -- the warm-up at the spawn point had half of its then-known cells in view
   -- and valued 0.01 ("office, scanned, no toilet there") on every loop point after. Its door frontier at
   (7.5, 3.4) never formed an opening node. The agent peeked nine openings elsewhere (two of them 82- and
   99-action approaches) and the toilet's cells were still unknown at action 500.
3. **Darden/000034** (`4ae716bacfab`, a 53-action success in the validation re-run): the bathroom (room 5) was
   released `unreachable` twice on ONE refused plan each -- at 112 after 16 actions and at 493 after a
   65-action walk that had brought the agent from 11.7 m to 2.4 m of it -- with the agent 0.5 m from the
   spot the validation run had first seen the toilet from, facing the other way. DTG 1.33 m at the cap.
4. **Collierville/000036** (`6bc7ace42dcf`): 488 of 500 actions as `fallback_hold` ("off the observed passable
   map; the idle turn is the only move"), 0.0 m walked, in both runs. The agent spawned in the strip behind a
   sofa; the depth camera ringed its 0.5 m pocket with occupied cells and no inflated route left it.
5. **Both of the "unreachable" releases above happened on the action a glance began** (Darden/000034 at 493,
   Corozal/000076's peek toward (-3.3, 4.4) at 282, 1.5 m short after a 51-action walk on a committed route),
   and the re-run with fixes 1-4 still lost ~150 actions of Corozal/000076 to peeks that ended `unreachable`
   30-62 actions into a walk that was making progress.

**Root cause:**
1. `RPTSearchPolicy._blocked_since` is reset only by a forward step's worth of displacement (the Leonardo
   wedge lesson of 2026-10-05) and is NOT paused with the supervisor's deadlines during a suspended phase
   (`supervisor.pause` shifts `_goal_s`, `_search_end_s`, ... through a glance, a peek's look, the warm-up;
   the blocked clock ran on), so after a ten-turn glance any new transit was 6 s "blocked" at once. The
   repeated wall-pushing: `ObservedMap.notify_blocked` added ONE hit (`l_occ` 0.85) to a cell the floor rays
   had driven to `l_min` (-4); it needed ten bumps to cross the 65 % threshold, decayed again under the rays
   during the next glance, and meanwhile the body-footprint clearing forced it free for the planner.
2. `RoomScanLedger._status` reached `seen_from_scan` without reading the room's live frontier: "unknown cells
   block sight, so nothing unresolved is credited" was true of the room's KNOWN cells and said nothing about
   the unknown beyond its open boundary. Only the frontier-free tests (`seen_through`, `fragment`,
   `explored`) required `frontier == 0`.
3. `_transit` turned the first `_navigate` refusal into `route_failed`, which the supervisor ends as
   `UNREACHABLE` the same action. The committed entry was a frontier cell beside the door jamb that the map
   re-snapped under (or swallowed in the inflation) as the agent closed in; the room's other frontiers, its
   vantage point and its centroid were never asked about.
4. The fallback's last rung was the hold. LESSONS 2026-10-08 already said "the hold rung has no recovery: the
   idle turn does not change the map" -- and rightly forbade opening "genuine" obstacles around the start in
   the planner's map. Nothing physical was ever tried: the simulator's collision check, not the 0.1 m grid
   with a 0.18 m inflation, is what says whether the body passes a 0.4 m gap.
5. `ObjectSearchSupervisor.pause` shifted `_plan_s_at_goal` with the deadlines. That stamp is the ordinal
   mark "a plan newer than this was made for the goal in force" (`planned = last_plan_s > _plan_s_at_goal`);
   a committed route is planned once and followed without re-planning, so `last_plan_s` stays put, a pause
   of a glance's length moved the mark past it, and with `elapsed > plan_grace_s` (5 s) the next update
   released the node "no route in 5 s -- unreachable". Every transit longer than the grace was one glance
   away from release.

**Fix:** (1) `notify_action` shifts `_blocked_since` by the action time whenever it pauses the supervisor
(external or released actions), and `notify_blocked` → `LogOddsGridCostmap.mark_occupied` sets the cell a
forward step would have reached to `l_max` outright (it decays like any evidence, but it flips at once and
the footprint clearing keeps occupied cells). (2) `status()` withholds `seen_from_scan` while the known live
frontier count is above zero; `scan_point_inside` stands. (3) `_reaim_after_refusal`: the room's next three
view-gain frontiers under `explore`, then the vantage point, then the centroid, before `route_failed`
(`route_reaimed` events; the route memory's `no_progress` refusal still releases). (4) `methods/escape_probe.py`
as rung 7 of the fallback: the grid read WITHOUT inflation along every turn heading, the heading with the
thinnest occupied band (≤ 0.4 m) and the most floor beyond it (≥ 0.5 m) gets a turn or one forward step;
a blocked probe is remembered per cell and heading for 80 actions. Collierville/000036 was out of the
pocket in 12 actions. (5) `pause` leaves `_plan_s_at_goal` alone; the deadlines still shift. Regressions: `test_blocked_evidence.py`, `test_escape_probe.py`,
`test_room_scans.py::test_a_room_seen_from_a_scan_with_a_live_frontier_left_...`,
`test_explore_visit.py::test_a_plan_the_planner_refuses_re_aims_...`,
`test_object_search_supervisor.py::test_a_pause_does_not_unplan_a_committed_route`.

**Don't:**
- Don't compare two 15-episode campaigns of the 3B model by their means and call the difference a
  regression or a gain: the same episode went 53 actions / SPL 1.0 → 500 actions / DTG 1.33 with identical
  code, diverging at the first loop point (Ollama is not bit-reproducible at temperature 0). Read the
  per-episode traces; the four mechanisms above were each visible in one.
- Don't reset the blocked clock on a glance (a wedged agent that glances every few actions would never trip
  the verdict) -- pause it, exactly as the supervisor's deadlines are paused.
- Don't make the bump mark permanent: a cell the agent clipped entering a 0.7 m door would close that door
  for the episode under the inflation. Saturate it and let the free-space rays wear it down.
- Don't write the probe's judgement into the planning map; it is an action, and the simulator answers it.
- Don't shift an ordinal timestamp when you pause a clock: a deadline moves with the pause, a "since when"
  mark does not. Look for releases that coincide with the first action of a glance, a peek look or the
  warm-up; the supervisor's note says "no route in N s" about a route the follower is standing on.

**Result and an open item:** with the five fixes the same 15 episodes scored SR 0.667 / SPL 0.293 / DTG 2.19 m
(mean 301 actions), against 0.600 / 0.258 / 2.02 m before the day's work and 0.533 / 0.233 / 3.01 m after its
first half; Corozal/000076, Darden/000034 and Wiconisco/000062 came back as successes, Collierville/000036 left
its pocket. Three of the four remaining failures are the 3B oracle's order (Darden/000190 as in the entry below;
Wiconisco/000014's bathtub room kept offered but ranked behind a re-scan; Markleeville/000131 first saw its toilet
at action 493 and was mid-approach at the cap). The open item is infrastructural: that evening three `gibson.run`
jobs died natively on the software renderer -- one left a `Zl` zombie (main thread gone, 32 Mesa/Magnum threads
alive, `ffmpeg` child waiting) at action 40, two were SIGSEGV at actions 388 and 108 of the same episode, none
reproduced under `PYTHONFAULTHANDLER=1`, no core (`apport` ignores a conda python, `dmesg` is restricted).
`five_scene` now kills a job whose `live.json` is `--stall-minutes` (20) old (`STALLED`, scene failed) instead of
waiting on the zombie, and the unreached episodes are re-run with `--episode-ids`. If it recurs, run the job
under `catchsegv`/`gdb -batch -ex run -ex bt` or set `kernel.core_pattern` for the user before guessing.
- Don't judge the probe by this spawn alone: with a dead planner on open floor it walks straight until a
  wall, which the fallback tests now expect (`test_a_dead_planner_is_one_probing_step_per_action...`).

---

## 2026-10-13 — ObjectNav Darden/000190: the toilet's bathroom was "not worth a step" -- the sight ledger's pocket rule settled the unseen half of a small room, three runs in a row

**Symptom:** The recorded episode (`runs/benchmark_toilet_15episodes_20261013T110223Z/Darden/recordings/ec8d17aeaaaa`)
and two re-runs after the efficiency fixes of the day all ended at the 500-action cap with the agent having
stood *inside the success region* -- at the bathroom door, DTG 0.0 at actions 321-333 of the first re-run,
0.15 at action 119 of the recording -- without a detection, a frontier or an opening pointing into the room.
The per-step record showed no `accessible_frontiers` entry anywhere in the bathroom while the final map had
0.9 m2 of unknown there, walled on three sides, with the toilet in it.

**Root cause:** `methods/sightlines.py::enclosed_pockets`: an unknown component of at most `pocket_max_m2`
(3 m2) that does not touch the map's edge is a *pocket* and is written OCCUPIED for the frontier logic. The
rule was written for the strip behind a bed and the blind disk under the spawn point, and its comment argued "a
toilet does not fit in 3 m2 behind a wardrobe". From the door the camera saw the bathroom's far wall, which
closed the unseen half of the room off from the rest of the unknown; 92 cells = 0.92 m2 -> pocket -> settled.
No frontier, no opening, no room with `frontier > 0`, so nothing ever sent the agent through the door, and
the frames from the threshold showed a towel rail and a mesh hole, never the toilet.

**Fix / workaround:** `SightSettings.pocket_max_width_m` (0.6): a pocket is settled only if nothing could
stand in it -- its inscribed width (twice the component's largest EDT) is under the bound -- or the agent has
itself stood on it (the spawn island). With it the bathroom reads as two live frontier clusters of the bedroom
(`(2.5, 4.8)`, `(3.5, 5.4)` in the re-runs). Beside it `OpeningSettings.min_cells` went 8 -> 5: the bathroom
door was 7-8 frontier cells wide and flickered in and out of the openings between loop points.

**Don't:** Do not read "the agent stood in the success region and saw nothing" as a detector failure first.
Check the frontier record for the room the goal is in; a region that is never a frontier was settled by
something. And do not expect the pocket fix alone to turn this episode green under the 3B oracle: with the
frontier offered, `qwen2.5:3b-instruct` valued the bedroom holding it at 0.01 ("bedroom, entered, no toilet
there") on every loop point of the fourth run, below `min_prob`, so RPT* never chose it. The 14B model is the
benchmark model for a reason.

---

## 2026-10-12 — "unable to find CUDA device 0 among 3 EGL devices": the renderer died because `apt` upgraded the NVIDIA userspace under the running kernel module, and the iGPU is no fallback

**Symptom:** Every `gibson.run` (and the `run_benchmark.sh` smoke) died in two seconds with
`Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA device 0 among 3 EGL devices in
total / WindowlessContext: Unable to create windowless context`. `nvidia-smi` said `Failed to initialize NVML:
Driver/library version mismatch (NVML library version: 595.99)`, so `gpu_plan.py` and the `gibson.run` GPU gate
died too -- while `torch.cuda` in the detector venv allocated on the card without complaint.

**Root cause:** `/var/log/apt/history.log`: `nvidia-driver-595-open` 595.91.07 -> 595.99.02 at 09:46, seventeen
minutes after boot; `/proc/driver/nvidia/version` still said 595.91.07 and DKMS had the 595.99 module built for the
next boot. CUDA tolerates the skew, NVML and the NVIDIA EGL vendor do not: with
`__EGL_VENDOR_LIBRARY_FILENAMES=10_nvidia.json` forced, `eglQueryDevicesEXT` returned **zero** devices, so the three
EGL devices glvnd listed were all Mesa's (the NVIDIA card through a driverless Mesa device, the Intel iGPU, and
`EGL_MESA_device_software`), none carrying `EGL_CUDA_DEVICE_NV`, and habitat-sim's `gpu_device_id=0` CUDA match
had nothing to match. The obvious fallbacks failed for reasons worth knowing: `gpu_device_id=-1` (no CUDA match,
EGL device 0) created a context on the **Intel iGPU** and rendered **black RGB with valid depth** -- Magnum's GPU
validation showed `GL_OUT_OF_MEMORY in glTexStorage2D(texture too large)`: a Gibson `.glb` is one mesh with one
**16384 x 16384 JPEG** texture, which iris cannot allocate (hiding the ASTC extension so Basis transcodes to BC7
changed nothing: the texture is a JPEG); `LIBGL_ALWAYS_SOFTWARE=1` segfaulted at context creation (kms_swrast over
the driverless NVIDIA node); and `MAGNUM_DEVICE=2` was ignored -- habitat fixes Magnum's device to 0 (`found 3 EGL
devices, choosing device 0`), so the software device is unreachable while any hardware device is enumerated.

**Fix / workaround:** The reboot (the DKMS module is built). Until then, `gibson.run --software-render`
(habitat `gpu_device_id=-1`, GPU gate skipped, `renderer` recorded in `run.json`) under an unprivileged user
namespace that hides the DRM nodes, so Mesa enumerates ONLY its software device and habitat's device 0 is
llvmpipe: `unshare -rm sh -c 'mount -t tmpfs none /dev/dri && exec "$@"' _ python -m ... gibson.run
--software-render ...`. `run_benchmark.sh HABITAT_RENDER=software` (with `GPU_PLAN=off`, since the plan needs
`nvidia-smi`) does exactly that for the run, the preflight and the five-scene campaign, and `record_episodes.sh`
re-wraps a run whose argv carries `--software-render`. Measured: a Wiconisco frame renders correctly (door,
refrigerator, textures) at ~30 ms a step on llvmpipe (LLVM 20, 32 threads), with ~1.3 GB of RAM for the texture
and its mips per simulator process -- the renderer was never the bottleneck, the CPU LLM is. The former
`HABITAT_SOFTWARE_RENDER=1` path (`LIBGL_ALWAYS_SOFTWARE=1`, device unchanged) never rendered on software and
now routes to this.

**Don't:** Don't reload the kernel module to "fix" the skew on a laptop whose desktop (gnome-shell on Wayland)
holds `/dev/nvidia0` -- it needs root and ends the session. Don't trust an RGB frame by `max() == 255`: the alpha
channel of habitat's RGBA sensor is 255 everywhere; check the mean and std of the first three channels. Don't
read `torch.cuda.is_available()` as "the GPU stack is fine": CUDA, NVML and EGL are three libraries with three
tolerances for the same skew. And don't run the Intel iGPU as a renderer for Gibson even when it creates a
context: the depth comes back plausible and the RGB is black, which the detector reads as an empty scene with
no error anywhere.

---

## 2026-10-11 — A more accurate target memory made a success fail: the standoff re-selection threw away a goal that had a path

**Symptom:** The regression check of the toilet closing rules on the nine inspection slots
(`runs/gibson_val_20261008T120112Z.inspection` vs `.inspection-fix`) came back 2 improved, 6 identical, and one
REGRESSED: tv Collierville/000194 went from SR 1 (STOP at action 253, SPL 0.25) to SR 0 (step limit, final DTG
4.14 m). Nothing about the television, the rooms or the LLM had changed -- the two decision logs are identical
through action 237.

**Root cause:** From action 238 the along-ray association cylinder (`association_depth_m`) kept the real
television's frames that the old 0.50 m sphere about the anchor had dropped (`depth_associations` 11), so the
3-D memory was refined by 0.11 m toward the true position. At action 242 that drift crossed `refine_distance_m`
(0.15 m) from the reference the standoff goal was selected at, and `TargetApproachPath` did what it always did on
a refinement: dropped the goal and re-selected one from a bearing-relative ring (8 offsets) about the new
position. The ring is a *different set of points*, each needing a navmesh projection and an A* path, and from
(0.94, 1.36) none of the eight had one -- while the dropped goal at (-0.28, 2.51), 0.84 m from the refined
memory, still did. Five pathless actions later the lock was released as "no safe path from here" (a boxed-in
release), the agent walked away, re-took the television from 4 m and released it the same way again. The
baseline only succeeded because its memory never moved: at action 246 a bottom-cut box of something else
0.95 m away was associated through the anchor, pulled the memory 0.4 m, and the OLD goal walked the agent into
the success region by luck. Two smaller things on the way: after a failed re-selection `reference_xyz` stayed
stale, so `goal_refinements` climbed by one every action (3 -> 8 in five actions, meaningless); and the goal
ring's radius is `terminal_distance_m - goal_tolerance_m - resolution` = 0.70 m, so a 0.15 m memory shift can
push a goal that was fine past the 0.75 m selection limit by construction -- the rule was tuned for selection,
not for keeping.

**Fix / workaround:** `TargetApproachPath.command` keeps a standoff goal through a refinement when it still
lies within `terminal_distance_m` (1.0 m) of the refined memory -- arriving there puts the agent inside the
terminal radius, which is all a standoff is for, and the inspection reads the fresh range from there
(`kept`, `goal_kept` in the diagnostics) -- and re-references on every refinement. A kept goal that a slow
drift carries past the terminal radius is re-selected the action it falls out (`stale`), reference or no
reference. `test_reobservation_refines_goal_but_missing_frames_do_not` pins all three.

**Don't:** Don't read "the re-run reproduced the original" (`--expect same`) as "the closing behaved the
same" -- compare the phase timelines; the baseline STOP here was a different mechanism (a history STOP on a
memory pulled by a stranger's box) from the one the fix would have needed. Don't revert an association
improvement because a downstream consumer happened to depend on the old memory standing still; find the
consumer that cannot take a moving input. And don't treat a re-selection as free: every selection needs a
navmesh projection and a path, and a point that had both is worth more than eight that might.

---

## 2026-10-11 — Nine failure recordings of the full Gibson run: the toilet was in the frame at 0.92 confidence and the inspection "saw nothing"

**Symptom:** toilet SR 0.288 over 208 episodes (131 of 132 failures were 500-step timeouts, median final DTG
4.5 m), couch SR 0.41 in Wiconisco against >= 0.86 elsewhere, tv/bed successes with SPL 0.07-0.25. Nine
representative episodes were re-run with `--record` (`runs/gibson_val_20261008T120112Z.inspection/`, all nine
reproduced on the first attempt) to look for corridor bottlenecks and skipped doorways (toilet), the misleading
entity (couch) and bad RPT* room orders (tv/bed). The decision logs answered before the videos were watched.

**Root causes (from `steps.jsonl` + the evaluator's `trajectory.csv`, which has the true DTG per step):**
- **Toilet timeouts are not search failures.** In all three, the agent found the toilet, walked up to it
  (true DTG 0.00 -- standing inside the success region), ran INSPECT and released it with `inspection saw
  nothing in any view`, then spent the remaining 315-458 actions elsewhere (Darden/000193 did it twice, at 132
  and 409). The detector had the toilet in every one of those frames at confidence 0.90-0.96. What the
  inspection counted as nothing is a box touching the **bottom edge of the frame** (`y2 >= 479`): a toilet is
  ~0.4 m tall, the camera is at 0.88 m with a 63-degree vertical field and the Gibson action set has no
  LOOK_DOWN, so at the closing's inspection range (1.2-1.6 m) the toilet's base is always below the frame and
  its box is always bottom-cut; the border rule (meant for things half out of the picture) then rejects every
  view. Whole run: 93 of the 148 toilet failures (63 %) had locked on and released a candidate -- 40 with this
  exact release as the last one, 23+11+7 with the `unverified` family -- and only 53 (36 %) never sighted a
  toilet at all. A STOP at the moment of release would have made all three recordings successes.
- **Couch false positives: no inspection happened.** Wiconisco/000123 STOPped at action 2 and /000010 eight
  actions after the sighting, both on `fresh terminal target confirmation` at 0.8-0.9 m -- the terminal path
  accepts a 'sofa' box without the inspection the toilet gets; /000114 ran 27 INSPECT frames, could not decide
  and STOPped on `inspection exhausted` = accept-by-default. Two of the three (000123, 000114) locked on the
  **same object** at world (0.4, 5.7), 0.35-0.63 m high, 2.7-9.8 m from any annotated couch; 000010 on another
  at (5.5, -3.6). The detector reports it as 'sofa' at 0.81-0.93 (and 'couch' 0.78-0.91): the frames
  (`_verify.jpg`/`_stop.jpg`) decide whether it is a bench/daybed or a real sofa the SemExp annotation lacks.
- **Low SPL is not RPT* ordering.** Collierville/000043: the *real* TV was in view from action 1 at 3.9 m and
  the closing released it four times with `no safe path from here` (the success is incidental: inside 1 m when
  the budget ran out). Collierville/000194 and Corozal/000133: three false candidates each (a 'television' at
  2.6-4.7 m from the real one; three 'beds' at 0.6-0.8 m range, all at 0.72-0.84 m height -- table height, not
  bed height), each costing a 13-action BACK_OFF plus a 13-action VERIFY before an `unverified` release
  (~75 actions per episode), the real target found at 222 / 448. In 000194 the real TV was also seen at action 3
  and released `unverified`. The rooms the oracle ordered were few and mislabelled (Corozal: 7 of 12
  `living_room`); the time went to the closing, not to the order.

**Fix / workaround:** pending -- the recordings exist so the fix is chosen on evidence, not on this entry.
The candidates, in order of expected yield: (1) in the inspection, accept a box cut **only at the bottom** when
its top edge is well inside the frame -- for a floor-standing object at close range that cut is geometry, not
occlusion; the bearing is sound and the range can be read from the box's upper part (or inspect low classes
from >= 1.5 m, where the toilet fits); (2) make the terminal confirmation of a sofa/bed-height candidate go
through the same inspection a toilet does, and never accept on `inspection exhausted`; (3) treat `no safe path
from here` to a visible target as a planning problem (goal projection / inflation), not a release. Measure each
on the 148 toilet failures and the 18 Wiconisco couch STOPs before touching defaults.

**Don't:** Don't take a timeout with a far final DTG as "never found it" -- the lean record's `target_closing`
counters (`releases`, `inspection_releases`, `last_release`) say whether it was found and let go, at no cost.
Don't read the recording's `decision.info.distance_to_goal_m` as the score: it is the agent's own number
(0.00 at action 0); the evaluator's is `trajectory.csv`. Don't infer "the agent reached it" from the closing's
range alone either -- cross it with the true DTG, as above. And don't widen the border rule globally: it exists
because a box cut at the top or the sides has no trustworthy centre.

---

## 2026-10-11 — The full Gibson run died at episode 919 of 1,000: one published start lies 3.3 cm outside the navmesh, and a turn "moved" the agent

**Symptom:** `runs/gibson_val_20261008T120112Z`, 36 h in, 918/1000 done (SR 0.666): `EnvContractError: episode
'Wiconisco/000118', action 1: TURN_LEFT should turn in place (move at most 0.020 m), but the agent moved 0.033 m
(0.033 m across the floor at -73.3 deg from its heading, +0.000 m up)`. The supervisor replayed the run with
`--resume`; it died on the same line, and -- correctly -- was not relaunched a third time. The launcher's
exit trap had also killed the detector and the Ollama; the supervisor restarted both from its snapshot.

**Root cause:** The 2026-10-07 fix for off-mesh published starts excused the *vertical* placement
(`settle_m`, 0.30 m) and kept "the horizontal bounds stand". Measured against all 1,000 starts with the
shipped navmeshes (`PathFinder.snap_point` / `try_step(start, start)`), exactly one, Wiconisco/000118, lies
**outside the mesh horizontally** -- 0.033 m, the only start `is_navigable` rejects; the other 999 are on it
to the millimetre. In habitat-sim 0.2.4 `turn_left`/`turn_right` are registered as *body* actions, so the
step filter runs for a turn too and `try_step(start, start)` returns the nearest point *of the mesh*: the
warm-up turn pulled the agent 3.3 cm onto the edge (and 6.4 cm down -- excused). A kinematic self-check
refusal is an `EnvContractError`, not an agent error, so it leaves `run_benchmark` and ends the run. Two
things the earlier entry got wrong on the way: "turns bypass the navmesh filter" (they do not -- Darden/000000
is lifted 0.10 m by its first *turn*, which the runner's "first action that displaces the agent" rule
absorbed by luck of phrasing), and that the start validation at preflight would catch a bad start (it
audits the FMM field, not the navmesh).

**Fix / workaround:** `KinematicTolerance.settle_xy_m` -- the horizontal counterpart of `settle_m`, taken off
the first action that moves the agent along its own direction, up to the bound; Gibson sets 0.05 m (one map
cell, the width `forward_overshoot_m` already allows a step), measured: 0.033 m is the only non-zero value
in the split. Finishing the run then needed a resume whose frozen configuration differed from the recorded
one in `kinematics` and `source_sha256` -- which the harness refuses, rightly, by default. The honest path
was an *explicit, bounded, recorded* exception: `MetricsLogger(accept_changes=...)` /
`gibson.run --resume --resume-accepting-changes kinematics source_sha256` (choices limited to
`RESUMABLE_CONFIG_CHANGES`: `kinematics`, `source_sha256`, `gpu_plan` -- provenance, or a self-check that
only refuses an episode, never an input to a decision or a score). The recorded config stays the run's;
the resume's `run.json` entry carries `config_changes` with both values; `comparison.md` / `audit.json`
title and disclose it; the supervisor passes it via `RESUME_ARGS`. Verified before relaunch: the replayed
preflight differed in exactly those two keys and the logger accepted the resume without writing. The run
continued past 000118 (chair, SR 1, 32 actions) on 2026-10-11 05:07Z.

**Don't:** Don't excuse a first-action displacement without a bound -- the point of the contract is to catch
a sim that moves on a turn. Don't widen `position_m` (the turn-in-place bound) instead: every turn of every
episode would then tolerate 5 cm of drift. Don't make the resume comparison lenient by default or silently
drop keys from it -- the override is named per key, per resume, in the record. Don't trust a protocol
docstring's account of the simulator's mechanics without measuring: `try_step(start, start)` takes one line.
Follow-up: a preflight check that reports starts off the mesh beyond `settle_xy_m` *before* a day of compute
(the measurement script is three loops over `PathFinder`).

---

## 2026-10-08 — A Gibson run cannot be resumed by the script that launched it, and three edits that would kill a day-long run

**Symptom:** Preparing to supervise the 1,000-episode `runs/gibson_val_20261008T120112Z` unattended, the
documented resume -- `run_benchmark.sh --output <dir>` -- would have been refused as "cannot resume with a
different config (gpu_plan differ)" the moment the run died, although nothing about the experiment had changed.

**Root cause:** `MetricsLogger._resume` compares the *entire* config text to the recorded one, and the config
embeds the whole `gpu_plan.json` -- `free_mib`, `used_mib` and the "fits: N MiB of M left" reasons. The
launcher re-runs `gpu_plan.py` on every start, so the numbers are whatever the card happens to hold at
that second (949/23103 MiB at 12:01, 916/23136 at 12:06 for the same machine). The config also embeds
`source_sha256`, the hash of **every `.py` under `sparx_agency/`** (`run.py::source_fingerprint`), tests
and tools included. Two consequences nobody had written down: (1) under `GPU_PLAN=auto` the script's
resume essentially never passes; `GPU_PLAN=off` does not help either (its plan record is `source: none`,
which differs too); (2) adding or editing *any* `.py` in the package while a run is going -- a new helper,
a test -- makes that run unresumable, forever. And a third hazard found on the way: bash reads a running
script incrementally, so editing `run_benchmark.sh` while its `bash` is blocked in the `gibson.run | tee`
pipeline corrupts what it executes after the pipeline (the summary step, the exit code).

**Fix / workaround:** The resume that reproduces the configuration byte for byte replays the *original
processes*: `gibson/supervise_benchmark.sh` snapshots the live `gibson.run`, detector and Ollama (argv,
environment, cwd from `/proc/<pid>/`) into `<output>/replay/`, restores `gpu_plan.json` to the recorded
one, and relaunches the saved argv + `--resume` under the saved environment via `os.execvpe`. Verified
non-destructively: the replayed argv with `--preflight --resume` produced a `configuration` identical to
`run.json`'s. Rules for the run's lifetime: no `.py` edits under `sparx_agency/` (the supervisor itself is
a `.sh` for exactly this reason), no edits to a `.sh` that a live bash is executing (restart the stateless
supervisor instead), and a new `.py` only once `summary.json` exists. The proper fix -- `_gpu_plan_record`
dropping the volatile keys, or the resume comparison excluding provenance -- is a `.py` change and so waits
for the run to finish.

**Don't:** Don't read "It resumes" in BENCHMARK.md (before 2026-10-08) as tested for the auto-plan path --
it held only for `GPU_PLAN=off` runs whose services were still answering. Don't `cp /proc/<pid>/cmdline`
and expect to overwrite it later: the copy inherits the `r--r--r--` mode (`cp --no-preserve=mode`). Don't
expect `tr ... < /proc/<pid>/environ 2>/dev/null` to be silent for another user's process: the
redirection fails before `2>` applies; test `-r` first.

---

## 2026-10-08 — The 3x3 on Darden/Markleeville/Wiconisco: a manoeuvre that looped, a capsule that closed a corridor, a toilet that left through the floor

**Symptom:** `runs/benchmark_9episodes_20261008T082632Z`: SR 6/9. Markleeville/000000 and /000001 (bed)
spent 106-114 actions in `target_back_off` and 69-97 in `target_verify` -- six to eight backing manoeuvres,
each ending `action bound reached`, each followed by an `unverified` release and a re-takeover of the same
0.73 m-high surface read as a bed -- then STOPped on it, 5-6 m from the real bed. In /000000 the stair
capsule also cut the 0.55 m corridor between the stairwell and the wall to a 0.25 m gap (75 free cells
beyond it unreachable). Wiconisco/000000 (toilet) saw a bottom-cut box at 1.32 m, took the verification
step toward it, lost it through the bottom of the frame, and repeated that four times before releasing.

**Root causes and fixes:** (a) the manoeuvre's arrival test (0.25 m) was tighter than the converter's, so
at 0.25 m from the goal the route read complete, the idle result became a TURN, and eleven turns on the
spot spent the bound -- arrival is now the converter's tolerance plus half a step, or three idle actions
while facing the goal, with a retreat budget of two half-turns plus the steps. (b) The re-verification swept
without re-seeing the box; the release was `unverified`, which `override_confidence` (0.80) bypasses, and
the search walked two steps back to the same spot -- the loop. One manoeuvre per anchor
(`backoff_repeats`), and a candidate the wider perspective never confirmed is released as
`unverified from a wider perspective`: the failed-inspection radius, no override, refused from twice the
same-spot radius. The re-fly then showed two more holes in the same wall: the *legacy* target pursuit
walked to the disproved table and STOPped on it nine actions after the closing had given it back
(`refuses_far_candidate` only bound far candidates -- now every release that disproved its candidate from
close binds it at any range), and without a manoeuvre to spend, the same surface was re-taken 35 times
through the `unverified` override (the second unverified release of one anchor now escalates to
`unverified again`, non-overridable). With both, Markleeville/000001 found the real bed (SR 1, 331
actions); /000000 still STOPped on a `bed` whose surface stands 0.56 m above the base -- a mattress
height, inside every band -- which no geometry check separates from a bed. (c) The fixed capsule margin (0.6 m plus a cell) double-counts the body radius the
planner inflates by anyway; `adaptive_capsule` shrinks it per flight until no body-passable region is
split or erased (0.3 m there). (d) `_approach_pitch` on every step toward the target -- the pitch for
where it will be after the step -- and, in the LOOK-less Gibson protocol, no verification step that
pushes a bottom-cut box out of the frame (`_step_keeps_box`); a fresh near edge inside the terminal
radius is terminal evidence wherever the centroid is, and a fresh centred reading beyond it resumes the
approach instead of spinning out the budget. The false `bed` itself -- a table top at 0.73 m -- stands:
`target_context.CLASS_HEIGHT_BANDS` flags a bed surface too LOW (a counter), not too high.

**Don't:** don't let a manoeuvre's own arrival rule disagree with the converter's -- the converter decides
when a route is done and what idle looks like. Don't make a release overridable when it came from a second
viewpoint; the override exists for *far* flickers. Don't trust a 40 px synthetic box at 0.8 m in a test:
a real detector's box reaches the floor and is cut by the bottom edge, and `clipped_box` has a rule for
exactly that. And don't predict a turn's effect on a box horizontally only: a low box stretches toward the
bottom edge by cos(b)/cos(b').

---

## 2026-10-08 — The stair mask trapped the agent: a capsule of "genuine" obstacles with the agent inside it

**Symptom:** `runs/smoke_gpu_plan_20261008T070438Z` (Collierville/000000, toilet): the agent
walked a doorway peek past the foot of a staircase, the connector was placed at action 190, and from
action 191 the exploration fallback printed `off the observed passable map; the idle turn is the
only move` for the remaining 309 actions -- `hold: 309` in `exploration_fallback.stats`, `footing: 0`
(the floor around the agent was *known*, so no footing sweep), the closing never engaged, SR 0 at
the step limit. The recorded planning grid showed the agent's cell `occupied`, inside a smooth
1.3 m-wide capsule running along the flight's centreline and 0.65 m past its foot into the room.

**Root cause:** `peek_stairs.stair_peek_mask` rasterises each seen connector polyline clipped to
`height ± departure_m` (0.45 m) with a capsule radius of body radius + arrival tolerance + one cell
(0.53 m -> 6 cells of 0.1 m = ±0.65 m), and `SpawnFloorGuard.observe` writes that into the planning
world as `occupied`. The polyline begins *on the floor* at the foot anchor (z = storey height), so the
floor in front of the stairs was in the band, and the capsule covered the corridor the agent used to
pass the foot 0.60 m from the centreline. The weighted A* relaxes a blocked start only for its own
cell and its inflation *skirt* -- cells "lethal but not truly occupied" -- and the mask cells are
truly occupied by construction, so every plan from inside the capsule returned `NO_PATH`;
`peek_planning_world` re-applied the same capsule for the fallback's own plans. The hold rung has no
recovery: the idle turn does not change the map, so the episode was over at action 191.

**Fix:** two rules in `peek_stairs` (and the guard, which combines stairs with observed drops):
the floor run of a flight within `FLOOR_RUN_M` (0.10 m) of the storey plane is not drawn (the treads
above and below still are), and the cells within body radius + one cell of every pose the agent has
taken on the storey -- `SightLedger.walked_mask`, plus the current pose, which the ledger records only
after the world is confined -- are carved out of the mask (`walked_exemption`, counted as
`walked_exempt_cells`). The agent was physically there; nothing written for planning may outvote
that. Regression: `test_spawn_floor_guard.py::test_an_agent_inside_the_stair_capsule_...`.

**Don't:** don't widen `_clear_start_footprint` to open "genuine" obstacles around the start -- that
is the one guarantee that A* never threads a real wall; the mask had to stop claiming the cell
instead. Don't read `footing: 0` as "the agent could see its feet": a mask makes the blind disk
*known*, which is why the footing sweep -- the rung meant for exactly this -- never ran. And don't
compare this run's SR with the morning's CPU run of the same episode as evidence about the closing
changes made the same day: the detector on CUDA and the LLM on the GPU give different numerics,
the search diverged before any target was seen, and the failure was in a module neither change
touched.

---

## 2026-10-07 — The Allensville frame-by-frame audit: four failures that were not where the video said they were

**Symptom:** Three recordings of the 5x3 benchmark (`runs/zson-benchmark-5x3-20261005/campaign/frontier/
Allensville`) read as: table objects not registered; a room graph with R0 alone for 30 actions, then R0
and R2 with no R1; a kitchen island STOPped on as a `bed`; a strong bathroom valued 0 for a toilet; a
0.95 toilet ignored at 2.4 m; the approach to it turning in place and reverting to exploration.

**Root causes (from `steps.jsonl`, not from the video):**
- The table's vases projected with valid depth and were dropped as `same_frame_association`: the
  landmark map associated ANY class within 0.70 m of a landmark's centroid, and the fused cabinet had
  already been fed that frame. The same rule voted a toilet (4 votes) and the bathtub 0.5 m beside it
  into one landmark, which flipped to `bathtub`, released the toilet's lock as "contradicted by the
  map" with a 2 m rejection radius -- which is why the 0.95-0.97 toilet in plain view at 2.39 m was
  refused nine frames running (beyond `rejection_min_range_m`).
- The pid followed the LARGER half of a split (the IoU rule): the spawn room was R0, became R1 when a
  door 0.5 m away cut it off, merged back as R0, and re-split as R2 after the 10-tick memory had
  expired, while the hallway the agent had just walked into carried R0 and the spawn room's
  "entered" record. The step-40 "doorway" is a uniformly narrow hallway (0.30-0.50 m clearance for
  forty actions on the saved map) -- no partition could have split it there; the far rooms appear at
  step 80 when their floor is first seen.
- The counter front read as `bed` had its supported points 0.67-0.70 m above the agent base (every
  real bed of the run: 0.11-0.52 m) inside a STRONG kitchen, and locked on two frames from one spot.
- The bathroom's 0 was not the type prior: the warm-up spin at the spawn point "saw" >50 % of its
  cells through the door (`scanned:seen_from_scan`) and finished it with the bathtub inside.
- Within 2 m a toilet's box always touches the BOTTOM image edge; `clipped_box` called that a
  sliver (`border_rejections` 3 -> 26), no fresh frame counted, the ±30-degree sweep ran for 12
  actions and released `unverified`, twice. And at 1.18 m a centred, counted frame was followed by
  TURN_LEFT: the hold was satisfied, idle, and the headless agent executes idle as a turn.

**Fixes:** classes never merge in the landmark map (no class vote; a cup on a table is two instances),
same-class instances told apart by size (0.35 m centroid floor, footprint IoU 0.25); a height-aware
same-frame rule; the bottom-edge floor cutoff is not a sliver; birth anchors and an episode-long memory
in the room registry; walked-through clearance dips carved like snapped doors; the home-object floor
and the no-finish-from-outside guard; `override_confidence` 0.80 past the unverified/map memory; the
context check (room type, height band) with a four-frame, two-viewpoint lock; a verification frame
that keeps the box whole with the pitch following the target's elevation (LOOK_DOWN below the camera,
LOOK_UP above -- a fixed look-down drove a wall-mounted television out of the frame), and the same
dynamic pitch on the close approach inside 1.3 m. Progress entry 024.

**Don't:** read "the lock was wrong" off a map release without checking what landmark the anchor fell
on -- a neighbour of another class within 0.70 m was enough to outvote it. Don't let classes merge in
the object memory at all: every cross-class rule the map had (centroid radius, then footprint overlap)
collapsed something resting on or beside something else. Don't take `scanned:*` for the type prior in
the HUD; `excluded` says which. Don't hand a VERIFY candidate a hold that is already satisfied -- in
Habitat idle is a turn -- and don't hardcode the direction of the LOOK that replaces it: the target's
elevation decides. And on this machine `.venv/bin/python` has no pytest; the shell's `venv/bin/python`
(3.12, pytest 7.4, numpy 1.26) is the one that runs the suites.

---

## 2026-10-07 — Before the 1,000-episode run: the harness itself would have ended it, and the budget went to peeks

**Symptom:** a review of the Gibson adapter before the full val split found that the kinematic
contract -- our own self-check, not the scorer -- raised `EnvContractError` out of `run_benchmark` on
(a) a published start settled onto the navmesh by habitat-sim on the first translation (starts lie up
to 0.28 m below the mesh; Darden/000000 0.10 m) and (b) an on-mesh forward step that climbs more than
0.20 m, which the shipped Collierville/Wiconisco navmeshes do about once in 300 steps (treads, split
levels inside the storey). Either ends the run, and a `--resume` replays the same refusal for ever.
Two smoke episodes then spent 247 of 500 actions on ten doorway peeks and 120 on target verification
(18 and 10 releases) for one bed.

**Root causes and fixes:** `KinematicTolerance.settle_m` (0.30, additive, on the first action that
moves the agent -- turns bypass the navmesh filter, so with a warm-up rotation that is action ~13,
not action 1) and `climb_m` 0.60 for Gibson; a near candidate released from the spot the agent still
stands on is the same evidence and is refused until it moves; a completed peek covers the cone it
looked into so the frontier it reveals through the same door is not a new opening; the oracle is told
that unexplored places share the chance of the rooms still missing (0.35 on each of fourteen doorways
became 0.20 on each of six -- still generous; the peek's real price is the walk, charged as travel).

**Don't:** key a "first action" excuse to `sent == 1`; trust that a 14B on a 32-thread CPU is cheap --
64 s median per oracle call, 14 calls, 15 of a 21-minute episode; or read a `pass` verdict the model
never wrote: a missing key is the first pass, never the second.

## 2026-10-05 — The 5x3 same-storey benchmark: eight failures, five mechanisms, none of them the reasoning

**Symptom:** `runs/zson-benchmark-5x3-20261005`: SR 7/15, SPL 0.27, DTG 3.6 m. Every success was a
STOP at an annotated instance in 59-299 actions. The failures: two STOPs at a kitchen counter front
read as `bed` from 0.55 m (one at action 2, the real bed in view behind it); 160 actions of
MOVE_FORWARD into an obstacle the map did not show, locked on a real sofa, until the closing's
160-action bound raised an agent error; 350 actions wedged on a transit, the pose jittering 5-10 cm;
a loop transit that walked down an unseen staircase into the basement; two Klickitat episodes in a
split-level house standing on a half-level the floor band did not recognise (361 idle turns; 319
relocations); two large floors that ran out of budget peeking openings.

**Root causes and fixes:** (a) the STOP rule takes two frames of a target-class box at terminal range
and the map had no other landmark at the counter to outvote it -- a detector limit, left as the
user's design call (a VLM look, or a first sighting from farther away); (b) the closing had no
stall clock -- added (`approach_stall_actions`); (c) the policy's blocked clock was reset by any
5 cm displacement, which a collision slide provides -- now 60 % of a forward step; and a BLOCKED
opening was re-chosen by the supervisor's escape hatch at one oracle call per seven actions -- now
retired; (d) the stair-blocked planning world covers seen stairs on fallback routes only; (e) the
floor band is anchored to one storey height, so a half-level 0.5 m down is neither floor nor obstacle.

**Don't:** read "an upper floor" off a storey's rank alone -- the first launch of this campaign said
so for every storey above the lowest and had the 14B send the couch search upstairs from a ground
floor with a living room found on it; Newfields, Leonardo and Klickitat all have a basement. Give
the rank from both ends and let the rooms decide. Don't poll a wedged episode by its oracle calls:
with the 14B each costs 40 s, and 34 of them made a 500-action episode take 58 minutes.

## 2026-10-05 — End-to-end review before the 5x3 benchmark: the room numbers churned because a snapped door did not cut the floor

**Symptom:** Ranchester couch recording (`runs/zson-couch-crossfloor-3x-20261005-b`): the room behind
one door was numbered R23, R26, R28 and R52 across ticks; pids reached 56 for ~10 rooms;
`partition_revision` 86 in 290 actions. Earlier, Hanson renumbered one bedroom R11 -> R14 -> R16
while the agent stood in it, and a corridor swallowed a room for a tick and spat out a new one.

**Root cause (two layers, found by replaying the watershed on the saved map along the trajectory):**
`door_carve_mask` snaps a detected door to the nearest medial-axis choke and carves a disk sized to
the clearance there. Near the frontier the clearance at a skeleton cell is the distance to an
UNKNOWN cell, so the disk is narrower than the passage and does not sever the floor -- 47 % of the
snapped disks in the replay. Such a disk is not in the merge barrier (the design relied on the two
rooms lying on different sides of the carved mask), so the room behind the door merged into the
hallway on the ticks the snap found a choke (`depth 0.00 < 0.2`) and split off again on the ticks it
did not (the skeleton changes every action; one door toggled 44 times). Then the registry: it
compares fresh rooms to the PREVIOUS tick only, so a room absent for one tick is retired for good,
and when it re-separates the only containment candidate -- the hallway -- is already consumed by its
IoU match, so a new pid is issued. Replay: 36 pids for 10 rooms, 26 deaths.

**Fix:** a snapped disk that leaves the floor around it in one 4-connected piece falls back to the
plain barring disk (`_severs`); the registry keeps a vanished room's mask for ten updates and a fresh
room nobody claims re-adopts its pid (`RoomRegistry(memory_ticks=10)`), ties broken by overlap.
Replay: 11 pids for 11 rooms, 3 deaths. The scan ledger re-judges a sticky verdict when the room
grows past 1.5x the size it was reached on (a sliver finished as a fragment, walked into under the
same number).

**Don't:** diagnose churn from `room_partitions.npz` -- it holds only the final partition per storey;
replay the segmentation along `trajectory.csv` on `floor_maps.npz` instead (the reviewer's scripts
grew a 3 m disc of the final map along the path -- a proxy, but it reproduced the numbers).
Don't make the choke sticky per door first: it treats the symptom; the non-severing disk is the cause.

## 2026-10-05 — Three more from the same review: a storey measured on the stairs, a tilt that never turned, an evaluator gate on the median height

**Symptom / cause / fix, briefly:**
- The atlas created Hanson's upper storey at 0.277 m (navmesh 0.108 m) from the three poses it settled
  on -- the eased last treads -- and 218 of 300 later poses read 0.10 m, outside `stable_height_m`
  (0.12): no staircase seen up there ever became a portal, the map's floor-clear mask never fired, and
  the oracle's storey line said rank 2 of 3. A storey first reached by a ground-truth traversal is now
  pinned to the connector's navmesh height when within `floor_match_m` (`storey_height_pinned`).
- A fresh, in-range sighting 20 degrees off the centre column at a pitch other than the one the
  target's height predicted produced LOOK_DOWN/LOOK_UP for the whole 24-action inspection budget: the
  converter tilts before it faces, so asking for the predicted pitch never let the centring turn
  happen. A fresh off-centre sighting is centred at the pitch it was seen at (`camera_pitch=None`).
- `MultiFloorDistance.success` gated on the goal region's MEDIAN height; Klickitat's chair samples
  lie at +0.18 m and -0.41 m (a split-level floor), so a STOP beside the lower chair scored 0 four
  centimetres from the region. The gate is the nearest goal sample's height now. Signature in any
  finished run: `success: false` with `distance_to_goal_m < 1.0`.
- The 3B oracle copies the worked example's words as well as its numbers ("home: living room,
  sometimes a bedroom" for a CHAIR). The 14B does not, and it is the model the prompt was written
  for; on this CPU it answers in ~40 s (generation-bound, ~6 tok/s; the system prompt is prefix-cached).
  Given only "(this one at +3.1 m)" it read an upper storey as the ground floor: tell it the storey's
  rank in words (`storey_position`), not its height.

**Don't:** run a review of this size serially -- five read-only reviewers over disjoint layers,
each with the records to confirm against, took an hour and found eleven real defects; reading the
same 25k lines alone would not have.

## 2026-10-05 — Ranchester couch from upstairs: the model said "downstairs" sixteen times and the agent peeked thirteen doors instead

**Symptom:** `runs/zson-couch-crossfloor-3x-20261005`, Ranchester, couch, start on the upper storey.
The oracle's STEP 2 read "upper floor (bedroom found); no living room here; living rooms are
downstairs" at all sixteen loop points; the staircase, once seen (action 166), was valued 0.60 with
"unvisited ground floor holds the living room"; the agent stood 0.9 m from its foot at action 171
and walked away. Every RPT* order put the stairs LAST behind 0.25-valued gaps; thirteen peeks at
~25 actions each (one 85, a 13 m walk to a gap); 500 actions, never downstairs. Also: "Source
changed during the frozen comparison" aborted the campaign after the first episode.

**Root cause:** The 3B model copies the worked example's numbers -- 25 on every unexplored node,
60 on the stairs -- whatever its own STEP 2 says, so rule 2b's "less (10-20) when the home type does
not belong on this storey" was never applied, and the code's uncertainty floor (0.25) could not tell
the two cases apart. Under RPT*'s expected-time objective five cheap nodes at 0.25 beat one far node
at 0.60 every time (and rightly, for those numbers). The campaign abort: the runner fingerprints
every `.py` under `sparx_agency/` at start and per episode; I edited the generator and its tests
while the campaign flew.

**Fix / workaround:** Ask for the STEP 2 verdict as a field (`home_here`: found / missing /
elsewhere) -- small models answer it reliably in words -- and let the code apply rule 2b's
arithmetic: `elsewhere` reads every unexplored node at `unexplored_elsewhere` (0.10) exactly;
`found` / `missing` keep the 0.25 floor. Documented in `search_node_oracle.py`, `LoopSettings`,
the README and progress entry 022. Do not edit Python under `sparx_agency/` while a frozen
campaign runs; Markdown is not fingerprinted.

**Don't:**
- Don't lower the floor globally to fix this: the 0.25 floor exists because the same model wrote
  0-1 on every door of the Hanson toilet storey (bathrooms DO belong upstairs) and left at action 27.
- Don't parse the free-text `storey` line for "downstairs": it is right in words and the numbers
  still disagree with it; the structured field is what the code can act on.
- Don't make the stairs cheaper (the 8 m floor-change leaf) to win the order: the leaf is what a
  storey change costs, and the problem was the gaps' worth, not the stairs' price.

---

## 2026-10-05 — Hanson re-fly: a railing is not an exit, a pocket is not a room, and a spawn beside a bed is a 160-action spin

**Symptom:** `runs/zson-hanson-3x-20261005` (the three Hanson episodes after the door-cut /
floor / glance changes). The toilet episode reached its toilet at action 359 but spent most of
them on places a person would not have gone: the two ends of a balcony it had seen whole from
the threshold (O2 and O6, actions 25-51), an island of unknown between the spawn point and the
bed in front of it (O4, "GAP", actions 51-82 -- back to within a metre of where it started), a
strip behind a bed segmented as its own room (R13/O9, actions 150-192), the balcony again
(R5, offered at 0.25 as "never entered" at action 206, 158 actions after the agent had stood
at its far end), and a bedroom it had seen the bed and the television of through the door
(R10, actions 270-289). At action 101 a bathroom vanity (`cabinet 0.84`) was cut off at the left
edge of the frame and the scheduler glanced right. At action 150 a room labelled `bedroom
strong` was still valued by the oracle at 0.10. Room numbers churned (R11 -> R14 -> R16 for one
bedroom) and each new number read `entered=no`. The potted-plant episode spawned beside a bed
with the plant in view 4.4 m away and spun 159 actions on "no safe target path" to an agent
error. The chair episode stopped at action 14 on a child's ride-on horse read as `chair` 0.9.

**Root cause:**
1. The frontier is "free beside unknown", and every unknown cell counted. A balcony railing
   (the depth returns nothing at floor height beyond it), a window and a mesh hole are frontier
   for ever; so is an unknown island enclosed by known cells. Nothing in the method said
   "I looked there and there is nothing to resolve".
2. A room was finished only by a full rotation inside it (or a scan point that saw half of
   it). Standing at the end of a balcony after walking its length finishes nothing; a 3-step
   peek is not a scan.
3. The watershed carved the strip behind the bed into a room; nothing recognised a doorless
   2 m2 region with no unknown boundary as a fragment.
4. The loop computed the exclusions, THEN re-classified the rooms inside the oracle call: a
   label that became strong on that action was excluded one loop point late.
5. One kind of object was by rule a weak label (the 2026-10-05 merge guard). A bed is not a
   guess.
6. The glance scorer values unknown floor only; a small room beside the route shows it little,
   and optimistic rays through a window value the garden.
7. `RoomRegistry` matched by IoU >= 0.15; a room seen through its door grows tenfold as the
   agent walks in and loses its number -- and with it `last_inside` and `searched`.
8. A level camera 0.88 m up sees the floor from 1.4 m; the spawn footprint is the only known
   free around an agent that has not moved; unknown is impassable; the locked target waited for
   A* for ever. Nothing ever looked down.
9. YOLO-World lands on the nearest prompt it was given; there was no toy in the vocabulary.

**Fix / workaround:** The sight ledger (`methods/sightlines.py`: looked-through unknown within
2.5 m from two poses, pockets under 3 m2, written OCCUPIED for the frontier logic only); the
`seen_through` and `fragment` verdicts in `room_scans.py` (no live frontier, and looked into /
walked through if narrow, or doorless and under 3 m2); `ObservedSceneGraph.refresh_labels`
before the exclusions; `SIGNATURE_OBJECTS` making a bed/toilet/oven label strong on its own;
cue glances toward a confident box cut off by the frame's edge; containment matching in the
registry; the footing sweep (LOOK_DOWN 30 degrees, a circle, LOOK_UP) for a pathless lock and
for the fallback's boxed-in case, with a no-rejection release of the lock after six more
pathless actions; a 0.35 m start-clearance rule in the samplers; `toy` / `rocking horse` /
`stuffed animal` / `bathtub` in the vocabulary. Progress entry 022.

**Don't:**
- Don't resolve unknown by range alone: beyond ~2.5 m the floor samples of a stride-8 depth
  image are 0.2-0.3 m apart, and an unknown cell between two seen rows is a sampling hole, not
  a railing. The looked-through band stops at 2.5 m for that reason.
- Don't write settled unknown into the planner's map: occupied is inflated by the body radius,
  unknown is not, and a settled railing would shrink the balcony the agent can still stand on.
- Don't finish a wide room because the agent stepped into it: the walls beside its door are
  behind the camera. Narrow rooms (clearance under 0.9 m) are walked through; wide ones need a
  pose that saw half of them, or the scan.
- Don't read a per-pixel "no return" as a settled cell in Gibson: the meshes have holes, and a
  ray through a hole in a wall lands in the next room's floor and would settle it from the
  wrong side. The 2-D cast through known free cells stops at the wall.
- Don't make the one-signature-object rule general: a chair, a cabinet, a desk, a plant names
  nothing on its own; the Ranchester hallway named "bathroom" by a toilet glimpsed through a
  door is still the reason weak labels never exclude.

---

## 2026-10-05 — Hanson/000000: a bedroom in two halves, furniture in no room, every door valued at zero, a peek cancelled on its own route

**Symptom:** Nine-episode campaign of 2026-10-04 (`runs/zson-campaign-3x3-20261004`). In the
Hanson toilet episode the HUD showed R0 and R6 as separate rooms for one bedroom while R7 and
the corridor were split correctly at a door; the bed, the desks and the wardrobe appeared in no
room's object list (the bedroom was "living room?" from its one chair); at action 27 the agent
walked to the stairs with two never-entered rooms and six doorways still unlooked-into
(`order: [stairs]`); at 164 it stood in front of a doorway with a sink through it and the oracle
had not even been asked about the room; and the one peek it did commit to (166) was cancelled at
196, "approach budget of 30 spent 5.8 m short", while the route it was following was fine.

**Root cause:** Four unrelated ones, all visible in `steps.jsonl` once decoded:
1. The door cut is a 0.75 m disk at the *detected* door position. YOLO's door landmark sat
   0.6 m off its doorway, in the bedroom's own floor at a junction; the disk severed the room
   around its bed (every border through the disk was "barred", so the merge could not repair
   it) and missed nothing it should have cut.
2. `_room_objects` credited a landmark to the room whose mask holds its *cell*. Furniture
   stands on cells the map reads OCCUPIED; the watershed labels free cells only; so beds,
   desks and wardrobes were `room: null` and rooms were named by whatever stood on free floor.
3. The 3B oracle wrote "small unknown room, no toilet fits" on every never-entered room and
   "no toilet glimpsed through gap" on every doorway -- the *words of the prompt's own example*
   (`"bathroom-sized room, no television fits","p":1`) -- and `min_prob` (0.05) then dropped
   them from the order, leaving the stairs alone. Later a sofa made the room with the sink a
   weak "living room" and the type prior ruled it out for the toilet beside the sink.
4. The peek's approach bound was sized from the frontier inventory's geodesic (a Dijkstra on
   the cost grid at the *body* radius, 3.5 m), but A* plans at the *preferred* clearance and
   relaxes only when nothing else gets through, so its route went 12.5 m round a squeeze. The
   RPT* instance used the same optimistic distances.

**Fix / workaround:** (1) `room_watershed`: a detected door's cut is snapped to the nearest
*choke* within 0.9 m -- the medial-axis cell of least clearance whose disk parts the skeleton
into two substantial pieces -- and sized to the passage; basins on different sides of the
carved mask never merge, basins on one side still may; plain disk when no choke is in reach.
Checked on all 15 recorded floors: fixed the one, moved the rest by at most a room. (2) The
nearest room floor within the footprint plus 0.6 m is the object's room. (3) An **uncertainty
floor** (`unexplored_floor`, 0.25): a never-entered unknown room or an opening with nothing
glimpsed is never read below it, whatever the model wrote; `entered=no` is in the prompt; the
example no longer writes anything off for its size; a weak (one-kind) label never excludes; a
room holding a *home object* of the target (a sink, for a toilet) is never excluded. (4) The
approach bound follows the route the planner actually adopted; the inventory and the RPT*
instance measure distances at the preferred standoff where it reaches, body radius elsewhere.

**Don't:** Do not "fix" (1) by shrinking the disk: at 0.45 m the misplaced door cut nothing
and R7 merged into the bedroom; at 0.5 m and above it severed the bedroom. No disk radius
works when the landmark is 0.6 m off; the cut has to move. Do not fix (3) by raising
`min_prob` or adding a "peek before you descend" rule: the user's model is that the LLM's
probability expresses it, and the floor keeps the unexplored nodes in RPT*'s hands, where
distance decides. Do not read `distance_m` in a recording as the route length: until this
change it was the body-radius geodesic.

---

## 2026-10-01 — a peek route across a stair head becomes a coverage-veto rotation loop

**Symptom:** Sofa same-spawn run: a visible path near steps 297–303, then stationary
left/right rotations. The agent never changed height or started a stair traversal.

**Root cause:** At 293, `DoorwayPeek` replaced the room-7 transit with an interior
peek route through the seen stair head. Steps 297/299/301/302 really emitted
MOVE_FORWARD. At 303 the follower corrected a 17.19-degree heading error; at 304
it requested forward with -12.81 degrees error (inside the 15-degree deadband),
but `FloorDepartureGuard` vetoed it to TURN_LEFT because pending room scans still
blocked departure. The guard cancelled the peek and cleared the path, but the
supervisor kept room 7: subsequent `transit/7` replans hit the same veto. This was
not the committed stair controller oscillating: `transition` stayed null.

**Fix / workaround:** One peek attempt per floor-local room identity, consumed on
start without refund on cancellation; scanned, initially classified (even weak),
and previously peeked rooms are exempt from selection AND floor-exit requirements.
The exemption follows substantial spatial overlap through renumber/split/merge
without asserting that every child was scanned. Seen floor-local connector cells
are excluded from peek goals and a peek-only A* map; active stair/atlas transitions
preempt peeks. A gated room transit is released rather than replanned indefinitely.
The generic converter retains its half-turn band and gains optional path-only
hysteresis (ObjectNav 1 degree), never overriding collision recovery or safety vetoes.

**Validation:** 1,206 core/runtime tests passed. CPU-only native-NavMesh Ranchester
probe with production hysteresis: ascent 37 actions, descent 38, both zero blocked
steps, recoveries or retreats. No new end-to-end benchmark was run.

**Don't:** Do not call a planar `doorway_peek_approach` a committed stair descent,
or weaken obstacle checks to fix a movement veto. Do not leave an exempt/attempted
room pending in the floor gate: that demands a second peek the one-time rule forbids.

---

## 2026-10-01 — target closing can freeze search correctly and still steer away from the bbox

**Symptom:** `runs/toilet_closing_verify_20261001`, Ranchester/000000, target toilet:
takeover 169, confirmation 170, four forward steps, then target lost at 203. No
LOOK_DOWN or STOP. The original video is complete through all 203 emitted actions.

**Root cause:** The global latch worked (LLM/classifier/RPT counters never changed),
but a centered bbox does not mean the A* route heading is centered. At steps 176/178
the command was a follow path with bbox error inside the yaw dead band; the converter
turned right toward its route, away from the target. The relative alternating
reacquisition turns subsequently oscillated outside the last successful heading.
Closest observed target range was 1.76 m, so the 1.2 m LOOK_DOWN trigger was never reached.
Separately, `TargetClosing.fail()` raises `ObjNavInternalError`, which `agent_contract`
explicitly treats as infrastructure failure even with `on_agent_error=record`.
Thus ordinary failed-episode scoring/completion notification did not run.

**Fix / workaround:** Unresolved navigation issues; this request authorized exactly
one episode, not tuning/retries. Original traces are retained and a separate,
clearly labeled post-hoc failure report was produced. Follow-up needs route/bbox
heading arbitration, reacquisition referenced to a known target bearing, and a
recordable method-failure contract. A runner-forced STOP must not be reported as
a successful target-distance STOP.

**Follow-up resolution (persistent-lock request, same date):** `target_path.py` now
retains the selected NavMesh standoff and reuses collision-qualified A* while the
target is occluded; bbox yaw no longer competes with path turns. Refinements over
0.15 m reproject/replan, smaller updates leave the goal stable. Only terminal
inspection at 1.0 m stops translation and requires a fresh bbox/depth confirmation.
Pitch-dependent projection moved a synthetic target by 2 mm at the exact radius;
a 0.05 m confirmation tolerance avoids a fully-satisfied hold deadlock without
allowing STOP on remembered range. Expected failure is now `RuntimeError`, recordable
by the harness, not `ObjNavInternalError`.

The single authorized replay in `runs/toilet_persistent_lock_20261001` kept its path
through occlusion at steps 177/178/180, including MOVE_FORWARD at 177, LOOK_DOWN at
183 and actual policy STOP at 185 (0.874 m range, no agent error). Global counters
stayed frozen. SR/SPL still zero and DTG 10.8924 m because the scored goal is on the
reference floor downstairs; do not relabel that result as benchmark success.

**Don't:** Do not infer live closing success from the obstacle-free synthetic rollout.
Do not equate bbox range with evaluator DTG: this multistory manifest only scores
reference-floor targets, whereas the detection here was upstairs. Do not backfill
an aborted run's empty benchmark ledger as if normal scoring completed.

---

## 2026-09-30 — "Ground-truth stairs" meant a map of every staircase; it should have meant a perfect detector

**Symptom:** With `stair_source=ground_truth` the Pomaria recording (`campaign-14b-20260929/campaign/
frontier/Pomaria`, 000000, 500 actions) committed to the stairs at action 1 -- before the camera had
ever had them in frame -- then paced 460 actions against a landing wall halfway up. Both Ranchester
episodes committed to the only staircase within 16-43 actions and spent the rest of the episode
"traversing" without ever standing on a tread. The user's brief was the opposite: stairs are known when
they are SEEN, placed on the map where they are seen, and never assumed -- "not while I'm in the
bathroom".

**Root cause:** Four, layered:
1. **The policy was handed the building's connector list at reset** and made every connector a portal
   and an RPT* node on the first action. Knowledge of the stairs was a file, not a sighting.
2. **One XY cluster per stairwell.** Pomaria's stairwell serves three storeys; the connector ran from
   -2.66 m straight to +2.72 m, THROUGH the middle storey no connector touched. Its polyline was the
   navmesh's string-pulled shortest path -- hugging the inner wall of every flight -- and the traversal's
   "reached" test was a 0.35 m sphere, so at the first landing the cursor jumped from vertex 0 to 9 (the
   flight above passes 0.3 m over the flight below) and the route aimed through the landing wall.
3. **A blocked vertex was skipped.** On a staircase the polyline is the only way; skipping a vertex aimed
   at the vertex after it through the same wall (Ranchester smoke: three skips, retreat, twice).
4. **Anchors round a corner and hooks of 15 cm legs.** Where the floor meets the flight at an angle the
   storey anchor sat round the stair-head corner; the pathfinder repair inserted a hook of 0.15-0.25 m
   legs; the agent "reached" three of them at once from 0.3 m off and aimed at the fourth from beside the
   centreline -- into the wall of a 0.5 m opening (Hanson, Coffeen).

**Fix / workaround:** `methods/stair_sightings.py` -- a perfect detector: each connector's navmesh
surface sample is projected into the frame (`project_to_image`) and depth-tested against the depth
image; six visible samples make a sighting with a bounding box (drawn orange on the frame, carried in
`detections` with `source: ground_truth`). Only a SEEN connector becomes a portal (`portal_placed`,
with `seen_step` and the sightings' `footprint` the map draws), a node, or the explanation of a height
departure (`height_departure_ignored: connector N underfoot has never been seen`). `gibson/
stair_connectors.py` splits a stairwell at every storey (Pomaria: 2 connectors, 7.6 m each), walks the
CENTRELINE (one vertex per 0.2 m band, the component chain that touches band to band), places anchors on
the flight's own line, stride-checks every leg with `PathFinder.try_step` -- the simulator's own
`move_forward` -- repairs through the pathfinder only where a stride fails, merges legs under 0.3 m, and
finds a verified flat EXIT point off each end. `ground_truth_traversal.py` never skips: a blocked tight
aim sends the agent back to the last vertex it reached; the route handed to the converter ends at the
next bend (corner walked TO, not cut by the 0.5 m lookahead); a vertex reached from off to one side is
led back THROUGH (`VIA_M`); `REACHED_XY_M` is 0.26 -- just over the converter's 0.25 m arrival, never
wider. `tests/rollout_stairs_navmesh.py` climbs every connector of every scene both ways on the real
navmesh with the real converter: 29/29 traversable climbs complete, 25 of them with zero blocked steps
(Pomaria 4/4 in ~40 actions). In the real Pomaria run the flight was seen at action 7, committed at 9,
climbed 25-69 with zero blocked steps and confirmed at 75.

**How to see it next time:** `building.events` now reads as a story: `stairs_seen` (bbox, distance,
`traversable`), `portal_placed` (`seen_step`, `footprint_points`), `traversal_recovery` (`back_to`),
`way_back_held`. `rollout_stairs_navmesh.py --scene X --verbose` reproduces a climb headlessly in seconds;
run it before any Habitat campaign that touches the stairs.

**Don't:** Don't widen `REACHED_XY_M` to make vertices easier to reach -- the agent then leaves them from
that far off, which is exactly the drift that hit the wall. Don't hand the converter the whole polyline
past a bend. Don't let the observed map's A* replace the stride check: the navmesh IS the collision
model. Don't treat an `unwalkable` leg as a tuning problem before checking `island_radius` at its ends.

---

## 2026-09-30 — Six "unwalkable" flights were navmesh islands, and two actions after arriving the agent went back down

**Symptom:** After the centreline rewrite, `rollout_stairs_navmesh.py` still failed Hanson c0, Leonardo
c0, Marstons c1/c2 and Shelbyville c0/c1 (12 blocked steps, retreat) although their treads were the
middle of the flight. And in the first real Pomaria run with sightings, the agent confirmed the middle
storey at action 75 and was descending the same flight at action 78.

**Root cause:** (1) `PathFinder.island_radius` differs across the bad leg and `find_path` has no route:
those storeys lie on DIFFERENT navmesh islands (0/40 random floor-to-floor paths in Hanson, Leonardo,
Marstons, Shelbyville). `try_step` -- what Habitat applies to `move_forward` -- never crosses, so no agent
can; they are mesh artifacts, not stairs. (2) On the new storey the map was two frames old: no rooms, no
frontier; the exploration fallback read the floor as exhausted and its last resort -- the stairs -- had
one candidate, whose entry the agent stood at (within the 0.65 m that starts a climb outright).

**Fix / workaround:** A leg with no navmesh path between its ends marks the connector `traversable:
False` (`walkability.disconnected`); it is still SEEN (bounding box, `stairs_seen` with `traversable:
false`) but never a portal. The probe reports such connectors as SKIP, not FAIL. The staircase the agent
arrived by is withheld from BOTH the loop's node list and the fallback rule for
`MultiFloorParams.arrival_grace_actions` (40) after the arrival (`MultiFloorSearch.way_back_held`,
`way_back_held` event): the second real Pomaria run showed the 14B node oracle -- told `arrived_by`,
`arrived 0 s ago`, one tentative room -- choosing "stairs down" at 0.98 on the arrival action three times
(78, 270, 493), each after a flawless climb, so 234 of 500 actions were spent on the stairs. A storey is
looked at for the length of a climb before "not here" means anything; any OTHER staircase is offered at
once, and walking onto the stairs by accident still completes the climb.

**Don't:** Don't try to repair an island-split leg with a longer `path_between` or a wider snap -- there
is no path. Don't make the grace a cooldown on the portal itself: `cooldown_until` is the failed-approach
deferral and the arrival's own bookkeeping reads it. Don't make it long enough to trap the agent on a
tiny storey either -- the fallback holds the way back too, and a floor that is genuinely exhausted at
action 20 waits out the rest of the grace on `fallback_hold`; 40 is about one climb.

---

## 2026-09-29 — A per-action "relabel on new evidence" trigger asked the room LLM 22 times in 60 actions

**Symptom:** The first clue-driven room relabel (re-ask the classifier for the room in force the
moment its evidence changes) looked right in every unit test, and the headless trace probe showed
identical navigation decisions -- and `label queries 22` in 60 actions, one classifier call every
three actions. On the CPU service (1-2 s per call) that is half a second per action, silently.

**Root cause:** Two multipliers, neither visible in a test that installs rooms by hand:
1. The evidence signature is count-sensitive by design (`("bed", 3)` is not `("bed", 2)`: it once
   told a bedroom from a living room), and the landmark map opens a NEW landmark whenever a
   re-observed object projects more than the dedupe radius from its last centroid -- so the
   count, and the signature, changed on most actions.
2. The watershed re-partitions the floor as the map grows, and a re-partition clears every
   per-room label cache on purpose (a cached label must not follow a pid onto a different
   region). Every clear made every room "pending" again, and the classifier's own cache was
   bypassed with `refresh=True` on every changed signature.

**Fix / workaround:** The mid-visit clue is a new KIND of object, not a new count
(`RevisableRoomLabels.pending`); a verdict for a set of kinds is kept and reused across
re-partitions (`_kind_cache`), and a changed signature the classifier has already judged reads
`RoomTypeClassifier.cached()` instead of buying it again -- the label is a function of the
evidence, not of the region. Counts are still re-asked, at loop points, as before. 5 calls in the
same 60 actions, identical decisions.

**How to see it next time:** The trace probe prints `label queries`, `llm queries` and
`affinity queries`; run it before and after any change that adds a model call to a per-action
path, and read the counts, not just the decisions. `telemetry.latencies["room_relabel"]` carries
the per-call cost in a real run.

**Don't:** Don't judge an LLM trigger by unit tests with a fake model -- the fake answers in
microseconds and the tests never re-partition. Don't clear the classifier's evidence→label
cache with the per-room caches: the per-room cache is about WHICH region a label belongs to,
the classifier's is about WHAT the evidence means, and only the first goes stale on a split.

---

## 2026-09-28 — Ground-truth stair RETREAT ate 60 % of the episode because "done" was gated on the atlas settling

**Symptom:** Two recorded Ranchester episodes on 299c20a1
(`runs/fallback_gtstairs_yolo_20260928T130941Z`, YOLO-only, seed 17): the room-search loop
ran cleanly for ~140 actions, the ground-truth floor decision picked the real staircase, the
agent descended 0.8 m — and then spent the remaining 316 / 293 actions in `RETREAT`,
shuffling in a 0.2 × 0.3 m box at the top of the stairs, camera pinned 60° down, never
re-entering SEARCH. Final DTG 7.2 / 7.6 m, no success. The previous commit (observed stairs)
walked the same staircase down at action 324–369 of the same episode.

**Root cause:** Two things, in order of impact.
1. `GroundTruthTraversal.plan` expires the retreat budget (`retreat_actions = 80`) by setting
   `completion_reason = ground_truth_retreat_budget_exhausted` and `arrival_allowed = True` —
   but nothing *ends* the transition. `MultiFloorSearch.observe` only abandons when
   `not atlas.in_transition`, and `FloorAtlas._settled()` needs 3 samples with height range
   ≤ 0.12 m **and** net translation ≥ 0.35 m. An agent stuck on the top treads (z wobble
   0.18 m, no net travel) never settles, so `in_transition` stays True forever and the
   traversal keeps issuing a 0.43 m follow command it can't complete.
2. The connector polyline from `stair_connectors` (7 points, top → bottom, 5.5 m) ends on the
   intermediate landing at z 1.84, 1.8 m above the destination (`best_rise_m` 0.80 of 2.59).
   `cursor` hit the last vertex with `at_height` false, forward was blocked 4× on the
   landing → `retreat_started reason=blocked`. Same spot in both episodes, so it is the
   geometry, not luck: either the navmesh shortest path between the two anchors is being
   truncated to the first flight, or the follower cuts the U-turn.

**Fix / workaround:** (1) give retreat a hard exit: when the budget is spent, call
`_abandon()` (or hand the action to the exploration fallback's relocation rung) and let the
atlas re-anchor by height alone, without waiting for `_settled()`. (2) Check
`scene_structure_from_pathfinder` on Ranchester: compare the connector polyline against
`pathfinder.find_path(top, bottom)` and make sure the full path, including the landing turn,
is what the policy receives. Until both are in, `stair_source: "observed"` is the better
default on this building.

**Don't:** don't read `idle_steps = 0` and `exploration_fallback.failures = 0` as "the
episode went well" — the retreat shuffle is forward/turn actions, so it is invisible to both
counters. Look at the `traversal_state` column of `trajectory.csv` or the timeline tool.

**Status (2026-09-30, remote line):** both root causes are addressed in e217f8f1. A spent
retreat within `floor_match_m` of a measured storey now confirms that storey through
`FloorAtlas.settle` and the coordinator abandons the transition. The connector is
re-extracted along the flight centreline, the follower never skips a vertex, and
`tests/rollout_stairs_navmesh.py` walks every connector both ways. Still open: a
retreat spent more than `floor_match_m` from every known storey (stuck on the treads)
still waits for the plateau test. The office fix for this (`FloorAtlas.cancel_transition`,
commit 4adf14c0, never pushed) was not carried over. Under the departure gate it would
restart the retreat in a cycle.

---

## 2026-09-22 — ObjectNav spent half its actions spinning, and the step log said why only once decoded

**Symptom:** In the Ranchester Gibson recording (`runs/grounded_vlm_gibson_3x3_20260922T084950Z/
.../e7c4f2ad5402`, 472 actions, `toilet` not found) the robot turned in place for ~50 actions
after finishing a room, tilted the camera down and up every ~24 actions while following a
path, then turned 120° back, and repeatedly planned a 7-9 m route only to replace it on the
next action. Video-wise it looked like four unrelated bugs.

**Root cause:** Four code paths, each individually "correct" for the 1 Hz aircraft it was
written for, all paid in discrete actions:
1. `ObjectSearchSupervisor.SEARCH` only ended on a 30 s stall / 90 s budget / a *stale*
   cluster count reaching zero, while the policy's own in-room goal generator was already
   empty. Every empty action became the headless agent's idle `TURN_LEFT` — 52 in a row once.
2. `MultiFloorSearch.plan` marked a stair inspection "due" on every action after
   `floor_search_actions` (140), so one fired at every 24-action cooldown in every new 1 m
   cell, mid-route, and the follower then spent four turns undoing the sweep's 120°.
3. `in_room_frontier_goals` ordered clusters by size only, with no reachability check: the
   robot reversed to a big cluster behind it (step 297), and three goals in the first thirty
   actions had no known-free path — one failed A* and one idle turn each.
4. `Release` fell through to a floor-wide frontier on the release action; `SELECT` replaced
   that route one action later. And `CommittedRoute.adopt` overwrote the clear reason with
   `new_goal_or_invalid_route`, so the log could not distinguish "new goal" from "replanned
   around an obstacle".

**Fix / workaround:** Caller-driven supervisor exits (`frontier_exhausted`, `route_failed`);
`FrontierSweep` with one bounded look-around per swept room and release-and-reselect on the
same action; `frontier_ranking.ranked_frontier_goals` (geodesic + facing, unreachable dropped,
several goals tried per action); inspection gated on "no committed route" plus a named
trigger and a 100-action unprompted cadence; `route_replaced` in the route info.

**How to see it next time:** `steps.jsonl` already carries everything — bucket each row by
`decision.info.status` (`arrived` with no waypoints = idle turn), `command.info.kind`
(`stair_inspection`, `room_scan`, `transit/N`, `frontier`) and `command.info.route`
(`new_goal_or_invalid_route` = a route was adopted; `route_replaced` now says why). A
twenty-line script over the log attributes every action in seconds; do that before watching
the video frame by frame.

**Don't:** Don't tune `search_timeout_s`/`frontier_stall_s` down to "fix" the spin — they are
backstops for a room whose count never clears, and the real signal (the goal generator is
empty) was always available on the same tick. Don't remove the room look-around entirely
either: the detector has to see the room from inside; `room_scan_turns=0` is a setting, not
the default. And don't issue two edits to the same file in one parallel tool batch — only
one of them lands, silently (cost an hour of "why is my edit gone" today).

---

## 2026-09-28 — Every Ranchester episode died on "Room oracle failed", and the model was never the problem

**Symptom:** Three Ranchester attempts across two campaigns (472, 281 and 18 actions) all ended
with `ObjNavInternalError: Room oracle failed; refusing a uniform fallback run`. The natural
reading -- qwen2.5:3b cannot produce the JSON -- was wrong: the recorded oracle rounds before
each abort were all `source=llm`.

**Root cause:** Latency. The frozen campaign ran Ollama CPU-only (`CUDA_VISIBLE_DEVICES=-1`)
with `LLM_TIMEOUT_S=30`. Measured against that service, qwen2.5:3b needs ~2.2 s per room in the
prompt: 6 rooms 14 s, 9 rooms 22 s, 12 rooms 29-44 s, 13-14 rooms 45-90 s. The oracle is asked
over EVERY room (only the solver caps at `max_rooms`), so the first loop point after the map
reached 13-14 rooms timed out, the base call fell to uniform, the one repair call timed out too,
and the runtime -- correctly -- refused to fly on a uniform prior. All three recordings died at
the moment the room count crossed that line.

**Fix / workaround:** `LLM_TIMEOUT_S=120` (a documented env setting; recorded in the run's
`preflight.json` under `method.llm`). With it both Ranchester episodes completed: 10 oracle
rounds, none failed, median 34 s, max 48 s. The loop's LLM-at-loop-points cadence and the
oracle's prompt-keyed reply reuse keep the count of calls small; they do not make a call fast.
The system Ollama on 11434 answers the same prompts in 2-5 s with llama3 -- but it runs on the
GPU, which this project's policy reserves for the detector and the renderer, so it was not used.

**How to see it next time:** Probe before launching: `RepairingSearchOracle(LLMClient(cfg))
.probabilities("toilet", rooms)` at 6/12/14 rooms against the real service, and read the wall
time, not just `source`. If the 14-room call is within a factor of two of the timeout, the
episode will die the moment the building has 14 rooms.

**Don't:** Don't lower `max_rooms` to shorten the prompt -- the oracle sees every room anyway.
Don't switch the room-language service onto the GPU to make it fast without saying so in the
run's provenance; the frozen campaigns are CPU-LLM by design.

---

## 2026-09-28 — 31 idle turns at a transit's route end: arrived by the converter, "not arrived" by the loop

**Symptom:** In the first recorded room-search-loop episode (`runs/room_search_loop_ranchester_
20260928T091752Z`, 000000), steps 41-71 are 31 consecutive `TURN_LEFT` with
`decision.info.status=arrived` and a committed `transit/0` route: the converter said the route
was finished, and the agent turned in place for 31 actions until the room was released as
`unreachable`.

**Root cause:** The transit's entry frontier was re-aimed mid-route (the map re-segmented from
3 to 7 rooms while the agent walked), the agent reached the new route end, and there the loop's
arrival test -- the agent's cell INSIDE the room's mask -- was false: the spot now belonged to a
neighbouring room, and even a correct entry frontier sits on the boundary that the watershed's
eroded mask stops short of. The supervisor's own distance test still pointed at the ORIGINAL
entry 2.4 m away. Neither could fire; `_navigate` kept reusing the committed route as
`committed_safe_route`; the converter, at its end, emitted idle turns; and only
`CommittedRoute`'s 30-action no-progress clock ended it -- as `unreachable`, which is also the
wrong verdict for a room the agent was standing next to.

**Fix / workaround:** `RoomSearchLoop._transit` now (a) counts arrival when the route has ended
and the agent is within the sweep's `mask_slack_m` of the room (the boundary case), and (b)
when the route has ended somewhere the room is not, retires that entry, re-aims once at the
room's frontier nearest NOW, and releases the room as unreachable if it has none -- all on the
same action (`entry_lost` stat/event).

**How to see it next time:** `decision.info.status == "arrived"` with no `command.waypoints` is
an idle turn; bucket them by `command.info.kind`. A run of them under one `transit/N` is this
bug or a relative of it. The confirmation run of the same episode has zero.

**Don't:** Don't make the supervisor's `arrival_tol_m` larger to cover this -- the supervisor's
goal was the stale entry, so no tolerance short of the room's diameter would have fired, and a
large one declares arrival at rooms never entered. Don't shorten `progress_timeout_steps`
either: it is the last line, not the first.

---

## 2026-09-28 — The room-search loop released a room with three clusters showing, on every action

**Symptom:** The first headless trace of the room-search loop (`tests/trace_room_search_loop.py`,
one room, 12 actions) showed the same room released `exhausted` on five of them, a room LLM
call for each, and the in-room sweep never lasting more than one action while the scene graph
reported `frontier_clusters=3` for that room the whole time. The supervisor's own tests and the
loop's own tests were green.

**Root cause:** Three separate mechanisms, none of them in the loop's control flow:
1. **The frontier goal was the cluster centroid.** A frontier is usually an arc (the edge of
   the depth range, bowed around the robot), and the centroid of an arc lies inside the floor
   already seen. `FrontierSweep._current_goal` keeps a goal only while unknown space remains
   within 1 m of it, so a freshly adopted goal was "resolved" on the very next action. Its
   replacement -- the next arc's centroid, 0.25 m further on -- fell inside the 0.4 m
   "recently retired" radius and was refused. No admissible goal, so `frontier_exhausted`.
2. **The count and the sweep disagreed about which cells belong to a room.**
   `count_frontier_clusters` credits a cluster to a room by majority vote over the labels at
   its cells; the sweep required the frontier cell itself to lie *inside* the watershed mask,
   which is eroded by `min_clearance_m`, so boundary cells sit just outside it. Three
   clusters counted, zero offered.
3. **The supervisor's every-room-cooling escape hatch.** With one room, an `EXHAUSTED`
   release cooled it, the hatch immediately repeated it ("a search that stops is worse than
   one that repeats"), the loop re-reasoned (a loop point) and re-entered it. Correct for an
   aircraft with nothing else to do; wrong for a loop that has a floor-wide frontier to fall
   back to.

**Fix / workaround:** `room_costs.frontier_clusters` stands a cluster for by its *member*
cell nearest the centroid (on the boundary, informative by construction); the in-room sweep,
the room entry point and the transit re-aim all draw their goals from
`frontier_goals_by_room` (the count's own majority vote) through one admissibility filter;
`ObjectSearchParams.repeat_verdicts` lets the loop keep `EXHAUSTED`/`MAPPED` rooms out of the
hatch. Separately, the N-step budget check sat *inside* `_local`, after a plain supervisor
tick, so a budget release spent a round before its flag was raised and the same-action chain
overran the 3-round guard, dropping a pending `frontier_exhausted` on the floor and planning
a throwaway route; the flag is now raised before the first tick, the guard is 5 and an overrun
is counted and logged.

**How to see it next time:** Run the trace probe before trusting the unit tests -- every
mechanism above was invisible to tests that install rooms by hand, because the disagreement
was between the *real* segmentation, the *real* frontier geometry and the sweep. Read
`room_search_loop.events` (release verdicts with `local_steps`) against `graph.facts`: a
release with `local_steps < 3` and a non-zero cluster count is this bug or a relative of it.

**Don't:** Don't widen `visited_radius_m` down or `informative_radius_m` up to make the sweep
survive -- both hide the geometric mismatch instead of removing it. Don't put the room LLM
back on an action timer to "smooth" the calls; the calls were a symptom.

---

## 2026-09-28 — 132 actions "acquiring tread support" on a bathroom floor and a stair head: depth-found stairs are not stairs

**Symptom:** In the confirmation recording of the room-search loop
(`runs/room_search_loop_ranchester_20260928T093541Z_transitfix`, 000000, 500 actions,
`toilet` not found) the phase timeline shows `TRAVERSE` with `reason=acquire connected tread
support` twice: steps 187-306 (120 actions) at the top of the real staircase, ending in a
`RETREAT`, and steps 488-499 (12 actions, to the step cap) after the agent walked into a
bathroom on the ground floor. Both look on video like the camera spinning in place with the
pitch pinned down. The episode ended 9 m from any stairs, "traversing".

**Root cause:** Two, one per incident, both in the observed-only stair discovery:
1. **A height departure of `stable_height_m` (12 cm) started a traversal.** `MultiFloorSearch
   .prepare_observation` began a `StairTraversal` whenever `|z - floor| > 0.12` with no
   committed portal -- the agent at the stair head dipped 12 cm onto the first tread, a
   traversal began with a one-point "portal" at the agent's own feet, and `StairTraversal
   ._surface_route` then needed depth-measured *connected* tread support to plan on. From the
   top of a flight looking down, the treads' risers occlude their surfaces, the support graph
   is empty, and the transition's answer to "no path" is a `hold(final_yaw=+30°)`: one turn per
   action, twelve turns per `failure`, four failures before retreat. Checked against the navmesh
   afterwards: the agent was 0.24 m from the real stair polyline -- these WERE the stairs, and
   the observed mode still could not walk them from where it stood.
2. **A 13 cm raised bathroom floor is indistinguishable from a first tread by height alone.**
   The same rule fired at step 488 on a threshold 9.3 m from the staircase. `stair_min_rise_m`
   (0.40) guards the *proposal* path (`StairTerrain._candidate_paths`), not the unplanned
   departure path, which has no rise check at all.

The YOLO `stairs`/`staircase` labels played no part in either -- they are context-only and
were already correct in not creating portals. The loop's own fallback for "nothing to do" was a
plain `hold()`, so wherever the transition did *not* own the action, the headless agent's idle
`TURN_LEFT` produced the same spin from a different cause.

**Fix / workaround:** `MultiFloorParams.stair_source = "ground_truth"` (default). The
evaluator reads each scene's navmesh once (`gibson/stair_connectors.py`) and hands the policy
the storeys and stair connectors through the episode metadata; a transition begins only ON a
known connector (within 0.75 m of its polyline, beyond `departure_m` = 0.45 m), follows the
connector's own polyline with a FOLLOW on every action, and never scans in place. The bathroom
step never reaches 0.45 m; the stair-head wobble is on the polyline and now completes the climb.
Separately, every "nothing to do" and every decision exception goes through
`exploration_fallback.py` (nearest frontier anywhere → stairs → retired frontier → relocation),
so an empty hold is issued only when the agent is off the passable map.

**How to see it next time:** `timeline_room_search_loop.py` on `steps.jsonl`, then look for
long runs of `TURN_*` under one `kind` with no `waypoints` in the command. `reason=acquire
connected tread support` is the observed stair mode; `kind=fallback_hold` is the new fallback's
last resort and should be rare -- if it is not, the agent is boxed in, not the planner.
`building.events` now carries `height_departure_ignored` (with the height) and `floor_decision`
(with every candidate's verdict), so "why did/didn't it take the stairs" reads straight off the
record.

**Don't:** Don't raise `stable_height_m` to stop the false starts -- the atlas uses it to
decide when the agent is *off* a floor, and a larger value lets real stair treads read as level.
Don't put a rise check on the unplanned path and call the observed mode fixed: it would still
have spun at the real stair head, where the failure was tread support, not detection. Don't
disable `multifloor` to avoid it all: the target floor in the multi-story protocol is by
construction not the start floor.

---

## 2026-09-22 — Grounding DINO shared-head loading can look clean but substitute tensors

**Symptom:** Transformers 5.17.0 reported no missing/unexpected weights for the
official Grounding DINO Base snapshot, but direct comparison found a saved
decoder bias had been substituted. Reusing LLMDet's independent-head alias fix
then left thirty decoder parameters on the meta device, failing on `.to(cpu)`.

**Root cause:** This snapshot saves only `model.decoder.bbox_embed.0` (six
tensors), with shared decoder heads. Upstream's chained alias map can use the
unsaved `bbox_embed.0` as canonical. LLMDet's distinct-head layout is different;
its loader correction is not transferable unchanged.

**Fix / workaround:** A local Grounding DINO subclass maps all decoder/output
aliases directly to the saved head. Validate that exact checkpoint layout,
compare the six saved tensors with loaded values, and reject any remaining meta
parameters/buffers before serving. The upstream architecture/forward and model
weights are unchanged. Real CPU inference and HTTP mode smokes passed.

**Don't:** Do not trust an empty loading-info report alone, initialize missing
learned parameters randomly, call `to_empty()` to conceal the defect, or apply
the independent-head LLMDet correction to a shared-head model.

---

## 2026-07-29 — new detector container: numpy/CLIP/TensorRT-version traps, in sequence

**Symptom:** Building `docker/Dockerfile.detector` (torch + ultralytics on top of the
`perception` layer, for the YOLO-World detector) hit three separate, unrelated failures, each
only visible after fixing the one before it: (1) `pip install torch` timed out mid-download
with a default 120s timeout, (2) after fixing that, `import pycuda.autoinit` failed with
`AttributeError: _ARRAY_API not found`, (3) after fixing that, the detector node failed with
`RuntimeError: Failed to deserialize one of the engines.`

**Root cause:** Three independent issues, not one:
1. torch's CUDA 13 wheels are large; pip's default read timeout can trip mid-stream well
   before anything is actually wrong (same as `Dockerfile.perception`'s own tensorrt install
   already documents) — needs a much longer `--timeout`, not just `--retries`.
2. `ultralytics` declares `numpy` unconstrained, so pip resolved `numpy==2.2.6` — silently
   overriding `perception`'s deliberate `numpy==1.26.4` pin (there specifically so `pycuda`,
   compiled against numpy 1.x's C-API, works). numpy 2.x breaks pycuda outright, not just a
   warning.
3. `git+https://github.com/ultralytics/CLIP.git` "succeeded" but built a wheel named literally
   `UNKNOWN-0.0.0` — the base image's stock pip 22.0.2/setuptools 59.6.0 mis-resolve that
   repo's `pyproject.toml` metadata, so no `clip` module was actually importable despite a
   clean-looking install log. Separately, once TensorRT engines finally loaded, they were
   built on the HOST (TensorRT 10.16.1.11) and failed to deserialize in the container
   (TensorRT 10.15.1.29) — `Dockerfile.perception` already documents that engines are locked
   to the exact TRT build that produced them, not just a major-version match.

**Fix / workaround:** In order: raised the pip timeout to 600s/8 retries for the torch
install; added an explicit `pip3 install numpy==1.26.4` as the LAST install step (after
ultralytics/CLIP, so nothing downstream re-upgrades it) — confirmed `cv2` (a *different*
package, pip's own `opencv-python`, not `perception`'s apt `python3-opencv`) tolerates the
downgrade fine despite pip flagging it as a dependency conflict; upgraded
`pip`/`setuptools`/`wheel` before the CLIP install so it resolves the real package name; and
rebuilt just the TensorRT engine (not the ONNX export — that's portable) inside the container
that will actually run it.

**Don't:** Don't trust a "Successfully installed X" pip log line as proof X actually works —
check the real package name landed (not `UNKNOWN`) and that nothing else silently got
up/downgraded as a side effect. Don't assume a TensorRT engine built on one machine/container
works in another with a different TensorRT version, even the same GPU — rebuild the engine
(cheap, seconds) inside wherever it will actually run, and check `Dockerfile.perception`'s own
comments before re-debugging a version-mismatch class already documented there.

---

## 2026-07-29 — stale `ros2` CLI daemon kept showing zero R1 topics after switching networks/domains back and forth

**Symptom:** After going real-drone (`ROS_DOMAIN_ID=1`, CycloneDDS pointed at the real
network) and then switching back to Sphera (`ROS_DOMAIN_ID=9`, CycloneDDS reverted to
`172.16.17.10`), `ros2 topic list`/`ros2 topic echo` inside `it` showed ZERO `/R1/...`
topics, even though R1's own container logs showed a perfectly healthy boot (FCU
connected, position valid, no CDR/NIC errors) and both `it`'s and R1's cyclonedds
configs correctly agreed on domain 9 / `enp129s0`/`172.16.17.10`.

**Root cause:** `ros2` (Foxy) CLI commands are served by a long-lived background
`_ros2_daemon` process that caches graph/discovery state and does NOT restart or
rebind when `CYCLONEDDS_URI`/`ROS_DOMAIN_ID` change in a later shell — it keeps
running with whatever network state it had when it first started. A daemon that had
been running since earlier in the session (started during the real-drone/domain-1
work) kept serving stale results to every later `ros2 topic list` call, even ones that
correctly exported `ROS_DOMAIN_ID=9` — the CLI just asks the existing daemon, it
doesn't re-discover itself. `ros2 daemon stop` itself hung (never completed) rather
than fixing it.

**Fix / workaround:** `docker exec it kill -9 <daemon-pid>` (find via `ps aux | grep
daemon` inside `it`) — a fresh daemon spins up automatically on the next `ros2` CLI
call and immediately sees the correct, current graph.

**Don't:** Don't assume "both configs agree and the container's own logs look healthy"
rules out a networking/discovery problem — the `ros2` CLI's own daemon is a separate,
stateful layer that can lag behind a config change made after it started. Any time
`ROS_DOMAIN_ID`/`CYCLONEDDS_URI` changes mid-session (e.g. switching between Sphera and
a real drone), kill and let the daemon restart before trusting a "no topics" result.

---

## 2026-07-27 — duplicate method definition silently shadowed the real navigation output

**Symptom:** FALCON's `waypoint_follower` logged `nav=RUN done=False` continuously and
appeared to be actively navigating after a BEV click, but the drone never moved.
`cmd_vel_gate`'s "commands passed" counter was frozen at a fixed number instead of
climbing — meaning `/cmd_vel_raw` was never actually being published on any tick.

**Root cause:** `waypoint_follower_node.py` defined `_publish_twist_multi` TWICE in the
same class. The correct version (with `vz` altitude-hold support, yaw-pitch-bias,
`cmd_vy_sign`) came first; a much older, simpler version (no `vz` parameter at all) was
defined later in the file and silently overrode it — that's just how Python class bodies
work, no error at import time. Every tick called
`self._publish_twist_multi(cmd.vx, cmd.vy, cmd.wz, vz=self._alt_vz)`, which raised
`TypeError: _publish_twist_multi() got an unexpected keyword argument 'vz'` against the
live (second) definition — right before it would have reached `.publish()`. The
exception was swallowed by rospy's timer callback error handling (logged, not fatal),
so the node stayed "alive" and kept logging as if navigating.

**Fix / workaround:** Deleted the older, shadowing definition. Confirmed via
`grep -n "_publish_twist_multi" *.py` before assuming a single definition existed.

**Don't:** Don't trust a log line saying `nav=RUN` as proof that output is actually being
published — check the downstream gate/counter (`cmd_vel_gate`'s passed count) for real
movement, and grep for duplicate method names in a file before deep-diagnosing a "looks
alive but doesn't work" node.

---

## 2026-07-27 — half-applied coordinate handedness fix (yaw negated, position not)

**Symptom:** The drone's reported heading looked correct at low yaw, but flying it
(pure `forward` command) produced a real-world displacement whose Y sign didn't match
`tan(yaw)`'s sign — i.e. the map/localization built in the mirrored lateral direction
from the drone's actual physical motion.

**Root cause:** `rooster_ground_truth_localization.py` converts Sphera/Unreal telemetry
(left-handed, clockwise-positive yaw) to ROS (right-handed, counter-clockwise-positive).
A previous fix (documented inline, dated earlier) negated yaw to fix a "map built behind
the drone instead of in front" bug — but a full handedness conversion needs ONE LINEAR
AXIS negated too, and that half was never done. `position.y` was passed straight
through unflipped, leaving rotation and translation using inconsistent conventions.

**Fix / workaround:** Verified quantitatively before touching code: commanded a pure
`forward` move, recorded ground-truth `(dx, dy)` before/after, and compared
`dy/dx` against `tan(yaw)` from the reported orientation. The sign only matched after
mentally flipping `dy`, confirming which axis and which fix. Negated `position.y` in
`rooster_ground_truth_localization.py`.

**Don't:** Don't assume a rotation-only handedness fix (negating yaw alone) is complete —
check whether a corresponding linear axis needs the same treatment. And don't guess which
axis without a live before/after ground-truth measurement; a wrong guess flips it a
different, still-wrong way.

---

## 2026-07-27 — fixing the Y-axis sign broke the map/BEV bounds (expected, but easy to miss)

**Symptom:** Immediately after the Y-axis localization fix above, the drone appeared to
spawn "outside the BEV click map" and RViz's 3D voxel view showed no drone marker at all.

**Root cause:** `maps/sphera_jail.yaml` (`init_y`, `map_min_y`/`map_max_y`, `box_min_y`/
`box_max_y`, `vbox_min_y`/`vbox_max_y`) and the launch args documented in the
`fly-rooster-sphera` skill (`bev_ymin`/`bev_ymax`/`goal_y`) were all tuned against the
drone's OLD (unflipped) reported Y position. The drone's real, physical spawn point
never moved — but the sign fix above changed which number `/R1/localization` reports for
it (`+14.66` → `-14.66`), which took every Y bound out of range at once. The map yaml's
own comments already document that being out of bounds FATAL-aborts `exploration_node`
with an out-of-range voxel-array index.

**Fix / workaround:** Negated AND min/max-swapped every Y bound in both files (negating
a range reverses which end is the min and which is the max).

**Don't:** Don't change a world-frame axis sign convention in one place without auditing
every spatial config tuned against the old convention (map bounds, BEV click bounds,
default goal coordinates) — they're a matched pair, not independent.

---

## 2026-07-27 — twist-control adapter's stop-watchdog cancels any in-progress takeoff/land

**Symptom:** Sending `arm` then `takeoff` armed the drone, but the climb got cancelled
(`"Climb cancelled - holding z=..."`) within ~50-100ms every time, regardless of
throttle/altitude-hold tuning. Same thing happened trying to `land` while airborne.

**Root cause:** `rooster_twist_control_adapter.py` runs a 20Hz watchdog that publishes
`{"action": "stop"}` on `/R1/cmd_nav` whenever it hasn't seen a `/cmd_vel` message in the
last `cmd_timeout_sec` (default 0.4s) — reasonable as a "planner went silent, stop
moving" safety net. But `rooster_command_unit.py`'s `RoosterUnit.stop()` unconditionally
cancels ANY in-progress `takeoff`/`land` sequence (`busy_action in ("takeoff", "land")`),
with no distinction between "the planner's own no-op stop" and "the user wants to abort
a takeoff." Since FALCON only publishes `/cmd_vel` while actively driving toward a goal,
this adapter spams stop essentially constantly outside of active navigation — killing
any manual arm/takeoff/land test within one or two 50ms timer ticks.

**Fix / workaround:** Kill (or don't yet start) the twist-control adapter before any
manual arm/takeoff/land test; only start/restart it once the drone is confirmed hovering
and you're about to test click-to-fly.

**Don't:** Don't leave the twist-control adapter running "just in case" during a manual
flight test — it will silently sabotage takeoff/land and the failure looks like a
throttle/timing bug in `rooster_command_unit.py`, not a second process fighting it.

**Update 2026-07-30 — same root cause, a much more misleading second symptom:** left the
twist-control adapter running from an earlier FALCON click-to-fly test, then moved on to a
manual-flight session via the Tkinter `ui.py`. The drone took off and immediately flew hard
into a wall; it LOOKED exactly like a yaw/turn-direction bug (see the "BEV click turned the
wrong way" investigation earlier the same day) — plausible enough that real time went into
checking bearing math and body-frame conventions before the actual cause surfaced. A
command+pose logger (subscribing to `/R1/cmd_nav` and `/R1/localization`, no
publishing) showed the real story: continuous `{"action": "move", "axes": {"x": 599, "y":
0, "r": 0}}` commands at ~50ms intervals, present in **6672 of 6765** logged commands
(only 93 had `x==0`) — i.e. an almost-constant ~60%-forward push with zero steering,
starting before `arm` and continuing straight through takeoff. `ui.py` itself only ever
sends discrete named actions (`forward`/`turn_left`/etc., grepped — no `"move"` string
anywhere in it), so this wasn't the UI's own doing at all: `{"action": "move", "axes":
{...}}` is `rooster_twist_control_adapter.py`'s own publish format, and it was still alive
from earlier, faithfully translating FALCON's `waypoint_follower` (still cruising toward
its own leftover default goal) into a continuous forward push that fought every manual
input the whole session.

**Don't (extended):** A drone that "drifts/turns into a wall immediately on takeoff" is
NOT proof of a yaw-sign or turn-direction bug — check `ps aux | grep
rooster_twist_control_adapter` (or any other `/R1/cmd_nav` publisher) FIRST, before
re-deriving bearing/handedness math again. A cheap command+pose logger (subscribe-only,
never publish) that records every `cmd_nav` message alongside ground-truth pose settles
this class of question in seconds instead of live-flight-testing guesswork — keep using
one whenever "what actually got commanded" is in doubt.

**Update 2026-08-02 — third occurrence, this time looked like a "turning left instead of
right" planning/direction bug across THREE separate takeoffs.** After building a live
telemetry dashboard for click-to-fly debugging, the twist-control adapter was left running
(needed for that testing) and never stopped before the user switched to flying manually via
`ui.py`. All three flights showed the drone hovering nearly in place with yaw oscillating
50-100°+ repeatedly right after takeoff — this time the interference wasn't a constant
forward push (2026-07-30's symptom) but an alignment fight, because FALCON's
`waypoint_follower` always has a non-empty `self.goal` from the moment it launches
(`~goal_x`/`~goal_y` args are read at init, before any click ever arrives — see
`astar_planner_node.py`), so the adapter kept trying to turn the drone toward that static
default goal while the user tried to fly a different direction by hand. The command log
confirmed it instantly: 300 of 301 logged commands in each flight were `{"action": "move",
...}` (the adapter's format), only 1 was the actual `takeoff` action — zero named actions
(`forward`/`turn_left`/etc.) from the UI made it through uncontested.

**Don't (extended further):** Having a live dashboard that shows this process's up/down
status does NOT prevent this bug by itself — the status was accurate and visible the whole
time, but nothing prompted anyone to actually stop the process when the testing mode
switched from click-to-fly to manual flight. Whenever handing control back to a human pilot
after any autonomous-navigation testing, explicitly kill every `/R1/cmd_nav` publisher that
isn't the UI first — don't rely on remembering, and don't treat "the dashboard would show it
as running" as equivalent to "someone will notice and stop it."

---

## 2026-07-28 — camera rig/mount visible in its own FOV, fused as a permanent phantom wall

**Symptom:** FALCON's A* planner reported "boxed in - no A* route" almost immediately
after a BEV click, falling back to NavDP, which then searched/spun erratically — and
the resulting map looked noisy/speckled (a "V"-shaped or cauliflower-like blob rather
than clean walls).

**Root cause:** The bottom ~25% of every single RGB frame showed a near-constant
`~0.17-0.35m` depth reading, completely stable across consecutive frames regardless of
drone motion — confirmed by sampling `/tmp/rooster_depth/*.npy` directly with numpy.
That's the drone's own camera rig/mount, visible in its own field of view, not real
environment. `cam_min_depth` was `0.1` — below that artifact's range — so it passed
the near-depth filter and got fused into the map as a permanent phantom wall directly
in front of the drone on every single frame, regardless of where the drone actually
was or which way it was facing.

**Fix / workaround:** Raised `cam_min_depth` from `0.1` to `0.45` (comfortably above
the observed `~0.35m` ceiling of the artifact, similar order of margin to
`astar_planner`'s own `inflate_radius_m=0.4m`). Confirmed occupied-cell counts dropped
substantially in a comparable flight window after the fix (889 → 315).

**Don't:** Don't assume a "boxed in" planner result or a noisy map is necessarily a
planner/mapping-logic bug — check the raw depth data FIRST (a quick numpy stat check
across a handful of `.npy` frames caught this in minutes) before chasing coordinate
transforms or freeze-timing theories. The freeze-timing (turning) theory investigated
first was real but secondary — this was the dominant cause.

---

## 2026-07-28 — vbox/box bounds set exactly equal to map bounds crashed exploration_node

**Symptom:** Raised `box_max_z`/`vbox_max_z` from `1.8` to `4.0` (to stop RViz truncating
the room below its real ceiling) and set `map_max_z` to the same `4.0` — `exploration_node`
crashed within seconds of the next `falcon` restart: `voxel_mapping::ESDF::getDistance`
→ glog FATAL "Address out of range", from `BsplineOptimizer::calcDistanceCost` during
the exploration planner's own internal trajectory optimization.

**Root cause:** ESDF distance/gradient queries read neighboring cells near a queried
point. With `vbox_max_z` exactly equal to `map_max_z`, there's no room left inside the
map's own allocated grid for those neighbor reads near the top boundary — an
out-of-range access is inevitable, not just possible. The documented ordering rule
(`map ⊇ vbox ⊇ box`) technically allows equality, but "supports the ordering" and
"leaves a working margin" are not the same thing, and the code that enforces this
never checks for the margin.

**Fix / workaround:** Raised `map_max_z` to `5.0` instead of leaving it at `4.0`,
restoring a real 1.0m margin between `vbox_max_z` (4.0) and `map_max_z` (5.0) — smaller
than the original config's 2.2m gap, but comfortably more than the handful of 0.1m grid
cells any real stencil actually reads.

**Don't:** Don't set `vbox_max_z`/`box_max_z` equal to `map_max_z` (or `_min` bounds
equal to each other) even though the ordering comment technically permits it — always
leave real headroom. If this crash class recurs (`ESDF::getDistance`/`getDistanceAndGradient`
→ glog FATAL "Address out of range"), check the map/vbox/box margins first, not just
whether the ordering constraint holds.

---

## 2026-07-28 — RViz showed a completely empty scene after the Y-axis localization fix

**Symptom:** After negating `position.y` in `rooster_ground_truth_localization.py`
(see the handedness-fix lesson above), RViz's 3D view showed absolutely nothing — no
error, no warning, just an empty gray scene. The 2D BEV (matplotlib) window meanwhile
showed real, live, correctly-updating data (drone pose, A* planning), proving the
underlying pipeline was fine.

**Root cause:** `maps/sphera_jail.rviz`'s saved "Current View" camera had a hardcoded
`Focal Point: X: -54.75, Y: 14.66` — captured in an earlier session, before the Y-axis
sign fix. The drone's real, physical spawn point never moved, but the sign reported
for it flipped (`+14.66` → `-14.66`), so the saved camera was now pointed at the
mirror-image empty location where the room used to be under the old convention.

**Fix / workaround:** Updated the saved Focal Point to `Y: -14.66` to match. RViz must
be restarted (not just have the `.rviz` file edited) to pick up a changed saved view,
since it only reads that file at startup.

**Don't:** When fixing a world-frame axis sign, don't assume you've found every place
it's baked in once the code and config bounds are fixed — RViz's own saved viewport
state is invisible until you actually open it, and "no error" does not mean "nothing
is wrong." Check any saved camera/view files for the same axis too.

---

## 2026-07-28 — mapping_sync's rotation freeze got permanently stuck once it started working

**Symptom:** After adding `rooster_demo_mode_manager.py` (a Rooster-only demo-mode
arbiter that didn't exist before), the map stopped updating entirely — even with the
drone sitting disarmed on the floor. `mapping_sync`'s heartbeat showed `gate=FROZEN`
with the `frozen` counter climbing every single cycle, never recovering. Restarting
`mapping_sync` itself (re-arming its warm-up sequence, which force-fuses the first 2
pairs and calls `reset_freeze()`) only unstuck it for a few frames before it froze
again immediately.

**Root cause:** `waypoint_follower_node.py`'s own rotation supervisor was found to be
continuously, on every single tick, requesting `"turning"` mode — while in its normal
`RUNNING` navigation state, targeting the default startup goal, with the drone either
disarmed on the floor or being flown manually (bypassing FALCON's own `/cmd_vel`
entirely). Its "turn complete" condition apparently never resolves without real flight
dynamics to confirm against. Before `rooster_demo_mode_manager.py` existed, nothing
ever echoed this request onto `/R1/demo_mode`, so the freeze silently never engaged —
masking this bug entirely. Fixing the arbiter (correct behavior) exposed a real,
pre-existing bug in the rotation supervisor (incorrect behavior) that had never
mattered before.

**Fix / workaround:** No real fix yet. Set `freeze_on_turning_mode:=false` on
`mapping_sync` in `sphera_drone.launch`, reverting to the freeze-less behavior
everything had already been verified against. This gives up the turning-smear
protection the freeze exists for (confirmed real and needed — see the original
ring-artifact map from 2026-07-27) until the supervisor bug itself is found and fixed.

**Don't:** Don't assume "the demo mode value looks right when I check it manually" (it
did — `rostopic pub` confirmed `/R1/demo_mode` genuinely read `"fly_straight"`) means
the freeze will clear — the freeze-clearing logic in `depth_fusion_gate.py` re-evaluates
on every NEW mode message, so if something else keeps re-requesting `"turning"` a moment
later, a manual override is immediately overwritten. Check what's actually driving the
requests (`rosnode info /waypoint_follower`, or grep its own status log for `mode=`)
before assuming a one-off manual fix will stick.

**Update 2026-07-28 (later same day):** the actual supervisor bug above was found and
fixed (`MultiAxisCommand.yaw_engaged`, see `multi_axis_follower/types.py` +
`waypoint_follower_node.py:_supervisor_cmd_wz()`), and `freeze_on_turning_mode` was
re-enabled (`true`) in `sphera_drone.launch`. It recurred anyway during a pure manual
flight test (drone driven directly via `/R1/cmd_nav`, `waypoint_follower_node.py` never
consulted at all) — but this time the mechanism was different and worth telling apart:

- `mapping_sync`'s heartbeat showed `gate=FROZEN` with `frozen` climbing, exactly like
  before. But `sensor_gate`'s own heartbeat log showed `state=FUSING`,
  `mode_turning=False` the entire time — a red herring. `sensor_gate_node.py` and
  `mapping_sync_node_sphera.py` each hold their **own separate** `DepthFusionGate`
  instance (per `sensor_gate_node.py`'s own docstring: sensor_gate's copy "does not
  freeze" the authoritative one in mapping_sync). Checking sensor_gate's health proves
  nothing about mapping_sync's gate — always check `mapping_sync`'s own heartbeat.
- Publishing to `/sensor_gate/reset_mode_freeze` (the documented manual-recovery topic)
  did nothing, because it only resets sensor_gate's copy, not mapping_sync's.
- The real cause this time: `rooster_demo_mode_manager.py`'s mode publisher is
  `latch=True` *and* re-published on a 1 s timer from its own `self.current_mode`
  — so a single stale `"turning"` request from earlier (before the manual-flight test
  even started) stayed latched indefinitely with nothing to overwrite it, once the node
  that sent it was gone.

**Actual fix:** publish a fresh non-turning string straight to the *source*,
`/R1/demo_mode_request` (not `/R1/demo_mode` itself, and not
`/sensor_gate/reset_mode_freeze`) — e.g. `rostopic pub -1 /R1/demo_mode_request
std_msgs/String 'data: fly_straight'` (`fly_straight` is the manager's own
`~initial_mode` default). This updates the manager's internal `current_mode`, so the
1 s timer starts re-publishing the correct value instead of re-latching `"turning"`.
Confirmed: `mapping_sync`'s `gate` flipped to `FUSING` and `emit` resumed climbing
within one heartbeat cycle, `frozen` stopped growing.

## 2026-07-29 — Jetson's `agency_ws` isn't a git-tracked clone of this repo

**Symptom:** Wired a new `mission_control.py` service to run
`dir_watch_path_publisher.py` from `{JETSON_REPO}/sparx_agency/robots/common/` on the
Jetson — the file is committed in this repo (`647701b9`) and clearly present locally,
but `ls` on the Jetson showed the directory without it at all.

**Root cause:** `{JETSON_REPO}` (`/home/user/agency_ws`) is not kept in sync via
`git pull`. `git status`/`git log` there show an empty, commit-less `master` branch
with unrelated home-directory dotfiles as untracked — its `sparx_agency/` tree was
manually copied over at some point in the past and never touched since for files
outside whatever was actively being used. Any file committed after that copy simply
isn't there, regardless of how long ago it was committed here.

**Fix / workaround:** `scp`'d the one needed file directly to the matching path on the
Jetson. No general sync mechanism exists — don't assume one does.

**Don't:** Don't assume `{JETSON_REPO}` matches this repo just because paths/service
definitions reference it as if it does. Before wiring any new `mission_control.py`
Jetson-side service, verify the specific file(s) it needs actually exist there first
(`ssh ... ls <path>`), rather than debugging a confusing `FileNotFoundError` after the
fact.

---

## 2026-07-29 — testing `mission_control.py`'s orchestration logic without driving the UI

**Symptom:** Needed to verify `start_service`/`stop_service`/`get_all_states` actually
work for two new services, but no browser-automation tool was available to click
through the Streamlit UI, and the script has no `if __name__ == "__main__":` guard —
it's UI calls top-to-bottom, seemingly not meant to be imported as a plain module.

**Root cause / finding:** Streamlit tolerates being imported outside a real
`streamlit run` session. Every `st.*` call just logs "missing ScriptRunContext! This
warning can be ignored when running in bare mode." and no-ops, rather than raising —
this is documented Streamlit behavior ("bare mode"), not a hack specific to this file.

**Fix / workaround:** `import sparx_agency.tools.mission_control as mc` directly (via
`importlib`, with the repo root on `sys.path`) and call `mc.start_service(svc)` /
`mc.stop_service(svc)` / `mc.get_all_states()` against real `Service` objects pulled
from `mc.ALL_SERVICES`. Output is drowned in bare-mode warnings — grep for your own
`print()` markers rather than trying to read the raw output.

**Don't:** Don't assume a Streamlit app needs a real browser session (or refactoring
into a separate importable module) just to unit-test its orchestration logic — try a
plain import first.

## 2026-07-29 — broken torch import: orphaned cu12 packages + two corrupted cu13 installs

**Symptom:** `import torch` failed with `undefined symbol: ncclCommWindowDeregister` in
`libtorch_cuda.so`. After removing the conflicting packages (see Root cause), the same import
then failed with `libcudnn.so.9: cannot open shared object file`, then after fixing that,
`libnccl.so.2: cannot open shared object file` — three distinct failures in sequence, each
looking like it could be "the" bug.

**Root cause:** Two independent problems layered on top of each other in the project venv:
1. Both cu12 and cu13 variants of nearly every NVIDIA package (`nvidia-cublas`,
   `nvidia-cudnn`, `nvidia-nccl`, etc.) were installed side by side. `torch==2.11.0` declares
   `nvidia-nccl-cu13` etc. as its actual dependency, but the orphaned cu12 cluster (verified via
   `pip show <pkg>` — `Required-by:` was empty for every one of them) was shadowing/conflicting
   at import time, producing symbol mismatches against the older cu12 library.
2. Independently, `nvidia-cudnn-cu13` and `nvidia-nvshmem-cu13` had their `.dist-info` metadata
   present (`pip list`/`pip show` reported them installed) but their actual `.so` library files
   were missing from disk entirely — an interrupted or corrupted earlier install, unrelated to
   the cu12/cu13 conflict. Fixing #1 alone still left torch broken because of #2.

**Fix / workaround:** Diagnosed systematically rather than guessing package-by-package:
`pip show -f <pkg>` lists every file a package installed; checked each installed nvidia-* cu13
package's files against disk (`[[ -f "$location/$f" ]]`) to find exactly which ones were
actually missing (`nccl`: 1/1 missing, `nvshmem`: 12/12 missing — everything else was fine).
Uninstalled the entire orphaned cu12 cluster (`pip uninstall -y nvidia-*-cu12` for every one
with an empty `Required-by`), then `pip install --force-reinstall --no-deps` the two genuinely
broken cu13 packages specifically (not a blanket torch reinstall).

**Don't:** Don't assume the first import error is the only problem — each fix can unmask a
different, independent failure underneath. Don't `pip uninstall`/reinstall packages just
because they look suspicious; check `Required-by` first so a shared venv (many things in this
project depend on it) doesn't lose something another part of the codebase actually needs.
Don't reinstall a big package (torch) wholesale to fix what's actually a missing-file problem
in one of its dependencies — `pip show -f` + a file-existence check finds the precise culprit
in seconds and avoids re-downloading everything.

## 2026-07-30 — Rooster twist-control adapter's max_yaw_rate was ~4x too low

**Symptom:** No live crash or error — found via a logged manual flight, not a reported bug.
`rooster_twist_control_adapter.py` scales an incoming Twist to a `/R1/cmd_nav` axis value as
`axis = twist_component / max_component * 1000`, with `max_yaw_rate` defaulting to 0.5 rad/s
(never live-validated — same "assumed from doc convention" class of issue already seen once
with `turn_left`/`turn_right`'s r-axis sign, see the earlier stop-watchdog entry above).

**Root cause:** A subscribe-only command+pose logger (`manual_flight_logger.py`, joins
`/R1/cmd_nav` and `/R1/localization` by timestamp) captured a full manual flight and let the
actual axis->rate relationship be measured directly instead of assumed. 8 isolated
`turn_right` segments (axis r=500, each bounded by the next command so no averaging across
unrelated motion) gave a consistent ~55 deg/s (~0.96 rad/s), extrapolating to axis 1000 (full
deflection) -> ~1.9 rad/s. The configured 0.5 rad/s max meant any planner (FALCON's
`waypoint_follower_node.py` via `rooster_twist_control_adapter.py`) requesting even a modest
angular.z was actually commanding a turn ~4x faster than it thought it was asking for.
3 `turn_left` segments gave a lower ~42 deg/s (~1.5 rad/s at full scale) — a real left/right
asymmetry, or noise from the small sample, not yet resolved (see
`docs/progress/entries/007-rooster-velocity-controller.md`).

**Fix / workaround:** Recalibrated `max_yaw_rate`'s default from 0.5 to 1.8 rad/s (both the
constructor default and the CLI `--max-yaw-rate` default) in `rooster_twist_control_adapter.py`,
with the derivation documented inline. `max_linear_x`/`max_linear_y` were left unchanged —
this same flight's forward/lateral segments were too short and interleaved with adjacent turns
(leftover momentum contaminated each segment; several "forward" segments even showed net
*negative* displacement) to derive a trustworthy number. A dedicated calibration flight
(isolated single-axis moves, no interleaving) is the planned follow-up for those and to
confirm whether the yaw asymmetry is real.

**Don't:** Don't assume a `max_*` "rate at full deflection" constant is correct just because
no one has complained — it can be silently wrong for a long time if nothing downstream
saturates it in an obviously visible way. Don't try to calibrate axis->rate gains from a
flight that wasn't designed for calibration (this one mixed forward/turn commands back to
back) unless the segment is long/isolated enough that adjacent-command momentum can't
contaminate it — turn-rate segments here were trustworthy because each was bounded cleanly by
stops; translational segments were not, for the same reason.

## 2026-07-30 — flew after an R1 crash without restarting falcon, map was poisoned

**Symptom:** After `rooster_manager` crashed inside `R1` and Sphera was restarted, only
`ros1_bridge` and `video_trigger.py` were restarted (the documented fix for those two going
stale on R1 recreation) — `falcon` itself, running continuously since well before the crash,
was left alone on the reasoning "the drone respawned at the same spawn point, so the map
should still be valid." The next click-to-fly test got repeated `astar_planner` "boxed in" /
"PLAN FAILED ... pinched at (x,y) — thinner than the airframe, so most likely a mis-detected
voxel" warnings, the drone jumped several meters at a time while nominally stopped, and landed
~11m from the intended goal.

**Root cause:** `exploration_node`'s voxel map is long-lived state for the whole `falcon`
container's lifetime, and FALCON's TSDF/ESDF mapping has no exposed decay/forgetting or
hit/miss-probability knob at all (confirmed by reading the vendored source directly) — bad
fused data isn't temporally erased, it just sits there until enough new correct observations
outweigh it, if that happens at all before the bad region matters for planning. During the
`rooster_manager` crash window, `video_trigger.py` kept running and (per the existing
"does NOT reconnect after R1 respawn" gotcha above) very likely froze on its last decoded
frame and kept feeding stale/close-up depth into the still-running map as if it were real
geometry. "Same spawn point" says nothing about whether the map's accumulated voxels are
still trustworthy — that reasoning was the actual mistake.

**Fix / workaround:** Elevated to a first-rule item in the `fly-rooster-sphera` skill: any
time `R1` crashes or gets recreated, fully restart `falcon` (not just `ros1_bridge`/
`video_trigger.py`) before flying again. A full container restart costs under a minute and
guarantees a clean map, which is far cheaper than diagnosing "is the occupancy map buggy" as
if it were a deep, novel bug every time this happens.

**Don't:** Don't reason about map validity from drone *position* alone — the map is a
function of everything that got fused into it since the container started, not of where the
drone currently is. Don't skip restarting `falcon` after an `R1`-side crash just because the
immediately-obvious staleness fixes (bridge, video) are already in the routine — the map
itself is the thing most silently damaged by exactly that kind of disruption.

## 2026-07-30 — a directional cmd_nav action doesn't clear the other axes

**Symptom:** Drone got pinned against a wall during an autonomous click-to-fly test. Killed
`rooster_twist_control_adapter.py` (the competing publisher) and immediately sent a single
`{"action": "backward", "value": 300}` to back it away. Instead of retreating cleanly, the
drone's position and the altitude-hold ranger both went wild (ranger jumped 1.7m -> 3.1m ->
4.06m within 2s, velocity swinging -1.84 -> +0.94 -> -0.17 m/s) and it ended up far from where
a straight retreat should have put it.

**Root cause:** `rooster_command_unit.py`'s `_MOVE_ACTIONS` handling (`_on_cmd_nav`) only
overrides the ONE axis a named action maps to (`"backward": dict(x=-1)` only ever touches
`x`) — `y` and `r` are explicitly preserved from `unit.axes`'s current value, by design, so
that turning while flying doesn't zero the drone's throttle/altitude-hold `z` axis. But that
same preservation applies to `y`/`r` too: whatever `rooster_twist_control_adapter.py` had
last written (it had been actively commanding a nonzero yaw rate while fighting the stuck-
against-wall situation, right up until the moment its process was killed) stayed latched in
`RoosterUnit.axes` and got carried straight into the `backward` command. The drone was very
likely retreating correctly relative to its own nose (confirmed via `rooster_unit.py:164-168`
publish_manual - `x/y/r` pass straight into the vendor `ManualControl` message with no frame
transform in our code, and FCU `ManualControl` axes are body-relative on essentially every
FCU convention) while ALSO still spinning from the stale `r` - a body-relative retreat plus
continuous yaw looks, from the world frame, exactly like "backward went somewhere wrong."

**Fix / workaround:** Send `stop` first (a distinct action, not part of `_MOVE_ACTIONS` -
zeroes x/y/r fully) to clear ALL latched axes, THEN send the desired directional action.
Never assume killing a competing publisher process also clears the state it already wrote
into `RoosterUnit` - the axes it set are still sitting there until something explicitly
overrides or zeroes them.

**Don't:** Don't reach for a single named directional cmd_nav action as an emergency override
without a preceding `stop` when a different controller (autonomous or otherwise) was just
active - it inherits whatever that controller left on every axis except the one you're
setting. Don't assume "the drone moved in a weird direction" means the axis convention itself
is wrong (body- vs world-frame) before checking what was actually latched on the OTHER axes
first - the frame convention here was correct; the bug was leftover state.

---

## 2026-08-02 — ros1_bridge crashed with std::bad_alloc after every Sphera restart, no voxels in RViz

**Symptom:** After restarting Sphera/R1, `ros1_bridge` (launched via `run_bridge.sh` with only
`ROS_DOMAIN_ID=9` set) crashed within ~1s of the first real messages flowing (`terminate called
after throwing an instance of 'std::bad_alloc'`), right after logging "Passing message from ROS 2
geometry_msgs/msg/PoseStamped..." or a plain `std_msgs/String`. `falcon`'s `mapping_sync` heartbeat
stayed at `gate=WARMUP`, `last_pose` growing forever, so no voxels ever appeared in RViz. Recreating
`R1`, restarting `it`, and retrying the bridge repeatedly all failed to fix it.

**Root cause:** `run_bridge.sh`'s own header comment says Rooster/R1 requires
`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` plus `CYCLONEDDS_URI=file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml`
(R1 is Jazzy/CycloneDDS; XTEND, the script's default, is Foxy/FastRTPS) - only `ROS_DOMAIN_ID=9` was
being passed. The bridge ran on `rmw_fastrtps_cpp` trying to deserialize CDR from a CycloneDDS
publisher; the cross-vendor CDR mismatch is what threw `bad_alloc`, not the previously-documented
Sphera CycloneDDS interface corruption bug ([[project_sphera_cyclonedds_interface]] in memory) - that
one only applies when both sides already agree on CycloneDDS.

**Fix / workaround:** Always launch the bridge for Rooster/R1 with all three overrides together:
`ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///home/user1/rqs_iai_ws/src/cyclonedds.xml ./run_bridge.sh`.
`ROS_DOMAIN_ID` alone is not enough. Confirm by checking the bridge's own startup banner prints
`RMW (ROS2): rmw_cyclonedds_cpp`, and confirm real flow with `rostopic hz /R1/localization` inside
`falcon` (not by grepping the bridge log for "Passing message" - that line only ever prints once per
topic type, so it can't tell you whether the stream later stalled).

**Don't:** Don't assume a repeat `bad_alloc` after a Sphera restart is the known CycloneDDS-corruption
bug and reach straight for "restart Sphera again" - check the bridge's actual RMW banner first. Don't
trust `mapping_sync`'s heartbeat alone to mean "no bridge running" - `pose=2, buf=1` stuck and aging is
what it looks like when the bridge greeted the topic once and then silently dropped it, distinct from
`pose=0, last_pose=-1.0s` (never connected at all).

---

## 2026-08-03 — Rooster's published yaw had TWO independent sign/reference bugs, not one

**Symptom:** BEV map/mapped geometry appeared on the wrong side of the drone from the way it was
actually facing in Sphera (e.g. drone visually facing +Y, mapped area showing up toward -Y). Separately,
`turn_right`/`turn_left` from the manual UI always worked correctly, but a controller computing a
specific relative turn (e.g. "turn +30deg") sometimes went the wrong way.

**Root cause:** Two distinct, independent bugs in `rooster_ground_truth_localization.py`'s `_on_state`,
both hiding behind similar-looking symptoms:
1. **Rotational sense.** The code negated yaw (`-msg.rotation.yaw`) based on a comment claiming
   Sphera/Unreal yaw is left-handed/clockwise-positive. Confirmed live via a real commanded `turn_right`
   while airborne (compare raw `msg.rotation.yaw` against published `/R1/localization` yaw at the same
   timestamps): raw yaw DECREASED during the real right/clockwise turn, i.e. it was already standard
   REP103 CCW-positive. The negation was taking an already-correct value and flipping it.
2. **Zero-reference axis offset (a completely separate degree of freedom from #1).** Confirmed on a
   fresh, untouched spawn (no commands sent since the last Sphera restart): the drone visually faces
   world +Y there, but its yaw read ~-94deg, not the +90deg that FALCON's own camera-to-world depth
   integration assumes (yaw=0 means forward aligns with world's own +X, CCW-positive) for a drone facing
   +Y. Off by ~180deg - independently explains the BEV map symptom, since FALCON would then integrate
   depth data toward almost the exact opposite of where the drone was really looking.
   `sphera_jail.yaml`'s `init_yaw: 0.0` for this same spawn point is consequently also wrong, but that's
   a repo-wide placeholder default used identically on every map file, not something calibrated
   per-map, so it was left alone (real localization data supersedes it within milliseconds anyway).

**Fix / workaround:** `yaw = atan2(sin(raw_yaw + pi), cos(raw_yaw + pi))` - no negation (fixes #1), plus
a `+pi` (180deg) offset before wrapping (fixes #2). Verified: the corrected value on that same fresh
spawn came out to 85.6deg, matching the expected +90deg closely (small residual because the real spawn
heading isn't perfectly axis-aligned - not a bug).

**Don't:** Don't assume "yaw problem" is one bug just because both symptoms involve yaw and both got
fixed by edits to the same function - these were two independently-derived, independently-confirmed
fixes with completely different evidence (a live turn test vs. a static fresh-spawn quaternion reading).
Fixing #1 alone did not fix #2, and there was no a-priori reason #2 had to exist at all once #1 was
fixed. Also don't put a fix like this inside `core/planning/trackers/waypoint_follower` (tried and
reverted mid-session) - that package is deliberately drone-agnostic and assumes standard REP103 yaw;
the correct place is the Rooster-specific localization boundary that produces the yaw in the first
place, so every consumer (core's own bearing math, FALCON's C++ mapping, the BEV click arrow) gets
fixed at once without needing its own compensating hack.

---

## 2026-08-03 — computed turns went backwards while manual turn_left/turn_right always worked

**Symptom:** The manual UI's `turn_right`/`turn_left` buttons always turned the drone the correct way.
But any controller-computed turn (waypoint, drift_pid, ...) that needed a specific relative correction
(e.g. "turn +30deg") sometimes executed in the opposite direction, and turns driven by `/cmd_vel` never
settled - repeatedly overshooting past the target and correcting back the other way, forming a sustained
pendulum that never completed.

**Root cause:** Two separate, additive problems, both inside `rooster_twist_control_adapter.py` (the
one, shared conversion point every controller's `/cmd_vel` funnels through before reaching the FCU -
manual UI actions never go through this file at all, straight to `rooster_command_unit.py`'s named-
action dict instead, which is exactly why manual always worked):
1. **Sign mismatch.** Standard REP103 (which `waypoint_follower`/`drift_pid` correctly assume, since
   `core/` is deliberately drone-agnostic) has positive `angular.z` = CCW = left. This drone's FCU axis
   convention has positive `r` = right (confirmed live, same convention already baked into
   `rooster_command_unit.py`'s `turn_left: r=-1` / `turn_right: r=+1`). The adapter was doing a direct,
   unflipped `r = angular.z / max_yaw_rate * 1000` - so a controller correctly computing "turn left"
   executed as "turn right", the opposite of intended, for every controller.
2. **No damping on top of an undamped FCU loop.** PX4's own yaw-rate control loop has zero derivative
   gain (`MC_YAWRATE_D=0.0`, confirmed via `/R1/fcu/param/get_float` - P/I only). The adapter was
   forwarding whatever r-axis value a controller requested as an instantaneous step change every tick,
   exactly the kind of abrupt input that excites a P/I-only loop into sustained oscillation. Per the
   user's explicit preference, this was NOT fixed by touching the PX4 parameter itself.

**Fix / workaround:** (1) Negate: `target_r = -angular.z / max_yaw_rate * 1000`. (2) Added a
`slew()` rate-limiter (new helper in `robots/common/math_utils.py`) so the adapter ramps its own r-axis
output toward the target by at most `max_yaw_axis_step_per_sec` (2500, a deliberately conservative
first guess - live-test and retune) per second, instead of snapping to it - smoothing the input without
touching PX4. `stop_motion()` resets the ramp state directly so a stop is still immediate.

**Don't:** Don't assume a shared symptom across every controller means the bug is in one specific
controller's tuning - check what code path is actually COMMON to all of them first (here, the single
Twist->cmd_nav converter) before hunting through per-controller PID gains. Don't conflate "the turn
never settles" (damping/oscillation) with "the turn goes the wrong way" (sign) - they were reported
and diagnosed as one thing at first, but are two independent bugs with two independent fixes; a
sign-flipped, undamped loop produces a much more violent, sustained oscillation than an undamped-but-
correctly-signed one, so a report of severe swinging is a hint to check the sign FIRST, not just reach
for damping.

---

## 2026-08-10 — `robotican_dev` container: wrong name + missing engine mount broke three Rooster services

**Symptom:** After the `robotican_dev` container exited (stale, 3 days old) and was brought back up with
`docker compose -f docker-compose.robotican.yml up -d`, mission_control's Rooster Frame Capture, Depth
Processor, and Twist Control Adapter all failed identically: `Error response from daemon: container
<hex-id> is not running` — quoting the ID of the OLD, exited container, even though a freshly-started
one was clearly running. Once that was fixed, Depth Processor then failed with `FileNotFoundError` for
the DA3 TensorRT engine, despite the file existing fine on the host.

**Root cause:** Two independent bugs in `docker-compose.robotican.yml`, both invisible until something
tried to recreate the container from scratch: (1) no `container_name:` was set, so `docker compose up`
named it `theagency-robotican-1` (the default `<project>-<service>-<index>` pattern) instead of the
literal `robotican_dev` every `run_*.sh` wrapper and `mission_control.py`'s `proc_container` hardcode —
the previous, long-lived container only had the right name because it predated this compose file /
was started some other way, so the mismatch never surfaced until it needed recreating. (2)
`~/depth_anything_ws` (where the DA3 engine + ONNX live) was never in the compose file's volume list at
all, so the path doesn't exist inside the container regardless of naming.

**Fix / workaround:** Added `container_name: robotican_dev` to the compose file's service block, and
added a `${HOME}/depth_anything_ws:${HOME}/depth_anything_ws` volume (read-write, not `:ro` — a TRT
version mismatch has needed rebuilding the engine from inside this exact container before, see the
2026-07-29 entry above).

**Don't:** Don't assume "container is not running" means the container just needs restarting — check
`docker ps -a` for the EXACT name being `docker exec`'d into first. A compose-managed container can
come up successfully under a completely different name than every script expects, and the error message
alone won't tell you that.

---

## 2026-08-10 — FALCON `exploration_node` stuck in a "no path to next viewpoint" loop with the drone still on the ground

**Symptom:** After wiring `nav_mode:=exploration` (`exploration_node` → `traj_server` →
`falcon_exploration_follower_node.py`), the drone never moved. `exploration_node` was clearly alive and
computing (HGrid/TSP/SOP all logging every ~30-50ms, correctly picking a next-viewpoint target), but
every attempt ended in `[ExplorationManager] planTrajToView: No path to next viewpoint using default
A*` / `... using coarse A*` / `[FSM] Plan fail`, repeating in a tight loop — with the map otherwise
healthy (`mapping_sync` reporting `gate=FUSING`, pose and depth both flowing).

**Root cause:** The drone was armed but had never taken off — sitting at ground level. Exploration's
frontier viewpoints are chosen in free 3D space at flight altitude; A* has no reachable route from a
ground-level start pose to one of those. `falcon_exploration_follower_node.py` correctly recognized it
had no valid trajectory and held station the whole time (by design, via `ReferenceTracker3D`'s own
stale-reference handling) rather than fabricating motion — so nothing in the follower's own logs pointed
at "not flying yet" as the cause.

**Fix / workaround:** Take off and get the drone hovering first, *then* switch to `nav_mode:=exploration`.
Confirmed live: within seconds of takeoff, `exploration_node` started finding paths, `traj_server`
reported real flight-time/path-length progress, and coverage climbed past 97%.

**Don't:** Don't debug this as a planner/map bug (checking safety clearance, map connectivity, frontier
placement, etc.) before confirming the aircraft is actually airborne — the log signature is identical
either way, and chasing the map/planner side first wastes time on a non-issue.

---

## 2026-08-10 — Rooster Frame Capture: `PermissionError` writing into `/tmp/rooster_frames`

**Symptom:** `rooster_frame_dir_publisher.py` (Rooster Frame Capture, runs inside
`robotican_dev`) crashed on its very first frame with `PermissionError: [Errno 13]
Permission denied: '/tmp/rooster_frames/frame_00000001.tmp'`.

**Root cause:** `/tmp/rooster_frames` and `/tmp/rooster_depth` are shared host directories
bind-mounted (at the same path) into `robotican_dev`, `falcon`, and `detector_dev`. Docker
auto-creates a bind-mount's host-side directory as `root:root` mode `0755` if it doesn't
already exist when a container starts — and since `/tmp` is typically cleared on reboot,
whichever of those containers happened to start first ended up creating the directory as
root. `robotican_dev` (the one that actually needs to WRITE the frame/depth files) runs as
non-root `uid 1000`, so it couldn't write into a root-owned 0755 directory. `falcon` and
`detector_dev` both mount these paths read-only, so neither could fix it from inside
either — the fix had to come from the host.

**Fix / workaround:**
```bash
sudo chmod 777 /tmp/rooster_frames /tmp/rooster_depth
```
Do this any time `/tmp` has been freshly cleared (reboot) before the first bring-up of the
day.

**Don't:** Don't assume a `PermissionError` here means something wrong with
`rooster_frame_dir_publisher.py` itself — check `ls -ld` on the two shared dirs first;
whichever container starts first on a clean `/tmp` decides the ownership for everyone else.

---

## 2026-08-13 — `rooster_ground_truth_localization` silently received nothing after Sphera restart

**Symptom:** After restarting Sphera, `/R1/localization` published nothing (confirmed via a
real subscriber script, not `ros2 topic hz` — see the entry below on why that tool lied).
Restarting `rooster_ground_truth_localization` didn't fix it either.

**Root cause:** Sphera's own engine publishes `/R1/sphera/state` at `BEST_EFFORT` reliability
from a bare DDS participant (not rclcpp). `rooster_ground_truth_localization.py`'s subscriber
used a plain integer queue depth (`create_subscription(..., 10)`), which rclpy expands to the
default `RELIABLE` profile — incompatible with a `BEST_EFFORT` publisher, so it silently
received zero messages. `ros2 topic info --verbose` showed the exact warning: `New publisher
discovered on this topic, offering incompatible QoS... RELIABILITY_QOS_POLICY`.

**Fix / workaround:** Subscribe with an explicit `QoSProfile(depth=10,
reliability=ReliabilityPolicy.BEST_EFFORT)` instead of a bare int. Confirmed live: ~140Hz
real pose data end-to-end afterward. The same two-publishers-different-QoS pattern also
shows up on `/R1/state` (ranger) — one `BEST_EFFORT` bare-DDS publisher, one `RELIABLE`
`rooster_manager` publisher — but that one has always been benign since the real consumer
subscribes `RELIABLE` (compatible with `rooster_manager`), so don't chase the warning there.

**Don't:** Don't assume "the process is alive and logs 'ready'" means it's receiving data.
Check `ros2 topic info --verbose` for a QoS mismatch warning before assuming the bug is
somewhere else (Sphera itself, the bridge, network).

---

## 2026-08-13 — `ros2 topic hz` / `rostopic hz` report nothing on `BEST_EFFORT`-only topics

**Symptom:** Repeatedly checked "is data flowing?" with `ros2 topic hz` / `rostopic hz` and
got zero output, even when a real, independently-confirmed consumer (a node's own log
showing incrementing counts) proved the topic was very much alive.

**Root cause:** Both tools subscribe with default (`RELIABLE`) QoS. Any topic whose only
publisher is `BEST_EFFORT` (Sphera's own sim-engine topics are, throughout this stack) is
QoS-incompatible with that default subscription, so the tool silently sees nothing. This
ROS2 distro (Foxy) doesn't even have a `--qos-reliability` override flag for `ros2 topic hz`.

**Fix / workaround:** Don't trust `hz`/`rostopic hz` as ground truth on any Sphera-originated
topic. Write a one-off subscriber script with explicit `BEST_EFFORT` QoS instead (a few lines
of rclpy/rospy) — that's what actually caught the real data rate every time this came up today.

**Don't:** Don't conclude a topic is dead from `hz` alone. Cross-check against a real
consumer's own log, or a hand-rolled subscriber with matching QoS, before restarting things.

---

## 2026-08-13 — Altitude-hold PD loop ran at 1Hz, ~10x slower than its own sensor

**Symptom:** `rooster_command_unit`'s altitude hold would drift 0.5-1m past target and
either never recover, or oscillate in a stuck band for minutes — confirmed live across many
flights, at multiple `altitude_hold_max_correction` values (200, then 380).

**Root cause:** Two separate bugs stacked on each other.
(1) `altitude_hold_interval_sec` defaulted to `1.0`, but `/R1/state` (the ranger source)
actually updates at ~10Hz — the loop was reacting to only 1 in 10 fresh readings on a true
double-integrator plant (throttle → accel → velocity → position), way too slow to converge.
(2) Naively raising the loop rate to match (~10Hz) then aliased against the sensor's own
~10Hz rate: on ticks where no new sample had arrived yet, `ranger == prev_ranger` computed a
false `velocity=0`, producing an undamped P-only correction that alternated with properly
-damped ticks every other cycle — visible directly in the log as a correction value bouncing
between wildly different numbers tick to tick.

**Fix / workaround:** `altitude_hold_interval_sec` raised to `0.1` (still runs the ROS timer
at 10Hz), AND `_altitude_hold_tick` now skips entirely when `ranger` hasn't actually changed
since the last real sample, using the true elapsed wall-clock time (`time.monotonic()`) for
the velocity term instead of the fixed loop interval. Confirmed live: a clean, monotonic
convergence to within 2cm of target in under 5 seconds on one flight (vs. minutes of
oscillation before) — though a later flight still didn't converge cleanly, and turned out to
be physically wedged against a door frame at the time, not a control-loop failure.

**Don't:** Don't just widen the correction clamp again if this recurs — check `/R1/state`'s
actual publish rate first (a hand-rolled subscriber, not `hz` — see the entry above) before
assuming the gains or the clamp are the problem.

---

## 2026-08-13 — Voxel map looks fragmented/floating; root cause is altitude instability, not FALCON

**Symptom:** RViz's Voxel Mapping display repeatedly showed disconnected chunks of occupied
cells floating above the main wall/floor structure, with visible gaps — reproduced fresh on
multiple flights, on a completely reset map each time, so not stale/leftover data.

**Root cause:** Confirmed via direct query (subscribing to `/voxel_mapping/occupancy_grid_occupied`
and comparing X/Y footprints across height bands): the vast majority of "floating" cells had
zero occupied structure directly below them, while every low cell also had structure above
it. This matches a coverage gap, not corrupted data: the drone scans real walls at whatever
height it happens to be flying, but the still-imperfect altitude hold means that height
varies flight to flight (and even within one flight), so the same X/Y spot's lower wall
section sometimes never gets scanned from a floor-hugging vantage. FALCON's own exploration
logic doesn't force a second, different-height pass over an already-"seen" frontier.

**Fix / workaround:** None yet — the aliasing fix above improves *convergence* but the hold
still isn't perfectly steady through a whole flight, and that residual variance is enough to
leave gaps. The real fix is making our own altitude hold hold a genuinely constant height
throughout exploration, not a FALCON-side change (FALCON's frontier logic just faithfully
maps whatever height band the camera was actually pointed at).

**Don't:** Don't reach for FALCON's own exploration/frontier code to fix this — the root
cause is in our altitude-hold stability (`rooster_unit.py`), not upstream FALCON logic.

---

## 2026-08-13 — Launching Sphera + Rooster has no CLI/service hook; it's GUI-only past the container restart

**Symptom:** Wanted to script/automate the full Sphera + Rooster bring-up. Found and confirmed
`~/.sphera/sphera-restart.sh` handles the container restart itself cleanly (`docker compose
down` + `up -d run_sphera`), but the drone never actually became controllable after just that.

**Root cause:** Everything past the container restart is a click inside Sphera's own rendered
(Unreal) window, with zero corresponding shell command, ROS topic, or file change to key off
of: a "Welcome to SPHERA" screen needing a "Continue" click, a role-select screen ("Manager" is
the only enabled option in this build), a "Manager mode" scenario picker (`Start` →
`Standalone`), then assigning the drone to an operator and pressing the green ▶ Play button.
`ros2 service list`/binary `strings` turned up no scenario-control service or CLI flag.
**Play is also what spawns the separate `R1` backend container** (`sphera-backend:rooster`,
`docker run --name R1 ...`) — it's not a distinct manual step, just an automatic side effect of
that one click. If `R1` dies mid-scenario (vs. the container never having existed), Sphera does
NOT auto-respawn it — replaying the exact captured `docker run` command by hand works fine.

**Fix / workaround:** No automation exists for the GUI portion. `xdotool` could theoretically
click through it (proportional coordinates within the window's current geometry, not absolute
screen position — Sphera's window isn't pinned to one place or size), but that was explicitly
decided against for now given the fragility of blind GUI-coordinate clicking. Manual for now.

**Don't:** Don't assume "restart the container" is equivalent to "the drone is flyable again" —
always confirm ROS2 nodes/topics under `/R1/*` and `/Rooster_1` actually reappear before moving
on to restarting the falcon-side stack.

<!--
Example, in the style already proven useful in project-specific skills:

## 2026-07-22 — hover_z drift at 560

**Symptom:** Drone gets airborne and holds briefly at hover_z=560, then slowly drifts to
the ceiling over 1-2 minutes. Lower values (550) sink to the floor instead; higher values
(575+) drift up faster.

**Root cause:** Unresolved as of this writing — shortening climb_duration_sec (tested at
1.5s) did NOT fix the drift, which rules out simple integral windup as the sole cause.

**Fix / workaround:** No real fix yet. Use hover_z=560 and budget for manual landing before
~1 minute on any longer test.

**Don't:** Don't assume climb_duration_sec is the whole story — already tested and ruled out.
-->

---

## 2026-08-26 — InternVLA-N1 on SJTU: three knobs that were read by nobody, and a wall that rotation could not escape

**Symptom:** With everything documented and configured, the hospital flights produced
almost no continuous trajectory: 18 of 22 committed routes were 0.25 m two-point stubs,
the drone spent 43% of a flight stationary, five separate twelve-second stalls in ninety
seconds, and one run sat 0.45–0.70 m from a wall for seventy seconds re-committing the same
forward step. From the outside it looked like the model producing short plans.

**Root cause:** Four independent things, none of which raised anything.

1. **The agent hands the curve over and then queues its own discretisation.** One System-1
   pass renders twice — the continuous path *and* the list of 0.25 m / 15° steps that
   approximates it. Upstream returns `idx[0]` now and queues `idx[1:]` for the next three
   calls, which is right for a discrete client and double-counting for one flying the curve:
   it had already covered that ground, and three of every four decisions carried no curve at
   all. Server-side, needs a patch (PATCH 7, `sys1_continuous_only`).
2. **`commit.expected_speed_mps` and `commit.commit_grace_s` were in the YAML and never read
   by the node.** Every commitment therefore fell back to the flat `max_commit_s: 12` — the
   exact twelve-second stalls the YAML comment claims those two knobs removed. The config
   file described a fix that did not exist. Found by an offline dry run, not by a flight.
3. **A discrete TURN action was flown as a bent 0.25 m waypoint.** Upstream a turn is a pure
   rotation (`trajectory_to_discrete_actions_close_to_goal` advances `pos` only on a forward
   action), but everything downstream consumes polylines, so it was rendered as a step — and
   a *holonomic* pursuit satisfies that by crabbing sideways. The model asked to look
   somewhere and the aircraft shuffled 0.25 m without changing where it was pointing.
4. **`/agent/init` against a live agent is a server-side no-op**, so changed intrinsics,
   `sys2_max_forward_step` and the new flag all reached a server started before the change
   and were silently ignored — the flight behaves like the old configuration while every
   file on disk says otherwise.

**Fix / workaround:** PATCH 7 in the vendored agent (opt-in, defaults to upstream behaviour);
pass `expected_speed_mps`/`commit_grace_s` through to `CommitSpec`; fly a turn as a rotation
via `core/planning/vlas/common/turn_in_place.py` on a `/n1/yaw_goal` topic; warn loudly when
`init_agent` short-circuits with settings in hand, and restart the server after changing any.
Also: hold the aircraft still while System 2 thinks (`hold_to_think`) — System 2 takes 2.6–10 s
and the route is anchored at the pose the *frame* was taken from, so a moving aircraft
anchors its next route metres behind itself.

**Then the failure moved.** With turns flown as real rotations, an aircraft that got inside
its own stopping distance of a wall could no longer escape: the wall stays inside the depth
corridor across most of the arc, so every heading reads blocked. Measured: thirteen rotations,
zero metres. **Rotating is only half an escape** — it has to break contact first
(`core/planning/recovery/escape_maneuver.EscapeManeuver`: brake, 0.5 m of reverse, settle,
*then* rotate). Holding still to think makes this worse, not better: a blocked aircraft asking
from a stationary frame gets a byte-identical answer for ever.

**Don't:**
- Don't trust a config comment that describes a fix. Grep for the key in the code that reads
  it. Two of the four above were knobs nobody had ever passed.
- Don't debug this from flights. `tasks/planning/sjtu_internvla_n1/scripts/dry_run.py` flies
  the real nodes against a fake drone and a scripted server in ninety seconds, with no Gazebo,
  no GPU and no model, and it found (2) and the incomplete escape before a run was spent.
- Don't run `record_campaign.sh` from a shell that has not sourced ROS — until this was fixed
  it sourced nothing itself, so `ros2` failed on an importlib traceback, the odom wait burned
  its full sixty seconds, and every run was written off as "could not reach the area" with
  nothing anywhere naming the cause. It only ever worked by accident.

---

## 2026-09-01 — `bool()` on an LLM's boolean field is a silent always-true

**Symptom:** The scene-graph target watcher latched `/target_seen` on the first object of the
flight — a *shelf*, for the target "wheelchair" — and the JSON it published carried the model's
own reason: "CLASS is a different object and only an accessory of TARGET". Every subsequent
object also "matched". The mission's stop condition fired within seconds of the first
detection, every run.

**Root cause:** `qwen2.5:3b-instruct` answers `{"match": "false"}` with the word **quoted** — a
string, not a JSON boolean. The verdict was read as `bool(reply.get("match", False))`, and
`bool("false")` is `True` in Python because the string is non-empty. The prompt, the model's
reasoning and the logged reason were all correct; only the coercion was wrong, which is why
every log line read as a correct rejection while the code recorded a hit.

**Fix / workaround:** `core/mapping/topology/llm_client.py` now exports
`coerce_bool(value, default=False)` — real bools, numbers, and the words
`true`/`false`/`yes`/`no`/`1`/`0`, defaulting to `False` for anything unrecognised (a watcher
that stops the search is worse than one that keeps looking). `target_matcher._ask_llm` uses it.
Pinned by `core/mapping/topology/tests/test_target_matcher.py`
(`test_quoted_false_is_not_a_match` plus a parametrised coercion table).

**Don't:** Don't reach for `bool()` on ANY field that came out of a language model — it fails in
the dangerous direction. Numbers are safe (`float("0.9")` is `0.9`), so this is specific to
booleans. And don't trust a reason string in a log as evidence the verdict was right: here the
two disagreed for the whole flight.

---

## 2026-09-01 — a door list that cannot cut the building, measured against ground truth

**Symptom:** The scene-graph room segmentation reported the entire explored wing of the hospital
as ONE room — 22,704 BEV cells (~511 m²) holding 64 objects. Every object landed in the same
room, room typing had nothing to separate, and the per-room probability ranking was meaningless.

**Root cause:** Two things, and the second only became visible after fixing the first.
1. The 24 hardcoded doors ported from the `sjtu_project` stack are not enough to partition this
   building: measured on the ground-truth map they leave one component at **41% of the floor**.
2. Measuring it at first showed a room LARGER than the free-cell count, because `hospital.npz`
   has free space on BOTH sides of the outer wall — the outdoor space wraps the building and
   connects everything, so no door cut can sever anything until the interior is isolated with
   `core/planning/environment/grid_regions.largest_enclosed_region` (that helper's own docstring
   documents this, and `robots/SJTU/maps/README.md` warns about sealed voids reading as free).

**Fix / workaround:** `robots/SJTU/maps/hospital_doors.yaml` is now the flown 24 merged with the
11 non-duplicate lintel-band portals from `hospital_regions.yaml` (co-registered, same frame,
regenerated by `tasks/mapping/gazebo_world_occupancy/build_regions.py`); duplicates within 0.75 m
were dropped. Measured on the isolated interior: flown-24 alone 23 rooms / largest 41% of the
floor; the merged 35 doors 29 rooms / largest 13%. Live, the explored wing went from 2 rooms to
4, with 25 of the 35 doors discovered.

**Don't:** Don't measure a room segmentation against a ground-truth occupancy map without
isolating the interior first — the outdoors silently joins every room into one and makes the door
list look innocent. And don't assume a hand-tuned door list ported from another stack covers the
world it is pointed at; this repo already ships a derived portal set for this hospital, so
reconcile with it rather than trusting the port.
