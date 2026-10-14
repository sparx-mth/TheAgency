# 022 - The Hanson re-fly: the sight ledger, finished-by-looking rooms, cue glances, the footing sweep

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** implemented; 1869 regressions pass across the runtime, exploration, topology and objnav packages (1040 runtime/core before); re-flown twice; the cross-floor couch campaign flown twice (Ranchester SR 1, SPL 0.56 on the second)
**Roadmap item:** ObjectNav exploration efficiency (010) and room-search loop (011); follows 021

## Diagnosis (runs/zson-hanson-3x-20261005, after the changes of 021)
Read against the user's notes on the three recordings.

**Hanson/000000 (toilet, STOP at 359 beside a real toilet; SR=0 is the annotation's doing):**
1. **O4 at action 25 ("GAP") and the walk back to it at 51-82.** The unknown between the spawn
   point and the bed in front of it is an island: the camera's blind radius leaves the floor
   under its own feet unseen, the bed and the walls close it. Nothing distinguished an enclosed
   pocket from an exit; it was an opening valued 0.01 and taken first because it was near.
2. **O2/O6 at 25-51.** The balcony was seen whole from its threshold at action 38; the unknown
   beyond its railing (the depth returns nothing at floor height) stayed frontier for ever, and
   the loop walked to both ends to peek.
3. **R0 after action 25.** Right by the type prior; the walk to O4 was the island, not the room.
4. **No look left at 101.** A bathroom vanity read `cabinet 0.84` on the left edge of the frame;
   the glance scorer values unknown floor, and the right side (a larger unknown) won twice.
5. **R11 -> R14 -> R16.** `RoomRegistry` matched by IoU >= 0.15; a room seen through its door grows
   tenfold as the agent walks in and lost its number at each growth spurt -- and with the number
   `last_inside`, so each new number read `entered=no`.
6. **O9 at 150, behind the bed.** The strip behind the bed was a room of its own (R13, 2 m2, no
   door) with a gap into an enclosed pocket; valued 0.35 by the 3B oracle. The bedroom R14 was
   `bedroom strong` on that action and still valued 0.10: the loop computed the exclusions
   BEFORE the oracle call re-classified the rooms.
7. **R5 at 206.** The balcony was offered again at 0.25, "never entered", 158 actions after the
   agent had stood at its far end: a room was finished only by a full rotation or a scan point
   that saw half of it, never by having walked it.
8. **R10 at 270-289.** A bed and a television through the door; "bedroom?" weak (one kind of
   object) by the 2026-10-05 merge guard; the room entered and scanned for a toilet.

**Hanson/000002 (potted plant, 161 actions, agent error):** spawned 0.3 m from a bed with the
plant in view 4.4 m away. The locked target had no safe path -- every cell around the feet
unknown, unknown impassable -- and the closing waited for A* with a centring hold (an idle
TURN_LEFT/TURN_RIGHT oscillation) for 159 actions until the 160-action bound.

**Hanson/000001 (chair, 14 actions, false STOP):** a child's ride-on horse read as `chair` 0.9
at 0.9 m. No toy was in the vocabulary for the open-vocabulary detector to land on.

## Goal
Unknown the camera has looked at without a return, and unknown enclosed by what is known, is
not a place to go; a room the agent has walked through or looked into with nothing left to
look at is finished; a bed through a door is a bedroom; an object at the edge of the frame is
looked at; a room keeps its number as it grows; an agent that cannot plan a step looks at the
floor around its feet; a start is never sampled beside a bed; a toy is a toy.

## Steps
- [x] `methods/sightlines.py`: `SightSettings`, `floor_band`, `cone_seen`, `enclosed_pockets`,
      `SightLedger` (per-floor looked counts with pose bins, pockets, `resolved`, `overlay`,
      poses, `stood_in`); `RPTSearchPolicy.sight`, observed every action after the map update
      (the closing path too); `sight` in `configuration()` and `episode_info()`.
- [x] The overlay in the frontier logic: `ObservedSceneGraph.resolved_provider` /
      `frontier_world` (set in `FloorContextBank.new`), `count_frontier_clusters` and
      `accessible_frontiers(frontier_world=)` read it; `UnknownView(resolved=)` for the glance
      scorer; `floor_panels` draws settled unknown darker (`settled_cells`).
- [x] `room_scans.py`: `SEEN_THROUGH` / `FRAGMENT`, `status(frontier=, doored=)`,
      `_status_without_frontier`, `_looked_into` (progressive, `pose_batch`), `mark`;
      `LoopSettings.fragment_max_m2` / `walkthrough_clearance_m`; the loop's `_live_frontier`,
      `_exclusions(frontier=...)`, `_finished_on_the_way` in transit (`finished_in_transit`).
- [x] `ObservedSceneGraph.refresh_labels`; `RoomSearchLoop._reason` refreshes the labels and
      re-runs the exclusions before the oracle call.
- [x] `room_priors.SIGNATURE_OBJECTS` / `signature_type`; `RoomLabelSettings.signature_objects`;
      `RevisableRoomLabels` strength (`signature` in the metadata).
- [x] `RoomRegistry(containment_threshold=0.6)`: a second tier of matches under the IoU bar.
- [x] `path_glances.py`: `cue_*` settings, `_cue` / `_start_cue`, `cue` on the look's events and
      HUD info; `cue_turns` bounds the look.
- [x] `camera_control.py`: `begin_inspection(reason="footing")`, `footing_pitch_deg` /
      `footing_turns`, `footing` property, the takeover's camera ownership keeps a footing sweep;
      `target_closing.py`: `footing_*` settings, `_no_path`, `_footing_command`, the boxed-in
      release (`_release(remember=False)`, `boxed_in`, `_boxed_in_here`); the exploration
      fallback's `footing` rung before the hold.
- [x] `gibson/generate_development.py`: `MIN_START_CLEARANCE_M`, `start_clearance`,
      `generate_start(min_clearance_m=)`; `multifloor_generation.cross_floor_starts` the same.
- [x] `core/planning/objnav/labels/datasets/gibson.py`: `toy`, `rocking horse`, `stuffed animal`,
      `bathtub`.
- [x] Regressions: `test_sightlines.py` (13), `test_room_scans.py` (+1), `test_room_registry.py`
      (+3), `test_path_glances.py` (+3), `test_target_closing.py` (+3), `test_multistory_repair.py`
      (+2), `test_exploration_fallback.py` (+1), `test_distinct_buildings.py` (clearance);
      `test_opening_nodes.py` turns the ledger off in its fixture (its "patch by the door" is a
      pocket) and, with `test_room_scans.py` / `test_room_search_loop.py`, uses a sink where a
      weak label is meant (a toilet is strong now).
- [x] README (closing, what is not a node, the two verdicts, openings, fallback rungs, cue
      glances, the sight ledger section, room numbers, vocabulary), MULTISTORY.md (clearance),
      CHANGELOG, LESSONS.
- [x] Re-fly the three Hanson episodes with a restarted detector (the vocabulary changed) and
      read the record (`runs/zson-hanson-3x-20261005-b`, below).
- [x] Fix what the second fly showed in the target closing (footing cut short when a path appears;
      pitch from the target's height, not its label; STOP on a fresh in-range sighting at any pitch;
      step toward a fresh candidate within one turn of the centre; the legacy target evidence
      refuses what the takeover refuses) with regressions (`test_target_closing.py` +4, 1 updated).
- [x] The third Hanson fly was interrupted; the user judged the second fly's videos good and asked
      for a target that is NOT on the start storey instead.
- [x] `generate_development --multistory --categories couch` (`cross_floor_starts(categories=)`);
      a three-building cross-floor couch manifest (`runs/zson-couch-crossfloor-3x-20261005*/episodes.json`).
- [x] From the first couch fly: the oracle's `home_here` verdict and `unexplored_elsewhere` (0.10);
      from the second: `distinctive_required` for a strong label. Regressions in
      `test_search_node_oracle.py` (+3), `test_rooms.py` (+1), `test_distinct_buildings.py` (+1).

## Open questions
- The looked-through far limit (2.5 m) is reasoned from the depth stride, not measured on a
  flight. A per-pixel "no return at the floor crossing" test would reach 5 m but is unsafe in
  Gibson's holed meshes (a ray through a wall hole settles the next room's floor); the 2-D cast
  through known free cells stops at the wall. Worth measuring on a recording with the map
  replayed step by step.
- `seen_through` by one pose that saw half the room shares the blind spot of `seen_from_scan`:
  the walls beside the door. Narrow rooms are the clear case; for a wide room looked into from
  its door this trades a scan for the chance of a TV beside the door.
- The fragment bound (3 m2) and the pocket bound (3 m2) are the same number for different
  reasons; a cup-sized target would want both smaller.
- The cue's gain bar (1.5 m2 for a non-home class) is set so a chair in a mapped room is not
  two turns; a vanity in a bathroom the map has barely seen clears it. Not measured on a flight.
- The boxed-in release leaves the plant where it is: the landmark node (`T<n>`) finds it again
  once the map connects, which depends on the footing sweep having mapped a way out.

## Notes
- 2026-10-05: the first `seen_through` was "stood in it and no frontier"; a wide room stepped
  into one pace facing in would have been finished with its door-side walls unseen. The narrow
  (walk-through) and looked-into (cone) halves are what remained.
- 2026-10-05: the `exits`-based "no live boundary" (openings >= 0.8 m) was replaced by the
  facts' accessible frontier count (clusters >= 4 cells): a toilet hidden behind a door jamb
  leaves a 6-10 cell frontier that is no opening but is a reason not to call the room done.
- 2026-10-05: with the ledger on, the two-room test world's patch by the door is a pocket --
  exactly the thing it was built to be a stand-in for. The opening tests turn the ledger off
  rather than lose the patch; `test_sightlines` owns the pocket.

## Result
Shipped as listed under Steps. CHANGELOG `[Unreleased]` 2026-10-05 entries; LESSONS
`2026-10-05 -- Hanson re-fly`.

**Second fly** (`runs/zson-hanson-3x-20261005-b`, same seed and models, 5 min for the three):
every episode ended in a STOP on a real instance of its category -- scored SR=0 by the dataset's
one annotated instance each (DTG 12.5-13.1 m), which is the annotation, not the search.

| episode | before (021) | now | what the record shows |
|---|---|---|---|
| toilet | 359 actions, the upstairs bathroom | **77**, STOP 0.92 m from the en-suite toilet | the en-suite the first fly walked past at 101 (the vanity cue); 1 cue glance (`bed`), no pocket or railing walked to; 24 of the 77 actions were the inspection ping-pong (fixed below) |
| chair | 14, false STOP on a ride-on horse | **106**, STOP 0.96 m from a desk chair | the horse reads `rocking horse 0.75` + `chair 0.51`; the rocking-horse landmark outvoted three later `chair 0.8` boxes on it (`map_rejections 3`); R4 `bathroom` (toilet x9, bathtub x3); R4-sized strip `scanned:fragment`; the chair was found by a cue glance (`chair` on the right edge at 80); 33 actions lost to a flaky far chair (12 of verification ping-pong, 9 of legacy pursuit after the release, 12 of a restarted warm-up) |
| potted plant | 161, agent error spinning in place | **101**, STOP 0.85 m from the plant | footing sweep at action 2, path at 5, the plant reached at 39; then "inspection saw nothing" (camera stuck at the sweep's 30 degrees), a re-lock at 75 and 24 actions of LOOK_UP/LOOK_DOWN before the exhaustion STOP |

**Cross-floor couch campaign** (`runs/zson-couch-crossfloor-3x-20261005` and `-b`; three
buildings, one couch episode each, starts on another storey than the annotated couch):

| episode | first fly | second fly (with `home_here`) |
|---|---|---|
| Ranchester (start upstairs, couch 18.6 m downstairs) | 500 actions upstairs: the model said "living rooms are downstairs" at all 16 loop points and 25 on every upstairs gap, the stairs (0.60) last in every order, 13 peeks, standing 0.9 m from the stairs at 171 | **SR 1, SPL 0.56, 290 actions**: stairs chosen at the first loop point after they were seen with `elsewhere` in force (208), down at 227-271, the couch locked 4 actions after arriving. 37 actions lost to one `missing` verdict (a cabinet+plant "living_room" upstairs) |
| Pomaria (start in the basement, couch a storey up) | (campaign aborted) | STOP at 9 actions on a `couch 0.94` 0.86 m ahead -- a real sofa in the basement library the annotation does not know (DTG 14.1 m) |
| Hanson (start upstairs, couch 7 m downstairs) | (aborted) | STOP at 58 actions on a `sofa 0.85-0.95` in a bedroom: a small upholstered settee with pillows; a detector call to review, not a search error |

The first campaign aborted after Ranchester with "Source changed during the frozen comparison": the
runner fingerprints every `.py` under `sparx_agency/`, and I edited the generator while it flew.

What the Hanson re-fly showed, fixed the same afternoon: the footing sweep's pitch stood for the whole approach
(owner `FOOTING` from action 5 to 63); `_pitch` forced 30 degrees down for every "potted plant";
the STOP test demanded the *predicted* pitch, so a fresh centred sighting at another pitch was
refused and the inspection alternated until its budget (both the toilet, 56 -> 76, and the plant,
76 -> 100); the verification centred a far box and lost it four cycles running; and the legacy
target hint was re-armed by the frame that had just been released.
