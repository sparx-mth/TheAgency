# 024 - Allensville audit: table objects, room numbering, the bathtub prior, the low-target lock

**Branch:** `feat/objnav-habitat-gibson-nadav` (a `fix/` branch opened in the morning was deleted unpushed at the user's request)
**Status:** done -- code, 1975 regressions (1891 before), docs, one verification episode flown (Allensville/000000 toilet: SR 1, SPL 0.791, DTG 0.00 m, 33 actions; the benchmark's run of the same episode: SR 1, SPL 0.148, 299 actions)
**Roadmap item:** ObjectNav robustness (012) and exploration efficiency (010); follows 023

## The ask
Three recordings of the 5x3 benchmark (`runs/zson-benchmark-5x3-20261005/campaign/frontier/
Allensville/recordings/`) were reviewed frame by frame:

- `1e3ec4018b4a` (Allensville/1, couch): a table with several visible items in frames 1-2 and
  only the cabinet registered; the room graph showing R0 alone from step 10 to 43, then R0 and
  R2 (no R1), the partition "kicking in" at 80-89.
- `807d9d0107aa` (Allensville/2, bed): a kitchen island read as `bed` and STOPped on.
- `b536f3c4fe05` (Allensville/0, toilet): a strong bathroom (bathtub) valued 0 for the toilet at
  step 26; a 0.95-0.97 toilet in plain view at 2.4 m ignored around step 50; the approach at
  ~85 turning in place, losing the lock and going back to exploration.

Fix the perception filters, the topological mapper, the prior table and the closing state
machine, then fly one verification episode and report SR / SPL / DTG with the HUD video.

## What the traces actually show (read before changing anything)
- **Table items.** Step 0 of the couch run: `cabinet 0.65` fused; `vase 0.48` and `vase 0.40`
  projected with valid depth and were dropped as `same_frame_association`; `cup 0.34` fell under
  the 0.35 confidence floor. The landmark map associates ANY class within 0.70 m of a landmark's
  centroid, and `fuse()` re-matched the already-fed cabinet without the height check. The same
  rule folded a toilet (4 votes) and the bathtub beside it into one landmark in the toilet run,
  which flipped to `bathtub` and released the lock as "contradicted by the map" at step 9 with a
  2 m rejection radius -- the reason the 0.95 toilet at 2.39 m was refused at steps 45-53.
- **Room numbers.** A door 0.5 m from the spawn point split the spawn room off as R1 at step 8;
  the pid R0 followed the LARGER half (the IoU rule), so the spawn room became R1 and then merged
  back at step 10; when it re-split at step 44, 34 ticks later, the registry's 10-tick memory had
  expired and it was numbered R2 while R0 followed the agent into the hallway. The whole route to
  step 62 is a 0.8 m hallway (clearance 0.30-0.50 m on the saved map) with no clearance dip at
  the step-40 "doorway", so the watershed could not have split it earlier; the bedroom, kitchen
  and living room appear at step 80 when their floor is first seen and are labelled at the loop
  point (84-89).
- **Kitchen island.** `bed 0.53-0.75` at 0.55-0.59 m, supported points 0.67-0.70 m above the
  agent base (real beds in the same run: 0.11-0.52 m), inside a STRONG `kitchen` (refrigerator,
  oven); four unverified takeovers from one spot (31/47/63/79), then two frames from one spot at
  113-114 -> STOP, DTG 2.18 m.
- **Toilet VERIFY.** Within 2 m a toilet's box always touches the BOTTOM image edge (y2 = 477-480);
  `clipped_box` called that a sliver (`border_rejections` 3 -> 26), no fresh frame counted, the
  agent swept left/right around the anchor for 12 actions and released `unverified`, twice (54-66,
  75-87). At 1.18 m a centred, counted frame was followed by TURN_LEFT because a satisfied hold
  is idle and the headless agent's idle action is a turn. Step 26's 0 % for R1 was not the type
  prior: the warm-up spin had "seen" >50 % of the bathroom's cells through its door
  (`scanned:seen_from_scan`) and finished it with the bathtub inside and the toilet unseen.

## Steps
- [x] Landmark association: cross-class votes need footprint overlap, not a 0.70 m centroid
  radius (vase on a cabinet, toilet beside a bathtub); the same-frame rule is height-aware.
- [x] `clipped_box`: a box cut only by the bottom edge, standing in the lower half of the frame,
  is the camera's floor cutoff, not a sliver.
- [x] Detection confidence floor 0.35 -> 0.30 (the takeover's own tracking threshold).
- [x] Room registry: a room keeps its number through a split by its birth anchor, not by which
  half is larger; vanished rooms stay re-adoptable for the episode (bounded count).
- [x] Scene graph: a clearance dip the agent walked through is carved like a snapped door
  (severing chokes only, never a plain disk), so a room is instantiated at the threshold.
- [x] Priors: a room holding a home object of the target is never finished by sight from outside;
  such a node is read at a probability floor (`home_floor`); `bathtub` already names a bathroom.
- [x] Closing: `override_confidence` (0.80) starts a takeover past the unverified/map-contradicted
  rejection memory and the cooldown; a context-suspect candidate (implausible strong room type or
  class-height conflict) is penalised and needs four frames from two viewpoints; VERIFY keeps
  the box whole (LOOK_DOWN for a low target, else the view that keeps it in frame) instead of an
  idle turn; `look_down_distance_m` 1.30.
- [x] Regressions for each; the existing suites green.
- [x] README / CHANGELOG / LESSONS.
- [x] One verification episode: Allensville/000000 (toilet), HUD video, SR / SPL / DTG.
- [x] Follow-up (same day): classes never merge in the landmark map (no class vote), same-class
  instances told apart by size (0.35 m / IoU 0.25); the verification LOOK and the close approach's
  pitch follow the target's elevation (LOOK_UP above the camera, LOOK_DOWN below), no fixed look-down.

## Open questions
- The trail-dip doorway cut is a heuristic over observed geometry; a furniture gap the agent
  walks through with open floor on both sides would be taken for a doorway. It did not fire in the
  verification episode (the takeover owned every action) and has only synthetic coverage so far.
- The height bands of the context check are calibrated on three Allensville recordings relative
  to the agent base (the navmesh height); another building's navmesh offset could shift them.

## Notes
- 2026-10-07: entry opened from the trace review above.
- 2026-10-07: the closing tests used `.9` as their generic "confident" box, which the 0.80 override
  now carries past the memory those tests exercise; the memory/cooldown fixtures read `.7`, and the
  two synthetic toilets at camera height (a height conflict) run with `context_check=False`. The
  map-outvote fixture's `bed` votes get the chair's own footprint: under the new rule a bed four
  times the chair's footprint around it is a second object.
- 2026-10-07: this machine has no `qwen2.5:*-instruct` in Ollama and `.venv/bin/python` has no
  pytest; the suites ran under the shell's `venv/bin/python`, and the episode under
  `objnav-habitat` with `LLM_MODEL=llama3.2:3b`, `LLM_REASONING_MODEL=llama3.1:8b` (the user's
  choice over an 11 GB pull) -- neither model was called: `room_llm: 0, room_classifier: 0, rpt: 0`.
- 2026-10-07 (later): the user asked for a strict cross-class fusion ban and a dynamic pitch. The
  morning's refinement of the class vote (footprint-only across classes) was replaced by no
  cross-class association at all; the vote stays in the library behind `class_votes=True`, so the
  map-release path (`contradicted_by_map`) is dead in this runtime and the three closing tests that
  asserted it now assert the opposite (a second instance, the lock stands). The fix branch opened in
  the morning was deleted unpushed; everything is committed on `feat/objnav-habitat-gibson-nadav`.

## Result
`runs/allensville-audit-verify-20261007/` -- `episodes.json` (the 5x3 manifest with paths rewritten to
this machine; identities unchanged), `preflight.json`, `episode/` (the run; `completion.jsonl`,
`metrics.csv`, `recordings/b536f3c4fe05/{video.mp4,steps.jsonl,trajectory.csv}`), `episode.log`,
`vocabulary.txt`, `source-commit.txt`, `source-changes.diff`.

**Allensville/000000 (toilet): SR 1, SPL 0.791, DTG 0.00 m, 33 actions, 14.4 s.** The trace: the
toilet locked at step 1 from 5.4 m as before; at step 3 the bathtub beside it became its OWN landmark
(24 votes to the toilet's 21 at the end; `map_releases 0`, `releases 0`, nothing in the rejection
memory) where the benchmark's run had voted it into the toilet's landmark and released the lock as
"contradicted by the map" at step 9; the lock held through a seven-action occlusion (CLOSE_OCCLUDED
4-10), the approach counted every bottom-cut frame at 1.4-1.7 m (`border_rejections 0`), and the
end was CLOSE at 1.05 m -> INSPECT LOOK_DOWN -> one centring turn -> STOP at 0.61 m on a fresh 0.97
sighting. The benchmark's run of the same episode took 299 actions (SPL 0.148). Video: H.264,
1600x900, 35 frames at 6 fps. The vase (2 votes), the potted plant (6) and the cabinet (10) beside the
bathroom all registered as separate landmarks.

Regressions: 1975 pass (`objnav_benchmark_runtime/tests`, `core/mapping/topology/tests`,
`core/mapping/objects/tests`, `core/planning/exploration`, `core/planning/objnav`); new:
`tests/test_perception_association.py` (6), `tests/test_target_context.py` (11), plus anchor/memory,
threshold, home-floor, finish-guard, class-ban and size-rule cases in the registry, watershed, oracle,
scan and landmark suites. CHANGELOG `[Unreleased]` (Added / Fixed, 2026-10-07) and the LESSONS entry
of the same date.

**Re-flown under the fusion ban and the dynamic pitch** (`episode-fusion-ban-dynamic-pitch/`):
**SR 1, SPL 0.828, DTG 0.00 m, 32 actions, 13.2 s.** At step 27 (1.19 m, inside the 1.30 m band) the
approach requested +30 degrees because the toilet's centroid stands 0.7 m below the camera -- the box
went from bottom-cut (y2 = 480) to whole (112-388) -- the approach continued pitched and STOPped at
0.63 m on a fresh 0.96 sighting; `map_releases 0`, `releases 0`, nothing in the rejection memory. The
map ended with twelve landmarks, every class its own: toilet, bathtub, cabinet, vase, potted plant beside
one another in the bathroom. The stricter same-class rule split the toilet (15 + 4 observations, 0.4 m
apart) and the bathtub (22 + 1) into two instances each -- the expected cost of a 0.35 m floor on an
object whose measured centroid walks with the viewing angle; the closing's own association radius
(0.50 m) is unaffected and the lock never wavered.

