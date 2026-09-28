# 012 - ObjectNav: never spin on a failure, ground-truth stairs, YOLO-only default

**Branch:** `feat/objnav-habitat-gibson-nadav` (continues 011; the user decides on a split)
**Status:** in-progress
**Roadmap item:** [ObjectNav robustness](../ROADMAP.md)

## Goal

Three things the confirmation recording of the room-search loop
(`runs/room_search_loop_ranchester_20260928T093541Z_transitfix`, 500 actions,
`toilet` not found) showed were still wrong:

1. **A failed plan is a move, not a spin.** Whatever fails -- A* with no path,
   the RPT* instance, the room LLM, a bug in the decision -- the agent goes to
   the nearest reachable frontier anywhere on the floor (leaving the current
   room if that is where it is), then the stairs, then a retired frontier, then
   a relocation. An idle hold is issued only when the agent stands off the
   passable map.
2. **Stairs are ground truth, and the up/down decision is explicit.** The
   simulator's navmesh tells the policy where the staircases are and which
   storeys they join; a floor transition starts only at one of them; the choice
   of which one, and whether it goes up or down, is a readable rule with a
   recorded verdict per candidate. YOLO-based stair detection is deferred.
3. **YOLO only is the default detector.** Grounding DINO / BLIP-2 stay
   selectable, off by default, everywhere: service JSON, service CLI, every
   Gibson entrypoint.

## Why

- Steps 187-306 and 488-499 of that recording were `TRAVERSE ... acquire
  connected tread support`: turns in place, 120 actions at the real stair head
  (a 12 cm height wobble started an observed traversal, which then could not
  find treads from where it stood) and 12 at a raised bathroom floor 9 m from
  any stairs. The observed stair discovery mistook both for first treads.
- The loop's "nothing to work in" answer was `NavigationCommand.hold()` with no
  yaw -- one idle `TURN_LEFT` per action from the headless agent -- and a room
  LLM failure was an `ObjNavInternalError` that ended the episode as an agent
  error (three of the four earlier Ranchester attempts died that way).
- The hybrid VLM mode is 16 GB of weights on the CPU and its stair verification
  is the piece that is not ready; the run that matters now needs YOLO only.

## Steps

- [x] Evaluator: `gibson/stair_connectors.py` reads storeys and stair connectors
      from the navmesh once per scene (area-supported levels by the generator's
      own rule; off-level surface clusters as connectors with floor anchors and
      the navmesh shortest path); `HabitatRGBDSimulator.scene_structure()`;
      `GibsonEnv.reset` attaches the block to the episode metadata (no goal, no
      distance, no target floor; the harness's tripwire accepts it). Verified on
      the real Ranchester navmesh: two storeys, one connector, 5.5 m, the flight
      the recording climbed.
- [x] Policy: `MultiFloorParams.stair_source` (`ground_truth` default, `observed`
      kept); `stair_ground_truth.py` (connectors oriented per floor, XY distance
      to a flight); `floor_decision.py` (eligible → reachable now → worth it →
      unvisited first, nearest entry; one `floor_decision` event per distinct
      verdict); `ground_truth_traversal.py` (FOLLOW the polyline every action,
      exit stub past the far anchor, bounded retreat that never halts);
      `multifloor_policy.py` runs either source. Ground-truth mode: no look-down
      inspections, no depth veto on the committed route, a height departure off
      every connector is logged once and left to the atlas.
- [x] `exploration_fallback.py` + wiring: the loop's `_fallback` delegates to it;
      `_reason` turns a room-LLM failure into a recorded, backed-off
      `RoomReasoningUnavailable`; `RPTSearchPolicy._plan` wraps the decision and
      routes any exception to it; `perception_cycle.observe` gives the detector
      the same back-off. Failure records and stats in the episode record and
      `configuration()`.
- [x] Detector default: `DEFAULT_DETECTOR_BACKEND = "yolo_world"` in
      `detector_options.py` (all five entrypoints), `gibson_perception.json`
      `backend: yolo_world`; runbooks updated.
- [x] Tests: 9 (connector reading, orientation), 16 (decision, coordinator,
      traversal: bathroom floor, accidental climb, arrival cools the way back,
      exhausted floor takes the stairs, no inspections), 13 (fallback: LLM
      back-off and recovery, refused goal, dead planner, relocation, retired
      frontier, boxed in, decision exception, failing fallback, detector
      back-off, settings, record), 2 (metadata attach + tripwire, YOLO default
      everywhere). 264 pass across the runtime suite; observed-mode fixtures
      select `stair_source: "observed"` explicitly.
- [x] README (runtime, gibson, MULTISTORY, serve, GROUNDED_VLM), CHANGELOG,
      LESSONS, this entry.
- [x] `objnav_benchmark_runtime/QUICKSTART.md`: end-to-end setup + run order for the
      Gibson simulation, linked from the root README. Validated on this workstation:
      the YOLO service start (18095, `yolo_world`/cpu/26 classes/conf 0.05) and the
      `run_development --preflight-output` step passed as written against the frozen
      Ranchester manifest and the CPU Ollama. `provision_grounded_vlm --model none
      --include-yolo` added so YOLO-only weights can be provisioned without the VLMs.
- [ ] Re-run Ranchester 000000 with YOLO only and ground-truth stairs; read the
      `floor_decision` events and the fallback stats against this entry (user
      decides when; needs the detector service restarted on `yolo_world`).

## Open questions

- `near_connector_m` (0.75 m) and `departure_m` (0.45 m) decide when an
  unplanned height change counts as "on the stairs". Ranchester's flight is
  1.2 m wide; a wider or spiral flight may need the radius measured per scene
  from the connector's own sample spread rather than one constant.
- The decision reads "worth visiting" from the saved floor context's frontier
  counts. A storey visited before the map grew (frontier counts stale) is
  treated as finished; the portal cooldown bounds that, but a fresher signal
  would be the atlas floor's unknown-cell count.
- The fallback's relocation targets the farthest reachable cell. On a large
  swept floor that can be a long walk for one new vantage point; capping it
  at, say, 6 m geodesic would trade coverage for actions. Not measured yet.

## Notes

- 2026-09-28: Verified the extraction on the real Ranchester navmesh before
  writing the policy side: `build_navmesh_vertices` gives 599 triangles; the
  area-by-height histogram shows 62.9 m² at 0.0 and 44.2 m² at 2.6 with the
  stair triangles between (0.9-2.2 m); one XY cluster of off-level samples;
  floor anchors at (13.35, 1.56, 0.06) and (15.21, 1.77, 2.60) ENU;
  `find_path` between them returns the 7-point stair polyline. The recorded
  false traversal at step 187 was 0.24 m from that polyline (a real stair head
  with a 12 cm wobble, so the ground-truth rule would have completed the climb
  or ignored it, not scanned for treads); the one at step 488 was 9.3 m away.
- 2026-09-28: In ground-truth mode the atlas's `plateau_observer` (depth-measured
  flat support before a storey is confirmed) is not installed: the storey
  heights are known and the 1.5 m separation rule alone keeps landings from
  becoming floors. The observed mode keeps it.
- 2026-09-28: The FALCON orchestration rig monkeypatches `mapping.update`, so
  the atlas has no floors when discovery first runs; ground-truth discovery now
  reads the pose height in that case, as the observed path always did.

## Result

Implemented; unit-tested; not yet flown. The connector reading is validated
against the real scene. The Ranchester re-run is the next step.

