# Changelog

All notable changes to this project are logged here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/) — one line per change, written for
future-you, not for a commit log.

## [Unreleased]
### Added (2026-11-22, the frame log: every RGB frame of an episode kept under its capture pose -- the first piece of the VLM pipeline, `core/planning/object_search/frame_log`)
- **What it keeps.** `EpisodeFrameLog` holds every RGB frame the room-search policy decides on, named after its step
  (`step_000137.jpg`), and a lookup dictionary from a frame's name to its `FrameRecord`: the step and the exact
  capture pose (the observation's `AgentPose` -- x, y, z, yaw, camera pitch, the frame of the map and of the RPT\*
  node positions). Where the images live is a `FrameStore`: `DiskFrameStore` (one folder per episode, the images
  beside `frames.jsonl` and `episode.json`; `read_frame_log` reads a folder back) or `MemoryFrameStore` (RAM, dropped
  per episode). A frame is encoded once, above the store (`FrameCodec`: JPEG quality 95 by default, PNG for
  lossless), so both stores hold byte-identical images and swapping them changes nothing a reader of the log sees.
- **Where it attaches.** `RPTSearchPolicy(..., frame_log=None)`: begun at every reset and fed at the top of every
  decision, before perception. Without one the policy, `RPTSettings` and `configuration()` are what they were.
  `OBJNAV_FRAME_LOG_DIR=<dir>` turns the disk log on for every dataset entrypoint through the one policy builder
  (`policy_factory.build_policy`) and is inherited by a campaign's children; the frozen configuration gains
  `method.frame_log` only then, so recorded runs resume as before; `--agent falcon-yolo` and `stop` refuse the
  variable instead of ignoring it. N actions log N frames: the frame the last action returns is shown to no policy.
- **How it fails.** `FrameLogError` is an `ObjNavInternalError`: a frame that cannot be encoded or stored stops the
  run instead of being scored as a failed episode, and no record exists without its image. An episode logged again
  (the one a resumed run died in) replaces its own folder; finished episodes are not touched.
- **Checked.** 85 new tests: the codec, both stores against one contract and the episode log
  (`frame_log/tests`, 71), the policy feeding it (`tests/test_policy_frame_log.py`, 6), and the policy builder, a
  whole run through the real runner on the harness's fake building and the failure's attribution
  (`objnav_benchmark_runtime/tests/test_frame_log_run.py`, 8); twelve planted faults (no feed, a channel swap, an
  unchecked step order, stale frames kept, a rounded pose, ...) each fail at least one of them. One real MP3D episode
  (`pLe4wQe7qrG/000000`, software rendering, the stand-in LLM, ground-truth perception), flown with the log and
  without: 243 actions, 243 frames, every pose equal to the recorder's `trajectory.csv`, and the same actions, poses
  and row either way -- the frozen configurations differ in `method.frame_log` alone; 111 kB a frame at the median,
  27 MB for the episode. The existing suites are as they were (object_search 234, objnav and harness 1,287, runtime
  rig 1,368 passed), one failure included that this change does not touch:
  `test_doorway_actions::test_peek_evidence_flows_through_classifier_oracle_and_real_rpt_star`.
- **Not done, on purpose.** Nothing reads the log during an episode yet. Choosing the frames that stand for a
  frontier cluster, the VLM client and RPT\* over frontier clusters are the next steps
  (`docs/papers/vlm-search-scoring-20261119`).

### Changed (2026-11-18, the paper's draft cut to a submission draft: `docs/paper/zson_paper.tex`)
- **From an audit of the code to a paper** (34 -> 28 pages, pseudocode and figure included; pdfLaTeX of TeX Live 2026:
  no error, no warning, no over- or underfull box; every reference cited). The 32 `\TODO` notes are gone, with the
  placeholder macros. Removed: the section "Which code produced the numbers" (360 lines), the development-history
  remarks inside the Method ("earlier revisions ...", the "located sides" paragraph), the implementation map and the
  run-records appendix. What they said about the reported runs is now an asterisk on the numbers (`\oldrev`), one
  footnote in the abstract and a short subsection "Checks of the current code" (6.3) that keeps the stop-criterion
  table.
- **The numbers are still the two 14B full-split runs, marked**: HM3D SR 71.0 / SPL 39.8, MP3D SR 65.6 / SPL 29.1
  (entries 047, 049), start-storey SR 83.5 / 77.4. Section 6.3 names what changed since and what was checked: closing
  on 180 MP3D episodes (77.8 -> 78.9 % without the navmesh query, 78.9 -> 82.2 % with the stop criterion), the last
  prompt revision on 90 HM3D episodes (86.7 -> 87.8 %), the smoke of 2026-11-17 end to end (16 of 20 against 15) --
  and that the removal of the priors table and of the prescriptive prompt was not measured in isolation.
- **Method statements corrected against the code** (each read in the module): a refused step does NOT end the steps
  toward a depth-less candidate (the withdrawn fix was still described; up to 24 such steps are outside the 12-action
  budget); a landmark is retracted after `max(4, observations)` clear misses; a doorway needs three observations and
  one view 0.2 m or 20 degrees from the first, and is not used "only" for the partition (scrape guard, exits, room
  facts); the door cut goes to the NARROWEST choke; thresholds are also read off sight lines to exits; floor glimpsed
  beyond a forced cut is a room from four cells; a query is deferred while a node is in force; on timeout RPT* returns
  its incumbent, and a triangle-inequality violation falls to the random draw; a lattice step costs the cost map's
  cells, a turn one unit; routes are simplified (Douglas-Peucker), not string-pulled; a refused step is a lattice
  pose AND a map write unless a scrape, and a scrape needs a confirmed doorway within 40 degrees of its axis;
  finished-room tests (c) and (d); which STOPs are taken without a fresh box; the walk's end conditions; what a
  release remembers; the stair-start switch after 220 actions; the 0.45 m line for crossed cells; adopted split
  levels; depth out of range is discarded, not clipped.
- **Filled from the records instead of deleted**: RPT* proven optimal in 99.3 % of the 8,657 solver calls of the HM3D
  run (the other 0.7 % violated the triangle inequality; at most 11 ms and 11 nodes); the adapter's start-to-goal
  geodesic distances within 1 mm of the published ones on all MP3D and 1,992 of 2,000 HM3D episodes; HM3D SoftSPL by
  storey 48.1 / 7.5, and overall 41.0 (41.1 was a double rounding of 0.41045).
- **Related work and references**: four systems that also plan a visiting sequence are cited beside the novelty
  claim (WTRP-Searcher, BeliefMapNav, SSMG-Nav, SEEK); Gong et al. is the RA-L 2026 article (title and authors
  changed), RPT* the T-RO early-access article, InstructNav PMLR 270 (2025). Entries checked against Crossref, arXiv,
  PMLR and OpenReview.
- **Pseudocode and figure.** `algorithm2e` is replaced by `algorithm` + `algpseudocode`, in the style of the example
  the paper's owner gave: numbered lines, procedure calls in small capitals, short comments at the right, `end`
  lines shown. Algorithm 1 is the episode loop (per step, as `RPTSearchPolicy._plan` / `_search` and
  `RoomSearchLoop.plan` decide it), Algorithm 2 `UpdateSceneGraph` (map, landmarks, doorways, rooms), Algorithm 3
  `RoomProbabilities` (the reply's checks and the adjustment by exploration state, which replace two paragraphs and
  Eq. 3) and Algorithm 4 `TargetClosing`. A*, RPT*, the watershed, the detector and the LLM are single calls;
  Algorithm 1's caption names the rules it leaves out. The overview figure is redrawn in TikZ: perception and map,
  room scene graph, the two LLM queries, RPT*, node visit, target closing, route planning and execution, with the
  two feedback loops. The target category is `c*` (`c(u,v)` is the cost) and the actions spent inside a room are
  `n_v` (`tau_v` is the search time of Eq. 2).
- **The pseudocode was read line by line against the code** and corrected where it differed: room labels and
  probabilities are asked under one condition (not after an unreachable release); the candidates are the rooms not
  finished or re-offered; a transit ends on arrival, not on standing inside the room; the exits are read by the loop
  after the warm-up, not by the graph update; the rooms are rebuilt on fewer actions only during a committed walk.
  In the closing: the lock and the verification's release are tested before the verify action; a candidate out of
  view for 6 sweep actions is released; the inspection begins only if no fresh box reads farther; the STOP on a
  fresh box needs it aligned (centred, or tracked over the approach); an inspection that tried every view without a
  box releases. Two rules that end an episode outside the closing are now stated: the walk to a target landmark on
  which no takeover started (STOP within 0.75 m after three separated views; 1 of 1,345 STOPs and 40 of the first
  1,810 episodes of the 3B run in progress) and the stair walk's halt after a failed retreat (4 episodes there).
- **After a review of the revision**: the footnote claims "a similar success rate", not comparable results; "earlier
  revisions" (the two runs used different ones); the 90-episode check is said to compare two revisions that both
  lacked the priors and the prescriptive prompt; the old prompt and priors table are described as they were; the
  0.8 m stair-start band is listed among the looks at validation data; the labeller prompt's hand-written hints are
  listed as hand-designed; the launcher, not the library, defaults to the 3B.
- Open for the authors: the method's name (`\ours`, still the working name LGOS), the author block (`Anonymous
  Authors`) and acknowledgements. `sync_prompts.py --check` passes; its docstring names Appendix B.
### Added (2026-11-17, the 3B-then-14B HM3D v1 GT comparison as one resumable command -- hm3d/README.md section 3.6)
- **`hm3d/run_gt_3b_then_14b.sh`**: two full ground-truth runs of the v1 val split in sequence, `qwen2.5:3b-instruct` then
  `qwen2.5:14b-instruct` in both LLM roles under the same prompt version `facts-qwen2.5-3b-v1`; checks the branch and
  commit, the models in Ollama (pulls them), the preflight's prompt digest and roles, the services and the first
  `run.json`; resumes by re-run; fixed parallelism (20 workers / 10 slots, 8 / 4) instead of `--jobs max`; and after each
  stage writes, commits and pushes `hm3d/results/<run name>/` (never force; a failed push is logged and the next stage runs).
- **`hm3d/results_summary.py`** and the `hm3d/results/` convention (as `gibson/results/`): SR and SPL overall and per
  category with intervals, episodes, the LLM per role, the prompt version and digest and whether it was tuned on the model
  that answered, the source commit and tree state, perception, start and end times; `episodes.csv`, `run.json`,
  `campaign_summary.json` beside it.
- **`hm3d.run --record-log-only`** and `hm3d/run_benchmark.sh --record-log-only`, MP3D's flag ported: every episode's
  `steps.jsonl`, `trajectory.csv` and map journal, no video and no FFmpeg (~9 MB an episode on HM3D).
- Smoke of 2026-11-17 (one episode per scene per stage, committed as `hm3d/results/hm3d_v1_gt_3b_vs_14b_smoke_*`): the 3B
  at 20 workers takes VRAM 81 %, CPU up to 63 %, RAM 37 %, its oracle calls 25 s each (Ollama's one sampling thread shared
  by 20 workers); the 14B at 9 workers reached 91 % of VRAM, so the full run uses 8.
### Changed (2026-11-16, evening: the STOP is taken inside each benchmark's own success zone; the blind-step fix and the 0.20 m retry clearance are withdrawn; the launcher's oracle model is the 3B)

Five items asked for after the day's first check (the entry below, which this one revises): bring back the
approach before STOP, make the stop distance match each benchmark's success definition exactly, restore the
previous blind-step behaviour, choose the retry clearance on the measurements in hand, and make the 3B the
launcher's oracle default. Measured on the same 180 MP3D episodes with the same stand-in for the language model:
**SR 74.4 % -> 77.8 % with the approach back at the radius the method already had, a planar metre (item 1: 6
episodes won, none lost) -> 82.2 % with the radius and the measure of MP3D's own definition, 0.9 m in 3-D from the
floor (item 2: 8 more won, none lost); SPL 0.332 -> 0.341 -> 0.354.** That is 6 won and none lost against this
morning's "before" (78.9 % / 0.346), whose rules it replaces. HM3D and Gibson were not measured.

#### 1. The approach before STOP is back, as a stop criterion the benchmark's adapter states

- **Deciding to stop and where to stand are two things.** The closing DECIDES on its own evidence, as before: a
  fresh, whole, confident box of the target inside the terminal range (1.05 m). How near the target it must then
  STAND is not the method's to choose -- a benchmark defines a success zone round each goal object -- so it is a
  criterion of two settings, `TargetClosingSettings.stop_within_m` and `stop_measure`, read by one pure function
  (`core/planning/object_search/closing/stop_criterion.py`, new): `planar` is the horizontal range to the target's
  nearest measured point; `euclidean` is that range and the height of the target's lowest measured point above the
  floor under the agent, as the two legs of a triangle (an underside under `stop_elevation_min_m`, 0.30 m, is an
  object on the floor). Both are the agent's own depth measurements, and both err on the safe side: the reading
  is never shorter than the true distance to the object's box (but for that 0.30 m threshold, worth under 5 cm).
- **The walk.** When the STOP is decided and the range reads beyond the distance, the closing walks straight on
  toward the 3-D memory (phase `ADVANCE`) and STOPs when the range is within it. The range is the deciding
  frame's, carried forward by the agent's own displacement; no frame is consulted meanwhile (under a raised target
  the box leaves the top of the frame). It also ends, with the STOP, at a refused step, after four steps or seven
  actions, when another step would leave the memory under 0.30 m or the measured near edge inside the body radius,
  when the memory stands more than a quarter turn off, or on the last action the episode's budget holds (a decided
  STOP is always called). A target whose underside is raised by the stop distance
  or more is within it of no place on the floor and is stood under as closely as that allows
  (`stop_approach_shortfalls`). The record carries `stop_decision` (what BOTH measures read on the deciding frame,
  whatever the criterion), `stop_approach` (where the walk ended and what the range read there) and the counters
  `stop_approaches`, `stop_approach_steps`, `stop_approach_shortfalls`; `mp3d/stuck_report.py` tables them.
- **The method's own default is no criterion** (`stop_within_m: 0`): a deployment that names no success zone STOPs
  on the deciding frame. The benchmark entry points set it below `--policy-config`, so
  `{"target_closing": {"stop_within_m": 0}}` is the arm without it. The name is not `stop_distance_m`:
  `RPTSettings.stop_distance_m` is the legacy pursuit's standoff, which the closing never reads.
- **What is not back.** The unconditional step after every STOP (Gibson's footprint rule of 2026-10-14) and the
  switch by which evaluator scores the run: the criterion steps only while the range reads beyond the distance.
  The frontal standoff of raised planar targets: it was not asked for, and the Euclidean measure addresses the same
  episodes from the definition instead of from recorded view points (measured below: `f257a0ce`'s steps WITHOUT
  its frontal standoff score one episode more than with it).
- **Where the line is.** What reaches the method is one distance and one word per dataset, the same for every
  episode and object. What does not: any view point's position, any goal's position, any distance to them, at
  run time. `tests/test_agent_gets_no_scene_geometry.py` and the episode contract are untouched.

#### 2. The stop distance matches each benchmark's success definition

- **habitat-lab's definition, for HM3D and MP3D alike:** STOP called, and the geodesic distance from where it was
  called to the nearest *view point* of any goal object of the category under `SUCCESS_DISTANCE` = 0.1 m
  (`DISTANCE_TO: VIEW_POINTS`). The challenge text says "within 1.0m Euclidean distance from any instance ... AND
  the object can be viewed by an oracle"; what is scored is the view points, and they are published with the
  episodes, so the geometry of "success" is whatever each dataset's generator did.
- **The generator** (habitat-lab's ObjectNav dataset code as kept in `Ram81/habitat-imitation-baselines`,
  `object_nav_generator.build_goal`, and for the HM3D lineage `naokiyokoyama/ovon`,
  `ObjectGoalGenerator._make_object_viewpoints`): candidates on a horizontal grid round the object's axis-aligned
  box, half the agent's radius apart, out to `max_distance` = 1.0 m; kept when within `max_distance` of the box,
  navigable, and showing the object's pixels in frames turned toward it. The grid was confirmed on the published
  episodes: the nearest neighbour of a view point is 0.090 m away on HM3D v1 and MP3D v1 and 0.085 m on HM3D v2.
- **Which distance "within 1.0 m of the box" is differs between the datasets, and it was measured, not assumed**
  (every validation goal's view points against the object's box as habitat-sim loads it):

  | dataset | goals, view points | farthest view point from the box, in the plane | in 3-D from the floor point | so the metre is |
  |---|---|---|---|---|
  | HM3D v1 | 432 goals of 20 scenes, 111,406 | 1.00 m at every object height (none beyond 1.01 m) | up to 1.3-1.8 m for a mounted television or a raised plant | **planar** |
  | HM3D v2 | 417 goals annotated as in v1 | 1.00 m | -- | **planar** |
  | MP3D v1 | 1,316 goals of 11 scenes, 291,810 | by underside height: 1.00 m on the floor, 0.96 m at 0.1-0.3 m, 0.89 m at 0.3-0.6 m, 0.74 m at 0.6-0.9 m, 0.57 m at 0.9-1.0 m (medians over goals) | 1.00 m throughout (1.11 m for the 23 goals over 0.9 m up); no goal's underside is 1.0 m up | **Euclidean from the floor** |

  The code that survives tests in the plane; the 2020 MP3D dataset evidently was not made with it. About 5 % of
  MP3D's view points stand beyond 1.0 m in the plane, where the generator moved them onto the navigation mesh.
- **How the other published zero-shot methods stop** (their released code), all three by a PLANAR range to the
  nearest point of the target's point cloud, on both datasets: **VLFM** STOPs when it is under 0.9 m
  (`pointnav_stop_radius`); **ASCENT** steps straight on inside 1.0 m and STOPs at 0.6 m, or where a step no
  longer brings it 0.1 m nearer; **ApexNav** plans to the free cell nearest a point 0.5 m from the object (0.70 m,
  then 0.85 m, when that has no route), STOPs within 0.2 m of it (0.45 m when stuck), and its evaluation configs
  set `success_distance: 0.2` for both datasets. None measures in 3-D: on MP3D a planar 0.9 m STOP is outside the
  zone of any goal raised more than about 0.4 m.
- **In each adapter.** `hm3d/protocol.py` and `mp3d/protocol.py` state their dataset's zone as a constant
  (`GOAL_ZONES`: `GoalZone(distance_m, measure, grid_m)`, `habitat/goal_zone.py`, new) and expose it as
  `protocol.goal_zone` -- a property, so the frozen protocol record and the manifests are unchanged.
  `habitat/entrypoint.py::method_defaults` turns it into the criterion with `goal_zone.stop_criterion`: the zone's
  measure, and its distance less `STOP_MARGIN_M` = 0.1 m. **HM3D: `planar`, 0.9 m. MP3D: `euclidean`, 0.9 m.**
  Gibson (`gibson/run.py::method_defaults`): `planar`, 0.9 m -- SemExp's protocol counts the agent inside the goal
  footprint dilated by 1.0 m on the 5 cm map, in the plane, and the STOP is taken two map cells inside it.
- **Why a decimetre inside the zone.** A STOP exactly at the edge is within 0.1 m of a view point only if the
  grid happens to have a candidate there. No place is farther than half the grid's diagonal (0.064 m) from a
  candidate, so a STOP 0.9 m out has a candidate INSIDE the zone within the success distance whichever way the
  grid lies (`test_goal_zone.py` checks it on the geometry, for boxes at any height and grids at any offset); the
  rest of the decimetre is for what the argument leaves out -- a candidate is a view point only if it is navigable
  and sees the object, and the agent's range is a measurement. 0.9 m is also VLFM's number. The protocol's success
  distance is checked against the grid, not subtracted: a run scored at ApexNav's 0.2 m stops where the official
  one does.
- **Disclosed.** The two zones were established on validation goals, with the scene annotations' object boxes --
  only validation scenes are on this machine. That is a use of evaluation data and annotations in development; the
  paper says so (section 5.5). Nothing of it is read at run time.
- **Not measured: HM3D and Gibson.** The check is MP3D's. On HM3D the only evidence is older and about another
  rule (an unconditional step after the STOP never helped on 70 recorded STOPs and walked one out of the zone);
  the criterion steps only from beyond 0.9 m. On Gibson the criterion replaces "one step always, a second beyond
  0.9 m" of the runs recorded there. Both are flagged in their documents.

#### 3. The blind verification-step loop: the previous behaviour, restored

- `TargetClosing._glimpse_stalled`, the counter `glimpse_stalls`, the pursuit's `GLIMPSE` source and their four
  tests are out; `target_closing.py` and `target_pursuit.py` verify a glimpsed candidate exactly as at `f257a0ce`.
  The loop is a known, open fault, noted at `max_verify_approach_steps` and in `LESSONS.md` with its cause and
  what the withdrawn fix measured (five episodes changed: three still succeeded, two were lost).

#### 4. The standing-point retry clearance: the body radius, 0.18 m

- **Chosen on the checks already run:** the two arms that differ in this number alone gave SR 73.3 % / SPL 0.325
  at 0.18 m against 71.1 % / 0.322 at 0.20 m (2 won, 6 lost, exact McNemar p = 0.29); the price is refused forward
  steps, 6.79 an episode against 5.92. `standing_retry_clearance_m` defaults to `None`, the policy's own body
  radius (`RPTSearchPolicy.standing_retry_clearance_m()`), and a number between the body radius and
  `min_clearance_m` selects another rung (`0.20` is the one measured). The retry planner stays the closing's own
  (`policy.standing_retry_planner`, counter `retry_standoffs`); at the body radius it plans exactly what the
  fallback's squeeze planner planned for it before today.

#### 5. The launcher's oracle default is the 3B

- `benchmark_services.sh`: `LLM_REASONING_MODEL` defaults to `qwen2.5:3b-instruct` (was `qwen2.5:14b-instruct`),
  so both LLM roles run on the model the laptop holds unless the environment names another;
  `tests/test_launcher_defaults.py` pins it. QUICKSTART, `hm3d/README.md`, `gibson/BENCHMARK.md` and the runtime
  README say so and name the 14B as the model of the runs published before today. **Not changed:**
  `LLMConfig.from_env` and `gpu_plan.plan_from_environment` still fall back to the 14B when the variable is unset
  -- code that builds the client without the launcher gets it; every launcher exports the variable.

#### The check

- **The arms**, all on the 180 episodes of `sets/single_storey_flat_v1.json` with the launcher, the flags and the
  stand-in of the entry below (ground-truth perception, the frontier explorer, seed 17, CPU rendering, every room
  `living_room`, the nodes an equal share). The base is that entry's arm A2 -- the rules removed, STOP on the
  deciding frame -- which this tree reproduces when the criterion is off (the controls, below):

  | per episode (180) | SR (successes) | SPL | won / lost against the base (exact McNemar p) | failed STOPs of 158 |
  |---|---|---|---|---|
  | *base*: the rules removed, STOP on the deciding frame (A2) | 74.4 % (134) | 0.332 | -- | 24 |
  | *the approach as it was*: `f257a0ce`'s steps under a raised target, its frontal standoff off | 79.4 % (143) | 0.348 | 9 / 0 (0.004) | 15 |
  | *this morning's "before"*: those steps and the frontal standoff (`f257a0ce`) | 78.9 % (142) | 0.346 | 8 / 0 (0.008) | 16 |
  | **item 1**: the approach back, to the radius the method had: `planar` 1.0 m | 77.8 % (140) | 0.341 | 6 / 0 (0.03) | 18 |
  | item 2, the margin alone: `planar` 0.9 m (VLFM's rule; what HM3D's definition gives) | 80.6 % (145) | 0.350 | 11 / 0 (0.001) | 13 |
  | item 2, the measure alone: `euclidean` 1.0 m | 80.6 % (145) | 0.352 | 11 / 0 (0.001) | 13 |
  | **item 2**: MP3D's own definition, `euclidean` 0.9 m -- the adapter's default | **82.2 % (148)** | **0.354** | 14 / 0 (0.0001) | 10 |

- **Item 1 alone** (base -> `planar` 1.0 m): 6 won, none lost (p = 0.03), SPL +0.009; 145 trajectories identical.
  The STOP is decided inside the terminal range and its tolerance (1.05 m); 35 STOPs were decided between 1.00 and
  1.05 m in the plane and took one step toward the target (three ended short of the metre). The six that turned
  were decided at 1.006-1.041 m and had STOPped
  0.11-0.26 m from the nearest view point: a stool, two tables, a sink, a chair and a picture -- stops a few
  centimetres outside the metre, with the target in view.
- **Item 2 alone** (`planar` 1.0 m -> `euclidean` 0.9 m): 8 won, none lost (p = 0.008), SPL +0.014; 89
  trajectories identical. It is two changes, and all four arms of the two were run:

  | successes of 180 | at the zone's edge, 1.0 m | a decimetre inside, 0.9 m | the margin |
  |---|---|---|---|
  | `planar` | 140 | 145 | 5 won, 0 lost (p = 0.06) |
  | `euclidean` | 145 | 148 | 3 won, 0 lost (p = 0.25) |
  | the measure | 5 won, 0 lost (p = 0.06) | 3 won, 0 lost (p = 0.25) | |

  *The margin* (planar, 1.0 -> 0.9 m): a cabinet, a towel, a table and two pictures decided at 0.92-0.99 m in the
  plane -- within the metre -- and 0.14-0.32 m from the nearest view point. Stopping at the zone's edge is not
  stopping inside it; on MP3D's own measure the same step is worth three (two of them STOPs 0.99 m from an object
  on the floor). *The measure* (planar -> Euclidean, 0.9 m): three pictures whose planar range ALREADY read inside
  0.9 m when the STOP was decided (0.88, 0.90 and 0.49 m) while their distance in 3-D from the floor did not (1.31,
  1.35 and 0.95 m; undersides 0.82-1.01 m up): under the planar measure no step is taken and the STOP fails
  0.11-0.42 m from a view point. This is the case MP3D's definition predicts and a planar rule cannot see. Neither
  half is significant on 180 episodes by itself; together they are, each is the direction the dataset's geometry
  dictates, and no step of the four loses an episode. The adapter's criterion has the highest SR and the highest
  SPL of the seven arms.
- **The approach as it was** (base -> `f257a0ce` with its frontal standoff switched off): 9 won, none lost
  (p = 0.004) -- every one a raised target: six pictures, a cabinet, a sink and a chair. Adding the frontal
  standoff to it LOSES one and wins none (zsNo4HB9uLZ/000045: from the frontal standoff the steps toward the
  picture ended at a refused one, 0.101 m from a view point): on this set the standoff was worth nothing, and it
  is not restored. The criterion wins the same nine and five more, all five on the floor (three tables, a stool,
  a towel), which a rule for raised targets never stepped toward.
- **Against this morning's "before"**: 6 won, none lost (exact McNemar p = 0.03), SPL +0.008. The six are those
  three tables, the stool and the towel, and the picture the frontal standoff had left 0.10 m from a view point.
- **What the walk is and costs.** In 116 of the 158 episodes that decided a STOP the range read beyond the
  distance and the agent walked: one step in 89, two in 15, three in 12 (155 forward steps and 20 turns in all,
  under one action an episode). 88 walks ended within the distance (85 succeeded); 14 at a refused step (10
  succeeded); 14 under the target, where another step would have left the memory under 0.30 m (12 succeeded). On
  the 134 episodes that succeed either way the SPL falls by 0.012 (0.25 m of path a step); over all 180 it rises
  by 0.022. Refused forward steps 1,417 -> 1,432.
- **What still fails at a STOP** (10 of 158): two STOPs 4.8 and 6.1 m from any view point (not the walk's to
  mend); four pictures 0.9-1.4 m up, 0.26-0.80 m from a view point, where a step was refused or the agent already
  stood under the picture; and four on the floor (two beds, a counter, a sofa) that STOPped 0.71-0.87 m from the
  near edge they measured and 0.24-0.83 m from a view point. The last four are inside the zone by the agent's own
  measurement and outside the published one. Why was not established (the surface the agent measured may not be
  the goal instance's box; a zone may lack view points where the generator's visibility test failed); no stop
  distance drawn from the definition answers it, and none was fitted to it.
- **Why the pairing can be trusted.** The pipeline is deterministic under the stand-in. (a) In the 64 episodes of
  the adapter's arm in which the walk took no action -- 22 that never decided a STOP, 42 decided inside the distance
  or with the target behind the agent -- the trajectory is the base's, step for step; and in all 180 every other
  counter of the closing is the base's (releases, standing points at the retry clearance 89, blind verification
  steps 314, the same five episodes with twenty or more of them). So this tree with the criterion off IS this
  morning's A2, which is also what items 3 and 4 claim. (b) The `planar` 1.0 m, `planar` 0.9 m and `euclidean`
  1.0 m arms were run on every episode in which they can differ from the base -- a STOP decided at or beyond that
  distance by that measure, read off the adapter arm's own `stop_decision` record: 36, 105 and 66 episodes -- and
  the last two on 8 controls each, all identical to the base; the four re-run episodes in which no step was
  walked are the base's too. Elsewhere their rows are the base's. (c) The tree itself: ten episodes in which the
  walk acts, run with the criterion off (`{"target_closing": {"stop_within_m": 0}}`), are the base's trajectories,
  ten of ten; twelve run again after the setting was renamed, twelve after the action-budget rule, and the same
  twelve on the tree as committed (after the cleanup below), are the adapter arm's, twelve of twelve each time.
  (d) The reference arm was run on the 14
  episodes in which the frontal standoff had acted this morning and on 12 controls -- six in which the raised
  target's steps had acted, and the six carried episodes with a remembered pursuit, through which a standoff could
  have been chosen uncounted -- all identical to "before"; elsewhere its rows are "before"'s.
- **How it was run.** The adapter's arm ran from a snapshot of the tree (`stop_20261116_e`); the setting was
  renamed afterwards (`stop_distance_m` -> `stop_within_m`, to keep it apart from the legacy pursuit's setting of
  that name) and the partial arms ran from the renamed snapshot (`stop_20261116_f`); the controls after that ran
  from snapshots of the tree as it then was (`stop_20261116_g`, and `stop_20261116_h` for the tree as committed),
  and the reference arm from this morning's snapshot of `f257a0ce`. Rows under `runs/cpu_ab_20261116/` (not
  tracked): `s2_euclid09`, `f_planar10`, `f_planar09{,_z6}`, `f_euclid10{,_z6}`, `f_s2check`, `f_s0control`,
  `g_final_check`, `h_final_check`, `h_hm3d_smoke`, `x_head_frontal_off{,_pursuits}`.
- **What this check is not.** The limits of the entry below stand: ground-truth perception, a stand-in for the
  language model, software rendering, one seed, 180 episodes of eight MP3D scenes. Two more. The zone was
  established on the validation goals of these same scenes, and the two bounds of the walk (four steps, 0.30 m)
  were chosen after looking at how HEAD's steps had ended on these episodes (within: 23 of 23 succeeded; at its
  0.45 m bound: 16 of 19; blocked: 9 of 13) -- so the check is not independent of the design. And nothing here
  says what the criterion does on HM3D or Gibson.

#### The defaults, after the check

- **MP3D: `euclidean` 0.9 m.** The arm with the highest SR is also the one with the highest SPL -- 82.2 % / 0.354
  against 80.6 % / 0.352 and 80.6 % / 0.350 for the two nearest arms and 78.9 % / 0.346 for `f257a0ce` -- so the
  tie-break asked for (where the SR difference is not significant, the better SPL) decides nothing here. It is
  what `mp3d/protocol.py` states and `habitat/entrypoint.py::method_defaults` hands the policy: built with no
  `--policy-config`, the MP3D settings equal that arm's recorded settings in all 228 fields (one of them under
  its new name). The rest of that arm is the default with it: the retry at the body radius (0.18 m), the blind
  verification steps as at `f257a0ce`, no frontal standoff.
- **HM3D: `planar` 0.9 m** (`hm3d/protocol.py`, v1 and v2) -- its own definition's measure, the same margin.
  Between the two planar arms the SR difference is not significant (5 won, none lost, p = 0.06) and 0.9 m has
  the better SPL (0.350 against 0.341); that was measured on MP3D, and HM3D itself has only been smoke-run
  (below). **Gibson: `planar` 0.9 m**, not run.
- **The core's own default stays "no criterion"** (`TargetClosingSettings.stop_within_m = 0`): a success zone is
  a benchmark's, so the distance lives in the adapters, and every HM3D, MP3D and Gibson entry point builds its
  policy through one (`hm3d/run.py`, `mp3d/run.py`, `gibson/run.py`: the three callers of `build_policy`).
- **Run on the defaults, end to end, on the tree as committed** (CPU, the stand-in, snapshot `stop_20261116_h`).
  MP3D: the twelve control episodes are the adapter arm's trajectories, twelve of twelve (`h_final_check`).
  HM3D, through `hm3d.run` with the criterion for the first time: the first eight published v1 episodes of two
  scenes (`h_hm3d_smoke`) -- no agent error; 9 STOPs, all 9 successes, each 0.01-0.06 m from the nearest view
  point; seven were decided 0.93-1.05 m out in the plane and walked one step each, to 0.68-0.90 m; two were
  decided inside the distance and took none. Sixteen episodes say that the path works, not what it is worth.
  Both benchmarks' preflights pass with the launchers' own arguments. The launchers themselves were not started
  (they load the model on the GPU): `bash -n` and `--help` only.

#### Cleaned up before the commit (no behaviour changed; the controls above ran after it)

- `gibson/run.py` reads the one `STOP_MARGIN_M` of `habitat/goal_zone.py` instead of a copy of its own, and
  the adapters' defaults are asserted in `tests/test_goal_zone.py` alone (they were repeated in
  `tests/test_target_closing.py`).
- `NodeOraclePrompt` refuses a system prompt that does not ask for every key the oracle's parser reads
  (`REPLY_KEYS`, which only the tests had read).
- The check's tools are cut by responsibility: `mp3d/ram_gate.py` holds the RAM start gate (`RamGate`, with its
  own tests) and `mp3d/cpu_stub_campaign.py` the launcher (`build_parser`, `selected_ids`, `main`); the probe's
  scenarios moved out of its runner into `tests/probe_node_oracle_scenarios.py`.
- Two long functions split: `TargetClosing._stop_approach_end` (why the walk ends) out of
  `_stop_approach_step`, `policy_factory.oracle_model_record` out of `build_policy`.
- `mp3d/stuck_report.py` lost the row of the withdrawn blind-step fix's counter and its two assigned lambdas.
- Stale words corrected: `mp3d/run.py::_method`'s docstring (the entry point does hand the method the success
  zone) and the runtime README's description of the probe.
- **In the commit, deliberately:** the paired check's harness (`stub_llm.py`, `mp3d/cpu_stub_campaign.py`,
  `mp3d/ram_gate.py`, `mp3d/stuck_report.py`, with tests) -- the check that the READMEs, this file and the paper
  cite is run with it, and `LESSONS.md` 2026-11-16 records what its absence from the repository cost. **Not in
  it:** the rows (`runs/`, ignored), the code snapshots (outside the repository), and the per-arm policy
  configs, id lists and analysis scripts of the day's scratch work.

#### Tests

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core` (the three torch-importing files
  ignored): **3,916 passed, 2 skipped** (this morning 3,906 / 2). `objnav_benchmark` and
  `objnav_benchmark_runtime`: **1,851 passed, 1 skipped** (this morning 1,834 / 1). New:
  `core/.../tests/test_stop_criterion.py` (the two measures), `tests/test_goal_zone.py` (each adapter's zone; the
  margin's argument checked on the geometry for boxes at any height and grids at any offset; Gibson's radius),
  `tests/test_launcher_defaults.py` (the launcher's model defaults, in a clean environment),
  `tests/test_ram_gate.py` (the start gate, moved out of the launcher's tests), and in
  `tests/test_target_closing.py` the walk to the stop distance: its start, its measure, every way it ends, the
  action budget, the record. Rewritten: the retry-clearance tests (the body radius by default, 0.20 m as a
  setting), the entry point's defaults, the stuck report's rows. Removed: the four tests of the blind-step fix.
- Every edited file under `core/` parses as Python 3.8 (`stop_criterion.py` is standard library only), pyflakes
  reports nothing that `f257a0ce` did not, and `docs/paper/sync_prompts.py --check` passes (the prompts were not
  touched).

### Changed (2026-11-16, before the long run: the oracle's prompt leaves the judgement to the model, the closing loses its evaluator-shaped rules, the stair nodes are deleted, and four faults are fixed)

Seven items asked for in one task, each with its tests; the paper (`docs/paper/zson_paper.tex`) describes the
method as it now is and says, in its section 5.6, that the reported numbers predate all of it. Ground-truth
perception and the depth filter were not touched. Nothing here was run on a benchmark with a real model: what was
measured is a 180-episode paired check with a stand-in for the language model (items 2, 4 and 6, each isolated in
an arm of its own) and a probe of the prompt on hand-written scenarios (item 1). **In that check every one of the
three items lowers the success rate: 78.9 % before, 74.4 % without the evaluator-shaped rules, 73.3 % with the
blind-step fix, 71.1 % with the 0.20 m retry clearance** -- and refused forward steps fall from 7.95 to 5.92 an
episode. Only the first step is significant at n = 180; the numbers are in the last section.

**Revised the same evening (the entry above), and where the two disagree about the tree the entry above is
right:** the STOP on the deciding frame of item 2 became a stop criterion each benchmark's adapter states from
its own success definition; the blind-step fix of item 4 and the 0.20 m retry clearance of item 6 were withdrawn
(the code is as it was before them); and the launcher's oracle model became the 3B. Items 1, 3, 5 and 7 stand as
written. Sections 2, 4 and 6 and the check are kept as they were measured, because the later decisions rest on
them.

#### 1. The node oracle's prompt: the model's own knowledge, and none of ours

- **The principle.** The system prompt tells the model to use its own reasoning and its own knowledge of standard
  homes and buildings -- how they are laid out, in which rooms furniture and objects of each kind are normally
  found -- and to derive every probability from that. It holds no category knowledge and no rule of judgement: no
  room named for any object, no probability, band or typical value, no list of a benchmark's goal categories, no
  count of rooms a home has, no "coverage first, time second", no rule on what an exit is worth or how the budget
  weighs, no worked example. What it does hold is what a model cannot know: what a node is, what each field of a
  line measures and how it was obtained, what the reply looks like, that the numbers are independent, and that
  travel is the planner's. The one object a model sees named is the target of the episode, on the `TARGET:` line
  of every query. The default version, `facts-qwen2.5-3b-v1`, is 3,471 characters and is written for the 3B this
  laptop runs: what a small model would have to infer is SAID, as a fact -- a node is a room together with the
  unseen space its exits open onto; a detector looks for the target in every frame and has not found it on the
  floor a line lists as covered; the number is the probability that the target is in the part of the node not
  checked yet; `unknown` may be any kind; what 0 and 100 mean -- and the question is restated, with the target's
  name, as the last line of every query. That last definition (the part not checked yet) is the one place where
  the prompt goes a step past "what a field measures": it says what the number is about, not how to get it.
- **What it replaced.** 14,751 characters: the typical rooms of a toilet, a bed, a sofa, a television, a chair and
  a potted plant (the six HM3D goal categories; 13 of MP3D's 21 had none), the rooms and room sizes of a typical
  home, a band of numbers for every room status ("0-2", "30 or more", "60 or more", "at most HALF"), the rules
  COVERAGE FIRST / TIME IS SECONDARY / EXITS / SPREAD, a worked toilet example with numbers, and "NEVER use ...
  here=yes as a reason". Saved verbatim beside the new one in `runs/prompt_side_by_side_20261116.md` (not
  tracked).
- **Division of labour.** The model gives one probability per node. Distance, and the weighing of distance
  against the probabilities, is RPT*'s: the prompt carries no distance, no position and no word on where the robot
  stands -- `here=yes` is no longer rendered (`SearchNode.here` is kept for the record), and how long ago the robot
  left a room is read for the status word alone.
- **The room line carries everything the scene graph stores about the room, except where it is.** New on it:
  the objects with their counts, the most numerous first (`seen: chair x4, table`; a cut list says so, `+3 more`);
  the types of the rooms a door or a walked threshold joins it to (`connected to: bedroom x2, hallway`,
  `ObservedSceneGraph.connections()`, the graph's own edges); the length of boundary still open to unseen space
  (`open edge 2m`); effort in actions, the unit of the HOUSE line's budget (`searched=40 actions`,
  `scanned(full rotation, 120 actions ago)`, in the same coarse steps as before, so an unchanged map is an
  unchanged prompt); and `target: detected here, not reached` on a room whose confirmed objects include one of the
  target's own kind (`SearchNode.target_seen`, from `TargetLabels.accept_labels`) -- which is what lets the prompt
  say that the covered floor was looked at without finding it. Left off, and why: the room's centroid, the lines
  its walls stand on and the agent's distance to each (position is distance), the classifier's confidence (the `?`
  is its reading), elongation (the two dimensions are there), the previous round's numbers, and bookkeeping ids.
- **"Final" was a promise the code cannot keep; the words were changed, not the rule.** `size=12m2 (4x3m, all
  walls seen)` was explained as "FINAL -- the room is that big and no bigger". Of 252 rooms so declared in 40 HM3D
  recordings, 73 were over a quarter bigger by the end of the episode. Of the 233 growths behind that, 178 were the
  partition merging two rooms and 51 a cut between two rooms moving -- floor that was mapped, and listed under the
  neighbouring node, all along -- 1 a fragment and 3 a side read as located that was not. What the extents know is
  that no side of the room, as the partition draws it now, opens onto unmapped space; they cannot know that the
  watershed will not join it to its neighbour. So the line reads `all sides mapped` / `1 of 4 sides not mapped
  yet`, and the prompt says "though the map's division into rooms may still be revised". The alternative --
  withholding the verdict wherever a side is a cut rather than a wall -- would have told the model that mapped
  floor is unmapped, to guard against a merge the map does not show coming. `room_extent.FINAL` keeps its name
  (recordings and `room_extent_replay.py` carry it) with a docstring saying what it does not promise.
- **The multi-storey supplement is gone** (`STAIRS_SUPPLEMENT`, the STAIRS node kind, the storey lines of the
  user prompt, the `home_here` verdict and `LoopSettings.unexplored_elsewhere` that read it): the search is
  single-storey (item 7). The openings supplement, for the deprecated exits-as-nodes mode, is cut to what an
  OPENING line means.
- **The reply format is unchanged** -- `{"home","house","stage","pass","nodes":[{"id","why","p"}]}` -- and so is
  the parser. The schema-repair note no longer shows the shape filled in (`"id":0 ... "p":50` put a number in
  front of a model that had just failed to produce its own): every value is a placeholder.
- **Prompt versions, and the run record** (`core/mapping/topology/search_node_prompts.py`, new). A wording is
  read by one model and may be misread by the next, so a wording is a named *version* (`NodeOraclePrompt`: system
  prompt, openings supplement, user frame, `tuned_on`), written against one model and probed on it. A run selects
  one by name (`LoopSettings.oracle_prompt`, policy config `{"room_search": {"oracle_prompt": "<name>"}}`; an
  unregistered name is refused at start-up) and its record carries
  `node_oracle.prompt = {version, tuned_on, sha256, system_chars}` and, from the benchmark factory,
  `node_oracle.model = {name, digest, prompt_tuned_on_it}` -- the model that answered and whether it is the one
  the version was tuned on (a mismatch is also said once at start-up). `tests/test_search_node_prompts.py` holds
  EVERY registered version to the contract above, so a version added for the PC's model cannot slip a rule or a
  room name past it. The module docstring is the four-step procedure for adding one. One version exists,
  `facts-qwen2.5-3b-v1`; a run on another model uses it and says so (`prompt_tuned_on_it: false`) until that
  model has its own. No launcher default was changed here: `benchmark_services.sh` and `LLMConfig.from_env`
  defaulted `LLM_REASONING_MODEL` to `qwen2.5:14b-instruct`, so a laptop run set
  `LLM_REASONING_MODEL=qwen2.5:3b-instruct` (as `mp3d/run_single_storey.sh` already does). (The launcher's
  default became the 3B that evening: entry above.)
- **Probed on the model this laptop runs** (`qwen2.5:3b-instruct`, CPU only, temperature 0, one call per
  scenario; `tests/probe_node_oracle.py`, rewritten: `--prompt <version>`, `--set P|O|PO`, `--only`, `--repair`).
  Nineteen hand-written scenarios, each with one ordinal claim a reasoner who knows homes would not get wrong
  (never shown to the model): `P`, twelve as a run sends them by default, and `O`, the first probe's seven. Claims
  held at the model's own numbers / replies usable / claims held after the loop's rescaling:

  | wording | system prompt | `P` (12) | `O` (7) | a call |
  |---|---|---|---|---|
  | before today (rules and priors) | 14,751 chars | 8 / 12 / 10 | 5 / 7 / 6 | 91-130 s |
  | the first facts-only wording (this morning, not registered) | 2,955 chars | 3 / 11 / 6 | 2 / 6 / 5 | 54-63 s |
  | `facts-qwen2.5-3b-v1` | 3,471 chars | 5 / 12 / 10 | 4 / 7 / 6 | 50-56 s |

  So on the 3B the facts-only prompt holds fewer claims at the model's own numbers than the prompt of rules did
  (9 of 19 against 13) and as many after the loop's rescaling (16 of 19 each); every reply is usable and no
  never-entered room is put at 0 (the first wording did that in 4 of 19 queries). The 3B's own knowledge of
  where things are is mostly sound ("beds are typically found in bedrooms") and sometimes wrong (a toilet
  "typically found in bathrooms or kitchens"); what it does not do is combine fields. The two scenarios that
  ask what coverage without a detection means (a room of the fitting type covered whole against a barely seen
  room of another type) were held by the old prompt, which stated the rule, and by none of the facts-only
  wordings tried -- the loop's discount of an entered room carries both. What helped and what did not (a
  negation on every line, explaining the detected-target note, `left=` as its own field, an `unchecked=`
  framing, a scaffold in the readings, a shorter legend) is in `core/mapping/topology/README.md`. Nineteen
  scenarios, one call each, a model that one sentence moves by three scenarios and that Ollama does not answer
  bit-identically twice: one or two claims of difference are noise. On `qwen2.5:14b-instruct` the first
  facts-only wording held 5 of the 7 `O` claims (CPU, one run, before the instruction to stay on the 3B); no
  version has been tuned on it.
- **Not changed, and to be decided:** the code still rescales the model's numbers by exploration state (a
  never-entered room x1.5 and at least 0.15; an entered one x0.5 with a 20-action half-life, at most 0.2 after 20
  actions with one frontier left; under 0.05 not offered) and offers a target-landmark node at a fixed 0.85. These
  were hand-set under the old prompt, which stated the same rules to the model; the new prompt states none, so
  they are now the one hand-set adjustment between the model's number and RPT* -- and the model is shown the
  evidence they act on, so it may be counted twice. The room labeller's prompt and its fixed set of room types
  were not touched either.

#### 2. Rules shaped by the evaluator: two removed, one kept with its reason

*(Revised that evening: the STOP on the deciding frame described here cost eight successes, and it was replaced
by a stop criterion derived from each dataset's own success zone -- which also answers the "Open" point at the
end of this section. The frontal standoff stays removed. Entry above.)*

- **Removed: every step after a decided STOP** (`stop_overshoot_*`, `elevation_overshoot`, the `ADVANCE` phase
  and the entrypoint's switch). The planar overshoot walked one or two
  steps toward the object because a near-edge reading is short of the footprint Gibson's evaluator measures to,
  and `habitat/entrypoint.py::method_defaults` switched it off when the evaluator scores view points: a rule
  switched by which evaluator scores the run. The raised-target variant (2026-11-10) took up to three steps under a
  picture or a wall cabinet "because its view points stand closer", with its reach and its bounds sized on MP3D's
  validation view points. `method_defaults` now hands the method the agent's body and nothing else, and the
  closing issues its STOP on the frame that decides it: a fresh, complete box of the target within 1 m, the task
  definition's own distance.
- **Removed: the frontal standoff of raised planar targets** (`frontal_standoff`, `perception/face_normal.py`
  and its tests, the normal carried through `projection.py` and `target_path.py`): a plane fitted to the target's
  depth pixels, standoffs tried about its normal first, an in-range STOP from beside it deferred -- "no view point
  is" beside a picture, sized on the view points of the set's 78 pictures. The standoff ring starts from the
  agent's own bearing to the target, as before 2026-11-10.
- **Kept: the reach of the storey over steps** (`SpawnFloorGuard.step_reach_m`, `step_surface_band_m`), with the
  reason now stated in the code and the paper, and it is the agent's own: a step is support the body can walk
  onto (risers of at most `max_step_m` with head room); from a storey walked level, support further off the plane
  than the atlas's `departure_m` (0.45 m) is where the agent's own storey tracking reads it as leaving and walks
  it back; from a provisional spawn storey the whole band (0.8 m) stays under the least separation of two storeys
  (1.5 m). The 2026-10-31 look at validation shortest paths on the navmesh confirmed the 0.45 m bound and is not
  its reason; the paper lists it under development on evaluation data. The 0.8 m band is a hand-set value inside
  those limits and was also looked at against validation goals; it is left and flagged.
- **A policy config that still names a removed setting is refused** (`TypeError`), never silently ignored:
  `mp3d/sets/policy_fix_closing.json` is deleted and the other four configs lost their `target_closing` block.
- **What the removal costs, measured** (arm A2 of the check below: the last commit with the rules, both switched
  off, against the same commit with them on): SR 0.789 -> 0.744 (8 lost, 0 won, exact McNemar p = 0.008), SPL
  0.346 -> 0.332. Every loss is a STOP with the target in view and within 1 m in the plane, 0.11-0.42 m from
  the nearest view point (success needs under 0.1 m): five pictures, a cabinet, a sink, a chair. Seven of the
  eight had taken the raised target's steps (which acted in 54 episodes: 48 successes with them, 41 without),
  one the frontal standoff (14 episodes: 8 and 5). The 123 episodes in which neither rule had acted are
  identical in the two arms, step for step. **Open:** the task definition's distance is Euclidean ("within 1.0m Euclidean distance from any instance
  of the target object category AND the object can be viewed by an oracle from that stopping position"), and the
  closing measures its 1 m in the plane -- for a raised target the two differ. Deciding the STOP on a 3-D distance
  would be a statement about the task and not about an evaluator; it was not done here (the removed rule was not
  that: under a target more than about 0.9 m up its test could never pass and its three-step bound decided).

#### 3. `ground_truth_semantics` in the run record

- `configuration()["ground_truth_semantics"]` was the constant `False`, in a `--perception ground-truth` run too.
  It is now read from what the perception services declare (`interfaces.ground_truth_semantics(detector,
  segmenter)`: only the literal `True` of a `ground_truth_semantics` attribute counts, so a test double that
  answers every attribute declares nothing). `GroundTruthDetector` and `GroundTruthSegmenter` declare it; the
  policy and the FALCON+YOLO baseline record it. The method never branches on it. (The two declarations are the
  only lines changed under `ground_truth/`.)

#### 4. The blind verification-step loop

*(The fix below was withdrawn that evening, to be handled on its own: `_glimpse_stalled` and `glimpse_stalls` are
not in the tree, and the loop is a known, open fault. What it measured stands. Entry above.)*

- **Cause.** A box of the target's class with no usable depth (a *glimpse*) cannot be placed, so the verification
  steps along its bearing to get a depth. A step that would collide is refused whole and changes neither the pose
  nor the frame: the next action saw the same box and asked for the same step. The verification's 12-action
  budget did not bound it, because its clock pauses while the candidate is in view, which allowed about 36
  actions a verification. Five of the 180 episodes of the check took 36 to 70 such steps; three of them did so
  with the navmesh still in the closing.
- **Fix.** The first refused step ends it (`TargetClosing._glimpse_stalled`, counter `glimpse_stalls`): the
  glimpse goes to the pursuit, which plans a route on the map instead of a bearing; the spot is remembered as
  boxed in; the verification releases with no rejection (the candidate was not disproved). The bearing approach
  is unchanged -- it was never such a loop (it gives up after three steps that went nowhere). Cutting it to one
  refused step as well was tried and withdrawn: nobody had asked, and on the 180 episodes it changed 33
  trajectories (one success won, two lost) to save 139 refused steps.
- **Measured alone** (arm A24 below against A2): exactly the five episodes change. Blind verification steps
  314 -> 46 in all, refused forward steps 1,417 -> 1,223, no episode with twenty blind steps left. Three of the
  five still succeed (128, 89 and 80 steps where they took 205, 82 and 116); the other two had ended well after
  their 70 refused steps and are lost at the step limit (2azQ1b91cZZ/000108 keeps returning to a cushion it
  cannot reach: nine pursuits, ninety escape probes; QUCTc6BB5sX/000057 walks off after remembered targets and
  never sees its towel again). SR 74.4 % -> 73.3 % (0 won, 2 lost, exact McNemar p = 0.5).
- **Not fixed, and next in line:** what refused steps remain are other mechanisms. In the final arm twelve
  episodes have twenty or more refused steps (524 of the 1,066), nine of them in Z6MFQCViBuw; the fallback's
  escape probe runs twenty times or more in eight episodes (eleven before today) and 210 times in one.

#### 5. Failing tests

- **`test_monitor.py::test_finished_is_judged_by_the_pid...`: the bug was in the code.** `exit_code()` compared
  the code file's `st_mtime` with a wall-clock `since`; a file is stamped from the kernel's coarse clock, up to a
  timer tick behind `time.time()` (1.15 ms measured), so a code written just after `since` read as older and was
  ignored -- a run that failed with a code was reported as "killed (no exit code recorded)", and a monitor started
  without a PID never saw it end. `Monitor` now takes its fallback start from a stamp of the run's own directory
  (`file_clock_now`) and `exit_code()` allows a wall-clock `since` `STAMP_LAG_S` (20 ms). The failing test passes
  unmodified; two new tests pin the race. Not fixed: a PID reused after a reboot reads as alive.
- **`test_potential_mapper.py` (8 failing): one bug in the code, seven stale tests.** The code: OpenCV 5 returns
  `HoughLinesP` as `(N, 4)` where 4.x returned `(N, 1, 4)`, so `lines.squeeze(axis=1)` raised inside every
  `update()` that found a wall -- now `reshape(-1, 4)`, right under both. The tests: six called `step()`, removed
  on 2026-03-16 (`9d5de68c`) -- now `update(_backproject(...))`; two of those also asserted the pre-NaN contract of
  unobserved cells; and `test_target_navigation` was stale in its frame (the robot at the bottom centre), its law
  and its laziness (the fields are computed in `update()`). A `def` line lost in an earlier edit is restored
  (19 -> 20 tests). `zeta` is an unused knob and the constructor prints a debug line; both left.

#### 6. Standing-point clearance: 0.25 m, then 0.20 m

*(Decided that evening on the measurement below: the retry clearance is the body radius, 0.18 m, by default --
`standing_retry_clearance_m: None` -- and 0.20 m is a setting. The retry planner of the closing's own stays.
Entry above.)*

- The closing's standing points are planned at the default 0.25 m from every observed obstacle and, only when no
  route is found, at 0.20 m (`target_closing.standing_retry_clearance_m`, `policy.standing_retry_planner`) -- no
  longer at the body radius (0.18 m), where the squeeze planner took them until now. On the 0.1 m grid the nearest
  obstacle cell is then at least 0.283 m and 0.224 m from the agent's cell. The counter is `retry_standoffs`
  (`squeezed_standoffs` in rows recorded before; `stuck_report` reads either). The fallback's and the room
  entries' squeeze planner is untouched.
- **How "0.20 m" was read, and what it is on the grid.** The planner blocks a cell whose clearance is at most
  its inflation, for 0.25 m and for 0.20 m alike, so the retry keeps MORE than 0.20 m: the nearest obstacle cell
  centre is 0.224 m away, one cell class further than the 0.200 m the body-radius squeeze allowed. Read
  inclusively ("0.20 m is enough") the retry would be the old squeeze exactly and item 6 would change nothing.
- **Measured alone** (after against A24): 42 episodes change. Standing points reached at the relaxed clearance
  92 -> 52; releases for want of a route 29 -> 49; refused forward steps 1,223 -> 1,066 (in the 42 episodes:
  13.2 -> 9.5 an episode); SR 73.3 % -> 71.1 % (2 won, 6 lost, exact McNemar p = 0.29), SPL 0.325 -> 0.322. Three
  of the six lost are STOPs on the target, 0.10-0.34 m from the nearest view point; three run to the step
  limit, two of them after releases with "no safe path from here" (one spends 210 actions in the fallback's
  escape probe). The margin costs success here and buys fewer refused steps; whether that is the right trade is
  the user's call, and `{"target_closing": {"standing_retry_clearance_m": 0.18}}` in a policy config gives the
  old behaviour back.

#### 7. Stair nodes deleted

- Dormant since 2026-11-15 (no stair source says where a flight leads): `rooms/stair_nodes.py`,
  `multifloor/stair_approach.py`, `tests/test_stair_nodes.py`; the loop's stair paths; `MultiFloorSearch.commit`,
  `portal_by_id`, `way_back_held`, the `stairs_taken` callback; `MultiFloorParams.floor_change_cost_m` and
  `arrival_grace_actions`; the oracle's STAIRS node and supplement (item 1); the stair rows of the search panel,
  the snapshot and the recording. `graph.stair_probs` is `extra_probs` (the deprecated opening nodes still use
  it). With traversal allowed a floor change is the building coordinator's allowance rule, as before stair nodes
  existed. It is all in the history at `f257a0ce` for the multi-storey work.
- Removing settings and parameters changes the frozen configuration: `--resume` refuses a campaign recorded
  before today, and a config naming `stairs_as_nodes`, `unexplored_elsewhere`, `floor_change_cost_m`,
  `arrival_grace_actions` or a removed closing setting raises.

#### The check, and its harness

- **In the repository now:** `objnav_benchmark_runtime/stub_llm.py` (the stand-in for the language model: every
  room `living_room` at 0.8, the nodes of an oracle prompt an equal share of 90 %; it keys on the node ids alone,
  so the prompt's wording does not change an arm), `mp3d/cpu_stub_campaign.py` (one process per scene on Mesa's
  llvmpipe, started against free RAM: `shard_pool.run_pool` with the first-fit `RamGate` of `mp3d/ram_gate.py`,
  for which `run_pool` gained a `choose(pending)` hook), `mp3d/stuck_report.py` (the paired stuck-indicator table and the
  identical-trajectory count), with tests. Yesterday's check had been run from a scratch folder; the launcher
  was rebuilt and, on the tree yesterday's second arm ran, reproduced its stored rows (33 of 33 trajectories on
  two scenes) -- so that stored arm is today's "before". A first "before" arm run with a differently written
  stand-in was discarded (16 of its 92 episodes were other trajectories).
- **Four arms on the 180 episodes of `sets/single_storey_flat_v1.json`** (ground-truth perception, the frontier
  explorer, seed 17, CPU rendering, the stand-in LLM), each differing from the one before it in one item:
  *before* = `f257a0ce` (the stored arm); *A2* = `f257a0ce` with the raised target's steps and the frontal
  standoff switched off (item 2); *A24* = this tree with the retry clearance set back to 0.18 m (plus item 4, and
  items 3 and 7 and the prompt, which must not move anything); *after* = this tree (plus item 6).

  | per episode | before | A2 | A24 | after |
  |---|---|---|---|---|
  | success rate (successes of 180) | 78.9 % (142) | 74.4 % (134) | 73.3 % (132) | 71.1 % (128) |
  | SPL | 0.346 | 0.332 | 0.325 | 0.322 |
  | steps | 196.8 | 196.0 | 197.8 | 195.9 |
  | refused forward steps (in all) | 7.95 (1,431) | 7.87 (1,417) | 6.79 (1,223) | 5.92 (1,066) |
  | blind verification steps, in all | 314 | 314 | 46 | 41 |
  | standing points at the relaxed clearance, in all | 89 | 89 | 92 | 52 |
  | releases for want of a route, in all | 25 | 25 | 29 | 49 |
  | episodes ended without a STOP | 22 | 22 | 24 | 25 |
  | against the arm before it: won / lost (exact McNemar p) | -- | 0 / 8 (0.008) | 0 / 2 (0.5) | 2 / 6 (0.29) |
  | trajectories identical to the arm before it | -- | 124 | 175 | 138 |

  Before to after: 1 won, 15 lost (p = 0.001), 94 trajectories identical. SoftSPL barely moves (38.5, 38.9, 38.7 %
  for before, A2, after): the agent gets as near as it did, and what is lost is lost at the STOP -- failed STOPs
  within 1 m of the goal go 14 -> 22 -> 25.
- **Why the pairing can be trusted.** The pipeline is deterministic under the stand-in, so an episode in which a
  change never acted is the same trajectory in both arms: the 123 episodes of A2 in which neither removed rule
  had acted; A24's 175 (the five that differ are the five with a blind step ended at a refusal, and nothing else
  in today's change -- the stair-node deletion, the record, the prompt, the refactoring -- moved one episode);
  12 of 12 control episodes run again under A24; 27 of 27 run again under the final tree. `stuck_report` counts
  the identical trajectories with every table, and its signature now includes where the episode ended (two
  arms once took the same 21 actions over the same 1.5 m and ended 5 cm apart).
- **How the last two arms were put together, said plainly.** *after* was first run with the bearing approach cut
  (item 4, above); when that was withdrawn, the 81 episodes that had started a bearing approach were run again
  on the final tree, with 27 controls, and replace their first rows. A24 was run on the 45 episodes in which
  *after* differs from A2, with 12 controls; in the other 135 neither item 4 nor item 6 acted, so A24 is A2
  there. The arms run from snapshots taken during the day carry the prompt version's name with the digest of a
  draft wording; the stand-in never reads the system prompt, and no model has been asked under that draft in a
  run. The rows are under `runs/cpu_ab_20261116/` (not tracked): `a2_head_item2_off`, `a24_retry018{,_quc}`,
  `after`, `after_c`, `after_c_quc`; a later root given to `stuck_report` or `single_storey_report` overrides an
  earlier one (`after=runs/.../after,runs/.../after_c,runs/.../after_c_quc`).
- **What this check is not.** Ground-truth perception, a stand-in for the language model that gives every node
  the same share, software rendering, one seed, 180 episodes of eight scenes that the rules now removed were
  partly sized on. Only its paired differences carry meaning, and of the three steps only the first is
  significant.

#### Tests

*(The counts at the end of the day are in the entry above; these are the counts of this state.)*

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core` (the three torch-importing files
  ignored): **3,906 passed, 2 skipped** (before: 3,883 passed, 8 failed -- the eight of `test_potential_mapper.py`,
  which `CLAUDE.md` still lists as known-red). `objnav_benchmark` and `objnav_benchmark_runtime`: **1,834 passed,
  1 skipped** (before: 1,840 passed, 1 failed -- the monitor test). The counts moved with what was deleted
  (`test_stair_nodes.py`, `test_face_normal.py`, the tests of the overshoot and the frontal standoff) and what was
  added (the prompt versions, the oracle's line, the stand-in, the launcher, the stuck report, the refused blind
  step, the retry planner, the run record, the monitor's clock). `test_monitor.py` 25 times over: no failure.
  Every edited file under `core/` parses as Python 3.8, and pyflakes reports nothing the change introduced.
- **Outside those suites, untouched and unchanged:** `tasks/mapping/scene_graph` 344 passed, 1 failed, as at
  `f257a0ce` (`test_llmdet.py::test_persistent_false_hypothesis_can_still_pass_existing_stop_gate`);
  `tasks/planning/routing_benchmark` and `tools/` pass. A blanket run over `tasks/`, `robots/` and `tools/` met
  two more failures that `f257a0ce` has as well (`launch_params/tests/test_discovery_and_store.py::
  test_the_real_mission_sources_layer_the_way_the_script_reads_them`, `yolo_world_trt/tests/
  test_hardware_policy.py::test_backbone_uses_dla_on_orin_and_is_static`) and then hung in
  `falcon_pegasus/link/tests/test_socket_link.py::test_a_message_arrives_intact`, where it was stopped: the
  tests after that point (about a third) were not run. The first of these exercises the object-search policy
  and fails on the same line before and after (the policy STOPs one frame earlier than the test expects); the
  other three read nothing this change touches.

### Added (2026-11-16, room size and coverage for the node oracle, read off the agent's own occupancy map: the room extents of `feat/objnav-room-extents-nadav`, ported by hand)
- **What came over, and how.** The room extents were written on 2026-11-11 on `feat/objnav-room-extents-nadav`
  (`6739f3d3`, `acb837bd`, `991e9be8`), which forked from this branch at `0dd7d39c` -- before `room_priors.py` and the
  navmesh were removed -- so nothing was merged: the room-size work was carried over by hand onto the current files,
  and nothing those two removals deleted came back with it. The branch is untouched and still holds everything listed
  under "left there" below.
- **`core/mapping/topology/room_extent.py`** -- `room_extents()` reads, for every room of the live partition, its
  boundary faces classified by what lies beyond them (a wall, another room, settled unknown, unseen unknown, floor of
  no room) and grouped into four sides in the storey's wall frame (the dominant wall direction from local second
  moments, so a diagonal wall's staircase is one side); a side is SEEN when at most `open_tolerance` (20 %) of its
  faces open onto unseen space in runs wider than a doorway (`doorway_max_m` 1.2 m, a jamb at each end, in a side
  that is mostly located); the room's dimensions along its axes and its footprint area (free cells plus enclosed
  furniture islands); `bounded` (all four sides seen: the size is FINAL) or `open_sides` (a LOWER BOUND); and the
  floor covered -- `covered_m2`, `covered_fraction` (the share of the whole room; an upper bound while a wall is
  unseen) and `left_m2`, furniture islands credited from the observed floor around them. Recomputed from the live
  map and partition on every update and keyed by cell, so a room the watershed merges, splits or renumbers reads
  right. The reading itself is the branch's code, unchanged.
- **It is given the agent's own map and nothing else.** `room_extents()` takes the storey's occupancy grid, the
  partition the watershed cut out of that grid, the sight ledger's settled and observed cells, and the agent's
  position. No navigation mesh, no pathfinder, no room or region annotation of the simulator or a dataset, nothing
  of `room_priors.py` (which no longer exists): the module imports numpy and scipy and nothing of this repository.
  Three things hold it there: `core/mapping/topology/tests/test_room_extent.py` pins the module's imports and the
  parameters of `room_extents()`; `test_package_boundaries.py` lists it among the `core/mapping` modules that may
  not import `core/planning`; and `test_no_privileged_geometry.py` names it, `scene_graph.py` and `sightlines.py`
  among the modules its scan must have covered. A live test re-reads the size from `policy.last_world.grid`, the
  graph's own labels and the ledger's two records and gets what the scene graph holds and the oracle was shown.
  `configuration()["room_extents"]` records the thresholds and the rule ("observed map only: ...").
- **`SightLedger.observed_mask`** (`world_model/sightlines.py`): per storey and per cell, the free cells seen within
  `SightSettings.detect_range_m` (4 m, the loop's `scan_seen_range_m`) with a clear line of sight, at a pitch that
  puts an object body `object_band_m` (0.3-1.0 m) tall in the frame (`object_band`, `cone_observed`) -- cast over
  the agent's own map from its own poses. A disabled ledger returns no record (None) rather than an empty one, so a
  rig without sight shows a size and no coverage instead of "0%".
- **The oracle's room line** (`search_node_oracle.format_node`): `size=12m2 (4x3m, all walls seen)` when final,
  `size=at least 12m2 (5+ x 3m, 1 wall not seen yet)` when not (the `+` marks the dimension cut short; a sliver reads
  `under 1m2 mapped so far`), followed by `covered=10m2 (80% of the room, 2m2 left)` -- or, when the size is a lower
  bound, `covered=3m2 (at most 40% of the room, at least 4m2 left)`, the share rounded up and the remainder a lower
  bound. `SearchNode` carries `footprint_m2`, `length_m`, `width_m`, `length_final`, `width_final`, `walls_seen`,
  `bounded`, `covered_fraction`, `covered_m2`, `left_m2`, `open_perimeter_m`; a node without them renders exactly as
  before. Everything else on the line is unchanged: `frontier=`, `exits=` with the unseen floor behind them,
  `status=`, `searched=`, `here=yes`, `seen:`, in the same order.
- **The system prompt** explains the two fields, states that every number on a node line is a final result to be
  read and not recomputed, and gains two rules: `HOW MUCH IS LEFT -- COVERAGE FIRST` (coverage is the main fact for
  what is left of a room, the size says what the remainder can hold, the model judges whether the target fits) and
  `TIME IS SECONDARY` (coverage overrides `searched=` both ways). Three sentences that read time, or an assumption,
  where the coverage is now given were replaced with the branch's: the `entered` status no longer says the camera
  "has seen most of what is mapped"; STEP 4's entered-room bullet values what is LEFT (all walls seen and 90 %
  covered is as good as scanned) instead of "`searched=` 20s or longer with `frontier=` 0 or 1"; and "the longer
  `searched=` reads on an entered room, the lower its number" is gone. STEP 2 adds that a size of "at least" is not
  the room's size, and the worked example shows the new fields. The prompt is 14,751 characters (11,778 before);
  the reasoning route asks for an 8,192-token window and refuses a truncated prompt.
- **Recorded and shown:** `last_reasoning["extents"]` (every room's full record: sides, lines, wall distances), the
  step snapshot's `extent` per room, a line under each room of the search column (`>=6m2 4.9x2.4 walls 3/4 cov
  <=43% >=3m2 left`). `room_extent_replay.py` replays a recording's map journal and poses through the reading
  offline and checks its promises (the observed floor conserved across partition changes; growth after a FINAL
  reading classified as a merge, a moved cut, a fragment or a BREACH).
- **Left on `feat/objnav-room-extents-nadav`, deliberately.** (1) `mp3d/room_extent_ground_truth.py` and its test:
  it grades the sizes against Matterport's `.house` room polygons -- a dataset's region annotations -- and nothing
  that reads those was brought, nor the accuracy figures it produced (they are in that branch's CHANGELOG). (2) The
  united "unknown floor behind a room's frontiers" figure (`rooms/frontier_unknown.py`, `frontier=2 (~30m2 unknown
  behind them)`) and with it the change of the `exits=` brackets: a different measurement from a room's size; this
  branch keeps `exits=2 (1 doorway, ~30m2 unseen behind them)`. (3) The other prompt changes of `acb837bd` that are
  not about size or coverage: dropping `here=yes`, the reordering of the line, "the probability that the target is
  there" for "that going there NEXT finds the target", and the rewording of the "places the robot has NOT been near"
  bullet. (4) The A/B harness (`hm3d/ab_room_extents.sh`, `ab_room_extents_report.py`), the fixed 200-episode set
  (`hm3d/representative_set.py`, `sets/hm3d_v1_val_rep200*.json`) and the HANDOFF entries. (5) The prompt-word
  helpers `room_extent.py` carried beside the oracle's own (`size_words`, `seen_words`, `coverage_words`,
  `area_words`, `coarse_metres`, `coarse_fraction`: called only by their tests -- the phrasing has one copy, in
  `search_node_oracle.py`) and `side_index` (unused). (6) The snapshot commit `95aae785`: already here, in newer
  form.
- **Checked.** Tests: the whole of `core` 3,883 passed with the 8 known failures of `test_potential_mapper.py`
  (3,849 before), the benchmark harness and runtime 1,840 passed, 1 skipped, with the one failure of
  `test_monitor.py` that fails on the untouched tree too (1,830 before); the three boundary files pass. The skip is
  the replay of one named HM3D recording, which is not on this machine; pointed at the two recordings of that
  episode that are (`OBJNAV_EXTENT_RECORDING`, `runs/glance_ab/{A,B}_gpu`), it passes. **The port
  against the branch's own flight record:** the A/B's arm "with" (`runs/ab_room_extents_20261111T2000Z/with`) was
  flown with the branch's code and logged every room's extent at every action; 40 of its recordings (2 per scene,
  10,123 actions, 60,607 room-action readings) were replayed through this tree's `room_extents()`. At the 11,028
  readings the live code made afresh, the area, length and width are equal in all of them, the covered floor
  in all of them, and the verdict (final or lower bound, walls seen) in 10,944 (99.2 %). Of the other 84, 81 lie
  inside the band the one-update hysteresis can move a side in (the live code carries its memory from its last
  rebuild, the replay from the last action); 3 do not -- two of them a 1.9 m2 room at actions where the storey's
  wall frame flipped (next bullet), the third at action 1 of an episode. **The promises, on recordings:** the
  same 40 HM3D recordings -- 252 rooms declared final and seen again,
  233 growths of over a quarter after a final reading: 178 merges, 51 moved cuts, 1 fragment, 3 breaches (a room
  that grew through a side read as located; all within the first 28 actions of an episode); 36 MP3D recordings of
  `runs/ss_gt_fixes_v4` (8 scenes, 8,350 actions) -- 167 such rooms, 93 merges, 29 moved cuts, 1 breach (action 2).
  The observed floor is conserved across partition changes (median |delta| 0.24 and 0.38 m2).
- **Seen while checking, and left as the branch had it.** (1) "FINAL" is a statement about the map of that action:
  of the 252 HM3D rooms above, 73 were over a quarter bigger by the end of the episode, mostly through the merges
  and moved cuts counted above -- which the reading is right to follow, but the prompt tells the model "the
  room is that big and no bigger". (2) The storey's wall direction is estimated modulo 90 degrees, and in a house
  aligned with the grid it alternates between about 0 and about 90 degrees -- on 312 of the 10,123 actions, in 36 of
  the 40 recordings. The names of the four sides swap with it, and the per-side hysteresis is keyed by name, so it
  is spent on the wrong side after each flip. The cost is small: of 52,213 room-action pairs whose own cells did
  not change, the final / lower-bound status flipped in 124 and the count of walls seen changed in 461; with the
  frame folded into (-45, 45] degrees (in the analysis only) 117 and 393. (3) The extents are shown to the model
  and recorded; no code reads them for a decision. The code-side rescaling of an entered room's probability still
  goes by actions spent inside and frontier count (`LoopSettings.entered_discount`, `dwell_*`), while the prompt
  now tells the model that time is secondary. (4) The worked example in the prompt exceeds the limits the prompt
  sets: its `house` string is 32 words (limit 25) and two of its `why` strings are 13 (limit 12).
- **Not measured.** No episode was flown with this tree: no simulator, no language model, no GPU. What the model
  does with the new lines is known only from the branch's A/B, which is not this configuration (the 3B model in
  both roles, ground-truth perception, the room priors still in, and that branch's full prompt change): on 200
  HM3D v1 val episodes SR 0.690 -> 0.670 (95 % CI of the difference [-0.060, 0.020], exact McNemar p 0.45), i.e.
  within the noise, with the oracle's choice changed on identical evidence in 113 episodes and its stores holding
  1,143 oracle answers against 1,064 without. The live-model probe (`tests/probe_node_oracle.py`) was not run.
- **Not comparable with earlier runs, on purpose.** The oracle's prompt differs, so answer stores recorded before
  this change no longer replay its answers (the room labeller's still do); the frozen configuration differs
  (`room_extents`, two new `sight` fields), so `--resume` refuses a campaign recorded before it. Both full-split
  runs behind `docs/paper/zson_paper.tex` predate it; the paper says so (Section 5.6 and the run table), describes
  the reading (Section 4.6, the constants table, the module map), carries a TODO about the Matterport check in its
  "Development on the evaluation data" paragraph (Section 5.5), and its Appendix C was regenerated from the code
  (`docs/paper/sync_prompts.py`). The paper was not built here (no LaTeX on this machine).
- Docs: this file, `LESSONS.md` (the branch's 2026-11-11 entry on the sliver seen through a door), the topology,
  `object_search` and runtime READMEs, the paper.

### Removed (2026-11-15, the agent's privileged scene geometry: the navmesh projection of the closing's standing points and the ground-truth stair source)
- **The ZSON agent is given no simulator geometry any more.** An audit found two things reaching `core/` that ASCENT,
  ApexNav, VLFM and MFNP do not give their agents. (1) Every run script bound `policy.target_projector` to the
  environment's navmesh snap (`HabitatRGBDSimulator.project_target`), and the closing projected every standing point
  through it at three call sites -- the standoff ring (`closing/target_path.py`), the back-off spot
  (`target_backoff.py`) and the nearest-floor goal (`target_closing.py::_nearest_reachable`). (2) Both environment
  adapters put the navmesh's storeys and stair connectors into `ObjNavEpisode.metadata`, which
  `multifloor.stair_source = "ground_truth"` read. Both are deleted, with the seam on every side: the three call
  sites, `target_projector` on `RPTSearchPolicy` and `FalconYoloBaseline`, the binding in `hm3d/run.py`, `mp3d/run.py`,
  `gibson/run.py` and `gibson/run_development.py`, the `target_projector` property of `HabitatObjNavEnv` and
  `GibsonEnv`, `project_target` and `scene_structure` on the simulator bridge, the `target_navmesh_projection` record,
  `_scene_metadata`, `multifloor/stair_ground_truth.py`, `stair_sightings.py`, `ground_truth_traversal.py`, the
  fallback's up/down rule (`floor_decision.py`, whose geometry half is now `stair_approach.py`), the storey count and
  rank in the oracle's context (`stair_nodes.storey_position`) and the four `MultiFloorParams` knobs only that
  traversal read. `stair_source` accepts `"observed"` only and raises on the old value;
  `configuration()["ground_truth_stairs"]` is the constant `False`. The navmesh stays outside the agent: simulator
  physics, episode setup, scoring, the harness's checks, analysis (`habitat/stair_connectors.py`, now documented as
  analysis-only).
- **What replaced the projection: `core/planning/object_search/world_model/standing_floor.py`.** A standing point is
  a cell of the agent's own occupancy map that (1) was observed free, (2) has every cell within the body radius
  (0.18 m) observed free too -- an unknown cell there disqualifies it as an occupied one does -- and (3) is joined to
  the agent through such cells. One floor, two planners: the default planner is asked first, of the standing cells
  that keep its own standoff (0.25 m) from every *observed obstacle*; the squeeze planner then, of all of them
  (`TargetClosing.standing_planners`; unknown cells cast no standoff). A candidate is moved
  to the nearest such cell within `standing_snap_m` (0.5 m; never across an obstacle the map shows) or dropped, and
  the route is planned to that cell exactly (`PlanRequest.options["exact_goal"]`, new in the weighted A\* planner).
  **Unobserved space is never used**: the closing's planner through unknown cells (`through_unknown`,
  `through_unknown_cost`, the policy's `optimistic_planner`) is removed with the projection -- a config naming them
  is a `TypeError`. What the agent cannot yet stand on it looks at: the walk to the observed floor nearest the target,
  then the footing sweep, and only then a standoff that was set aside -- one a detour long (now measured as the walk
  over standing floor, one Dijkstra instead of an A\* per candidate), or, while floor in front of the target is still
  unobserved, one beside it, behind it or out of the map's sight of it (`closing/map_sight.py`). New counters:
  `squeezed_standoffs`, `standoffs_revoked`, `far_side_standoffs`; `navmesh_goal_xyz` is `standoff_goal_xyz`,
  `through_unknown_routes` is gone.
- **What the stairs lost.** With traversal forbidden (the benchmark default) nothing: the flights masked were already
  the ones the robot's depth measured. With it allowed, floor changes are the coordinator's allowance rule, as under
  the observed source before. The loop's staircases-as-nodes (`rooms/stair_nodes.py`, `MultiFloorSearch.commit`) had
  no source but the navmesh: a portal is offered only when it says which storey its flight leads to, the observed
  discovery does not measure that, so today no staircase is a node. The loop's side is kept and tested on hand-placed
  portals (`tests/test_stair_nodes.py`). `search_node_oracle.STAIRS_SUPPLEMENT` still words a storey RANK that only
  the navmesh's levels supplied; the prompt text is untouched (it is what the evaluated runs sent) and is flagged
  beside it.
- **Tests that keep it out.** `core/planning/object_search/tests/test_no_privileged_geometry.py` fails if anything
  under `sparx_agency/core` can reach the navmesh, the pathfinder or ground-truth storey data -- by import (the
  transitive import closure of every `core` module, lazy and relative imports followed, computed imports registered),
  by name (no identifier may spell the navmesh, the pathfinder's API, the projector seam or the storey structure) or
  through the episode (no `core` module reads `ObjNavEpisode.metadata`; a policy handed the navmesh's structure there
  acts exactly as one handed nothing) -- with every scanner run against planted leaks. On the pre-change tree it fails
  at exactly the audit's findings. The adapters' half is
  `objnav_benchmark_runtime/tests/test_agent_gets_no_scene_geometry.py` (no episode built with metadata, nothing of an
  environment bound to a policy), and the harness's reset check now refuses `navmesh`, `pathfinder`, `stair`,
  `storey` and `floor` metadata keys (`env_contract.PRIVILEGED_KEY_PARTS`). `tests/test_standing_floor.py` pins the
  three rules on small grids and random maps (painting every unknown cell occupied changes no answer) and the
  planner standoff (observed obstacles only).
- **Measured: the same 180 episodes with and without the change.** MP3D v1 val, the fixed single-storey set
  (`mp3d/sets/single_storey_flat_v1.json`: 8 scenes, 19 categories), the tree at `38bd79e8` against this change,
  ground-truth perception, the frontier explorer, seed 17 -- and NOT the configuration of the reported runs: software
  rendering on the CPU (the GPU was not to be taken) and an in-process stub in place of the language model (every
  room `living_room` at 0.8). The absolute numbers are therefore not the paper's; only the paired differences mean
  anything. The pipeline is deterministic: an A/A control repeated 20 of 20 trajectories exactly, so every difference
  is the change's. **Outcome:** success 0.778 -> 0.789 (7 episodes won, 5 lost; exact McNemar p = 0.77), SPL 0.349 ->
  0.346, steps 190.1 -> 196.8 (65 episodes longer, 35 shorter), step limit reached in 13.3% -> 12.2%. **Stuck
  indicators:** refused forward steps 6.67 -> 7.95 an episode (+1.28, bootstrap 95% [+0.12, +2.60]; median
  difference 0, 40 episodes up and 36 down). The increase is a tail in two places. 133 of the 230 extra refusals are
  five episodes of the verification's blind "glimpse" steps -- a loop of the old code too (3 such episodes before, 5
  after), not a matter of where the agent stands. And in the 34 other episodes where a standoff had to be taken with
  the squeeze planner refusals went 6.59 -> 9.97 (median +1); in the remaining 141 episodes 5.65 -> 5.52. The closing
  more often has nowhere to stand from where it is: boxed-in releases 0.072 -> 0.139 an episode (22 episodes against
  10), and the rungs that look first are used -- the walk to the nearest observed floor 0.072 -> 0.217 an episode,
  the footing sweep in 33 episodes against 1. The first version of the floor (two floors, the default planner's a
  0.25 m disc observed free) was run on 85 of these episodes: the squeeze planner took a standoff in 32 of them and
  boxed-in releases were 0.176 an episode; the one floor has 17 and 0.094 on the same 85 (LESSONS 2026-11-15). **The
  five losses:** two STOPs 0.14 and 0.19 m outside the success region, at a second table, after the first -- seen
  over an obstacle, no floor observed beside it -- was released as boxed in (pLe4wQe7qrG/000085, /000109; the navmesh
  runs walked round through unobserved floor); one STOP on the threshold (zsNo4HB9uLZ/000045, 0.10 m);
  Z6MFQCViBuw/000098, the action bound twice in front of a three-cell strip between a sofa and a table that the
  squeeze planner keeps routing through; Z6MFQCViBuw/000093, a 155-action pursuit. **Not measured:** either full
  split, the real language model, HM3D, a second seed.
  `test_ground_truth_floor_transitions.py`, `test_stair_sightings.py` and `rollout_stairs_navmesh.py` went with the
  mode they tested; the fixtures that handed an episode a scene structure place a flight as the coordinator places a
  measured one.
- **Not comparable with earlier runs, on purpose.** The frozen configuration differs (`standing_points` for
  `target_navmesh_projection`, no `through_unknown`), so `--resume` refuses a campaign recorded before the change.
  Every number in `docs/paper/zson_paper.tex` and `docs/progress/entries/047`, `049` was produced WITH the navmesh
  projection and the through-unknown routes (and without ground-truth stairs); the paper now says so
  (Section 5.6) and describes the standing-floor rule. Docs: this file, the `object_search`, runtime, HM3D, harness
  and contract READMEs, `QUICKSTART.md`, `gibson/MULTISTORY.md`, `GROUNDED_VLM.md`.

### Added (2026-11-15, the paper's first complete draft: `docs/paper/zson_paper.tex`)
- **`docs/paper/zson_paper.tex`: one self-contained LaTeX file (27 pages, bibliography embedded, builds clean), written
  against the working tree after the `room_priors.py` removal.** Single-storey method, experimental setup, the two 14B
  full-split results (HM3D v1 val SR 71.0 / SPL 39.8; MP3D v1 val SR 65.6 / SPL 29.1), failure analysis, limitations,
  and appendices: module map, constants (read from `RPTSearchPolicy.configuration()`), both prompts, run records. It
  says what the numbers are NOT: both runs carried the hand-written room priors (Section 5.6), the MP3D run
  (`a77c946c`) also predates the five HM3D-branch closing/navigation commits that the HM3D run (`0dd7d39c`, method code
  identical to HEAD) has, perception is the simulator's and its target labels are the publisher's goal ids, the
  closing snaps standing points to the navmesh, and the rules were developed on the evaluated val episodes. 25 `TODO`
  notes mark what is not done (re-run with the current code first). No ablation and no model-size comparison, by
  request. Supersedes `docs/paper/zero_shot_objectnav_paper.tex`, `docs/method_algorithms.tex`,
  `docs/algorithm_exploration.tex` and `docs/Method.docx`, which describe earlier versions and were left in place.
- **`docs/paper/sync_prompts.py`**: copies the room labeller's and the node oracle's prompts from the code into the
  paper's Appendix C between `%%PROMPT-BEGIN/END%%` markers; `--check` exits 1 when the paper is stale. Run it after
  any prompt change.
### Removed (2026-11-15, the hand-written room priors: `core/mapping/topology/room_priors.py` and every consumer)
- **`room_priors.py` is deleted.** It told the search where things are, by hand: `TARGET_HOMES` (the room types each
  target category is likely and rare in -- rendered into every node-oracle prompt as a `WHERE IT LIVES` line since
  2026-11-09, which the system prompt told the model to copy into its `home` step), `IMPLAUSIBLE_ROOMS` (the hard
  `type_prior` exclusion, off by default, and the room half of the closing's context check, on), `HOME_OBJECTS` (a room
  holding one was kept a node, not finished by sight from outside, a clue that interrupted a walk for an oracle call, a
  cue for a glance, and floored by `home_floor`), `SIGNATURE_OBJECTS` and `GENERIC_OBJECTS` (how strong a room label
  is, and whether a room is labelled at all). Where a target lives is now the node oracle's own judgement.
- **Gone with it:** `LoopSettings.type_prior`, `.home_floor`, `.weak_type_max_openings`; `RoomLabelSettings
  .signature_objects`, `.distinctive_required`, `.generic_evidence_gate`; `SearchNode.home`, `SearchContext
  .target_home`, `HOME_FLOOR` and the `home_floor` argument of the oracle; the `WHERE IT LIVES` sentence of
  `SYSTEM_PROMPT` (so recorded LLM answer stores no longer replay: the prompt hash changed); `target_context
  .room_conflict`; the fallback's type demotion (`split_exits(type_rule=...)`, `type_demoted`); the loop counters
  `weak_type_kept`, `home_object_kept`, `excluded_type`; the home-object component of `discovery.semantic_signature`
  (now `(storey, rooms on offer, their types)`). A config or `--policy-config` naming a removed field is refused
  (`TypeError`), not ignored.
- **What a room label's strength is now:** `strong` on two or more kinds of object and a confident classifier, `weak`
  otherwise -- a count of the evidence. A room named from one bed reads `bedroom?` again; a room of a cabinet alone is
  classified (weakly) instead of staying `unknown`.
- **Audit: the two full-split results on record were NOT clean of this file.** `gt_oracle_2000ep_14b_20261111T114322Z`
  (HM3D v1 val, tree `0dd7d39c`) and `gt_oracle_mp3d_2195ep_14b_20261111T095333Z` (MP3D v1 val, method code of
  `60c9823a`) both ran with `room_priors.py` wired in: `type_prior` was off, but every oracle prompt carried the
  `WHERE IT LIVES` line, `signature_objects` / `distinctive_required` / `generic_evidence_gate` were on, the
  home-object guard, clue and cue were live, and `context_check` read the implausible-room table (the frozen
  `preflight.json` of the MP3D run says so in `method.node_oracle.asks`; the code at both commits calls
  `target_home_line` unconditionally; and every oracle prompt in the two answer stores of the same period that are
  kept on this machine -- `runs/ss_gt_fixes_llm_answers`, 612 of 612, and `runs/single_storey_v1_llm_answers`, 496 of
  496 -- carries the line; the two runs' own stores are on the office machine and were not opened). Their numbers
  are results of the method WITH the table; a table-free number needs a re-run.
- **Still hand-written, deliberately left and listed** (`core/planning/object_search/README.md`, "No hand-written
  semantic priors"): the two prompts (the oracle's system prompt names typical homes for the six HM3D categories and
  gives numeric bands), the code-side scaling of the oracle's numbers by exploration state, `CLASS_HEIGHT_BANDS`,
  `SUPPORT_SURFACE_CLASSES`, `SHADOWLESS_CLASSES`.
- Tests: the suites of the three packages pass as before the change (1,697 passed; the one failure,
  `test_monitor.py::test_finished_is_judged_by_the_pid...`, fails on the untouched tree too). Tests of the removed
  behaviour were replaced by tests of its absence (`test_no_module_of_hand_written_room_priors_exists`,
  `test_the_oracle_takes_no_hand_written_prior_about_where_the_target_lives`, ...).
### Added (2026-11-13/15, the stairs-on arm of the MP3D 14B run and the comparison -- docs/progress/entries/049)
- **`runs/gt_oracle_mp3d_2195ep_14b_stairs_20261112T130740Z`: the same split, model, seed and settings as run 1 with
  `allow_stair_traversal` TRUE, replaying run 1's answers (15 % of its questions): SR 56.4 % (CI 54.3-58.4), SPL 25.7 %,
  DTG 4.33 m -- 9.3 SR points UNDER stairs off (CI -11.2 to -7.4, McNemar p < 0.001), +132 / -336 flips.** Same storey
  77.4 -> 65.2 % (-12.1 pts), cross storey 11.1 -> 14.9 % (+3.9 pts, p 0.09). One mechanism: 541 of run 2's 619 wrong
  STOPs are the observed-stairs traversal's `SAFE_HALT` (`retreat_budget_exhausted_on_unconfirmed_support`), a STOP the
  scorer takes as the answer at a median 9 m from the goal, half the budget left, on 91 of 200 sampled halts within
  0.12 m of the source floor's height -- the `RETREAT`-never-exits gap of HANDOFF.md 2026-10-01, measured. 149
  traversals completed in 601 attempts (129 episodes, SR 41 % among them); 26 of 58 cross-storey successes followed a
  floor arrival. 11.3 h at 194 episodes/h. The fix to make before stair traversal is measured again is written in the
  entry: a spent retreat standing on the storey it left should release the portal and resume the search, not STOP.
### Added (2026-11-13, stair traversal on against off on HM3D v1 val: the same 2,000 episodes, paired -- docs/progress/entries/048)
- **Run 2, `allow_stair_traversal: true`** (`runs/gt_oracle_2000ep_14b_stairs_20261112T115622Z`, `hm3d/sets/policy_stairs_on.json`,
  everything else as Run 1, Run 1's LLM answers replayed): complete in 12 h 31 min, **SR 63.0 %, SPL 36.8 %, DTG 3.12 m**.
  Paired against Run 1: **overall −8.0 SR points** [−10.0, −5.9]; **cross-storey goals +13.5** (11.2 → 24.7 %, DTG 12.7 → 8.3 m);
  **same-storey goals −12.5** (83.5 → 71.1 %). 406 episodes changed storey (mean 0.28 transitions an episode; 0.63 on
  cross-storey goals, 0.20 on same-storey ones, where none was needed); 546 of the 740 failures end inside the traversal
  machinery -- 358 `SAFE_HALT` STOPs and 188 step limits mid-traversal, 355 of them same-storey episodes whose traversal
  started at a median action 63 before the storey was searched.
- **`hm3d/compare_campaigns.py`**: two campaign roots paired episode by episode -- SR / SPL / DTG per arm and the paired
  B − A differences with bootstrap intervals and McNemar, overall, per category, by storey and per category × storey;
  storey transitions per episode per arm; flips with the mechanism each arm failed by; and a pairing-integrity check on
  the episodes in which the change never acted (including that A's spawn-floor guard painted nothing, since the guard is
  inert once traversal is allowed).
- **`campaign_breakdown.py` reads traversal-on runs**: `stop_safe_halt`, `limit_in_traversal`, `limit_no_transition`
  tags from `building.allow_stair_traversal` / `phase` / `completed_transitions`, a storey-transitions-per-episode
  section, and single-scene roots.
### Added (2026-11-12, the full MP3D val split with the 14B in both roles under ground truth, stair traversal off -- docs/progress/entries/049)
- **`runs/gt_oracle_mp3d_2195ep_14b_20261111T095333Z`: SR 65.6 % (CI 63.6-67.6), SPL 29.1 %, DTG 3.82 m over all 2,195
  episodes** (entry 043's 3B run with none of the fixes of entries 044-046: 52.1 % / 23.6 % / 4.42 m); at ApexNav's
  0.2 m 66.7 % / 29.7 %. Same storey (1,807 episodes) 77.4 % / 34.3 % / 1.91 m, cross storey (388) 11.1 % / 4.6 % /
  12.69 m with stair traversal off. 754 failures: 392 step limits after a released lock, 222 never seen (341 of the
  668 limits are cross-storey), 54 wedged, 44 near-miss STOPs (23 beside the object), 21 through a barrier, 12 the
  sensor's phantoms. 27.2 h at 80.7 episodes/h, LLM-bound (50,149 answers recorded, 22.8 an episode, all replayable);
  the card's compute 37 % busy, its memory 85 %. Every episode's step log recorded without video. The stairs-on arm
  (`gt_oracle_mp3d_2195ep_14b_stairs_20261112T130740Z`, replaying run 1's answers) runs next.
### Added (2026-11-11, the HM3D v1 val ground-truth run with the 14B in both LLM roles: what it needed to run unattended and be read)
- **A per-step heartbeat for the stall watchdog** (`progress.ProgressTracker.heartbeat`, `heartbeat.json` beside
  `progress.json`, rewritten at most every 30 s from `ProgressEnv.step`; `scene_pool.HEARTBEAT_FILES` watches it):
  without `--record` a scene child wrote nothing between episodes (`live.json` is the recorder's), so the 30-minute
  watchdog would have killed any live worker whose episode ran longer -- the Nov-4 run had episodes of 45-66 minutes.
- **`utilization_log.py`**: the machine sampled every N seconds into a CSV for the life of a campaign (GPU
  utilisation and VRAM, each GPU process's VRAM, CPU, RAM, disk, load; `campaign_monitor.resources` underneath) and
  `--summarise` for the means and percentiles a run report quotes.
- **`hm3d/campaign_breakdown.py`** and the shipped level survey **`hm3d/sets/hm3d_v1_val_levels.csv`** (1,652
  same-storey goals, 348 cross-storey): SR / SPL / DTG with intervals overall, per category and same- vs cross-storey,
  one failure-mechanism tag per failed episode from the full records, the last closing releases, the LLM's health,
  the storey events, per-scene runtime and the utilization summary. Read against the Nov-4 3B run it tags 579 failures:
  300 cross-storey goals (stairs off), 179 never seen, 60 seen and not converted, 25 wedged, 15 failed STOPs.
- **Operating findings of the launch** (RTX 4090, 24 GB): a rendering GT worker takes ~740 MiB of VRAM (560 before
  its first frame); the 14B runner takes 15.1 / 16.6 / 18.2 GB at 4 / 5 / 6 Ollama slots (+1.5 GB a slot at 8,192
  tokens) and the 3B 3.5 GB at 3-4 slots; with the 3B beside it, 4 slots + 8 renderers left ~450 MiB and the 14B
  runner died of `CUDA error: out of memory` loading a kernel on its first busy request, after which Ollama evicted
  one model for the other on every call. The run went out as 14B-only, 4 slots, 9 workers: 22.2 GB steady.
- **The full HM3D v1 val split under ground-truth perception with `qwen2.5:14b-instruct` in both LLM roles**
  (`runs/gt_oracle_2000ep_14b_20261111T114322Z`, `docs/progress/entries/047-hm3d-gt-oracle-14b-val.md`): 2,000
  episodes in 20 scenes, complete, 23 h 59 min, 0 agent errors, 0 stalls, 1 oracle refusal in ~50,000 calls.
  **SR 71.0 %, SPL 39.8 %, DTG 3.15 m**; same-storey goals (1,652) SR 83.5 % / SPL 47.2 %, cross-storey goals (348)
  SR 11.2 % / SPL 4.8 % with stair traversal at its default (off). Of 581 failures: 307 cross-storey goals, 160 same-
  storey targets never seen, 61 seen and not converted, 36 wedged, 17 scored STOPs. GPU memory 86-90 %, GPU compute
  39 % (the 14B's JSON sampling pins one CPU core per request; median call 9.0 s), 9 workers at 96 % occupancy.
### Changed (2026-11-11, one Habitat adapter base: each dataset adapter holds only what is its own)
- **`habitat/env.py::HabitatObjNavEnv`** holds the habitat-lab ObjectNav episode, the view-point measurement, start
  validation against the publisher's geodesics, navmesh provenance and the oracle hand-off (frames, depth, pose,
  camera, instance boxes) once; `hm3d/env.py` and `mp3d/env.py` subclass it with their `BENCHMARK` name and, for
  MP3D, the one override -- an instance is named by both its `.house` category and its mpcat40 class
  (`_annotations`). The two files were ~380 near-identical lines each; they are ~40. The dataset interface the base
  reads: `episode_ids`, `episodes`, `scene_assets`, `goals_for`, `goal_instances`, `semantic_scene_dataset_config()`
  (`HM3DDataset` gains that name beside its `annotated_scene_dataset_config`).
- **`habitat/entrypoint.py`** holds what both `run.py` entrypoints shared below their dataset: `run_main` (the whole
  `main`, given the adapter's `parser`, label table, protocol registry, `prepare` and report writer),
  `method_defaults(protocol)`, the geodesic preflight check, the oracle liveness gate, the runtime record, the GPU
  gate, the live hooks and the frozen-config constants. Each `run.py` keeps its `parser`, `prepare` and
  `ground_truth_issues`, and re-exports the shared names it always had.
- **Shared campaign code left the HM3D adapter**: `hm3d/geodesic.py` -> `habitat/geodesic.py` (habitat-lab's
  measure), `hm3d/scene_pool.py` -> `scene_pool.py`, `hm3d/smoke_plan.py` -> `smoke_plan.py`, and the child-argument
  and episode-plan rules of `hm3d/campaign.py` -> `campaign_common.py`; the MP3D campaign, shards and launcher import
  from the shared homes, and `hm3d/` keeps re-export shims so every old import and `python -m` path still works. No
  module under `mp3d/` or `habitat/` imports from `hm3d/` any more.
### Changed (2026-11-11, the HM3D closing work of 2026-11-05 merged into the infra branch beside the MP3D closing work: one algorithm, the rules combined)
- **A step-out frame is the inspection's by either side's rule** (`closing/target_closing.py`): HM3D kept the
  inspection on a centred frame within one forward step of the limit while the step rule's budget lasts, arrived or
  not (`step_out_inspections_kept`); MP3D kept it from an ARRIVED standoff whatever the budget and added the
  side-clipped withhold and the resumption cap. Merged: the resumption is skipped when either applies, then the
  clipped withhold, then the cap, then the resumption as before.
- **The standoff ring is searched ring, arc, planner, bearing set, offset** (`closing/target_path.py`): MP3D's
  rings (the facing normal's at the standoff radius and closer, then the agent's, then the widened) each tried HM3D's
  way -- the front arc with every planner, the unknown included, before any point beside or behind with any -- and a
  route longer than `standoff_detour_factor` times the range plus the slack remembered as the fallback across every
  ring, taken only when nothing else has a route (`standoff_detours`; the candidate remembers which ring it came from,
  so `frontal_standoffs` stays right). The HM3D "old order" regression now turns `frontal_standoff` off too: a flat
  target's normal ring already puts the front first.
- **The overshoot reads the ceiling that applies to the target** (`TargetClosing._terminal`): HM3D turned the
  planar overshoot off under a view-point evaluator (`stop_overshoot_steps`/`stop_overshoot_max_steps` 0: on 70
  recorded STOPs it never helped), MP3D gave a RAISED target more steps (`stop_overshoot_elevated_max_steps`,
  `elevation_overshoot`: its view points stand closer). Merged: an object on the floor reads the planar ceiling, a
  raised one its own, so the view-point defaults no longer silence the elevated steps and both at 0 is the overshoot
  off for every target. `method_defaults` moved to the shared `habitat/entrypoint.py` and applies to **MP3D too**
  (its scoring rule is the same); the MP3D numbers recorded before the merge were taken with the planar overshoot on.
- **The scrape guard excuses a refusal only in a door-named channel, counted by place** (`navigation/scrape_guard.py`):
  MP3D's `doorways_only` first, then HM3D's place-keyed count (`same_place_m`) and the closed memory
  (`closed_memory_actions`); the stats carry both `no_door` and `closed`.
- **`hm3d/run.py`'s shared helpers moved to `habitat/entrypoint.py`** (`method_defaults`, `geodesic_agreement_issue`,
  `oracle_alive`, the geodesic constants, `RESUMABLE_CONFIG_CHANGES`, `RUNTIME_PACKAGES`); `hm3d/run.py` re-exports
  them and `mp3d/run.py` imports them there instead of from the HM3D adapter.
- Everything else of both sides merged without overlap: HM3D's plain-A\* lattice, squeeze-planner transit, clipped
  bearing start, fewer look-downs, terminal re-sighting, `history_stop_max_unseen`, nested-default merge in the
  factory; MP3D's tiny-room fragments, frontier tri-state, oracle priors, labeller hysteresis, face-normal standoff,
  bounded pursuit, on-object oracle filter.
- **Regression check** (2026-11-11, ground-truth perception, qwen2.5:3b for both LLM roles, 4 scenes x 2 episodes
  per benchmark, each pre-merge branch on its own tree against the merged tree; the merged arm replayed the branch
  arm's recorded answers and asked live where the prompt had changed). **MP3D** (the single-storey set's first two
  episodes of 2azQ1b91cZZ, zsNo4HB9uLZ, Z6MFQCViBuw, pLe4wQe7qrG): SR 7/8 -> 7/8, SPL 0.331 -> 0.377, no flips; 79
  of 197 oracle answers were new questions. **HM3D v1** (the 2026-11-05 regression set: 00843/000000, /000007,
  /000023, 00853/000017, 00880/000007, /000009, 00891/000006, /000027): SR 6/8 -> 7/8, SPL 0.221 -> 0.398; gained
  00843/000000 (500 -> 101 actions) and 00880/000009 (the clipped television), lost 00843/000007 (nine rooms searched,
  the toilet never seen: a search-order difference under live answers -- the `WHERE IT LIVES` line makes every node
  oracle call a new question, so 279 of 554 answers were live). `instances_with_boxes` equals `annotated_instances`
  on every HM3D and MP3D scene (351-898): the on-object filter has its boxes on both.
### Changed (2026-11-11, the MP3D work merged into the benchmark infra: the ground-truth fixes reach every Habitat adapter)
- **The oracle's on-object filter applies on HM3D as on MP3D** (`hm3d/env.py`): the HM3D adapter now hands the
  `GroundTruthSource` the annotations' instance boxes in ENU once per scene and each frame's depth, pose and camera,
  as the MP3D adapter did -- until now an HM3D run under `--perception ground-truth` recorded `on_object_m: 0.25`
  while the filter judged nothing and the method STOPped on whatever the semantic mesh painted. The conversion is
  written once, `habitat.simulator.semantic_boxes_enu` (both adapters call it; `enu_box` is `habitat_pose`'s own
  mapping, tested as such), and `HM3DEnv` uses the shared `habitat_position` instead of its own sign flips.
- **A filter with nothing to judge says so** (`GroundTruthDetector.last_diagnostics["unjudged"]`,
  `unjudged_instances`): an instance reported as rendered because the frame came without depth, pose or camera or the
  scene without a box for it is counted in every frame's detector diagnostics (recorded through the perception
  cycle's `detector_evidence`), beside the scene report's `instances_with_boxes` -- a missing hand-off is read off
  the record, not inferred from a score. The per-frame rule is unchanged: such an instance is still reported.
- **The stall watchdog reaches the HM3D campaign** (`hm3d/campaign.py --stall-minutes`, 30 by default, 0 disables;
  `hm3d/run_benchmark.sh --stall-minutes`): `scene_pool.run_concurrently(stall_s=...)` was shared code that only
  the MP3D campaign called. A worker that loses its main thread is the harness's failure on any dataset.
- **Both launchers take `--ground-truth-on-object-m` by name** (`GT_ON_OBJECT_M` in `benchmark_services.sh`), as
  they took `--ground-truth-min-pixels`; the HM3D launcher documents the knob.
- **Every goal category of every registered benchmark has a WHERE IT LIVES prior** -- pinned by a test over
  `default_label_mapper_registry()` (`tests/test_room_scans.py`), so the oracle briefing written for MP3D's fifteen
  unprimed categories stays a contract on HM3D, Gibson and whatever table is registered next.
### Added (2026-11-11, the full MP3D val run with the 14B: log-only recordings, the shard watchdog, a resource monitor)
- **`mp3d.run --record-log-only` / `run_benchmark.sh --record-log-only`** (`recording.EpisodeRecorder(video=False)`,
  `evaluation.run_evaluation(record_video=)`): every episode's `steps.jsonl`, `trajectory.csv`, map journal and
  `metrics.json` are written exactly as with a video, and no dashboard frame is rendered or encoded -- no `video.mp4`,
  stills, `final.jpg`, `latest.jpg` or review page, no FFmpeg. `live.json` still updates on every action (the stall
  watchdog's heartbeat). The full split's failure mechanisms can now be read from the logs of a run that recorded
  no video; a video of a chosen episode is re-made by replaying it with its recorded LLM answers.
- **The stall watchdog covers sharded campaigns** (`shard_pool.run_pool(stall_s=)`, `PoolJob.directory`,
  `job_heartbeat`, `sharded_campaign.run_sharded_campaign(stall_s=)`; `campaign.py --stall-minutes` now reaches the
  sharded path, default 30): a shard runs in its own session and one whose `live.json` / `progress.json` /
  `episodes.jsonl` / console log are silent for the period is killed with its session and recorded as exit -9, so the
  pool moves on and a resume of the root continues the shard. Before this, `--stall-minutes` was accepted and ignored
  with `--shard-episodes`.
- **`single_storey_failures.recordings_by_episode` reads a merged scene's `shard_*/recordings/`** (the merge moves
  the rows, not the recordings), so the failure tagging runs on a sharded campaign root as it does on a per-scene one.
- **`mp3d/storey_report.py`**: one or several campaign roots tabulated overall and on the same-storey against the
  cross-storey episodes (entry 043's +-0.5 m view-point rule, and the level survey as a second reading), per category
  and per category by storey, paired against the first arm (bootstrap dSR / dSPL, exact McNemar, flips), with the
  storey transitions per episode -- the floor atlas's `floor_arrival` events and completed stair traversals from the
  full records, the settled-level changes of each recorded trajectory, and how many cross-storey episodes ended on
  the goal's storey. The table of the stairs-off / stairs-on comparison. Tests in `tests/test_mp3d_storey_report.py`.
- **`objnav_benchmark_runtime/resource_monitor.py`**: `record out.csv --interval 15` samples GPU utilisation, VRAM,
  CPU, RAM, load and the live worker count; `summarize out.csv [--since-utc ..]` gives mean / median / peak of each
  and the share of samples the GPU spent above 80 % -- the utilisation line of a campaign's report, which until now was
  read off a terminal.
- **Measured for the 14B on the 4090** (`mp3d/README.md`): `qwen2.5:14b-instruct` at 4 slots and 8k context takes
  15.1 GB; the client's `format: json` caps it at ~0.5 answers/s (0.36 against 0.93 requests/s with and without the
  format on an idle card at 4 concurrent; 8 slots, 12 workers or a q8_0 KV cache change nothing, the grammar's sampling
  being CPU work serialised in Ollama's runner), so 7-8 MP3D renderers (270-1,180 MiB each by scene) saturate it at
  ~110 episodes/h with the card's compute ~40 % busy and its memory full.
### Added (2026-11-10, MP3D single storey: a fixed set, a deterministic baseline, the flat-p ablation, and the single-storey fixes -- docs/progress/entries/046)
- **A fixed single-storey evaluation set and its launcher** (`mp3d/single_storey_set.py`, `mp3d/sets/single_storey_flat_v1.json`:
  180 same-storey episodes in 8 flat houses, the split-level scenes excluded, seeded; `mp3d/run_single_storey.sh <name>`
  runs a campaign on it from a FROZEN rsync of the source tree with a shared, episode-scoped LLM answer store, so a
  re-run of unchanged code is bit-identical -- 14/14 episodes, 0 live calls -- and two runs part only where a change
  asks something new; `REPLAY=` serves earlier stores read-only). `single_storey_report.py` pairs the arms (SR, SPL,
  per-category, bootstrap dSR, McNemar, flips); `single_storey_failures.py` tags each failure's mechanism from its
  recording and the published goals, separates the ground-truth sensor's phantom ids from the method's failures, and
  reads a near-miss STOP against the goal's `.house` box (beside / in the band / far).
- **The flat-p ablation** (`LoopSettings.flat_p`, `SearchNodeOracle.flat`, `source="flat"` through the runtime): every
  node at p 0.15 instead of the model's number. SR 59.4 % against the baseline's 62.8 % (dSR -3.3 pts [-6.7, +0.0],
  p 0.11, 131/180 episodes identical): the 3B oracle's probabilities are worth about three points on this set.
- **`--policy-config` reaches the room-search loop and the labeller** (`RPTSettings.room_search`, `room_labels`:
  JSON objects of `LoopSettings` / `RoomLabelSettings` overrides), and the stall watchdog kills a scene worker that
  has not written for `--stall-minutes` (30) and records exit -9 for the resume (`scene_pool.run_concurrently`).
- **`llm_answer_cache --resolve-conflicts <dir> [--keep earliest|latest]`** repairs a store two concurrent runs
  recorded the same key into with different answers (the loader refuses such a store).
### Fixed (2026-11-10, the single-storey failures the recordings explain)
- **A refusal is excused as a doorway scrape only in a channel a confirmed door landmark names**
  (`ScrapeSettings.doorways_only`): the geometry-only channel test treated every squeeze of the route between
  furniture as a doorway, excused the refusals that would have mapped the furniture, and 2azQ1b91cZZ/000058 released
  a table in plain view 3 m away as "went nowhere" and lost it 26 more times. Two wedged episodes of the baseline
  succeed; none regress.
- **Two labeller rules, written, measured and turned OFF by default** (`RoomLabelSettings.dining_min_chairs` 0,
  `shape_rule` False; 3 and True enable them): a `dining_room` verdict on fewer than three chairs read `unknown`
  (the benchmark's `table` is any table, and the 3B model had made 225 dining rooms of desks, side tables and coffee
  tables), and an empty passage-like room was `hallway` by rule without a classifier call (257 calls on rooms with
  nothing in them, answered at random). The briefing got cleaner (dining rooms 225 -> 122, calls -17 %) and the
  search got no better: +5 / -7 flips together, and alone each trended negative (65.4 % and 67.9 % against gt-base's
  71.2 % and 73.6 % on the first 52-53 episodes of its arm). A workaround for a 3B model the user is replacing; off.
- **The STOP overshoot stands closer under a raised target** (`TargetClosingSettings.elevation_overshoot`,
  `elevation_min_m` 0.30, `stop_overshoot_elevated_max_steps` 3): the within-radius test reads the memory's planar
  range lifted by the target's lowest measured point above the base. MP3D's view points of an object raised 0.3 m or
  more reach 0.84 m from its box (median over the set's 242 raised goals) against 1.02 m for one on the floor,
  because the generator keeps only the candidates the object shows from. Two near-misses succeed, none regress.
- **The standoff of a planar target is in front of it** (`frontal_standoff`, `frontal_off_normal_deg` 35,
  `frontal_inner_radii_m` (0.65, 0.45), `frontal_max_deferrals` 12; `perception/face_normal.py`,
  `target_path._ring_bearings`, `TargetClosing._frontal_deferral`): the ring started from the agent's own bearing, so
  a picture first seen at a grazing angle was approached and STOPped on from beside it -- inside the 1 m band but
  0.1-0.5 m from every view point, which for a thin wall object lie within 0.26 m of its width (8 of the baseline's 9
  near-misses stood beside their object, 45-70 degrees off its normal). A sighting whose mask is a standing plane
  (every depth pixel fitted: spread under 5 cm, normal near horizontal) gives the lock the plane's facing normal (the
  mean over sightings); the ring is tried about it first, at the standoff radius and then closer, the agent's ring is
  the fallback, and an in-range frame from more than 35 degrees off the normal walks on to the frontal standoff for at
  most 12 actions before the STOP rules apply. The pursuit's walk-back remembers the normal. A first version read the
  "front" off the map's free cells, which point to the explored side and move as the map fills; replaced the same day.
- **A target glimpsed without depth is one target, and the pursuit's chase is bounded for the episode**
  (`PursuitSettings.bearing_merge_m` 1.5, `bearing_max_walk_actions` 48, `max_total_walk_actions` 150;
  `closing/target_pursuit.py`): in Z6MFQCViBuw's atrium house a chair on the far side of the void was seen in RGB
  without depth again and again, every abandoned bearing approach remembered a new point 6 m along its bearing and
  each was walked toward for up to 120 actions -- eight single-storey episodes spent 275-453 of 500 actions "walking
  back to the released chair" (ground truth, filter on). A bearing target near a remembered one of the same label is
  the same object (its guess updated, its attempts kept), a bearing pursuit walks 48 actions at most, and all the
  episode's pursuits together 150.
- **The frontal-standoff rule applies to RAISED planes only** (`frontal_min_elevation_m` 0.6): fitted on every
  standing plane it fired on 90 of 180 locks -- a table's apron, a chair's backrest, a cushion, a floor cabinet --
  whose view points are all round, and cost 1,358 actions and 1.7 SPL points over them for +8 / -3 episodes; gated to
  a lock whose lowest point stands 0.6 m or more above the base it touches 17 locks (12 pictures) for +3 / -1 at no
  SPL cost.
- **The ground-truth detector's on-object filter is ON by default** (`--ground-truth-on-object-m` 0.25;
  `GroundTruthDetector`, `GroundTruthSource.publish(depth, pose, camera)`, `HabitatRGBDSimulator.semantic_boxes`):
  a rendered pixel counts as an instance's only where the frame's depth puts it on the instance's annotated box, so
  the ids MP3D's semantic mesh leaks through walls and stair risers (32 of the baseline's 67 failures: 23 slivers under
  10 cm tall, 9 whole boxes) are not reported. Perception layer only, the user's decision; nothing in the navigation
  reads it.

### Fixed (2026-11-09, the MP3D run's graph and oracle bookkeeping, and the oracle's priors for every MP3D category)
- **The node oracle is briefed with a WHERE IT LIVES line for every target category**
  (`room_priors.TARGET_HOMES`, `target_home_line`, `SearchContext.target_home`): the 3B model had no prior for the
  fifteen MP3D categories HM3D lacks and wrote "a table is in the bathroom", "a stool in the hallway"; `TARGET_HOMES`,
  `IMPLAUSIBLE_ROOMS` and `HOME_OBJECTS` now cover all 21 under the dataset's names and the accept labels
  (`_TARGET_ALIASES`), so `target_context`'s room-conflict penalty is live for those categories too.
- **A room under `LoopSettings.tiny_room_m2` (1 m2) with no live frontier is a fragment whatever its doors**
  (`room_scans._status_without_frontier`), and a home object of the target credited to it does not keep it a node
  (the `home_object_kept` exception is for a room whose jamb may hide the target; a sliver has none): slivers of
  0.2-0.8 m2 were offered in 40 % of the oracle calls and were the model's first choice in one in eight. A sliver with
  frontier left stays a node.
- **The HOUSE line's frontier statement is a tri-state** (`house_context.frontier_state`: reachable / only slivers /
  unreachable from here / none; the loop's `_frontier_state` reads "unreachable" off the rooms' own frontier counts,
  since the inventory holds only what the agent's free component reaches): a boxed-in agent had told the model "no
  unexplored frontier left", and the phrase flipped back in 167 of 266 episodes that used it. The oracle shows a node
  under a square metre as `size=under 1m2`, and the HOUSE line is built AFTER the labels are refreshed, so it no
  longer reads "rooms found: none yet" over a node list that says `type=bathroom`.
- **A mid-walk reconsideration is a clue, not any new object class** (`discovery.semantic_signature`,
  `semantic_change`): a room on or off the offer, a room's type, a home-object kind new to the storey. Geometry alone
  -- the watershed renumbering, a merge, a sliver the exclusions withhold at once -- waits for the loop point when the
  rooms on offer are the ones valued at the last call or that call was under `reconsider_min_actions` (8) ago,
  without arming the loop point itself (armed, the SELECT state spent the call on the same action). The same window
  holds SELECT's "a room without a probability buys a call" while a valued room stands (`unvalued_waits`), and a
  release for `unreachable` / `blocked` / `transit_timeout` re-selects on the standing numbers instead of re-asking
  (`releases_unasked`): QUCTc6BB5sX/000069 had asked 27 times in 230 actions inside a 5 m patch, on consecutive
  actions over the same two nodes. `reconsiders_spaced` counts the waits; `semantic_replan` events carry `clue` and
  `since`.
- **The room labeller holds a verdict that differs without a new kind of evidence until it repeats, and a verdict on
  a kind LOST to the partition for `refresh_steps` actions** (`RoomLabelSettings.relabel_hysteresis`, `held`,
  `room_labels_held` in the episode info; the hold survives the partition changes the watershed makes on most
  actions): rooms had swung bedroom / living room / dining room up to eight times an episode on the same objects, and
  a 10 m2 dining room read bathroom for the 40 actions its table stood across a moved boundary; each swing was a
  semantic clue and an oracle call. A new kind of object relabels at once.
  `docs/progress/entries/045-mp3d-layer4-and-category-priors.md` for the re-run.
- **`mp3d/campaign.py main` passes `runner=subprocess.run` at call time**: bound as the default it ignored the tests'
  monkeypatch, and `test_a_sweep_whose_scenes_crashed_does_not_exit_zero` spawned a real `run.py` against the live
  Ollama, which hung the rig for 20 minutes while a campaign kept the server busy.
### Fixed (2026-11-09, the MP3D "seen, not converted" failures: the closing's share, measured on 15)
- **The target closing's re-sighting is the projection nearest the 3-D memory** (`TargetClosing._best_projected`), a
  whole box before a sliver, instead of the most confident box: with several instances of the class in one frame at one
  confidence (the ground-truth detector) the range was measured on the neighbour and the centring turns chased a sliver
  at the frame's edge (zsNo4HB9uLZ/000078: 211 actions to the STOP, now 48).
- **An arrived standoff with the fresh range at most one forward step beyond the limit is the inspection's**, whose
  step rule takes the one MOVE_FORWARD; the arrived route's idle result was executed as a TURN and the lock was lost as
  "no safe path to the target 1.1 m away". The closest-approach STOP still needs a widened standoff.
- **The inspection resumes the approach at most `max_inspection_resumptions` (3) times per lock, never on a
  side-clipped box.** Four new per-episode counters; `docs/progress/entries/044-mp3d-seen-not-converted-closing.md`
  for the 15-episode re-run and what the recordings say the other mechanisms are (the spawn-floor guard after raised
  and stair starts, wedges, near-miss standing spots, a STOP from the wrong level, ground-truth slivers).
### Added (2026-11-05/06, MP3D v1 val under ground-truth perception, measured)
- **A sharded MP3D campaign** (`objnav_benchmark_runtime/mp3d/campaign.py --shard-episodes N`, `shard_plan.py`,
  `shard_pool.py`, `sharded_campaign.py`, `campaign_status.py`, `run_benchmark.sh --llm-servers N` / `--status`):
  - Every scene is cut into shards, so no scene becomes the tail.
  - Fresh shards are dealt over several native Ollama servers, because one server answers 0.7-0.8 requests/s
    whatever its slot count.
  - A shard starts only while used VRAM plus one renderer's 1,400 MiB stays under 88 % of the card, and running
    work is never paused for VRAM.
  - A merged scene reads like an unsharded one.
- **The full objectnav_mp3d_v1 val split under ground-truth perception, 2,195 episodes in 11 scenes**
  (`runs/gt_oracle_mp3d_2195ep_20261105T141854Z`, `docs/progress/entries/043-mp3d-gt-oracle-val.md`):
  - **SR 52.1 %, SPL 23.6 %**; at ApexNav's 0.2 m, 53.3 % / 24.2 %. DTG 4.42 m, agent errors 0.
  - The 388 episodes (17.7 %) with no goal on the start storey score SR 7.0 %, against 61.8 % for the rest, because
    stair traversal stays off by default.
### Fixed (2026-11-05, evening: the goal reached and not STOPped on, 00891/000006 re-run)
- **A centred frame one step outside the limit is the inspection's step to take, not the approach's route**
  (`closing/target_closing.py`, `step_out_inspections_kept`): actions 167-207 at 0.04 m from the goal. The inspection
  began on a left-clipped frame that measured 1.046 m against the 1.05 m limit and turned to centre the box; the
  centred frame measured 1.071 m, the resumption rule ("seen fresh beyond the range: walk, not look") cancelled the
  inspection and the approach's route to the standoff began with the turn back -- forty-one actions of
  TURN_LEFT/TURN_RIGHT inside the success region, then the stall release ("went nowhere"). The inspection already
  owns a fresh range within one forward step of the limit (one MOVE_FORWARD puts the next frame inside it, three
  steps per inspection); the resumption rule now leaves such a frame to it, and the budget spent hands it back.
- **A strong box within the terminal range is the locked target, whatever the anchor says**
  (`TargetClosingSettings.terminal_resighting`, `terminal_resightings`): actions 259-292 at 0.08 m from the goal, the
  only toilet in the room whole and centred in twenty frames, every one dropped by the association cylinder: it is
  drawn about the ANCHOR, the first far frame's centroid (the tank through a doorway, the wall's depth mixed in), and
  the close frames' centroid (the bowl, pitched down) lay 0.41 m across its ray and 0.35 m below it, 3.5 cm outside
  the 0.50 m radius -- "inspection from 0.9 m saw nothing", released. While the lock stands a strong box of the class
  whose near edge measures within the limit plus one step is a sighting (the benchmark counts any instance within
  the range); farther off the cylinder judges as before.
- **The STOP overshoot is off under the HM3D view-point evaluator** (`hm3d/run.py` `method_defaults`,
  `target_closing.stop_overshoot_steps` and `stop_overshoot_max_steps` 0 there; `policy_factory.build_policy` merges
  a nested default group with the policy config's field by field): with the two fixes above the re-run of
  00891/000006 decided its STOP at 0.02 m from the goal and the overshoot's step toward the toilet -- the margin
  walk tuned on 2026-10-14 for Gibson's footprint evaluator -- ended blocked against it at 0.16 m, a failure. HM3D
  scores the geodesic distance to the goal's view points, where the deciding frame already stands: on the 70
  recorded HM3D STOPs since 2026-10-28 the step never brought a frame from outside 0.1 m to inside it, cost 2-14 cm
  of margin in 18 and gained it in 2. The Gibson default stands. Disclosed in `hm3d/PROTOCOL_AND_TUNING.md` as the
  one `run.py` default judged on recorded validation episodes.
### Fixed (2026-11-05, afternoon: two more bugs read off the re-run of the ten regressions)
- **A standoff reached by a detour is the ring's last choice** (`closing/target_path.py`,
  `TargetClosingSettings.standoff_detour_factor` 1.5, `standoff_detour_slack_m` 1.0, `standoff_detours`): the memory
  of 00891/000006 placed half a metre into the bathroom wall, the ring's front-arc point at 60 degrees landed in the
  wall and the navmesh projection moved it to the room beyond -- a 6.6 m known-floor route for a target 3.4 m away,
  walked out of sight round the outside of the bathroom to inspect a wall (the front-arc order of the morning could
  not see it: the projected point counted as front). A route longer than the factor times the range plus the slack
  is kept as the fallback while the other bearings and planners are asked; the through-unknown route by the doorway
  wins, and the detour is walked only when nothing else has a route.
- **A target cut by the frame's edge is looked at, not dropped** (`closing/target_closing.py`
  `TargetClosingSettings.bearing_clipped`, `bearing_clipped_starts`): the border gate drops a sliver at an image edge
  -- its label and depth centroid are unreliable -- and dropped it silently; 00880/000009 walked past its television
  in ten consecutive frames (actions 327-337), the box cut by the TOP edge for nine of them because the doorway
  look-down held the camera 30 degrees down, then by the left edge as it walked by, and never turned its head. A
  strong sliver now begins the bearing approach: the camera is tilted toward the cut edge (levelled under the
  look-down) and the body faced to the box, so the next frame shows the object whole and the ordinary start is
  judged on it; the bearing's bounds, its two-frame persistence rule and its retry memory apply unchanged.
- **Fewer look-downs** (`navigation/settings.py` `LookDownSettings.approach_m` 0.9 m, was 1.5; `max_width_m` 0.9 m;
  `done_radius_m` 1.0 m): on the 214 look-downs of twenty recorded episodes they were 7 % of all actions; 69 were
  released before the agent reached the channel because the route changed first (requested from a median 1.7 m to
  the throat, against 1.1 m for those that did reach it), 49 were at channels a metre wide, 50 within a metre of a
  throat already looked down at -- and the plan through the channel changed on the refreshed map in 21 % of them,
  the same share whatever the width, the unknown floor about the throat or whether any cell near it changed.
### Fixed (2026-11-05, five navigation and planning complaints read off the ten replayed non-stair regressions)
- **The lattice follower plans with plain A\*** (`planners/lattice/params.py` `heuristic_weight` 1.0, was 1.5;
  `navigation/settings.py` `KinematicSettings.heuristic_weight`): the weighted search returned the first goal it
  popped, and because a forward step lowers the route's cost-to-go while a turn does not, it popped the plan that
  steps first and turns later -- the agent seen walking two steps off a route 16-35 degrees from its heading before
  turning toward it (a route 35 degrees off walked `F L L F F` where `L F F F` was open). On the 741 recorded plans
  of the ten episodes: **19.3 actions and 6.4 turns per plan against 20.5 and 7.1**, cheaper on 523 of them, 3.4 ms a
  plan (40 ms at worst) against 1.4 ms. On a straight route 16 degrees off the heading both orders cost the same
  actions; the planner then centres its zigzag on the line, which is offset-optimal -- `route_offset_cost` is the
  knob that trades turns for tracking, not the first heading.
- **A committed transit entry the floor standoff refuses is asked of the body radius before the room is re-aimed
  or released** (`rooms/room_search_loop.py` `_navigate_transit`, `transit_squeezed`, the `transit_squeeze`
  event; `policy._navigate(retire=)`): one jamb cell painted by the doorway look-down drew a 0.5 m way as 0.4 m,
  the route to R9 1.7 m away was refused, every re-aim behind the same gap too, and 00843/000023 released the room
  -- and on the same map the bathroom behind it -- as `unreachable`, re-solved the order to the other end of the
  house, walked 13 actions west, was sent back, and looped 100 actions round the building before it walked through
  that gap at the floor standoff (actions 479-483). Four of the seven `unreachable` releases in the ten episodes had
  a route at the body radius. A squeeze route in force is judged by the planner that made it.
- **Doorway scrapes are counted by place, and a place that ran out of excuses stays closed** (`navigation/
  scrape_guard.py`; `ScrapeSettings.same_place_m` 0.5, `closed_memory_actions` 120): the count was per channel,
  keyed by the throat rounded to half a metre, and since every refusal clears the route and the re-plan names a
  throat a few centimetres over, 00843/000007 was refused at one spot at four headings 90 degrees apart (a hole in
  the mesh the depth camera sees nothing of, drawn free by the rays through it), each excused as "scrape 1 of 3"
  of a new channel, came back 80 actions later and was excused three times more: 74 actions and three room
  selections on the room behind it, 0 of 14 refusals a sticky bar.
- **The standoff ring tries the front arc with every planner before any point beside or behind the target**
  (`closing/target_path.py` `BACK_ARC_FROM_DEG`, `back_arc_standoffs`): the toilet seen through the bathroom doorway
  from 3 m, the doorway's floor still unknown, and the only known-floor route led to the point behind the toilet --
  in the next room, 7.5 m round the outside of the bathroom -- while the front arc through the unknown was never
  asked (00891/000006 and /000027). **A STOP on the memory alone needs the last fresh box within
  `history_stop_max_unseen` = 15 actions** (`TargetClosingSettings`, `history_stops_unseen`): both episodes walked
  36 and 41 actions with the toilet out of view, inspected the wall from 0.9 m and STOPped on a memory 39 and 45
  actions old, 6 m of geodesic from the goal.
### Fixed (2026-11-01)
- **The LLM answer store keys answers by episode** (`llm_answer_cache.request_key(scope=)`, `LLMAnswerStore.scope`,
  `LLMClient.set_answer_scope`, set by `RPTSearchPolicy.reset`): the first glance A/B pilot recorded arm A from three
  scene workers into one directory; two asked the same room-label question, got two answers, and every scene that
  started after that refused at preflight (`shards disagree`). 50 of 300 episodes ran; the pilot restarted clean.
- **A resumed scene refused its own configuration** (`LLMAnswerStore.describe` recorded `loaded`, the answers already
  on disk, which grows as the run records): the pilot's RAM pause (00:14 UTC, 61 % for 30 s with two Mesa workers on
  large scenes) interrupted two scenes, and neither could resume. The frozen description no longer carries it; the
  two scenes rerun afresh, their finished episodes replaying from arm A's record.
### Added (2026-11-01)
- **`objnav_benchmark_runtime/paired_ab.py`**: two campaigns on the same episodes compared pair by pair -- SR and
  SPL differences with paired-bootstrap intervals, exact McNemar, flips both ways and the flip rate, the
  never-acted integrity check, the glance decision rule, and the pairs a decision needs at the measured flip rate.
### Added (2026-10-31, evening: reproducible LLM answers for a paired A/B)
- **An LLM answer store that records and replays answers keyed on the exact request**
  (`core/mapping/topology/llm_answer_cache.py`, `LLMClient(answers=)`, `LLM_ANSWERS_RECORD` / `LLM_ANSWERS_REPLAY`,
  `llm_answers` per episode row, the frozen config's `llm_answers`): both LLM routes, prompts kept in the shards, the
  preflight probe never served from it. Arm B of an A/B replays arm A's answers.
- **A cold-start mode for Ollama** (`core/mapping/topology/llm_exclusive.py` `ExclusiveColdStart`, `LLM_EXCLUSIVE=1`):
  a host-wide lock and a model unload before every request, the one mode measured to make an answer depend on its
  request alone (a flush and a per-request tag did not; `LESSONS.md` 2026-10-31); 6.4 s a call against 3.5 s.
### Fixed (2026-10-31, evening)
- **The kinematic follower judged plans against an older map** when a new planning world was allocated at the freed
  one's address (`navigation/kinematic_follower.py` `_collision_mask`, keyed on `id(world)`, now on a digest of the
  occupied cells): two otherwise identical runs of 00813/000000 parted on it at action 83.
### Fixed (2026-10-31, the GT sweep's dead oracle and its single-storey failures, measured on 40 of them)
- **The node oracle answers again with one model on both routes**, the run stops on a dead oracle, steps and spawns
  on treads are walkable, standoffs on a step are standoffs, released targets are pursued, memory STOPs need line of
  sight: 40 failed or poor-SPL single-storey episodes of `gt_oracle_2000ep_3b_20261028T152707Z` re-run with
  geometric glances off: **SR 0.125 -> 0.600, SPL 0.014 -> 0.321**, 19 fixed, no success lost
  (`docs/progress/entries/040-hm3d-failure40-fixes.md`, `runs/failure40/report_final/`).
- **Geometric path glances judged on capped rays, and off by default pending an A/B**
  (`core/planning/exploration/view_gain.py` `UnknownView.unknown_through`, `glance_gains(unknown_run_m=)`;
  `exploration/path_glances.py` `GlanceSettings` `unknown_run_m` 1.0 m, `geometric_min_gain_m2` 1.0,
  `geometric_min_gain_per_action_m2` 0.15; `RPTSettings.geometric_glances`, False, the run's switch through
  `--policy-config`): a ray counts only the first metre of unknown it enters, so an unmapped wall stops promising the
  room behind it. On the 232 recorded glances the promise went from r = -0.17 to 0.39 (0.62 geometric) with what the
  map gained; cue glances unchanged. The same 40 cases with them on: **SR 0.650, SPL 0.283** (7 fixed, 5 lost
  against off; `runs/failure40/report_glances/`) -- but the oracle answers the same prompt differently run to run, and
  the 6 episodes whose runs parted on such an answer before any geometric glance carry the whole SPL drop (-0.067);
  the 27 a glance set on a new course went 7 fixed / 2 lost, +0.032 SPL.
### Changed (2026-10-28/29, the narrow-passage navigation overhaul, measured)
- **Every committed route is walked on the discrete-action lattice** (`core/planning/planners/lattice`,
  `core/planning/object_search/navigation`): Hybrid A* over (x, y, heading) with the benchmark's step and turn as its
  only primitives, the 2D route's medial-axis line (the planner's new `centreline_weight` layer) as its corridor, the
  converter executing the plan verbatim under strict tracking, a refused step a pose rather than a cell on the map, a
  doorway scrape excused and a look-down on the approach. HM3D v1 val under ground-truth perception, 2,000 episodes
  (`runs/gt_oracle_2000ep_v2_20261028T134440Z`, `docs/progress/entries/039-hm3d-gt-oracle-v2-narrow-passage.md`):
  **SR 65.3 % (was 58.4 %), SPL 0.353 (0.328), DTG 3.43 m (3.97 m)**, refused steps 6.1 per episode (11.9), agent
  errors 0 (34); 17 of 20 scenes up, 00832, 00839 and 00876 within noise of before.
### Fixed (2026-10-27, the kinematic self-check and habitat-sim's step filter)
- **A body action is judged from where habitat-sim's step filter started it** (`objnav_benchmark/kinematics.py`
  `check_motion(placed=...)`, `HabitatRGBDSimulator.last_placed_pose`, `HM3DEnv.placed_pose`, the runner): the second
  full HM3D v1 sweep (`runs/hm3d_v1_val_j8p2_20261026`, 1,926 of 2,000) lost `00878-XB4GS9ShBRE` at its 96th episode to
  a MOVE_FORWARD carried 0.197 m across its heading and `00832-qyAac8rV8Zk` at its 30th to one that dropped 0.679 m --
  the filter's own placements, read in habitat-sim 0.2.4's source and reproduced with the harness's simulator:
  `tryStep` starts every body action from the 3-D-nearest polygon point, can leave the agent off the mesh (a stride on
  00878's stairs advanced 0.063 m, dropped 0.474 m and stood 0.316 m from the nearest mesh point), moves nothing when
  it refuses a step from there, and carries the agent onto the mesh with the next turn or step. The simulator now
  reports that projection (`pathfinder.snap_point`, which *is* `tryStep`'s start) and the check judges a turn or step
  from it toward where the action aimed -- the end anywhere on that segment, or the pose itself, the height judged from
  the mesh; a slide off the segment, a step past it or a backward stride is still refused, which no scalar
  `reproject_m` over 0.20 m could promise.
  `HM3DProtocol.kinematics()` `climb_m` 0.60 -> 0.85 m, derived from the recomputed navmesh (a riser under two cell
  heights, a 45-degree run over the stride, the detail mesh's sample error). 148,500 seeded random actions per scene
  with the harness's simulator reproduced both placements (turns displaced up to 0.313 m, strides climbed up to
  0.740 m), which pass under the new check (LESSONS.md 2026-10-27).
### Added (2026-10-26, the perception upper bound)
- **Ground-truth (oracle) perception for the HM3D ObjectNav harness** (`objnav_benchmark_runtime/ground_truth/`,
  `hm3d.run --perception ground-truth`, `hm3d/run_benchmark.sh --perception ground-truth --jobs max` ->
  `runs/gt_oracle_2000ep_<stamp>`): habitat-sim's semantic sensor (per-pixel HM3D-Semantics instance ids, rendered by
  `HabitatRGBDSimulator(semantic=True)` through the publisher's annotated scene dataset config and published by
  `HM3DEnv` to a `GroundTruthSource`) answers the method's `Detector` and `Segmenter` contracts in place of YOLO-World
  and SAM -- every visible annotated instance in the label table's vocabulary as a confidence-1.0 detection with its
  tight box, the instance's exact pixels as the target's mask; 100 % precision/recall against the annotations, no box
  drift, no label noise, one declared floor (`--ground-truth-min-pixels`, 64). Goal categories take the publisher's own
  goal instance ids per scene (`HM3DDataset.goal_instances`; an instance named for a listed category but not among its
  goals is dropped and counted); the context vocabulary comes from an exact raw-name table (`hm3d_categories.py`,
  SHA-256 in the frozen configuration). Nothing of the method changes -- dedup, gates, doors, projection, landmarks, LLM
  room labels, RPT*, frontier/FALCON, side-looks, A*, closing -- and the run is `publishable: false`. The source
  answers only for the frame it was given; another frame ends the run. Verified end to end on the public HM3D example
  scene (908 annotated instances; sensor ids reproduce `semantic.txt`; one full RPT episode: 320 frames, 145 labelled
  instances, 336 door detections -> 8 door landmarks, 10 rooms, closing to STOP). The licensed val annotations
  (`hm3d_val_semantic_{annots,configs}_v0.1`, ~1.4 GB) are a hard prerequisite: `hm3d.assets --check --semantic` and
  `--instructions` step 5 name them, the preflight refuses the run by file name without them.
- **`--jobs max`** (`hm3d/scene_pool.max_workers`): one scene child per scene at most, bounded by cores and by the free
  VRAM at `HABITAT_VRAM_MIB` per renderer.
### Fixed (2026-10-26)
- **A resident Ollama is no longer re-placed by the GPU plan** (`benchmark_services.sh`): with the card held by an
  already-running LLM the plan read "no room for the LLM" and exported `LLM_NUM_GPU=0`, so every child of the new run
  would have asked the *shared* server for `num_gpu=0` and reloaded both models on the CPU for the live campaign too.
  An LLM answering at `LLM_BASE_URL` before the plan is budgeted as placed and its device left alone.

### Added (2026-10-22)
- **Five-scene campaign on the committed-target changes** (`runs/benchmark_25episodes_20261022T081925Z`, 25 of 25 scored in
  16 minutes, everything recorded with the map journal; `docs/progress/entries/036-objnav-committed-target-and-log-review.md`):
  SR 0.880, SPL 0.631, SoftSPL 0.652, DTG 0.61 m, 71 actions and 37.5 s per episode. On the nineteen episodes the
  04:43 run had scored before it was stopped: SR 0.895 -> 0.895, SPL 0.631 -> 0.693, DTG 0.76 -> 0.58 m, 98.7 -> 60.6
  actions, 130 -> 32.8 s per episode; Corozal/000003 toilet SR 0 -> 1 (500 -> 202 actions, three LLM calls where
  there were nineteen). Two wrong STOPs on detector false positives (Markleeville/000000 `bed`, Wiconisco/000003
  `sofa` with a 0.34 m footprint), diagnosed from the logs; the closing's, not the search's.
### Changed (2026-10-22, the Corozal/000003 toilet recording `c1af8954a378`: a target set is a target reached)
- **The committed-target rule** (`rooms/room_search_loop.py` `LoopSettings.transit_commitment` on, `rebind_reach_m`
  0.6; `object_search_supervisor.py` `rebind`). Eleven of the episode's eighteen selections were `unreachable`
  releases of a room the watershed had merely renumbered on the way to it (`room_missing_held` -> `release` three
  actions later), each buying a room-LLM call (10 s) and an RPT* solve, and at action 137 the agent re-planned to the
  toilet's room while committed to its neighbour. A transit now ends only by arrival, a route the planner refuses or
  the follower makes no progress on, the follower's wedge, the supervisor's clocks, or the target takeover: the room
  in force is RE-BOUND to the live room holding the committed entry point (else the lost room's centre) under its new
  number, with its entry, estimate and probability (`room_rebound` events, `rooms_rebound`); a visit in force follows
  the room renumbered under the agent's feet; no classifier call is spent in transit (`relabels_deferred`; the arrival
  asks) and the ledger's `finished_in_transit` no longer ends a walk -- both are judged at arrival. The place walked
  to with no room under it at all is given up after `room_missing_grace` (10) actions as before. Off, the 2026-10-21
  behaviour; the regressions that encode it say so.
- **The scene graph is not rebuilt on every action of a committed transit** (`policy.py` `graph_period`,
  `refresh_graph`; `RPTSettings.transit_graph_period_steps` 5). The watershed, registry, frontier inventory and
  per-room facts were 1.4 s of a 1.7 s median decision, 500 times; the follower needs the map and its route, not a
  fresh partition. Rebuilt every fifth action while the route is walked, on the action the transit ends (before the
  visit's first decision), and at every loop point as before; every action in SEARCH and SELECT as before. The
  body-radius cost grid is built once per map and shared (`navigation_cost` cache; the loop asked for it twice).
- **A place the agent keeps failing to reach is given up** (`LoopSettings.entry_failure_limit` 2,
  `entry_failure_radius_m` 0.75, `entry_failure_backoff` 2; `entry_failure` events, `unreachable:<n>` exclusions).
  `unreachable`, `blocked` and `transit_timeout` releases are remembered by the entry's POSITION, so the memory
  survives the renumbering the supervisor's `max_attempts` does not: a room whose failed places add up to the limit,
  or whose candidate entry lies at one, is withheld for `failure_cooldown_steps * 2 ** (n - 2)` actions, and the
  explore entry passes such places over for the room's next frontier or doorstep. 0 disables.
- **A frontier wrapping a block of unknown is one cluster per side** (`core/mapping/topology/frontier_split.py`;
  `room_costs.frontier_clusters`, `room_stats.count_frontier_clusters` `split_sides`). The floor mapped along three
  sides of an unmapped room was one 8-connected component with one goal, on whichever side its centroid fell nearest
  -- action 291: the gap O30 on one side of an unknown square and nothing on the side the agent could reach. A
  component whose cells face different sides of their unknown and whose centre lies on the unknown side of them (it
  wraps a block; an arc bulging into the open unknown has its centre behind it and stays whole) is cut at the corners
  into one cluster per side of at least a doorway's width (`SIDE_MIN_M` 0.9); the goal maker and the scene graph's
  count make the same cut. The per-cluster extraction reads each component's bounding box, not the whole grid.
- **Trodden ground ranks under ground barely visited** (`frontier_ranking.py` `FrontierRankingParams.visit_weight`
  0.05 / `visit_radius_m` 1.5, `trail_visits`; the loop's `_visits`, the fallback's `_by_view_gain`). A frontier
  goal's utility is divided by `1 + 0.05 * actions spent within 1.5 m of it`, read off the sight ledger's trail: the
  route back through the open room crossed five times (action 291) now ranks behind the frontiers in the wings the
  agent never walked. 0 disables.
- **The unseen floor behind a room's exits is on its oracle line** (`opening_nodes.RoomExits.unseen_m2`,
  `search_node_oracle.SearchNode.unseen_m2`, `format_exits`): `exits=2 (1 doorway, ~30m2 unseen behind them)`, the
  rays' reach from the exits' thresholds summed, and the prompt told that a room whose exits open onto a large
  unseen area is where the rooms not found yet must be -- the model can prefer the way to another room over the
  living room it has seen.
- **The floor panel draws the accessible frontier goals** (`floor_panels.py` `FRONTIER_COLOR`; `search_panel.py`):
  the column counted "blue diamonds" the map did not show, so a recording could not say which way the inventory
  pointed. One small diamond per cluster.
- **A recording is reviewed from its log, by step, never from a frame** (`floor_panels.py` map journal ->
  `map_journal.npz`; `visualization.py` `search_snapshot` `frontier_goals` / `transit` / `entry_failures`, the
  explore visit's `ranked` goals; `review_recording.py`; `recording.py` and `episode.json` say so). The per-step
  row had the frontier goals as bare coordinates and the only saved map was the final one, so "what lay to the
  agent's left at action 261 and why was none of it a goal" was answerable only from a video frame. Now every
  action's occupancy, settled-unknown and room-partition deltas are journaled, each frontier goal rides with its
  size, walk, facing, utility and room, and `review_recording.py <dir> summary | timeline | step N | map N |
  events` answers from the log -- `step N` with the map of that action by sector round the agent (free /
  occupied / unknown / settled area, raw frontier cells, the goals and openings in the sector). Older
  recordings fall back to the final map and say so. `LESSONS.md` has the entry.
### Fixed (2026-10-22, after 19 episodes of the 25-episode campaign: stair false positives, classifier churn, update cost)
- **An observed staircase must have a flight's profile** (`world_model/stair_terrain.py` `stair_profile`;
  `MultiFloorParams.stair_min_risers` 3, `stair_riser_min_m` 0.10, `stair_max_tread_m` 0.55): any climbable chain of
  support surfaces 0.4 m up was a candidate -- Corozal/000003 placed a "staircase" on a coffee table's shelf and top
  and the sofa behind them, Darden/000000 one on a bed, each masking a square metre or two of the room on the planning
  grid and excluding it from the partition (Corozal's toilet episode ran to the 500-step limit, DTG 8.5 m). Now the
  path must climb one way in at least three risers of 0.10 m to `max_step_m`, with no level between them walked for
  more than 0.55 m: a table top, a seat or a mattress is a tread half a metre deep or more, and no furniture is three
  tread-sized risers tall; a landing ends one stretch and the next flight starts at it. `risers` / `profile_rise_m`
  on the candidate. One new test in `test_multifloor.py`.
- **The room classifier is keyed by a room's objects alone** (`room_classifier.py` `_signature`,
  `revisable_room_labels.py`): with the geometry bucket in every cache key, a room re-asked the model whenever it grew
  from small to medium or gained a doorway -- 80-100 classifier calls per long episode. The bucket now keys only a
  room with no objects (the shape-only question); the geometry is still shown in the prompt when a query happens.
- **`_parts_skeleton` judges a choke on the skeleton's bounding box, not the whole grid** (`room_watershed.py`): the
  800x800 grid was labelled twice per candidate, a hundred times per scene-graph update -- 0.4 s of the update's
  0.76 s on Corozal/000003's map, the update being 1.4 s median over the 500 steps. Exact (9,600 random verdicts on
  three recorded maps unchanged, the Corozal partition identical); the update halves to 0.37 s on that map.
### Changed (2026-10-21, afternoon: observed stairs, the unvisited raised, wide glances, the room taxonomy)
- **Stairs are observed, never read from the navmesh** (`core/localization/floor_atlas.py`
  `MultiFloorParams.stair_source` default `observed`; `core/planning/object_search/multifloor/multifloor_policy.py`,
  `policy.py`, `spawn_floor_guard.py` docstrings). The ground-truth connector's footprint was the navmesh centreline,
  which in Darden begins 0.45 m before the first riser on a sloped threshold -- drawn on the grid the flight stood
  lower than the stairs do, and with the capsule's foot it closed the junction below. The method now takes its stairs
  from its own RGB-D: a reachable run of measured treads rising `stair_min_rise_m` (0.40 m) within 5 m
  (`world_model/stair_terrain.py` candidates) is a portal of the storey, placed from the measurement, masked by the
  same capsule (`peek_stairs`), with the observed drops and the map's obstacle band as before -- no simulator geometry
  is consulted, the footing of ApexNav, SG-Nav and OSG-Nav. `configuration()["ground_truth_stairs"]` reads False;
  `stair_source="ground_truth"` is the declared ablation and the fixtures that hand an episode a scene structure ask
  for it (`tests/test_method.py setup_policy`, the baseline's `baseline()`).
- **A room never stood in is raised; a room entered decays with the time spent inside** (`rooms/room_search_loop.py`
  `_scaled_values`, `LoopSettings.unvisited_boost` 1.5 / `unvisited_floor` 0.15 / `dwell_half_life_steps` 20 /
  `dwell_floor` 0.05; the oracle prompt's STEP 4, `search_node_oracle.py`). The models read a never-entered room's
  `frontier=0` or `1` as little to find, when the count measures the map and not the room -- one boundary cleared
  opens the whole of it. A never-entered, unfinished room the model gave anything is read at 1.5x its number and no
  less than 0.15 (a room given 0 by the model or the type prior stays 0; the room the agent stands in keeps its
  number); a room entered and left keeps `entered_discount * 0.5 ** (steps inside / 20)`, floored at 0.05, the
  lingered rule still capping it. The prompt: never below 15 for a never-entered unknown room, never lowered for a
  low frontier count; the longer `searched=` reads, the lower. `test_search_economy.py`: the lingered test follows
  the decay; one new test.
- **Side glances turn to 120 degrees when the far band pays** (`exploration/path_glances.py`
  `GlanceSettings.wide_rad` 2pi/3, `GlancePlan.turn_rad`, `wide` plans and counters). A right angle with the 79-degree
  camera shows the floor to about 130 degrees off the route; a pocket beside and a little behind the agent -- the
  alcove just passed, the room whose door opens back toward it -- lies in the 130-160 band only a wider look
  reaches. Every candidate point is scored for the wide look too, taken only when the extra band alone pays for the
  two extra turns each way at `min_gain_per_action_m2`; the sector memory, the committed plan and the HUD carry
  the angle. Two new tests in `test_path_glances.py`.
- **The room taxonomy is residential, and a label is never locked** (`core/mapping/topology/room_classifier.py`
  `HOME_LABEL_SET`, `LABEL_ALIASES`, geometry facts in the prompt and the cache key; `revisable_room_labels.py`
  `shape_only_rooms`; `rooms/scene_graph.py` `_room_geometry`; `core/mapping/objects/landmarks.py` `max_misses`;
  `perception/cycle.py` `_count_misses`; `RPTSettings.landmark_max_misses` 4 and the miss view bounds). Hallway /
  corridor, foyer / entryway, dining room, office / study, laundry / utility, balcony / patio, closet / pantry /
  storage and garage are the set offered first, synonyms and slashed forms are read as the set's labels, and the
  prompt names the detector's confusions (a table or a sofa for a bed). A room is told its floor area, its shape and
  the doorways off it, and a room with nothing in it that reads as a passage -- long and narrow, or two or more
  doorways, 2 m2 or more -- is asked about on its shape alone, once per geometry bucket, WEAK and provisional (it
  rules nothing out). The lock: confirmation was one-way, so a dining table YOLO called `bed` at 0.52 twice named a
  kitchen a bedroom for the episode (Darden/000000). A confirmed landmark that stands in a CLEAR view -- inside the
  frame by 40 px, 0.8-3.5 m away, the depth at its pixel reaching it -- without a detection of its class is missed,
  once per distinct viewpoint (0.5 m or 30 degrees from the last counted miss); as many misses as
  `max(landmark_max_misses, its observations)` -- four for the twice-seen table, twelve for a sofa seen twelve times
  (a flat four retracted 29 of Markleeville's 51 confirmed objects, most of them real) -- retract it from the
  confirmed set (the room's evidence, the target landmark nodes), the evidence gate
  takes the room's label back at the next scene-graph update and the label history records it; the next detection
  that folds into it reinstates it. `perception.counts` `landmark_misses` / `landmark_views_occluded` /
  `landmark_retractions`, `perception.retractions` in the record. Tests: `test_room_classifier.py` (3),
  `test_rooms.py` (1), `test_perception_association.py` (3).
### Fixed (2026-10-21, the audit after the first room-level run: doorways, glances, closing, stairs, spins)
- **The doorway partition: the space beyond an observed open door is a node of its own** (`core/mapping/topology/
  room_watershed.py`, `core/planning/object_search/rooms/scene_graph.py`, `room_search_loop.py`). The sliver of floor
  glimpsed through a doorway had no clearance peak, so no watershed label, and `_reclaim_carved` handed it to the
  nearest seeded room ACROSS the very cut carved to part them (the toilet seen through the door named the hallway a
  bathroom). Now: `ObservedSceneGraph.note_openings` (called by the loop with every action's openings and the viewing
  pose) reads the doorway off the line of sight from the viewpoint to the opening -- `sightline_thresholds`, the
  floor narrowing between OBSERVED jambs with the unknown passable, so an open room's frontier is no doorway -- snaps
  the dip to the doorway's choke (`SIGHTLINE_SNAP_REACH_M` 0.8) and keeps it as a sticky threshold
  (`threshold_events` `source: sightline`); the carve there is a two-cell LINE across the passage (`_severing_cut`,
  the disk of old swallowed a thin sliver or left two crescents of it), chokes are ranked by clearance to observed
  obstacles (`ChokeFields`), a short far side counts when it leads into the unknown (`_parts_skeleton`, `leads_on`,
  `LEADS_ON_CELLS`), a choke must be walled on both sides across the passage (`_walled_both_sides`) and must have a
  cut that parts the floor; and the seedless floor beyond a cut that borders unknown is seeded as a **stub room**
  (`_stub_rooms`, `WatershedRoomParams.stub_min_cells`, 4 in `DEFAULT_SEGMENTATION`, 0 = the older behaviour), exempt
  from `min_room_cells`, keeping its number as it grows. `segment_rooms_watershed(unknown_mask=...)`,
  `door_carve_mask(unknown_mask=...)`. The scene graph's edges now include every threshold with the room pair it
  joins through the doorway coordinate (`graph.doors`, kind `threshold`, index `t<k>`; doors carry kind `door`), and
  a door's annulus starts at the door cell so the stub beyond it is linked. Tests: `tests/test_doorway_partition.py`
  (2), four in `test_room_watershed.py`.
- **Side glances were scheduled and never taken** (`core/planning/object_search/exploration/path_glances.py`). The
  audit of the fifteen recordings of 2026-10-21 found the scheduler wired and firing (40 glances, 33 of them side or
  cue), but the geometric side looks re-planned away before the agent reached them -- Collierville/000000: 27
  scheduled, 4 taken; Markleeville/000000: 27 and none -- the re-scoring every two actions moving the plan to the next
  doorway as the map grew. The plan in force is now **committed**: re-scored on the fresh map, kept while its gain
  clears the floors, replaced only by a candidate at or nearer than its point, taken when its point is reached or
  passed. Three new tests.
- **Redundant circles** (`path_glances.py`): a full circle must pay for its four extra turns with the circle's extra over
  the better side glance (`_full_worth_it`), and is never taken inside a room the search has finished, that has no
  frontier left, or whose frontier walk is in force (`_full_allowed_at`): the recordings' circles promised 5.5 and
  18.5 m2 and revealed 0.8 and 1.9. The loop already re-selects on the action a room's visit ends; no other rotation
  source exists (the sweep's look-around allowance is 0, the fallback never spins).
- **The takeover starts on an RGB detection without depth** (`core/planning/object_search/closing/target_closing.py`):
  a confident whole box of the target's class with no usable depth -- beyond the sensor's ~5 m -- begins a bounded
  BEARING approach once the class has been in the frame on two consecutive actions (`bearing_start`): face the box by
  its image-space bearing, step toward it, re-face it each frame it is seen, until a frame projects with depth and the
  ordinary VERIFY starts there with its own clock. Given up without a rejection memory after `bearing_max_actions`
  (16), `bearing_lost_frames` (3) frames without the class, three steps that moved the agent nowhere,
  `bearing_max_walk_m` (3.0 m) walked toward the box -- a box first seen within the depth cap plus that distance is in
  range by then, and one still depthless is unmeasurable (a television screen is a hole in the Gibson mesh,
  Corozal/000001: the unbounded walk went 3.75 m, never projected, and left the agent to STOP on a second television
  6.95 m from the goal; with the bound the episode succeeds) or not where the bearing pointed -- or a box
  `bearing_near_box_frac` (0.30) of the frame's height tall without depth, which is within reach and cannot be
  positioned, so no blind STOP is taken on it; no retry within `bearing_retry_m` (1.5 m) of the spot the walk began.
  `phase=target_bearing`; counters `bearing_starts` / `bearing_resolved` / `bearing_abandoned` / `bearing_refusals`.
  `test_invalid_depth_never_starts_a_takeover` became `test_invalid_depth_never_locks_but_begins_a_bearing_approach`;
  four more tests.
- **The stair capsule's half-width is a setting, slightly reduced** (`settings.py` `RPTSettings.stair_margin_m` 0.45 m,
  five cells drawn; `world_model/peek_stairs.py`): before, body radius + arrival tolerance + a cell, 0.6 m drawn, which
  with the planner's body-radius inflation on top narrowed the passages beside Markleeville's stairwell on the 2D grid.
  Stairs only; the adaptive capsule still narrows it to the body radius where a corridor would close. Not scene-specific
  by design: the method names no scene. Tests updated in `test_spawn_floor_guard.py`.
- **The stair capsule's storey end is square** (`world_model/peek_stairs.py` `_foot_of` / `_square_foot`, Darden,
  same day): OpenCV's thick line ends in half-discs, and the one at the storey end lay on the floor in front of the
  first riser -- the margin's whole radius of plain floor, 0.55 m with five cells drawn. Darden's flight has its foot
  in the junction of the hallway and the room beside it, and that half-disc, with the planner's inflation on top,
  drew the junction as one oval (`Darden/000000`, the campaign of 2026-10-21: 204 cells, 50 of them hallway floor
  between y 1.03 and 1.53 across the junction's whole width). The end nearest the storey plane is now cut flat across
  the flight's direction, inside every candidate radius of `adaptive_capsule` as well; the sides keep their margin and
  the far end its cap, more flight being behind it. The connector's anchor itself -- the storey floor in front of the
  flight -- is no longer masked or refused as a goal; the first treads still are. Not scene-specific. One new test in
  `test_spawn_floor_guard.py`; two probes moved from the anchor to the first treads. *What the square foot does not
  open:* the doorways between Darden's west and east halves are 0.2 m humps in the mesh (the navmesh walks them; the
  observed map's obstacle band begins 0.15 m above the plane), see progress entry 035.
### Changed (2026-10-21, room-level GAPs: the decision hierarchy of the room search)
- **Exits are a room's attributes, never nodes; the room's visit walks them** (`core/mapping/topology/
  search_node_oracle.py`, `core/planning/object_search/rooms/{room_search_loop,opening_nodes,scene_graph}.py`,
  `policy.py`, `objnav_benchmark_runtime/search_panel.py`). A standalone LLM number per doorway or gap let the RPT*
  order chase open passes for their own sake -- every gap of a living room, in a search for a toilet -- instead of
  the room-level prior. Under `explore` every exit read off the live map is now folded into the room it opens from
  (`_fold_openings`); the room's node line carries its exits as attributes (`SearchNode.exits` / `doorways` /
  `glimpsed`, rendered `exits=2 (1 doorway)  glimpsed through its exits: toilet`, `opening_nodes.room_exits`;
  `ObservedSceneGraph.reason(exits=...)`); the system prompt's new EXITS rule ties them to the room's type (exits
  raise a room only where its type fits the target's home; a hallway's or an unidentified room's by what the home
  still lacks behind them; a living room's gaps never for a toilet), its worked example values rooms by their
  exits, and the OPENING node rules moved to `OPENINGS_SUPPLEMENT`, appended only when an exit node is offered.
  The global decision is over rooms and stairs; the look through a doorway is a step of the chosen room's local
  exploration (the explore visit's frontier walk), which ends when the room's GAPs are exhausted or
  `explore_visit_steps` runs out and whose release is the loop point that re-selects. A ledger-finished room with
  GAPs left to walk (`_gaps_left`: significant live frontier on a room with an exit) is **reopened** for them
  (`_exclusions` -> `_reopened`, event `room_reopened`, stat `rooms_reopened`, `reopened_for_gaps` in the
  diagnostics, `exits` / `reopened` in the estimate events): shown as `status=scanned(...)` with its `exits=`,
  entered at its best frontier, priced by its frontiers, discounted as an entered room, walked and not re-scanned;
  a mid-visit rotation finishes the room in the ledger but no longer ends the visit while GAPs are left, and a
  room in transit with GAPs left is not released `finished_in_transit`. A new exit on an offered room buys the
  oracle a call at most every `revalue_actions` (`_exits_at_call`), as a new opening node did; the HOUSE line
  counts every exit of the storey. `OpeningSettings.exits_as_nodes` (default False) keeps the 2026-10-04 design
  as a deprecated mode -- the opening-node regressions in `test_opening_nodes.py` (scan mode, unchanged) and
  `test_explore_visit.py` run it explicitly; `scan` and `sweep` visits keep exits as peek nodes regardless, and
  target landmarks are nodes in every mode. Visited-node decay (`entered_discount`, `lingered_discount`, the
  supervisor's cooldowns) and the footprint tracking are untouched. Nine regressions in
  `tests/test_room_level_gaps.py`; the room line's `exits=` field is pinned in `test_search_node_oracle.py`.
### Added (2026-10-19, HM3D under the Gibson-validated method)
- **`feat/objnav-benchmark-infra-nadav` merged into `feat/objnav-habitat-hm3d-nadav`**: the HM3D adapter now runs
  the method from `core/planning/object_search/` exactly as the Gibson val result did. `core/` differs from the
  infra branch by the HM3D label table and its registry line only; the HM3D branch's two algorithm-side edits were
  dropped on purpose (the `goal_changed` watchdog exemption in `route_memory.py`, a behaviour change the Gibson
  number was not measured with, and the `target_closing_enabled` switch, a no-op knob). A frozen HM3D
  configuration and a frozen Gibson one differ in the detector's deployment (URL, device, class order) and in
  nothing of the 548 method keys.
- `hm3d/run_benchmark.sh`: the one-command HM3D run -- services, a mesh-free preflight, then `hm3d/campaign.py`
  (one `hm3d.run` process per scene, resumable, aggregated into `campaign_summary.json`) or one `--scene`;
  `--smoke`, `--record`, `--status`, `--preflight`, `--version v1|v2`, `--agent falcon-yolo` as on Gibson.
- `objnav_benchmark_runtime/benchmark_services.sh`: the service orchestration both launchers share (GPU plan,
  Ollama, the YOLO-World detector started with the benchmark's own vocabulary, the segmenter, FFmpeg, teardown),
  lifted verbatim out of `gibson/run_benchmark.sh`, which now sources it; a detector already up at the port is
  checked to serve this benchmark's class order before it is reused (HM3D defaults to port 18092, Gibson to 18095).
### Changed
- `habitat/simulator.py` is the HM3D branch's superset (body height separate from the camera mount,
  `navmesh="published"|"agent"`, `pathfinder`, `navmesh_provenance()`), with the reset-time height
  reconciliation applied under `navmesh="agent"` only: under `"published"` the start is placed verbatim, as every
  Gibson run did. `HabitatRGBDSimulator` now requires `height_m=`; `gibson/env.py`, `smooth_replay.py` and
  `stair_diagnostic.py` pass `GibsonProtocol.agent_height_m` (0.88 m, the camera height -- no change in what runs).
- `provenance.episodes_of_category(ids, episodes, category, categories)` is the one category filter; Gibson's
  `run.episodes_of_category` delegates to it.
- The shared runtime keeps the HM3D branch's benchmark-neutral `dashboard.py` (titles from the summary's
  benchmark/split, unreachable DTGs omitted from the plot), `recording.py` (optional telemetry hook, encoder log
  closed on a failed start) and `run_evaluation`'s `progress` / `wrap_env` / `agent_class` seams.
- **The object-search method moved out of the benchmark runtime into `core`** (`core/planning/object_search/` new,
  `tasks/planning/objnav_benchmark_runtime/methods/` gone, 2026-10-19). A structural change only -- every module
  moved with `git mv`, no algorithm, setting or default changed; a run after this commit differs from one before
  it in `source_sha256` alone. The method (`policy.py`, `settings.py`) is now one package split by concern --
  `perception/`, `rooms/`, `world_model/`, `exploration/`, `multifloor/`, `closing/`, `falcon/`, `baselines/`,
  `route_memory.py` -- and its policy-free pieces went to the subsystem they belong to: `box_extent`, `box_occluder`,
  `detection_dedup`, `target_evidence`, `support_surface` in `core/mapping/objects`; `target_memory_3d` in
  `core/mapping/tracking`; `room_priors`, `revisable_room_labels` in `core/mapping/topology`; the camera-tilt
  controller and the box-driven pitch servo in `core/control/camera` (`pitch_controller`, `box_pitch_servo`: where to
  look is a control process beside where to move, merged in the `NavigationCommand`); the storey localiser
  `floor_atlas` in `core/localization` (from `core/planning/exploration`). Five inline ``atan2(sin, cos)`` wraps now
  call the shared `normalize_angle`; the box aliases use `core.common.math.bbox.BBox`. The seam to the
  services is `object_search/interfaces.py` (`Detection`, the `Detector` / `Segmenter` / `RoomReasoner` protocols,
  the two failure families, which the perception cycle now tells apart without importing `requests`); the HTTP
  clients moved to `objnav_benchmark_runtime/services/` (`http_detector`, `http_segmenter`, `llm`). The runtime
  shares one method factory across datasets (`policy_factory.build_policy`, as on the HM3D branch), lifts
  `detector_options.py` and `gpu_plan.py` to its root (Gibson keeps import shims), folds `select_episodes`,
  `explicit_episodes` and `gpu_plan_record` into `provenance.py` (the duplicate `source_fingerprint` in `gibson/run.py`
  is gone -- same digest), lifts `gibson/progress.py` to `progress.py` (shim kept), and moves the navmesh storey/stair
  reader to `habitat/stair_connectors.py`. The method's
  regression rig stays with the runtime (`tests/`); `core/planning/object_search/tests/` adds the boundary scan (no
  `tasks`, simulator or transport import anywhere in the method), a service-free smoke run and the failure
  contract. READMEs: `core/planning/object_search/README.md` (new), the runtime README's "Where the code is" and
  "Adding a dataset adapter", `core/control` (the camera branch), `core/mapping/{objects,tracking,topology}`,
  `core/planning/objnav`, `falcon`. Verified structural-only: every moved module is code-identical to its original
  (imports and docstrings aside) and four deterministic 140-step rollouts with fake services -- frontier and FALCON
  explorers, with and without a target, and the FALCON+YOLO baseline -- produce byte-identical traces, diagnostics
  and frozen configurations before and after.
### Added
- **Gibson validation result** (`objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md`, `gibson/results/gibson_val_20261018T105339Z/`, 2026-10-19): SR 0.788, SPL 0.465, DTG 0.96 m over 1000 of the 1,000 published SemExp v1.1 val episodes (5 scenes x 200), lean records, habitat-sim 0.2.4; per scene and per category, beside the paper's Table 2 rows, with the frozen configuration, in the file.
- **Gibson validation result** (`objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md`, `gibson/results/gibson_val_20261018_14b_sam/`, 2026-10-19): SR 0.784, SPL 0.480, DTG 1.00 m over 1000 of the 1,000 published SemExp v1.1 val episodes (5 scenes x 200), lean records, habitat-sim 0.2.4; per scene and per category, beside the paper's Table 2 rows, with the frozen configuration, in the file.
- **The FALCON + YOLO + closing baseline on the Gibson val split** (`gibson/results/gibson_val_falcon-yolo_20261018T173302Z/`, 2026-10-19): SR 0.665, SPL 0.337, DTG 1.75 m over the same 1,000 episodes as the full method's SR 0.792, SPL 0.464, DTG 0.94 m; per scene and per category in entry 032. `supervise_benchmark.sh` kills a run stalled for `STALL_MINUTES` (a habitat-sim segfault left a zombie it waited on for five hours) and calls an episode deterministic after three deaths.
- **The FALCON + YOLO + closing exploration baseline** (`objnav_benchmark_runtime/methods/falcon_yolo_baseline.py` new, `gibson/run.py --agent falcon-yolo`, `five_scene.py`, `run_benchmark.sh --agent`, `launch_detached.sh` `RUN_ARGS`, `supervise_benchmark.sh`, `floor_panels.py`, `visualization.py`, `search_panel.py`, `gibson/progress.py`, `gibson/FALCON_YOLO_BASELINE.md`, 2026-10-18): goal-free planar FALCON over the storey (bounded transactions in a 4 m disc about the agent, A* relocation to the nearest frontier when the disc is exhausted, a look-around and a STOP when the storey is) with the full method's YOLO perception cycle, `TargetClosing` takeover and legacy pursuit, camera controller and spawn-floor confinement unchanged, and no LLM, RPT*, scene graph, rooms, doors, openings, peeks or glances. The recorder draws the explorer's column in place of the search panel; lean records keep `exploration`. Twenty-one tests in `tests/test_falcon_yolo_baseline.py`. The swept-motion veto judges FALCON's own routes only (a relocation's or the pursuit's A* route is the planner's, as in the full method's frontier explorer; Wiconisco/000000 had spent 179 vetoes in its first doorway). The first fifteen recorded episodes (`runs/benchmark_falcon-yolo_15episodes_20261018T152349Z`): SR 0.800, SPL 0.433, DTG 0.94 m against the full method's 0.867 / 0.641 / 0.16 m on the same episodes.
- **Gibson validation result** (`objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md`, `gibson/results/gibson_val_20261016_small_llm/`, 2026-10-18): SR 0.792, SPL 0.464, DTG 0.94 m over 1000 of the 1,000 published SemExp v1.1 val episodes (5 scenes x 200), lean records, habitat-sim 0.2.4; per scene and per category, beside the paper's Table 2 rows, with the frozen configuration, in the file.
- **The box read around a surface in front of the object, and a standoff for a big object** (`objnav_benchmark_runtime/methods/box_occluder.py`
  new, `perception.py`, `target_path.py`, `target_closing.py`, 2026-10-15). Two Collierville tv failures that
  reproduced at the same actions in two 1,000-episode runs. (1) 000018: the television's box held 79 % television
  at 4.1 m and 20 % of a seat back's rail at 1.3 m in its lower middle; the 30th-percentile depth read the rail,
  the ±0.12 m support band kept 36 % of the centre patch against a 35 % gate, and the closing STOPped 3.3 m from the
  television after "approaching" the rail. `occluder_in_front` asks the whole box four measured things -- a second
  surface at least 0.5 m behind, spanning 80 % of the box's height AND width, covering more of the box than the
  near one, the near one leaving the top quarter of the box alone -- and `_supported_centroid` then sets the near
  pixels aside and reads the box again (`occluder_depth_m`, `occluder_px` in the row). A chair in front of a wall
  reaches the top of its box; a couch's backrest, a toilet's tank, a bed's headboard span the width only: none
  fires. The mask pipeline is untouched. (2) 000009: a 1.5 m television on its cabinet put every 0.70 m standoff
  point inside the cabinet, against it or behind it, in both planners, and three locks with the memory 0.02-0.14 m
  from the real television were released from 3 m as "no safe path from here"; a standoff 1.01 m from the refined
  memory with a seventeen-waypoint route was dropped as stale for a ring with none. `TargetApproachPath` widens the
  ring in the front arc out to the terminal radius when neither planner reaches the standoff ring (`widened`,
  `widened_standoffs`) and keeps a stale goal within terminal + tolerance when no ring point has a route
  (`stale_kept`, `stale_standoffs_kept`); `TargetClosing` begins the inspection from such a standoff with the fresh
  range within a forward step of the limit and STOPs when the inspection's steps are spent or blocked
  (`closest_approach_stop` on by default, `closest_approach_stops`; `note_blocked` / `note_action` tell the
  inspection whether its step moved). Re-run of the three recorded episodes
  (`runs/gibson_Collierville_tv_fix_20261015T064436Z`): 000018 SR 0 -> 1 (SPL 0.81, 45 actions, the occluder set
  aside on 10 frames, the lock at the real television), 000009 SR 0 -> 1 (SPL 1.00, 32 actions, one widened
  standoff, no boxed release); 000014, a detector false positive, unchanged. The toilet yardstick
  (`runs/benchmark_toilet_10episodes_20261015T064732Z`, the seeded ten): SR 0.90, SPL 0.530, DTG 0.23 m against the
  2026-10-14 reference's 0.90 / 0.456 / 0.59 -- the same nine successes and the same one timeout. Tests: `tests/test_box_occluder.py`
  (9) and six standoff/closest-approach tests in `tests/test_target_closing.py`; the runtime suite 860 pass. The
  per-episode ground-truth tools behind the diagnosis are `runs/_analysis/closing_vs_truth.py` and
  `replay_depth_dissection.py` (gitignored). LESSONS has the entry.
- **The STOP overshoot: decide, then walk the margin** (`objnav_benchmark_runtime/methods/target_closing.py`,
  `rpt_policy.py`, 2026-10-14 night). Every terminal reading is biased short of the object's footprint -- the near
  edge of a box with floor in it, a centroid fused from near surfaces -- and the evaluator measures to the
  footprint: Markleeville/000188 STOPped on a 0.84 m near edge 1.27 m from the toilet (DTG 0.27, SR 0) in the run the
  user chose, `runs/benchmark_toilet_10episodes_20261014T143259Z` (SR 0.80, SPL 0.366, DTG 0.62 m;
  Collierville/000036 129 actions, SPL 0.67, with the back-off fix). Once a STOP is decided, on whichever of the
  seven paths of `_inspect`, `TargetClosing._terminal` walks `ClosingSettings.stop_overshoot_steps` (1) more
  MOVE_FORWARD toward the 3-D memory -- a second, up to `stop_overshoot_max_steps` (2), only while the memory
  still stands beyond `stop_overshoot_within_m` (0.90 m) after the first -- and STOPs then: no frame is consulted (`observe` short-circuits), nothing is
  re-verified; a step is taken only while the memory would still stand `stop_overshoot_min_range_m` (0.45) beyond
  the agent and the deciding frame's near edge a body radius, the agent faces within a quarter turn of the memory
  (beyond one turn the converter turns toward it first), a step the converter reports blocked ends the stage
  (`note_blocked`, from the policy's `notify_blocked`; `note_action` counts the steps) and
  `stop_overshoot_max_actions` (5) ends it without the hooks. Phase `ADVANCE`; `stop_overshoot` in the command
  info and the diagnostics; counter `overshoot_commands`. 0 restores the STOP on the deciding frame -- the shared
  test fixture runs the decision tests so. Three fixes tried in between (`unvalued_prob` 0.05, a fused-memory
  range for small objects, a fold gate on the unknown behind a doorway; run `…150615Z`, SR 0.56) were reverted --
  the fused-memory range flipped CLOSE and INSPECT on alternate frames for 25 actions -- and are in `LESSONS.md`.
  Verification run with two steps, `runs/benchmark_toilet_10episodes_20261014T155701Z`: SR 0.90, SPL 0.450,
  DTG 0.59 m; every episode retraced `…143259Z` within two actions, Markleeville/000188 went from DTG 0.27 to SPL
  1.0, seven of nine overshoots took both steps and two stopped after one at the minimum range. Two steps were
  judged too dear for the SPL (every other STOP 0.01-0.07 lower); one step always (`…162832Z`) gave SR 0.80,
  SPL 0.357 with 000188 STOPped 3 cm outside the disc. The memory at the deciding frame tells the two apart
  (0.88-1.19 m in the eight successes, 1.25 m in 000188), hence the conditional second step -- verified in
  `…170224Z`: SR 0.90, SPL 0.456, DTG 0.59 m, the second step taken on 000188 and Collierville/000023 only.
- **A door beside the stairs, a camera at half a step, and a bump the rays erased** (`objnav_benchmark_runtime/methods/opening_nodes.py`,
  `room_search_loop.py`, `target_closing.py`, `observed_map.py`, 2026-10-14 evening). The openings' stair exclusion
  is directional -- only masked cells and portal points on the frontier's unknown side count (`near_stair_mask(...,
  heading)`, `beyond`): Darden/000034's bathroom door, 1.3-1.5 m beside the flight, was struck off the openings for a
  whole episode and the search walked 17 m to an "entered" living room instead. The closing's geometric pitch is
  quantised with hysteresis (`ClosingSettings.pitch_hysteresis` 0.15, `_quantise_pitch`): a wanted angle at half a
  tilt step chattered LOOK_DOWN / LOOK_UP with the memory's millimetre flicker and the agent never stepped. A second
  bump within `ObservedMap.repeat_bump_m` (0.5 m) of an earlier one leaves a sticky, body-wide bar in every export
  (`_touched`, `sticky_bumps`): the single `l_max` cell decays under the rays that look over a ledge the navmesh
  refuses, and Corozal/000019 bumped three times at one spot and lost its three nearby nodes to the releases.
  The back-off's retreat spot must show the target on the observed grid (`BackoffManeuver._sees`, a Bresenham
  ray stopped short of the target's cells): Collierville/000036 retreated 0.5 m from the goal viewpoint to a spot
  behind the door jamb, disproved the 0.90 toilet from there and lost 380 actions.
  Third verification run `runs/benchmark_toilet_10episodes_20261014T132341Z`: SR 0.80, SPL 0.477, DTG 0.88 m.
- **An exit the oracle has not valued is a node at a prior, not at 0.0** (`objnav_benchmark_runtime/methods/room_search_loop.py`,
  `opening_nodes.py`, `visualization.py`, 2026-10-14 evening). The oracle's numbers are replaced wholesale at every
  call and the fold is read against the numbers before it, so an exit folded into its room at the call and unfolded
  by the reply that dropped the room under `min_prob` came out with no number and read 0.0 -- Corozal/000142 lost
  the toilet's own doorway (12 m2 unseen, 6 m away) to a 12 m gap at 10 % with no SELECT in the 54 actions left.
  `RoomSearchLoop._exit_prob` hands such an exit to RPT* at the oracle's last number for it (`_exit_memory`) or at
  `OpeningSettings.unvalued_prob` (0.10); `prob_source` (`oracle`/`remembered`/`provisional`) in the estimate and
  the HUD record, event `opening_provisional`, stat `openings_provisional`. The node still buys the next call.
  Second verification run `runs/benchmark_toilet_10episodes_20261014T124355Z`: SR 0.90, SPL 0.522, DTG 0.30 m.
- **Five losses of one Collierville recording, and a flight the guard let through** (`objnav_benchmark_runtime/methods/exploration_fallback.py`,
  `opening_nodes.py`, `room_search_loop.py`, `path_glances.py`, `spawn_floor_guard.py`, `rpt_settings.py`;
  `core/planning/exploration/view_gain.py`, 2026-10-14 afternoon). The object-shadow rule no longer drops a frontier
  for a clock or a vase beside it (`SHADOWLESS_CLASSES`, `shadow_min_radius_m` 0.25) nor one with a room's floor
  right behind it (`shadow_exemption`: `shadow_exempt_m2` 2 within `shadow_exempt_range_m` 1.5, through the walls;
  `UnknownView.unknown_in_sector(max_range_m=...)`), and `detect_openings` applies the same rule -- Collierville/
  000023 lost the frontier onto its toilet's room as "the shadow of refrigerator 0" and a doorway as a clock's. A
  peek the supervisor's 0.6 m room-centre tolerance declared arrived short of its threshold walks the remainder and
  looks two turns to each side of the way it came through (`OpeningSettings.step_through`, `through_look_turns` 2,
  `through_steps` 8; `peek_step_through`). A transit whose room is missing from the registry keeps walking for
  `LoopSettings.room_missing_grace` (3) actions before the room is given up (the watershed flickers). The glance
  state records the gains at the agent's own position (`glance.here`). The spawn-floor guard's drop threshold is a
  knob, `RPTSettings.step_down_m`, kept at 0.30 m: 0.15 m masked Corozal/000142's own storey (the plane is the pose
  height). A floor-cut box is judged by its height, with an 8 % width floor (`clipped_box`): a toilet's tank at 1.3 m
  is 65-83 px wide and the 15 % bar refused four 0.91-0.95 frames. The floor-support gate puts an object on the
  storey when ONE terrain sample near it is the storey's floor (`PerceptionCycle._floor_support`): the lowest sample
  beside a toilet at a stairwell was the stairwell's. LESSONS 2026-10-14 (afternoon) has the traces.
- **The toilet re-run's nine behaviours, fixed from their traces** (`objnav_benchmark_runtime/methods/target_closing.py`,
  `target_path.py`, `room_search_loop.py`, `exploration_fallback.py`, `opening_nodes.py`, `perception_cycle.py`;
  `core/mapping/topology/search_node_oracle.py`, 2026-10-14). Target closing: a re-sighting of the ACTIVE candidate
  counts past the rejection memory (it judges a start, not a frame; `resighting_exemptions`); a depthless box on the
  anchor's bearing is a `glimpse` that carries the chain, is faced and stepped toward without running the verify clock
  (`max_verify_approach_steps` 24); the inspection takes one step when the fresh range is a step beyond the limit
  (`inspection_max_steps` 3) and reads a turn that would undo the last turn as aligned (`dead_band_alignments`);
  a locked target with no route on the known floor is walked to through the unknown before the spot is given up
  (`through_unknown`, `through_unknown_cost` 8, the policy's `optimistic_planner`); a candidate first read at
  `backoff_trust_confidence`, or one the detector had been firing on (`backoff_persistence_frames` 2 of
  `backoff_persistence_window` 6), is not backed off from; a fruitless sweep cycle releases the verification
  (`verify_sweep_max_actions` 6). The map: a target landmark seen once at `landmark_sighting_confidence` 0.50 is a
  `T` node, only a release that DISPROVED a candidate hides one (`landmark_refused`), and the floor-support band
  is 0.60 m. The loop: `second_pass` and `scanned_to_oracle` off (a room scanned once is finished and is no node of
  the oracle's), `lingered_discount` 0.2 for an entered room searched 20+ actions with at most one significant
  frontier (and the prompt says so), and a frontier-less room is entered at its doorstep, not its vantage point.
  The fallback: exits ranked by the unknown a look from each reveals, slivers under `explore_min_gain_m2`
  demoted, and the goal in force kept across a glance (`goal_memory_actions` 12). The same 10 toilet episodes
  (2 per scene, seed 17) went from SR 0.700 / SPL 0.425 / DTG 0.94 m to SR 0.900 / SPL 0.489 / DTG 0.23 m
  (`runs/benchmark_toilet_10episodes_20261014T085618Z`, before round 2); LESSONS 2026-10-14 has the traces.
- **Gibson validation result** (`objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md`, `gibson/results/gibson_val_20261012T141500Z/`, 2026-10-14): SR 0.713, SPL 0.409, DTG 1.27 m over 1000 of the 1,000 published SemExp v1.1 val episodes (5 scenes x 200), lean records, habitat-sim 0.2.4; per scene and per category, beside the paper's Table 2 rows, with the frozen configuration, in the file.
- **The detached Gibson run as one command, and the same command after a reboot**
  (`objnav_benchmark_runtime/gibson/launch_detached.sh`, 2026-10-12): launcher in its own session, monitor and
  supervisor beside it, `PROGRESS_ENTRY` required so publishing closes the right entry, the supervisor's settings in
  `<output>/launch.env`; on an existing run it starts the Ollama container and a fresh supervisor that resumes from the
  replay snapshot; `--stop` ends a run for good. `--after-reboot` is for a card that is dead until the machine restarts
  (driver upgraded under the running kernel module): the launch's variables go to `launch.env` and a one-shot `@reboot`
  crontab entry launches once `nvidia-smi` and `docker` answer -- never on the CPU, since the device layout is frozen
  into the run. The supervisor's `alive()` now counts a PID only if its command line names the run (PID reuse after a
  reboot). BENCHMARK.md 2.3 documents both.
- **The box in the frame drives the camera pitch** (`objnav_benchmark_runtime/methods/bbox_pitch.py`,
  `TargetClosing._servo`, `TargetClosingSettings.bbox_pitch / bbox_pitch_distance_m 1.50 / bbox_pitch_region
  0.25`, 2026-10-11): within 1.5 m of the target, with the LOOK pair and a fresh box, the pitch is an
  image-space servo's and not the 3-D memory's -- a box in the lower quarter of the frame, or predicted to be
  cut by the lower edge on the next step toward it, is looked DOWN at by the rounded number of tilts; one in
  the upper quarter or about to be cut above, UP; one in the centre band leaves the camera where it is, even
  where the geometric controller would force a tilt (the two would hunt). No class, height or memory enters:
  a frame cut by the lower edge biases the memory exactly when the pitch matters, and every terminal STOP of
  the nine Gibson inspection recordings fired on such a frame. The geometry remains the fallback for a frame
  without a fresh box; without the LOOK pair (the published Gibson protocol) nothing changes. The band is
  never narrower than one tilt's rounding dead band, so the servo cannot hunt; pitch limits cut a tilt back.
- **The terminal STOP is taken on a whole, confident box** (`TargetClosing._before_stop`, `stop_centring`,
  `stop_confidence` 0.50, `weak_stop_max_holds` 2, 2026-10-11): with the LOOK pair, a fresh in-range box the
  servo can re-frame earns ONE centring tilt per inspection and the STOP is taken on the next frame -- on the
  whole box if the detector shows it, on the previous frame's evidence if it does not (the tilt may cost an
  action, never the STOP); a fresh in-range box under `stop_confidence` (the takeover's own start threshold:
  detect high, associate low, STOP high) is held for the one frame that keeps it whole at most twice, then the
  STOP goes ahead on the evidence there is. The STOP's info carries `confidence`, `bbox_region` and
  `depth_valid`; `bbox_pitch_corrections`, `stop_centring_tilts`, `weak_stop_holds`, `bbox_region`,
  `bbox_pitch_steps`, `box_drift_rows` and `goal_kept` join the diagnostics.
- **Regression check of the toilet closing rules on the nine inspection slots** (`runs/gibson_val_20261008T120112Z
  .inspection-fix`, progress entry 027, 2026-10-11): toilet 2/3 improved (Darden/000193 SR 1 at 129 actions,
  Wiconisco/000031 SR 1 at 253), couch 3/3 the same outcome (the Wiconisco false positives are the detector's,
  at 0.72-0.91; 000114 STOPs 31 actions earlier on the same wrong object via the blind-cone rule), bed the same,
  tv Collierville/000043 the same, tv Collierville/000194 regressed -- traced and fixed below.
- **Re-record named episodes of a run for failure inspection** (`objnav_benchmark_runtime/gibson/record_episodes.sh`,
  `BENCHMARK.md` 2.7, 2026-10-11): `record_episodes.sh <run> <out> [--after-run] label=Scene/NNNNNN[,fallback...]...`
  re-runs each slot with `--record` under the run's own frozen settings -- the `run.json` argv addressed to one
  episode (`--scene S --shards 200 --shard-index k --limit 1`), the snapshotted environment, the services reused
  or replayed -- keeps the first candidate whose re-run reproduces the original outcome (a GPU LLM is not
  bit-exact) and tries the next otherwise, and writes `videos/<NN>_<label>_<Scene>_<k>.mp4`, the final frame,
  for a wrong STOP the frames at the closing's first VERIFY / INSPECT / STOP decisions, and `index.md` with the
  original beside every attempt. `--after-run` queues it behind a benchmark still scoring; `--preflight` checks
  the slots without recording. First use: nine representative failures of the 1,000-episode validation run
  (toilet timeouts, Wiconisco couch false positives, circuitous tv/bed successes).
- **Detached Gibson run monitor** (`objnav_benchmark_runtime/gibson/monitor.py`, `tests/test_monitor.py`,
  `BENCHMARK.md` 2.3-2.5, 2026-10-08): the companion of a `nohup`/`setsid` launch of the 1,000-episode
  benchmark that nobody is watching. Every tick it appends one status line (UTC, done/total, running
  SR/SPL/DTG, ETA, the episode in progress, the last outcome) to `<output>/monitor.log` and keeps
  `<output>/episodes.csv` current -- the scalar columns of `episodes.jsonl` alone (id, scene, target,
  success, SPL, SoftSPL, path length, shortest path, start and final distance to goal, steps, STOP,
  termination, wall time, agent error), no diagnostics. Completion is judged by the launcher's PID
  (`launcher.pid`, written by the launcher itself because `setsid` forks), `exit_code.txt` supplies the
  code and is ignored when older than the PID file (a preflight into the same directory leaves one);
  the finished summary goes to the log and `FINAL_SUMMARY.txt`. `BENCHMARK.md` now gives the detached
  recipe, the stop command (`kill -TERM -- -PGID`, the lock being an advisory `flock` so a resume
  follows) and lists the new files.
- **LLM-first node oracle for the single-storey benchmark** (`core/mapping/topology/search_node_oracle.py`,
  `objnav_benchmark_runtime/methods/{house_context,room_search_loop,scene_graph}.py`, 2026-10-07): the 14B
  model is shown EVERY room of the storey -- finished and type-excluded ones included, each with
  `status=never_entered|entered|scanned(<how>, <ago>)` -- and a HOUSE line (room types found with their
  status, rooms unidentified, openings unlooked, whether frontier is reachable, `actions used about N of
  500`), and asked to reason as a person who knows homes does (where the target lives, incl. an en-suite
  toilet reached through a bedroom and a television in a living room then a bedroom, never a bathroom;
  what the rooms found imply for the unidentified ones -- one kitchen, one living room; and how the
  budget weighs: unexplored first, scanned home-type rooms late) before any number, writing `home`,
  `house`, `stage` and a `pass` verdict (`first`/`second`). Under `second` (or a floor with no unfinished
  room, exit or frontier) finished rooms are nodes again and a revisit scans from a spot
  `LoopSettings.revisit_standoff_m` (1.5 m) from the earlier scan points (`second_pass`,
  `second_pass_forced`, `revisits_*` stats). The three arithmetic floors (`unexplored_floor`,
  `unexplored_elsewhere`, `home_floor`) and the hard type exclusion (`type_prior`) are OFF by default; the
  storey/stairs rules became a `STAIRS_SUPPLEMENT` shown only when a staircase is a node. The reasoning
  model is not spent on a loop point with nothing offerable (`oracle_calls_skipped`), in `reconsider()`
  too. `tests/probe_node_oracle.py`: five single-storey scenarios, all five pass on `qwen2.5:14b-instruct`.
- **One-click Gibson benchmark** (`objnav_benchmark_runtime/gibson/run_benchmark.sh`, `BENCHMARK.md`,
  `gibson/progress.py`, `gibson.run --lean --progress-file`): services up (Ollama docker/native/external,
  the CPU YOLO-World detector), preflight, the 1,000 published episodes with LEAN records (the scores
  and the counters that explain an outcome, ~10 KB per episode instead of ~100 KB), a console progress
  bar per episode, `progress.json` rewritten atomically after every episode (done/total, ETA, the
  episode in progress, running SR/SPL/SoftSPL/DTG overall, per scene and per category), `--status DIR`
  and `--summary`, automatic resume into an existing `--output`, services down on exit. The README
  covers install, data, weights, one episode, the full split, sharding, reading the results.
- **The context check on a target candidate** (`objnav_benchmark_runtime/methods/target_context.py`,
  `TargetClosingSettings.context_*`, 2026-10-07): a box whose footprint lies in a room whose STRONG
  type the target is not searched for in (`room:kitchen`), or whose measured centroid height
  contradicts its class (`CLASS_HEIGHT_BANDS`, calibrated on the Allensville recordings: a bed's or
  a toilet's surface under 0.60 m above the agent base, `height:0.69`), counts 0.70 of its confidence
  toward the takeover's start threshold and needs four consecutive frames from two viewpoints
  0.30 m apart to lock. Allensville/2 and Newfields/2 of the 5x3 benchmark STOPped on a kitchen
  counter front read as `bed` from 0.55 m after two frames from one spot (`suspect`,
  `context_rejections`, `suspect_locks_held` in the diagnostics).
- **`override_confidence` (0.80)** on the takeover: a box at that confidence starts a takeover past
  the rejection memory of `unverified` and `contradicted by the map` releases and the release
  cooldown -- a 0.95 toilet in plain view at 2.4 m is not a flicker (the Allensville toilet run
  refused nine such frames) -- while a failed-inspection spot, a stalled approach, a boxed-in spot,
  border clipping and the map's own contradiction still hold; the rejection memory records `why`.
- **A verification frame that keeps the box whole, with the pitch following the target's
  elevation** (`verify_keep_in_frame`, `TargetClosing._elevation`, `elevation_band_m` 0.15 m): a
  fresh, aligned candidate too close to step toward used to be held, and a satisfied hold is executed
  as a TURN (the headless agent's idle action), which moved the centred Allensville toilet out of the
  frame at 1.18 m. The frame is now a LOOK in the direction of the target's elevation -- its 3-D
  centroid against the camera's height, or its box centre against the horizon -- DOWN at a toilet,
  UP at a wall-mounted television, never a fixed direction; level with the camera, the LOOK or TURN
  whose predicted pixel shift keeps the box in the frame.
- **Dynamic pitch on the close approach** (`look_down_distance_m` 1.30 m is now the band of the
  controller, both directions): CLOSE used to force a level camera until the terminal inspection;
  inside the band the approach carries the target's own geometric pitch (`_pitch`, at least one tilt
  toward its elevation, the same function the inspection uses), so a low toilet or a high television
  is held whole on the terminal frames. `_pitch` itself is symmetric now: a target above the camera
  within the band gets LOOK_UP where the rounded geometry said level.
- **Thresholds the agent walked through are cuts** (`room_watershed.trail_thresholds`,
  `WatershedRoomParams.threshold_*`, `ObservedSceneGraph.trail/thresholds`): a clearance dip of
  0.55 m or less with 0.30 m more clearance within 1.5 m of travel on both sides is a doorway,
  carved at the severing choke within 0.5 m exactly as a snapped door is (never a plain disk), so
  the room beyond separates the tick the agent is through it; a uniformly narrow corridor never dips.
- **A home object is a prior** (`LoopSettings.home_floor` 0.60, `SearchNode.home`,
  `SearchNodeOracle(home_floor=)`): a room holding a confirmed home object of the target (a bathtub
  for a toilet) is never read below the floor, whatever the model wrote or its storey verdict says,
  and is never finished by sight from outside it (`seen_from_scan` / `seen_through`) -- the
  Allensville toilet run finished the bathroom with the bathtub inside from the spawn point's spin
  and valued it 0 (`home_object_kept` events with `scanned`).
- Progress entry `docs/progress/entries/024-allensville-audit-perception-rooms-priors-closing.md`,
  regressions in `tests/test_perception_association.py`, `tests/test_target_context.py` and the
  registry/watershed/oracle/scan suites (1969 pass across the runtime, exploration, topology and
  objnav packages; 1891 before).

### Fixed
- **A refined target memory no longer throws away a reachable standoff goal** (`objnav_benchmark_runtime/
  methods/target_path.py`, `TargetApproachPath.kept`, 2026-10-11): a memory shift over `refine_distance_m`
  (0.15 m) used to drop the standoff goal and re-select one from a bearing-relative ring about the new
  position -- a different set of points, each needing a navmesh projection and a path. Collierville/000194:
  the along-ray association refined the television's memory by 0.11 m at 2.5 m, the ring found no path from
  a spot the old goal was reachable from, and the real television was released twice as "no safe path from
  here" (SR 1 -> 0 against the baseline, whose un-refined memory kept its goal). A goal that still ends
  inside `terminal_distance_m` of the refined memory is now kept, the reference is refreshed on every
  refinement (`goal_refinements` no longer climbs every action after a failed re-selection), and a kept goal
  a slow drift carries past the terminal radius is re-selected the action it falls out.
- **The full Gibson run died at episode 919 on a start 3.3 cm outside the navmesh; finishing it is
  explicit and recorded** (`objnav_benchmark/{kinematics,logger,run_info}.py`, `gibson/{protocol,run,
  report}.py`, `gibson/supervise_benchmark.sh`, 2026-10-11): habitat-sim's step filter runs for a turn
  too (`turn_left` is a body action), and `try_step(start, start)` pulls a start that lies *outside* the
  mesh horizontally onto its edge -- Wiconisco/000118, the only such start of the 1,000 (measured),
  "moved" 0.033 m on its warm-up turn and the 0.020 m turn-in-place bound ended the run, twice. New
  `KinematicTolerance.settle_xy_m` (Gibson: 0.05 m, one map cell) excuses that much horizontal placement
  on the first action that moves the agent, like `settle_m` does vertically. Because the fix changes the
  frozen configuration (`kinematics`, `source_sha256`), the resume that finished the run needed
  `gibson.run --resume --resume-accepting-changes kinematics source_sha256`: `MetricsLogger` accepts a
  difference in the named keys only (`RESUMABLE_CONFIG_CHANGES`: provenance or a self-check, never a
  decision or score input), keeps the recorded config, writes both values into the resume's `run.json`
  entry, and the report titles and discloses it. The supervisor forwards `RESUME_ARGS` on every replay.
- **The 1,000-episode run no longer dies on the kinematic contract** (`objnav_benchmark/{kinematics,
  env_contract,runner}.py`, `gibson/protocol.py`, 2026-10-07): published starts lie up to 0.28 m below
  the shipped navmesh and habitat-sim settles the agent on its first TRANSLATION (turns bypass the
  navmesh filter); `KinematicTolerance.settle_m` (0.30 for Gibson) takes that off the first moving
  action additively, and `climb_m` is 0.60 for Gibson because the evaluated storey itself steps up to
  0.50 m on one stride (treads, split levels; the 0.20 m bound refused ~1 step in 300). An agent position
  off the floor map reads as the unreachable sentinel instead of raising; `merge_runs` refuses
  diagnostic shards and compares `target_override`/`publishable`/`kinematics`.
- **Target closing** (`methods/target_closing.py`): a near candidate (inside 2 m) released from the spot
  the agent still stands on is refused until it moves (the rejection memory records where the agent
  stood; `same_spot_rejections`), a suspect candidate may step inside the terminal range to earn its
  second viewpoint, a locked target without a path is kept in frame instead of the idle hold that the
  headless agent executed as a turn (233 reversals in one Darden episode), and a frame the detector did
  not see is NO evidence: the chain is carried, the clocks pause (`blind_frames`) -- before, a transient
  detector failure released a correct lock with a 2 m rejection around the real target.
- **Perception**: a malformed detector reply for one frame is a frame failure (`DetectorFrameError`,
  back-off) rather than the end of the run; only identity drift (vocabulary/model) still is. The
  per-frame `/health` GET is gone (the `/detect` reply echoes the identity). An LLM request that runs
  into its timeout is never retried (one hung call cost 20 minutes).
- **Room-search loop**: a SEARCH-state `route_failed` (the scan's vantage unreachable with the agent let
  in by the arrival tolerance) now ends the turn `unreachable` instead of being dropped for 36 actions of
  blind fallback; a completed peek covers the cone it looked into, so the frontier it reveals through
  the same door is not a new opening (the "twenty openings" drain); the vantage-point EDT runs on the
  room's bounding box; the map-edge `ValueError` is caught inside the fallback guard.
- **Half-levels** (`methods/spawn_floor_guard.py`, `core/planning/exploration/floor_atlas.py`): a settled
  plateau 0.5-1.5 m off the spawn plane that the atlas read as a landing for ever (Klickitat: 361 idle
  turns) is adopted as a level of the spawn storey after 30 actions (`FloorAtlas.adopt_level`,
  `half_level_adopted`); the storey below is not a drop, goals on the level are allowed; a plateau a
  storey away stays confined.
- **Classes never merge in the landmark map, and same-class instances are told apart by size**
  (`core/mapping/objects/landmarks.py`, `floor_context.new`, `RPTSettings.landmark_dedupe_radius_m`
  0.35 / `landmark_footprint_iou` 0.25, 2026-10-07): the ObjectNav map no longer takes the class vote
  (`class_votes=False`) -- a cup on a table, a toilet beside a bathtub, a vase on a cabinet are separate
  instances whatever their proximity or footprint overlap -- and two observations of one class are one
  instance only within 0.35 m (the ported 0.70 m merged two dining chairs) or at a footprint-disc IoU
  of 0.25 (about one radius apart: a bed re-seen from its other side is one bed). The two vases on the
  cabinet in the first frame of the Allensville couch run were dropped as `same_frame_association` on
  every frame, and the toilet and the bathtub 0.5 m beside it were voted into one landmark that flipped
  to `bathtub` and released the toilet's lock as "contradicted by the map" with a 2 m rejection radius.
  The vote (`contradicted_by_map`, `map_rejections`, `map_releases`, `outvoted_detections`) stays in
  the library behind `class_votes=True` and reads zero in this runtime; the per-frame alias filter
  (a box of another class over the same pixels, IoU ≥ 0.9) is the one cross-class rule kept, on the
  detector's output, not the map. `RPTSettings.detection_confidence` 0.35 -> 0.30 (the takeover's
  tracking threshold; a landmark needs two observations, so a weak box costs nothing alone).
- **A box cut only by the bottom edge is not a sliver** (`perception.clipped_box`): a low object at
  1-2 m is cut by the bottom edge on every level frame; a box touching only that edge, with its top
  in the lower half and 15 % of the frame in both dimensions, counts. The Allensville toilet run
  refused 23 such frames of a 0.97 toilet at 1.2-1.7 m and released the candidate twice.
- **Room numbers follow the room, not the larger half** (`RoomRegistry(anchors=True)`,
  `TrackedRoom.anchor`, `memory_rooms`): a fresh mask holding a previous room's birth anchor is
  matched to it before any IoU score (oldest pid first on a merge), and the scene graph's registry
  remembers vanished rooms for the episode (`REGISTRY_MEMORY_TICKS` 1000, bounded to 48 masks). The
  Allensville spawn room was R0 -> R1 -> (merged) R0 -> R2 across steps 0-44 while the hallway the
  agent walked into carried R0 and its "entered" record.

### Added (2026-10-05)
- **The sight ledger** (`objnav_benchmark_runtime/methods/sightlines.py`, `SightSettings`,
  2026-10-05): every action the camera's cone is cast over the map it has just updated; an
  unknown cell a ray ends at inside the floor's visible band (beyond the blind radius, within
  2.5 m) from two distinct poses is **looked through** -- a railing, a window, a mesh hole -- and
  a connected unknown region under 3 m2 enclosed by known cells is a **pocket**. Both are
  settled: written OCCUPIED on the map the frontier logic reads (`ObservedSceneGraph.
  frontier_world` via `resolved_provider`; `accessible_frontiers(frontier_world=)`;
  `UnknownView(resolved=)`), so they are no frontier, no opening, no fallback exit and no
  glance gain; the planner and the display keep the real map (the panels draw settled unknown
  darker, `settled_cells`). Hanson 2026-10-05 walked to both ends of a balcony seen whole from
  its threshold, back to an island between the spawn point and the bed (O4, 31 actions) and
  behind a bed (O9, 42 actions). `sight` in the configuration and the episode record.
- **Two more finished-room verdicts** (`room_scans.py`, `LoopSettings.walkthrough_clearance_m`
  0.9 m / `fragment_max_m2` 3 m2): with no live frontier left, a room is `seen_through` when one
  recorded pose had half of it in the camera's cone with clear sight, or the agent stood in it
  and it is narrow (a corridor, a balcony); and a `fragment` when it is under 3 m2 with no door
  on it (the strip behind a bed). A room finished by what the walk to it showed is released
  before it is entered (`finished_in_transit`). The Hanson balcony was offered again at 0.25,
  "never entered", 158 actions after the agent had stood at its far end.
- **Cue glances** (`path_glances.py`, `GlanceSettings.cue_*`): a confident detection spanning
  30 % of the frame's height and cut off by its left or right edge turns the agent toward it -- enough turns to centre it and one
  more; the target's class and its home objects outright, any other class while that side still
  holds unknown floor; once per class and side within 2 m. Hanson action 101: a bathroom vanity
  (`cabinet 0.84`) on the left edge, two right glances taken, the toilet 250 actions later.
- **The footing sweep** (`camera_control.begin_inspection(reason="footing")`,
  `TargetClosingSettings.footing_*`, the exploration fallback's sixth rung): one LOOK_DOWN to 30
  degrees, a full circle and a LOOK_UP map the floor under the camera's blind radius, which an
  agent that has not moved cannot plan across (unknown is impassable); taken only while a quarter
  of the cells within 1.2 m are unknown. A LOCKED target with no
  safe path asks for it after two pathless actions and, six pathless actions after it, releases
  the lock **without a rejection** (`boxed_releases`; no far takeover from within 1 m of the
  spot) so the search can walk; the fallback takes it before its last-resort hold. Hanson/000002
  spawned beside a bed with the plant in view 4.4 m away and spun 159 actions on "no safe target
  path" to an agent error.
- **Signature objects** (`room_priors.SIGNATURE_OBJECTS`, `RoomLabelSettings.signature_objects`):
  a bed, a toilet, a shower, a bathtub, an oven, a stove or a refrigerator makes the classifier's
  agreeing label STRONG on its own, so the type prior rules the room out at once; the home-object
  and target-seen guards still hold. Hanson action 270: a bed and a television through the door,
  the "bedroom?" kept weak, the room entered and scanned for a toilet.
- **A start-clearance rule in the episode samplers** (`generate_development.
  MIN_START_CLEARANCE_M` 0.35 m, `start_clearance`; both `generate_start` and
  `cross_floor_starts`; `start_clearance_m` in the audits): no start within 0.35 m of the navmesh
  boundary. Frozen manifests keep their starts.
- **`generate_development --multistory --categories couch`** (`multifloor_generation.
  cross_floor_starts(categories=)`): a cross-floor campaign restricted to the goal categories named,
  skipping buildings without one on the reference floor; `generation.goal_categories` in the manifest.
- **Distractor prompts** in the Gibson context vocabulary (`toy`, `rocking horse`,
  `stuffed animal`) and `bathtub`: Hanson/000001 stopped at action 14 on a child's ride-on horse
  read as `chair` 0.9. A running detector service must be restarted with the new vocabulary.
### Changed
- **End-to-end review before the 5x3 benchmark (2026-10-05, evening)** -- small fixes only, found by
  reading every layer and replaying the Ranchester recording:
  - *Room ids* (`room_watershed.door_carve_mask`, `RoomRegistry(memory_ticks=10)`): a door snapped to a
    choke whose disk did not actually sever the floor had no separating power (the medial axis was
    parted, the floor was not; 47 % of the snapped disks in the Ranchester replay), so the room behind
    it merged into the hallway on the ticks the snap found a choke and split off under a new number on
    the ticks it did not -- R23, R26, R28, R52 for one room. Such a disk now falls back to the plain
    barring disk; and the registry keeps a vanished room's mask for ten updates, so a room absorbed
    for a tick re-adopts its own pid (ties broken by overlap, not raster order). The replay goes from
    36 pids for 10 rooms (26 deaths) to 11 pids for 11 rooms (3).
  - *Finished rooms re-judged on growth* (`RoomScanLedger(regrow_factor=1.5)`, `regrown` in the
    diagnostics): a sticky `fragment` / `seen_through` verdict reached on a sliver no longer finishes
    the room that grows out of it under the same number; a scan point still inside finishes it again.
  - *Generic objects alone classify nothing* (`RoomLabelSettings.generic_evidence_gate`): one cabinet
    made a "kitchen" at 0.9 on Ranchester's upper storey and the storey summary showed the oracle a
    kitchen found; weak labels are marked `kitchen?` in that summary as they are on the node lines.
  - *The node oracle prompt*: STEP 2 reads the storey from the rooms found AND its RANK in the building
    (`stair_nodes.storey_position`: "this storey is the highest of 2 known storeys, 1 above the lowest"
    instead of "(this one at +3.1 m)", which the 14B read as a ground floor; the rooms decide, and the
    line names no floor -- a first version that called every storey above the lowest "an upper floor"
    had the 14B say the living room lives elsewhere on Newfields' ground floor, whose lowest storey is
    a basement, with a living room found on it); an opening whose glimpsed objects
    name the home type is rule 1 (60 or more -- the 14B valued a sink-glimpsed gap at 0.20 for a
    toilet, "a bathroom, no toilet"); the worked example carries a toilet counter-example so a small
    model cannot copy its `elsewhere`; `parse_home_here` reads a negated "found" and a bare "no" as
    `missing`, never `elsewhere`.
  - *The room classifier* reads "Living Room" / "living-room" as `living_room` instead of `unknown`.
  - *The evaluator* (`MultiFloorDistance.success`): the height gate is the nearest goal sample's, not
    the region's median -- Klickitat's chair samples lie at +0.18 m and -0.41 m (a split-level floor),
    and a STOP beside the lower chair scored 0 four centimetres from the region. The campaign report
    writes `statistics.json` and `RESULTS.md` before it raises on a missing video (`missing_recordings`).
  - *Generation*: `generate_development --multistory --start-storey same` samples the start on the
    annotated storey (SemExp-style same-floor starts on the multistory harness, so SR is measurable),
    and a building with fewer annotated categories than episodes is ineligible (`distinct_categories`;
    Onaga's three episodes would all have been the couch).
  - *Target closing*: a fresh, off-centre sighting at a pitch other than the predicted one is centred
    at the pitch it was seen at (the converter tilts before it faces, so the predicted pitch LOOKed
    away from a target in view for the whole 24-action budget); an inspection the filtered estimate
    entered while every fresh frame measures the surface beyond the limit is ended and the approach
    resumed (`inspection_resumptions`) instead of releasing a target in plain view as "saw nothing";
    a glance in force is aborted the action a takeover starts; the landmark map keeps voting during
    the takeover so the documented map release can fire (the legacy target evidence stands down,
    `takeover_active`); the closing's own release turn is not charged to the room loop; a cue glance
    skips a box perception placed on another storey, outside the map or on a refused candidate; a
    detector whose vocabulary, model or configuration changed under the evaluation ends the run
    instead of entering the transport back-off.
  - *Exploration fallback*: a blind-radius demotion is sticky (the goal is retired and the goal in
    force cleared -- the step toward the next exit promoted it back and the fallback turned between
    two exits for ever, 0.25 m a step so no watchdog fired); the relocation target is kept until
    reached (the farthest cell from a moving agent flips between the two ends of a hall); routes are
    planned with every seen staircase written occupied (Pomaria 2026-10-04 walked down a flight it had
    not chosen behind a floor-wide frontier); the per-step record names this action's rung.
  - *After the 5x3 benchmark (same evening; not in its numbers)*: the closing releases a lock whose
    approach went nowhere -- `approach_stall_actions` (20) CLOSE actions inside `approach_stall_m`
    (0.20 m) -- with a rejection (`stall_releases`; Leonardo pushed into an unseen obstacle for 160
    actions and raised the closing bound); the search's blocked clock is reset by 60 % of a forward
    step, not a collision slide (Leonardo wedged for 350 actions); an opening released BLOCKED,
    UNREACHABLE or TRANSIT_TIMEOUT is retired for the storey instead of re-chosen by the escape hatch.
  - *Stairs*: a storey first reached by a ground-truth traversal is pinned to the connector's navmesh
    height (`storey_height_pinned`; the atlas had measured Hanson's upper storey 0.17 m high from
    poses on the eased last treads, so no staircase seen up there ever became a portal and the map's
    floor-clear mask never fired); `storey_position` uses the atlas's match tolerance; the fallback
    rule's "destination already searched out" verdict re-checks in 10 actions instead of hiding the
    stairs node from the oracle for 100; `_reapproach` re-snaps the entry by geometry alone; no arrival
    cooldown on the source portal in ground-truth mode (the grace and the rooms left hold the way back).
- **A strong room label needs a distinctive kind of object** (`RoomLabelSettings.distinctive_required`,
  `room_priors.GENERIC_OBJECTS`): a cabinet and a potted plant no longer make a strong "living_room".
- **The oracle's STEP 2 is a structured verdict** (`search_node_oracle.parse_home_here`, `home_here` in the
  reply schema and the worked example; `SearchNodeOracle(unexplored_elsewhere=)`, `LoopSettings.
  unexplored_elsewhere` 0.10, `NodeOracleResult.capped` / `home_here`): with `home_here="elsewhere"`
  every unexplored node is read at 0.10 exactly instead of the 0.25 floor. The Ranchester cross-floor
  couch episode (`runs/zson-couch-crossfloor-3x-20261005`) had the 3B model say "living rooms are
  downstairs" sixteen times and 25 on every upstairs gap sixteen times, and spent its 500 actions on
  thirteen peeks with the stairs (0.60) last in every order -- standing 0.9 m from them at action 171.
- **Target closing, after the second Hanson fly of 2026-10-05**: the footing sweep is cut short the
  action a path exists (the sweep's 30 degrees had stood for the whole approach); the inspection
  pitch follows the target's measured height, not its label (a plant in a tall planter is at eye
  level); a fresh, aligned, in-range sighting STOPs at whatever pitch it came at (the toilet
  projected at 60 degrees where its height predicted 30: 20 actions of LOOK_UP/LOOK_DOWN); a fresh
  unlocked candidate within one turn of the centre is stepped toward rather than centred (the
  centring turn dropped a far chair four cycles running); and the legacy target evidence refuses
  what the takeover refuses (`refuses_far_candidate`; nine actions walked toward a just-released
  chair, and the move restarted the warm-up).
- **Labels are read before the nodes are chosen** (`ObservedSceneGraph.refresh_labels`, called
  by the loop's `_reason` before the exclusions and the oracle): a room that became a strong
  bedroom on the loop point's own re-classification was shown to the oracle, valued 0.10 and
  kept in the order for one more loop point (Hanson action 150).
- **Room ids survive growth** (`RoomRegistry(containment_threshold=0.6)`): a pair under the IoU
  threshold matches when 60 % of the smaller mask lies in the larger; IoU matches are consumed
  first. One Hanson bedroom was R11 -> R14 -> R16 while the agent stood in it, and each number
  lost the record of having been entered.
- The exploration fallback's retired-frontier rung no longer retries a goal under the agent's
  feet (the converter has no action for it: an idle turn per action).
- The oracle's per-room `frontier=` counts, the openings and the fallback's exits are read off
  the sight ledger's overlay; `test_opening_nodes` turns the ledger off in its fixture because its
  two-room world's "patch by the door" is exactly the pocket the ledger settles.
### Added (2026-10-05, earlier the same day)
- **Informative glances along a route** (`objnav_benchmark_runtime/methods/path_glances.py`,
  `core/planning/exploration/view_gain.py`, 2026-10-05): the route in force is scored every few
  actions for the one point where a look to the left, to the right or all round would reveal the
  most unknown floor the walk itself will not (optimistic rays through the unknown, stopped by
  walls, the forward cones along the route subtracted); if the gain clears 3 m2 and 0.5 m2 per
  action, the look is performed when the agent gets there -- one in-place turn per action, a
  suspended phase like the warm-up (no loop charge, clocks paused, the route's watchdogs told the
  pause). A spot where a full rotation already stood is never glanced from; a target sighting
  or a stair traversal aborts a glance. `GlanceSettings` on the policy, `glances` in the method
  configuration, `episode_info` and the per-step `search` record, a HUD line.
- **An uncertainty floor on unexplored nodes** (`search_node_oracle.SearchNodeOracle(unexplored_floor=)`,
  `LoopSettings.unexplored_floor`, default 0.25): a never-entered room still `unknown`, or an
  opening nothing was glimpsed through, is read at no less than the floor whatever the model
  wrote (`floored` in the result and the estimate events). The room line says `entered=no`; the
  prompt gains rules 2b (unexplored places owe a look; never "too small") and 2c (a home-type
  object in a room of another type means the map merged two rooms); the worked example no
  longer writes a small unknown room off. Hanson 2026-10-04 left its storey at action 27 with
  two never-entered rooms and six doorways valued at 0-1% by the 3B model.
- **Door cuts snapped to the choke** (`core/mapping/topology/room_watershed.py`
  `WatershedRoomParams.door_snap_reach_m`, on in the ObjectNav scene graph at 0.9 m): a detected
  door is moved onto the nearest medial-axis constriction whose removal parts the skeleton into
  two substantial pieces, and the disk carved there is sized to the passage; basins on different
  sides of the carved mask never merge (`merge_basins_by_dynamics(basin_sides=)`), basins on one
  side still may; a door with no choke in reach keeps the plain disk. The Hanson bedroom (R0/R6)
  is one room again, R7 and the corridor stay apart; the other fourteen recorded floors move by
  at most one room.
- **Furniture belongs to the room around it** (`ObservedSceneGraph.object_room` / `room_near`):
  a landmark standing on occupied cells is credited to the nearest room floor within its
  footprint radius plus 0.6 m (the bed, the desks and the wardrobe of Hanson's bedroom were
  `room: null` and the room was a "living room?" from its chair).
- **Home objects** (`room_priors.HOME_OBJECTS`, `ruled_out`): a room holding an object of the
  kind the target lives with (a sink or a shower, for a toilet) is never ruled out by type.
### Changed
- **The type prior rules out STRONG labels only**: a weak label (one kind of object) never
  excludes a room; the oracle sees it as `type=bedroom?` and values it (`weak_type_kept`, once
  per room; `LoopSettings.weak_type_max_openings` is retained but no longer consulted). The
  relabel-ends-the-visit rule follows the same test.
- **The peek's approach bound follows the route the planner adopted**
  (`peek_approach_extended` events): Hanson's peek of action 166 was sized on a 3.5 m geodesic
  and cancelled 5.8 m short on a 12.5 m route still being followed.
- **Distances are measured the way A\* flies them**: `accessible_frontiers(preferred_cost=)` and
  `build_instance(preferred_cost=)` take the cost grid at the planner's preferred standoff and
  measure every reachable distance there, falling back to the body-radius graph where only a
  squeeze connects; `RPTSearchPolicy.preferred_cost` supplies it. Reachability is unchanged.
- **The room visit is a scan** (`objnav_benchmark_runtime/methods/room_search_loop.py`,
  `LoopSettings.visit="scan"`, 2026-10-04): transit to the room's vantage point -- the reachable
  interior cell of greatest clearance (`room_vantage.py`) -- turn a full circle on measured yaw,
  and leave; the room is finished for the episode. The **scan ledger** (`room_scans.py`) remembers
  where every rotation stood, not which room id it was credited to, and finishes a room when a scan
  point lies inside it or a scan saw more than half of it through observed free space; the verdict
  is sticky per `(floor, pid)`. Finished rooms and rooms whose identified type cannot hold the
  target (`room_priors.py`; a confirmed target object in the room outranks the prior) are **not
  nodes**: not shown to the oracle, not solved, not entered (`excluded` events/estimates/HUD).
  A clue mid-visit is deferred to the loop point; a relabel ends the visit only when the new type
  rules the room out. The former bounded sweep is kept as `visit="sweep"`.
- **The warm-up is one full rotation** (12 x 30 deg), recorded as the storey's first scan so the
  spawn room is finished before any room is chosen; repeated on every storey first entered.
- **Stairs are never a room**: seen stair footprints are removed from the free mask before the
  watershed (`ObservedSceneGraph.update(exclude=policy.room_exclusion(world))`).
- **Object identity is a per-frame vote** (`core/mapping/objects/landmarks.py` `class_votes=True`,
  wired in `floor_context.py`/`perception_cycle.py`): detections associate by position (dedupe
  radius or footprint-disc IoU >= 0.15, `perception.footprint_radius_m`) with a 0.35 m height
  check; each landmark tallies `votes` per class, one per frame; its class is the plurality,
  relabels are recorded, confirmation needs a clear plurality; target evidence is judged on the
  map's class (`contradicted_by_map`), so a `sofa` box on a five-times-confirmed bed is outvoted.
- Target closing on a big object at close range: a fresh box spanning the image centre column is
  aligned whatever its centre says; the terminal range is the near edge of the measured surface
  (`range_near_m`); an exhausted inspection after an in-range sighting STOPs
  (`stop_on_exhausted_inspection`) instead of raising. The border gate is shared with the legacy
  target-evidence path (`perception.clipped_box`). The rejection memory of a released candidate
  does not block a view from within `rejection_min_range_m` (2 m), and the lock's association
  radius grows with range (`association_range_gain`).
- Under `scan`, the staircase the agent arrived by is withheld while the storey still has a room
  that is neither finished nor ruled out (`way_back_held` loop events); a warm-up moved more than
  0.5 m mid-rotation starts its circle again.
- **Target closing starts only on a depth-projected candidate** and **tracks low once active**
  (`target_closing.track_confidence`, 0.30): a confident box the depth sensor cannot place no longer
  buys a twelve-step verification ("waiting for valid target depth", Ranchester attempts 4-5), and a
  re-sighting of the active candidate on its anchor counts from the lower threshold -- the agent's own
  centring turn dropped the Ranchester couch from 0.68 to 0.42 and reset the consecutive-frame run on
  every cycle (24 frames in view, two releases). `weak_resightings` in the diagnostics.
- **The exploration fallback ranks exits first** (`exploration_fallback.py`, `FallbackSettings.exits_first`):
  a frontier that is an object's shadow (beside a confirmed landmark's footprint, no longer than it
  could cast; `object_shadow`) or belongs to a room the loop ruled out by type waits in a new
  `frontier_demoted` rung behind every genuine opening and the stairs. The Ranchester recording's
  actions 61-102 chased ten strips of unknown between the beds while the passage to the stairs stood
  two metres away. `last_demoted` in the fallback record; `RoomSearchLoop.excluded` is public.
- **The fallback commits to its goal** (`goal_switch_gain` 2.0, `goal_match_m` 0.6): the frontier in
  force is driven until it is gone or refused unless another of its rung is worth twice as much; the
  greedy per-action order, re-ranked by the agent's own turning, produced a 180-degree turn toward one
  goal and a 210-degree turn back (Ranchester attempt 6, actions 42-55). Frontiers inside the camera's
  blind radius (`near_blind_m` 1.2) -- the floor under the agent that no step toward resolves -- are
  demoted with the shadows. The per-step record carries `method.fallback` (stage, goal, demotions).
- Per-storey loop events/stats/exclusions in `building.floor_contexts`; HUD shows the visit budget,
  scan phase and the rooms that are not nodes. Progress entry 019.
- **Openings are nodes** (`objnav_benchmark_runtime/methods/opening_nodes.py`, `LoopSettings.openings`,
  2026-10-04): every genuine exit of the mapped floor -- a frontier that is not an object's shadow,
  not under the agent, at least 0.8 m wide after merging within 1.5 m, not at the foot of a seen
  staircase -- is offered to the oracle and to RPT* beside the rooms and the stairs, with a sticky
  building-wide id (`O<n>`), the room it opens from, `doorway`/`gap` by a confirmed door frame, and
  the objects glimpsed through it; priced at a peek's 4 actions (`build_instance(service_s=...)`).
  Its visit is a **peek**: walk to the threshold (bound: the distance's worth of steps x 1.6 + 6, at
  least 20; one re-aim when the cell goes occupied), face the unknown, one look to each side, and the
  opening is retired for the storey (`peek_look`/`peek_complete`/`peek_abandoned`/`peek_reaimed`
  events, `OpeningRegistry` in the episode record). The heading into the unknown is from the known
  floor around the frontier cell toward the unknown; a glimpse is a landmark 0.3-3 m beyond the
  threshold inside a 35-degree cone (attempt 8 read the stair passage as a bedroom door from a
  hallway bed). A new opening while nothing is in force buys an oracle call at most every 6 actions.
  The oracle's prompt explains OPENING nodes and
  shows one in its worked example. Ranchester attempt 7, action 75: a toilet glimpsed through a door
  named the hallway a bathroom and every door off it was demoted with it.
- **A weak type label does not rule out a room with more than `weak_type_max_openings` (1) openings**
  (`weak_type_kept` events): the one object that named it may belong to the room behind a door. Openings
  count as "rooms left" for the way-back hold.
- **Room numbers are unique across the building**: a storey first entered numbers its rooms after the
  highest pid any storey has used (`RoomRegistry(first_pid=...)`, `FloorContextBank.new`), so `R0`
  names one room in a recording. Stair node ids are now a band (100 000-199 999); openings start at
  200 000. HUD: `OPENINGS (nodes)` section, `O<n>` markers with the heading into the unknown on the
  active floor map, the peek in force. Progress entry 020.
- **A confirmed landmark of the target's class is a node** (`opening_nodes.landmark_openings`,
  `OpeningSettings.landmarks_enabled` / `landmark_prob` 0.85 / `landmark_standoff_m` 1.5; `T<n>` on the
  HUD): offered to RPT* at a fixed probability without an oracle call, at a standoff on the agent's side,
  visited by the same peek (walk facing it, one look to each side); inspected landmarks and spots in the
  takeover's rejection memory are not offered again (`TargetClosing.rejected_near`). Ranchester attempt
  9 walked forty actions away from a confirmed `sofa` 3.8 m from the stair foot.
- **A locked target whose terminal inspection never saw it is released, not an episode error**
  (`target_closing.release_on_failed_inspection`, default on; `failed_inspection_radius_factor` 2.0): the
  anchor joins the rejection memory with twice the radius, the release action is one turn in place, and
  the search resumes. The Ranchester same-storey run ended as `agent_error` on a sofa-from-four-metres
  that was nothing from one, the real couch 2.85 m away. The release comes as soon as every inspection
  view has been tried once with nothing seen (not after the 24-action budget), and at once when the
  map's class vote at the anchor outvotes the lock (`map_releases`; the sofa from four metres is the bed
  from two). The rejection memory now records a radius per spot (`rejected[].radius_m`);
  `inspection_releases`, `map_releases` and `last_release` in the diagnostics.

### Changed
- `doorway_peek.enabled` and the room-coverage floor gate (`doorway_peek.gate_floor_departure`)
  default to **off**: a peek costs a scan and finishes nothing, and the gate held the Ranchester
  descent 14 times. Both remain available as ablations.
- Development runs use `qwen2.5:3b-instruct` as the reasoning model; the 14B stays the benchmark model.
- The Ranchester couch-downstairs development episode (`runs/zson-couch-downstairs-20261004-scan`)
  now succeeds in **150 actions (SR 1, SPL 0.71)** where the 2026-10-04 morning runs failed at 28 and
  500; the seven attempts of the day (185 / 215 / 167 / 150 actions on the successful ones) are kept
  under `attempt-*` with the diff each ran.
### Added (2026-10-01, HM3D: shared navigation core)
- HM3D ObjectNav now runs the same navigation core as Gibson. `feat/objnav-habitat-gibson-nadav`
  is merged into `feat/objnav-habitat-hm3d-nadav`: `objnav_benchmark_runtime/methods/` (RPT*,
  room-search loop, frontier sweep, exploration fallback, multi-floor coordinator with
  ground-truth stairs, YOLO-World default detector), `habitat/` and the shared
  evaluation/recording path are one core; `gibson/` and `hm3d/` are the two task modules
  that own their environments, loaders and protocols. The core imports neither task
  (`tests/test_runtime_boundaries.py`). `hm3d.run` gains `--explorer frontier|falcon`,
  `--detector-backend` and `--detector-timeout-s`; `HM3DEnv` attaches the navmesh storeys
  and stair connectors to the episode like `GibsonEnv` does. See
  `docs/progress/entries/013-*.md`.
- `hm3d/README.md` rewritten as the HM3D execution guide: environments, dataset and weight
  downloads, one recorded episode with the live dashboard, and the scene-per-process
  campaign, sharding and freeze-then-evaluate workflow for the full published split.
### Changed (2026-10-01, HM3D merge)
- `habitat/simulator.py` is the union of both branches: explicit `height_m` and the
  `navmesh="published"|"agent"` protocol choice with `navmesh_provenance()` (HM3D), plus
  `scene_structure()`, `last_collision` and RGB/depth extrinsic checks (Gibson). Gibson's
  protocol now states `agent_height_m = 0.88` explicitly instead of reusing the camera height.
- `gibson/stair_connectors.py` moved to `habitat/stair_connectors.py` (it reads any habitat-sim
  navmesh); the old path re-exports it.
- Target-closing recording HUD shows active/confirmed lock, phase, observed target range,
  elapsed time and FPS; per-frame LLM/classifier/RPT/A* counters permit freeze verification,
  and normally completed videos end with SR/SPL/DTG/runtime metrics.
- Single live toilet verification retained with failure evidence: takeover/lock worked, but
  route steering and reacquisition lost the target before LOOK_DOWN or STOP. No retry was run;
  the aborted run has a separately labeled diagnostic report, not a completed benchmark row.
- Irreversible ObjectNav target takeover: configurable confidence, two consecutive depth-consistent
  frames, bbox yaw centering, close low-target LOOK_DOWN and fresh verified explicit STOP; room/LLM/RPT*
  exploration cannot resume after takeover, including through failures or detection loss.
- Geometry-only local Habitat NavMesh projection for observed target standoff goals, followed by
  collision-qualified A*; availability is declared in run provenance, with no evaluator target/DTG access.
- Development evaluation now prints finalized video paths and SR/SPL/DTG/runtime/FPS immediately
  after each episode, supports timed inspection pauses, and saves completion and throughput summaries.
- Recorded HUD now includes evaluator-only DTG, scene, current room type/confidence and a
  coordinate-based active frontier ID, without exposing goal distance to the navigation policy.
- ObjectNav room-coverage floor gate: every observed room, including the spawn room, needs a
  completed interior peek and measured 180-degree scan before normal or fallback stair selection;
  new rooms revoke an approach, failures remain pending, and accidental early stair entry retreats safely.
- Regression coverage for real-converter interior scans, floor-local visit identity, failed and
  split rooms, stair-choice bypasses, adaptive floor-map scaling and persistent room partitions.
- **Stairs are known when they are SEEN** (`objnav_benchmark_runtime/methods/stair_sightings.py`):
  a perfect stair detector replaces the connector list the policy used to be handed at reset. Each
  navmesh connector's surface sample is projected into the current frame (`camera_geometry
  .project_to_image`) and depth-tested against the depth image; six visible samples make a
  sighting with a bounding box, drawn orange on the RGB overlay and carried in `detections` with
  `source: ground_truth`, `distance_m`, `visible_points`. Only a seen staircase becomes a portal
  (`portal_placed` with `seen_step`, `seen_from_m` and the sightings' `footprint`), an RPT* node
  or the explanation of a height departure -- a raised bathroom floor is a bathroom floor, and a
  staircase never in frame does not exist to the search (`height_departure_ignored: connector N
  underfoot has never been seen`). `GroundTruthStairs.touching/nearest(..., among=seen)`,
  `StairSightings` (first/last step, sightings, nearest distance, deduplicated footprint),
  `building.stairs_seen` in the diagnostics.
- Stair connectors rebuilt (`gibson/stair_connectors.py`): one connector per pair of ADJACENT
  storeys (a stairwell serving three storeys is two connectors, not one 5.4 m flight through the
  middle storey); the polyline is the flight's CENTRELINE -- one vertex per 0.2 m height band, the
  band component that touches the previous band's -- with anchors on the flight's own line and a
  verified flat EXIT point off each end (`bottom_exit_xyz`/`top_exit_xyz`); every leg is
  stride-checked with `PathFinder.try_step` (the simulator's `move_forward`) and repaired through
  the pathfinder only where a stride fails, legs under 0.3 m merged; `surface_xyz` (the detector's
  sample), `walkability` counts and `traversable` (False when the flight joins two navmesh islands:
  Hanson, Leonardo, Marstons, Shelbyville -- seen, never climbed) ride with each connector.
- `tests/rollout_stairs_navmesh.py`: every connector of every scene climbed both ways on the real
  navmesh by the real `GroundTruthTraversal` and `DiscreteActionConverter`, MOVE_FORWARD applied
  with `try_step` and sliding. 29/29 traversable climbs complete, 25 with zero blocked steps;
  island-split connectors reported SKIP. Exit 1 on any failed climb, so a campaign can gate on it.
- `tests/test_stair_sightings.py`: the detector (in frame, hidden behind a nearer wall, half
  hidden, behind the agent, beyond range, depth slack, polyline stand-in) and the sightings memory.
- Floor maps (`floor_panels.py`): every SEEN staircase drawn where its treads were seen (footprint
  dots, entry square, `S<n> up/down`); node labels only for the first six of the RPT* order plus the
  node in force and the next, every other room a dot; labels placed on the first free side and
  dropped rather than written over another; dark outline under every label; object dots smaller.
  Panel title counts the stairs.
### Changed
- Room peeks are now strictly one attempt per episode-local room: initial classifications,
  completed scans and consumed attempts bypass both repeat peeks and impossible floor-exit
  requirements. No cancellation refunds an attempt; overlap preserves the allowance through
  segmentation renumbering/splits/expansion without claiming a scan completed.
- Seen, current-floor stair connectors are excluded from peek viewpoints and peek-only A*
  routes; committed stair/atlas transitions cannot be interrupted by peeks. A coverage-vetoed
  room transit is released instead of repeatedly replanning into the same left/right loop.
- ObjectNav enables a bounded 1-degree path-heading hysteresis margin to prefer safe forward
  progress over reversing-turn chatter; generic converter behavior remains unchanged by default.
  1,206 core/runtime tests pass; headless Ranchester ascent/descent complete without blocked steps.
- Same-spawn Sofa verification (one Ranchester episode, unchanged policy/source) reached 500
  actions with SR=0/SPL=0/DTG=12.5852 m. Stairs were seen at 135, but 62 room-coverage vetoes
  kept the agent upstairs; descent, lower-floor target closing and STOP were not exercised.
  Full HUD video, exact-spawn proof and official metrics retained in `runs/sofa_same_spawn_20261001`.
- Target closing now retains its confirmed 3D target and NavMesh standoff path through occlusion;
  A* owns transit heading, fresh observations refine the goal, and stationary terminal inspection
  at 1.0 m (0.05 m confirmation tolerance) owns pitch/bbox servo and explicit STOP.
- Expected closing failures are recordable method errors rather than infrastructure aborts;
  the HUD distinguishes persistent lock from live visibility. Added occlusion/refinement/corner
  regressions; 645 runtime and converter tests pass.
- One recorded Ranchester toilet check verified unchanged LLM/RPT* counters, path continuation
  across three occluded decisions, LOOK_DOWN and policy STOP at 0.874 m, with no agent error.
  Official SR=0/SPL=0/DTG=10.8924 m remain: this manifest scores downstairs reference-floor targets,
  not the upstairs detection. This is closing-control verification, not benchmark success.
- Doorway peeks go a further metre inside the room with stopping-tolerance compensation and a
  deepest-safe-viewpoint fallback for small rooms; simply crossing a doorway no longer counts as scanned.
- Floor maps now show thin logical room outlines, short IDs and minimal route/robot/stair markers;
  object dots, probability text, frontier diamonds and visit-order chains stay off the map. A shared
  observed-house viewport and scale bars replace the fixed 32 m crop; room partitions persist per floor
  and are saved separately as numeric `room_partitions.npz`. Camera, depth and search column are unchanged.
- Development check of the room-peek gate: four recorded seed-17 episodes across Ranchester and
  Pomaria completed with no runtime failures, but **0/4 navigation successes**. Thirty measured
  half-turn scans completed; room repartitioning caused 37 peek cancellations and repeated scans,
  leaving rooms pending and preventing every floor change. The prior successful Ranchester couch
  episode regressed to the action limit. The 368 passing runtime tests validate behavior, not a
  performance gain; visit continuity across repartitioning remains an unresolved follow-up.
- `GroundTruthTraversal` never skips a vertex: a tight aim blocked twice sends the agent BACK to the
  last vertex it reached (`traversal_recovery`, `recoveries` in the diagnostics) and the route is
  resumed from there; the route handed to the converter ends at the next bend (`BEND_RAD`, 35
  degrees), so a landing corner is walked to, not cut by the half-metre lookahead; a vertex reached
  from off to one side is led back THROUGH (`VIA_M`); reached means 0.26 m in plan view and 0.45 m
  in height (`REACHED_XY_M`/`REACHED_Z_M`), replacing the 0.35 m sphere; the exit is the
  connector's verified exit point, then a stub swung 45/90 degrees either side and never behind.
  Pomaria's two flights: 40 actions each way, 0 blocked, in the probe and in the recorded run.
- `MultiFloorParams.arrival_grace_actions` (40): the staircase the agent arrived by is withheld from
  the loop's node list and from the fallback rule (`MultiFloorSearch.way_back_held`, `way_back_held`
  event) until the new storey has been looked at -- the node oracle chose the way back on the
  arrival action three times in one recording. Any other staircase is offered at once.
- `floor_decision`/`stair_nodes` see only seen, traversable connectors; `test_room_search_loop
  .stair_policy` marks the fixture's staircase seen.
### Earlier in this cycle
- ObjectNav node oracle (`core/mapping/topology/search_node_oracle.py`): the room LLM is now
  asked ONE question per loop point over every node the search could go to next -- each room
  (type or `unknown`, size, frontier left, time searched and how long ago, objects seen, whether
  the robot stands in it) and each staircase (up/down, whether the other storey was visited and
  what was found there, whether the robot arrived by it) -- for the probability, in percent,
  that going there NEXT finds the target, plus `elsewhere`. The system prompt states the
  judgement in full: an irrelevant type or a fully observed room reads ~0; a room searched long
  and recently reads low; an `unknown` room is an exploration node valued by size, frontier and
  the room types still missing; a hallway with frontier by what it leads to; a staircase by what
  this storey turned out to be; and never by distance, which RPT* charges. Code keeps only the
  contract (parse, drop invented ids, omitted nodes get a small share, rescale, clamp below 1,
  refuse a reply with no usable node, reuse a byte-identical prompt's reply). `RepairingNodeOracle`
  retries one malformed reply. `LLMClient` gains a REASONING route -- `LLM_REASONING_MODEL`
  (default `qwen2.5:14b-instruct`, 9 GB at Q4, chosen to fit beside Habitat in 30 GB of RAM),
  `LLM_REASONING_TIMEOUT_S` (600 s), `LLM_REASONING_MAX_TOKENS`, `LLM_REASONING_NUM_CTX` (8192,
  requested explicitly because Ollama truncates an outgrown window from the front) -- for this
  one call; the default model keeps the cheap, frequent ones; `VerifiedLLMClient` checks both are
  provisioned. Runtime is not the constraint: the search waits for the answer. Measured on the
  laptop's CPU container: ~25-30 s per call once loaded, ~75 s cold. `tests/probe_node_oracle.py`
  runs three scenarios against the live model; the prompt was tightened on its first run (size
  judged against the target's usual room -- a toilet's room is small; a known kitchen elsewhere
  outweighs a room that might be one; an example that cannot be parroted).
- Staircases as nodes of the RPT* order (`methods/stair_nodes.py`, `room_costs.build_instance
  (leaves=...)`): every reachable, uncooled portal of the floor is a node beside the rooms, with
  an id above every room pid, valued by the oracle, and sitting at the foot of its flight with a
  LEAF of the flight's length plus `MultiFloorParams.floor_change_cost_m` (8 m) charged on every
  arc into AND out of it -- so going upstairs puts every room down here that much further away
  and the expected-time-to-find objective sees the cost of changing floors. When the order puts a
  staircase first the loop commits it to the building coordinator (`MultiFloorSearch.commit`),
  the climb begins at the foot and the node's turn ends `traversed` (`ObjectSearchSupervisor
  .finish`); the way back down is a node like any other, marked `arrived_by`, with no cooldown.
- The clue rule: one confirmed object names a room (`RoomLabelSettings` 1/1/1; a single class is
  a *weak* label, shown to the oracle as `type=kitchen?`), the classifier is re-asked for the
  room in force or in transit the action a NEW KIND of object lands in it (a verdict for a set of
  kinds is reused across the watershed's re-partitions; 22 → 5 calls in the 60-action trace),
  and a changed name ends the room's turn with the new neutral verdict `RECLASSIFIED`
  (`room_reclassified` exit, in SEARCH or in TRANSIT before entering; not cooled, no attempt
  charged) so the estimate and the order are redone over the new fact.
- Committed ground-truth traversal (`methods/ground_truth_traversal.py`, `MultiFloorParams
  .commit_failures/commit_stall_actions/confirm_actions`, `FloorAtlas.settle`): a blocked step
  tightens the following to a straight line at the next vertex before any vertex is skipped;
  turning back takes 12 blocked steps or 30 stalled actions; the transition budget no longer
  turns the climb round; a storey the agent stands on at the far anchor is confirmed outright
  after 24 actions when the plateau test cannot fire (`destination_forced`); a failed approach
  is re-snapped and retried before the connector is deferred.
- Dashboard search column (`objnav_benchmark_runtime/search_panel.py`, `visualization
  .search_snapshot`, `floor_panels.render(search=...)`): the target and the oracle's split
  between the nodes and elsewhere, the RPT* visit order with its head (`S0` a staircase), the
  node in force or in transit, every room's type/probability/reason/frontier/time/objects, every
  staircase's probability, direction, climb cost and storey beyond, the objects by room and the
  loop's last events; rooms and stair nodes named and valued on the active floor map with the
  order drawn through them and the next node ringed. The same block rides every `steps.jsonl`
  row and `live.json`; `room_search_loop.estimate_events` records every oracle round.
- Tests: `core/mapping/topology/tests/test_search_node_oracle.py`, the leaf test in
  `test_room_costs.py`, `objnav_benchmark_runtime/tests/test_search_panel.py`, plus the
  reclassify, stair-node, fallback-rule, commitment and forced-confirmation regressions in the
  existing modules. `tests/trace_room_search_loop.py` drives the policy through the real
  `DiscreteActionConverter` on a point agent.
### Changed
- **No step budget decides a floor change in ground-truth mode.** `floor_search_actions` is the
  observed mode's allowance only; `floor_decision.py` is the exploration fallback's last resort
  (`rule: fallback`), asked when the floor is exhausted and the loop has nothing to offer, and it
  charges the climb through the same `stair_cost_m` as the RPT* leaf.
- `ObjectSearchSupervisor`: `RECLASSIFIED` (neutral) and `TRAVERSED` (productive) verdicts,
  `update(room_reclassified=...)`, `finish(verdict, note, now)`, `is_cooling()`; `NEUTRAL`
  verdicts charge no attempt.
- `ObservedSceneGraph.reason(..., extra_nodes, context, here_xy)`; `graph.probs` are the oracle's
  find-probabilities unnormalised (they sum with `stair_probs` to `p_present`); `last_inside`
  tracks when the agent was last in each room.
- `RoomTypeClassifier.cached(classes)` reads a held verdict without a call.
- `search_oracle.py` (the 3B-era per-room affinity oracle) is no longer used by the runtime; it
  remains for the ROS scene-graph stack.
### Earlier in this cycle
- ObjectNav exploration fallback (`objnav_benchmark_runtime/methods/exploration_fallback.py`):
  every failure of the decision pipeline -- an A* with no path, an RPT* instance that cannot
  be built, a room LLM that times out or answers badly, a bug in the loop -- now ends in a
  command that moves: the nearest reachable frontier anywhere on the floor (leaving the
  current room if that is where it is), then the stairs, then a retired frontier, then a
  relocation to the farthest reachable known cell; a hold only when the agent stands off the
  passable map. `RPTSearchPolicy.plan` wraps the decision and routes any exception there, so
  a model failure is no longer an agent error that ends the episode. Failed model services
  (room LLM, detector) get a doubling back-off (25 → 200 actions) during which they are not
  asked; failures are counted, kept with type/message/origin under `exploration_fallback` in
  the episode record, and logged once per type. See `docs/progress/entries/012-*.md`.
- Ground-truth stairs for the multi-story policy (`MultiFloorParams.stair_source`, default
  `ground_truth`; `observed` keeps the RGB-D discovery for comparison). The evaluator reads
  each scene's navmesh once (`gibson/stair_connectors.py`): area-supported storey heights and
  every off-level surface cluster as a connector with floor anchors and the navmesh shortest
  path, attached to the episode metadata (no goal, distance or target floor). The policy
  (`methods/stair_ground_truth.py`, `floor_decision.py`, `ground_truth_traversal.py`) makes
  each connector a portal, decides explicitly when to change floors -- eligible, reachable
  on the observed map now, destination unvisited or with frontier left, nearest first; up or
  down falls out of the connector -- with one recorded `floor_decision` event per distinct
  verdict, and walks the connector's own polyline with a FOLLOW command on every action.
  `configuration()["ground_truth_stairs"]` declares it. Verified on the real Ranchester
  navmesh: two storeys, one 5.5 m connector, the flight the recordings climbed.
- `objnav_benchmark_runtime/tests/{test_stair_ground_truth,test_ground_truth_floor_transitions,test_exploration_fallback}.py`
  (38 tests) plus the metadata/tripwire and YOLO-default regressions.
- `objnav_benchmark_runtime/QUICKSTART.md`: the end-to-end runbook for the Gibson ObjectNav
  simulation -- the three environments, the licensed dataset and weight downloads, then the
  services, frozen episodes, preflight, smoke run and campaign in strict order, with the
  refusals a newcomer meets and their fixes. Linked from the root README and the runtime and
  Gibson READMEs; its service and preflight steps were executed as written.
- `provision_grounded_vlm --model none --include-yolo` provisions YOLO-World X-v2 + CLIP alone
  (~0.5 GB) instead of forcing a Grounding DINO / BLIP-2 download for the YOLO-only default.

### Changed
- YOLO-World only is the default object-detection pipeline everywhere: `--detector-backend`
  defaults to `yolo_world` in `gibson.run`, `run_development`, `distinct_buildings`,
  `compare_explorers` and `stair_diagnostic` (`detector_options.DEFAULT_DETECTOR_BACKEND`),
  and `scene_graph/serve/configs/gibson_perception.json` selects `yolo_world`. Grounding
  DINO / BLIP-2 (`grounded_vlm`, `hybrid`) remain selectable but disabled until their stair
  verification is revisited; stairs no longer depend on the detector at all.
- In ground-truth stair mode the building coordinator runs no look-down inspections and
  applies no depth veto on the committed route; an unplanned height departure starts a
  transition only within 0.75 m of a known connector's polyline (`near_connector_m`) and
  beyond `departure_m`, and any other departure is logged once (`height_departure_ignored`)
  and left to the atlas. A spent retreat lets the atlas settle where the agent stands instead
  of halting the episode (`SAFE_HALT` is observed-mode only).
- `RoomSearchLoop._fallback` delegates to the exploration fallback; `_reason` turns a room-LLM
  failure into a recorded, backed-off `RoomReasoningUnavailable` instead of raising out of
  `plan`. `LoopSettings`/`RPTSettings` unchanged; `FallbackSettings` added to
  `configuration()`.

### Fixed
- The multi-story policy no longer starts a "stair traversal" on a raised bathroom floor or on
  a 12 cm height wobble at a real stair head and then turns in place "acquiring tread support"
  -- 120 + 12 actions of the 500 in `runs/room_search_loop_ranchester_20260928T093541Z_transitfix`.
  See LESSONS.
- The room-search loop's "nothing to work in" answer is no longer an empty hold that the
  headless agent spends as one idle `TURN_LEFT` per action until the step budget runs out.

### Added (earlier, same branch)
- ObjectNav frontier explorer now runs the room-search loop as specified
  (`objnav_benchmark_runtime/methods/room_search_loop.py`): bounded local exploration of the
  room in force -- `LoopSettings.local_steps` (10) actions or no frontier left -- with every
  in-room route planned on a copy of the map in which the other rooms are blocked, so no
  route crosses a door; then, on the same action the room ends, re-classify every room from
  every object so far, re-estimate P(target) per room, re-solve the RPT* order (local budget
  folded in as service time) and transit to the *nearest frontier inside* the chosen room
  (centroid when it has none); arrival is the agent's cell inside the room mask and resets
  the counter. The room LLM runs at loop points only (plus once at the start and when a room
  the estimate has never seen appears while nothing is in force); scene-graph geometry
  refreshes every action (`graph_period_steps` default 10 -> 1). The episode record carries
  `room_search_loop` (per-room estimates with the distance the order charged, and one event
  per transit/arrival/release). See `docs/progress/entries/011-objnav-room-search-loop.md`.
- `ObjectSearchSupervisor`: caller-driven `arrived` (TRANSIT -> SEARCH by the caller's own
  mask test) and `budget_spent` (SEARCH -> `BUDGET_SPENT` by the caller's action count);
  `frontier_exhausted` now also ends a TRANSIT whose room the map filled in from outside
  (`exhausted_in_transit` stat). New params `cooldown_verdicts` (which verdicts cool a room),
  `repeat_verdicts` (which cooling rooms the every-room-cooling escape hatch may repeat) and
  `resolve_on_release` (re-ask the solver after every release). Defaults keep the flown
  behaviour; the loop leaves `BUDGET_SPENT` out of the cooldown and `EXHAUSTED`/`MAPPED` out
  of the hatch.
- `frontier_ranking.frontier_goals_by_room`: every room's ranked, reachable frontier goals in
  one extraction and one Dijkstra, clusters credited to rooms by the same majority vote
  `count_frontier_clusters` uses -- so a room's goals and its reported cluster count are one
  population. `room_costs.frontier_clusters` exposes cluster member cells for it.
- `SearchOracle` reuses the model's last reply when the prompt it would show is
  byte-identical, re-applying the code-side time/frontier factors (`OracleResult.reused`,
  `oracle_reuses` in the episode record): a loop point over an unchanged map costs no call.
- `ObservedSceneGraph.labels` (the pid+1 room label image) and `room_at()`.
- Two full recorded Ranchester episodes of the room-search loop from two start positions
  (`runs/room_search_loop_ranchester_20260928T091752Z`: 000000 toilet from (2.97, -11.49),
  000001 couch from (-1.03, -19.97)) -- the first Ranchester episodes ever to run to
  completion on this workstation; every earlier attempt aborted on the room LLM. Plus
  `objnav_benchmark_runtime/tests/timeline_room_search_loop.py`, which prints a recording's
  `steps.jsonl` as a collapsed phase/room/command timeline with the oracle rounds and the
  detections that reached the target track.

### Changed
- A frontier cluster's goal is now its member cell nearest the centroid (on the boundary),
  not the centroid itself: for the arc-shaped clusters a depth range produces the centroid
  lies inside seen floor, so every goal was "no longer informative" on the action it was
  adopted and the room read exhausted with clusters still showing (see LESSONS).
- `frontier_sweep.py` is reduced to goal generation, committed-goal lifetime and the optional
  look-around; `SweepSettings.room_scan_turns` defaults to 0 (the loop's termination rule is
  *N steps or nothing left*) and `supervisor_rounds` moved to `LoopSettings` (default 5, the
  longest legitimate same-action chain; an overrun is counted and logged, not silent).
  In-room goals, the room entry point and the transit re-aim share one admissibility filter.

### Fixed
- Room-search loop transit no longer idles at a route end that the re-segmenting map moved out
  of the target room: arrival now also counts on the room's boundary (within the sweep's mask
  slack -- the eroded watershed mask stops short of the very frontier the transit aims at), and
  a route that ends where the room is not re-aims once at the room's current nearest frontier
  or releases the room as unreachable (`entry_lost` stat and event). The first recorded
  Ranchester run (`runs/room_search_loop_ranchester_20260928T091752Z`, episode 000000) spent
  31 of 299 actions turning in place at exactly such a spot until the route memory's 30-action
  no-progress clock released the room. See LESSONS.
- ObjectNav frontier explorer no longer spins in place for the supervisor's 30/90 s clocks
  once a room is swept: the new `frontier_sweep.py` gives a swept room one bounded
  look-around (a full rotation by default) and then ends it at once through two new
  `ObjectSearchSupervisor` exits, `frontier_exhausted` (verdict `EXHAUSTED`, productive) and
  `route_failed` (`UNREACHABLE` without the plan-grace wait). A released room is replaced by
  the next transit on the same action instead of by a throwaway floor-wide route. The
  Ranchester recording `e7c4f2ad5402` spent 24 % of its 472 actions on these idle turns. See
  `docs/progress/entries/010-objnav-exploration-efficiency.md`.
- Stair (look-down) inspections no longer fire every cooldown while following a path once the
  floor allowance is spent: they now start only for a named reason (`floor_exhausted`,
  `stair_detection`, `periodic`), the latter two only at an action with no committed route,
  and unprompted ones at most every `periodic_inspection_actions` (100). The same recording
  ran ten sweeps (17 % of actions) plus 7 % of turns recovering the heading each left behind.
- `CommittedRoute.reusable` no longer overwrites the recorded clear reason with `unplanned`,
  so `route_replaced` (new in every adopted route's info) and the per-reason
  `cleared:<reason>` stats say why a route was really dropped.

### Changed
- Frontier goals are ranked by utility (`core/planning/exploration/frontier_ranking.py`):
  sub-linear cluster size over geodesic distance on the planner's passable graph, with facing
  as a discount. Clusters with no known-free path are dropped instead of proposed, and up to
  three ranked goals are tried per action, so a refused A* no longer costs an idle turn. A
  committed frontier goal is kept while it is passable, still borders unknown space and is
  near the room being swept -- a re-segmented room mask alone no longer replaces it every
  ten actions, and a boundary the camera has already resolved no longer keeps it.
  `in_room_frontier_goals` (size order) remains for pose-free callers and the scene-graph
  count; both now share `frontier_cluster_cells`.

### Fixed (earlier)
- `rooster_twist_control_adapter.py`'s `max_yaw_rate` recalibrated from a never-validated
  0.5 rad/s to 1.8 rad/s, derived from a logged manual flight's actual turn-rate behavior
  (~4x too low previously — any planner-requested yaw rate was executed much faster than
  intended). See LESSONS.md and `docs/progress/entries/007-rooster-velocity-controller.md`.
- Broken `torch` import in the project venv (`venv/`): removed an orphaned cu12 NVIDIA package
  cluster conflicting with `torch==2.11.0`'s actual cu13 requirement, and force-reinstalled two
  cu13 packages (`nvidia-cudnn-cu13`, `nvidia-nvshmem-cu13`) whose library files were missing
  despite being reported installed (corrupted/interrupted earlier install). See LESSONS.md.

### Added
- Explicit, default-off shared-GPU authorization for the detector and frozen
  Gibson development campaigns; records the permission in model/run identity
  without changing navigation, scoring, CUDA checks or the detector memory cap.
  See `docs/progress/entries/009-grounded-vlm-gibson-evaluation.md`.
- Selectable Gibson perception modes: YOLO only, Grounding DINO Base + BLIP-2
  FLAN-T5 XL, or all three. One JSON `backend` setting controls model loading;
  raw/verified frame evidence is recorded without changing depth, floor or STOP
  safeguards. Added pinned/checksummed offline weight provisioning, explicit
  local CLIP loading, bounded verification and forwarded detector timeouts.
  See `docs/progress/entries/008-grounded-vlm-perception.md`; native staircase
  accuracy is not yet established.
- New `detector` container (`docker/Dockerfile.detector`, `bake.hcl` sibling target off
  `perception`, `docker-compose.detector.yml`, started persistently as `detector_dev`) running
  the YOLO-World detector sidecar — torch/ultralytics/CLIP kept out of `perception`/`robotican`
  since `yolo_world_trt` is shared task infra, not ROBOTICAN-specific. `run_object_mission_sphera.sh`
  now launches the sidecar via `docker exec` into it instead of the bare host venv, and passes
  `rgb_topic:=/R1/rgb_frame_path` explicitly (previously unset, silently defaulting to XTEND's
  topic). See LESSONS.md for three build issues hit and fixed along the way (numpy pin conflict,
  a mis-built CLIP wheel, TensorRT engine/version lock).
- Built the YOLO-World-`s` TensorRT engine (`yolo_world_trt/build_all.sh s`) end-to-end for
  the first time on this PC (installed `ultralytics`, downloaded `yolov8s-worldv2.pt`) —
  `./run_object_mission_sphera.sh --detector-only` now actually starts the detector sidecar.
  `m`/`l`/`x` are not built yet.
- Sphera/Rooster fork of the "pick an object, then fly to it and land" mission stack:
  `adapter/launch/object_mission_sphera.launch`, `config/mission_sphera.yaml`,
  `run_object_mission_sphera.sh`, and a placeholder `objects_sphera.json` (no real object
  catalog exists for `sphera_jail` yet). Mirrors XTEND's existing `object_mission.launch` stack
  the same way `sphera_drone.launch` mirrors `real_drone.launch` — additive, XTEND's originals
  untouched. Dry-validated (`mission_config.py` + `--help`); not yet flown live. See
  `docs/progress/entries/005-yolo-object-navigation.md`.
- Two new `mission_control.py` services (`rooster_jetson` group): `Rooster Frame Relay ->
  Jetson (R1)` and `Rooster Jetson Frame Watcher (R1)`, wiring the existing
  `dir_push_relay.py`/`dir_watch_path_publisher.py` mechanism (previously XTEND-only, never
  tested for Rooster) so captured frames can be forwarded to the Jetson over rsync/SSH,
  additive to (doesn't change) the existing local Rooster/Falcon vision pipeline. Verified
  end-to-end through the orchestrator's own start/stop/status code, not just ad-hoc commands —
  see `docs/progress/entries/006-rooster-frame-jetson-relay.md`.
- Closed-loop PD altitude hold for the ROBOTICAN Rooster (`rooster_unit.py`), replacing
  the previous open-loop throttle constant that reliably drifted to floor or ceiling.
- `demo_mode_topic`/`demo_mode_request_topic` params on `nav_stack.launch`, letting
  `sphera_drone.launch` route Rooster's demo-mode handshake through `/R1/...` topics
  instead of the XTEND-shaped `/xtend/...` defaults.
- `rooster_demo_mode_manager.py` (new, `falcon_adapter` ROS1 node): a minimal Rooster
  equivalent of `xtend_drone_demo_manager.py` — echoes a requested demo mode back as
  the authoritative current mode, with no other side effects (deliberately does not
  auto-land on FINISH the way the XTEND version does).

### Changed
- `sphera_drone.launch` now overrides FALCON's navigation controller to `multi_axis`
  for Rooster (a genuinely holonomic platform), instead of the default `roll_assist`.
- Every Y-axis bound in `maps/sphera_jail.yaml` (`init_y`, `map_min_y`/`map_max_y`,
  `box_min_y`/`box_max_y`, `vbox_min_y`/`vbox_max_y`) and the Y-axis launch args
  documented in the `fly-rooster-sphera` skill (`bev_ymin`/`bev_ymax`/`goal_y`) —
  negated and min/max-swapped to match the corrected localization sign (see Fixed).
- `map_max_z` raised `4.0` → `5.0` and `box_max_z`/`vbox_max_z` raised `1.8` → `4.0`
  in `maps/sphera_jail.yaml`, so RViz shows real room geometry up to the actual
  ceiling instead of truncating at an artificially low height, while keeping a
  real (1.0m) margin between the map and box/vbox bounds (see Fixed for why the
  margin matters).
- `cam_min_depth` raised `0.1` → `0.45` in `sphera_drone.launch` (see Fixed).
- `mapping_sync`'s `freeze_on_turning_mode` set to `false` in `sphera_drone.launch`
  (see Fixed) — turning-smear protection is temporarily disabled pending a real fix
  to the rotation-supervisor bug it exposed.
- Stale saved RViz camera position in `maps/sphera_jail.rviz` (`Focal Point Y: 14.66`)
  updated to `Y: -14.66` to match the corrected localization sign.

### Fixed
- `waypoint_follower_node.py`: `_publish_twist_multi` was defined twice in the same
  class; the second (older) definition silently shadowed the first and didn't accept
  the `vz` keyword the caller passed, so every navigation tick threw a `TypeError`
  before ever calling `.publish()`. FALCON's internal state showed `nav=RUN` the whole
  time, but `/cmd_vel` was never actually emitted — the drone never moved toward a
  clicked BEV goal regardless of controller/topic configuration.
- `rooster_ground_truth_localization.py`: `position.y` was passed straight through
  from Sphera/Unreal telemetry while yaw was already negated for the left-handed →
  right-handed conversion, leaving position and rotation handedness inconsistent.
  Completed the conversion by negating `position.y` too (see LESSONS.md for how this
  was verified).
- Click-to-fly deadlock for Rooster: the default `roll_assist` controller's demo-mode
  confirmation handshake never resolves because nothing publishes to `/xtend/demo_mode`
  for this platform — fixed via the `multi_axis` controller switch above (Rooster is
  holonomic and doesn't need the turn-then-forward handshake `roll_assist` requires).
- Noisy/speckled voxel map: the bottom ~25% of every RGB frame showed a near-constant
  `~0.17-0.35m` depth reading (the camera rig/mount itself, visible in its own FOV,
  not real environment) that was below `cam_min_depth` and so got fused as a permanent
  phantom wall directly ahead of the drone on every frame — very likely the real cause
  of "boxed in, no A* route" seen the same day. Fixed by raising `cam_min_depth`
  (see Changed).
- `exploration_node` crashing (`voxel_mapping::ESDF::getDistance` → glog FATAL
  "Address out of range") introduced by setting `vbox_max_z` exactly equal to
  `map_max_z` — ESDF's neighbor-cell queries need a real margin near a boundary, not
  just `map ⊇ vbox` ordering. Fixed by raising `map_max_z` instead (see Changed).
- RViz appearing completely empty (no error, just nothing rendered): a stale saved
  camera focal point in `sphera_jail.rviz` from before the Y-axis fix pointed the
  camera at the mirror-image empty location where the room used to be under the old
  sign convention (see Changed).

### Known issues (not yet fixed)
- `waypoint_follower_node.py`'s rotation supervisor gets stuck permanently requesting
  `"turning"` mode while its navigation loop targets the default startup goal with no
  real flight dynamics ever confirming a turn is complete — this permanently froze
  `mapping_sync`'s rotation-freeze mechanism once `rooster_demo_mode_manager.py` made
  it actually engage. Worked around by disabling `freeze_on_turning_mode` for now
  (see Changed); the supervisor bug itself is unresolved.
- Possible left/right (lateral) mirroring in the built map: reported once during a
  BEV-driven flight (drone next to the real left wall in Sphera; map showed it next
  to the right wall). Forward/back and altitude are both confirmed correct via
  quantitative ground-truth tests, so this is a different bug from the Y-axis fix
  above — most likely in how the camera's local lateral axis projects into world-frame
  points, not the drone's own tracked pose. Not yet conclusively confirmed or
  root-caused; a physical landmark was added to the test hallway to check this
  properly next session.

<!--
Example of a real entry once you have one:

## [Unreleased]
### Fixed
- Hover z-axis drift at hover_z=560 traced to accumulated integral windup in the altitude
  controller, not the sim's ranger noise as first suspected. See LESSONS.md.
-->
