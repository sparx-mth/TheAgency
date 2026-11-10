# 046 - MP3D single storey: a fixed set, a deterministic baseline, the flat-p ablation, and the remaining failures

**Branch:** `feat/objnav-habitat-mp3d-nadav`
**Status:** done (changes in the working tree, uncommitted; four campaigns on the fixed set complete)
**Roadmap item:** the method's full potential on ONE storey, before the storey model (layer 1) is touched. Entries
044 and 045 fixed the closing's own share and the graph bookkeeping; this entry makes the measurement repeatable,
asks what the 3B oracle contributes, and fixes the single-storey mechanisms the recordings show.

## 1. Making results measurable

**Deterministic LLM output.** Every run of the study shares one LLM answer store
(`runs/single_storey_v1_llm_answers`, `LLM_ANSWERS_RECORD`; `llm_answer_cache.py`, keyed by episode, prompt,
model, temperature 0, seed 17). A question an episode has already had answered gets the same answer in every later
run, so two runs part only where a change under test makes them ask something new. Check: 14 baseline episodes
re-run with the store replaying (`runs/ss_determinism`) -- **0 live calls, 14 of 14 rows identical** in success,
steps, SPL, path length and DTG. The live model stays nondeterministic (entry 041); the store makes a re-run of the
same code a replay of the first, which is what a comparison needs. `LLM_EXCLUSIVE=1` (a cold model per call, 6.4 s)
would make even the live answers a function of the request, at 1.8x the LLM time; not used.

Two things learned the hard way (`LESSONS.md` 2026-11-10): a store shared by two runs that hold the SAME episode at
the same time ends with two answers to one key and every later process refuses it
(`llm_answer_cache --resolve-conflicts <dir>` repairs it, keeping one line per key); and a scene worker that loses
its main thread stalls the campaign -- `campaign.py --stall-minutes` (default 30) now kills such a worker and
`--resume` reruns the scene (it fired once in the baseline, at 2azQ1b91cZZ/000118 step 74, 30 minutes silent).

**The fixed set** `mp3d/sets/single_storey_flat_v1.json`: 180 episodes drawn uniformly (seed 17) from the 1,281
val episodes that start on a storey (`episode_levels` start_kind floor), whose nearest goal view point lies on that
storey within 0.2 m of the start height, whose shortest path never rises or drops 0.2 m, in the eight houses that
are not split-level (x8F5xyUWy9e, X7HyMhZNoso, EU6Fwq7SyZv out). Per scene: 2azQ1b91cZZ 24, 8194nk5LbLH 6,
QUCTc6BB5sX 21, TbHJrupSAjP 20, Z6MFQCViBuw 29, oLBMNvg9in8 21, pLe4wQe7qrG 27, zsNo4HB9uLZ 32; 19 categories
(table 37, chair 25, picture 21, cushion 20, cabinet 14, seating 11 ... clothes, gym_equipment, toilet 1 each).
`single_storey_set.py` regenerates it from the shipped survey; `run_single_storey.sh <name>` runs a campaign on it
from a frozen rsync of the source (a scene child imports the repository at its start, so a campaign that runs for
hours beside an edited tree would mix two methods).

**Every run here:** ground-truth perception (`min_pixels` 64; the on-object filter OFF in sections 2-5, ON in 5b),
`qwen2.5:3b-instruct` in both LLM roles on the resident Ollama (NUM_PARALLEL 2), seed 17, `--record --lean`, 2-4
scenes at once on the laptop's 8 GB card; 4-5 h of worker time per run. The LLM is not the bottleneck once a run
replays earlier stores (the fix run: 1,334 live calls, 0.7 h of model time in 4.6 h of worker time); rendering is,
at 0.56 s per action, so two arms with two workers each run side by side.

## 2. Baseline and the flat-p ablation

`runs/ss_baseline` (the current code at `96cd374c` plus the harness of section 1) and `runs/ss_flat_p` (the same
code with `{"room_search": {"flat_p": 0.15}}`: every node read at 0.15 -- the median the 3B oracle wrote over the
2026-11-09 re-runs -- without a call, so RPT* orders the nodes by travel cost alone; the labels, exclusions, landmark
nodes and closing unchanged):

| campaign | SR | SPL | SoftSPL | DTG (m) | steps | STOP / limit | wrong STOPs (within 1 m) | dSR vs baseline [95 %] | McNemar p | dSPL [95 %] | flips +/- |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | **62.8 %** | **27.6 %** | 34.3 % | 2.58 | 164 | 161 / 19 | 48 (10) | | | | |
| flat-p | 59.4 % | 26.1 % | 33.8 % | 2.87 | 162 | 162 / 18 | 55 (11) | -3.3 pts [-6.7, +0.0] | 0.11 | -0.015 [-0.034, +0.002] | 2 / 8 |

131 of the 180 episodes are identical in the two runs (the closing took over before any order mattered, or the
first order was the same); the oracle changed 49 and netted six successes. **The 3B oracle contributes about three
SR points and 1.5 SPL points over nearest-first, not significant at n = 180** (the interval touches zero). Its
cost: 493 oracle calls (plus 2,753 label calls, which the flat run makes too). The per-category and per-scene
columns are in `runs/ss_report_step2.json`; by scene the oracle's gain is oLBMNvg9in8 +3, zsNo4HB9uLZ +2,
QUCTc6BB5sX and TbHJrupSAjP +1, Z6MFQCViBuw -1.

Per-category SR, baseline (successes / n): table 30/37, cushion 17/20, seating 10/11, picture 13/21, cabinet 8/14,
chair 9/25, plant 5/7, shower 4/5, chest_of_drawers 4/8, bed 3/6, fireplace 3/3, stool 2/3, towel 2/6, sofa 1/5,
counter 1/4, toilet 1/1, clothes 0/1, gym_equipment 0/1, sink 0/2.

**Decision on the LLM (the user's, 2026-11-10):** the oracle stays as it is and model-agnostic -- no prompt, model
or probability change, nothing tuned around the 3B's weaknesses; the model moves to a 7B/14B on a 4090 later, as a
config change. **Decision on ground-truth errors (the user's, same day):** they are of no interest -- the on-object
filter is ON by default for ground-truth runs, phantom-id failures are outside the failure analysis, and nothing
GT-specific lives in the navigation logic (the closing-level sliver gate written for them was removed again).

## 3. What the 67 baseline failures are (`single_storey_failures.py` over the recordings and the published goals)

| mechanism | n | reading |
|---|---|---|
| **phantom ids** -- a STOP on a lock whose 3-D memory stood more than 1.5 m from every goal object of the class | **32** | the semantic sensor painted the class's instance ids where no such object stands: 23 are slivers under 10 cm of visible height (a "chair" 1.4 cm tall at 1 m, a "plant" 1.7 cm tall), 9 are whole boxes (71 x 94 px chairs, curtains and vases painted across the wall of Z6MFQCViBuw's staircase, 3-7 m from the chairs). 136 of the 475 locks the closing took (29 %) were such phantoms; by scene QUCTc6BB5sX 36 of 67, pLe4wQe7qrG 28 of 46 (harmless there: SR 100 %), Z6MFQCViBuw 25 of 67 |
| near-miss STOP (DTG 0.14-0.82 m) | 9 | 7 are raised targets -- pictures with their underside 0.8-1.35 m up, a wall cabinet at 1.08 m, a towel at 0.93 m, a cushion at 0.58 m. **How the view points are made** (habitat's generator, read from its source): a grid of candidates at the object's centre height, kept within 1 m of the object's box in that plane, dropped vertically onto the navmesh, kept only where the object's pixels show with the camera turned to it at three pitches. Over the set's 1,020 goals (`.house` boxes): the farthest view point of a floor object is 1.02 m from its box (median) but 0.84 m for an object raised 0.3 m or more (pictures 0.81, wall cabinets 0.72) -- a raised object must be stood under more closely -- and the view points of a thin wall-mounted object stay within 0.26 m of its width along the wall (median; 0.6-0.9 m for every other category): **in front of it, never beside it**. 8 of the 9 near-misses stood beside their object (0.2-0.4 m beyond its width, 45-70 degrees off its normal), inside the 1 m band |
| seen through a barrier or from another level | 4 | a cabinet and a gym machine seen across an opening, two pictures seen from under their landing |
| never seen in 500 actions | 10 | search failures proper: the room order never reached the room (five in 2azQ1b91cZZ) |
| wedged in clutter | 3 | 53-88 refused MOVE_FORWARDs; 2azQ1b91cZZ/000058 refused six from one spot at five headings in 20 actions, the first three excused as doorway scrapes by a guard that treated every squeeze of the route as a doorway, and the closing released a table in plain view 3 m away as "went nowhere" -- then locked and lost it 26 more times |
| locks released, step limit | 6 | re-sightings refused at the frame's edge, an inspection that saw nothing, a stall |
| other STOPs | 3 | DTG 1.0-2.2 m |

The labeller over the 180 episodes: 2,753 calls; `dining_room` adopted 225 times, 23 of them on `table` alone, 22
on `chair + table`, 13 on `cabinet + chair + table`; 32 episodes ended with two or more dining rooms. 257 verdicts
were on rooms with NO objects (the shape question): hallway 140, foyer 56, unknown 35, storage_closet 18,
living_room 8.

## 4. The fixes (all behind settings; defaults changed as noted)

| where | what | setting |
|---|---|---|
| `navigation/scrape_guard.py` | a refusal is excused as a doorway scrape only in a channel a confirmed door landmark names; between furniture it is written on the map at once, and the second one leaves the sticky bar the planner routes around | `ScrapeSettings.doorways_only` (True; False = the geometry-only channel) |
| `revisable_room_labels.py` | a `dining_room` verdict on fewer than 3 chairs reads `unknown` (a desk, a side table and a coffee table are tables under the benchmark's label); an empty passage-like room is `hallway` by rule, weak, without a classifier call | `RoomLabelSettings.dining_min_chairs` (3; 0 off), `shape_rule` (True), `shape_rule_confidence` (0.5) |
| `closing/target_closing.py` | the STOP overshoot reads the memory's planar range lifted by the target's lowest measured point above the base (objects under 0.3 m up read as before), and takes up to 3 steps for a raised target | `TargetClosingSettings.elevation_overshoot` (True), `elevation_min_m` (0.30), `stop_overshoot_elevated_max_steps` (3) |
| `closing/target_path.py`, `closing/wall_normal.py` | the standoff ring of a raised target mounted on something starts from the direction into the room -- toward the free cells within 0.8 m of a memory with something occupied within 0.3 m of it -- instead of the agent's own bearing, so the standoff is in front of a picture, not beside it; the pursuit's walk-back uses the same rule (the lock's lowest point is remembered) | `frontal_standoff` (True), `frontal_wall_radius_m` (0.8) |
| `ground_truth/perception.py`, `source.py`, `mp3d/env.py`, `habitat/simulator.py` | the on-object filter: a rendered pixel is the instance's only where the frame's depth puts it on the instance's annotated box (inflated by M); an instance mostly off its box is not reported. Perception-layer only; nothing in the navigation reads it | `--ground-truth-on-object-m M` / `GT_ON_OBJECT_M` (**0.25 m, ON by default**; 0 off) |
| `settings.py`, `policy.py` | `--policy-config` reaches the loop and the labeller: `{"room_search": {...}, "room_labels": {...}}` | `RPTSettings.room_search`, `room_labels` |
| `search_node_oracle.py`, `room_search_loop.py`, `scene_graph.py`, `oracle_retry.py` | the flat-p ablation (`source="flat"`, accepted by the runtime) | `LoopSettings.flat_p` (None) |
| `hm3d/scene_pool.py`, `mp3d/campaign.py`, `run_benchmark.sh` | the stall watchdog | `--stall-minutes` (30) |
| `llm_answer_cache.py` | `--resolve-conflicts <dir>` | |

Tests: 1,963 pass across the runtime, topology, object_search and exploration rigs (one pre-existing `test_monitor`
failure deselected, unrelated). New: the set (3), the watchdog (3), the flat oracle (2), the scrape guard (1), the
labeller rules (2), the raised-target overshoot (1), the frontal standoff (1 + 5 on `wall_normal`), the on-object
filter (1), the store repair (1).

## 5. The fixed set re-run with the first three fixes (`runs/ss_fixes`: scrape guard, labeller rules, elevation overshoot; same perception as the baseline, filter OFF)

| campaign | SR | SPL | STOP / limit | wrong STOPs (<= 1 m) | dSR vs baseline [95 %] | McNemar p | flips +/- |
|---|---|---|---|---|---|---|---|
| baseline | 62.8 % | 27.6 % | 161 / 19 | 48 (10) | -- | -- | -- |
| fixes | 64.4 % | 27.0 % | 162 / 18 | 46 (8) | +1.7 pts [-2.8, +6.1] | 0.63 | 10 / 7 |

Not significant on 180 episodes; the flips say what each fix did (`/tmp`-free: `single_storey_report.py` lists them,
the first divergence of each pair of recordings names the fix):

* **Elevation overshoot: +2, -0.** 2azQ1b91cZZ/000025 (wall cabinet) and TbHJrupSAjP/000156 (picture) take one more
  step than the baseline at the same STOP decision and succeed (DTG 0.22 and 0.14 before).
* **Doorway-only scrape guard: +2 (+1 shared), -0.** 2azQ1b91cZZ/000058 (the wedged table) succeeds at action 183
  against 500 wedged; Z6MFQCViBuw/000090 (chair, wedged) at 333; 2azQ1b91cZZ/000108 (cushion) diverges at a refusal
  written on the map and then STOPs on the lifted 3-D range.
* **Labeller rules: +5, -7, net -2 -- a reshuffle.** Every other flip diverges at a *transit* node choice: a
  different room label changed the oracle's briefing and the planner's order. 3 of the 5 gains and 3 of the 7
  losses are GT phantoms met or avoided on the new route (outside the analysis); the method's own are +3
  (000020 barrier, QUCTc6BB5sX/000017 locks released, TbHJrupSAjP/000040 never seen) and -4 (three never seen
  within 500 actions after long searches that the baseline finished at actions 302-431, one lock released in
  clutter). 2,753 -> 2,000 classifier calls; `dining_room` adoptions 225 -> 92; shape-only calls 257 -> 0.

The failure census of this run (30 the method's, 34 the sensor's): near-miss 7 (beside 4, in the band 2, far 1),
barrier 4, other 1, never seen 15, wedged 1, locks released 2. The near-misses are what the fourth fix is for.

## 5b. The arms with the on-object filter ON (identical perception; each fix toggled by `--policy-config`)

RESULTS_GT_ARMS

## 6. What remains on a single storey (the method's failures, GT artefacts excluded)

REMAINING

**Mechanisms that are single episodes, documented and not fixed here:**

* **The map closes the door behind the agent -> escape loop.** zsNo4HB9uLZ/000153 (sink): the agent enters a 2 x
  2.5 m bathroom; seen from inside at close range the doorway is drawn 0.2-0.3 m wide, the inflated map boxes the
  agent in, and the exploration fallback's `escape` ("probing the thinnest barrier by a forward step") spends 394 of
  500 actions turning (260 turns, 86 steps) inside a 1 x 1.3 m area. Deterministic: 394 actions in the baseline and
  the fix run, 376 under flat-p. The only other episode with 100+ escape actions (oLBMNvg9in8/000129, 219) succeeded.
* **A real instance that is not a goal.** Z6MFQCViBuw/000098 (chair): the lock's memory is 3.4 m from every goal
  chair -- a chair the dataset dropped (no view point reaches it), unreachable; the approach was refused 83 times,
  released 4 times and walked back to 15 times. Any detector sees it; the method cannot know it does not count.
* **Clutter: no route to any standoff.** oLBMNvg9in8/000028 (table): seen, locked, released 4 times as "no safe
  path from here" (396 squeeze-planner calls); the baseline reached the same table from the other side at action 96.
* **In the band, off every view point.** Z6MFQCViBuw/000121 (picture): frontal, 0.69 m from the box, but the goal's
  view points are one 0.1 m strip (furniture under the picture); the overshoot's second step was blocked.
  oLBMNvg9in8/000083 (picture): 0.21 m from a view point in a straight line, 0.51 m by the navmesh.
  QUCTc6BB5sX/000184 (counter): the first overshoot step blocked 0.94 m from the box, whose view points end at 0.76.

## Harness notes
- One stall in the baseline (the 2026-11-09 zombie worker), killed by the watchdog at 30 minutes and resumed; the
  resumed episode succeeded at step 124, so the stall was the harness's.
- Two store conflicts (33 keys) from running the flat-p ablation beside the baseline's resume on the same scene;
  repaired keeping the baseline's answers (the baseline stays replayable; two flat-p episodes do not).
- GPU: Ollama 2.7 GB + 0.83 GB per renderer; five renderers fit beside the model, the LLM caps the throughput at
  0.5 requests/s whatever the worker count.
