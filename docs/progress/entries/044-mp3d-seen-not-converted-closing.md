# 044 - MP3D "seen, not converted": what the recordings show, the closing fix, 15 episodes re-run

**Branch:** `feat/objnav-habitat-mp3d-nadav`
**Status:** done (fix in the working tree; verification run complete)
**Roadmap item:** the failures of the MP3D ground-truth run (entry 043), layer by layer; this is the closing

## Goal
Entry 043's analysis flagged, from the LLM prompts alone, 147 same-storey episodes in which a room's `seen:` list named
the target class and the search still ran 100 or more actions ("seen, not converted"). Fix what the target closing does
wrong in them, and check the fix on 15 such episodes. Nothing else (the storey model, the oracle, the room graph) is
touched here.

## Reproduction
MP3D is now installed on the laptop (`~/datasets/scene_datasets/mp3d`, the 11 val scenes of `mp3d_habitat.zip`;
episodes under `~/datasets/objectnav/mp3d/v1/val`). Every run here: ground-truth perception, the GPU renderer,
`qwen2.5:3b-instruct` in both LLM roles, the campaign's 46,582 recorded answers replayed (`LLM_ANSWERS_REPLAY`), two
scenes at once, `--record`.

- 24 flagged episodes re-run on the unmodified code (`runs/diag_seen3`, `runs/baseline21`): **10 of 24 succeed here**
  where the campaign failed them. The method is not deterministic across machines (the renderer differs, a prompt the
  record does not hold goes to the live model), so a per-episode verdict needs a local baseline; the 14 local failures
  plus zsNo4HB9uLZ/000078 are the 15 below.
- `tasks/planning/objnav_benchmark_runtime/hm3d/episode_levels.py` run over MP3D val (`runs/fixed15_support/levels.csv`):
  162 of 2,195 starts are on stairs, 160 on a platform off every storey; in x8F5xyUWy9e and X7HyMhZNoso 31-40 % of
  the "same storey" episodes have a shortest path that changes level by more than 0.45 m.

## What the recordings show
"Seen, not converted" is several mechanisms, and most of them are not the closing's:

| mechanism | local failures | layer |
|---|---|---|
| the spawn-floor guard masks the storey after a raised or stair start (3,000-4,200 drop cells, 200-500 goals refused, the agent boxed in; the target 1.3-1.9 m away in view) | EU6Fwq7SyZv/000100, /000118, /000125, /000083; zsNo4HB9uLZ/000153; TbHJrupSAjP/000112 | 1 (storey) |
| wedged in clutter: 20-action stall releases on every lock (75 refused steps of 204) | Z6MFQCViBuw/000154, /000005 | navigation |
| a STOP from the wrong level: a picture whose lowest point is 1.42 m above the agent's floor, view points on the level above (DTG 4.49 m) | TbHJrupSAjP/000147 | 1 + closing (the planar range) |
| a near-miss STOP: the memory within 1 m but the agent not where the view points are (DTG 0.26, 0.49 m) | TbHJrupSAjP/000157, 2azQ1b91cZZ/000108 | closing (standing spot) |
| a ground-truth sliver: a sink's rim 24 x 8 px at 3.7 m, a picture seen edge-on 4 x 100 px, a cabinet strip 6-25 px wide -- above the oracle's 64 px floor, below any real detector | x8F5xyUWy9e/000038, Z6MFQCViBuw/000128, zsNo4HB9uLZ/000001 | the oracle's definition |
| **the closing itself:** an ARRIVED standoff with the fresh range 1.05-1.30 m, the re-sighting measured on a neighbouring instance, INSPECT / CLOSE looping until "no safe path to the target 1.1 m away" | zsNo4HB9uLZ/000078 (actions 83-106) | 2 |

So entry 043 over-attributed to the closing: of 14 local failures, one is the closing's own mechanism, two are its
standing spot, one its planar range on another level.

## The fix (`core/planning/object_search/closing/target_closing.py`)
Three rules, each from zsNo4HB9uLZ/000078's first lock (a cushion among cushions on a sofa):

1. **An arrived standoff with the fresh range at most one forward step beyond the limit is the inspection's.** The ring
   is drawn about the memory's centroid and the executor ends within its tolerance of the ring point, so the near edge
   of a small object reads 1.05-1.30 m where the route completes. Before, such an arrival fell through to CLOSE and the
   completed route's idle result was executed as a TURN: three turns swung the agent 90 degrees off the cushion. Now the
   inspection's step rule takes the one MOVE_FORWARD that puts the next frame in range. The flag is read after the
   route's own command, since the action the route completes on is the one that sets it. The closest-approach STOP
   still needs a widened standoff.
2. **The re-sighting is the projection nearest the 3-D memory** (`_best_projected`), a whole box before a sliver, not
   the most confident box -- with the ground-truth detector's confidence 1.0 on every instance the old `max` took
   whichever the detector listed first, so the range was measured on the neighbour and the centring turns chased a
   sliver at the frame's edge. Before any memory: confidence, then box area.
3. **The inspection resumes the approach at most `max_inspection_resumptions` (3) times per lock, and never on a
   side-clipped box** whose near range is the visible, farther part's.

Four counters record them per episode (`arrived_step_out_inspections`, `resumption_cap_hits`,
`clipped_resumptions_withheld`, `resighting_reselections`). Tests: three new or rewritten in
`tests/test_target_closing.py` (134 pass), 194 across the closing rigs, 58 in the package. One old test moved its
fixture from 1.3 m to 1.6 m: 1.30 m is now "one step out" and the step rule's, not a closer ring's.

## The 15 episodes (`runs/fixed15`, baseline in `runs/baseline21` + `runs/diag_seen3`)

| episode | target | baseline | fixed | parted at | mechanism |
|---|---|---|---|---|---|
| zsNo4HB9uLZ/000078 | cushion | SR 1, 211 steps | **SR 1, 48 steps** | 40 (re-sighting) | the closing's |
| zsNo4HB9uLZ/000001 | cabinet | SR 0, STOP at 1.87 m | **SR 1, 198 steps** | 177 (re-sighting) | sliver cabinet |
| Z6MFQCViBuw/000154 | table | SR 0, limit, 3.89 m | SR 0, STOP at 1.34 m | 87 (re-sighting) | wedged |
| EU6Fwq7SyZv/000083 | cushion | SR 0, limit, 0.03 m | SR 0, limit, 1.98 m | 0 (re-sighting) | guard, stair start |
| EU6Fwq7SyZv/000100 | chair | SR 0, limit, 1.40 m | identical | -- | guard |
| EU6Fwq7SyZv/000118 | chair | SR 0, limit, 1.57 m | identical | -- | guard |
| EU6Fwq7SyZv/000125 | chair | SR 0, limit, 1.02 m | identical | -- | guard |
| zsNo4HB9uLZ/000153 | sink | SR 0, limit, 4.88 m | identical | -- | guard |
| TbHJrupSAjP/000112 | cushion | SR 0, limit, 1.15 m | identical | -- | guard |
| TbHJrupSAjP/000147 | picture | SR 0, STOP at 4.49 m | identical | -- | wrong level |
| TbHJrupSAjP/000157 | picture | SR 0, STOP at 0.26 m | identical | -- | near miss |
| 2azQ1b91cZZ/000108 | cushion | SR 0, STOP at 0.49 m | identical | -- | near miss |
| x8F5xyUWy9e/000038 | sink | SR 0, limit, 11.05 m | identical | -- | sliver sink |
| Z6MFQCViBuw/000005 | chest_of_drawers | SR 0, limit, 11.80 m | identical | -- | wedged |
| Z6MFQCViBuw/000128 | picture | SR 0, limit, 10.17 m | identical | -- | sliver picture |

Every change parts exactly at a frame where `_best_projected` chose a different box than the old `max` (the counter
and the parting step agree: 14 reselections over the 15 episodes), so the four changes are the fix's and the eleven
identical runs are episodes in which no new rule had anything to act on. Rule 1 never fired in these 15 runs
(`arrived_step_out_inspections` 0): rule 2 changed 000078's trajectory before the arrival it was written for recurred,
so it rests on the baseline trace (actions 83-106) and its tests. The fix converts the mechanism it was built for
(4.4x fewer actions on 000078) and one sliver case; it cannot touch the guard, the wedge, the standing spot or the
wrong level.

## Harness
Z6MFQCViBuw/000005's worker twice lost its main thread mid-episode (steps 255 and 307) and lingered with 32 live
threads; `campaign.py` waited on it for an hour (`LESSONS.md` 2026-11-09). Killed by hand, resumed with
`--resume-accepting-changes gpu_plan`; the third attempt ran to 500 steps.

## Open
- The storey model for split-level and stair starts (6 of the 14 local failures): layer 1, entry 043's first item.
- The standing spot: STOP where a view point can be -- in front of the object along its normal, not at the ring point
  the route happened to reach (the 88 wrong STOPs within 1 m of the campaign).
- A planar range on another level: `measured` needs the object's floor, which needs the level model.
- The ground-truth detector admits slivers (a 4 x 100 px picture, a 24 x 8 px sink rim) that no detector reports; a
  minimum box side (about 10 px) in `ground_truth/perception.py` would define "ideal perception" more honestly. Changing
  it changes the upper bound's definition, so it is a decision, not a fix.
- The campaign pool should end a worker whose `steps.jsonl` stops moving.
