# 043 - MP3D v1 val, ground-truth perception, the full official split

**Branch:** `feat/objnav-habitat-mp3d-nadav`
**Status:** done
**Roadmap item:** the perception upper bound of the room-search method, on a third benchmark (HM3D: entry 039)

## Goal
Measure the method under ideal perception on the complete official `objectnav_mp3d_v1` val split: 2,195 episodes in
11 scenes, with nothing dropped, filtered or reordered. Habitat-lab's MP3D ObjectNav settings stay unchanged:
- 500 steps; 0.25 m forward; 30° turns and tilts.
- 640x480 camera at HFOV 79°, mounted at 0.88 m.
- Success is a STOP within 0.1 m of a goal view point.
- The navmesh is recomputed for the agent.

The run uses commit `e0e7c16c`, with `qwen2.5:3b-instruct` in both LLM roles (labels and node oracle) at the same
8,192-token context.

## Result (2026-11-06 02:34 UTC)

Run: `gt_oracle_mp3d_2195ep_20261105T141854Z`. It took 12 h 16 min (2026-11-05 14:19 to 2026-11-06 02:34 UTC), 179
episodes an hour. How it was launched:
- Every scene is cut into shards of 10 episodes, taking the publisher's order strided.
- Up to 16 shard processes. A shard starts only while used VRAM plus one renderer's 1,400 MiB stays under 88 % of the
  24 GB card. That held it to 15 processes at most, 13 typically, and running work was never paused.
- The shards are dealt over three native Ollama servers with 4 request slots each.
- Lean records, no video.

The LLM's answer queue limits throughput, at about 21 answers an episode. Peaks: RAM 46 %, CPU 52 %, VRAM 87 %,
GPU 53 %.

| | episodes | SR | SPL | SoftSPL | DTG (m) | mean steps | mean path (m) | steps (successes) | STOP / limit / error |
|---|---|---|---|---|---|---|---|---|---|
| **official, 0.1 m** | 2195 | **52.1 %** | **23.6 %** | 30.7 % | 4.42 | 202 | 20.9 | 136 | 1785 / 410 / 0 |
| same rows at ApexNav's 0.2 m | 2195 | 53.3 % | 24.2 % | | | | | | |

`campaign_summary.json` gives:
- SR 0.521 (95 % CI 0.500-0.542) and SPL 0.236 (CI 0.226-0.247).
- complete split: True, with 2,195 of 2,195 published episodes in 11 of 11 scenes.

The 0.2 m row re-scores the same rows, and 26 episodes change to success. Of the 641 wrong STOPs, 26 ended under 0.2 m
from a goal view point, 62 at 0.2-1 m, 120 at 1-3 m and 433 at 3 m or more.

### Per scene

| scene | episodes | SR | SPL | SR 0.2 m | SPL 0.2 m | DTG (m) | mean steps | no goal on the start storey |
|---|---|---|---|---|---|---|---|---|
| 2azQ1b91cZZ | 200 | 42.5 % | 20.6 % | 44.0 % | 21.2 % | 4.44 | 224 | 10 |
| 8194nk5LbLH | 201 | 21.4 % | 10.0 % | 22.4 % | 10.2 % | 10.74 | 227 | 138 |
| EU6Fwq7SyZv | 198 | 50.0 % | 26.0 % | 54.0 % | 28.7 % | 3.77 | 166 | 45 |
| oLBMNvg9in8 | 201 | 69.2 % | 33.7 % | 69.7 % | 33.9 % | 1.57 | 157 | 28 |
| pLe4wQe7qrG | 200 | 100.0 % | 33.8 % | 100.0 % | 33.8 % | 0.04 | 86 | 0 |
| QUCTc6BB5sX | 197 | 50.3 % | 19.8 % | 51.8 % | 20.6 % | 4.86 | 285 | 1 |
| TbHJrupSAjP | 200 | 57.5 % | 28.2 % | 58.5 % | 28.8 % | 3.83 | 238 | 33 |
| X7HyMhZNoso | 200 | 25.0 % | 12.2 % | 25.0 % | 12.2 % | 9.99 | 292 | 103 |
| x8F5xyUWy9e | 200 | 44.0 % | 23.0 % | 44.5 % | 23.2 % | 3.42 | 292 | 30 |
| Z6MFQCViBuw | 200 | 38.0 % | 15.8 % | 38.5 % | 16.0 % | 4.84 | 111 | 0 |
| zsNo4HB9uLZ | 198 | 75.8 % | 36.9 % | 78.3 % | 38.1 % | 1.02 | 143 | 0 |

### Per category

| category | episodes | SR | SPL | SR 0.2 m | SPL 0.2 m | DTG (m) | no goal on the start storey |
|---|---|---|---|---|---|---|---|
| bathtub | 9 | 44.4 % | 25.8 % | 44.4 % | 25.8 % | 10.64 | 2 |
| bed | 48 | 41.7 % | 20.0 % | 43.8 % | 20.7 % | 10.82 | 19 |
| cabinet | 180 | 45.0 % | 21.7 % | 46.1 % | 22.3 % | 3.99 | 19 |
| chair | 434 | 45.2 % | 19.5 % | 45.2 % | 19.5 % | 4.37 | 96 |
| chest_of_drawers | 63 | 38.1 % | 20.2 % | 39.7 % | 21.1 % | 6.81 | 7 |
| clothes | 25 | 32.0 % | 18.6 % | 36.0 % | 20.5 % | 9.65 | 1 |
| counter | 66 | 34.8 % | 19.8 % | 36.4 % | 20.6 % | 6.12 | 12 |
| cushion | 239 | 66.9 % | 34.6 % | 66.9 % | 34.6 % | 2.46 | 37 |
| fireplace | 30 | 66.7 % | 28.7 % | 66.7 % | 28.7 % | 3.07 | 1 |
| gym_equipment | 44 | 9.1 % | 1.8 % | 9.1 % | 1.8 % | 11.47 | 37 |
| picture | 161 | 47.8 % | 25.4 % | 55.9 % | 29.9 % | 2.42 | 11 |
| plant | 87 | 42.5 % | 20.6 % | 42.5 % | 20.6 % | 4.88 | 33 |
| seating | 87 | 74.7 % | 24.7 % | 74.7 % | 24.7 % | 2.38 | 6 |
| shower | 70 | 67.1 % | 34.1 % | 67.1 % | 34.1 % | 5.01 | 11 |
| sink | 82 | 67.1 % | 31.9 % | 68.3 % | 32.5 % | 3.69 | 9 |
| sofa | 53 | 32.1 % | 16.7 % | 35.8 % | 18.4 % | 9.03 | 15 |
| stool | 68 | 44.1 % | 20.6 % | 47.1 % | 22.0 % | 8.39 | 22 |
| table | 340 | 70.6 % | 26.7 % | 71.2 % | 26.9 % | 1.91 | 25 |
| toilet | 28 | 35.7 % | 11.9 % | 35.7 % | 11.9 % | 9.10 | 9 |
| towel | 73 | 34.2 % | 16.7 % | 34.2 % | 16.7 % | 6.21 | 10 |
| tv_monitor | 8 | 12.5 % | 10.4 % | 12.5 % | 10.4 % | 15.92 | 6 |

### Start storey and goal storey
`allow_stair_traversal` stays at its default, False, so the method searches only the spawn storey (the spawn floor
guard's plane, ±0.5 m). That default assumes Gibson SemExp v1.1 and HM3D, which score goals on the spawn storey. MP3D
does not:

| a goal view point within ±0.5 m of the start height | episodes | SR | SPL | wrong STOPs (3 m or more away) | step limit |
|---|---|---|---|---|---|
| yes: same storey | 1807 (82.3 %) | 61.8 % | 28.0 % | 471 (270) | 219 |
| no: other storey | 388 (17.7 %) | 7.0 % | 3.1 % | 170 (163) | 191 |

At ±1.0 m, 1,929 same-storey episodes score SR 58.6 % / SPL 26.5 %, and 266 other-storey episodes score 5.3 % / 2.4 %.
Most episodes in the two weakest scenes are cross-storey:
- 8194nk5LbLH: 138 of 201. All 35 of its gym_equipment episodes start downstairs, and every gym_equipment goal is
  upstairs.
- X7HyMhZNoso: 103 of 200.

The records show the agent seeing a real goal on the other floor, closing to about 0.9 m of planar range and stopping
on its own floor. For example, `8194nk5LbLH/000123`, a cabinet, stopped 32.5 m from the nearest goal view point.

### Checks
- **Episode set.** The rows map one-to-one onto the official files: 0 missing, 0 extra, 0 duplicated, 0 category
  mismatches. A row is keyed `<scene>/<6-digit position>` because the published `episode_id` repeats within a scene
  (2azQ1b91cZZ: 200 episodes, 38 distinct ids).
- **Geodesics.** The live start geodesic matches the publisher's `info.geodesic_distance` to within 0.02 mm on every
  episode.
- **Ground-truth goals.** Each goal was rendered from its own view points (`goal_visibility.py`). 1,314 of 1,316 goal
  objects appear under their own `object_id`, and every annotation box is centred on the published goal position. The
  two that never appear have a best view-point iou of 0.0002 and 0.00007 in the release itself.
- **LLM answers.** Every answer is recorded for exact replay: 46,582 answers in 222 files. All 47,031 `/api/chat`
  requests returned HTTP 200 (this count includes warm-ups and preflight probes).
- **Run hygiene.** No video or images were written, and the resource guard (88 % caps) never paused a worker.

### Published MP3D val rows
These methods learn their perception; this run uses ground truth. That makes this row an upper bound, not a
like-for-like comparison.

| method | SR | SPL | source |
|---|---|---|---|
| SemExp | 36.0 % | 14.4 % | SG-Nav, arXiv:2410.08189v1, Table 1 |
| PONI | 31.8 % | 12.1 % | SG-Nav, arXiv:2410.08189v1, Table 1 |
| ZSON | 15.3 % | 4.8 % | SG-Nav, arXiv:2410.08189v1, Table 1 |
| CoW | 7.4 % | 3.7 % | SG-Nav, arXiv:2410.08189v1, Table 1 |
| ESC | 28.7 % | 14.2 % | SG-Nav, arXiv:2410.08189v1, Table 1 |
| L3MVN | 34.9 % | 14.5 % | SG-Nav, arXiv:2410.08189v1, Table 1 |
| OpenFMNav | 37.2 % | 15.7 % | SG-Nav, arXiv:2410.08189v1, Table 1 |
| VLFM (SG-Nav's table) | 36.2 % | 15.9 % | SG-Nav, arXiv:2410.08189v1, Table 1 |
| SG-Nav-LLaMA | 40.1 % | 16.0 % | SG-Nav, arXiv:2410.08189v1, Table 1 |
| SG-Nav-GPT | 40.2 % | 16.0 % | SG-Nav, arXiv:2410.08189v1, Table 1 |
| VLFM (ApexNav's table) | 36.4 % | 17.5 % | ApexNav, arXiv:2504.14478v1, Table I (0.2 m radius) |
| VLFM* (shortest-path planner) | 32.5 % | 15.9 % | ApexNav, arXiv:2504.14478v1, Table I (0.2 m radius) |
| ApexNav | 39.2 % | 17.8 % | ApexNav, arXiv:2504.14478v1, Table I (0.2 m radius) |
| **this run, ground truth** | **52.1 %** | **23.6 %** | 0.1 m; 53.3 % / 24.2 % at 0.2 m |

## Notes
- **Upper bound.** The run measures the perception UPPER BOUND (`publishable: false`). Habitat-sim's semantic sensor
  replaces the detector and segmenter, and the method itself is unchanged:
  - Only the publisher's goal instances carry the goal label.
  - Context labels come from the raw `.house` names, then mpcat40 (`ground_truth/mp3d_categories.py`).
- **Where the files are.** `runs/` is not tracked.
  - Per-episode rows: `<root>/<scene>/episodes.jsonl`.
  - The frozen configuration: `<root>/<scene>/run.json`.
  - In `runs/gt_oracle_mp3d_2195ep_20261105T141854Z_support/`: the report with the storey split and the goal check
    (`final_report.txt`, `goal_visibility.jsonl`), the scripts that produced them, and `llm_answers/` (270 MB).
- **Open.** 270 same-storey episodes also end in a STOP 3 m or more from every goal. A possible cause is a target on
  another storey being accepted while one exists on the spawn storey, in the shared target closing. Not examined.
- **Not run.** The same split with stair traversal on (`--policy-config` with `{"allow_stair_traversal": true}`, no
  code change), about 12 h.
