# ObjectNav arrival spin and Nadav's frontier algorithm: handoff, 2026-11-24

Continues `todos/objnav-room-score-handoff-2026-11-22.md` (room-score and frame-capture study).
Two branches: this one (`daphna/objnav-verify-diagnosis-rebased`, analysis code and handoffs) and
`daphna/objnav-arrival-spin` (code change on top of Nadav's `feat/objnav-vlm-frontiers-nadav`,
worktree `/home/user1/GIT/TheAgency.worktrees/daphna-arrival-spin`). Runs were still going when this
was written; see "Runs in flight".

## What Nadav's branch is (read from code and the Algorithm 1 / 2 figures)

- `--agent rpt` is the figure's algorithm: frontier nodes with sticky ids, 1-3 logged frames per
  node chosen by greedy coverage of its cells, one VLM score per node, then RPT\* and A\*.
  The old room planner is kept as `--agent rpt_room_llm`.
- A3 of Algorithm 1: reaching a frontier marks it visited; the next decision is taken without a
  360-degree scan; a cluster still present 12 actions after its first visit is blocked. The only
  scan is the opening one (10 turns of 30 degrees with a 79-degree camera).
- The code differs from the figure: it also takes glances along the route (`exploration/path_glances.py`:
  a side or full-circle turn when the unknown floor gained pays for the actions). The figure has none.
  Decision with Nadav: add glances to the figure later; for now the baseline includes them.
- Score once (A2): a node is scored the first action it has a usable view and never again. Nadav's
  view: no point rescoring, but worth testing some day.
- The question to the VLM (`frontier_views/scoring.py`): a red box on the node in 1-3 same-camera
  views stitched into one image, asking the chance of finding the target by exploring beyond that
  boundary, JSON probability in [0, 1]. Nadav chose the box because the image, the space and the
  node's position are geometrically related. My room-level wording tests had no box and asked about the
  room, so they do not transfer.
- RPT\* minimises expected cost from survival probabilities; the values enter the cost, not only their
  order, and p must be below 1 (the code divides by 1 - p). Normalising scores to sum to 1 would change
  decisions and can produce an invalid 1.0. Not tried.
- "Arrived" has two definitions: a node is marked visited when the agent is within
  `goal_tolerance_m` (0.25) plus one map cell (0.1) of its point, and only while the node still exists;
  a route has arrived (`route_memory.arrived`) when the whole path is walked and the agent is within
  tolerance of its goal. The second is the figure's `Arrived`.

## Within-house metric (answers "0.5 is not the right chance")

`room_dataset/report_within_scene.py` (4 tests): each room judged against the other rooms of its own
house, 4B model, original percent question, 178 rooms in 52 houses, `~/hm3d_spins/report_within_scene.md`.

| spin | AUROC in house | top-1 hits | chance | top-1 minus chance |
|---|---|---|---|---|
| arrival view only | 0.590 | 0.533 | 0.398 | +0.135 [+0.065, +0.205] |
| 3 turns | 0.646 | 0.558 | 0.398 | +0.159 [+0.073, +0.237] |
| 6 turns | 0.627 | 0.495 | 0.398 | +0.097 [+0.019, +0.182] |
| 10 turns | 0.613 | 0.513 | 0.398 | +0.115 [+0.036, +0.199] |

Rooms where the target is in none of the spin's views: AUROC 0.51 to 0.60, top-1 0.44 to 0.47 against a
chance of 0.36, differences +0.07 to +0.10 with intervals reaching 0. Houses here have about 3 rooms,
the ones the planner visited, and many hold the target, so chance for top-1 is 0.40, not 0.5.
Spin differences on top-1 are within noise.

## The arrival spin (branch `daphna/objnav-arrival-spin`, local commits pushed with this handoff)

- `RPTSettings.arrival_spin_turns` (0 by default, non-negative int): turns of 30 degrees taken in place
  once per frontier route, when the route is walked to its goal, even if the frontier is already gone.
  Reuses the warm-up rotation (`policy.start_rotation`), so the frames enter the frame log and later
  nodes can be scored from them. A new floor still gets the full warm-up
  (`full_rotation_turns`). Set it with `--policy-config file.json`, e.g. `{"arrival_spin_turns": 3}`.
- Commits: `d5984136` the setting and hook; `d5b07f49` fix: the spin cut the opening 10-turn warm-up to 3
  in 48 of 58 long episodes (a node within tolerance at spawn), so it now starts only when the current
  rotation is done; `af90c224` trigger moved from "a node is under the agent" to "the route ended"
  (first version fired only 19 times in 14 of 68 episodes).
- Tests: `frontier_views/tests/test_arrival_spin.py` (9), whole `object_search` suite 394 passing.
  Run with `~/GIT/TheAgency/.venv/bin/python -m pytest` (has `skimage`; `navdp` does not).
- Wiring: `PERCEPTION=service`, detector with `DETECT_PY=~/miniconda3/envs/detector_bench/bin/python`,
  Ollama external at 11434 with `qwen3-vl:8b-instruct` for both roles (no download; the launcher's
  default `qwen2.5vl:7b` is not installed and Nadav's own full MP3D run uses it, on his machine).
  HM3D v2 val scenes are in `~/datasets/scene_datasets`; MP3D scenes are in
  `~/datasets/mp3d/scene_datasets` (`MP3D_SCENES_DIR`), not the default path.

## Pilot results so far (HM3D v2 val, first 2 episodes of each scene, 68 episodes, same ids per condition)

First version of the trigger (node under the agent), spin 0 / 3 / 6 on `runs/arrival_pilot` of the
worktree: SR 0.294 in all three, SPL 0.131 / 0.133 / 0.127, mean steps 172 / 177 / 177, no episode won
or lost by the spin. Not informative: the spin fired in only 14 of 68 episodes. Outputs of the
truncated-warm-up runs are in `spin3_truncated_warmup`, `spin6_truncated_warmup` (invalid, do not use).
The old-trigger outputs are in `spin3_node_trigger`, `spin6_node_trigger`.

## Runs in flight (check before reading anything into them)

- HM3D, new trigger, spin 3 then spin 6 into `runs/arrival_pilot/spin3`, `spin6`
  (about 30 minutes each); `~/verify_diag/run_arrival_pilot.sh`, log `~/verify_diag/arrival_pilot_rerun2.log`.
- MP3D v1 val, 6 episodes per scene (66), spin 0 / 3 / 6, queued to start after the HM3D run:
  `~/verify_diag/run_arrival_pilot_mp3d.sh`, outputs `runs/arrival_pilot_mp3d/spin{0,3,6}`, logs
  `~/verify_diag/arrival_mp3d_spin*.log`. The script unloads the model at the end.
- Report: `~/verify_diag/arrival_pilot_report.py <runs/arrival_pilot>` prints the validity check
  (opening warm-up whole), how often the spin fired, SR, SPL, steps, and paired wins and losses with a
  bootstrap over scenes. Condition: more than a couple of episodes either way with the spin firing
  in most episodes; 68 episodes at SR 0.29 cannot show a difference of two or three.

## Open items

1. Read the HM3D and MP3D pilot results; if the spin fires often and shows a signal, scale to about 300
   episodes per condition (about 1.5 hours each); consider k = 10, the figure's full scan on arrival.
2. Frontier-level comparison of questions and prompts: oracle-frontier label (the candidate minimising
   geodesic agent-to-node plus node-to-nearest-goal), top-1 against chance 1/N over the candidates of a
   decision, Nadav's boxed prompt against the room-level percent and yes/no wordings, 4B and 8B.
   Images can be re-rendered from the recorded poses; his replay tool
   (`frontier_view_replay.py`) gives the views each node would be shown from a recording alone.
3. Measure how much the full frame history raises the number of frames that see a cluster, compared with
   my earlier window count (median 1 frame, half of the clusters seen by none).
4. Normalised or soft-normalised scores into RPT\* (only values below 1), after reading the solver.
5. Rescoring a node when a better view arrives (Nadav: probably not worth it, test later).
6. Add the glances to the Algorithm 1 figure.
7. Direct comparison to Nadav's full MP3D run needs `qwen2.5vl:7b` here (about 6 GB download, declined
   for now) or the same model on his side.

## Standing constraints

Push only when she says so; never use or enter credentials or the Matterport login; held-out scenes only
for final numbers; no tuning on val; no per-label thresholds or object-to-room dictionaries; tests before
code; comments at most 2 lines; HM3D frames are never uploaded to hosted artifacts; never kill her processes
(Ollama on 11435, label page on 8765; both were down after the reboot of 2026-11-24 and not restarted);
free the GPU when done (Ollama on 11434 is mine). The branch `daphna/objnav-arrival-spin` tracks Nadav's
branch locally: never push to his branch.
