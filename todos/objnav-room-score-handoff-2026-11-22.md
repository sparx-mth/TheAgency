# ObjectNav room score and frame capture: handoff, 2026-11-22

Branch `daphna/objnav-verify-diagnosis-rebased`. All code is in
`sparx_agency/tasks/planning/objnav_benchmark_runtime/room_dataset/` (tests in `tests/`).
Data and logs live on the desktop under `~` (not in git); paths below.

## Goal

Improve the object-search algorithm (figure: VLM-guided search with frontier clusters, a
warm-up scan, an informative-frame choice, a VLM score per cluster, then RPT\*). Questions
under study: how to choose informative frames, how to capture them with no extra actions,
how many, where to stand; whether rooms should be nodes; what to ask the VLM.
Metric: AUROC per target over rooms (chance of ranking a room that holds the target above
one that does not; 0.5 chance, 1.0 perfect), scene-clustered bootstrap, "hidden" = target in
none of the frames. Bar set in advance for a wording or rule: hidden-room AUROC >= 0.70 with
the lower end of the 95% interval above 0.60 (gate: gain over random with lower end above 0).

## Results so far (4B model unless stated; every number is on rooms, not on frontier clusters)

- **Planner's own room score and exit count** are no better than nearest-first
  (`report_planner_baseline.md`; the test has almost no headroom, 39 episodes).
- **Choosing frames** (sweep, planner buffer): no rule beats random. More views help up to
  about 3-6. Position (door, middle, far) and heading spread do not matter.
- **Entrance scans** of 3 and 6 views added nothing over the planner path; the even sweep
  beat the path by +0.098 but is not free.
- **Wordings** (`~/hm3d_prompts/report_prompts.md`, 255 rooms, 3 frames): the original
  percent wording is best or tied for 4B and 8B. "unseen" is worse, "nearby" ties, "keep"
  is worse (8B "keep" is below chance, probably the first-token yes/no reading). None clears
  the bar: hidden-room AUROC 0.633 (4B), 0.661 (8B).
- **Informative-frame gate** (a yes/no question per view, read from log-probabilities;
  `~/hm3d_gate/report_gate.md`, `report_gate_8b.md`): not shown at K=3 (+0.015 4B, +0.006 8B,
  intervals include 0). At K=1 it helps all rooms (+0.057 4B, +0.069 8B, intervals above 0)
  and hidden rooms in 4B (+0.083 [+0.008, +0.143]) but not 8B (+0.041, interval includes 0).
  At 3 or more views random choice is as good as any rule.
- **Spin in place at the arrival pose of each room** (`~/hm3d_spins/report_spins.md`, 178
  rooms in 52 scenes, 30-degree turns, original percent question on every view): all rooms
  0.606 (arrival view alone), 0.647 (3 turns), 0.642 (6), 0.641 (10). Hidden rooms 0.548,
  0.586, 0.613, 0.614. Three turns give most of the gain for all rooms; six turns for
  hidden rooms; ten add nothing for the room score. Extra actions per episode for a spin at
  each room entry: about 4 rooms x k turns, i.e. 12 / 24 / 40 actions (5% / 10% / 17% of
  the mean 238-step recorded episode).
- **Out-of-view AUROC stays at about 0.6-0.66** across models, wordings, frame choices and
  routes. It looks like a limit of what the VLM knows, not of the prompt or the capture.

## Frontier-level findings (the unit the algorithm actually scores)

- The recorded planner (`~/planner_runs/batch160`, `batch20`, `dev2_batch`; 299 recordings,
  no images saved) is a room-based LLM planner, not the figure's algorithm. It has the
  10-turn warm-up but not a scan on arrival, and its frontier clusters are rebuilt from the
  map at every action, with no link to the frame that created them.
- Clusters (`core/planning/exploration/room_costs.py:527`, `frontier_ranking.py`): frontier
  cells (free next to unknown) grouped by adjacency, at least 4 cells, split at corners of
  an unknown block, one goal cell per cluster, room by majority vote.
- A step is 0.25 m forward or a 30-degree turn, one frame per step.
- **Frames per cluster** (`~/hm3d_membership/`): 2,238 clusters at 424 picks. Map line of
  sight is too strict (low furniture blocks the top-down map but not the camera); facing
  only is too loose. Rendered depth on 80 sampled picks (443 clusters,
  `render_check.md`): median 1 frame sees a cluster in the window since the previous pick;
  50% of clusters are seen by none, 32% by 1-2, 16% by 3-5, 2% by 6 or more. So the
  problem is coverage, not choice between frames, and one frame per cluster (as in the
  figure) is typical, which is where the gate helps.
- Not yet split by after-warm-up picks versus later picks. The figure scans before every
  plan, so after-warm-up picks are the closer match.

## Design intent from the user (not yet tested end to end)

A warm-up at the start of the episode and at the start of every room: the same structure,
a general "perceive the area" step, not necessarily the same scan. The spin result above
suggests 3-6 turns suit the room score; a full 10 turns would be for covering exits in all
directions.

## Code added (local commits, pushed with this handoff)

`planner_scores.py`, `eval_planner_baseline.py` (planner baseline); `sweep_poses.py`,
`sweep_room_poses.py`, `sweep_rules.py`, `report_sweep.py`; `buffer_rules.py`,
`select_planner_buffers.py`, `report_buffers.py`; `select_entrance_scans.py`,
`report_same_rooms.py`; `mp3d_target_score.py` (styles, `--walk-method`, `--only-usable`);
`make_prompt_frames.py`, `report_prompts.py`; `frame_gate.py`, `report_gate.py`;
`frontier_membership.py`, `count_frontier_membership.py`, `report_membership.py`,
`check_membership_render.py`; `select_spin_scans.py`, `report_spins.py`.
Run the new tests with the `navdp` conda env (it has pytest), `PYTHONPATH` at the worktree.
Experiments run with `~/miniconda3/envs/objnav-habitat/bin/python`.

Scripts: `~/verify_diag/run_*.sh` (sweep, buffers, same_rooms, prompts, gate, gate_8b,
spins), logs next to them.

## Open items and next steps

1. Split the membership count by after-warm-up picks versus later picks, to see how many
   frames a cluster gets when a scan precedes the plan.
2. Frontier-level test: score the frames that see each cluster, aggregate per cluster
   (max, mean, all frames at once), and run the ordering test with distances and RPT\*.
   The label for "the target is in the space beyond the frontier" is not defined yet
   (default: same room as the frontier or the room it leads into).
3. Optional: mark the cluster in the image (project its goal cell, draw a marker, ask about
   the marked region). A pairwise "which of two rooms" wording is also unbuilt.
4. Rooms as nodes versus a flat graph (step 4 of the plan) comes after the above.
5. The 8B was run for wordings and the gate, not for the spin test.
6. Merge or rebase of the 2 laptop commits on `origin/daphna/detector-compare`
   (it is 32 commits behind and 2 ahead; no fast-forward). The detection and verification
   line (bbox check with 4B/8B on 1,616 boxes) from the laptop plan was not started.
7. Caveats: small n (99-255 rooms, small hidden subsets), ground-truth perception, held-out
   scenes untouched (final numbers only there), the "0 extra steps" claim holds only for
   planner-derived frames.

## Standing constraints

Push only when she says so; never use or enter credentials or the Matterport login;
held-out scenes only for final numbers; no tuning on val; no per-label thresholds or
object-to-room dictionaries; tests before code; comments at most 2 lines; HM3D frames are
never uploaded to hosted artifacts; never kill her processes (Ollama on 11435, label page
on 8765); free the GPU for Nadav when done (it is free now, Ollama on 11434 is mine and
holds no model).
