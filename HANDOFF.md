# Handoff — HM3D ObjectNav benchmark

## 2026-10-29 — The ground-truth oracle ran the full v1 val split with the 3B model in both LLM roles

Branch: `feat/objnav-habitat-hm3d-nadav`, tree at `50bbd877`. The HM3D-Semantics v0.1 val annotations are now
installed (`hm3d.assets --version v1 --check --semantic`: `semantic_ready: true`), which unblocked the run below.

**What ran.** `runs/gt_oracle_2000ep_3b_20261028T152707Z`: HM3D v1 val, 2,000 episodes / 20 scenes,
`--perception ground-truth` (habitat-sim's semantic sensor in place of YOLO-World and SAM, exact instance masks),
RPT agent otherwise unchanged, **`qwen2.5:3b-instruct` for both the room labels and the node oracle**
(`LLM_MODEL=LLM_REASONING_MODEL=qwen2.5:3b-instruct`; the 14B default was deliberately replaced to measure the
small model). 20 scenes at once (`--jobs max` on an idle card). `complete: true`, exit 0, 15:27 → 10:11 UTC
(~16.7 h, ~0.6 episode/min across the 20 workers). Non-publishable by construction (oracle perception).

**Results** (`campaign_summary.json`, `campaign_metrics.csv`, `<scene>/comparison.md`):

| SR | SPL | SoftSPL | DTG | mean steps | unreachable ends |
|---|---|---|---|---|---|
| **0.691** (CI 0.670–0.711) | 0.362 (CI 0.348–0.376) | 0.372 | 3.18 m | 243 | 0 |

Per-scene SR spans 0.24 (`00820-mL8ThkuaVTM`, SPL 0.10) and 0.27 (`00878-XB4GS9ShBRE`) at the bottom to 0.97–0.98
(`00829-QaLdnwvtxbs`, `00813-svBbv1Pavdk`) at the top; the two worst scenes are the first place to read
`evaluation_diagnostics.jsonl` (`ground_truth.scene_report`, see the 2026-10-26 entry) since perception is ideal
there and the misses are the search's.

**Two launch lessons, both environmental** (nothing in the harness changed):

- `DETECT_PY` defaults to `~/.venvs/objnav-detector/bin/python`, which no longer exists on this machine; the
  conda env `detector_bench` has ultralytics and the detection server imports cleanly. Pass
  `DETECT_PY=~/miniconda3/envs/detector_bench/bin/python` (the launcher needs it even for `--perception
  ground-truth`, and `--status` needs `HAB_PY`).
- With no Ollama resident, the GPU plan put the LLM on the CPU (its default 30 GB Habitat reserve exceeded the
  card), and 20 workers on a CPU 3B model saw 30–100 s per call: ~2 episodes in 10 min. Pinning it back on the
  card fixed it (1.5–11 s per call): `GPU_PLAN_LLM=gpu LLM_VRAM_MIB=4096 HABITAT_VRAM_MIB=700
  OLLAMA_NUM_PARALLEL=10` -- 4.9 GB for Ollama plus 20 renderers came to ~18 GB of the 24 GB card.

The exact command that produced the run:

```bash
cd ~/GIT/TheAgency
HAB_PY=~/miniconda3/envs/objnav-habitat/bin/python DETECT_PY=~/miniconda3/envs/detector_bench/bin/python \
LLM_BASE_URL=http://127.0.0.1:11435 OLLAMA_MODE=native \
LLM_MODEL=qwen2.5:3b-instruct LLM_REASONING_MODEL=qwen2.5:3b-instruct \
OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_NUM_PARALLEL=10 GPU_PLAN_LLM=gpu LLM_VRAM_MIB=4096 HABITAT_VRAM_MIB=700 \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/hm3d/run_benchmark.sh --version v1 --perception ground-truth \
  --jobs max --output runs/gt_oracle_2000ep_3b_20261028T152707Z
```

Two earlier directories under `runs/` from the same session are dead starts and can be deleted:
`hm3d_v1_val_3b_20261028T150348Z` (YOLO perception, stopped after minutes by request) and
`gt_oracle_2000ep_3b_20261028T151516Z` (the CPU-LLM start, 2 episodes).

---

## 2026-10-26 — The perception upper bound (ground-truth oracle) is built; the 2,000-episode run waits on one download

Branch: `feat/objnav-habitat-hm3d-nadav`. Everything below is in the working tree (uncommitted at the time of
writing; see `git status`).

**What was asked.** Before comparing detectors/segmenters, run the HM3D-v1 val split (2,000 episodes, 20 scenes)
with *ideal* perception -- boxes, labels and instance masks from the simulator's annotations instead of
YOLO-World and SAM -- with every navigation layer unchanged, in parallel, into `runs/gt_oracle_2000ep_<stamp>`.

**What exists now** (`sparx_agency/tasks/planning/objnav_benchmark_runtime/`, operator's manual in
`hm3d/README.md` section 3.5, design notes in `README.md` "Ideal perception from the simulator"):

- `habitat/simulator.py`: `HabitatRGBDSimulator(semantic=True, scene_dataset_config=...)` renders habitat-sim's
  semantic sensor (HM3D-Semantics instance ids) beside RGB-D into `last_semantic`; `semantic_objects()` is the
  scene's `{instance id: raw category}`; a scene that loads no annotations is refused.
- `ground_truth/`: `GroundTruthSource` (env -> oracle hand-off, strict frame identity), `GroundTruthDetector` /
  `GroundTruthSegmenter` (the method's own `Detector`/`Segmenter` contracts), `hm3d_categories.py` (publisher's
  goal ids first, exact raw-name table second), `semantic_frame.py` (boxes/masks from an id frame).
- `hm3d/env.py` publishes each frame under `ground_truth=`; `hm3d/dataset.py` has `goal_instances`,
  `semantic_assets`, `annotated_scene_dataset_config`; `hm3d/run.py` has `--perception ground-truth` with a
  file-by-file annotation preflight; `policy_factory.build_policy(..., ground_truth=)` builds the pair, turns
  `segmentation` on, marks the run `publishable: false`.
- `hm3d/run_benchmark.sh --perception ground-truth --jobs max` (no YOLO/SAM started, output
  `runs/gt_oracle_2000ep_<stamp>`); `benchmark_services.sh PERCEPTION`; `scene_pool.max_workers` behind `--jobs max`.
- Tests: `tests/test_ground_truth_perception.py` (32). Suite status unchanged otherwise (the 20 pre-existing
  failures are `skfmm` missing in `.venv` and the uninstalled v2 split).

**Verified end to end** on Matterport's *public* HM3D example scene (`00861-GLAQ4DNUx5U`, fetched to
`/tmp/hm3d_example`, synthetic one-episode release in `/tmp/hm3d_example_run`; both disposable, both re-creatable
without a licence): habitat-sim 0.2.4 loads 908 instances through the annotated config, the rendered ids reproduce
`semantic.txt` exactly, and a full RPT episode ran through `hm3d.run --perception ground-truth` (320 oracle frames,
0 lookup misses, 145 labelled instances, 336 door detections -> 8 door landmarks, 10 rooms via 26 LLM calls,
closing to STOP; exit 0) and a second through the launcher's **campaign path**
(`run_benchmark.sh --perception ground-truth --scenes 00861-GLAQ4DNUx5U --limit 1 --jobs 2`: services, preflight,
`campaign.py` -> `scene_pool`, child, aggregate; 460 frames, STOP at 0.28 m from a view point after 459 actions,
`campaign_summary.json` with `complete: false` for the subset; exit 0). The synthetic episode's scores mean nothing
(its view points were a handful of random samples); the plumbing does.

**The one blocker: the HM3D-Semantics v0.1 annotations for the val scenes are not on this machine**
(`hm3d-val-semantic-annots-v0.1.tar.gz` + `hm3d-val-semantic-configs-v0.1.tar`, ~1.4 GB, Matterport API token --
the same licence as the scenes; nothing in this repo may hold the token). `hm3d.assets --version v1 --check
--semantic` reports 41 files missing (the config + 20 x {`.semantic.glb`, `.semantic.txt`}); the preflight
refuses the run by name. To unblock, the licence holder runs, in `objnav-habitat`:

```bash
python -m habitat_sim.utils.datasets_download --username <token-id> --password <token-secret> \
    --uids hm3d_val_semantic_annots_v0.1 hm3d_val_semantic_configs_v0.1 --data-path ~/datasets/
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.assets --version v1 --check --semantic
```

**Then launch** (the operator's environment of the running sweep, plus the oracle switch):

```bash
cd ~/GIT/TheAgency
HAB_PY=~/miniconda3/envs/objnav-habitat/bin/python LLM_BASE_URL=http://127.0.0.1:11435 OLLAMA_MODE=native \
OLLAMA_NUM_PARALLEL=2 OLLAMA_CONTEXT_LENGTH=4096 HABITAT_VRAM_MIB=400 \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/hm3d/run_benchmark.sh --version v1 --perception ground-truth --jobs max
sparx_agency/tasks/planning/objnav_benchmark_runtime/hm3d/run_benchmark.sh --status runs/gt_oracle_2000ep_<stamp>
```

`--jobs max` gave 10 while the YOLO sweep held the card (4.2 GB free at 400 MiB a renderer); on an idle card it
is 20 (one child per scene). **The YOLO baseline sweep `runs/hm3d_v1_val_j8p2_20261026` was running throughout**
(8 jobs, ~484 s/episode, ETA ~33 h from 09:20 UTC); the oracle run shares its Ollama, so the two slow each other
on the LLM -- starting the oracle after it finishes is the cleaner measurement. Starting it beside the sweep is
safe only because of the `benchmark_services.sh` fix described in `LESSONS.md` 2026-10-26 (a resident Ollama is
no longer re-placed; before it, the new run would have moved the shared LLM to the CPU).

**What to look at in the results**: `campaign_summary.json` (SR/SPL/SoftSPL/DTG, `complete`), each scene's
`evaluation_diagnostics.jsonl` `ground_truth.scene_report` (`reported_by_label`, `unmapped_raw_names`,
`named_for_a_goal_category_but_not_a_goal`) -- a frequent unmapped raw name that should be a context word is a
one-line addition to `ground_truth/hm3d_categories.py` (its hash is in the frozen config, so note the change).

---

## Handoff — HM3D ObjectNav benchmark, 2026-10-01

Branch: `feat/objnav-habitat-hm3d-daphna` (tracks `origin/feat/objnav-habitat-hm3d-nadav`).
Everything below is committed on this branch unless marked otherwise.

## The actual goal

Moshe asked for a larger, more diverse ObjectNav evaluation than the 15-building
Gibson pilot before drawing conclusions. Nadav's `objnav_benchmark_runtime/hm3d/`
package is that harness — the team's real method (scene graph → LLM room prior →
RPT\* room order → weighted A\*), run on HM3D-v2 (36 scenes, 1000 episodes).

Previous session (2026-09-29) found and partially fixed an exploration-freeze bug
(see `LESSONS.md` 2026-09-28/09-22 entries). Today's session merged in Nadav's
parallel work on the same bug class from `feat/objnav-habitat-gibson-nadav`.

## What happened today

1. **Merged `origin/feat/objnav-habitat-gibson-nadav` into this branch**
   (`a6fde50c`, plus a follow-up fix commit `f223becd` — see "Mistake" below).
   Nadav's branch had moved well past the fork point (`299c20a1`, 2026-09-28)
   with two more commits: `e217f8f1` (ground-truth stairs rewrite: seen-only
   detection, centreline traversal, never skip a vertex, withhold the
   just-climbed staircase from RPT\*) and `48f8f395` (today, 2026-10-01: one-time
   room peeks, floor-departure guard fixes, target-closing rewrite).
2. **Resolved 5 real conflicts**: `CHANGELOG.md`, `LESSONS.md`,
   `objnav_benchmark_runtime/README.md` (additive, kept both sides),
   `gibson/stair_connectors.py` (structural — see below), `habitat/simulator.py`
   (two unrelated new methods added at the same spot, kept both).
3. **Structural conflict**: this branch had already refactored
   `gibson/stair_connectors.py` into a thin re-export shim pointing at
   `habitat/stair_connectors.py` (done in the 2026-09-28 shared-core merge,
   `5a3d270f`), while Nadav's branch never did that rename and kept developing
   the real 650-line implementation directly in the old `gibson/` path. Resolved
   by keeping our shim and porting Nadav's rewritten implementation into
   `habitat/stair_connectors.py`, then fixing the shim's explicit re-export list
   (`_nearest_level`/`_polyline` no longer exist under the new design;
   nothing outside the shim imported them).
4. **Fixed 4 of Nadav's new tests** (`test_target_closing.py`): they called
   `HabitatRGBDSimulator(...)` without the `height_m` keyword this branch already
   requires (an HM3D-side requirement from the earlier shared-core merge that his
   branch never had). Added `height_m=0.88` to match the convention used
   elsewhere in the test suite.
5. **Verified no regressions**: full `tests/` + `core/planning/objnav` +
   `objnav_benchmark` + `core/mapping/topology` suite — 2093 passed, same 10
   pre-existing failures as on the unmerged branch (all `skfmm` not installed in
   `.venv`, confirmed by running the identical tests in a throwaway worktree at
   the pre-merge commit — not a merge regression).
6. **Mistake caught and fixed (`f223becd`)**: when first resolving the
   `gibson/stair_connectors.py` conflict, copied Nadav's new implementation into
   `habitat/stair_connectors.py` on disk but never `git add`ed it before
   committing the merge — `a6fde50c` landed the shim pointing at the *old*
   237-line implementation. Tests still passed because pytest reads the
   (unstaged) working tree, not the commit. Caught by checking `git status`
   before writing this handoff. **Lesson for next time: after resolving a
   conflict by replacing a file's content outright (not editing the conflicted
   copy in place), always `git add` it explicitly — don't rely on `git add -A`
   with a narrow pathspec to catch it.**

## Verification run: the exploration-freeze bug is IMPROVED, not fixed

Re-ran the known reproduction case, `00800-TEEsavR23oF/000000`
(`~/objnav_benchmark/hm3d_v2/postmerge_test_00800/`), same seed (17), same
`--explorer frontier`. **Used `qwen2.5:3b-instruct` for both LLM routes**
(including `LLM_REASONING_MODEL`, which defaults to `qwen2.5:14b-instruct` and
wasn't pulled yet at the time) — treat this run as a directional check, not a
clean/scorable result. `qwen2.5:14b-instruct` is now pulled and ready for a
proper re-run.

- **Before** (pre-merge, `stairfix_test_00800_v2`): 375 of 500 steps frozen in
  one continuous block.
- **After** (post-merge): ~236 of 500 steps frozen, split across four shorter
  windows instead of one. Real improvement, not noise.
- **Still broken — the episode's last 101 steps are unbroken `RETREAT`**, never
  exits for the rest of the episode. This matches the gap Nadav documented
  himself in `LESSONS.md` (2026-09-28 entry, "Status" note): a retreat stuck
  more than `floor_match_m` from every known storey never satisfies
  `FloorAtlas._settled()`'s plateau test, and the real fix for that
  (`FloorAtlas.cancel_transition`, commit `4adf14c0`) was **never pushed to any
  branch** — confirmed again today (`git cat-file -e 4adf14c0` fails in this
  repo).
- **Earlier freeze (steps 7–141)** centers on `doorway_peek_approach` near a
  stair head — looks related to *today's* newest lesson ("a peek route across a
  stair head becomes a coverage-veto rotation loop", `48f8f395`), whose own
  validation note says only "a CPU-only native-NavMesh probe... no new
  end-to-end benchmark was run." Plausibly not fully shaken out in a real run
  yet; could also be an artifact of using the 3B model for room reasoning
  instead of 14B.

## Environment (unchanged from last session, all local to this machine)

- `objnav-habitat` conda env (python 3.9, `habitat-sim==0.2.4`).
- `detector_bench` conda env — YOLO-World-X, GPU (`cuda:0`), `127.0.0.1:18092`.
  **Already running**; health-checked today, serving 29 classes correctly.
- Ollama, port **11435**, `~/models/objnav/ollama`. **Already running.**
  `qwen2.5:3b-instruct` was there already; `qwen2.5:14b-instruct` (~9 GB, Q4)
  pulled today and ready.
- Data: `~/datasets/objectnav/hm3d/v2/val/` + `~/datasets/scene_datasets/hm3d_v0.2/val/`
  (36/36 scenes present, verified today).
- Standard run env vars:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate objnav-habitat
  cd ~/GIT/TheAgency
  export LLM_BACKEND=ollama LLM_BASE_URL=http://127.0.0.1:11435 LLM_MODEL=qwen2.5:3b-instruct LLM_TIMEOUT_S=120
  export HM3D_EPISODES_DIR="$HOME/datasets/objectnav/hm3d/v2/val"
  export HM3D_SCENES_DIR="$HOME/datasets/scene_datasets"
  ```
  Add `export LLM_REASONING_MODEL=qwen2.5:3b-instruct` only for a quick check;
  omit it (use the real 14B default) for anything meant to be a real signal.

## Decision (confirmed with Moshe, 2026-10-04): priority for next week

**Do not keep chasing the `RETREAT` bug** unless it turns out to be a tiny,
contained fix (worth a quick look first, see below) — it is **not** next
week's task. The actual priority is to **bring FALCON into the algorithm**,
with two specific design asks:

1. **Multi-resolution FALCON — confirmed new work, not in the paper.**
   Checked the actual FALCON paper (arXiv 2407.00577): its own "hierarchical"
   structure is two levels of planning *abstraction* at one fixed resolution —
   a coarse-cell connectivity decomposition produces global coverage-path
   guidance, and local frontier/viewpoint optimization refines *within* that
   guidance. It is **not** multiple grid resolutions. What we actually want —
   literally re-running the grid decomposition at two different `cell_size_m`
   values, coarse pass first, then a finer pass — is confirmed (Moshe,
   2026-10-04) as an intentional extension beyond the paper, not a
   misunderstanding of it. Don't let `BOUNDED_FALCON.md`'s "hierarchical" talk
   create false confidence that this already exists — it doesn't; see below.
2. **2.5D, not 3D** — already true of the module that would be wired in (see
   below); this part needs no new work, only confirming it's the right module.

### What already exists (found 2026-10-01, not yet validated for this)

- `core/planning/exploration/falcon/` is **already** "the FALCON 2D/2.5D
  adaptation (not the ROS/UAV binary)" — its own module docstring says so
  explicitly (`params.py:1`). The real 3D, ROS-based FALCON lives elsewhere
  entirely: `TheAgency_moshe/sparx_agency/tasks/planning/falcon/adapter/`
  (drone/sim adapter nodes) and `TheAgency/sparx_agency/tasks/planning/
  falcon_pegasus/` (viz/memwatch, a different thing again). **Don't confuse
  the three** — "insert FALCON" almost certainly means wiring up the
  2D/2.5D planar one that's already in `methods/falcon_policy.py` and
  already reachable via `hm3d.run --explorer falcon`, not porting the ROS one.
- **No multi-resolution exists yet.** `FalconParams` (`falcon/params.py`) has
  one `cell_size_m` (4.0) and one `scope_radius_m` (4.0) — a single fixed
  decomposition. Building a coarse-then-fine pass is new work, not a toggle.
- FALCON was only ever validated on **Gibson**, not HM3D (`gibson/
  FALCON_HANDOFF.md`, `FALCON_RESULTS.md`), with a mixed, undecided result
  (frontier 2/2 vs FALCON 1/2 on the small development pair) — "keep frontier
  default" was the explicit conclusion. Treat that as the baseline to beat,
  not as proof FALCON already works here.

### Recommended order of work next session

1. Quick triage only: if the `RETREAT`-never-exits gap (see above) looks like
   a genuinely small, contained fix, take it — otherwise leave it and move on.
2. Confirm which FALCON module is meant is already settled: the 2D/2.5D
   `core/planning/exploration/falcon/` + `methods/falcon_*.py` one, reachable
   via `hm3d.run --explorer falcon` — not the ROS/UAV one under
   `TheAgency_moshe/.../falcon/adapter/`, nor `falcon_pegasus/`.
3. Design and build the actual coarse-then-fine multi-resolution pass for
   `FalconPlanner` (`falcon/planner.py`) / `FalconParams` (`falcon/params.py`):
   run the connectivity-aware decomposition at a coarse `cell_size_m` first for
   a high-level plan, then again at a finer `cell_size_m` for local refinement.
   This is new design work with no existing precedent in the paper or the
   current port — don't look for a flag to flip.
4. Re-run `00800-TEEsavR23oF/000000` with the proper `qwen2.5:14b-instruct`
   reasoning model (now pulled) at some point for a clean signal on where the
   frontier baseline currently stands — useful context before comparing a new
   FALCON variant against it.
