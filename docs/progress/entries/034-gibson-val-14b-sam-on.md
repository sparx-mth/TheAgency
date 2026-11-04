# 034 - The full Gibson validation run, 14B node oracle, SAM on (the B arm)

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** done -- `runs/gibson_val_20261018_14b_sam`, finished 2026-10-19, no resumes; result below
**Roadmap item:** ObjectNav -- the published Gibson val benchmark with Segment Anything reading the target's depth,
the `qwen2.5:3b-instruct` / `qwen2.5:14b-instruct` LLM pair, compared against the SAM-off runs

## Goal
The published Gibson ObjectNav validation split (1,000 episodes, 5 scenes x 200, protocol
`gibson-semexp-v1.1-val/2`, the same split and scoring as the SemExp / PONI / LFG / OSG-Nav rows in
`comparison.md`) run to completion on HEAD `59783f62` with the policy's `segmentation` setting **on**
(`SEGMENTATION=1`: MobileSAM prompted with the target's boxes, depth read on the object's own pixels) and the
3B/14B LLM pair (3B for room labels, 14B for the node oracle) on the 4090. This is the B arm entry 028 called
for; entry 030's `gibson_val_20261016_small_llm` (3B in both roles, SAM off) is the latest A arm.

## Why
Every full-split result so far was measured with box depth. The SAM pipeline was only compared on five-scene
campaigns (BENCHMARK.md 3.4). Whether the mask depth moves the population-level numbers -- and in which
categories -- needs the full split, run the same way as the previous runs.

## Steps
- [x] Preflight with the launch environment (`run_benchmark.sh --preflight`): 1000 episodes selected, FULL published
      split, publishable, lean records, LLM 3B labels + 14B oracle, detector `yolo_world` on cuda:0, renderer cuda gpu 0,
      segmentation ON (`mobile_sam.pt` on cpu at 18096).
- [x] Launch environment as entries 028/030: `HAB_PY=~/miniconda3/envs/objnav-habitat/bin/python`,
      `DETECT_PY=~/miniconda3/envs/detector_bench/bin/python`,
      `GIBSON_EPISODES_DIR=~/datasets/objectnav/objectnav/gibson/v1.1/val`, `GIBSON_SCENES_DIR=~/datasets/gibson/val/scenes`,
      `OLLAMA_MODE=native`, `SEGMENTATION=1`; defaults otherwise (seed 17, `GPU_PLAN=auto`, 14B oracle).
      MobileSAM provisioned into `~/models/objnav/mobile_sam.pt` per BENCHMARK.md 1.5.
- [x] Launched detached per BENCHMARK.md 2.3 with `launch_detached.sh`, `PROGRESS_ENTRY` pointing at this entry and
      `SUPERVISOR_ARGS=--dry-run`: the supervisor resumes a dead run at most twice and builds the result commit at the
      end without pushing -- publishing is the operator's step.
- [x] While it runs: no edit to any `.py` under `sparx_agency/` on this checkout.
- [x] When the run ends: read `FINAL_SUMMARY.txt` / `summary.json` / `comparison.md`; record SR, SPL, DTG here per
      category and per scene beside entry 030's result, and publish.

## Open questions
- Whether the SAM depth helps the categories where a foreground object sits in the box (couch, bed) without costing
  the small ones (tv) where a mask can fail and the target is then not read at all.

## Notes
- Written into the checkout by hand with the supervisor's `write_results`, since the supervisor ran with `--dry-run`.

## Result (2026-10-19)
Finished 2026-10-19T16:30:02Z after 0 resume(s): **SR 0.784, SPL 0.480, DTG 1.00 m** over 1000/1000 published episodes (full split: true; agent errors 0). Per-scene and per-category tables, the published-row comparison and the provenance: `sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md` (section `gibson_val_20261018_14b_sam`); the data under `sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/results/gibson_val_20261018_14b_sam/`. Published from the detached run by `supervise_benchmark.sh`.

Against entry 030 (`gibson_val_20261016_small_llm`: 3B in both roles, SAM off, laptop): SR 0.792 -> 0.784 (inside the
intervals), SPL 0.464 -> 0.480, DTG 0.94 -> 1.00 m. Per scene: Collierville 0.765, Corozal 0.740, Darden 0.840,
Markleeville 0.760, Wiconisco 0.815. Per category: chair 0.932, toilet 0.865, potted plant 0.819, couch 0.782,
tv 0.543, bed 0.520; terminations 946 STOP / 54 step-limit, no detector failures. Two things differ from entry 030 at
once (the 14B oracle and SAM), so the delta is not SAM's alone; bed and tv remain the weakest categories.
