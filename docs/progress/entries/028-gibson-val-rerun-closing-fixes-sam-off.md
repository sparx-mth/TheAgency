# 028 - The full Gibson validation run again, with the closing fixes, SAM off (the A arm)

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** in-progress -- the run is going (`runs/gibson_val_20261012T141500Z`), detached, supervised
**Roadmap item:** ObjectNav -- the published Gibson val benchmark re-run with the toilet closing fixes, SAM off, then SAM on, and the two compared

## Goal
The published Gibson ObjectNav validation split (1,000 episodes, 5 scenes x 200, the protocol of entry 026)
run again to completion under the code that fixed the toilet closing -- the three tilt-less closing rules
(the along-ray association cylinder, the floor-cut sighting, the below-the-frame STOP), the standoff-keeping
rule, and the image-space pitch servo of entry 027 (inert under the Gibson action set, which has no LOOK
pair) -- with the policy's `segmentation` setting **off** (the box-depth pipeline, `SEGMENTATION=0`). This is
the A arm of an A/B: the B arm is the same run with Segment Anything reading the target's depth on its own
pixels (`SEGMENTATION=1`), in its own entry, and the comparison of the two against each other and against
the 2026-10-08 baseline (SR 0.653, SPL 0.366, DTG 1.59 m; toilet SR 0.288).

## Why
Entry 026's run put toilet at SR 0.288 (131 of 132 failures were 500-step timeouts), and the nine failure
recordings showed the toilets found and released on bottom-cut frames. The fixes were measured on nine slots
only (toilet 2/3 improved, the rest unchanged, one tv regression traced and fixed). Whether they move the
population-level numbers -- and whether the SAM mask depth moves them further -- needs the full split, run
the same way as the baseline so the three results are comparable.

## Steps
- [x] Launch environment reproduced from the baseline run's replay snapshot (`runs/gibson_val_20261008T120112Z/replay/*.environ`),
      not from the script's defaults, none of which hold on this machine: `HAB_PY=~/miniconda3/envs/objnav-habitat/bin/python`,
      `GIBSON_EPISODES_DIR=~/datasets/objectnav/objectnav/gibson/v1.1/val`, `GIBSON_SCENES_DIR=~/datasets/gibson/val/scenes`,
      `OLLAMA_MODE=native`, `LLM_BASE_URL=http://127.0.0.1:11435`; everything else default (seed 17, `GPU_PLAN=auto`,
      lean records), plus `SEGMENTATION=0` stated explicitly.
- [x] The orphaned detector and native Ollama of the last session (reparented to `systemd --user`) stopped first, so
      the launcher starts, owns and tears down its own services as the baseline launch did.
- [x] Preflight into the run's directory: 1,000 episodes selected (`full_split`), lean records, publishable,
      segmentation off (box depth), renderer the CUDA GPU, LLM + YOLO-World + Habitat placed on the 4090.
- [x] Launch detached per `BENCHMARK.md` 2.3: the launcher (`launcher.pid`), the monitor (5-minute ticks), and after
      60 s the supervisor with `PROGRESS_ENTRY` pointing at this entry (its default is entry 026) -- it resumes a dead
      run at most twice, and publishes `BENCHMARK_RESULTS.md`, `results/<run>/`, this entry and the CHANGELOG line
      to the branch when `summary.json` appears.
- [ ] When the run ends: read `FINAL_SUMMARY.txt` / `summary.json` / `comparison.md`; record SR, SPL, DTG here, per
      category beside the baseline (toilet first), and a LESSONS entry if anything other than the policy's quality
      explains a number.
- [ ] The B arm: the same launch with `SEGMENTATION=1` (needs `~/models/objnav/mobile_sam.pt`, absent on this machine
      today), its own entry, and the A/B/baseline comparison table.

## Open questions
- The image-space pitch servo and the whole-box STOP of entry 027 need the LOOK pair; the Gibson action set has
  none, so this run measures the tilt-less rules and the standoff-keeping rule only. A tilt-capable protocol run
  would measure the rest.

## Notes
- 2026-10-12: `supervise_benchmark.sh`'s publish commit hardcoded `Progress: 026-...` in its trailer whatever
  `PROGRESS_ENTRY` said; now derived from the entry it was given. Fixed before this run's supervisor started
  (bash reads a running script incrementally; the file must not change while one runs).

## Result
(when the run ends)

