# 030 - The full Gibson validation run again, with the search and closing fixes of 2026-10-13/14

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** in-progress -- the run is going (`runs/gibson_val_20261015T045442Z`), detached, supervised
**Roadmap item:** ObjectNav -- the published Gibson val benchmark re-run on the code that fixed the toilet
campaign's search and closing sinks, compared against the 2026-10-12 run

## Goal
The published Gibson ObjectNav validation split (1,000 episodes, 5 scenes x 200, the protocol of entry 026) run
again to completion on HEAD `34e4bf49`, i.e. everything committed after the 2026-10-12 run's `709e3512`:

- `0fbc8950` -- goal-filtered runs, seeded start sampling, GPU-plan pins (tooling; no policy change).
- `4c217a47` -- rooms explored by frontier (`visit="explore"` the loop's default), every look-around cut to the
  camera's circle (ten 30-degree turns, not twelve), the reworked glance scheduler, openings read off the live map
  and folded into their room, the pocket rule narrowed, **and the Gibson protocol given Habitat's LOOK pair**
  (`camera_tilt`, protocol id `gibson-semexp-v1.1-val/2`, pitch -30..60 degrees) so the box-driven pitch servo of
  entry 027 fires on the approach.
- `211cc011` -- the toilet campaign's search and closing sinks: re-sighting exemptions, depthless glimpses that
  carry the chain, the one-step inspection and the dead-band alignment, the standoff ring planned through unknown,
  the trusted-confidence back-off skip, landmark nodes, the oracle's second pass gated on coverage, exits valued
  by the floor they reveal, the failure cooldown, the detour-factor squeeze, the directional stair exclusion, the
  escape probe, the blocked-clock reset.
- `34e4bf49` -- the STOP overshoot (decide, then walk the margin: one step, a conditional second), the pitch
  hysteresis, the back-off that refuses a spot behind a wall, shadowless classes and the shadow exemption, the
  step-through doorway peek, provisional exit probabilities, sticky repeated bumps, the 0.30 m spawn-floor drop
  threshold, the 8 % floor-cut box.

Same launch as entry 028 otherwise: `SEGMENTATION=0` (box depth), lean records, seed 17, `GPU_PLAN=auto`, the
`qwen2.5:3b-instruct` / `qwen2.5:14b-instruct` pair on the 4090, native Ollama on 11435.

## Why
The fixes above were measured on the 10-episode toilet campaigns only (`runs/benchmark_toilet_10episodes_*`:
SR 0.80/SPL 0.366 -> 0.90/0.456 on the same ten slots) and on single recordings. Whether they move the
population-level numbers -- and whether the LOOK pair, which the 2026-10-12 run did not have, costs SPL on the
approach -- needs the full split, run the same way, so the result is comparable with `gibson_val_20261012T141500Z`
(SR 0.713 [0.684, 0.740], SPL 0.409 [0.385, 0.433], DTG 1.27 m; toilet SR 0.514, bed SR 0.504, tv SR 0.557; 98 of
the 101 toilet failures were 500-step timeouts) and the 2026-10-08 baseline (SR 0.653, SPL 0.366, DTG 1.59 m).

## Steps
- [x] The 2026-10-12 run analysed for the comparison: overall, per scene, per category, scene x category, and the
      failure anatomy (`runs/gibson_val_20261012T141500Z/analysis.md`, beside the run's own `summary.json`).
- [x] Tests on HEAD before spending the GPU: the runtime README's suite under `.venv` (2478 pass; 13 fail only on
      `skfmm` missing from `.venv`), and those 13 modules again under the Habitat interpreter through a throwaway
      `/tmp` venv (73 pass) -- the Habitat env itself was not modified.
- [x] Launch environment as entry 028's: `HAB_PY=~/miniconda3/envs/objnav-habitat/bin/python`,
      `GIBSON_EPISODES_DIR=~/datasets/objectnav/objectnav/gibson/v1.1/val`, `GIBSON_SCENES_DIR=~/datasets/gibson/val/scenes`,
      `OLLAMA_MODE=native`, `LLM_BASE_URL=http://127.0.0.1:11435`, `SEGMENTATION=0`; a clean tree at `34e4bf49`.
- [x] Launch detached per `BENCHMARK.md` 2.3 with `launch_detached.sh` and `PROGRESS_ENTRY` pointing at this entry:
      the launcher (`launcher.pid`), the monitor (5-minute ticks), the supervisor once the run is past its preflight --
      it resumes a dead run at most twice and publishes `BENCHMARK_RESULTS.md`, `results/<run>/`, this entry and the
      CHANGELOG line to the branch when `summary.json` appears.
- [ ] While it runs: no edit to any `.py` under `sparx_agency/` on this checkout (the source hash is frozen into the
      run; a resume after an edit needs `RESUME_ARGS="--resume-accepting-changes source_sha256"`), and this entry
      committed and pushed before the run ends so the supervisor's worktree can close it.
- [ ] When the run ends: read `FINAL_SUMMARY.txt` / `summary.json` / `comparison.md`; record SR, SPL, DTG here beside
      the 2026-10-12 run per category (toilet, bed, tv first) and per scene (Wiconisco couch 0.500 and Corozal bed
      0.343 were the weakest cells), run the same analysis as the step above for the scene x category delta, and a
      LESSONS entry if anything other than the policy's quality explains a number.

## Open questions
- Protocol `/2` (with the LOOK pair) against `/1`: the pitch servo may add actions on every approach; SPL per
  category against the 2026-10-12 run tells whether the approach got dearer where SR did not move.
- Bed (SR 0.504, DTG 2.89 m, 28 of 61 failures a wrong STOP) was untouched by the toilet work; the run says whether
  the search changes moved it at all.

## Notes
- 2026-10-15: the previous run's `.gibson_val_current` pointer was stale (it still named the 2026-10-08 run); it
  now names this run.

## Result
(when the run ends)

