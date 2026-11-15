# 030 - The full Gibson validation run again, with the search and closing fixes of 2026-10-13/14

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** done -- the first run (`runs/gibson_val_20261015T045442Z`) was ended at 43/1000 episodes (2026-10-15, tv
closing, see the notes); the full run was relaunched on `f628a476` (entry 031's code) as
`runs/gibson_val_20261016_small_llm` and finished 2026-10-18. Result below.
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
- [x] When the run ends: read `FINAL_SUMMARY.txt` / `summary.json` / `comparison.md`; record SR, SPL, DTG here beside
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
- 2026-10-15: stopped at 43 episodes. The first 12 Collierville tv episodes went 3/12, with five wrong STOPs inside
  50 actions (000004, 000011, 000014, 000015, 000018) -- the same five as in the 2026-10-12 run, at the same step
  counts and distances -- so the failure is deterministic and not worth another 34 hours before it is understood.
- 2026-10-15: three recorded Collierville tv episodes, `runs/gibson_Collierville_tv_20261015T055552Z`
  (`run_benchmark.sh --scene Collierville --category tv --record --full-records --episode-ids ...`; H.264 MP4 with the
  HUD, `steps.jsonl` per action, key frames under `recordings/<key>/key_frames/`). All three reproduced their
  full-run outcome exactly. What the step records say:
  - **000018** (`recordings/991eeffec5a5`, wrong STOP at 14 actions, DTG 3.33 m): during the warm-up turn YOLO-World
    reads a **0.33 x 0.18 m** landscape rectangle 1.0 m away as `tv` 0.65 / `television` 0.44; the takeover verifies
    it in one step, inspects for four, advances one and STOPs on `tv` 0.63 ("fresh terminal target confirmation").
    A television-shaped object a third of a metre wide.
  - **000014** (`recordings/b10e28b0975a`, wrong STOP at 50 actions, DTG 5.88 m): `television` 0.60-0.77 on a
    **0.18 x 0.39 m portrait** rectangle (aspect 0.45-0.53); approach from 3.6 m to 1.0 m, then the inspection
    oscillated TURN_LEFT / TURN_RIGHT for 24 actions (box 0.42 / 0.29 on alternate frames -- the dead-band alignment
    of `211cc011` did not catch it), and `stop_on_exhausted_inspection` STOPped on the 0.42 frame.
  - **000009** (`recordings/873d8cf1d32d`, 500-step timeout): the opposite -- a tv box in 268 of 500 frames, 106 of
    them at >= 0.50, peak **0.90**; seven takeovers, every one released: three approached from ~4.5 m and released at
    ~3 m with "no safe path to the target 3.x m away", one reached 0.8 m and "the inspection from 0.8 m saw nothing".
    The 2026-10-12 run ended this episode 0.12 m from the goal without a STOP. Consistent with a wall-mounted,
    elevated television: its anchor lands where the planner has no floor, and from under it the level camera loses it.
  - Reading, revised with the ground truth (entry 031): two closing problems, neither the detector's confidence.
    (1) In 000018 the box WAS on the real television (same bearing, 113 px wide at 3.7 m = 1.1 m) and the range was
    read off a rail at camera height inside the box, 2.4 m in front of it -- a measurement fault, fixed by the box
    pipeline's occluder test. (2) In 000009 the memory was 0.02-0.14 m from the television and the 0.70 m standoff
    ring had no floor around a 1.5 m object on its cabinet -- a standoff fault, fixed by the widened ring, the kept
    stale goal and the closest-approach STOP. Both episodes now succeed (SR 1 / SPL 0.81 and 1.00); 000014 is a
    genuine detector false positive and is unchanged. Relaunch this run on the code of entry 031.

## Result
**SR 0.792 (95 % CI 0.766-0.816), SPL 0.464, DTG 0.94 m** over 1000/1000 published episodes on `f628a476`, protocol
`gibson-semexp-v1.1-val/2`, both LLM roles `qwen2.5:3b-instruct`, SAM off, lean records, seed 17, laptop RTX 5070 (8 GB);
against the 2026-10-12 run (3B/14B pair on a 4090, protocol /1): SR 0.713, SPL 0.409, DTG 1.27 m. Per-scene and
per-category tables: `gibson/BENCHMARK_RESULTS.md`, section `gibson_val_20261016_small_llm`. Weakest categories: bed
(SR 0.528), tv (0.643).

Provenance, to read before comparing:
- The YOLO-World detector died silently after episode 530 (Darden/000129); the run kept going on its
  frontier-only fallback, so episodes 531-573 (43, `detector_failures` > 0 in their rows) were not valid. Before the
  resume they were moved out of `episodes.jsonl`, `evaluation_diagnostics.jsonl` and `episodes.csv` (originals kept
  beside the run in `gibson_val_20261016_small_llm.quarantine_detector_down`) and re-run. All 1000 final rows have zero
  detector failures; one agent error.
- The run died three times for reasons outside the policy (machine crash 2026-10-17, then a gnome-shell SIGSEGV
  that ended the desktop session at 00:17Z on 2026-10-18, then a third death on Wiconisco/000072); the supervisor
  and the operator resumed it. The CPU package sat at 90-100 C under the run; after that it ran under a systemd scope
  (24 GB RAM, 6 CPU cores, balanced power profile, Ollama on 1.5 cores), RAM peak 64 %, VRAM 64 %.
- Hardware and model differ from the 2026-10-12 baseline (3B for both roles, laptop GPU, detector on GPU); the
  difference to it is the fixes plus that change, not the fixes alone.

## Result (2026-10-19)
Finished 2026-10-19T12:40:17Z after 0 resume(s): **SR 0.788, SPL 0.465, DTG 0.96 m** over 1000/1000 published episodes (full split: true; agent errors 0). Per-scene and per-category tables, the published-row comparison and the provenance: `sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md` (section `gibson_val_20261018T105339Z`); the data under `sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/results/gibson_val_20261018T105339Z/`. Published from the detached run by `supervise_benchmark.sh`.
