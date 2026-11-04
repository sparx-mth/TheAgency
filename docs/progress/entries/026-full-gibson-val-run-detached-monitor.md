# 026 - The full Gibson validation run, detached, with a monitor

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** done -- the run is going (`runs/gibson_val_20261008T120112Z`); code, tests and docs done
**Roadmap item:** ObjectNav -- the published Gibson val benchmark end to end (follows 025's one-click script)

## Goal
The published Gibson ObjectNav validation split -- the OSG Navigator protocol (arXiv:2508.04678, §7.1.1:
1,000 episodes evenly over the five val scenes, i.e. 200 per scene, the published seeds, starts and
targets; SR / SPL / DTG as in its Table 2) -- run once, to completion, detached from any terminal or
IDE session, with no video or frame output and only the per-episode scalars on disk, and a way to see
how far it has got without being there.

## Why
Every number so far came from 3-15 development episodes (`PROTOCOL_AND_TUNING.md` lists them as
contamination). The one-click script of 025 exists for this run; it had never been taken through the
whole split, and a multi-day job needs to survive the operator's session ending.

## Steps
- [x] Find the launcher's inputs on this machine (none of the documented defaults hold here): the
      Habitat interpreter is the `objnav-habitat` conda env, the data is under
      `~/datasets/objectnav/objectnav/gibson/v1.1/val` and `~/datasets/gibson/val/scenes`, Ollama is
      `native` on `127.0.0.1:11435` (the system Ollama on 11434 is a different model store).
- [x] Preflight into the run's directory: 1,000 episodes selected (`full_split`), lean records,
      publishable, LLM + YOLO-World + Habitat all placed on the 4090 by `gpu_plan.py`.
- [x] Launch detached: `setsid nohup bash -c 'echo $$ > launcher.pid; exec run_benchmark.sh --output …'`
      -- the launcher writes its own PID (setsid forks when it is a group leader), stdin from
      `/dev/null`, its own session so a dead terminal cannot HUP it. No `--record`, `--lean` (the
      script's default): ~8 KB per episode, no video, no frames.
- [x] `gibson/monitor.py` + `tests/test_monitor.py` (7 tests): one status line per tick into
      `monitor.log`, `episodes.csv` (the scalar columns of `episodes.jsonl` only), completion judged
      by the launcher's PID with `exit_code.txt` for the code, anchored on `launcher.pid`'s mtime so a
      preflight's older exit code does not count; the summary into `FINAL_SUMMARY.txt` at the end.
- [x] Monitor launched detached beside the run, 5-minute ticks.
- [x] `BENCHMARK.md`: the detached recipe, the stop command (`kill -TERM -- -PGID`), the monitor as
      the fourth progress view, the new files in the results table.
- [x] When the run ends: read `FINAL_SUMMARY.txt` / `summary.json` / `comparison.md`; record SR, SPL,
      DTG here and in a LESSONS entry if anything other than the policy's quality explains a number.
- [x] Qualitative failure inspection, queued behind the run (`gibson/record_episodes.sh --after-run`, output
      `runs/gibson_val_20261008T120112Z.inspection/`): nine slots with fallbacks -- toilet timeouts with final
      DTG > 3.5 m (Darden/000018: started 2.75 m from the toilet; Darden/000193 and Wiconisco/000031: at the
      medians of their pools), Wiconisco couch wrong STOPs > 2 m from the couch (000123: STOP after 3 actions
      at 9.75 m; 000010: a 95-action search ending 2.6 m off; 000114: 98 actions ending 9.1 m off), and the
      lowest-SPL successes (tv Collierville/000043 SPL 0.07 and /000194 SPL 0.25 from 3.5 m; bed
      Corozal/000133 SPL 0.18 after 47.7 m). Videos, final frames, VERIFY/INSPECT/STOP frames and `index.md`.
- [x] Watch the nine recordings against their inspection goals -- first pass done from the decision logs,
      cross-checked with the evaluator's per-step DTG (`trajectory.csv`), written up in LESSONS (2026-10-11,
      "Nine failure recordings"): the toilet timeouts are inspection releases of the real toilet (bottom-cut
      box at 1.2-1.6 m, no LOOK_DOWN in the Gibson action set; 63 % of all 148 toilet failures had locked on
      and released a candidate), the couch STOPs skipped the inspection (`fresh terminal target confirmation`
      at 0.8-0.9 m; two of three on one object at world (0.4, 5.7)), and the low SPLs are the closing's
      `no safe path` / `unverified` releases, not the room order. Visual confirmation of the couch entity
      (`_verify.jpg` / `_stop.jpg`) and of the toilet frames is the remaining human step.
- [x] Decide and implement the closing fixes the LESSONS entry lists (bottom-cut acceptance for floor-standing
      classes or a >= 1.5 m inspection stand-off; inspection before a sofa/bed terminal confirmation; `no safe
      path` as a planning repair) and measure them on the 148 toilet failures and the 18 Wiconisco couch STOPs.
      -> Done as the three tilt-less closing rules of 2026-10-11 (this entry's commit) and measured on the nine
      slots in entry 027: toilet 2/3 improved, couch/bed unchanged, one tv regression traced to the standoff
      re-selection and fixed there. The `no safe path` planning repair is the standoff-keeping rule of 027; the
      sofa/bed inspection-before-confirmation and the 148/18 population-level measurement remain open.

## Open questions
- The ask said "200 episodes across 5 scenes"; the paper's protocol is 200 *per* scene (1,000). The
  run is the full split. If 200 total was meant, `--shards 5 --shard-index 0` gives exactly 40 per
  scene (stride 5 over the published order) and is mergeable later with the other four shards.

## Notes
- 2026-10-11: **the run died at 918/1000 (2026-10-10 00:45Z, 36 h in)** on `Wiconisco/000118`: the only
  published start that lies outside the navmesh horizontally (0.033 m; measured against all 1,000), pulled
  onto the mesh edge by the warm-up turn, refused by the 0.020 m turn-in-place self-check. The supervisor
  resumed once (same crash: deterministic) and stopped, as designed. Fix: `KinematicTolerance.settle_xy_m`
  (0.05 m for Gibson). Finishing the run under a changed self-check and source hash needed the new, recorded
  `--resume-accepting-changes kinematics source_sha256`; resumed 2026-10-11 05:07Z, past 000118 at once.
  Metrics at 918: SR 0.666, SPL 0.373, DTG 1.47 m; one agent error (Darden/000108, `Target closing failed:
  target floor changed`, counted as a failure). Follow-up: a preflight check for starts off the mesh beyond
  `settle_xy_m`, so the next such start is refused before a day of compute, not at episode 919.
- 2026-10-08: the first four episodes took 286 s, 1 s, 3 s, 1 s; the ETA after four was ~20 h and will
  settle once the slow 500-step failures are averaged in (the five-scene runs this morning averaged
  ~2 min per episode on the GPU). Collierville/000000 (toilet) started 1.49 m from the goal and
  still ran 500 steps / 55 m without stopping -- a policy observation for later, not an infra fault.
- 2026-10-08: `test_progress.py::test_the_real_runner_writes_progress_and_lean_rows` fails in `.venv`
  with `No module named 'skfmm'` -- the FMM scorer is a Habitat-env dependency; pre-existing, not
  touched here.

## Result
Finished 2026-10-11T08:49:49Z after 1 resume(s): **SR 0.653, SPL 0.366, DTG 1.59 m** over 1000/1000 published episodes (full split: true; agent errors 1). Per-scene and per-category tables, the published-row comparison and the provenance: `sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md` (section `gibson_val_20261008T120112Z`); the data under `sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/results/gibson_val_20261008T120112Z/`. Published from the detached run by `supervise_benchmark.sh`.
