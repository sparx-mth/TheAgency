# 026 - The full Gibson validation run, detached, with a monitor

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** in-progress -- the run is going (`runs/gibson_val_20261008T120112Z`); code, tests and docs done
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
- [ ] When the run ends: read `FINAL_SUMMARY.txt` / `summary.json` / `comparison.md`; record SR, SPL,
      DTG here and in a LESSONS entry if anything other than the policy's quality explains a number.

## Open questions
- The ask said "200 episodes across 5 scenes"; the paper's protocol is 200 *per* scene (1,000). The
  run is the full split. If 200 total was meant, `--shards 5 --shard-index 0` gives exactly 40 per
  scene (stride 5 over the published order) and is mergeable later with the other four shards.

## Notes
- 2026-10-08: the first four episodes took 286 s, 1 s, 3 s, 1 s; the ETA after four was ~20 h and will
  settle once the slow 500-step failures are averaged in (the five-scene runs this morning averaged
  ~2 min per episode on the GPU). Collierville/000000 (toilet) started 1.49 m from the goal and
  still ran 500 steps / 55 m without stopping -- a policy observation for later, not an infra fault.
- 2026-10-08: `test_progress.py::test_the_real_runner_writes_progress_and_lean_rows` fails in `.venv`
  with `No module named 'skfmm'` -- the FMM scorer is a Habitat-env dependency; pre-existing, not
  touched here.

## Result
(when the run ends)

