# 028 - The full Gibson validation run on the 8 GB laptop: 3B oracle, box depth, lean records

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** in-progress -- launch tooling and preflight done; `runs/gibson_val_3b_laptop_20261012T142659Z` is scheduled to launch itself at the boot that loads the matching NVIDIA kernel module (`launch_detached.sh --after-reboot`)
**Roadmap item:** ObjectNav -- the published Gibson val benchmark under the small-model configuration (follows 026's office-PC run)

## Goal
The same 1,000 published SemExp v1.1 `val` episodes as entry 026 (five scenes x 200, 500 actions, SR / SPL / DTG /
SoftSPL), run once to completion on the development laptop -- one 8 GB GPU, 32 threads, 30 GB RAM -- with the
**small model in both LLM roles** (`qwen2.5:3b-instruct` labels the rooms *and* is the node oracle), the **box
depth pipeline** (no Segment Anything), lean records (no video, no frames), resumable across a restart of the
machine, and with everything on disk that a later analysis needs: which categories were found in which buildings,
the wrong STOPs (false positives), the step-limit timeouts, and the found-then-released cases.

## Why
Entry 026's result (SR 0.653, SPL 0.366, DTG 1.59 m) was produced with the 14B oracle on a 24 GB card. The paper
comparison (SemExp / PONI / LFG / OSG-Nav rows in `comparison.md`) needs the number this stack reaches with the model
that fits a laptop GPU, measured on the full split, not inferred from the 3-15 episode development campaigns.
The 3B's known weaknesses are on record (LESSONS 2026-10-05: it copies the worked example's numbers; the
`home_here` field and the uncertainty floor were the fixes) -- this run measures what they cost at scale.

## Configuration (frozen in the run's `run.json`)
- `LLM_MODEL=LLM_REASONING_MODEL=qwen2.5:3b-instruct`, `SEGMENTATION=0`, `--lean`, seed 17, the full split.
- Ollama in a GPU-enabled docker container, `ollama-scene-graph-gpu` (same model volume as `ollama-scene-graph`,
  `--gpus all`, `--restart unless-stopped`, `OLLAMA_CONTEXT_LENGTH=8192` so the label calls and the 8k-context oracle
  calls share one runner instead of reloading the model at every switch).
- `GPU_PLAN=auto` with `LLM_VRAM_MIB=2800` (the manifests live in a root-owned docker volume the planner cannot
  read): the plan places habitat (1.5 GB, GPU-only) first, then the LLM, then YOLO-World X (2 GB), and records the
  decision in `gpu_plan.json` and the frozen configuration. What does not fit falls back to the CPU and the plan says so.
- Launched by `gibson/launch_detached.sh <output>` (new in this entry): the detached recipe of BENCHMARK.md 2.3 as
  one command -- launcher, monitor, supervisor -- and the same command resumes the run after a reboot (it starts the
  Ollama container, which the supervisor cannot replay from `/proc`, and a fresh supervisor replays the detector and
  `gibson.run --resume` from the snapshot). `PROGRESS_ENTRY` points here, so publishing closes this entry.
  `--after-reboot` (2026-10-12, the GPU dead until the reboot) wrote this launch's variables to `<output>/launch.env`
  and a one-shot `@reboot` crontab entry that runs the launch once `nvidia-smi` and `docker` answer -- and refuses
  to launch at all if they do not within 15 minutes, since a CPU launch could never be moved to the card.

## Steps
- [x] Preflight of the exact configuration on this machine (`runs/preflight_3b_laptop/preflight.json`): 1,000 episodes
      selected (`full_split`), lean, publishable, 3B in both roles, segmentation off, the detector on the Gibson vocabulary.
- [x] `launch_detached.sh` (launch + reboot recovery + `--stop`), the supervisor's `alive()` hardened against PID reuse
      after a reboot (a live PID counts only if its command line names the run), the GPU Ollama container created.
- [x] `launch_detached.sh --after-reboot`: the run `runs/gibson_val_3b_laptop_20261012T142659Z` scheduled for the
      next boot with `LLM_MODEL=LLM_REASONING_MODEL=qwen2.5:3b-instruct`, `LLM_VRAM_MIB=2800`, `SEGMENTATION=0`,
      `GPU_PLAN=auto`, `OLLAMA_CONTAINER=ollama-scene-graph-gpu`, `PROGRESS_ENTRY` = this entry (`launch.env`).
- [ ] Reboot (the NVIDIA userspace was upgraded under the running kernel module this morning; `nvidia-smi` and the
      EGL renderer are dead until then -- LESSONS 2026-10-12). The run launches itself: `<output>/boot.log` is the
      transcript, then `launcher.log`; read `gpu_plan.json` and confirm the three placements (LLM and YOLO on the
      card beside habitat, or which one the plan offloaded and why) before leaving the run alone.
- [ ] While it runs: `run_benchmark.sh --status <output>`; `monitor.log` every 5 minutes; after a restart of the
      machine, `launch_detached.sh <output>` once.
- [ ] When the run ends: read `FINAL_SUMMARY.txt` / `summary.json` / `comparison.md`; record SR, SPL, DTG, SoftSPL
      here, beside 026's numbers.
- [ ] The analysis from `episodes.jsonl` (lean rows keep scene, target, success, termination, DTG, SPL, SoftSPL and the
      policy counters -- `target_closing` releases / inspection releases / last release and the STOP's class and
      confidence, `target_evidence`, `room_scans`, `llm_queries`) and `evaluation_diagnostics.jsonl`: SR / SPL / DTG per
      scene x category; wrong STOPs (`termination == "stop"`, `success == false`) by scene and by the class STOPped on;
      timeouts (`termination == "step_limit"`) split into never-sighted vs found-and-released by the closing counters;
      the per-category comparison against 026 (14B) and against the published rows.

## Open questions
- Whether the three processes fit the card beside the desktop (estimates: habitat 1.5 GB + LLM 2.8 GB + YOLO 2 GB +
  the 512 MiB reserve on 8 GB). If YOLO is offloaded the run is still valid -- the detector's numerics differ between
  CPU and CUDA (LESSONS 2026-10-08), which the frozen configuration records -- but about a minute per episode slower.
- The pace. This morning's five recorded episodes with every service on the CPU averaged 816 s; the office PC with
  everything on a 4090 averaged ~160 s. With the 3B on this GPU the oracle call drops from ~60 s to seconds, so the
  budget is 2-4 days; the ETA in `progress.json` settles after the first slow (500-step) episodes are averaged in.

## Notes
- 2026-10-12: a run's device layout -- the renderer, the detector's device (in its `/health` identity), the LLM's
  `num_gpu` -- is part of the frozen configuration and is *not* among `RESUMABLE_CONFIG_CHANGES`; a run launched on
  the CPU/software path today could never move to the GPU after the reboot. Hence: reboot first, launch once.
- 2026-10-12: lid-close still suspends this laptop (logind default); idle suspend on AC is off. Keep the lid open or
  set `HandleLidSwitch=ignore` for the run's lifetime.
- No `.py` under `sparx_agency/` changes while the run is going (the source hash is in the frozen configuration);
  the launcher and the supervisor fix are `.sh`, this entry and the docs are Markdown.
- 2026-10-12: the source the run measures is `786bed49` -- the toilet closing rules of entry 027 (box-driven pitch,
  STOP on a whole confident box, the kept standoff) and the segmentation flag, which this run leaves off; the SAM
  comparison is the next run, same split, `SEGMENTATION=1`.
- 2026-10-12: a launch born from cron has no desktop session; the supervisor's publish at the end still works because
  the remote is HTTPS with `credential.helper=store`. Were it SSH with an agent, the commit would land on the local
  `results/<run>` branch instead (the supervisor's documented fallback) and need a push by hand.

## Result
(when the run ends)
