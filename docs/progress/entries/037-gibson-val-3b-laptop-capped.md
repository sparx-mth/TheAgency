# 037 - The full Gibson validation run on entry 036's code: the 3B in both roles, on the laptop under 60 % caps

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** done -- finished 2026-10-24 09:55Z; launched 2026-10-22 as `runs/gibson_val_3b_cap60_20261022T161417Z` (detached, supervised,
dry-run publish), on `61873f98` with a clean tree; resumed once after the laptop hung (see Heat)
**Roadmap item:** ObjectNav -- the published Gibson val benchmark re-run on the code of entry 036, comparable with the
3B laptop run of entry 030

## Goal
The 1,000 published SemExp v1.1 `val` episodes (5 scenes x 200, 500 actions; SR, SPL, DTG) on HEAD `61873f98`:
entry 036's committed transit, throttled scene-graph rebuild, per-side frontier split and visit-weighted frontier
ranking, plus the parallel-shard tooling of `61873f98` (no policy change). The method configuration is
`runs/gibson_val_20261016_small_llm`'s (entry 030: SR 0.792, SPL 0.464, DTG 0.94 m): `qwen2.5:3b-instruct` in both
LLM roles, box depth (`SEGMENTATION=0`), lean records (no video, no frames), `--explorer frontier`, seed 17, one
process (no `--parallel`).

## Why under caps
The laptop has powered off mid-run repeatedly: the journal of the boots ending 2026-10-17 11:05 and 2026-10-21 20:09
stops without a shutdown, the second four minutes into a campaign with the 14B oracle on the CPU, and the
2026-10-16 run's `resources.log` shows the CPU package at 96-100 C. Every process of this run is held to 60 % of the
machine:

- **CPU** -- 60 % of 32 threads = 19.2 CPUs: the run's systemd user scope `CPUQuota=1770%` (launcher, `gibson.run`,
  the detector, the monitor, the supervisor and its replays, all at `nice 10`) plus the Ollama container `--cpus 1.5`.
- **RAM** -- 60 % of 31,684 MiB = 19,010 MiB: the scope `MemoryMax=14914M` (`MemoryHigh=14000M`) plus the container
  `--memory 4g`.
- **VRAM** -- 60 % of 8,151 MiB = 4,890 MiB: `GPU_RESERVE_MIB=3261` and the pins `GPU_PLAN_LLM=gpu
  GPU_PLAN_HABITAT=gpu GPU_PLAN_YOLO=cpu` -- the 3B (2.8 GB estimate) and the renderer (1.5 GB) on the card,
  YOLO-World X on the CPU (4 threads). The three together measured 5.3 GB (65 %) in the 2026-10-16 run. The
  software renderer was not the one to move: jobs died natively on it (LESSONS 2026-10-13) and the supervisor's
  replay does not re-wrap it.
- **GPU compute** -- no utilisation cap exists without root; the card's own power limit is 50 W of its 115 W.
- **Heat** (from the first resume on) -- the caps above held (the run used ~3 CPUs, 4.5 GB, 48 % VRAM), yet the
  package sat at a median 98 C (83 % of samples >= 90 C): a few cores boosting to 5+ GHz, which a CPU-time quota does
  not limit. The laptop hung at 16:50:40Z, 33 episodes in, with no kernel, OOM, MCE or Xid line. `start_capped.sh`
  now sets the power-saver profile on every start (EPP `power`, ASUS platform profile `quiet`; no root): a median
  85 C, none >= 90 C, for ~8 % on the CPU detector (466-495 vs 430-456 ms/frame). The policy's clock is simulated (a
  fixed `action_time_s` per action) and the service timeouts (detector 30 s, LLM 120 s) are far above the frame
  times, so the profile changes the wall time, not SR/SPL/DTG. Pinning the run to the 16 efficiency cores as well
  was measured and dropped: ~3 C more for 1,045 ms/frame and 1.94 vs ~1.0 s/step.

The one difference from entry 030's run is therefore the detector's device (CPU, not CUDA): its numerics differ
slightly (LESSONS 2026-10-08) and each action is slower; both are recorded in the frozen configuration.

## Steps
- [x] Nothing else running before the launch: no benchmark process, no Ollama, the card empty, a clean tree.
- [x] Launched with `runs/gibson_val_3b_cap60_20261022T161417Z/start_capped.sh`: the container's caps, then
      `launch_detached.sh` inside the capped scope, then `resources.log` (RAM, the scope's CPU and memory, the card's
      use, power and temperature, the CPU package temperature) every 20 s.
- [x] After a crash or a reboot: the same `start_capped.sh` once -- the same caps, and the supervisor resumes the run
      from its replay snapshot (`MAX_RESUMES=5`). Done once: the hang at 16:50:40Z (33 episodes on disk, all valid;
      `Collierville/000033` lost mid-episode), resumed at 17:05:10Z (resume 1 of 5) on the unchanged source, under the
      power-saver profile from then on (`resources.log` marks each change).
- [x] When the run ends: read `FINAL_SUMMARY.txt` / `summary.json` / `comparison.md`; SR, SPL, DTG beside entry 030's
      3B run, per scene and per category; whether the caps held (`resources.log`) and whether the laptop stayed up.

## Result
Finished 2026-10-24T09:55:50Z after 1 resume (no configuration change): **SR 0.750 [0.722, 0.776], SPL 0.458
[0.435, 0.482], DTG 1.48 m** (SoftSPL 0.489) over 1000/1000 published episodes (full split; agent errors 0; 900 stops,
100 action limits). Not published: the dry run built `7f5d26bf` and discarded it; the data are in the run directory.

Against entry 030's 3B run, paired over the same 1,000 episodes (95 % paired-bootstrap intervals):

| | SR | SPL | DTG (m) |
|---|---:|---:|---:|
| entry 030 | 0.792 | 0.464 | 0.94 |
| this run | 0.750 | 0.458 | 1.48 |
| difference | -0.042 [-0.069, -0.015] | -0.006 [-0.023, +0.012] | +0.54 [+0.34, +0.77] |

- **Per scene:** Collierville better (SR 0.780 -> 0.825, SPL 0.544 -> 0.593); Corozal (0.800 -> 0.740), Darden
  (0.815 -> 0.795) and Markleeville (0.750 -> 0.700) within noise; Wiconisco worse (0.815 -> 0.690, [-0.19, -0.06]).
- **Per category:** only toilet moved beyond noise: SR 0.894 -> 0.740 ([-0.226, -0.082]; 15 gained, 47 lost). 43 of
  the 47 lost end at the 500-action limit, a median 12.9 m from the goal, where entry 030 needed a median 320 actions:
  the long bathroom searches no longer finish. 24 of the 47 are in Wiconisco (toilet SR 0.87 -> 0.50). Failures
  overall 208 -> 250: action limits 63 -> 97, wrong stops 144 -> 153.
- **Confound:** the detector ran on the CPU here and on CUDA in entry 030, the one configuration difference besides
  the code. Failures at the action limit far from the goal point at exploration rather than detection, but this run
  cannot separate the two; the 62 toilet episodes whose outcome changed, re-run on entry 030's code with the CPU
  detector, would.
- **The caps and the laptop:** one hang (above), none after the power-saver profile: 40.6 h at a median 85 C (p95 88,
  max 97, 1.5 % of samples >= 90 C); the run at most 3.8 CPUs and 4.7 GB, RAM at most 44 %, VRAM 48 %, the GPU at
  most 68 C. Episode wall time 41.4 h against entry 030's 23.3 h (the CPU detector and the profile).







