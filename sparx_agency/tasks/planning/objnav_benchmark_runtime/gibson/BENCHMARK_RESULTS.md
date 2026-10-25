# Gibson ObjectNav benchmark -- results

One section per completed run of [`run_benchmark.sh`](BENCHMARK.md), written by `supervise_benchmark.sh` when the run ends; the data behind each section is under `results/<run>/`. Numbers are not comparable to the published rows beyond the caveats each section repeats.

## `gibson_val_20261008T120112Z` -- finished 2026-10-11T08:49:49Z

**Full Gibson validation: 1000 of 1000 published episodes.** Protocol `gibson-semexp-v1.1-val/1`: the published SemExp v1.1 `val` episodes, 500 actions, SR / SPL / DTG as in
arXiv:2508.04678 Table 2 (DTG = FMM distance to the 1 m success region; SoftSPL supplementary).

| | SR | SPL | DTG (m) | SoftSPL | mean steps | episodes |
|---|---:|---:|---:|---:|---:|---:|
| **this run** | **0.653** (95% CI 0.623-0.682) | **0.366** (95% CI 0.341-0.389) | **1.59** | 0.403 | 220 | 1000 |

Terminations: agent_error 1, step_limit 254, stop 745. Agent errors: 1. Episodes with no reachable goal at the end: 0.

### Per scene

| scene | n | SR | SPL | SoftSPL | DTG (m) | steps |
|---|---:|---:|---:|---:|---:|---:|
| Collierville | 200 | 0.645 | 0.440 | 0.454 | 1.20 | 207 |
| Corozal | 200 | 0.730 | 0.457 | 0.486 | 1.54 | 199 |
| Darden | 200 | 0.605 | 0.319 | 0.354 | 1.65 | 240 |
| Markleeville | 200 | 0.745 | 0.287 | 0.300 | 0.96 | 254 |
| Wiconisco | 200 | 0.540 | 0.325 | 0.420 | 2.60 | 200 |

### Per category

| category | n | SR | SPL | SoftSPL | DTG (m) | steps |
|---|---:|---:|---:|---:|---:|---:|
| bed | 123 | 0.528 | 0.229 | 0.257 | 2.65 | 301 |
| chair | 177 | 0.887 | 0.580 | 0.606 | 0.20 | 96 |
| couch | 229 | 0.769 | 0.533 | 0.611 | 0.65 | 119 |
| potted plant | 193 | 0.829 | 0.355 | 0.365 | 0.38 | 178 |
| toilet | 208 | 0.288 | 0.102 | 0.140 | 4.11 | 432 |
| tv | 70 | 0.500 | 0.329 | 0.352 | 2.16 | 202 |

### Provenance

- Source commit `d41fd686377c` with uncommitted changes at launch (`source-changes.diff` in the run directory); source sha256 `8403b164e6c47242...` (every `.py` under `sparx_agency/`).
- Method `sparx-rpt-llm-host-sweep`: local exploration `frontier`; room labels `qwen2.5:3b-instruct`, node oracle `qwen2.5:14b-instruct`
  (ollama at `http://127.0.0.1:11435`, seed 17, temperature 0.0); detector `yolo_world`.
- habitat-sim 0.2.4 (reference 0.1.5; `reference_sim_version_match: false`, acknowledged), numpy 1.26.4,
  scikit-fmm 2024.5.29; seed 17; lean records true; recording off (no video, no frames).
- GPU plan llm=gpu, yolo=gpu, habitat=gpu on NVIDIA GeForce RTX 4090 (pcn87652). Started 2026-10-08T12:06:13Z, finished 2026-10-11T08:49:49Z; 1 resume(s);
  episode wall time in total 1d16h (mean 2m25s per episode).
- **Resumed under a changed frozen configuration** (`gibson.run --resume-accepting-changes`, recorded in `run.json` under `resumes[].config_changes`): after 918 episodes at 2026-10-11T05:07:29Z, `kinematics` settle_xy_m None -> 0.05; `source_sha256` 8403b164e6c4... -> 33b08f49582b.... Each is provenance or a self-check that only refuses an episode -- never an input to a decision or a score; the scorer and the protocol were the same before and after.
- Data: `results/gibson_val_20261008T120112Z/summary.json` (every metric with its interval, per scene and per category), `episodes.csv`
  (the per-episode scalars: id, scene, target, success, SPL, SoftSPL, path length, shortest path, start and final
  distance to goal, steps, STOP, termination, wall time, agent error) and `run.json` (the frozen configuration, the
  dataset file hashes and the command line). The run directory was `runs/gibson_val_20261008T120112Z` on pcn87652 (gitignored).

### Beside the published rows (`comparison.md` of the run)

### Full Gibson validation -- resumed with recorded config changes (kinematics, source_sha256); see the audit

SR/SPL are episode means including failures; DTG here is FMM distance to the 1 m success region. SoftSPL is supplementary.

**Resume caveat:** 1 resume(s) continued this run under a frozen configuration that differed in `kinematics`, `source_sha256` -- each a key `gibson.run --resume-accepting-changes` admits because it is provenance (`source_sha256`, `gpu_plan`) or a self-check that only refuses an episode (`kinematics`), never an input to a decision or a score; the recorded and resumed values are in `run.json` under `resumes[].config_changes`. The episodes before and after were scored by the same scorer under the same protocol.

| Method | SR (%) | SPL (%) | SoftSPL (%) | DTG (m) | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| SemExp | 65.7 | 33.9 | — | 1.47 | arXiv:2508.04678v1, Table 2, p.12; TF=False; NM=False. Paper-reported, protocol equivalence not established. |
| PONI | 73.6 | 41.0 | — | 1.25 | arXiv:2508.04678v1, Table 2, p.12; TF=False; NM=False. Paper-reported, protocol equivalence not established. |
| LGX | 31.0 | 5.2 | — | 4.78 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| LFG | 64.5 | 40.6 | — | 1.81 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| FBE | 64.1 | 28.3 | — | 1.78 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| SemUtil | 69.3 | 40.5 | — | 1.49 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| OSG-Nav | 73.4 | 38.6 | — | 1.72 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=True. Paper-reported, protocol equivalence not established. |
| sparx-rpt-llm-host-sweep (ours) | 65.3 [62.3, 68.2] | 36.6 [34.1, 38.9] | 40.3 | 1.59 | this run: 1000 episodes, 95% intervals, 1 agent errors counted as failures |

**Comparison caveat:** the published rows are reference values, not paired re-evaluations. ApexNav and SG-Nav do not report Gibson. TF = training-free; NM = non-metric. Our RPT adaptation is TF, not NM. GT pose/depth do not mean GT semantic detections.

#### Audit

- **full_split_completed:** True
- **episodes:** 1000
- **selected_episodes:** 1000
- **agent_errors:** 1
- **reference_sim_version_match:** False
- **exact_osg_protocol_reproduction:** False
- **resumes:** 1
- **resumes_with_config_changes:** [{'utc': '2026-10-11T05:07:29Z', 'episodes_done': 918, 'changed': ['kinematics', 'source_sha256']}]
- **primary_metrics:** ['SR', 'SPL', 'DTG (SemExp distance-to-success, m)']
- **supplementary_metrics:** ['SoftSPL', 'steps', 'wall_s']
- **total_wall_s:** 145279.8156782955
- **notes:** OSG's exact evaluator/episode hashes are not published in the supplied paper. SemExp and current PONI code differ; no direct SOTA win is asserted.

## `gibson_val_20261012T141500Z` -- finished 2026-10-14T01:31:40Z

**Full Gibson validation: 1000 of 1000 published episodes.** Protocol `gibson-semexp-v1.1-val/1`: the published SemExp v1.1 `val` episodes, 500 actions, SR / SPL / DTG as in
arXiv:2508.04678 Table 2 (DTG = FMM distance to the 1 m success region; SoftSPL supplementary).

| | SR | SPL | DTG (m) | SoftSPL | mean steps | episodes |
|---|---:|---:|---:|---:|---:|---:|
| **this run** | **0.713** (95% CI 0.684-0.740) | **0.409** (95% CI 0.385-0.433) | **1.27** | 0.443 | 200 | 1000 |

Terminations: agent_error 2, step_limit 189, stop 809. Agent errors: 2. Episodes with no reachable goal at the end: 0.

### Per scene

| scene | n | SR | SPL | SoftSPL | DTG (m) | steps |
|---|---:|---:|---:|---:|---:|---:|
| Collierville | 200 | 0.675 | 0.456 | 0.470 | 1.09 | 203 |
| Corozal | 200 | 0.745 | 0.463 | 0.494 | 1.50 | 188 |
| Darden | 200 | 0.695 | 0.367 | 0.396 | 1.25 | 214 |
| Markleeville | 200 | 0.770 | 0.326 | 0.341 | 0.91 | 227 |
| Wiconisco | 200 | 0.680 | 0.432 | 0.516 | 1.62 | 168 |

### Per category

| category | n | SR | SPL | SoftSPL | DTG (m) | steps |
|---|---:|---:|---:|---:|---:|---:|
| bed | 123 | 0.504 | 0.207 | 0.233 | 2.89 | 299 |
| chair | 177 | 0.893 | 0.646 | 0.671 | 0.22 | 87 |
| couch | 229 | 0.803 | 0.589 | 0.666 | 0.55 | 97 |
| potted plant | 193 | 0.845 | 0.359 | 0.367 | 0.42 | 175 |
| toilet | 208 | 0.514 | 0.195 | 0.223 | 2.58 | 374 |
| tv | 70 | 0.557 | 0.350 | 0.373 | 1.94 | 199 |

### Provenance

- Source commit `709e351228c0` (clean tree); source sha256 `845ac98d7470729e...` (every `.py` under `sparx_agency/`).
- Method `sparx-rpt-llm-host-sweep`: local exploration `frontier`; room labels `qwen2.5:3b-instruct`, node oracle `qwen2.5:14b-instruct`
  (ollama at `http://127.0.0.1:11435`, seed 17, temperature 0.0); detector `yolo_world`.
- habitat-sim 0.2.4 (reference 0.1.5; `reference_sim_version_match: false`, acknowledged), numpy 1.26.4,
  scikit-fmm 2024.5.29; seed 17; lean records true; recording off (no video, no frames).
- GPU plan llm=gpu, yolo=gpu, habitat=gpu on NVIDIA GeForce RTX 4090 (pcn87652). Started 2026-10-12T14:12:28Z, finished 2026-10-14T01:31:40Z; 0 resume(s);
  episode wall time in total 1d11h (mean 2m07s per episode).
- Data: `results/gibson_val_20261012T141500Z/summary.json` (every metric with its interval, per scene and per category), `episodes.csv`
  (the per-episode scalars: id, scene, target, success, SPL, SoftSPL, path length, shortest path, start and final
  distance to goal, steps, STOP, termination, wall time, agent error) and `run.json` (the frozen configuration, the
  dataset file hashes and the command line). The run directory was `runs/gibson_val_20261012T141500Z` on pcn87652 (gitignored).

### Beside the published rows (`comparison.md` of the run)

### Full Gibson validation

SR/SPL are episode means including failures; DTG here is FMM distance to the 1 m success region. SoftSPL is supplementary.

| Method | SR (%) | SPL (%) | SoftSPL (%) | DTG (m) | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| SemExp | 65.7 | 33.9 | — | 1.47 | arXiv:2508.04678v1, Table 2, p.12; TF=False; NM=False. Paper-reported, protocol equivalence not established. |
| PONI | 73.6 | 41.0 | — | 1.25 | arXiv:2508.04678v1, Table 2, p.12; TF=False; NM=False. Paper-reported, protocol equivalence not established. |
| LGX | 31.0 | 5.2 | — | 4.78 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| LFG | 64.5 | 40.6 | — | 1.81 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| FBE | 64.1 | 28.3 | — | 1.78 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| SemUtil | 69.3 | 40.5 | — | 1.49 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| OSG-Nav | 73.4 | 38.6 | — | 1.72 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=True. Paper-reported, protocol equivalence not established. |
| sparx-rpt-llm-host-sweep (ours) | 71.3 [68.4, 74.0] | 40.9 [38.5, 43.3] | 44.3 | 1.27 | this run: 1000 episodes, 95% intervals, 2 agent errors counted as failures |

**Comparison caveat:** the published rows are reference values, not paired re-evaluations. ApexNav and SG-Nav do not report Gibson. TF = training-free; NM = non-metric. Our RPT adaptation is TF, not NM. GT pose/depth do not mean GT semantic detections.

#### Audit

- **full_split_completed:** True
- **episodes:** 1000
- **selected_episodes:** 1000
- **agent_errors:** 2
- **reference_sim_version_match:** False
- **exact_osg_protocol_reproduction:** False
- **resumes:** 0
- **resumes_with_config_changes:** []
- **primary_metrics:** ['SR', 'SPL', 'DTG (SemExp distance-to-success, m)']
- **supplementary_metrics:** ['SoftSPL', 'steps', 'wall_s']
- **total_wall_s:** 127150.13345551712
- **notes:** OSG's exact evaluator/episode hashes are not published in the supplied paper. SemExp and current PONI code differ; no direct SOTA win is asserted.

## `gibson_val_20261016_small_llm` -- finished 2026-10-18T13:35:17Z

**Full Gibson validation: 1000 of 1000 published episodes.** Protocol `gibson-semexp-v1.1-val/2`: the published SemExp v1.1 `val` episodes, 500 actions, SR / SPL / DTG as in
arXiv:2508.04678 Table 2 (DTG = FMM distance to the 1 m success region; SoftSPL supplementary).

| | SR | SPL | DTG (m) | SoftSPL | mean steps | episodes |
|---|---:|---:|---:|---:|---:|---:|
| **this run** | **0.792** (95% CI 0.766-0.816) | **0.464** (95% CI 0.441-0.487) | **0.94** | 0.491 | 152 | 1000 |

Terminations: agent_error 1, step_limit 67, stop 932. Agent errors: 1. Episodes with no reachable goal at the end: 0.

### Per scene

| scene | n | SR | SPL | SoftSPL | DTG (m) | steps |
|---|---:|---:|---:|---:|---:|---:|
| Collierville | 200 | 0.780 | 0.544 | 0.547 | 0.86 | 101 |
| Corozal | 200 | 0.800 | 0.479 | 0.508 | 1.26 | 172 |
| Darden | 200 | 0.815 | 0.416 | 0.434 | 0.72 | 173 |
| Markleeville | 200 | 0.750 | 0.367 | 0.391 | 1.16 | 189 |
| Wiconisco | 200 | 0.815 | 0.514 | 0.573 | 0.68 | 128 |

### Per category

| category | n | SR | SPL | SoftSPL | DTG (m) | steps |
|---|---:|---:|---:|---:|---:|---:|
| bed | 123 | 0.528 | 0.256 | 0.285 | 2.76 | 243 |
| chair | 177 | 0.915 | 0.689 | 0.703 | 0.23 | 78 |
| couch | 229 | 0.786 | 0.579 | 0.649 | 0.73 | 99 |
| potted plant | 193 | 0.798 | 0.376 | 0.386 | 0.70 | 113 |
| toilet | 208 | 0.894 | 0.357 | 0.366 | 0.52 | 260 |
| tv | 70 | 0.643 | 0.445 | 0.453 | 2.07 | 147 |

### Provenance

- Source commit `f628a476ca33` (clean tree); source sha256 `41c708f3acc1db8d...` (every `.py` under `sparx_agency/`).
- Method `sparx-rpt-llm-host-sweep`: local exploration `frontier`; room labels `qwen2.5:3b-instruct`, node oracle `qwen2.5:3b-instruct`
  (ollama at `http://127.0.0.1:11434`, seed 17, temperature 0.0); detector `yolo_world`.
- habitat-sim 0.2.4 (reference 0.1.5; `reference_sim_version_match: false`, acknowledged), numpy 1.26.4,
  scikit-fmm 2024.5.29; seed 17; lean records true; recording off (no video, no frames).
- GPU plan llm=gpu, yolo=gpu, habitat=gpu on NVIDIA GeForce RTX 5070 Laptop GPU (LP-Boston-17093). Started 2026-10-16T18:22:00Z, finished 2026-10-18T13:35:17Z; 3 resume(s);
  episode wall time in total 23h17m (mean 1m24s per episode).
- Data: `results/gibson_val_20261016_small_llm/summary.json` (every metric with its interval, per scene and per category), `episodes.csv`
  (the per-episode scalars: id, scene, target, success, SPL, SoftSPL, path length, shortest path, start and final
  distance to goal, steps, STOP, termination, wall time, agent error) and `run.json` (the frozen configuration, the
  dataset file hashes and the command line). The run directory was `runs/gibson_val_20261016_small_llm` on LP-Boston-17093 (gitignored).

### Beside the published rows (`comparison.md` of the run)

### Full Gibson validation

SR/SPL are episode means including failures; DTG here is FMM distance to the 1 m success region. SoftSPL is supplementary.

| Method | SR (%) | SPL (%) | SoftSPL (%) | DTG (m) | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| SemExp | 65.7 | 33.9 | — | 1.47 | arXiv:2508.04678v1, Table 2, p.12; TF=False; NM=False. Paper-reported, protocol equivalence not established. |
| PONI | 73.6 | 41.0 | — | 1.25 | arXiv:2508.04678v1, Table 2, p.12; TF=False; NM=False. Paper-reported, protocol equivalence not established. |
| LGX | 31.0 | 5.2 | — | 4.78 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| LFG | 64.5 | 40.6 | — | 1.81 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| FBE | 64.1 | 28.3 | — | 1.78 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| SemUtil | 69.3 | 40.5 | — | 1.49 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| OSG-Nav | 73.4 | 38.6 | — | 1.72 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=True. Paper-reported, protocol equivalence not established. |
| sparx-rpt-llm-host-sweep (ours) | 79.2 [76.6, 81.6] | 46.4 [44.1, 48.7] | 49.1 | 0.94 | this run: 1000 episodes, 95% intervals, 1 agent errors counted as failures |

**Comparison caveat:** the published rows are reference values, not paired re-evaluations. ApexNav and SG-Nav do not report Gibson. TF = training-free; NM = non-metric. Our RPT adaptation is TF, not NM. GT pose/depth do not mean GT semantic detections.

#### Audit

- **full_split_completed:** True
- **episodes:** 1000
- **selected_episodes:** 1000
- **agent_errors:** 1
- **reference_sim_version_match:** False
- **exact_osg_protocol_reproduction:** False
- **resumes:** 3
- **resumes_with_config_changes:** []
- **primary_metrics:** ['SR', 'SPL', 'DTG (SemExp distance-to-success, m)']
- **supplementary_metrics:** ['SoftSPL', 'steps', 'wall_s']
- **total_wall_s:** 83844.42688726992
- **notes:** OSG's exact evaluator/episode hashes are not published in the supplied paper. SemExp and current PONI code differ; no direct SOTA win is asserted.

## `gibson_val_20261018_14b_sam` -- finished 2026-10-19T16:30:02Z

**Full Gibson validation: 1000 of 1000 published episodes.** Protocol `gibson-semexp-v1.1-val/2`: the published SemExp v1.1 `val` episodes, 500 actions, SR / SPL / DTG as in
arXiv:2508.04678 Table 2 (DTG = FMM distance to the 1 m success region; SoftSPL supplementary).

| | SR | SPL | DTG (m) | SoftSPL | mean steps | episodes |
|---|---:|---:|---:|---:|---:|---:|
| **this run** | **0.784** (95% CI 0.757-0.808) | **0.480** (95% CI 0.456-0.503) | **1.00** | 0.508 | 142 | 1000 |

Terminations: agent_error 0, step_limit 54, stop 946. Agent errors: 0. Episodes with no reachable goal at the end: 0.

### Per scene

| scene | n | SR | SPL | SoftSPL | DTG (m) | steps |
|---|---:|---:|---:|---:|---:|---:|
| Collierville | 200 | 0.765 | 0.542 | 0.551 | 0.86 | 99 |
| Corozal | 200 | 0.740 | 0.482 | 0.517 | 1.43 | 156 |
| Darden | 200 | 0.840 | 0.459 | 0.474 | 0.73 | 160 |
| Markleeville | 200 | 0.760 | 0.403 | 0.417 | 1.14 | 164 |
| Wiconisco | 200 | 0.815 | 0.517 | 0.579 | 0.82 | 131 |

### Per category

| category | n | SR | SPL | SoftSPL | DTG (m) | steps |
|---|---:|---:|---:|---:|---:|---:|
| bed | 123 | 0.520 | 0.254 | 0.281 | 2.98 | 201 |
| chair | 177 | 0.932 | 0.697 | 0.710 | 0.19 | 72 |
| couch | 229 | 0.782 | 0.611 | 0.680 | 0.74 | 93 |
| potted plant | 193 | 0.819 | 0.427 | 0.433 | 0.68 | 93 |
| toilet | 208 | 0.865 | 0.362 | 0.372 | 0.75 | 271 |
| tv | 70 | 0.543 | 0.403 | 0.438 | 1.97 | 124 |

### Provenance

- Source commit `59783f62089c` with uncommitted changes at launch (`source-changes.diff` in the run directory); source sha256 `29e8a21030a53459...` (every `.py` under `sparx_agency/`).
- Method `sparx-rpt-llm-host-sweep`: local exploration `frontier`; room labels `qwen2.5:3b-instruct`, node oracle `qwen2.5:14b-instruct`
  (ollama at `http://127.0.0.1:11434`, seed 17, temperature 0.0); detector `yolo_world`.
- habitat-sim 0.2.4 (reference 0.1.5; `reference_sim_version_match: false`, acknowledged), numpy 1.26.4,
  scikit-fmm 2024.5.29; seed 17; lean records true; recording off (no video, no frames).
- GPU plan llm=gpu, yolo=gpu, habitat=gpu on NVIDIA GeForce RTX 4090 (PCN87653). Started 2026-10-18T14:33:29Z, finished 2026-10-19T16:30:02Z; 0 resume(s);
  episode wall time in total 1d01h (mean 1m33s per episode).
- Data: `results/gibson_val_20261018_14b_sam/summary.json` (every metric with its interval, per scene and per category), `episodes.csv`
  (the per-episode scalars: id, scene, target, success, SPL, SoftSPL, path length, shortest path, start and final
  distance to goal, steps, STOP, termination, wall time, agent error) and `run.json` (the frozen configuration, the
  dataset file hashes and the command line). The run directory was `runs/gibson_val_20261018_14b_sam` on PCN87653 (gitignored).

### Beside the published rows (`comparison.md` of the run)

### Full Gibson validation

SR/SPL are episode means including failures; DTG here is FMM distance to the 1 m success region. SoftSPL is supplementary.

| Method | SR (%) | SPL (%) | SoftSPL (%) | DTG (m) | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| SemExp | 65.7 | 33.9 | — | 1.47 | arXiv:2508.04678v1, Table 2, p.12; TF=False; NM=False. Paper-reported, protocol equivalence not established. |
| PONI | 73.6 | 41.0 | — | 1.25 | arXiv:2508.04678v1, Table 2, p.12; TF=False; NM=False. Paper-reported, protocol equivalence not established. |
| LGX | 31.0 | 5.2 | — | 4.78 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| LFG | 64.5 | 40.6 | — | 1.81 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| FBE | 64.1 | 28.3 | — | 1.78 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| SemUtil | 69.3 | 40.5 | — | 1.49 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| OSG-Nav | 73.4 | 38.6 | — | 1.72 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=True. Paper-reported, protocol equivalence not established. |
| sparx-rpt-llm-host-sweep (ours) | 78.4 [75.7, 80.8] | 48.0 [45.6, 50.3] | 50.8 | 1.00 | this run: 1000 episodes, 95% intervals |

**Comparison caveat:** the published rows are reference values, not paired re-evaluations. ApexNav and SG-Nav do not report Gibson. TF = training-free; NM = non-metric. Our RPT adaptation is TF, not NM. GT pose/depth do not mean GT semantic detections.

#### Audit

- **full_split_completed:** True
- **episodes:** 1000
- **selected_episodes:** 1000
- **agent_errors:** 0
- **reference_sim_version_match:** False
- **exact_osg_protocol_reproduction:** False
- **resumes:** 0
- **resumes_with_config_changes:** []
- **primary_metrics:** ['SR', 'SPL', 'DTG (SemExp distance-to-success, m)']
- **supplementary_metrics:** ['SoftSPL', 'steps', 'wall_s']
- **total_wall_s:** 93391.80004298949
- **notes:** OSG's exact evaluator/episode hashes are not published in the supplied paper. SemExp and current PONI code differ; no direct SOTA win is asserted.

## `gibson_val_20261018T105339Z` -- finished 2026-10-19T12:40:17Z

**Full Gibson validation: 1000 of 1000 published episodes.** Protocol `gibson-semexp-v1.1-val/2`: the published SemExp v1.1 `val` episodes, 500 actions, SR / SPL / DTG as in
arXiv:2508.04678 Table 2 (DTG = FMM distance to the 1 m success region; SoftSPL supplementary).

| | SR | SPL | DTG (m) | SoftSPL | mean steps | episodes |
|---|---:|---:|---:|---:|---:|---:|
| **this run** | **0.788** (95% CI 0.762-0.812) | **0.465** (95% CI 0.441-0.488) | **0.96** | 0.493 | 152 | 1000 |

Terminations: agent_error 0, step_limit 61, stop 939. Agent errors: 0. Episodes with no reachable goal at the end: 0.

### Per scene

| scene | n | SR | SPL | SoftSPL | DTG (m) | steps |
|---|---:|---:|---:|---:|---:|---:|
| Collierville | 200 | 0.785 | 0.551 | 0.554 | 0.82 | 99 |
| Corozal | 200 | 0.765 | 0.481 | 0.506 | 1.45 | 169 |
| Darden | 200 | 0.845 | 0.440 | 0.454 | 0.67 | 169 |
| Markleeville | 200 | 0.720 | 0.347 | 0.382 | 1.17 | 190 |
| Wiconisco | 200 | 0.825 | 0.506 | 0.569 | 0.68 | 133 |

### Per category

| category | n | SR | SPL | SoftSPL | DTG (m) | steps |
|---|---:|---:|---:|---:|---:|---:|
| bed | 123 | 0.553 | 0.240 | 0.265 | 2.82 | 248 |
| chair | 177 | 0.898 | 0.681 | 0.702 | 0.28 | 74 |
| couch | 229 | 0.795 | 0.581 | 0.654 | 0.59 | 102 |
| potted plant | 193 | 0.808 | 0.394 | 0.402 | 0.69 | 105 |
| toilet | 208 | 0.875 | 0.363 | 0.372 | 0.70 | 269 |
| tv | 70 | 0.586 | 0.432 | 0.447 | 2.12 | 130 |

### Provenance

- Source commit `f628a476ca33` with uncommitted changes at launch (`source-changes.diff` in the run directory); source sha256 `460987541927fa05...` (every `.py` under `sparx_agency/`).
- Method `sparx-rpt-llm-host-sweep`: local exploration `frontier`; room labels `qwen2.5:3b-instruct`, node oracle `qwen2.5:14b-instruct`
  (ollama at `http://127.0.0.1:11435`, seed 17, temperature 0.0); detector `yolo_world`.
- habitat-sim 0.2.4 (reference 0.1.5; `reference_sim_version_match: false`, acknowledged), numpy 1.26.4,
  scikit-fmm 2024.5.29; seed 17; lean records true; recording off (no video, no frames).
- GPU plan llm=gpu, yolo=gpu, habitat=gpu on NVIDIA GeForce RTX 4090 (pcn87652). Started 2026-10-18T10:53:55Z, finished 2026-10-19T12:40:17Z; 0 resume(s);
  episode wall time in total 1d01h (mean 1m33s per episode).
- Data: `results/gibson_val_20261018T105339Z/summary.json` (every metric with its interval, per scene and per category), `episodes.csv`
  (the per-episode scalars: id, scene, target, success, SPL, SoftSPL, path length, shortest path, start and final
  distance to goal, steps, STOP, termination, wall time, agent error) and `run.json` (the frozen configuration, the
  dataset file hashes and the command line). The run directory was `runs/gibson_val_20261018T105339Z` on pcn87652 (gitignored).

### Beside the published rows (`comparison.md` of the run)

### Full Gibson validation

SR/SPL are episode means including failures; DTG here is FMM distance to the 1 m success region. SoftSPL is supplementary.

| Method | SR (%) | SPL (%) | SoftSPL (%) | DTG (m) | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| SemExp | 65.7 | 33.9 | — | 1.47 | arXiv:2508.04678v1, Table 2, p.12; TF=False; NM=False. Paper-reported, protocol equivalence not established. |
| PONI | 73.6 | 41.0 | — | 1.25 | arXiv:2508.04678v1, Table 2, p.12; TF=False; NM=False. Paper-reported, protocol equivalence not established. |
| LGX | 31.0 | 5.2 | — | 4.78 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| LFG | 64.5 | 40.6 | — | 1.81 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| FBE | 64.1 | 28.3 | — | 1.78 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| SemUtil | 69.3 | 40.5 | — | 1.49 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=False. Paper-reported, protocol equivalence not established. |
| OSG-Nav | 73.4 | 38.6 | — | 1.72 | arXiv:2508.04678v1, Table 2, p.12; TF=True; NM=True. Paper-reported, protocol equivalence not established. |
| sparx-rpt-llm-host-sweep (ours) | 78.8 [76.2, 81.2] | 46.5 [44.1, 48.8] | 49.3 | 0.96 | this run: 1000 episodes, 95% intervals |

**Comparison caveat:** the published rows are reference values, not paired re-evaluations. ApexNav and SG-Nav do not report Gibson. TF = training-free; NM = non-metric. Our RPT adaptation is TF, not NM. GT pose/depth do not mean GT semantic detections.

#### Audit

- **full_split_completed:** True
- **episodes:** 1000
- **selected_episodes:** 1000
- **agent_errors:** 0
- **reference_sim_version_match:** False
- **exact_osg_protocol_reproduction:** False
- **resumes:** 0
- **resumes_with_config_changes:** []
- **primary_metrics:** ['SR', 'SPL', 'DTG (SemExp distance-to-success, m)']
- **supplementary_metrics:** ['SoftSPL', 'steps', 'wall_s']
- **total_wall_s:** 92776.04417727616
- **notes:** OSG's exact evaluator/episode hashes are not published in the supplied paper. SemExp and current PONI code differ; no direct SOTA win is asserted.
