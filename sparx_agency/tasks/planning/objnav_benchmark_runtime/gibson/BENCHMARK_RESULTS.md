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
