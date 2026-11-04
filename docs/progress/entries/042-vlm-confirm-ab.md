# 042 - VLM confirmation before STOP: paired A/B against the baseline

**Branch:** `feat/objnav-habitat-hm3d-nadav` (code in commit `41eb1bde`, "Ask a VLM to confirm the target before the
closing commits to STOP")
**Status:** 120-pair A/B on three scenes **done**; full 2,000-episode A/B **running** (2 of 40 arm-runs finished)
**Follows:** entry 041 (reproducible LLM answers, the paired A/B report)

This entry was written on 2026-11-04 by a session that did **not** launch the runs. Everything below is reconstructed
from the launch scripts, logs and result files left in the scratchpad. Where something could not be verified it says
so.

## 1. What is being tested

The object-search closing (`core/planning/object_search/closing/target_closing.py`) walks up to a detector candidate and
calls STOP. With `target_closing.vlm_confirm = true` it first crops the candidate's box out of the RGB frame, grows it by
`vlm_crop_margin` of its size on every side, and asks a vision-language model one yes/no question ("is this a
`<label>`?"). A "no" vetoes the STOP: the candidate is released and a rejection disc of
`vlm_rejection_radius_factor` x the usual radius is left behind so the search does not walk back to it. "yes" and
"unsure" (which includes any service error) let the STOP through, so a dead VLM server reproduces the baseline.

Code: `core/planning/object_search/closing/vlm_confirmation.py` (protocol, crop, question, answer parsing; 3.8-safe)
and `tasks/planning/objnav_benchmark_runtime/vlm_confirmer.py` (`OllamaConfirmer`, attached by `policy_factory.py`).
Each query is recorded in the episode row at `agent_info.policy.target_closing.vlm_checks[]` with the answer and
latency.

Defaults: `vlm_model = "qwen3-vl:8b"`, `vlm_crop_margin = 0.3`, `vlm_rejection_radius_factor = 1.0`, Ollama at
`http://127.0.0.1:11434`.

## 2. Setup shared by every run

- Benchmark: HM3D ObjectNav **v1 val**, `tasks/planning/objnav_benchmark_runtime/hm3d/run_benchmark.sh --version v1`,
  one scene per invocation, seed 17.
- Agent: `sparx-rpt-llm-host-sweep` - frontier explorer, YOLO-World X-v2 detector (conf 0.05, CLIP ViT-B/32 text
  encoder, served on `127.0.0.1:18092` from `~/.venvs/detector-eval`), room labels by `qwen2.5:3b-instruct`, node
  oracle by `qwen2.5:14b-instruct`, both via the host Ollama. The VLM arms additionally load `qwen3-vl:8b` (or `:2b`).
- Interpreters: Habitat side `~/miniconda3/envs/objnav-habitat/bin/python`; detector `~/.venvs/detector-eval`.
- Hardware: one RTX 4090 (24 GB), shared by Habitat, the detector and all Ollama models (`ALLOW_SHARED_GPU=1`).
- Between arms every Ollama model is unloaded (`ollama stop`) so each arm starts from a cold runner (entry 041: a
  warm runner's answers depend on request history).
- Scratchpad root (all outputs, not in git):
  `/tmp/claude-1000/-home-user1-GIT-habitat-matterport3d-dataset/6a2a27b3-66d6-42e1-b471-bb7db3c99f5b/scratchpad/`
  - referred to as `$S` below.
- Report tool: `python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.paired_ab <dirA> <dirB> --json out.json`.
  Pairs episodes by id, gives SR/SPL deltas with bootstrap 95 % CIs, McNemar (SR) and sign-flip (SPL) p-values,
  per-episode flips, and a sample-size estimate.

## 3. Arms

Episode subset for the small A/B: `$S/episodes_start_floor.txt` - 120 episodes, 40 each from scenes
`00824-Dd4bFSTQ8gi`, `00835-q3zU7Yy5E5s`, `00843-DYehNKdT76V` (start-floor episodes only).

| arm | policy config | purpose | launched by |
|---|---|---|---|
| `base` | none | the unchanged method | `ab_vlm.sh` |
| `vlm` | `{"vlm_confirm": true}` (qwen3-vl:8b, margin 0.3, radius x1) | the default VLM arm | `ab_vlm.sh` |
| `base2` | none | re-run of base: noise floor | `queue_next.sh` |
| `vlm2b` | `vlm_confirm` + `vlm_model: qwen3-vl:2b` | smaller/faster model | `queue_next.sh` |
| `vlmctx` | `vlm_confirm` + `vlm_crop_margin: 1.0` | wider crop, more context | `queue_more.sh` |
| `vlmr2` | `vlm_confirm` + `vlm_rejection_radius_factor: 2.0` | larger rejection disc after a veto | `queue_more.sh` |
| `vlm_b` | same as `vlm` | re-run of vlm: noise floor | `queue_more.sh` |

Outputs: `$S/ab_vlm/<arm>/<scene>/{episodes.jsonl,summary.json,run.log,...}`. The scripts chained on each other
(`queue_next.sh` waited for `ab_vlm.sh`, `queue_more.sh` for `queue_next.sh`, `queue_full.sh` for `queue_more.sh`) and
each printed a sentinel (`AB_DONE`, `QUEUE_DONE`, `MORE_DONE`) on completion. All three sentinels are present in the
logs; **`ab_vlm/` is complete**: 7 arms x 3 scenes, 21 `summary.json`, 6 paired reports written 2026-11-04 04:10.

`queue_next.sh` also ran a frame-level VLM evaluation independent of navigation (section 6).

## 4. Results: 120 paired episodes (FINAL)

From `$S/ab_vlm/report_base_vs_*.txt`:

| B vs `base` (A) | SR A | SR B | dSR (95 % CI) | McNemar p | SPL A | SPL B | dSPL | flips |
|---|---|---|---|---|---|---|---|---|
| `vlm` | 0.633 | 0.658 | +0.025 [0.000, 0.058] | 0.25 | 0.367 | 0.372 | +0.005 | 3 to success, 0 to failure |
| `vlm2b` | 0.633 | 0.658 | +0.025 [0.000, 0.058] | 0.25 | 0.367 | 0.372 | +0.005 | same 3 |
| `vlmr2` | 0.633 | 0.658 | +0.025 [0.000, 0.058] | 0.25 | 0.367 | 0.372 | +0.005 | same 3 |
| `vlmctx` | 0.633 | 0.658 | +0.025 [0.000, 0.058] | 0.25 | 0.367 | 0.370 | +0.003 | same 3 |
| `base2` | 0.633 | 0.633 | 0 | 1 | 0.367 | 0.367 | 0 | 0 |
| `vlm_b` vs `vlm` | 0.658 | 0.658 | 0 | 1 | 0.372 | 0.372 | 0 | 0 |

- The three episodes that flip to success in every VLM arm: `00824-Dd4bFSTQ8gi/000021`, `00824-Dd4bFSTQ8gi/000076`,
  `00843-DYehNKdT76V/000076`.
- **Reproducibility**: `base` vs `base2` and `vlm` vs `vlm_b` are identical to the episode (0 differences in success or
  step count). With cold Ollama runners the whole pipeline is deterministic, so every difference below is the
  feature, not noise.
- **Statistics**: +2.5 SR points is not significant (p = 0.25; the CI touches zero). The tool's own estimate: 312
  pairs to detect the observed effect, 489 pairs to detect a 2-point effect. That is why the full split is running.
- **How often the VLM is consulted** (from `vlm_checks`): 82 queries in 79 of 120 episodes; 78 "yes", 4 "no" in
  `vlm`, `vlm2b`, `vlmr2`, `vlm_b`; 84 queries, 6 "no" in `vlmctx`. Median latency 1.0 s (8B), 0.6 s (2B).
- **The 2B model returned the same 82 answers as the 8B model.** Plausible on 82 easy crops, but it means this
  experiment cannot separate the two models; the frame-level evaluation (section 6) does.

## 5. The "never glanced" integrity flag - resolved, not a bug in the runs

Every VLM report says:

> integrity: B never glanced in 120 episodes; 4 of them end differently from A ...
> decision: not readable: episodes in which B never glanced end differently from A

This looks alarming but is a mismatch between the report and the feature. `paired_ab.py` was written for the
*geometric-glance* A/B (entry 041); its integrity check counts `agent_info.policy.glances.stats` and demands that an
episode in which arm B took no geometric glance end exactly like arm A. The VLM arms never take geometric glances (the
feature is off in both arms), so the check fires on every episode that differs for any reason.

Cross-checking against `vlm_checks` shows the differences are exactly where they should be. In every arm the set of
episodes that differ from `base` (success or step count) **equals** the set of episodes in which the VLM answered
"no" at least once:

| arm | episodes with a "no" | episodes differing from base |
|---|---|---|
| `vlm`, `vlm2b`, `vlmr2`, `vlm_b` | 00824/000001, 00824/000021, 00824/000076, 00843/000076 | the same four |
| `vlmctx` | the four above + 00824/000017, 00824/000025 | the same six |
| `full2000/vlm/00877` (section 7) | 4 episodes (6 "no" answers) | the same four (0 SR flips) |

So the VLM veto is the only thing that changes an episode; three of the four vetoes turned a failure into a success,
one (`00824/000001`) changed the path without changing the outcome. There is no hidden side channel.

**Action item**: `paired_ab.py` needs a `--mode vlm` (or a generic "arm B's intervention log") so its integrity check
reads `target_closing.vlm_checks` instead of `glances.stats`; until then ignore the "not readable" decision line for
VLM reports and use the table above.

## 6. Frame-level VLM evaluation (FINAL)

Independent of navigation, `queue_next.sh` ran `$S/detector_study/vlm_confirm_eval.py` over the detector-study frames
(`$S/detector_study/data`, HM3D frames with ground-truth object lists). Each frame was shown whole to the VLM, which
was asked which of the six target categories it contains. Output: `$S/vlm_qwen3vl8b.jsonl` (2,324 frames) and
`$S/vlm_qwen3vl2b.jsonl` (2,021 frames; fewer because the run was cut by the queue, not by an error as far as the log
shows - unverified).

Per (frame, category) decisions against GT, computed for this entry:

| model | frames | TP | FP | FN | TN | precision | recall | median s / frame |
|---|---|---|---|---|---|---|---|---|
| qwen3-vl:8b | 2,324 | 1,479 | 779 | 300 | 11,386 | 0.655 | 0.831 | 6.0 |
| qwen3-vl:2b | 2,021 | 1,284 | 767 | 276 | 9,799 | 0.626 | 0.823 | 2.4 |

Caveat: this measures whole-frame presence, not the cropped yes/no the closing asks, and "potted plant" was matched to
GT "plant". It says the VLM is a weak filter on its own (precision 0.63-0.66) and that 2B is close to 8B at 2.5x the
speed. The closing only asks it about a detector candidate at close range, where the crops are easier - consistent
with 78/82 "yes" in section 4.

## 7. Full split: 2,000 episodes, 20 scenes (RUNNING)

`$S/queue_full.sh`: for each of the 20 val scenes, run `base` then `vlm` (default 8B config, all episodes of the
scene), then the paired report `$S/full2000/report_full_base_vs_vlm.{txt,json}` and the sentinel `FULL_DONE`.
Outputs in `$S/full2000/<arm>/<scene>/`.

Status at 2026-11-04 11:24 IST (read-only check; `queue_full.sh` PID 251479 alive, started 2026-11-02 ~17:19):

| arm-run | state | wall time |
|---|---|---|
| `base/00877-4ok3usBNeis` | done (99 episodes) | 04:11 -> 06:59 (2 h 48 m) |
| `vlm/00877-4ok3usBNeis` | done (99 episodes) | 07:00 -> 09:50 (2 h 50 m) |
| `base/00853-5cdEh9F2hJL` | running, 72/99, run's ETA ~34 min | started 09:51 |
| `vlm/00853` + 18 scenes x 2 arms | queued | 37 arm-runs |

(Times above are IST. Note that `run.log` stamps are **UTC**, two hours behind: a `[07:51:02]` header is a 09:51 IST
start. Each `run.log` has exactly one header; there were no restarted attempts.)

Rough ETA for the whole split: ~2.8 h per 99-episode arm-run (longer for scenes with more episodes), so 37 arm-runs
take about **4-5 days** from now.

Partial result, scene 00877 only (99 pairs, `paired_ab` run for this entry):

| | SR | SPL | flips | VLM queries |
|---|---|---|---|---|
| base | 0.737 | 0.377 | - | - |
| vlm | 0.737 | 0.376 | 0 / 0 | 71 queries in 65 episodes, 6 "no" |

Four episodes differ in path (the 4 with a veto), none in outcome. SPL -0.002 (p = 0.12): the vetoes cost a little
path without gaining a success here.

## 8. What we know so far

- The VLM veto fires in roughly 3-6 % of episodes and, when it does, it is the only thing that changes the episode.
- On 120 start-floor episodes it rescued 3 failures and broke none (+2.5 SR points, not significant). On the first
  full scene it rescued none and cost 0.2 SPL points.
- Model size (8B vs 2B), crop margin and rejection radius did not change a single outcome on the small set.
- Pipeline determinism is confirmed (two identical re-runs per arm), so the full split needs no repeats.
- The decision rule from entry 041 applies: keep the feature ON only if the full-split SR gain is significant at the
  95 % level. Current evidence does not support ON.

## 9. Open items

1. `paired_ab.py`: add a VLM-aware integrity check (section 5) and stop printing the geometric-glance decision for
   VLM arms.
2. Explain the 2B == 8B identity on all 82 closing queries (dump the crops for the 4 "no" cases and a sample of the
   "yes" cases; confirm Ollama actually served `qwen3-vl:2b` - the latency halving says it did).
3. When `FULL_DONE` appears: read `$S/full2000/report_full_base_vs_vlm.txt`, redo the section-5 cross-check on all 20
   scenes, and decide ON/OFF per the entry-041 rule.
4. Nothing under `$S` is in git. If the numbers are to survive, copy the `report_*.json`, the per-scene
   `summary.json` files and the two `vlm_qwen3vl*.jsonl` files into the repo's results area.
