# HM3D v1 val, ground-truth perception, qwen2.5:3b-instruct in both LLM roles, prompt facts-qwen2.5-3b-v1 (SMOKE: one episode per scene)

| | |
|---|---|
| episodes | 20 of 2000 published (INCOMPLETE) |
| scenes | 20 |
| LLM, room labels | `qwen2.5:3b-instruct` |
| LLM, node oracle | `qwen2.5:3b-instruct` (Ollama digest `357c53fb659c5076de1`) |
| oracle prompt | version `facts-qwen2.5-3b-v1`, digest `2d025f019f981fe5`, tuned on `qwen2.5:3b-instruct`; tuned on the model that answered: **True** |
| perception | ground-truth (on-object 0.25 m, min 64 px) |
| stair traversal | False |
| source commit | `a13214e2f18169c59a24bc8a9ee03f4faf51dba4` (tree NOT clean: see source-changes.diff in the run root) |
| protocol | hm3d-objectnav-v1-val/2, seed 17 |
| started / finished (UTC) | 2026-11-17T15:06:59Z / 2026-11-17T15:14:00Z (0.12 h) |
| run root (on the machine that ran it) | `/home/user1/GIT/TheAgency/runs/hm3d_v1_gt_3b_vs_14b_smoke/stage1_3b` |

## Metrics (means over every scored episode, failures included; 95 % intervals)

| group | n | SR [CI] | SPL [CI] | SoftSPL | DTG m | steps |
|---|---|---|---|---|---|---|
| **overall** | 20 | 0.750 [0.531, 0.888] | 0.487 [0.336, 0.620] | 0.489 | 2.93 | 200 |
| bed | 4 | 0.250 [0.046, 0.699] | 0.069 [0.000, 0.206] | 0.088 | 6.67 | 422 |
| chair | 4 | 1.000 [0.510, 1.000] | 0.607 [0.366, 0.847] | 0.603 | 0.03 | 97 |
| plant | 1 | 1.000 [0.207, 1.000] | 0.813 [0.813, 0.813] | 0.807 | 0.03 | 46 |
| sofa | 3 | 1.000 [0.439, 1.000] | 0.742 [0.705, 0.763] | 0.735 | 0.04 | 91 |
| toilet | 5 | 0.600 [0.231, 0.882] | 0.447 [0.125, 0.768] | 0.449 | 6.29 | 261 |
| tv_monitor | 3 | 1.000 [0.439, 1.000] | 0.589 [0.390, 0.698] | 0.585 | 0.04 | 100 |

Terminations: {"stop": 15, "step_limit": 5, "agent_error": 0}; agent errors: 0.

## Per scene

| scene | n | SR | SPL | DTG m |
|---|---|---|---|---|
| 00800-TEEsavR23oF | 1 | 0.000 | 0.000 | 23.37 |
| 00802-wcojb4TFT35 | 1 | 1.000 | 0.794 | 0.02 |
| 00813-svBbv1Pavdk | 1 | 1.000 | 0.390 | 0.04 |
| 00814-p53SfW6mjZe | 1 | 1.000 | 0.758 | 0.05 |
| 00820-mL8ThkuaVTM | 1 | 0.000 | 0.000 | 11.30 |
| 00824-Dd4bFSTQ8gi | 1 | 1.000 | 0.469 | 0.02 |
| 00829-QaLdnwvtxbs | 1 | 1.000 | 0.263 | 0.02 |
| 00832-qyAac8rV8Zk | 1 | 1.000 | 0.698 | 0.02 |
| 00835-q3zU7Yy5E5s | 1 | 1.000 | 0.763 | 0.04 |
| 00839-zt1RVoi7PcG | 1 | 0.000 | 0.000 | 7.95 |
| 00843-DYehNKdT76V | 1 | 1.000 | 0.626 | 0.07 |
| 00848-ziup5kvtCCR | 1 | 1.000 | 0.813 | 0.03 |
| 00853-5cdEh9F2hJL | 1 | 1.000 | 0.679 | 0.05 |
| 00873-bxsVRursffK | 1 | 0.000 | 0.000 | 9.55 |
| 00876-mv2HUxq3B53 | 1 | 1.000 | 0.868 | 0.04 |
| 00877-4ok3usBNeis | 1 | 1.000 | 0.274 | 0.04 |
| 00878-XB4GS9ShBRE | 1 | 0.000 | 0.000 | 5.81 |
| 00880-Nfvxx8J5NCo | 1 | 1.000 | 0.813 | 0.04 |
| 00890-6s7QHgap2fW | 1 | 1.000 | 0.827 | 0.03 |
| 00891-cvZr5TUy5C5 | 1 | 1.000 | 0.705 | 0.03 |

Written by `hm3d/results_summary.py` from the run root's `episodes.jsonl` rows; the rows, step logs, LLM answer store and console logs stay with the run root.
