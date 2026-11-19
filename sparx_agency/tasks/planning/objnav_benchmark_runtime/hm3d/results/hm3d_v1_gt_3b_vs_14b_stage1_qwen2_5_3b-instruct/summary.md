# HM3D v1 val, ground-truth perception, qwen2.5:3b-instruct in both LLM roles, prompt facts-qwen2.5-3b-v1

| | |
|---|---|
| episodes | 2000 of 2000 published (complete split) |
| scenes | 20 |
| LLM, room labels | `qwen2.5:3b-instruct` |
| LLM, node oracle | `qwen2.5:3b-instruct` (Ollama digest `357c53fb659c5076de1`) |
| oracle prompt | version `facts-qwen2.5-3b-v1`, digest `2d025f019f981fe5`, tuned on `qwen2.5:3b-instruct`; tuned on the model that answered: **True** |
| perception | ground-truth (on-object 0.25 m, min 64 px) |
| stair traversal | False |
| source commit | `0fb40f21b173bb91581b55a52bc027f1eb88ea53` |
| protocol | hm3d-objectnav-v1-val/2, seed 17 |
| started / finished (UTC) | 2026-11-17T15:50:00Z / 2026-11-18T11:52:24Z (20.04 h) |
| run root (on the machine that ran it) | `/home/user1/GIT/TheAgency/runs/hm3d_v1_gt_3b_vs_14b/stage1_3b` |

## Metrics (means over every scored episode, failures included; 95 % intervals)

| group | n | SR [CI] | SPL [CI] | SoftSPL | DTG m | steps |
|---|---|---|---|---|---|---|
| **overall** | 2000 | 0.714 [0.694, 0.733] | 0.374 [0.361, 0.387] | 0.385 | 3.10 | 256 |
| bed | 433 | 0.642 [0.596, 0.686] | 0.306 [0.278, 0.333] | 0.316 | 4.16 | 293 |
| chair | 428 | 0.848 [0.811, 0.879] | 0.491 [0.463, 0.520] | 0.494 | 1.17 | 175 |
| plant | 84 | 0.619 [0.512, 0.716] | 0.304 [0.237, 0.372] | 0.313 | 7.01 | 290 |
| sofa | 376 | 0.811 [0.769, 0.848] | 0.490 [0.457, 0.520] | 0.494 | 1.81 | 199 |
| toilet | 398 | 0.656 [0.608, 0.701] | 0.273 [0.247, 0.301] | 0.293 | 3.58 | 312 |
| tv_monitor | 281 | 0.601 [0.543, 0.657] | 0.310 [0.274, 0.352] | 0.334 | 4.30 | 310 |

Terminations: {"stop": 1438, "step_limit": 562, "agent_error": 0}; agent errors: 0.

## Per scene

| scene | n | SR | SPL | DTG m |
|---|---|---|---|---|
| 00800-TEEsavR23oF | 99 | 0.697 | 0.378 | 6.57 |
| 00802-wcojb4TFT35 | 99 | 0.727 | 0.395 | 1.22 |
| 00813-svBbv1Pavdk | 99 | 0.990 | 0.546 | 0.12 |
| 00814-p53SfW6mjZe | 99 | 0.465 | 0.246 | 5.72 |
| 00820-mL8ThkuaVTM | 99 | 0.404 | 0.175 | 5.64 |
| 00824-Dd4bFSTQ8gi | 99 | 0.949 | 0.457 | 0.16 |
| 00829-QaLdnwvtxbs | 99 | 0.990 | 0.565 | 0.04 |
| 00832-qyAac8rV8Zk | 99 | 0.515 | 0.235 | 6.27 |
| 00835-q3zU7Yy5E5s | 99 | 0.586 | 0.355 | 2.75 |
| 00839-zt1RVoi7PcG | 119 | 0.782 | 0.473 | 2.21 |
| 00843-DYehNKdT76V | 99 | 0.747 | 0.395 | 1.88 |
| 00848-ziup5kvtCCR | 99 | 0.838 | 0.408 | 0.98 |
| 00853-5cdEh9F2hJL | 99 | 0.859 | 0.408 | 1.29 |
| 00873-bxsVRursffK | 99 | 0.707 | 0.369 | 3.21 |
| 00876-mv2HUxq3B53 | 99 | 0.848 | 0.465 | 1.92 |
| 00877-4ok3usBNeis | 99 | 0.869 | 0.473 | 1.24 |
| 00878-XB4GS9ShBRE | 99 | 0.424 | 0.212 | 4.18 |
| 00880-Nfvxx8J5NCo | 99 | 0.879 | 0.425 | 0.79 |
| 00890-6s7QHgap2fW | 99 | 0.424 | 0.218 | 8.76 |
| 00891-cvZr5TUy5C5 | 99 | 0.566 | 0.266 | 7.24 |

Written by `hm3d/results_summary.py` from the run root's `episodes.jsonl` rows; the rows, step logs, LLM answer store and console logs stay with the run root.
