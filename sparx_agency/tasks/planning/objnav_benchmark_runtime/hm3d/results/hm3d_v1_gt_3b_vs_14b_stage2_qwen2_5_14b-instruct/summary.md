# HM3D v1 val, ground-truth perception, qwen2.5:14b-instruct in both LLM roles, prompt facts-qwen2.5-3b-v1

| | |
|---|---|
| episodes | 2000 of 2000 published (complete split) |
| scenes | 20 |
| LLM, room labels | `qwen2.5:14b-instruct` |
| LLM, node oracle | `qwen2.5:14b-instruct` (Ollama digest `7cdf5a0187d5c58cc5d`) |
| oracle prompt | version `facts-qwen2.5-3b-v1`, digest `2d025f019f981fe5`, tuned on `qwen2.5:3b-instruct`; tuned on the model that answered: **False** |
| perception | ground-truth (on-object 0.25 m, min 64 px) |
| stair traversal | False |
| source commit | `cba77767dce1e25700b4120d295838e507e351c3` |
| protocol | hm3d-objectnav-v1-val/2, seed 17 |
| started / finished (UTC) | 2026-11-18T11:52:48Z / 2026-11-19T08:11:56Z (20.32 h) |
| run root (on the machine that ran it) | `/home/user1/GIT/TheAgency/runs/hm3d_v1_gt_3b_vs_14b/stage2_14b` |

## Metrics (means over every scored episode, failures included; 95 % intervals)

| group | n | SR [CI] | SPL [CI] | SoftSPL | DTG m | steps |
|---|---|---|---|---|---|---|
| **overall** | 2000 | 0.713 [0.692, 0.732] | 0.382 [0.368, 0.396] | 0.392 | 3.21 | 253 |
| bed | 433 | 0.640 [0.593, 0.684] | 0.318 [0.289, 0.345] | 0.327 | 4.32 | 288 |
| chair | 428 | 0.850 [0.814, 0.881] | 0.501 [0.472, 0.529] | 0.502 | 1.23 | 172 |
| plant | 84 | 0.548 [0.441, 0.650] | 0.270 [0.204, 0.341] | 0.281 | 7.88 | 302 |
| sofa | 376 | 0.809 [0.766, 0.845] | 0.495 [0.461, 0.528] | 0.499 | 1.96 | 196 |
| toilet | 398 | 0.648 [0.600, 0.694] | 0.288 [0.260, 0.316] | 0.307 | 3.61 | 306 |
| tv_monitor | 281 | 0.626 [0.568, 0.681] | 0.315 [0.280, 0.355] | 0.336 | 4.25 | 310 |

Terminations: {"stop": 1435, "step_limit": 565, "agent_error": 0}; agent errors: 0.

## Per scene

| scene | n | SR | SPL | DTG m |
|---|---|---|---|---|
| 00800-TEEsavR23oF | 99 | 0.707 | 0.397 | 6.60 |
| 00802-wcojb4TFT35 | 99 | 0.747 | 0.418 | 1.24 |
| 00813-svBbv1Pavdk | 99 | 0.980 | 0.547 | 0.19 |
| 00814-p53SfW6mjZe | 99 | 0.455 | 0.254 | 6.00 |
| 00820-mL8ThkuaVTM | 99 | 0.394 | 0.161 | 5.80 |
| 00824-Dd4bFSTQ8gi | 99 | 0.929 | 0.501 | 0.32 |
| 00829-QaLdnwvtxbs | 99 | 1.000 | 0.560 | 0.03 |
| 00832-qyAac8rV8Zk | 99 | 0.515 | 0.261 | 6.38 |
| 00835-q3zU7Yy5E5s | 99 | 0.606 | 0.357 | 2.89 |
| 00839-zt1RVoi7PcG | 119 | 0.798 | 0.466 | 2.19 |
| 00843-DYehNKdT76V | 99 | 0.758 | 0.400 | 2.18 |
| 00848-ziup5kvtCCR | 99 | 0.828 | 0.415 | 1.02 |
| 00853-5cdEh9F2hJL | 99 | 0.889 | 0.416 | 0.76 |
| 00873-bxsVRursffK | 99 | 0.707 | 0.359 | 3.41 |
| 00876-mv2HUxq3B53 | 99 | 0.869 | 0.497 | 1.91 |
| 00877-4ok3usBNeis | 99 | 0.838 | 0.471 | 1.40 |
| 00878-XB4GS9ShBRE | 99 | 0.424 | 0.212 | 3.92 |
| 00880-Nfvxx8J5NCo | 99 | 0.879 | 0.451 | 1.02 |
| 00890-6s7QHgap2fW | 99 | 0.333 | 0.191 | 9.95 |
| 00891-cvZr5TUy5C5 | 99 | 0.576 | 0.287 | 7.26 |

Written by `hm3d/results_summary.py` from the run root's `episodes.jsonl` rows; the rows, step logs, LLM answer store and console logs stay with the run root.
