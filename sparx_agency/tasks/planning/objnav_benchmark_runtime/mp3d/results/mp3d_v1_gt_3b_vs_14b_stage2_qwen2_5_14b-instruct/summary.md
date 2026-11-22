# MP3D v1 val, ground-truth perception: qwen2.5:14b-instruct in both LLM roles, prompt facts-qwen2.5-3b-v1

| | |
|---|---|
| episodes | 2195 of 2195 published (complete split) |
| scenes | 11 |
| LLM, room labels | `qwen2.5:14b-instruct` |
| LLM, node oracle | `qwen2.5:14b-instruct` (Ollama digest `7cdf5a0187d5c58cc5d`) |
| oracle prompt | version `facts-qwen2.5-3b-v1`, digest `2d025f019f981fe5`, tuned on `qwen2.5:3b-instruct`; tuned on the model that answered: **False** |
| perception | ground-truth (on-object 0.25 m, min 64 px) |
| stair traversal | False |
| source commit | `222595a9d94d41023f15d7f160055319c49b9f12` (tree NOT clean: see source-changes.diff in the run root) |
| protocol | mp3d-objectnav-v1-val/2, seed 17 |
| started / finished (UTC) | 2026-11-19T18:53:33Z / 2026-11-22T00:40:12Z (53.78 h) |
| run root (on the machine that ran it) | `/home/user1/GIT/TheAgency/runs/mp3d_v1_gt_3b_vs_14b/stage2_qwen2_5_14b-instruct` |

## Metrics (means over every scored episode, failures included; 95 % intervals)

| group | n | SR [CI] | SPL [CI] | SoftSPL | DTG m | steps |
|---|---|---|---|---|---|---|
| **overall** | 2195 | 0.651 [0.631, 0.671] | 0.299 [0.288, 0.311] | 0.327 | 3.87 | 252 |
| bathtub | 9 | 0.667 [0.354, 0.879] | 0.333 [0.161, 0.504] | 0.346 | 5.77 | 304 |
| bed | 48 | 0.312 [0.199, 0.453] | 0.136 [0.074, 0.203] | 0.222 | 10.13 | 354 |
| cabinet | 180 | 0.728 [0.659, 0.788] | 0.333 [0.295, 0.370] | 0.379 | 2.03 | 211 |
| chair | 434 | 0.613 [0.566, 0.658] | 0.274 [0.250, 0.299] | 0.284 | 4.01 | 276 |
| chest_of_drawers | 63 | 0.476 [0.358, 0.597] | 0.249 [0.170, 0.326] | 0.274 | 7.77 | 337 |
| clothes | 25 | 0.480 [0.300, 0.665] | 0.192 [0.104, 0.288] | 0.265 | 5.61 | 341 |
| counter | 66 | 0.470 [0.354, 0.588] | 0.258 [0.184, 0.330] | 0.323 | 5.94 | 315 |
| cushion | 239 | 0.787 [0.730, 0.834] | 0.406 [0.369, 0.440] | 0.412 | 2.08 | 201 |
| fireplace | 30 | 0.700 [0.521, 0.833] | 0.291 [0.198, 0.385] | 0.305 | 4.35 | 276 |
| gym_equipment | 44 | 0.091 [0.036, 0.212] | 0.028 [0.002, 0.064] | 0.088 | 11.44 | 478 |
| picture | 161 | 0.677 [0.601, 0.744] | 0.300 [0.256, 0.343] | 0.393 | 1.68 | 170 |
| plant | 87 | 0.506 [0.403, 0.608] | 0.246 [0.186, 0.310] | 0.263 | 4.68 | 312 |
| seating | 87 | 0.759 [0.659, 0.836] | 0.209 [0.167, 0.251] | 0.261 | 2.28 | 190 |
| shower | 70 | 0.700 [0.585, 0.795] | 0.346 [0.277, 0.416] | 0.355 | 4.65 | 256 |
| sink | 82 | 0.732 [0.627, 0.816] | 0.296 [0.237, 0.360] | 0.316 | 3.37 | 282 |
| sofa | 53 | 0.434 [0.310, 0.567] | 0.221 [0.140, 0.305] | 0.287 | 7.59 | 308 |
| stool | 68 | 0.559 [0.441, 0.671] | 0.263 [0.200, 0.329] | 0.286 | 7.16 | 309 |
| table | 340 | 0.853 [0.811, 0.887] | 0.412 [0.387, 0.437] | 0.412 | 1.57 | 166 |
| toilet | 28 | 0.393 [0.236, 0.576] | 0.165 [0.082, 0.256] | 0.197 | 7.89 | 409 |
| towel | 73 | 0.452 [0.343, 0.566] | 0.170 [0.118, 0.225] | 0.202 | 6.08 | 370 |
| tv_monitor | 8 | 0.250 [0.071, 0.591] | 0.157 [0.000, 0.381] | 0.178 | 17.72 | 445 |

Terminations: {"stop": 1534, "step_limit": 661, "agent_error": 0}; agent errors: 0.

## Per scene

| scene | n | SR | SPL | DTG m |
|---|---|---|---|---|
| 2azQ1b91cZZ | 200 | 0.725 | 0.344 | 3.10 |
| 8194nk5LbLH | 201 | 0.259 | 0.121 | 11.29 |
| EU6Fwq7SyZv | 198 | 0.646 | 0.323 | 3.64 |
| QUCTc6BB5sX | 197 | 0.670 | 0.282 | 3.28 |
| TbHJrupSAjP | 200 | 0.670 | 0.297 | 3.15 |
| X7HyMhZNoso | 200 | 0.295 | 0.150 | 10.41 |
| Z6MFQCViBuw | 200 | 0.720 | 0.308 | 2.85 |
| oLBMNvg9in8 | 201 | 0.811 | 0.365 | 1.20 |
| pLe4wQe7qrG | 200 | 1.000 | 0.451 | 0.04 |
| x8F5xyUWy9e | 200 | 0.510 | 0.233 | 3.07 |
| zsNo4HB9uLZ | 198 | 0.859 | 0.420 | 0.51 |

Written by `mp3d/results_summary.py` from the run root's `episodes.jsonl` rows; the rows, step logs, LLM answer store and console logs stay with the run root.
