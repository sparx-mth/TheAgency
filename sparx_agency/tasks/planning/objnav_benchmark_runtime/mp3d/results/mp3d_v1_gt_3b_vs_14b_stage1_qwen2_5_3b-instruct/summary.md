# MP3D v1 val, ground-truth perception: qwen2.5:3b-instruct in both LLM roles, prompt facts-qwen2.5-3b-v1

| | |
|---|---|
| episodes | 2195 of 2195 published (complete split) |
| scenes | 11 |
| LLM, room labels | `qwen2.5:3b-instruct` |
| LLM, node oracle | `qwen2.5:3b-instruct` (Ollama digest `357c53fb659c5076de1`) |
| oracle prompt | version `facts-qwen2.5-3b-v1`, digest `2d025f019f981fe5`, tuned on `qwen2.5:3b-instruct`; tuned on the model that answered: **True** |
| perception | ground-truth (on-object 0.25 m, min 64 px) |
| stair traversal | False |
| source commit | `910e56f37d382b1cb053a563772dc191296f7406` (tree NOT clean: see source-changes.diff in the run root) |
| protocol | mp3d-objectnav-v1-val/2, seed 17 |
| started / finished (UTC) | 2026-11-19T06:10:52Z / 2026-11-19T18:51:23Z (12.68 h) |
| run root (on the machine that ran it) | `/home/user1/GIT/TheAgency/runs/mp3d_v1_gt_3b_vs_14b/stage1_qwen2_5_3b-instruct` |

## Metrics (means over every scored episode, failures included; 95 % intervals)

| group | n | SR [CI] | SPL [CI] | SoftSPL | DTG m | steps |
|---|---|---|---|---|---|---|
| **overall** | 2195 | 0.647 [0.627, 0.667] | 0.296 [0.284, 0.307] | 0.322 | 3.97 | 255 |
| bathtub | 9 | 0.444 [0.189, 0.733] | 0.214 [0.060, 0.402] | 0.253 | 8.01 | 371 |
| bed | 48 | 0.333 [0.217, 0.475] | 0.132 [0.079, 0.191] | 0.227 | 8.76 | 340 |
| cabinet | 180 | 0.733 [0.664, 0.793] | 0.336 [0.298, 0.374] | 0.374 | 2.14 | 214 |
| chair | 434 | 0.608 [0.562, 0.653] | 0.268 [0.243, 0.292] | 0.279 | 4.04 | 278 |
| chest_of_drawers | 63 | 0.476 [0.358, 0.597] | 0.272 [0.192, 0.350] | 0.285 | 8.59 | 321 |
| clothes | 25 | 0.480 [0.300, 0.665] | 0.167 [0.086, 0.255] | 0.212 | 7.87 | 367 |
| counter | 66 | 0.500 [0.383, 0.617] | 0.251 [0.179, 0.320] | 0.299 | 5.91 | 332 |
| cushion | 239 | 0.782 [0.726, 0.830] | 0.397 [0.361, 0.432] | 0.403 | 2.30 | 204 |
| fireplace | 30 | 0.767 [0.591, 0.882] | 0.376 [0.269, 0.478] | 0.388 | 3.89 | 236 |
| gym_equipment | 44 | 0.091 [0.036, 0.212] | 0.021 [0.002, 0.048] | 0.081 | 10.77 | 479 |
| picture | 161 | 0.696 [0.621, 0.761] | 0.314 [0.270, 0.356] | 0.384 | 1.73 | 175 |
| plant | 87 | 0.517 [0.414, 0.619] | 0.243 [0.184, 0.304] | 0.261 | 4.89 | 313 |
| seating | 87 | 0.724 [0.622, 0.807] | 0.197 [0.155, 0.238] | 0.253 | 2.69 | 188 |
| shower | 70 | 0.657 [0.540, 0.758] | 0.300 [0.237, 0.366] | 0.307 | 5.04 | 271 |
| sink | 82 | 0.683 [0.576, 0.774] | 0.277 [0.219, 0.336] | 0.311 | 3.35 | 278 |
| sofa | 53 | 0.434 [0.310, 0.567] | 0.231 [0.151, 0.319] | 0.307 | 7.46 | 311 |
| stool | 68 | 0.559 [0.441, 0.671] | 0.263 [0.199, 0.329] | 0.282 | 7.64 | 314 |
| table | 340 | 0.859 [0.818, 0.892] | 0.413 [0.389, 0.439] | 0.411 | 1.60 | 168 |
| toilet | 28 | 0.286 [0.153, 0.471] | 0.102 [0.040, 0.174] | 0.139 | 9.01 | 447 |
| towel | 73 | 0.425 [0.318, 0.539] | 0.169 [0.114, 0.224] | 0.216 | 5.47 | 373 |
| tv_monitor | 8 | 0.125 [0.022, 0.471] | 0.090 [0.000, 0.271] | 0.147 | 16.52 | 451 |

Terminations: {"stop": 1521, "step_limit": 674, "agent_error": 0}; agent errors: 0.

## Per scene

| scene | n | SR | SPL | DTG m |
|---|---|---|---|---|
| 2azQ1b91cZZ | 200 | 0.720 | 0.334 | 3.39 |
| 8194nk5LbLH | 201 | 0.259 | 0.119 | 11.20 |
| EU6Fwq7SyZv | 198 | 0.662 | 0.328 | 3.61 |
| QUCTc6BB5sX | 197 | 0.609 | 0.245 | 3.78 |
| TbHJrupSAjP | 200 | 0.670 | 0.305 | 3.41 |
| X7HyMhZNoso | 200 | 0.295 | 0.138 | 10.72 |
| Z6MFQCViBuw | 200 | 0.745 | 0.314 | 2.56 |
| oLBMNvg9in8 | 201 | 0.821 | 0.384 | 1.18 |
| pLe4wQe7qrG | 200 | 1.000 | 0.440 | 0.04 |
| x8F5xyUWy9e | 200 | 0.510 | 0.230 | 2.97 |
| zsNo4HB9uLZ | 198 | 0.828 | 0.417 | 0.72 |

Written by `mp3d/results_summary.py` from the run root's `episodes.jsonl` rows; the rows, step logs, LLM answer store and console logs stay with the run root.
