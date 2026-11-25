# MP3D v1 val, RTX 4090 profile: VLM frontier search (qwen2.5vl:7b) under ground-truth perception, detailed records

| | |
|---|---|
| episodes | 2195 of 2195 published (complete split) |
| scenes | 11 |
| pipeline variant | `rpt` |
| frontier VLM | `qwen2.5vl:7b` (digest `5ced39dfa4bac325dc1`), prompt `boxed-frontier-v2` |
| perception | ground-truth (on-object 0.25 m, min 64 px) |
| stair traversal | False |
| source commit | `56896d06622e7abb2452511dc90a73801ab2ea34` (tree NOT clean: see source-changes.diff in the run root) |
| protocol | mp3d-objectnav-v1-val/2, seed 17 |
| started / finished (UTC) | 2026-11-24T15:52:22Z / 2026-11-25T05:38:16Z (13.77 h) |
| run root (on the machine that ran it) | `/home/user1/GIT/TheAgency/runs/mp3d_v1_val_4090` |

## Metrics (means over every scored episode, failures included; 95 % intervals)

| group | n | SR [CI] | SPL [CI] | SoftSPL | DTG m | steps |
|---|---|---|---|---|---|---|
| **overall** | 2195 | 0.632 [0.612, 0.652] | 0.296 [0.284, 0.307] | 0.321 | 4.08 | 254 |
| bathtub | 9 | 0.444 [0.189, 0.733] | 0.243 [0.062, 0.446] | 0.260 | 9.19 | 358 |
| bed | 48 | 0.271 [0.166, 0.410] | 0.128 [0.067, 0.199] | 0.281 | 9.24 | 329 |
| cabinet | 180 | 0.733 [0.664, 0.793] | 0.361 [0.321, 0.401] | 0.396 | 2.25 | 203 |
| chair | 434 | 0.599 [0.552, 0.644] | 0.272 [0.246, 0.297] | 0.286 | 3.97 | 273 |
| chest_of_drawers | 63 | 0.524 [0.403, 0.642] | 0.231 [0.164, 0.303] | 0.242 | 7.09 | 343 |
| clothes | 25 | 0.520 [0.335, 0.700] | 0.254 [0.145, 0.368] | 0.289 | 7.61 | 308 |
| counter | 66 | 0.530 [0.412, 0.646] | 0.260 [0.194, 0.327] | 0.314 | 5.23 | 305 |
| cushion | 239 | 0.707 [0.647, 0.761] | 0.360 [0.324, 0.396] | 0.368 | 3.09 | 220 |
| fireplace | 30 | 0.633 [0.455, 0.781] | 0.334 [0.219, 0.451] | 0.373 | 5.06 | 246 |
| gym_equipment | 44 | 0.023 [0.004, 0.118] | 0.016 [0.000, 0.048] | 0.053 | 13.85 | 482 |
| picture | 161 | 0.621 [0.544, 0.692] | 0.315 [0.268, 0.360] | 0.378 | 2.28 | 196 |
| plant | 87 | 0.460 [0.359, 0.564] | 0.258 [0.192, 0.325] | 0.292 | 4.56 | 304 |
| seating | 87 | 0.759 [0.659, 0.836] | 0.199 [0.159, 0.241] | 0.245 | 2.31 | 211 |
| shower | 70 | 0.714 [0.599, 0.807] | 0.356 [0.292, 0.423] | 0.361 | 4.83 | 232 |
| sink | 82 | 0.695 [0.589, 0.784] | 0.324 [0.262, 0.387] | 0.332 | 3.66 | 276 |
| sofa | 53 | 0.377 [0.259, 0.512] | 0.162 [0.098, 0.231] | 0.231 | 7.81 | 346 |
| stool | 68 | 0.618 [0.499, 0.724] | 0.279 [0.214, 0.347] | 0.293 | 6.42 | 306 |
| table | 340 | 0.844 [0.802, 0.879] | 0.403 [0.377, 0.428] | 0.405 | 1.56 | 162 |
| toilet | 28 | 0.464 [0.295, 0.642] | 0.154 [0.084, 0.232] | 0.163 | 8.56 | 396 |
| towel | 73 | 0.425 [0.318, 0.539] | 0.165 [0.112, 0.218] | 0.183 | 6.54 | 374 |
| tv_monitor | 8 | 0.250 [0.071, 0.591] | 0.080 [0.000, 0.199] | 0.136 | 15.20 | 475 |

Terminations: {"stop": 1475, "step_limit": 720, "agent_error": 0}; agent errors: 0.

## Per scene

| scene | n | SR | SPL | DTG m |
|---|---|---|---|---|
| 2azQ1b91cZZ | 200 | 0.720 | 0.341 | 3.43 |
| 8194nk5LbLH | 201 | 0.259 | 0.120 | 11.08 |
| EU6Fwq7SyZv | 198 | 0.571 | 0.305 | 3.77 |
| QUCTc6BB5sX | 197 | 0.726 | 0.325 | 3.61 |
| TbHJrupSAjP | 200 | 0.650 | 0.288 | 3.27 |
| X7HyMhZNoso | 200 | 0.220 | 0.101 | 11.89 |
| Z6MFQCViBuw | 200 | 0.740 | 0.299 | 2.36 |
| oLBMNvg9in8 | 201 | 0.766 | 0.387 | 1.48 |
| pLe4wQe7qrG | 200 | 1.000 | 0.434 | 0.04 |
| x8F5xyUWy9e | 200 | 0.480 | 0.235 | 3.14 |
| zsNo4HB9uLZ | 198 | 0.823 | 0.422 | 0.77 |

Written by `mp3d/results_summary.py` from the run root's `episodes.jsonl` rows; the rows, step logs, LLM answer store and console logs stay with the run root.
