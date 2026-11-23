# MP3D mp3d-objectnav-v1-val/2: qwen2.5vl:7b (frontier VLM), ground-truth perception

| | |
|---|---|
| episodes | 180 of 2195 published (INCOMPLETE) |
| scenes | 8 |
| pipeline variant | `rpt` |
| frontier VLM | `qwen2.5vl:7b` (digest `5ced39dfa4bac325dc1`), prompt `boxed-frontier-v2` |
| perception | ground-truth (on-object 0.25 m, min 64 px) |
| stair traversal | False |
| source commit | `65a58747d23cbba2279ec53c54f098708e66544d` |
| protocol | mp3d-objectnav-v1-val/2, seed 17 |
| started / finished (UTC) | 2026-11-23T15:08:38Z / 2026-11-23T16:09:52Z (1.02 h) |
| run root (on the machine that ran it) | `/home/user1/GIT/TheAgency/runs/gt_rpt_pair_65a58747_seed17/rpt` |

## Metrics (means over every scored episode, failures included; 95 % intervals)

| group | n | SR [CI] | SPL [CI] | SoftSPL | DTG m | steps |
|---|---|---|---|---|---|---|
| **overall** | 180 | 0.828 [0.766, 0.876] | 0.377 [0.341, 0.413] | 0.400 | 1.40 | 184 |
| bed | 6 | 0.500 [0.188, 0.812] | 0.325 [0.082, 0.614] | 0.521 | 2.58 | 278 |
| cabinet | 14 | 1.000 [0.785, 1.000] | 0.503 [0.413, 0.595] | 0.497 | 0.04 | 96 |
| chair | 25 | 0.760 [0.566, 0.885] | 0.273 [0.192, 0.358] | 0.277 | 1.81 | 257 |
| chest_of_drawers | 8 | 0.750 [0.409, 0.929] | 0.339 [0.163, 0.528] | 0.342 | 2.58 | 268 |
| clothes | 1 | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.000] | 0.000 | 20.33 | 500 |
| counter | 4 | 0.250 [0.046, 0.699] | 0.057 [0.000, 0.172] | 0.200 | 4.16 | 385 |
| cushion | 20 | 0.850 [0.640, 0.948] | 0.428 [0.304, 0.548] | 0.430 | 0.98 | 180 |
| fireplace | 3 | 1.000 [0.439, 1.000] | 0.666 [0.630, 0.730] | 0.658 | 0.04 | 47 |
| gym_equipment | 1 | 1.000 [0.207, 1.000] | 0.208 [0.208, 0.208] | 0.204 | 0.04 | 96 |
| picture | 21 | 0.762 [0.549, 0.894] | 0.340 [0.237, 0.436] | 0.394 | 0.39 | 146 |
| plant | 7 | 1.000 [0.646, 1.000] | 0.518 [0.378, 0.655] | 0.515 | 0.04 | 145 |
| seating | 11 | 1.000 [0.741, 1.000] | 0.328 [0.218, 0.445] | 0.322 | 0.06 | 163 |
| shower | 5 | 0.600 [0.231, 0.882] | 0.328 [0.089, 0.567] | 0.327 | 11.08 | 254 |
| sink | 2 | 0.500 [0.095, 0.905] | 0.109 [0.000, 0.219] | 0.109 | 2.92 | 434 |
| sofa | 5 | 0.400 [0.118, 0.769] | 0.149 [0.000, 0.301] | 0.446 | 3.30 | 255 |
| stool | 3 | 0.667 [0.208, 0.939] | 0.473 [0.000, 0.730] | 0.470 | 3.33 | 216 |
| table | 37 | 1.000 [0.906, 1.000] | 0.469 [0.410, 0.531] | 0.462 | 0.03 | 104 |
| toilet | 1 | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.000] | 0.000 | 15.46 | 500 |
| towel | 6 | 1.000 [0.610, 1.000] | 0.435 [0.250, 0.627] | 0.433 | 0.04 | 240 |

Terminations: {"stop": 158, "step_limit": 22, "agent_error": 0}; agent errors: 0.

## Per scene

| scene | n | SR | SPL | DTG m |
|---|---|---|---|---|
| 2azQ1b91cZZ | 24 | 0.750 | 0.328 | 1.73 |
| 8194nk5LbLH | 6 | 1.000 | 0.505 | 0.03 |
| QUCTc6BB5sX | 21 | 0.762 | 0.344 | 4.14 |
| TbHJrupSAjP | 20 | 0.850 | 0.314 | 0.58 |
| Z6MFQCViBuw | 29 | 0.586 | 0.230 | 3.32 |
| oLBMNvg9in8 | 21 | 0.952 | 0.459 | 0.05 |
| pLe4wQe7qrG | 27 | 1.000 | 0.446 | 0.04 |
| zsNo4HB9uLZ | 32 | 0.875 | 0.473 | 0.43 |

Written by `mp3d/results_summary.py` from the run root's `episodes.jsonl` rows; the rows, step logs, LLM answer store and console logs stay with the run root.
