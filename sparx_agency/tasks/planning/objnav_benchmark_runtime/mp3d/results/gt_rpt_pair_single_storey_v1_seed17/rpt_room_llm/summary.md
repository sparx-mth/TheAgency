# MP3D mp3d-objectnav-v1-val/2: qwen2.5:3b-instruct (oracle) / qwen2.5:3b-instruct (labels), ground-truth perception

| | |
|---|---|
| episodes | 180 of 2195 published (INCOMPLETE) |
| scenes | 8 |
| pipeline variant | `rpt_room_llm` |
| LLM, room labels | `qwen2.5:3b-instruct` |
| LLM, node oracle | `qwen2.5:3b-instruct` (Ollama digest `357c53fb659c5076de1`) |
| oracle prompt | version `facts-qwen2.5-3b-v1`, digest `2d025f019f981fe5`, tuned on `qwen2.5:3b-instruct`; tuned on the model that answered: **True** |
| perception | ground-truth (on-object 0.25 m, min 64 px) |
| stair traversal | False |
| source commit | `65a58747d23cbba2279ec53c54f098708e66544d` |
| protocol | mp3d-objectnav-v1-val/2, seed 17 |
| started / finished (UTC) | 2026-11-23T13:27:50Z / 2026-11-23T15:08:13Z (1.67 h) |
| run root (on the machine that ran it) | `/home/user1/GIT/TheAgency/runs/gt_rpt_pair_65a58747_seed17/rpt_room_llm` |

## Metrics (means over every scored episode, failures included; 95 % intervals)

| group | n | SR [CI] | SPL [CI] | SoftSPL | DTG m | steps |
|---|---|---|---|---|---|---|
| **overall** | 180 | 0.811 [0.748, 0.862] | 0.345 [0.308, 0.381] | 0.371 | 1.84 | 200 |
| bed | 6 | 0.500 [0.188, 0.812] | 0.118 [0.034, 0.217] | 0.273 | 8.65 | 343 |
| cabinet | 14 | 0.929 [0.685, 0.987] | 0.438 [0.324, 0.541] | 0.442 | 0.10 | 135 |
| chair | 25 | 0.680 [0.484, 0.828] | 0.237 [0.156, 0.320] | 0.235 | 2.79 | 285 |
| chest_of_drawers | 8 | 0.500 [0.215, 0.785] | 0.229 [0.059, 0.420] | 0.237 | 4.90 | 352 |
| clothes | 1 | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.000] | 0.000 | 26.26 | 500 |
| counter | 4 | 0.500 [0.150, 0.850] | 0.130 [0.000, 0.261] | 0.240 | 1.23 | 411 |
| cushion | 20 | 1.000 [0.839, 1.000] | 0.528 [0.424, 0.618] | 0.525 | 0.03 | 146 |
| fireplace | 3 | 1.000 [0.439, 1.000] | 0.675 [0.617, 0.771] | 0.669 | 0.03 | 48 |
| gym_equipment | 1 | 1.000 [0.207, 1.000] | 0.056 [0.056, 0.056] | 0.055 | 0.01 | 318 |
| picture | 21 | 0.762 [0.549, 0.894] | 0.340 [0.233, 0.450] | 0.444 | 0.40 | 112 |
| plant | 7 | 0.857 [0.487, 0.974] | 0.281 [0.158, 0.389] | 0.279 | 2.62 | 281 |
| seating | 11 | 0.909 [0.623, 0.984] | 0.349 [0.235, 0.460] | 0.358 | 0.67 | 124 |
| shower | 5 | 0.600 [0.231, 0.882] | 0.220 [0.053, 0.388] | 0.218 | 7.77 | 288 |
| sink | 2 | 0.500 [0.095, 0.905] | 0.118 [0.000, 0.237] | 0.136 | 4.87 | 358 |
| sofa | 5 | 0.400 [0.118, 0.769] | 0.170 [0.000, 0.368] | 0.247 | 9.90 | 298 |
| stool | 3 | 1.000 [0.439, 1.000] | 0.508 [0.443, 0.573] | 0.505 | 0.02 | 122 |
| table | 37 | 1.000 [0.906, 1.000] | 0.446 [0.388, 0.506] | 0.439 | 0.03 | 123 |
| toilet | 1 | 1.000 [0.207, 1.000] | 0.281 [0.281, 0.281] | 0.279 | 0.05 | 476 |
| towel | 6 | 0.667 [0.300, 0.903] | 0.152 [0.061, 0.244] | 0.305 | 0.67 | 315 |

Terminations: {"stop": 156, "step_limit": 24, "agent_error": 0}; agent errors: 0.

## Per scene

| scene | n | SR | SPL | DTG m |
|---|---|---|---|---|
| 2azQ1b91cZZ | 24 | 0.750 | 0.335 | 3.00 |
| 8194nk5LbLH | 6 | 1.000 | 0.425 | 0.03 |
| QUCTc6BB5sX | 21 | 0.667 | 0.251 | 4.67 |
| TbHJrupSAjP | 20 | 0.800 | 0.261 | 1.42 |
| Z6MFQCViBuw | 29 | 0.586 | 0.223 | 4.21 |
| oLBMNvg9in8 | 21 | 0.905 | 0.408 | 0.09 |
| pLe4wQe7qrG | 27 | 1.000 | 0.445 | 0.04 |
| zsNo4HB9uLZ | 32 | 0.906 | 0.437 | 0.25 |

Written by `mp3d/results_summary.py` from the run root's `episodes.jsonl` rows; the rows, step logs, LLM answer store and console logs stay with the run root.
