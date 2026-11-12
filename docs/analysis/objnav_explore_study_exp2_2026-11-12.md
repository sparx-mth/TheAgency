# Experiment 2 -- local exploration: the frontier walk against the viewpoint explore visit (2026-11-12)

**Setting.** The same 39 single-storey MP3D val episodes, ground-truth perception, `qwen2.5:14b-instruct` in both
roles, temperature 0, seed 17 (`mp3d/sets/warmup_study_v1.json`). The `frontier` arm is the running campaign's own
rows (commit `a77c946c`, the method as it stands: ten-turn spin, then the frontier walk); the viewpoint arms ran from
frozen snapshots of this branch on the CPU renderer with the campaign's answer store replayed. Both arms keep the spin
warm-up, so the runs are identical until the first explore visit.

**What was built.** `LoopSettings.explore_goals == "viewpoint"` (`rooms/room_search_loop.py`, `_viewpoint_step`):
the room visit runs the greedy look planner (`exploration/greedy_look.py`, the same machine as the vantage warm-up)
scoped to the room in force. At every action it weighs a sweep of turns where it stands against a walk to a reachable
point of the room's floor (a half-metre lattice over the mask, the frontier cells among the candidates), each valued
by the expected useful area the camera would reveal from there per action it costs -- MapEx's viewpoint score with
this camera's sensor model (79-degree cone, rays to the 5 m detection range, the survival prior exp(-d / 2 m) on
crossed unknown; `viewpoint_transparent` drops the prior for MapEx's transparent unknown) over forward steps and
turns. The frontier population that keeps a room open is unchanged; when the planner finds nothing worth
`min_rate_m2_per_action` (0.25 m2) while significant frontier is still listed, that frontier is retired so the room
reads finished, and a room resolved from its edge gets one short step inside for the ledger's `explored` verdict.

**Arms.** `viewpoint` (survival prior, rate floor 0.25 m2/action), `viewpoint_norate` (prior, no floor: a frontier is
never given up while a look promises anything), `viewpoint_transparent` (no prior, floor 0.25 -- MapEx's raycast).

## Result

Episodes in every arm: 39

| arm | n | SR | SPL | SoftSPL | DTG (m) | steps | dSR vs frontier [95 %] | McNemar p | dSPL [95 %] | flips +/- |
|---|---|---|---|---|---|---|---|---|---|---|
| frontier | 39 | 82.1 % | 35.5 % | 38.2 % | 1.89 | 212 | -- | -- | -- | -- |
| viewpoint | 39 | 76.9 % | 34.4 % | 37.5 % | 1.68 | 218 | -5.1 pts [-15.4, +5.1] | 0.625 | -0.011 [-0.053, +0.029] | 1 / 3 |
| viewpoint_norate | 39 | 79.5 % | 34.8 % | 37.2 % | 1.93 | 218 | -2.6 pts [-12.8, +5.1] | 1.000 | -0.007 [-0.050, +0.032] | 1 / 2 |
| viewpoint_transparent | 39 | 82.1 % | 35.4 % | 37.4 % | 1.93 | 215 | +0.0 pts [-10.3, +10.3] | 1.000 | -0.001 [-0.048, +0.042] | 2 / 2 |

The warm-up itself:

| arm | warm-up actions | moves | walked (m) | area @ warm-up end (m2) | area @10 | area @20 | area @30 | area @50 | first sight: median action | seen within 30 | never seen | first lock: median | never locked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frontier | 8.5 | 0.00 | 0.00 | 27.7 | 24.4 | 28.7 | 31.6 | 37.2 | 42 | 38.5 % | 7.7 % | 43 | 7.7 % |
| viewpoint | 8.5 | 0.00 | 0.00 | 27.7 | 24.5 | 28.9 | 31.7 | 36.9 | 42 | 38.5 % | 7.7 % | 43 | 7.7 % |
| viewpoint_norate | 8.5 | 0.00 | 0.00 | 27.7 | 24.5 | 28.9 | 31.7 | 36.9 | 44 | 38.5 % | 2.6 % | 46 | 2.6 % |
| viewpoint_transparent | 8.5 | 0.00 | 0.00 | 27.7 | 24.5 | 28.9 | 31.7 | 36.8 | 44 | 38.5 % | 2.6 % | 46 | 2.6 % |

The local exploration (the explore visits):

| arm | explore actions / episode | area gained per explore action (m2) | explore visits / episode | frontier retired as not worth it / episode | steps inside / episode |
|---|---|---|---|---|---|
| frontier | 29.9 | 0.237 | 1.59 | 0.00 | 0.00 |
| viewpoint | 30.2 | 0.208 | 1.31 | 0.92 | 0.15 |
| viewpoint_norate | 39.0 | 0.210 | 1.51 | 0.54 | 0.31 |
| viewpoint_transparent | 37.3 | 0.176 | 1.36 | 0.69 | 0.21 |

Per-scene SR:

| scene | n | frontier | viewpoint | viewpoint_norate | viewpoint_transparent |
|---|---|---|---|---|---|
| 2azQ1b91cZZ | 5 | 5 (100.0 %) | 4 (80.0 %) | 4 (80.0 %) | 4 (80.0 %) |
| 8194nk5LbLH | 1 | 1 (100.0 %) | 1 (100.0 %) | 1 (100.0 %) | 1 (100.0 %) |
| QUCTc6BB5sX | 4 | 3 (75.0 %) | 3 (75.0 %) | 3 (75.0 %) | 3 (75.0 %) |
| TbHJrupSAjP | 3 | 1 (33.3 %) | 1 (33.3 %) | 1 (33.3 %) | 2 (66.7 %) |
| Z6MFQCViBuw | 8 | 4 (50.0 %) | 4 (50.0 %) | 4 (50.0 %) | 4 (50.0 %) |
| oLBMNvg9in8 | 5 | 5 (100.0 %) | 4 (80.0 %) | 5 (100.0 %) | 5 (100.0 %) |
| pLe4wQe7qrG | 6 | 6 (100.0 %) | 6 (100.0 %) | 6 (100.0 %) | 6 (100.0 %) |
| zsNo4HB9uLZ | 7 | 7 (100.0 %) | 7 (100.0 %) | 7 (100.0 %) | 7 (100.0 %) |

viewpoint against frontier: 1 to success (Z6MFQCViBuw/000120); 3 to failure (2azQ1b91cZZ/000142, Z6MFQCViBuw/000003, oLBMNvg9in8/000066); SPL better in 7 episodes, worse in 8 (by more than 0.01)

Why the changed episodes changed (viewpoint against frontier; a flip of success, or SPL moved by 0.05):
- 2azQ1b91cZZ/000061 (table): success -> success; SPL 0.580 -> 0.664; steps 240 -> 158; DTG 0.04 -> 0.03 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 195 -> 96, first takeover at 196 -> 97; LLM answers asked 28 -> 2, of which replayed 0 -> 10; the two runs part at action 81 (frontier was in frontier, viewpoint in viewpoint); frontier's closing: last release bearing approach given up: not moving: three steps toward it went nowhere; viewpoint's closing: last release bearing approach given up: lost: 3 frames without the class in view
- 2azQ1b91cZZ/000142 (counter): success -> failure; SPL 0.199 -> 0.000; steps 498 -> 500; DTG 0.03 -> 15.77 m; ended stop -> step_limit; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 457 -> --, first takeover at 458 -> --; LLM answers asked 72 -> 63, of which replayed 0 -> 5; the two runs part at action 47 (frontier was in frontier, viewpoint in viewpoint)
- QUCTc6BB5sX/000061 (towel): success -> success; SPL 0.462 -> 0.338; steps 161 -> 226; DTG 0.04 -> 0.05 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 138 -> 205, first takeover at 139 -> 205; LLM answers asked 21 -> 24, of which replayed 0 -> 8; the two runs part at action 64 (frontier was in transit/2, viewpoint in transit/5)
- QUCTc6BB5sX/000141 (cabinet): success -> success; SPL 0.665 -> 0.812; steps 77 -> 63; DTG 0.04 -> 0.04 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 59 -> 40, first takeover at 60 -> 41; LLM answers asked 6 -> 1, of which replayed 0 -> 2; the two runs part at action 36 (frontier was in glance, viewpoint in viewpoint)
- TbHJrupSAjP/000040 (stool): success -> success; SPL 0.415 -> 0.481; steps 164 -> 142; DTG 0.04 -> 0.03 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 138 -> 112, first takeover at 144 -> 113; LLM answers asked 9 -> 4, of which replayed 0 -> 4; the two runs part at action 24 (frontier was in frontier, viewpoint in viewpoint)
- Z6MFQCViBuw/000003 (chest_of_drawers): success -> failure; SPL 0.163 -> 0.000; steps 283 -> 500; DTG 0.03 -> 16.13 m; ended stop -> step_limit; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 6 -> 6, first takeover at 6 -> 6; LLM answers asked 22 -> 19, of which replayed 0 -> 15; the two runs part at action 145 (frontier was in glance, viewpoint in viewpoint); frontier's closing: last release bearing approach given up: not moving: three steps toward it went nowhere; viewpoint's closing: last release bearing approach given up: not moving: three steps toward it went nowhere
- Z6MFQCViBuw/000120 (bed): failure -> success; SPL 0.000 -> 0.398; steps 500 -> 319; DTG 39.55 -> 0.04 m; ended step_limit -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action -- -> 258, first takeover at -- -> 259; LLM answers asked 52 -> 18, of which replayed 0 -> 6; the two runs part at action 33 (frontier was in frontier, viewpoint in viewpoint); viewpoint's closing: last release bearing approach given up: walked 3.0 m toward it without the depth
- Z6MFQCViBuw/000123 (picture): success -> success; SPL 0.813 -> 0.239; steps 100 -> 378; DTG 0.04 -> 0.08 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 37 -> 45, first takeover at 38 -> 46; LLM answers asked 7 -> 21, of which replayed 0 -> 2; the two runs part at action 12 (frontier was in frontier, viewpoint in viewpoint); frontier's closing: last release bearing approach given up: near and unmeasurable: the box fills 42% of the frame without depth; viewpoint's closing: last release bearing approach given up: near and unmeasurable: the box fills 44% of the frame without depth
- oLBMNvg9in8/000066 (chair): success -> failure; SPL 0.154 -> 0.000; steps 412 -> 500; DTG 0.04 -> 4.06 m; ended stop -> step_limit; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 343 -> --, first takeover at 345 -> --; LLM answers asked 33 -> 42, of which replayed 0 -> 9; the two runs part at action 106 (frontier was in frontier, viewpoint in viewpoint); frontier's closing: last release bearing approach given up: not moving: three steps toward it went nowhere
- oLBMNvg9in8/000086 (picture): success -> success; SPL 0.305 -> 0.395; steps 93 -> 77; DTG 0.04 -> 0.05 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 57 -> 44, first takeover at 58 -> 45; LLM answers asked 8 -> 4, of which replayed 0 -> 4; the two runs part at action 18 (frontier was in transit/3, viewpoint in viewpoint)
- zsNo4HB9uLZ/000022 (bed): success -> success; SPL 0.150 -> 0.242; steps 373 -> 231; DTG 0.05 -> 0.00 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 332 -> 203, first takeover at 333 -> 203; LLM answers asked 33 -> 10, of which replayed 0 -> 5; the two runs part at action 45 (frontier was in frontier, viewpoint in transit/3)
- zsNo4HB9uLZ/000042 (picture): success -> success; SPL 0.293 -> 0.142; steps 84 -> 141; DTG 0.03 -> 0.02 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 74 -> 88, first takeover at 74 -> 88; LLM answers asked 5 -> 1, of which replayed 0 -> 5; the two runs part at action 68 (frontier was in glance, viewpoint in viewpoint)
- zsNo4HB9uLZ/000101 (cushion): success -> success; SPL 0.543 -> 0.458; steps 95 -> 125; DTG 0.03 -> 0.03 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 33 -> 33, first takeover at 33 -> 33; LLM answers asked 10 -> 7, of which replayed 0 -> 5; the two runs part at action 55 (frontier was in peek, viewpoint in viewpoint); frontier's closing: last release unverified; viewpoint's closing: last release bearing approach given up: not moving: three steps toward it went nowhere
- zsNo4HB9uLZ/000142 (chest_of_drawers): success -> success; SPL 0.426 -> 0.589; steps 214 -> 181; DTG 0.04 -> 0.01 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 184 -> 142, first takeover at 184 -> 142; LLM answers asked 17 -> 9, of which replayed 0 -> 3; the two runs part at action 96 (frontier was in frontier, viewpoint in glance)

viewpoint_norate against frontier: 1 to success (Z6MFQCViBuw/000120); 2 to failure (2azQ1b91cZZ/000142, Z6MFQCViBuw/000003); SPL better in 8 episodes, worse in 6 (by more than 0.01)

Why the changed episodes changed (viewpoint_norate against frontier; a flip of success, or SPL moved by 0.05):
- 2azQ1b91cZZ/000061 (table): success -> success; SPL 0.580 -> 0.664; steps 240 -> 158; DTG 0.04 -> 0.03 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 195 -> 96, first takeover at 196 -> 97; LLM answers asked 28 -> 0, of which replayed 0 -> 12; the two runs part at action 81 (frontier was in frontier, viewpoint_norate in viewpoint); frontier's closing: last release bearing approach given up: not moving: three steps toward it went nowhere; viewpoint_norate's closing: last release bearing approach given up: lost: 3 frames without the class in view
- 2azQ1b91cZZ/000142 (counter): success -> failure; SPL 0.199 -> 0.000; steps 498 -> 500; DTG 0.03 -> 9.57 m; ended stop -> step_limit; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 457 -> 481, first takeover at 458 -> 482; LLM answers asked 72 -> 41, of which replayed 0 -> 10; the two runs part at action 47 (frontier was in frontier, viewpoint_norate in viewpoint); viewpoint_norate's closing: last release bearing approach given up: not moving: three steps toward it went nowhere
- QUCTc6BB5sX/000061 (towel): success -> success; SPL 0.462 -> 0.338; steps 161 -> 226; DTG 0.04 -> 0.05 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 138 -> 205, first takeover at 139 -> 205; LLM answers asked 21 -> 0, of which replayed 0 -> 32; the two runs part at action 64 (frontier was in transit/2, viewpoint_norate in transit/5)
- QUCTc6BB5sX/000141 (cabinet): success -> success; SPL 0.665 -> 0.812; steps 77 -> 63; DTG 0.04 -> 0.04 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 59 -> 40, first takeover at 60 -> 41; LLM answers asked 6 -> 0, of which replayed 0 -> 3; the two runs part at action 36 (frontier was in glance, viewpoint_norate in viewpoint)
- TbHJrupSAjP/000040 (stool): success -> success; SPL 0.415 -> 0.481; steps 164 -> 142; DTG 0.04 -> 0.03 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 138 -> 112, first takeover at 144 -> 113; LLM answers asked 9 -> 0, of which replayed 0 -> 8; the two runs part at action 24 (frontier was in frontier, viewpoint_norate in viewpoint)
- Z6MFQCViBuw/000003 (chest_of_drawers): success -> failure; SPL 0.163 -> 0.000; steps 283 -> 500; DTG 0.03 -> 14.27 m; ended stop -> step_limit; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 6 -> 6, first takeover at 6 -> 6; LLM answers asked 22 -> 24, of which replayed 0 -> 30; the two runs part at action 145 (frontier was in glance, viewpoint_norate in viewpoint); frontier's closing: last release bearing approach given up: not moving: three steps toward it went nowhere; viewpoint_norate's closing: last release bearing approach given up: lost: 3 frames without the class in view
- Z6MFQCViBuw/000120 (bed): failure -> success; SPL 0.000 -> 0.378; steps 500 -> 334; DTG 39.55 -> 0.03 m; ended step_limit -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action -- -> 287, first takeover at -- -> 288; LLM answers asked 52 -> 10, of which replayed 0 -> 16; the two runs part at action 33 (frontier was in frontier, viewpoint_norate in viewpoint); viewpoint_norate's closing: last release bearing approach given up: near and unmeasurable: the box fills 30% of the frame without depth
- Z6MFQCViBuw/000123 (picture): success -> success; SPL 0.813 -> 0.218; steps 100 -> 375; DTG 0.04 -> 0.04 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 37 -> 45, first takeover at 38 -> 46; LLM answers asked 7 -> 15, of which replayed 0 -> 10; the two runs part at action 12 (frontier was in frontier, viewpoint_norate in viewpoint); frontier's closing: last release bearing approach given up: near and unmeasurable: the box fills 42% of the frame without depth; viewpoint_norate's closing: last release bearing approach given up: near and unmeasurable: the box fills 37% of the frame without depth
- oLBMNvg9in8/000023 (table): success -> success; SPL 0.309 -> 0.406; steps 232 -> 160; DTG 0.06 -> 0.04 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 207 -> 143, first takeover at 207 -> 143; LLM answers asked 13 -> 3, of which replayed 0 -> 3; the two runs part at action 65 (frontier was in frontier, viewpoint_norate in viewpoint)
- oLBMNvg9in8/000086 (picture): success -> success; SPL 0.305 -> 0.395; steps 93 -> 77; DTG 0.04 -> 0.05 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 57 -> 44, first takeover at 58 -> 45; LLM answers asked 8 -> 0, of which replayed 0 -> 8; the two runs part at action 18 (frontier was in transit/3, viewpoint_norate in viewpoint)
- zsNo4HB9uLZ/000042 (picture): success -> success; SPL 0.293 -> 0.142; steps 84 -> 141; DTG 0.03 -> 0.02 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 74 -> 88, first takeover at 74 -> 88; LLM answers asked 5 -> 0, of which replayed 0 -> 6; the two runs part at action 68 (frontier was in glance, viewpoint_norate in viewpoint)
- zsNo4HB9uLZ/000101 (cushion): success -> success; SPL 0.543 -> 0.458; steps 95 -> 125; DTG 0.03 -> 0.03 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 33 -> 33, first takeover at 33 -> 33; LLM answers asked 10 -> 0, of which replayed 0 -> 12; the two runs part at action 55 (frontier was in peek, viewpoint_norate in viewpoint); frontier's closing: last release unverified; viewpoint_norate's closing: last release bearing approach given up: not moving: three steps toward it went nowhere
- zsNo4HB9uLZ/000142 (chest_of_drawers): success -> success; SPL 0.426 -> 0.589; steps 214 -> 181; DTG 0.04 -> 0.01 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 184 -> 142, first takeover at 184 -> 142; LLM answers asked 17 -> 0, of which replayed 0 -> 12; the two runs part at action 96 (frontier was in frontier, viewpoint_norate in glance)

viewpoint_transparent against frontier: 2 to success (TbHJrupSAjP/000162, Z6MFQCViBuw/000120); 2 to failure (2azQ1b91cZZ/000142, Z6MFQCViBuw/000003); SPL better in 8 episodes, worse in 7 (by more than 0.01)

Why the changed episodes changed (viewpoint_transparent against frontier; a flip of success, or SPL moved by 0.05):
- 2azQ1b91cZZ/000061 (table): success -> success; SPL 0.580 -> 0.664; steps 240 -> 158; DTG 0.04 -> 0.03 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 195 -> 96, first takeover at 196 -> 97; LLM answers asked 28 -> 2, of which replayed 0 -> 10; the two runs part at action 81 (frontier was in frontier, viewpoint_transparent in viewpoint); frontier's closing: last release bearing approach given up: not moving: three steps toward it went nowhere; viewpoint_transparent's closing: last release bearing approach given up: lost: 3 frames without the class in view
- 2azQ1b91cZZ/000142 (counter): success -> failure; SPL 0.199 -> 0.000; steps 498 -> 500; DTG 0.03 -> 17.04 m; ended stop -> step_limit; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 457 -> --, first takeover at 458 -> --; LLM answers asked 72 -> 125, of which replayed 0 -> 6; the two runs part at action 47 (frontier was in frontier, viewpoint_transparent in viewpoint)
- QUCTc6BB5sX/000061 (towel): success -> success; SPL 0.462 -> 0.338; steps 161 -> 226; DTG 0.04 -> 0.05 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 138 -> 205, first takeover at 139 -> 205; LLM answers asked 21 -> 24, of which replayed 0 -> 8; the two runs part at action 64 (frontier was in transit/2, viewpoint_transparent in transit/5)
- QUCTc6BB5sX/000141 (cabinet): success -> success; SPL 0.665 -> 0.812; steps 77 -> 63; DTG 0.04 -> 0.04 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 59 -> 40, first takeover at 60 -> 41; LLM answers asked 6 -> 1, of which replayed 0 -> 2; the two runs part at action 36 (frontier was in glance, viewpoint_transparent in viewpoint)
- TbHJrupSAjP/000040 (stool): success -> success; SPL 0.415 -> 0.476; steps 164 -> 153; DTG 0.04 -> 0.04 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 138 -> 120, first takeover at 144 -> 120; LLM answers asked 9 -> 4, of which replayed 0 -> 4; the two runs part at action 24 (frontier was in frontier, viewpoint_transparent in viewpoint)
- TbHJrupSAjP/000162 (chest_of_drawers): failure -> success; SPL 0.000 -> 0.325; steps 500 -> 348; DTG 5.45 -> 0.05 m; ended step_limit -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action -- -> 328, first takeover at -- -> 328; LLM answers asked 59 -> 25, of which replayed 0 -> 25; the two runs part at action 126 (frontier was in transit/2, viewpoint_transparent in viewpoint)
- Z6MFQCViBuw/000003 (chest_of_drawers): success -> failure; SPL 0.163 -> 0.000; steps 283 -> 500; DTG 0.03 -> 12.60 m; ended stop -> step_limit; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 6 -> 6, first takeover at 6 -> 6; LLM answers asked 22 -> 22, of which replayed 0 -> 15; the two runs part at action 145 (frontier was in glance, viewpoint_transparent in viewpoint); frontier's closing: last release bearing approach given up: not moving: three steps toward it went nowhere; viewpoint_transparent's closing: last release bearing approach given up: not moving: three steps toward it went nowhere
- Z6MFQCViBuw/000120 (bed): failure -> success; SPL 0.000 -> 0.398; steps 500 -> 319; DTG 39.55 -> 0.04 m; ended step_limit -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action -- -> 258, first takeover at -- -> 259; LLM answers asked 52 -> 18, of which replayed 0 -> 6; the two runs part at action 33 (frontier was in frontier, viewpoint_transparent in viewpoint); viewpoint_transparent's closing: last release bearing approach given up: walked 3.0 m toward it without the depth
- Z6MFQCViBuw/000123 (picture): success -> success; SPL 0.813 -> 0.182; steps 100 -> 462; DTG 0.04 -> 0.04 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 37 -> 45, first takeover at 38 -> 46; LLM answers asked 7 -> 28, of which replayed 0 -> 2; the two runs part at action 12 (frontier was in frontier, viewpoint_transparent in viewpoint); frontier's closing: last release bearing approach given up: near and unmeasurable: the box fills 42% of the frame without depth; viewpoint_transparent's closing: last release bearing approach given up: near and unmeasurable: the box fills 34% of the frame without depth
- oLBMNvg9in8/000086 (picture): success -> success; SPL 0.305 -> 0.395; steps 93 -> 77; DTG 0.04 -> 0.05 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 57 -> 44, first takeover at 58 -> 45; LLM answers asked 8 -> 4, of which replayed 0 -> 4; the two runs part at action 18 (frontier was in transit/3, viewpoint_transparent in viewpoint)
- zsNo4HB9uLZ/000022 (bed): success -> success; SPL 0.150 -> 0.292; steps 373 -> 196; DTG 0.05 -> 0.03 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 332 -> 61, first takeover at 333 -> 62; LLM answers asked 33 -> 16, of which replayed 0 -> 7; the two runs part at action 45 (frontier was in frontier, viewpoint_transparent in viewpoint); viewpoint_transparent's closing: last release bearing approach given up: lost: 3 frames without the class in view
- zsNo4HB9uLZ/000042 (picture): success -> success; SPL 0.293 -> 0.170; steps 84 -> 133; DTG 0.03 -> 0.05 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 74 -> 92, first takeover at 74 -> 92; LLM answers asked 5 -> 1, of which replayed 0 -> 5; the two runs part at action 68 (frontier was in glance, viewpoint_transparent in viewpoint)
- zsNo4HB9uLZ/000101 (cushion): success -> success; SPL 0.543 -> 0.458; steps 95 -> 125; DTG 0.03 -> 0.03 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 33 -> 33, first takeover at 33 -> 33; LLM answers asked 10 -> 7, of which replayed 0 -> 5; the two runs part at action 55 (frontier was in peek, viewpoint_transparent in viewpoint); frontier's closing: last release unverified; viewpoint_transparent's closing: last release bearing approach given up: not moving: three steps toward it went nowhere
- zsNo4HB9uLZ/000142 (chest_of_drawers): success -> success; SPL 0.426 -> 0.484; steps 214 -> 290; DTG 0.04 -> 0.02 m; ended stop -> stop; warm-up actions 10 -> 10, walks 0 -> 0, walked 0.00 -> 0.00 m; target first in frame at action 184 -> 147, first takeover at 184 -> 148; LLM answers asked 17 -> 10, of which replayed 0 -> 3; the two runs part at action 96 (frontier was in frontier, viewpoint_transparent in glance); viewpoint_transparent's closing: last release unverified again


## Reading

1. **No arm of the viewpoint visit improves on the frontier walk; with the rate floor it is a little worse.**
   `viewpoint` SR 76.9 against 82.1 (-5.1 points, CI [-15, +5], McNemar p 0.63, flips 1 to success / 3 to failure),
   SPL 34.4 against 35.5; `viewpoint_norate` SR 79.5 (-2.6, flips 1 / 2), SPL 34.8; `viewpoint_transparent` SR 82.1
   = 82.1, SPL 35.4, flips 2 / 2. All three are inside the noise floor measured in experiment 1 (two arms with
   identical behaviour differed by 3 flips and 1.8 SPL points), so none is a demonstrated loss -- and none is a gain.
2. **Per action, the frontier walk reveals more.** Area gained per explore action: frontier 0.237 m2, viewpoint 0.208,
   norate 0.210, transparent 0.176. The planner's first decision was a walk in every visit, most often a walk
   followed by a full circle -- 20-32 actions for 9-17 m2 of *expected* area, rates of 0.3-0.85 m2/action -- while
   the frontier walk goes straight to the boundary, where the arrival frame shows what lies behind it, and never
   spends ten turns on a circle. The expected-area model did not predict realised gain better than "walk to the
   boundary that reveals the most per metre", which is already a greedy information-gain policy.
3. **The rate floor gives rooms up that the frontier walk finishes -- and removing it does not help.** With the
   floor, 14 of 45 `viewpoint` visits ended "nothing worth an action" with significant frontier still listed and
   retired it (36 clusters over the 39 episodes): a frontier 5 m away with 5 m2 of expected area is 25 actions for a
   rate of 0.2 and is given up, where the frontier walk's own floor is 1 m2 of gain at any distance. Without the
   floor (`norate`) the planner retired 21 clusters (only where no look promised anything), spent 39 explore actions
   an episode against 30, took 42 single-frame looks beside 24 circles, and ended at SR 79.5: the giving-up was not
   what the frontier walk had over it. The same two episodes fail in both viewpoint arms: `2azQ1b91cZZ/000142`
   (counter: the frontier arm walked six frontiers of rooms 2 and 1 between actions 46 and 89 and found it at 457;
   `viewpoint` retired room 2's last two frontiers at 66 and never saw it, `norate` saw it at 481 and ran out) and
   `Z6MFQCViBuw/000003` (chest of drawers: both arms locked it at action 6 and lost it to "not moving"; the frontier
   arm's 75 frontier actions found it again at 283, the viewpoint arms' 84-100 viewpoint actions and 90 pursuit
   actions did not). `oLBMNvg9in8/000066` (chair) fails only with the floor: the visit of room 10 spent 28 actions on
   one circle, retired the room's last frontier at 139, and the search never came back to the chair the frontier arm
   reached at 343. `Z6MFQCViBuw/000120` (bed) is the one gain, in all three viewpoint arms: the frontier arm walked
   209 frontier actions through 500 and never saw the bed, the viewpoint arms stood where they saw it (258-287) and
   STOPped at 319-334.
4. **Dropping the survival prior spends more and gains nothing.** The transparent arm spent 857 viewpoint actions
   over the 39 episodes against 616 (every ray through the unknown counts to full range, so more walks look
   worthwhile), with the lowest realised gain per action (0.176) and the same SR as the frontier walk.

**Conclusion.** Inside a room, standing where the camera is expected to see the most is not better than walking to
the boundary that reveals the most per metre: the frontier walk already is the greedy information-gain policy, its
goals are where new area is certain, and the viewpoint planner's circles cost ten turns each for an expected gain
the map does not repay -- with or without the rate floor, with or without the prior on the unknown. The viewpoint
visit stays off by default (`explore_goals: frontier`); the three arms remain in the laptop study
(`mp3d/explore_compare.sh`) for the 3B repeat.

Detailed per-episode explanations (every flip and every |dSPL| >= 0.05) follow in the report's `--explain` section
above; the JSON beside this file holds every number.
