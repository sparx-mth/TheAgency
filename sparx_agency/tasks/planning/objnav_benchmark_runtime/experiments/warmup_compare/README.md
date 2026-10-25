# Warm-up comparison on HM3D v1

Same 120 episodes (3 scenes x 40, seeded, goal view point on the start floor) under each configuration in `configs/`.

| config | start warm-up | room visit |
|---|---|---|
| `baseline` | full rotation | `explore` (frontier walk by view gain) |
| `scan360` | full rotation | vantage point, full rotation |
| `nearest_frontier` | full rotation | nearest frontier, no circle |
| `pattern_start` | gain-ranked pattern | `explore` |
| `pattern_rooms` | gain-ranked pattern | vantage point, pattern look-around |
| `prescan` | full rotation, then up to 100 actions of goal-free FALCON on a 0.3 m copy of the map | `explore` |

```bash
run_compare.sh                                   # all configs; OUT=runs/x to choose or resume a root
python summarize.py runs/warmup_compare_<date>   # SR, SPL, steps and paired flips against baseline
```

`select_episodes.py` regenerates `episodes_hm3d_v1_start_floor.txt`.

`prescan` counts its actions in the 500-step budget. FALCON plans on a block-reduced copy of the fine map (a block is occupied if any cell is, else unknown if any is), the fine map and scene graph keep every frame, and the room LLM first reasons once the prescan has ended. `run_compare.sh prescan` adds it to an existing output root.
