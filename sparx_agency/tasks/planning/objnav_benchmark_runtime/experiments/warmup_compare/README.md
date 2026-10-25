# Warm-up comparison on HM3D v1

Same 120 episodes (3 scenes x 40, seeded, goal view point on the start floor) under each configuration in `configs/`.
The three packages under comparison, each a warm-up (`WarmUp(k)`, every iteration, gated by its flag) and a room visit (`ExploreLocal(v, m, k)`):

| package | config | start warm-up | room visit |
|---|---|---|---|
| 1 | `nearest_frontier` | full rotation | nearest frontier, no circle |
| 2 | `pattern_rooms` | gain-ranked pattern | vantage point, pattern look-around |
| 3 | `falcon_full` | FALCON on a 0.3 m copy of the map (up to 100 actions), no rotation | FALCON viewpoints on the fine map, inside the room |

Ablations and the baseline:

| config | start warm-up | room visit |
|---|---|---|
| `baseline` | full rotation | `explore` (frontier walk by view gain) |
| `scan360` | full rotation | vantage point, full rotation |
| `pattern_start` | gain-ranked pattern | `explore` |
| `prescan` | full rotation, then up to 100 actions of FALCON on the 0.3 m copy | `explore` |
| `falcon_warmup` | FALCON on the 0.3 m copy, no rotation | `explore` |
| `falcon_rooms` | full rotation | FALCON viewpoints on the fine map, inside the room |

```bash
run_compare.sh                                   # all configs; OUT=runs/x to choose or resume a root
run_compare.sh falcon_full falcon_rooms          # only these, added to an existing OUT
python summarize.py runs/warmup_compare_<date>   # SR, SPL, steps and paired flips against baseline
```

`select_episodes.py` regenerates `episodes_hm3d_v1_start_floor.txt`.

FALCON warm-up and prescan count their actions in the 500-step budget. FALCON plans on a block-reduced copy of the fine map (a block is occupied if any cell is, and unknown only when wholly unseen), the fine map and scene graph keep every frame, and the room LLM first reasons once the warm-up has ended. In a room, FALCON plans on the fine map confined to the room's mask plus a margin and never relocates to frontiers elsewhere.
