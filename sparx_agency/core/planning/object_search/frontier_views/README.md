# frontier_views — which past frames show a frontier node

The second piece of the VLM pipeline analysed in
[`docs/papers/vlm-search-scoring-20261119`](../../../../../docs/papers/vlm-search-scoring-20261119/README.md),
between the [frame log](../frame_log/README.md) -- every RGB frame of the
episode under its capture pose -- and the model that will be asked how likely
the target lies behind a frontier. It answers, from geometry alone and before
any model is asked: **for each frontier node, which one to three logged frames
show it best, and where in each picture it lies.**

`RPTSearchPolicy` runs this on live observations. The offline replay
(`tasks/planning/objnav_benchmark_runtime/frontier_view_replay.py`) exercises
the same geometry without inference; its historical numbers below are not
live VLM accuracy or speed measurements.

```text
 FrontierInventory.clusters ─► frontier_nodes()  ─► [FrontierNode]      the places to ask about: cells, goal point, sticky id
 EpisodeFrameLog.records ────┐
 the map of this decision ───┼► select_views()   ─► {id: NodeViews}     1-3 frames a node, each with its window; or why none
 [FrontierNode] ─────────────┘        │
                                 SightScene.sight(pose)                  per frame and cell: out of range / out of the picture /
                                                                         occluded / seen -- and seen THROUGH or not
 ScoreLedger.observe / due / mark_scored / mark_visited                  score once, never again; still there after a visit: blocked
```

| module | holds |
|---|---|
| `nodes.py` | `FrontierNode`, `frontier_nodes()`, `NodeSettings`: the inventory's clusters merged into nodes with ids that stay |
| `sight.py` | `SightScene`, `FrameSight`: what one frame shows of a set of map cells -- which it sees, how well, whether it sees past them, and the pixels they land on |
| `selection.py` | `select_views()`, `NodeViews`, `FrameView`: the greedy choice of one to three frames per node, the window in each, the reason when there is none |
| `score_ledger.py` | `ScoreLedger`: the score-once rule -- which node is due, which is done, which is blocked |
| `settings.py` | `ViewSettings`: every range, share and threshold of the two modules above, in one frozen record |
| `scoring.py`, `search.py` | Mark selected frames, ask the VLM, retain independent probabilities and plan to the head of the RPT* order with A* |

## The unit: a frontier node

The search this prepares for hands RPT\* **frontier clusters** valued from
pictures, not rooms valued from the objects in them. The unit a picture is
asked about needs three things, and no existing type had all three:

| needed | `FrontierCluster` | `FrontierGoal` | `Opening` |
|---|---|---|---|
| the cells of the boundary (which frame looked at it is a question about cells) | yes | no | no |
| an id that stays (a node is asked about once and its answer kept) | no | no | yes |
| a point a planner accepts (RPT\* needs `id -> (x, y), probability` and nothing else) | yes | yes | yes |

A `FrontierNode` is built from the clusters of the frontier inventory
(`FrontierInventory.clusters`, kept since this change: the cells of every
goal's cluster, from the one extraction the goals came from) the way the
openings are built from exits, and by the same code:

1. the clusters, longest first, are merged greedily within `merge_m` (1.5 m)
   of a seed -- `rooms/opening_nodes.merge_near_seed`, extracted from
   `detect_openings`, which now calls it. Two boundaries a doorway apart open
   onto one unseen space, and an order that treats nodes as independent would
   count that space twice;
2. a group under `min_cells` (5: half a metre, the openings' own floor) is
   dropped;
3. the seed's goal point asks an `OpeningRegistry` for the id: a place within
   `match_m` (1 m) of a known one keeps that one's.

**Why not the `Opening` itself.** `detect_openings` reads the state of the loop
this pipeline replaces -- the multi-class landmarks whose shadows it discards,
room numbers, doors, peeks, the sweep's memory -- and runs only while that loop
does: of the six episodes below, three never listed an opening while their maps
held six to thirteen reachable clusters an action. What the opening got right
is kept by calling the same merge and the same registry.

On those episodes a node's id is stable enough to hang an answer on: 112 ids
over 719 actions for 5.4 nodes an action, from 9.9 clusters an action.

## What counts as a view

For one frame and one cell of a node, in order (`sight.py`):

1. **In range** -- within `max_range_m` (8 m) of the camera.
2. **In the picture** -- the band of heights that matters at the cell (floor to
   1.5 m) is clipped against the image exactly; the share left inside is the
   view's vertical worth, so a camera tilted down counts for less and one
   tilted further does not count, with no rule of its own for the pitch.
3. **Seen** -- the sight line over the map of *this decision* (a wall mapped
   since the frame was taken still stood when it was taken) crosses no occupied
   cell and no deep unknown.
4. **Seen through** -- the line, continued past the cell, runs straight into
   unknown the search has not settled: the picture shows what lies beyond the
   boundary there.

A view's **quality** is `min(1, good_range_m / distance)` x the share of the
band in the picture x (1 when seen through, `at_weight` when only seen); a node
is worth, to a set of frames, the average over its cells of the best quality
any of them gives the cell.

### Strategy B as proposed, and what the replay changed

Strategy B was proposed as *count the image columns whose ray leaves the map
through the node's cells*: only a frame that looks past the boundary is a view
of it. That is `at_weight = 0`, and the replay keeps it as the `through_only`
variant. It scores **32 %** of the nodes. The reason is structural, not a
threshold: the map is what the depth camera saw, so its boundary is the edge of
what every frame's depth resolved. A frame with a clear line across a boundary
cell and the far side inside its depth range would have mapped the far side.
A frame sees *through* a boundary only from beyond its depth range -- the
colour image carries farther than the depth -- and most frontiers are nearer.

So a cell seen but not seen through counts too, at `at_weight` (0.5): the
picture of the doorway, its jambs and the part of the room that is not hidden
is all there is, and it is evidence. That scores **81 %** of the nodes.

Two readings of "seen" had to give way for a boundary to be visible from the
frame that made it -- which is, for most nodes, the only frame there is:

* **the fringe**: unknown cells touching known floor do not hide what is
  behind them (59 % of nodes scored before, 63 % after);
* **a corner clipped by a cell** (`corner_slack_cells`, 1): the edge of what a
  camera saw past a door jamb is its own line tangent to the jamb, and on a grid
  a tangent clips the corner cell (63 % to 81 %).

**Two-dimensional on purpose, and what that costs.** The map marks a cell
occupied for anything between 0.15 and 0.93 m above the floor and the camera
stands at 0.88 m: it sees over a bench the sight line stops at. Such a node is
reported `occluded in every frame that faces it`, not silently absent; six
long-lived nodes of the 112 ended that way, and the replay draws each with the
frames that faced it. A per-frame depth profile (the proposal's Strategy C)
would recover them; it needs depth in the frame log and has not been needed
yet.

## Choosing the frames

`select_views()` adds frames greedily while the node's score grows by a fifth
(`min_gain_ratio`), three at most; the first is the best single view. A second
frame is rare -- 9 of 91 scorings -- and a third did not occur: a node is small
and one frame usually shows what there is to show of it.

Between frames worth the same -- the frames of a turn on one spot are worth
exactly the same -- the one that sees past more of the node is taken, then the
one with the node nearest the middle of its picture, then the latest.

Each node is chosen for on its own (one question per node is asked later), so
two nodes may be handed one frame, each with its own **window**: the box the
node's seen cells land in, floor to the top of the band. The median window is
82 pixels of 640 wide. A frame handed over without it is a picture of a room
with nothing saying which doorway is meant.

A node whose best frame shows too little has **no usable view**, and the answer
says which gate every frame failed: no frame of this storey, too far, in no
frame's picture, occluded, or seen but too thin.

## Score once

`ScoreLedger` is the bookkeeping of the rule the valuation runs under, kept
apart from whoever asks:

* a node is **scored once**, the first action it has a usable view -- the
  action it appears, when a frame already shows it, or a later one;
* a node already scored is **never asked about again**, however the frames
  that show it improve;
* a node **still there `settle_steps` (12) actions after the agent first stood
  at it** is not re-scored either: it is marked **blocked**.

It holds no probability and asks no model: `observe(step, ids)`, `due(views)`,
`mark_scored(id, step, frames)`, `mark_visited(id, step)`.

`min_age_steps` (0) holds a node back until it has been there that long. The
default is the rule as decided; the replay is why the knob exists (below).

## Live decisions

On each new observation the policy logs RGB, updates the observed map, and
builds the current floor's reachable inventory. In transit between storeys
there is no new floor map: the frame is kept but no planar nodes are offered.
`FrontierSearch.observe` selects views for pending nodes only. Each due node
gets its own red outline on each selected frame, even if another node uses
the same frame. One JPEG (up to three views side by side, each at most 640 px
on its long edge) and the target's name go in one request. Only a numeric,
finite `probability` in [0, 1] completes a score. A bad reply or outage arms
the fallback service back-off, not a fabricated score. Identity drift or a
broken frame log stops the run.

When search owns motion, `build_instance` gets the current candidate
positions and their stored probabilities without renormalization. It prices
travel from the current pose on the observed clearance map. The policy keeps
the selected ID while eligible and rechecks the A* route for safety each
step. Arrived, absent, blocked and retired goals are not offered. With no
usable route it invokes the existing geometric exploration fallback. No text
room classification or room-oracle call is made. The full episode record
includes the ledger, probabilities and frame names under `frontier_search`.

## What the first replay found (2026-11-22)

Six MP3D v1 val episodes (`pLe4wQe7qrG/000000`, `/000023`, `8194nk5LbLH/000087`,
`/000096`, `zsNo4HB9uLZ/000007`, `/000008`), flown on the CPU with the stand-in
language model and ground-truth perception, recorded with their frame logs:
719 actions, 719 frames, 112 node ids. Dated examples of one small set, to
set thresholds by -- not a result of the method.

| | selected | through only | the action's own frame |
|---|---|---|---|
| nodes scored (= model calls) | 91 of 112 (81 %) | 36 (32 %) | 86 (77 %) |
| scored on the action they appeared | 73 | 17 | 61 |
| node-actions with a score in hand | 81 % | 42 % | 77 % |
| calls per 100 actions | 12.7 | 5.0 | 12.0 |
| calls that look past the boundary | 46 % | 100 % | 45 % |
| median range of the first frame | 4.1 m | 4.8 m | 3.9 m |
| space mapped later behind the node, in the chosen frames | 9 % | 17 % | 9 % |
| ... in any frame logged so far | 12 % | 23 % | 12 % |

What follows from it:

* **The pictures exist for four nodes in five**, at about one question every
  eight actions.
* **The frame that made the node is usually the one chosen.** The median age of
  the first frame is zero actions, and without any memory -- the action's own
  frame alone -- 77 % of the nodes are scored against 81 %. The frame log buys
  the nodes the camera has turned away from by the time they appear, not better
  views of the rest.
* **What is asked about is the boundary, not the space behind it.** Of the
  space behind a node that the episode went on to map, the chosen frames had a
  clear line to 9 % -- and no frame logged so far to more than 12 %; for half the
  nodes, to (nearly) none. That is what a frontier is. A model shown these
  frames judges from what stands at the opening.
* **Waiting buys little.** Scored on the last action it existed instead of the
  first, a node's median score is 0.32 against 0.28 and its first frame is from
  3.4 m instead of 4.2 m; one node in eight would at least double its score.
* **Nearly one call in four is spent on a node gone two actions later** (21 of
  91) -- the edge of the camera's own field of view is a frontier until the next
  turn of a look-around. `min_age_steps = 3` leaves 4 such calls and 8.8 calls
  per 100 actions, and costs score-in-hand: 67 % of node-actions instead of 81 %.
  A delay is also where the frame log earns its keep: three actions on, the
  action's own frame no longer shows the node, and without the log the same
  delay leaves 49 % of node-actions with a score.
* **Ten nodes were still there well after the agent stood at them**: the
  blocked rule has something to do.

## The thresholds are starting values

Every default in `ViewSettings` and `NodeSettings` was set from that one
replay and is meant to be moved by the next. What moved the share of
node-actions with a score in hand (81 %) on the same six episodes, and what did
not: `at_weight` 0.5 to 0 loses 39 points; `corner_slack_cells` 1 to 0 loses 14;
`max_range_m` 8 to 6 and `min_cells` 5 to 10 lose one each.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core/planning/object_search/frontier_views -q
```

Small floor plans drawn in metres (`tests/rig.py`), the benchmark's camera and
frames by pose: a frame's sight of named cells, with the share of the height
band checked against the repository's own projection sampled along it
(`test_sight.py`); the selection (`test_selection.py`); the nodes over the
method's own inventory (`test_nodes.py`); the score-once rule
(`test_score_ledger.py`). The replay of a whole synthetic recording, and of a
real one when one is on the machine, is with the tool:
`tasks/planning/objnav_benchmark_runtime/tests/test_frontier_view_replay.py`.
