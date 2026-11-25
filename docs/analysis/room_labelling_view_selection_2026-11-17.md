# Room labelling in the ObjectNav planner: algorithm findings and proposals

Source of the findings: Daphna's offline room-classification work. Code read against `origin/feat/objnav-habitat-hm3d-nadav` at `a13214e2` (`feat/objnav-benchmark-infra-nadav` points to the same commit). Her reference code is on her local branch `daphna/objnav-verify-diagnosis-rebased`, under `sparx_agency/tasks/planning/objnav_benchmark_runtime/room_dataset/`: `geometry.py` (`visible_cells`, `greedy_views`, `near_share`, `hits_in_room_share`, `wall_hits`, `box_perimeter`) and `view_selection.py` (`describe_candidates`, `select`, `select_walls`).

## How rooms are labelled now

- A text LLM reads the confirmed object names of a room plus its area, shape and doorway count (`core/mapping/topology/revisable_room_labels.py`, `room_classifier.py`). No image reaches any model.
- `relabel_hysteresis` holds a differing verdict until the model repeats it, unless a new object kind appeared. A new kind relabels at once.
- On a re-partition the label of a room that keeps its id survives. The piece that keeps the id carries the whole-room label after a split, and the new piece starts with nothing.
- The scan (`_scan_visit`) walks to `vantage_point` (`rooms/room_vantage.py`), the cell with the most clearance, which is the middle of the room, and turns a full circle.
- In 14 recorded episodes made before the hysteresis existed: median 8 rooms and 9.5 label queries per episode (up to 111). 21 of 126 rooms changed from one real label to another and 10 changed twice or more. 42 of the 45 changes were tagged "evidence changed". A split or merge is tagged the same way, so the two causes cannot be told apart. This rate has not been re-measured with the hysteresis on.

## Measured offline (nothing has run in the planner)

Prompt, one image per call, temperature 0, JSON output: `Which room in a house was this photo taken in? Choose one of: <room names>. Answer with JSON {"room": "<one of the options>"}.` The label is the majority over the first K answers. The list holds room names only. Model: Qwen3-VL 4B.

Matterport3D train, 850 labelled regions, first K views per method:

| input | K=1 | K=3 | K=6 |
|---|---|---|---|
| 6 evenly spread views | 49.1% | 49.3% | 55.5% |
| views chosen by room coverage | 60.8% | 63.8% | 65.8% |

- Text LLM on detected objects: 34-42%. On true annotated objects: 49-53%.
- Coverage minus spread at K=3: +14.5 points, 95% interval +10.8 to +18.0.
- No difference from: adding detected objects to the view choice (-1.8, interval -3.9 to +0.4), a wall-perimeter measure instead of floor (-0.3 at K=3), or removing close-ups alone.
- HM3D dev scenes, 38 rooms labelled by one person who saw the coverage views: old views 55.3%, coverage views 78.9% at K=3. Gain +23.7, interval +5.3 to +39.5.

## The view score

For a candidate pose (floor point, heading):
1. Room cells seen: room floor cells inside the horizontal field of view whose distance is less than the depth at the horizon row in that direction, plus 0.3 m. No depth counts as blocked.
2. Reject the pose when more than 40% of the image is nearer than 1 m or has no depth.
3. Reject the pose when fewer than 70% of its horizon-row depth rays end within 1 m of a room cell. That is a view out through a door.
4. Pick poses greedily: each adds the most room cells no earlier pick saw. Stop below 2% new cells or at K.

Why coverage works: a pose that sees much of the room's floor is looking across the room, so furniture and fixtures are in frame. The failures it removes are views of a wall, curtain or door a short distance from the camera, and views into the next room.

## Proposed changes

1. **Votes by place, not room id.** Store each vision answer with the floor cells its view covered, in world coordinates. A room's tally is the votes whose place lies in its current mask. A merge adds tallies, a split divides them by mask, and a piece with no votes is `unknown` (it may borrow its parent's label for planning only). Where it applies: `revisable_room_labels.py` (`update`, `_update_room`), `rooms/scene_graph.py` (`update`). Expected effect: merges and splits cost no classifier call and no longer hand a whole-room verdict to a fragment.
2. **A margin as well as a repeat.** With a tally, a new object kind adds one vote and the label moves only when another type leads by a margin. The node oracle's room line now shows only `type=<label>` (with `?` when tentative), `mapped=`, `frontier=` and `searched=` (`search_node_oracle.format_node`), so the tally reaches it through the `?`: derive tentative from the margin, so a label held on a thin lead reads tentative. Expected effect: the label changes caused by a single new object kind (a potted plant making a bedroom a living room) stop.
3. **Vision model at two moments only.** Provisional: one or two frames at a doorway peek (`_transit`). Confirmed: three views kept during the room scan (`_scan_visit`, `_begin_rotation`). The rotation already yields a frame every 30 degrees: score each as it arrives, keep the best three, and allow an early end when the room is covered. Not in `_explore_visit` or ordinary transit. Expected effect: at most about three vision calls per room plus the glimpses.
4. **Scan stand point by coverage.** Replace `vantage_point` with the pose that maximises the score above, computed on the occupancy grid with a ray cast. The sight ledger's `observed_provider` may already give the "cells not yet observed" this needs. `room_extent.py` also tracks which walls are seen, and the wall-based measure was equal to floor offline, so either serves.

## Not established

- Nothing here has run inside the planner. Offline, the camera could stand anywhere in the room; the planner pays actions to turn and walk and has to reach each pose.
- Offline depth is the simulator's, not the planner's. The occupancy-map version of the score is untested.
- The HM3D result rests on 38 rooms and one labeller who saw the views the method chose. The old HM3D views included wall and curtain close-ups, so part of that gain is a weak baseline.
- The 36% of Matterport3D rooms still wrong at K=3 are mostly hallway, outdoor, closet and "other", where the dataset's labels are ambiguous.
- Open questions: the code comments refer to a 3B room labeller; whether an image-capable model is available in its place, how many of the 500 actions a scan may use, and whether a doorway stub (`stub_min_cells=4`) carries a label or inherits its parent's.
