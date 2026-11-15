# 045 - MP3D graph and oracle bookkeeping (layer 4), and priors for every MP3D category

**Branch:** `feat/objnav-habitat-mp3d-nadav`
**Status:** done (fixes in the working tree, uncommitted; 15-episode verification run complete)
**Roadmap item:** the failures of the MP3D ground-truth run (entry 043), layer by layer; after the closing (entry
044) this is the oracle's priors and the room-graph bookkeeping. The storey model (layer 1), the closing (2), the
3B oracle's flat numbers (3) and SPL (5) are not touched.

## Goal
Two items from entry 043's report:

1. The 3B node oracle had no prior for the fifteen MP3D categories HM3D lacks and wrote "a table is in the
   bathroom", "a stool in the hallway", "office equipment in a garage" for gym equipment. Cover all 21.
2. Layer 4 in full: slivers of 0.2-0.8 m2 offered as nodes (12.9 % of the node lines of the campaign, the model's
   first choice in 12 % of its calls), a false "no unexplored frontier left" that flipped back in 167 of the 266
   episodes that ever read it, room labels swinging on the same objects, and oracle bursts (six or more calls in 25
   actions in 183 episodes; QUCTc6BB5sX/000069 asked 27 times in 230 actions inside a 5 m patch).

## What the recordings show (24 local re-runs of entry 044, `runs/baseline21` + `runs/diag_seen3`)
- Tiny rooms (under 1 m2) offered 212 times in 521 oracle calls, the argmax in 65. All were watershed cuttings at a
  doorway or along a wall; a door on one made it "a room" under the fragment rule (which exempted doored rooms), and
  a home object of the target credited to one (the dining table standing across the line) kept it a node.
- Label churn: 66 label changes over 24 episodes, most of them a room losing or regaining one kind of object as the
  boundary moved (`dining_room` -> `hallway` -> `dining_room` on `chair, lamp` / `chair` / `chair, lamp`), the rest
  the 3B labeller answering the same evidence differently on a count change or the periodic refresh.
- Bursts: the mid-walk reconsideration fired on a change of ANY room's object-class set, so the ground-truth detector
  crediting a new class to some room on nearly every action re-asked the oracle on consecutive actions over the
  same two or three nodes (`semantic_replan` with `order: []`). In the SELECT state (nothing in force, the fallback
  exploring) a renumbered room id was "a node without a probability" and bought a call at once, each time.
- The HOUSE line was built BEFORE the graph's label update inside `reason`, so at the first call it read "rooms
  found: none yet; 6 rooms unidentified" over a node list that read `type=bathroom`.

## The fixes
| where | what |
|---|---|
| `core/mapping/topology/room_priors.py` | `TARGET_HOMES` (likely / rare room types) for all 21 MP3D categories, `IMPLAUSIBLE_ROOMS` and `HOME_OBJECTS` extended to all 21; `_TARGET_ALIASES` (dataset names, accept labels) kept apart from `_ALIASES` (object classes); `target_home_line`. Side effect, accepted: `target_key` resolving every category makes `target_context.suspect` (room conflict on a STRONG label) live for the fifteen it ignored -- `context_rejections` stayed 0 in the re-run. |
| `search_node_oracle.py` | `WHERE IT LIVES:` line in the user prompt (`SearchContext.target_home`), STEP 1 told to start from it; `size=under 1m2` instead of `0m2`; the HOUSE sentence covers slivers and unreachable frontier. |
| `rooms/room_scans.py`, `LoopSettings.tiny_room_m2` (1 m2) | a room under it with no live frontier is a fragment, door or no door; `_exclusions` does not apply the `home_object_kept` exception to such a fragment (no jamb to hide anything behind). |
| `rooms/house_context.py` | `frontier_state` tri-state (reachable / only slivers / unreachable from here / none) and `FRONTIER_WORDS`; the loop's `_frontier_state` reads "unreachable" off the rooms' own frontier counts (`graph.facts`, over the whole map) -- the inventory holds only what the agent's free component reaches, so read from it alone the tri-state collapsed to reachable / none (the second re-run showed no "unreachable" in 91 prompts). |
| `rooms/discovery.py` | `semantic_signature` = (storey, offered ids, type per offered id, home-object kinds per id); `semantic_change` reads a type change on a room common to both readings or a home-object kind new to the storey -- ids moving is geometry. The signature is re-read after the loop's exclusions so its own doing is not the next action's clue. |
| `rooms/room_search_loop.py` | `reconsider(semantic=...)`: geometry alone waits for the loop point when the offer after exclusions equals the last call's, or the last call was under `reconsider_min_actions` (8) ago -- without arming `_needs_reason` (a release arms it; armed, SELECT consumed it on the same action: 10 of 18 calls in QUCTc6BB5sX/000069's second re-run). SELECT's "a room without a probability buys a call" waits the same window while a valued room stands (`unvalued_waits`). A release for `unreachable` / `blocked` / `transit_timeout` re-selects on the standing numbers instead of re-asking (`releases_unasked`): a node the agent could not reach says nothing about the house, and Z6MFQCViBuw/000154 had asked at four consecutive unreachable releases (actions 32-35). `refresh_labels` before the HOUSE line whatever `type_prior` says. |
| `core/mapping/topology/revisable_room_labels.py` | `relabel_hysteresis`: a differing verdict without a new kind of evidence is held until it repeats; a kind LOST (objects a strict subset of those the label was adopted on, `_label_kinds`) is held through repeats for `refresh_steps` (50) actions -- a room that stays smaller that long was split for real. The hold survives `partition_changed`, which the watershed raises on most actions (cleared with the caches, it never held: room 3 of EU6Fwq7SyZv/000083 relabelled at once on a lost lamp). `held` counter, `room_labels_held` in the episode info and lean rows. |
| `mp3d/campaign.py` | test isolation: `main` passes `runner=subprocess.run` at call time; bound as the default it ignored the tests' monkeypatch and spawned a real `run.py` against the live Ollama (a stray child hung the rig for 20 min on 2026-11-09). |

## The 15 episodes (`runs/layer4_15`; baselines `runs/baseline21` + `runs/diag_seen3`, entry 044's `runs/fixed15`)
Twelve with a local baseline, chosen for the layer-4 symptoms (bursts, tiny argmax, churn, flips), plus three for
the category priors (counter, stool x2, gym equipment) without one. Ground-truth perception, `qwen2.5:3b-instruct`
in both roles, the campaign's recorded answers replayed where the prompt is unchanged (the oracle's is not: live
calls), two scenes at once, `--record`. An intermediate run with the first half of the fixes (`runs/layer4_15_mid`:
signature, same-offer rule, tiny fragments, tri-state, priors, same-kinds hysteresis) is kept as a data point.

Per episode: oracle calls / most calls in any 25 actions / tiny (<1 m2) nodes offered / calls whose argmax was a
tiny node / "no frontier" flips / label churn (distinct labels per room beyond the first, summed), then the outcome
with the DTG. The last column is the new counters: reconsiderations spaced / unvalued rooms made to wait / held
label verdicts.

| episode | category | baseline | `runs/layer4_15` | spaced / waits / held |
|---|---|---|---|---|
| QUCTc6BB5sX/000069 | sink | 27 / 6 / 9 / 5 / 2 / 0, S@449 | 7 / 1 / 0 / 0 / 0 / 1, F@500 1.2 m | 11 / 0 / 3 |
| EU6Fwq7SyZv/000083 | cushion | 64 / 15 / 1 / 0 / 0 / 4, F@500 | 7 / 2 / 0 / 0 / 0 / 3, F@500 2.0 m | 5 / 3 / 6 |
| EU6Fwq7SyZv/000118 | chair | 62 / 7 / 90 / 30 / 0 / 7, F@500 1.6 m | 6 / 3 / 0 / 0 / 0 / 3, F@500 1.0 m | 44 / 0 / 3 |
| Z6MFQCViBuw/000005 | chest of drawers | 27 / 10 / 17 / 11 / 1 / 1, F@500 11.8 m | 19 / 6 / 0 / 0 / 1 / 4, F@500 11.8 m | 23 / 27 / 7 |
| Z6MFQCViBuw/000154 | table | 10 / 5 / 11 / 7 / 0 / 3, F@500 3.9 m | 5 / 2 / 0 / 0 / 0 / 2, F@349 1.3 m | 2 / 5 / 2 |
| 2azQ1b91cZZ/000058 | table | 24 / 12 / 7 / 1 / 0 / 5, S@373 | 18 / 4 / 1 / 0 / 2 / 3, F@500 4.7 m | 20 / 1 / 24 |
| oLBMNvg9in8/000109 | picture | 25 / 9 / 2 / 0 / 0 / 3, S@295 | 11 / 3 / 0 / 0 / 0 / 1, S@314 | 11 / 1 / 1 |
| x8F5xyUWy9e/000038 | sink | 21 / 5 / 5 / 0 / 0 / 8, F@500 11.0 m | 3 / 1 / 0 / 0 / 0 / 2, F@319 3.7 m | 0 / 0 / 3 |
| TbHJrupSAjP/000112 | cushion | 60 / 7 / 10 / 0 / 0 / 6, F@500 1.1 m | 2 / 1 / 0 / 0 / 0 / 0, F@105 2.6 m | 5 / 1 / 0 |
| zsNo4HB9uLZ/000104 | cushion | 7 / 4 / 5 / 2 / 0 / 0, S@204 | 1 / 1 / 0 / 0 / 0 / 0, S@40 | 0 / 0 / 0 |
| Z6MFQCViBuw/000128 | picture | 7 / 1 / 9 / 3 / 0 / 3, F@500 10.2 m | 6 / 1 / 0 / 0 / 0 / 0, S@432 | 0 / 0 / 3 |
| oLBMNvg9in8/000189 | bathtub | 6 / 2 / 2 / 1 / 0 / 3, S@375 | 4 / 1 / 1 / 1 / 0 / 2, S@327 | 0 / 0 / 6 |
| 2azQ1b91cZZ/000134 | counter | (no local baseline) | 7 / 3 / 4 / 0 / 0 / 2, S@264 | 4 / 1 / 7 |
| zsNo4HB9uLZ/000180 | stool | (no local baseline) | 5 / 2 / 0 / 0 / 0 / 0, S@191 | 0 / 0 / 0 |
| QUCTc6BB5sX/000180 | gym equipment | (no local baseline) | 9 / 1 / 6 / 0 / 0 / 4, F@500 5.6 m | 0 / 0 / 7 |

Over the twelve with a local baseline recording:

| symptom | baseline | re-run |
|---|---|---|
| oracle calls (real, reuses excluded) | 340 -- one per 18 actions | 89 -- one per 49 actions |
| most calls in any 25 actions, summed | 83 | 26 |
| tiny nodes offered / the model's first choice | 168 / 60 | 2 / 1 |
| label churn | 43 | 21 |
| "no unexplored frontier left" then frontier again | 25 + 5 flips over the 24 baseline episodes, in the model's `house` reading (about 1.2 per episode) | 3 in 15 episodes by the prompt itself, 4 by the reading (about 0.2 per episode) |
| success | 5 / 12 | 4 / 12 (6 / 15) |

What the counters say fired: 125 reconsiderations spaced, 39 unvalued rooms made to wait, 43 unreachable releases
not re-asked, 72 label verdicts held (over 15 episodes, 5,341 actions). The intermediate run `runs/layer4_15_mid`
(the first half of the fixes) sat at 146 calls / 38 bursts / 38 tiny offered / 5 tiny argmax / 24 churn on the same
twelve, and `runs/layer4_15_v2` (all but the facts-based frontier state and the unasked releases) at 97 / 29 / 2 / 1 / 18.

**The priors.** With the WHERE IT LIVES line the model's `home` step follows the briefing for the categories it had
no prior for: counter -> "kitchen, bathroom, laundry room" in 7 of 7 calls; bathtub -> bathroom in 4 of 4;
chest of drawers -> bedroom in 7 of 7; sink -> kitchen or bathroom; picture -> "rare in bathrooms, storage
closets, laundry rooms". What it still gets wrong is its own: a stool read "chair or stool, often in a bedroom" once
in four, gym equipment "office equipment" once in seven, a cushion "a chair is most likely in a dining room" twice
in six (the 3B model confusing the target with the room's objects) -- layer 3's problem, the model's reading, not its
briefing.

**The outcomes are not the yardstick here.** The same code flipped x8F5xyUWy9e/000038 S@149 -> F@319 and
2azQ1b91cZZ/000134 S@493 -> S@264 between `runs/layer4_15_v2` and the final run: the 3B oracle's live answers vary
and the trajectories diverge, so on 15 episodes SR moves by chance. What changed hands against the baseline:
Z6MFQCViBuw/000128 (picture) and, in `v2`, x8F5xyUWy9e/000038 (sink) succeed where the baseline wandered 10-11 m
away; QUCTc6BB5sX/000069's baseline success was a closing lock at action 437 on a sink the agent had seen since
action 9 inside the same 6 x 4 m patch (14 border rejections this time, no lock -- entry 044's sliver); 2azQ1b91cZZ/
000058's table landmark at x = 1 m was released `unreachable` and the fallback explored east for the remaining 150
actions (a routing failure, not a bookkeeping one); x8F5xyUWy9e/000038 stopped at another sink instance 7 cm tall
(a ground-truth sliver, entry 044); TbHJrupSAjP/000112 locked a cushion 0.8 m away that counts for no goal view
point (DTG 2.64 m, the closing's standing-spot problem). None of the lost episodes shows a layer-4 mechanism in its
recording; `context_rejections` -- the `target_key` side effect -- stayed 0 everywhere.

**Still visible.** Z6MFQCViBuw/000005 (chest of drawers 11.8 m away, on another level: layer 1) still asks 19 times
with a burst of 6: four `room_reopened` events in eleven actions (41-57) each arm a call, and five geometry
reconsiderations after 450 pass the window with the offer changed. The "unreachable" frontier word never came up:
every "no unexplored frontier left" in the 15 coincided with zero frontier in every room and zero reachable goals --
none of these episodes is guard-boxed, which is layer 1's set.

## Harness
- The zombie-worker hazard of entry 044 did not recur (Z6MFQCViBuw/000005 ran to 500 four times). Load stayed under
  10 % CPU and 40 % GPU with two scenes.
- Two scenes of one attempt failed their PREFLIGHT: the oracle probe timed out after 900 s on the native Ollama while
  two other scenes were calling it; the server answered in 0.2 s a minute after the campaign ended, and the resume
  (same command, same `--output`) ran both. One EU6Fwq7SyZv worker died with exit -11 (the renderer) between its two
  episodes; the resume ran the second.
- The runtime rig hung for 20 minutes: `test_a_sweep_whose_scenes_crashed_does_not_exit_zero` spawned a real
  `run.py` (the monkeypatched `campaign.subprocess` never reached `run_campaign`'s bound default) that waited on the
  busy Ollama. Fixed in `campaign.main`; the rig takes 3 minutes again.
- Five launches were needed: each of the first four showed, in its first finished episodes, a mechanism the previous
  code had not addressed (the pid-keyed signature, the spaced clue consumed in SELECT, the cleared hold, the
  reachable-only inventory); the runs are kept as `runs/layer4_15_aborted*`, `_mid`, `_v2`.

## Open
- The watershed renumbering itself (ids reassigned at a boundary split, the anchor rule of 2026-10-05) is untouched:
  the fixes make the loop and the labeller indifferent to it rather than stop it.
- `reconsider_min_actions` = 8 and `tiny_room_m2` = 1.0 are first values, set from the recordings, not swept.
- The 3B oracle's numbers stay near-flat (layer 3): the WHERE IT LIVES line fixes what the model is TOLD, not what
  it does with it.
