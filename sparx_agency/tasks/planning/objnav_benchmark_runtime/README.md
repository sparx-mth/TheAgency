# ObjectNav runtime and Gibson adapter

This checkout composes observed mapping, object evidence, room reasoning,
RPT* and A*/WA* with the existing benchmark harness. Gibson `cc57809f` is the
base; the completed detector delta `5c465f97` is preserved without merging the
divergent infrastructure branch `60d91714` wholesale. Other simulator worktrees
remain separate.

## Exploration selection

- `--explorer frontier`: preserved observed-frontier baseline (default).
- `--explorer falcon`: bounded **FALCON 2D/2.5D ObjectNav adaptation** with
  connectivity-aware decomposition, coverage-path/SOP ordering, viewpoint
  refinement and free-space routes. This is not the unchanged ROS aerial binary.

The hierarchy is local exploration → room LLM → RPT* room selection → A*/WA*
transit, with bounded target interrupts/recovery. Mapping and detection continue
in both motion phases. All emitted actions count toward the unchanged Gibson
500-action limit; FALCON bursts have a separate 48-action default allowance.

See the [source mapping and runbook](gibson/BOUNDED_FALCON.md),
[results and limitations](gibson/FALCON_RESULTS.md), and
[Gibson protocol guide](gibson/PROTOCOL_AND_TUNING.md).

For the **15 distinct-building, 30-recording comparison**, see
[DISTINCT_BUILDINGS.md](gibson/DISTINCT_BUILDINGS.md). It uses frozen generated
ObjectNav starts in official Gibson training buildings, not repeated episodes
in the five validation scenes. Policies and the 500-action cap remain unchanged.

For **multi-story houses**, both explorers now retain per-floor scene graphs
linked by observed stair traversals. See [MULTISTORY.md](gibson/MULTISTORY.md)
for the architecture and the five-building × three-episode development campaign,
including one preselected recording per building and explicit scoring caveats.

## Reusable pieces

| Component | Responsibility |
|---|---|
| `methods/rpt_policy.py`, `rpt_settings.py` | Detector/mapper composition, baseline selection and recorded configuration |
| `methods/falcon_policy.py` | Explicit bounded hierarchy; existing RPT* and target-evidence integration |
| `methods/falcon_regions.py` | Observed local scopes, reachable entry goals and split/merge visit history |
| `methods/falcon_motion.py`, `falcon_routes.py` | Ground-safe action checks and paused route persistence |
| `methods/observed_map.py`, `floor_context.py` | Persistent floor-local maps, graphs, target evidence and paused room clocks |
| `methods/multifloor_policy.py`, `stair_terrain.py` | Observed stair proposals/traversal and building-level scheduling |
| `methods/route_memory.py` | Committed routes, snapped arrival and stagnation safeguards |
| `methods/object_evidence.py` | Alias/frame deduplication, multi-view support and bounded rejection |
| `methods/scene_graph.py`, `room_labels.py`, `doors.py` | Observed rooms/doors and revisable accumulated room reasoning |
| `methods/exploration_metrics.py` | Observed-area proxy curves, actions, revisits, stagnation and latency tails |
| `habitat/simulator.py` | Registered RGB-D, metric depth, ENU pose and existing discrete actions |
| `gibson/run.py`, `frozen_eval.py`, `compare_explorers.py` | Exact protocol execution, source/configuration checks and controlled comparison |

Algorithms live in `core/planning/exploration/falcon/`; they are ROS-free and
simulator-neutral. Host-side numpy/scipy dependencies do not enter the existing
Noetic exploration facade. The Gibson bridge retains its existing camera-height
body configuration; future simulators must supply their own explicit embodiment
rather than inherit Gibson defaults. Persistent floor identity/connectivity lives
in `core/planning/exploration/floor_atlas.py`; depth-based stair handling is the
host-side ground-robot adaptation, not an aerial-controller change.

## Completed detector preserved

**YOLO-World X-v2 is the completed-integration default**. Pretrained LLMDet
Swin-L remains selectable on a dedicated HTTP service; S/L are explicit
checkpoint rollback choices. No confidence thresholds, target-confirmation
rules or model architecture were redesigned for FALCON.

`HttpDetector` validates vocabulary, checkpoint/configuration, runtime and
prompt provenance, and optionally `expected_backend`. `gibson.run` exposes
`--detector-backend`. Missing or drifting services fail visibly. Models do not
load in the evaluator. Use the [detector setup and runbook](../../mapping/scene_graph/serve/README.md),
CPU model services and the existing Habitat environment for rendering.
Previously saved explicit demo checkpoint choices are not overwritten.

## Evaluation boundary

Ground-truth floor maps, goal distances and first success-region entry are
**evaluator-only**. The policy gets RGB, metric depth, pose and public task/action
information, not a navigation oracle. Additional native collision/entry
statistics are written to `evaluation_diagnostics.jsonl`; the existing primary
native-score schema and Gibson SR/SPL rules are unchanged.

Coverage is **ever-observed occupancy area**, a proxy, not ground-truth coverage.
The initial observation is excluded from gained-per-action. Curves sample
observations at decision indices. Masks are keyed by stable floor ID, so a
return visit does not recount known area. Transient stair terrain is excluded.
Historical reset-only results retain their original coverage limitation.

Recording and smooth replay remain optional; interpolated replay images are
never policy inputs. Use the existing Gibson freeze workflow for later published
validation, not the paired development runner. Previously inspected starts are
not an untouched holdout; no small-run or SOTA claim is supported.
