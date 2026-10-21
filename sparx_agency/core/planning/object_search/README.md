# object_search — the zero-shot object-search method

A `SearchPolicy` for the ObjectNav contract in
[`core/planning/objnav`](../objnav/README.md): given ground-truth RGB-D and
pose, find a named object category in an unseen home with a discrete action
set, and STOP inside the success region. Observed-only mapping, a room graph
with revisable LLM labels, a node oracle that values every room from the big
picture, RPT\* for the visit order, frontier or planar-FALCON local exploration,
A\*/WA\* routes, and a high-priority target closing. Training-free; no
simulator geometry beyond what the adapter declares.

```text
 ObjNavEnv (adapter, tasks/)                 RPTSearchPolicy (this package)
 ─────────────────────────────               ─────────────────────────────────────────────────────────
 ObjNavObservation ───────────────────────►  perception/   detector boxes → coherent 3-D centroids → landmarks, doors
                                             world_model/  per-storey log-odds occupancy, sight ledger, stair surfaces
                                             rooms/        watershed rooms → labels → node oracle → RPT* order → the loop
                                             exploration/  frontier sweep, glances, peeks, the fallback ladder, escape
                                             multifloor/   spawn-floor guard (default) or the building coordinator
                                             closing/      the target takeover: verify, approach, pitch, STOP
 NavigationCommand ◄──────────────────────  policy.py     one decision per frame, in world coordinates
```

The policy never names a simulator, a dataset or a wire format. It is built
with three services -- an object **detector**, an optional box-prompted
**segmenter** and a **room-reasoning LLM** -- behind the protocols of
[`interfaces.py`](interfaces.py); the HTTP clients that implement them, the
Habitat bridge, the Gibson files and the run CLI live in
[`tasks/planning/objnav_benchmark_runtime`](../../../tasks/planning/objnav_benchmark_runtime/README.md).
`tests/test_package_boundaries.py` enforces the line.

## Layout

| package | holds |
|---|---|
| `policy.py` | `RPTSearchPolicy`: composition, the per-frame decision order, `configuration()` and `episode_info()` |
| `settings.py` | `RPTSettings`, the one frozen-config record of the method's knobs (`--policy-config` overrides its fields) |
| `interfaces.py` | `Detection`, the `Detector` / `Segmenter` / `RoomReasoner` protocols, the frame-error classes and the failure contract |
| `perception/` | `cycle.py` (one synchronous prediction per frame, fusion, target evidence), `projection.py` (box → world centroid with coherent depth), `doors.py`, `target_context.py` (is this candidate plausible where it stands?) |
| `rooms/` | `scene_graph.py` (observed rooms, doors, labels, the oracle's nodes), `room_search_loop.py` (the loop), `room_scans.py`, `room_vantage.py`, `stair_nodes.py`, `opening_nodes.py`, `house_context.py`, `oracle_retry.py`, `floor_context.py`, `discovery.py` |
| `world_model/` | the maps the decisions plan on: `observed_map.py` (RGB-D → per-storey occupancy on the core log-odds integrator, keyed by the storey `core/localization/floor_atlas` reports), `sightlines.py`, `stair_terrain.py`, `stair_grid.py`, `peek_stairs.py` |
| `exploration/` | `exploration_fallback.py`, `escape_probe.py`, `frontier_sweep.py`, `path_glances.py`, `doorway_peek.py`, `doorway_candidates.py`, `exploration_metrics.py` |
| `multifloor/` | `spawn_floor_guard.py` (the benchmark default: one storey), `multifloor_policy.py` and the stair modules the development protocols enable |
| `closing/` | `target_closing.py`, `target_path.py`, `target_backoff.py`, `approach_history.py` |
| `falcon/` | the planar FALCON explorer wired to the room search (`local_exploration: "falcon"`) |
| `route_memory.py` | the committed-route memory: when a planned path is kept, re-used or released |
| `baselines/` | `falcon_yolo.py`: goal-free FALCON + the same perception and closing, no LLM (`--agent falcon-yolo`) |
| `tests/` | the package's contract tests: boundaries, a service-free smoke run, the failure contract; see below |

What the method shares with the rest of `core` it imports from there rather
than keeping a copy: the map vocabulary (`core/planning/environment`), the
A\*/WA\* planners, the FALCON port, the RPT\* room solver, frontier ranking and
room costs (`core/planning/exploration`), the room watershed, registry and node
oracle (`core/mapping/topology`), the landmark map (`core/mapping/objects`),
the camera geometry and the action converter (`core/planning/objnav`), the
storey localiser (`core/localization/floor_atlas`) and the camera-tilt
controller and servo law (`core/control/camera`: where to *look* is a control
process beside where to *move*; the two meet in the `NavigationCommand`). Its
own policy-free pieces went the other way, into the subsystem they belong to:
box extent and occluders, detection dedup, target evidence and support
surfaces in `core/mapping/objects`; the fused 3-D target memory in
`core/mapping/tracking`; the room priors and revisable room labels in
`core/mapping/topology`.

`world_model/` is mapping by nature and lives here by necessity: its products
are the planners' map vocabulary (`OccupancyGrid2D` of
`core/planning/environment`) and its inputs the contract's camera geometry,
and `core/mapping` imports nothing from `core/planning`. Moving the shared map
and pose vocabulary down into `core/common/types` would let it move; that is
a repository-wide change (over a hundred importers) and not this package's.

## How a component talks to the policy

The sub-components are constructed with the policy and read its shared state
through it (`self.policy.mapping`, `.episode`, `.route_memory`, `._navigate`):
one episode-scoped composition root, not a web of injected handles. The
convention is deliberate and the price is known -- a component is only
testable through a policy -- so the regression rig builds a real policy with
fake services on a synthetic map (`tests/fixtures.py` here; the benchmark
rig's `test_method.setup_policy`). Keep new components on the same pattern
rather than threading a fourth style through the package.

## Runtime

Python 3.8-compatible syntax, numpy and scipy; no ROS, no torch, and no
transport of its own -- the detector, segmenter and LLM clients are handed in
(`core/mapping/topology`'s facade still re-exports its HTTP `LLMClient`, so
`requests` is in the import closure; the method never calls it). It is not
imported by the FALCON Noetic container (which has no scipy) and is not part
of the `core/planning/objnav` import closure, so the contract's numpy-only
rule does not bind it. Importing the package root loads nothing; import the
policy by module path.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core/planning/object_search -q
```

The behavioural regression suite of the method -- the room loop, the closing,
the fallback ladder, several thousand assertions -- is the benchmark runtime's
rig: `tasks/planning/objnav_benchmark_runtime/tests/` (run it after every
change here). It lives there because it reasons about the Gibson camera, the
dashboard renders and the recorded traces, and because a rig belongs beside
the thing it exercises; the package's own tests check what the rig cannot --
that the method stays free of the adapter.
