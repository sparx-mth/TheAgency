# core/mapping/topology

Topological understanding of a mapped environment: split free space
into rooms, relate rooms to doors, objects and frontiers, and reason
about them (including via an LLM). numpy/scipy/skimage/networkx, plus
`requests` in the LLM modules below — no ROS. This is a host-owned
path, so scipy and skimage are allowed.

## Two room-splitting pipelines

Both split free space into rooms at doors; callers choose one.

**Grid-based (flown).** The pipeline that actually ran in the SJTU
hospital sim, ported from the old stack's `semantic_mapper_node.py`:

- `room_segmentation.py` — `compute_rooms()`: heal the free mask,
  medial-axis skeleton (DT-ridge fallback), punch a disk through the
  skeleton at each discovered door, 8-connected label, paint every free
  cell with its nearest skeleton pixel's label, drop tiny rooms.
  Also owns the occupancy value constants (`UNKNOWN`/`FREE_MAX`/`OCC_MIN`).
- `room_registry.py` — `RoomRegistry`: greedy best-IoU 1:1 matching of
  fresh room masks to the previous tick's, handing out persistent,
  monotonically increasing, never-reused pids.
- `room_watershed.py` — `segment_rooms_watershed()`: the same triple,
  derived from clearance geometry instead of the skeleton's topology.
  Every local maximum of the distance field seeds a room, the watershed
  pushes the boundaries into the narrow places, and a listed door is
  carved out of the flood mask so it always separates. Written because
  the skeleton cut COLLAPSES into one dominant room as coverage grows;
  the measured table is in its module docstring.
- `room_merge.py` — `merge_basins_by_dynamics()`: the watershed's one
  defect is over-segmentation (one room, two clearance peaks), repaired
  by merging adjacent basins whose *dynamics* — the clearance lost from
  the shallower peak down to the saddle between them — fall below a
  threshold. Union-find on a region adjacency graph built in one pass;
  a door border is a hard barrier and never merges.
- `room_adjacency.py` — `room_adjacency()`: which rooms genuinely touch,
  and `iter_label_borders()`, the single border scan both it and
  `room_merge` use. This is the room-to-room edge rule for the scene
  graph: proximity to a shared door is not connectivity.
- `room_stats.py` — free-function helpers over the grid and a room
  label image: door discovery (`discover_doors`), door-to-room linking
  via an annulus (`link_doors`), vetting those links against adjacency
  (`door_room_pairs`), frontier cluster counting
  (`count_frontier_clusters`), room-at-cell majority-vote lookup
  (`room_at_cell`), and the golden-ratio room color (`room_color`).

**Graph-based (MORE, untested in flight).** The Werby et al. (2025)
implementation operating on a Voronoi navigation graph:

- `voronoi.py` — `extract_voronoi_graph()`: occupancy → boundary cost
  field → Voronoi skeleton → sparse networkx navigation graph.
- `graph_utils.py` — graph sparsification and junction/dead-end queries.
- `room_separation.py` — `separate_rooms()`: Gaussian door-probability
  field, boundary integral per edge, cut high-scoring edges; connected
  components are the rooms.

## Scene reasoning

- `room_object_graph.py` — hierarchical root → rooms → objects graph
  (MORE convention); assigns objects to rooms by nearest Voronoi node.
- `llm_nav_planner.py` — two-stage LLM planner: prune irrelevant
  objects from the room-object tree, then produce route instructions
  using shortest paths on the Voronoi graph.

## LLM modules

The reasoning rungs the scene-graph search stack flies on. These add
`requests` to the package's dependencies (the only one that is not
numpy/scipy/skimage/networkx). **Read each one's failure contract, they
differ on purpose:** `search_oracle` and `target_matcher` degrade to an
offline answer when the LLM is off or unreachable, while
`room_classifier` lets transport/parse errors propagate so the caller
decides whether to keep a stale label — nothing is cached on failure.

- `llm_client.py` — Ollama / OpenAI-compatible HTTP client, plus
  `coerce_bool()` (never `bool()`: a small model answers with the word
  quoted, and `bool("false")` is True).
- `room_classifier.py` — object list → room type via LLM.
- `search_oracle.py` — per-room target probabilities.
- `target_matcher.py` — target-name matching: exact → cache → LLM →
  token-overlap fallback. The fallback rung is not implemented here: it
  delegates to `core/common/label_match.py`, which is the same rule the
  visual-servo acquisition gate acquires on.

## Tests

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core/mapping/topology/tests
```
