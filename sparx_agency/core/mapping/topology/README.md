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
  a door border is a hard barrier and never merges (a door snapped to a
  choke is a barrier by the carved mask's sides instead -- and, since
  2026-10-05, falls back to the plain barring disk when its disk does
  not actually sever the floor).
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
  Legacy callers still cache by class set. Online callers can request
  `min_classes`, `count_sensitive=True` and `classify(..., refresh=True)` to
  revise a label after more objects are observed or a region is resegmented,
  and `cached(classes)` to read a verdict already held for exactly that
  evidence before deciding whether a call is worth buying.
  Refresh failures propagate without replacing the previous cache entry.
- `search_oracle.py` — per-room target probabilities. The model is asked
  only when the prompt it would be shown has changed (room ids, labels,
  observed classes, area); `searched_s` and `frontier_clusters` are
  applied in code, so a query over an unchanged map re-scores the kept
  reply against the fresh effort numbers instead of spending a call
  (`OracleResult.reused`, `SearchOracle.reuses`). A reply that fell back
  to uniform is never kept.
- `search_node_oracle.py` — the successor of `search_oracle` for the
  room-search loop, written for a capable model (the runtime routes it to
  `LLMConfig.reasoning_model`, a 14B instruct model). One call per loop
  point over every NODE the search could go to next -- **every room of the
  storey**, finished ones included, each with its type (or `unknown`;
  `kitchen?` when named from a single kind of object), size, frontier
  left, `status=` (`never_entered` / `entered` / `scanned(<how>, <ago>)`),
  time searched, objects seen; each opening with the room it opens from
  and what was glimpsed through it; and, in the multi-storey development
  protocol only, each staircase -- asking for the probability that going
  there NEXT finds the target. **LLM-first since 2026-10-07:** the user
  prompt carries a HOUSE line (every room type found with its status, the
  rooms still unidentified, the openings not looked into, whether
  unexplored frontier is reachable, and `actions used about N of 500`),
  and the system prompt asks for three written steps before the numbers --
  `home` (where the target lives: a toilet in a bathroom or an en-suite
  reached through a bedroom; a television in a living room, then a
  bedroom, rarely a kitchen, never a bathroom), `house` (one kitchen, one
  living room, one to four bedrooms, one to three bathrooms: what the rooms
  found say about the rest), `stage` (early: the unexplored first; late:
  the detector may have missed the target in a scanned room of the right
  type) -- plus a `pass` verdict (`first` / `second`) the loop reads to
  decide whether scanned rooms are offered again. It forbids reasoning
  about distance, which RPT* charges. Code keeps only the contract:
  parse, drop invented ids, give an omitted node a small share, clamp
  below 1, refuse a reply with no usable node, reuse the reply when the
  prompt is byte-identical (effort and budget numbers are shown in coarse
  steps for that). The three arithmetic floors (`unexplored_floor`,
  `unexplored_elsewhere`, `home_floor`) are knobs, **off by default**; the
  storey/stairs rules (`home_here`) are a `STAIRS_SUPPLEMENT` appended only
  when a staircase is a node. `LLMClient.chat_json` takes `reasoning=True`
  to select the reasoning model, its timeout, its reply cap and its
  context window (`LLM_REASONING_MODEL`, default `qwen2.5:14b-instruct`;
  `LLM_REASONING_TIMEOUT_S`; `LLM_REASONING_MAX_TOKENS`;
  `LLM_REASONING_NUM_CTX` -- asked for explicitly because Ollama truncates
  an outgrown window from the front, which would drop the system prompt);
  a request that runs into its timeout is not retried.
  `tasks/planning/objnav_benchmark_runtime/tests/probe_node_oracle.py` runs
  five single-storey scenarios against the live model and checks the
  judgement, not just the schema.
- `target_matcher.py` — target-name matching: exact → cache → LLM →
  token-overlap fallback. The fallback rung is not implemented here: it
  delegates to `core/common/label_match.py`, which is the same rule the
  visual-servo acquisition gate acquires on.

## Tests

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core/mapping/topology/tests
```
