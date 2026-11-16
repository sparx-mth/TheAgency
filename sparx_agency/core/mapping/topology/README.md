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
- `room_extent.py` — `room_extents()` (since 2026-11-16; written 2026-11-11
  on `feat/objnav-room-extents-nadav` and ported by hand): for every room of
  the live partition, where its walls are (its boundary faces classified by
  what lies beyond them -- a wall, another room, settled unknown, unseen
  unknown, floor of no room -- grouped into four sides in the storey's wall
  frame), its dimensions and footprint area, whether that size is FINAL
  (every side seen: at most `ExtentParams.open_tolerance` of its faces open
  onto unseen space in runs wider than a doorway -- a statement about this
  room of this partition on this map, NOT a promise that the watershed will
  not join it to a neighbour later; the oracle's line words it `all sides
  mapped`, see `search_node_oracle.py` below) or a LOWER BOUND
  (`bounded`, `open_sides`), the agent's distance to each wall when it stands
  in the room, and how much of its floor the camera has covered from
  detection range (`covered_m2`; `covered_fraction`, the share of the whole
  room and an upper bound while a wall is unseen; `left_m2`; furniture
  islands credited from the observed floor around them). Recomputed from the
  map and the partition on every update and keyed by cell, so a room
  renumbered, merged or split by the watershed reads right.
  **It is read off the agent's own map and nothing else.** What
  `room_extents()` is handed is the whole of what it knows: the occupancy
  grid, the partition the watershed cut out of that same grid, the sight
  ledger's two per-cell records (the unknown it settled, the floor it
  observed) and the agent's position. No navigation mesh, no path query, no
  dataset's room or region annotation, no table of typical room sizes: the
  module imports numpy and scipy and nothing of this repository, and
  `tests/test_room_extent.py` pins both its imports and its signature. The
  words the oracle is shown are `search_node_oracle.format_size` and
  `format_coverage` (one copy of the phrasing, with the prompt it belongs
  to); `tasks/planning/objnav_benchmark_runtime/room_extent_replay.py`
  replays a recording through the reading offline.
- `frontier_split.py` — the cut shared by `count_frontier_clusters` and
  the planning side's goal maker (`core/planning/exploration/room_costs.py`
  `frontier_clusters`): a frontier component that wraps round a block of
  unknown is one cluster per side of the block (`frontier_sides`,
  `split_by_side`), so an unmapped room seen along three sides has a way
  in from each; an arc bulging into the open unknown stays whole.

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
  quoted, and `bool("false")` is True). Two optional attachments, both off
  unless a run names them:
  - `llm_answer_cache.py` — `LLMAnswerStore`, answers keyed on the exact
    request (backend, model, both prompts, temperature, seed, window, reply
    cap, format) within the episode in force (the policy sets the store's
    `scope` at each reset: two episodes asking one question get their own
    answers, and a replay gives each episode its own). A run records every
    answer it uses (`LLM_ANSWERS_RECORD=<dir>`, one JSON-lines shard per
    process, prompts included) and can replay another run's
    (`LLM_ANSWERS_REPLAY=<dir>[:...]`); a question asked again within an
    episode gets its first answer. Two runs that
    ask the same questions then get the same answers, whatever the server
    would have said: on 2026-10-31 the temperature-0, seeded 3B answered the
    same prompts differently by request order (`LESSONS.md`). The episode row
    counts `llm_answers` (`replayed` / `repeated` / `asked` / `recorded`).
  - `llm_exclusive.py` — `ExclusiveColdStart` (`LLM_EXCLUSIVE=1`): one
    request at a time across processes (a host-wide file lock), each on a
    freshly loaded model (unloaded, `/api/ps` checked, then asked). The only
    mode measured to make Ollama's answer depend on the request alone: a
    one-token flush or a per-request tag did not, because the server keeps
    the state of many earlier prompts. 6.4 s a call against 3.5 s.
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
  room-search loop (the runtime routes it to `LLMConfig.reasoning_model`).
  One call per loop point over every NODE the search could go to next --
  the rooms of the storey that are still nodes (and, under the deprecated
  exits-as-nodes mode, the openings). A node is a room **together with the
  unseen space its exits open onto** (the room's visit looks through them),
  and the model is asked for **the probability that the target is in the
  part of the node not checked yet**: the room's floor that is left, and
  that unseen space. The detector has looked at the covered floor in every
  frame without finding the target, so this is the probability that
  searching the node finds it -- the `p(v)` RPT* is handed.
  **The division of labour:** the model gives that probability and nothing
  else; distance, and the weighing of distance against the probabilities,
  is RPT*'s. The prompt carries no distance, no position and no word on
  where the robot stands (`here=yes` was dropped on 2026-11-16;
  `SearchNode.here` is kept for the record and never rendered).
  **The model's own knowledge, and none of ours (since 2026-11-16).** The
  system prompt tells the model to use its own reasoning and its own
  knowledge of standard homes and buildings -- how they are laid out, in
  which rooms furniture and objects of each kind are found -- and to derive
  every probability from that. It holds no category knowledge and no rule
  of judgement: no room named as the place of any object, no number or band
  a probability should fall in, no list of a benchmark's goal categories,
  no count of the rooms a home has, no "coverage first, time second", no
  rule on what an exit is worth or how the budget weighs, no worked
  example. The only object it ever sees named is the one on the `TARGET:`
  line of the user prompt -- the name of the object the current episode
  searches for, in every call. What it does hold is FACTS a model cannot
  know: what a node is, what each field of a line measures and how it was
  obtained, the reply format, that the numbers are independent. Before that
  day the prompt was 14,751 characters of our judgement.
  **What a room line holds: everything the scene graph stores about the
  room, except distance.** `type=` (or `unknown`; `kitchen?` when the
  evidence is thin); `size=` and `covered=` from `room_extent.py`;
  `frontier=`; `exits=` (how many stand at a door frame, the unseen floor
  behind them) and what was glimpsed through them; `status=`
  (`never_entered` / `entered` / `scanned(<how>, <ago>)`); `searched=`;
  `connected to:` the types of the rooms a door or doorway joins it to (the
  graph's edges, new on 2026-11-16); `seen:` the objects detected in it
  with their counts, most numerous first (`chair x4, table`; a cut list
  says so: `+3 more`); and `target: detected here, not reached` on a room
  whose confirmed objects include one of the target's own kind
  (`SearchNode.target_seen`) -- the one case in which "the detector looked
  at the covered floor without finding it" is not true, said on the line.
  With the action time in the context (`SearchContext.action_time_s`) the
  durations are written in actions -- the unit of the HOUSE line's budget
  (`searched=40 actions`) -- in the same coarse steps as the time
  rendering. The HOUSE line (every room type found with its status, the
  rooms unidentified, the exits not looked into, the frontier state,
  `actions used about N of 500`) is unchanged.
  **"Mapped", not "final".** A room whose four sides are all located reads
  `size=12m2 (4x3m, all sides mapped)`; one with a side the camera has not
  reached reads `size=at least 12m2 (5+ x 3m, 1 of 4 sides not mapped yet,
  open edge 2m)`, followed by `covered=10m2 (80% of the room, 2m2 left)` or,
  when the size is a lower bound, `covered=3m2 (at most 40% of the room, at
  least 4m2 left)`. As first written the first form read "all walls seen"
  and the prompt called it FINAL -- "that big and no bigger" -- which
  promised more than the reading knows: of 252 rooms so declared in 40 HM3D
  recordings, 73 were over a quarter bigger by the end of the episode, and
  229 of the 233 growths behind that were the PARTITION changing (178
  merges, 51 moved cuts) -- floor that was mapped, and listed under the
  neighbouring room, all along. What the extents know is that no side of
  the room opens onto unmapped space; the words now say that, and the
  prompt adds that the map's division into rooms may still be revised. The
  wording was changed rather than the code: withholding the verdict wherever
  a side is a cut and not a wall would have told the model that mapped
  floor is unmapped.
  **The reply** is unchanged: `home`, `house`, `stage` (three short readings
  in the model's words, recorded), a `pass` verdict (`first` / `second`,
  recorded; the loop's second pass is opened by the geometry) and one
  integer percentage per node, independent, never renormalised. Code keeps
  only the contract: parse, drop invented ids, give an omitted node a small
  share, clamp below 1, refuse a reply with no usable node, reuse the reply
  when the prompt is byte-identical. One arithmetic floor remains as a knob
  and is **off by default** (`unexplored_floor`).
  **No staircase node and no multi-storey supplement** (deleted on
  2026-11-16 with the loop's stair-node code, until the multi-storey work):
  `STAIRS`, `STAIRS_SUPPLEMENT`, the storey lines of the user prompt, the
  `home_here` verdict and `unexplored_elsewhere` are gone. The supplement
  for an exit offered as a node of its own (the deprecated mode) is the
  meaning of that line and nothing else.
  **The call.** `LLMClient.chat_json` takes `reasoning=True`
  to select the reasoning model, its timeout, its reply cap and its
  context window (`LLM_REASONING_MODEL`, default `qwen2.5:14b-instruct`;
  `LLM_REASONING_TIMEOUT_S`; `LLM_REASONING_MAX_TOKENS`;
  `LLM_REASONING_NUM_CTX` -- asked for explicitly because Ollama truncates
  an outgrown window from the front, which would drop the system prompt);
  a request that runs into its timeout is not retried. When one model
  serves both routes the default route asks for the same window
  (`LLMConfig.num_ctx`, `LLM_NUM_CTX`): Ollama keys a loaded model by its
  window, and under load the oracle's reload never came -- the GT sweep of
  2026-10-28 got no oracle answer in 2,000 episodes (`LESSONS.md`). A reply
  whose `prompt_eval_count` shows the prompt was cut, or whose prompt and
  reply filled the window, raises (`_refuse_truncation`).
  **History.** LLM-first since 2026-10-07 (the HOUSE line, the three
  written steps and the `pass` verdict); the frontier statement a tri-state
  and `size=under 1m2` since 2026-11-09; the `WHERE IT LIVES` line and the
  `home_floor` of 2026-11-09 went with `room_priors.py` on 2026-11-15 (below);
  size and coverage from the room extents came on 2026-11-16, with two rules
  ("coverage first", "time is secondary") that the same day's rewrite
  removed with every other rule.
- `search_node_prompts.py` — **the oracle's prompt, by version.** A wording
  is read by one model and may be misread by the next, so the texts a model
  is given beside the facts -- system prompt, openings supplement, the
  frame of the user prompt -- are a named `NodeOraclePrompt`, written
  against ONE model (`tuned_on`) and probed on it. `PROMPTS` holds them by
  name; a run selects one with `LoopSettings.oracle_prompt` (policy config
  `{"room_search": {"oracle_prompt": "<name>"}}`; an unregistered name is
  refused at start-up); the run record carries the name, the model it was
  tuned on and the SHA-256 of its text (`configuration()["node_oracle"]
  ["prompt"]`), and the benchmark factory adds the model that answered and
  whether it is that one (`["node_oracle"]["model"]`). A version is never
  edited after a run has used it -- a changed wording is a new version, and
  the digest shows when that rule is broken. `tests/
  test_search_node_prompts.py` holds EVERY registered version to the
  contract above (own knowledge; no object word, no room-type word, no
  number but the reply's; no rule; no distance; the reply format; a legend
  for exactly the fields a line carries), so a version added for another
  model cannot slip past it. **To add one:** copy the nearest version under
  a new name with the new model's tag, register it, probe it
  (`tasks/planning/objnav_benchmark_runtime/tests/probe_node_oracle.py
  --prompt <name>`, with `LLM_REASONING_MODEL` set to that model), reword
  and repeat, then record its figures in the table below.

  **The probe's figures** (`qwen2.5:3b-instruct`, CPU, temperature 0, seed
  0, one call per scenario, 2026-11-16; each cell is *claims held at the
  model's own numbers / replies usable / claims held as RPT\* receives them*):

  | wording | tuned on | system prompt | 12 default-mode scenarios (`P`) | the first 7 (`O`) | a call takes |
  |---|---|---|---|---|---|
  | the prompt before 2026-11-16 (rules and priors; not a version) | -- | 14,751 chars | 8 / 12 / 10 | 5 / 7 / 6 | 91-130 s |
  | the first facts-only wording of 2026-11-16 (not registered) | -- | 2,955 chars | 3 / 11 / 6 | 2 / 6 / 5 | 54-63 s |
  | `facts-qwen2.5-3b-v1` (the default) | `qwen2.5:3b-instruct` | 3,471 chars | 5 / 12 / 10 | 4 / 7 / 6 | 50-56 s |

  *As RPT\* receives them*: after the loop's rescaling by exploration state
  (a never-entered room x1.5 and at least 0.15; an entered one x0.5 with a
  20-action half-life; under 0.05 no node), which acts on whatever the model
  wrote. Read the table for what it is -- nineteen hand-written scenarios,
  one call each, a model that is brittle to wording (one sentence moves
  three scenarios) and not bit-reproducible on Ollama at temperature 0 (of
  three calls repeated, one came back with another number on one node): a
  difference of one or two claims is noise. What it does show:
  - **On the 3B the facts-only version holds fewer claims at the model's
    own numbers than the prompt of rules did (9 of 19 against 13), and as
    many once the loop has rescaled them (16 of 19 each).** Every reply is
    usable and no never-entered room is put at 0 -- the most harmful
    output, since the loop's floor for such a room applies only above 0;
    the first wording did that in 4 of 19 queries.
  - **The 3B's own knowledge is mostly sound; what it lacks is the
    inference.** Its `home` reading -- where it says the target is normally
    found -- is sensible in most of the nineteen queries ("beds are
    typically found in bedrooms", "refrigerators are commonly found in
    kitchens") and wrong in some (a toilet "typically found in bathrooms or
    kitchens" in three, "in bedrooms or kitchens" in one). What it does not
    do is combine fields. The two scenarios that ask exactly what coverage
    without a detection means -- a room of the fitting type covered whole
    against a barely seen room of another type (`P7`, `P11`) -- were held
    by the old prompt, which stated the rule, and by none of the facts-only
    wordings tried (twelve were asked `P7`, nine `P11`): the 3B answers 80
    to 100 % for the covered room in all but one of those replies ("highly
    likely in mapped area of living room"). The loop's discount of an
    entered room is what carries both.
  - **What helped, each a fact:** what a node is (the room and the unseen
    space behind its exits); that a detector looks for the target in every
    frame and has not found it on the covered floor; which part of a node
    the number is about; that `unknown` may be any kind; the two ends of
    the scale; the question restated, with the target's name, as the last
    line of the user prompt; the reply format on one line, "ending with one
    nodes entry per node" (a 3B closed the object after its three readings
    otherwise).
  - **What did not:** a negation on every line (`target not detected
    there`: never-entered rooms at 0); explaining the `target: detected
    here, not reached` note in the legend (the 3B read its absence as "not
    there": every node of a query at 0 in 4 of 19 queries, 4 of 19 claims
    held -- which is why the 3B's legend leaves the note to its own words);
    the floor left as a field of its own (`left=`: two of twelve replies
    unusable); an `unchecked=` framing; asking in the `home`/`house`
    readings for the kinds of room and which were found; a shorter legend.
  - **A larger model:** the first facts-only wording held 5 of the 7 `O`
    claims on `qwen2.5:14b-instruct` (CPU, one run) where it held 2 on the
    3B. No version has been tuned on it yet; the procedure is above.
- `room_priors.py` — **deleted on 2026-11-15.** It held hand-written tables
  that told the search where things are: the room types a target category
  is likely and rare in (`TARGET_HOMES`, the `WHERE IT LIVES` prompt line),
  the room types a search need not enter for it (`IMPLAUSIBLE_ROOMS`), the
  "home objects" that kept a room a node (`HOME_OBJECTS`), the "signature"
  objects that made a one-object label strong and the "generic" ones that
  kept a room unlabelled. None of it was measured by the method or asked
  of the model, so none of it belongs in a method whose claim is that the
  LLM supplies the semantics. Do not reintroduce a table under another
  name; if the model lacks a prior, that is a prompt or a model question.
- `revisable_room_labels.py` — `RevisableRoomLabels`, evidence-gated labels
  around `RoomTypeClassifier`: one confirmed landmark is enough to label, the
  label carries a strength (`weak` from one class of evidence or a hesitant
  model, `strong` from two or more classes and a confident model -- a count
  of the evidence, never a judgement of which objects they are), and a new
  kind of object revises it.
  Since 2026-11-09 a verdict that differs from the current label without a
  new kind of evidence -- a count that crossed a bucket, a periodic
  refresh -- is held (`held` counter, a `held: True` history entry) until
  the classifier repeats it; the MP3D ground-truth run had rooms swing
  `bedroom` / `living_room` / `dining_room` three to eight times an episode
  on the same objects, and every swing re-asked the oracle
  (`RoomLabelSettings.relabel_hysteresis`, on by default).
- `target_matcher.py` — target-name matching: exact → cache → LLM →
  token-overlap fallback. The fallback rung is not implemented here: it
  delegates to `core/common/label_match.py`, which is the same rule the
  visual-servo acquisition gate acquires on.

## Tests

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core/mapping/topology/tests
```
