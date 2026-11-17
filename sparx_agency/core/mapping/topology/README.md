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
  exits-as-nodes mode, the openings). The model is asked, for each room,
  for **the probability that the target is in the part of it the robot has
  not seen yet** -- a detector checks every frame, so that is where a
  target not yet found is (until 2026-11-17 a node was the room together
  with the unseen space its exits open onto).
  **The division of labour:** the model gives that probability and nothing
  else; distance, and the weighing of distance against the probabilities,
  is RPT*'s. The prompt carries no distance, no position, no neighbours and
  no word on where the robot stands (`here=yes` was dropped on 2026-11-16;
  `SearchNode.here` is kept for the record and never rendered).
  **The model's own knowledge, and none of ours (since 2026-11-16).** The
  system prompt opens with whose judgement is wanted -- "You reason like a
  person who knows how homes are laid out." -- and tells the model to use
  its own reasoning and its own knowledge of standard homes and buildings,
  in which rooms furniture and objects of each kind are found, and to
  derive every probability from that. It holds no category knowledge: no
  room named as the place of any object, no number or band a probability
  should fall in, no list of a benchmark's goal categories, no count of the
  rooms a home has, no worked example. The only object it ever sees named
  is the one on the `TARGET:` line of the user prompt -- the name of the
  object the current episode searches for, in every call. Before
  2026-11-16 the prompt was 14,751 characters of our judgement.
  **What the data means, in words (since 2026-11-17).** Beside what each
  field is, the prompt says what the search so far means for a room,
  because a 3B does not draw it from the fields (below): floor mapped and
  actions spent in a room without the target count against that room, and
  what can still hold the target is its unmapped floor and what lies behind
  its frontiers; a room never entered has not been searched, and is the
  likelier the more the other rooms were searched in vain; a room's number
  is about the target being IN the room -- walking through a searched room
  to another is the planner's. No number, threshold or example goes with it.
  **What a room line holds: four things (since 2026-11-17).** `type=` (or
  `unknown`, and then -- only then -- the objects seen in it: `type=unknown
  (seen: chair x4, table)`; `kitchen?` when the type is tentative);
  `mapped=6 of 12 m2 (50%)` -- the floor the camera has seen from detection
  range out of the room's floor, and their share, where the floor is the
  room's FREE cells (`RoomExtent.free_m2` / `observed_m2`): furniture and
  pockets the robot cannot stand on are not the room's floor; `frontier=`,
  the frontier clusters left in it; and `searched=40 actions`, the effort
  spent inside in the unit of the HOUSE line's budget, `searched=never` for
  a room not entered. Everything else the line held until that day -- the
  size and the sides not mapped, the exits and what was glimpsed through
  them, the status word, the rooms a door joins it to, the objects of a
  classified room, the note `target: detected here, not reached` -- is
  still on the `SearchNode`, for the recording, and is not shown to the
  model: connections and distances are the planner's. The HOUSE line
  (every room type found with its status, the rooms unidentified, the
  exits not looked into, the frontier state, `actions used about N of 500`)
  is unchanged.
  **The reply** is unchanged: `home`, `house`, `stage` (three short readings
  in the model's words, recorded), a `pass` verdict (`first` / `second`,
  recorded; the loop's second pass is opened by the geometry) and one
  integer percentage per node, independent, never renormalised.
  **Between the reply and RPT\* (`SearchNodeOracle.score`, since
  2026-11-17 in this shape).** The model need not write exactly what the
  planner takes; the code makes it so, from the reply and the nodes alone
  and with no knowledge of any room or object. *Format:* the number under
  `p` or a name a model drifts to, as a number or as text, a percent sign
  after it; invented ids dropped; a reply with no usable node refused (one
  schema repair, then the back-off -- never a made-up distribution).
  *Scale:* percentages, also when the whole reply was written in fractions
  of one (`written_as_fractions`), clamped into 0-100 and below 1 as
  probabilities. *Meaning:* a node left out is forgotten, not ruled out
  (1 %); **a room never entered is never zero** (`keep_unentered`: nobody
  has looked, so "certain it is not there" is read as not given -- 1 % --
  and the loop's floor for a never-entered room, 0.15, then holds for it;
  the ids raised are recorded in `NodeOracleResult.floored`), while the
  zero of a room the robot has been inside stands and keeps that room out
  of the order (`LoopSettings.min_prob`). The reply is reused when the
  prompt is byte-identical. One older floor remains as a knob and is **off
  by default** (`unexplored_floor`). After this the loop weighs the numbers
  by exploration state (`RoomSearchLoop._scaled_values`); since the prompt
  now tells the model that effort spent in a room counts against it, a
  searched room can be lowered twice -- by the model and by that discount.
  Neither was changed for it; the measurement is under `search_node_prompts.py`
  below (the paired check of 2026-11-17).
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
  removed with every other rule; on 2026-11-17 the line was cut to four
  things, the prompt was told what the search so far means, and the reply's
  post-processing took its present shape.
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
  the digest shows when that rule is broken. **It was broken once, on
  instruction:** on 2026-11-17 `facts-qwen2.5-3b-v1` was edited in place,
  with the room line it explains. A record carrying the digest
  `3ee6c00c94dc1753...` was asked under the text of 2026-11-16 (3,471
  characters, a line of ten fields); `2d025f019f981fe5...` is the text of
  2026-11-17 (3,091 characters, a line of four). `tests/
  test_search_node_prompts.py` holds EVERY registered version to the
  contract above (it opens with whose judgement is wanted; own knowledge;
  no object word, no room-type word, no number but the reply's; what the
  search so far means said in words and without a number, and no other
  rule; no distance; the reply format; a legend for exactly the fields a
  line carries), so a version added for another model cannot slip past it. **To add one:** copy the nearest version under
  a new name with the new model's tag, register it, probe it
  (`tasks/planning/objnav_benchmark_runtime/tests/probe_node_oracle.py
  --prompt <name>`, with `LLM_REASONING_MODEL` set to that model), reword
  and repeat, then record its figures in the table below.

  **The probe's figures of 2026-11-16** -- the wordings before the edit of
  2026-11-17; that day's figures follow them (`qwen2.5:3b-instruct`, CPU,
  temperature 0, seed 0, one call per scenario; each cell is *claims held
  at the model's own numbers / replies usable / claims held as RPT\*
  receives them*):

  | wording | tuned on | system prompt | 12 default-mode scenarios (`P`) | the first 7 (`O`) | a call takes |
  |---|---|---|---|---|---|
  | the prompt before 2026-11-16 (rules and priors; not a version) | -- | 14,751 chars | 8 / 12 / 10 | 5 / 7 / 6 | 91-130 s |
  | the first facts-only wording of 2026-11-16 (not registered) | -- | 2,955 chars | 3 / 11 / 6 | 2 / 6 / 5 | 54-63 s |
  | `facts-qwen2.5-3b-v1` as it stood on 2026-11-16 (digest `3ee6c00c...`) | `qwen2.5:3b-instruct` | 3,471 chars | 5 / 12 / 10 | 4 / 7 / 6 | 50-56 s |

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

  **The edit of 2026-11-17** (the room line cut to four things; what the
  search so far means said in words; the reply checked for format, scale
  and meaning). All figures: `qwen2.5:3b-instruct` on the GPU, temperature
  0, seed 17.

  *The nineteen scenarios* (`probe_node_oracle --set PO`, a warm server,
  one call each, the oracle's numbers as it hands them on):

  | text | claims held | replies usable | a call takes |
  |---|---|---|---|
  | 2026-11-16 (digest `3ee6c00c...`, 3,471 chars, ten-field line) | 6 of 19 | 13 of 19 | 1.9 s |
  | 2026-11-17 (digest `2d025f01...`, 3,091 chars, four-field line) | 13 of 19 | 19 of 19 | 2.8 s |

  The six replies that were unusable under the first text held the three
  readings and no `nodes` (a run repairs such a reply with a second call).
  Under the second text six claims do not hold, and two of them cannot:
  `P5` rests on what was glimpsed through a room's exit, which the line no
  longer shows, and `O4` asks that a room of the right type rank first when
  every room is searched whole, which is no longer what the number means
  (all three read 0: nothing is left unseen). The other four are the
  model's: two ties (`P8`, both rooms at the never-entered floor; `O5`,
  both at 100 %), a hallway with three frontiers put at 0 (`P12`), and a
  room mapped whole put at 100 % for its type (`P11`). **These scenarios cannot rank
  two wordings**: a draft and three variants of it that each changed one
  statement held 6, 8, 7 and 4 of the nineteen claims (cold starts), and
  the final text held 13 on a warm server and 9 when every call was a cold
  start (`LLM_EXCLUSIVE`, which is how a paired run asks).

  *A hundred recorded queries* -- what the wording was chosen on. The node
  lists of 100 oracle calls of a recorded HM3D run (269 room lines; the
  rooms, areas, frontiers and effort as the scene graph had them) were
  asked again under each candidate, every call a cold start so that a
  reply depends on its request alone. No claim is checked -- there is no
  ground truth for where the target was -- only what the replies are:

  | wording | usable at once | nodes left out | never-entered rooms at 0 | entered rooms at 0 | mean number: never entered / entered | rank correlation with actions inside / share mapped | every node 0 |
  |---|---|---|---|---|---|---|---|
  | the text of 2026-11-16, ten-field line | 91 | 2 % | 22 % | 28 % | 35 / 41 | -0.21 / -0.22 | 7 % |
  | A: the first draft ("the search goes on, so it has not been found yet"; closing question "is the target there?") | 99 | 14 % | 41 % | 35 % | 31 / 31 | -0.14 / -0.09 | 21 % |
  | A with the meaning said at each field instead of in a block | 100 | 13 % | 38 % | 34 % | 27 / 35 | -0.08 / -0.04 | 14 % |
  | B: A with the closing question naming every node by count | 100 | 8 % | 37 % | 33 % | 35 / 37 | -0.08 / -0.06 | 16 % |
  | C: B said in the affirmative ("the target is in a place the robot has not seen yet") | 99 | 7 % | 32 % | 30 % | 41 / 38 | -0.22 / -0.28 | 13 % |
  | **D: C with the number named, "the part of that room not seen yet" -- the text of 2026-11-17** | 98 | 6 % | 25 % | 34 % | 54 / 39 | -0.24 / -0.28 | 8 % |
  | D plus "unknown may also be the kind where this target is found" | 100 | 7 % | 30 % | 39 % | 51 / 41 | -0.15 / -0.29 | 8 % |
  | D with the never-entered sentence reworded (two variants) | 100 | 3-7 % | 24-32 % | 30-37 % | 48-53 / 40-43 | -0.11 to -0.35 / -0.23 to -0.26 | 8-16 % |

  What it shows:
  - **The text of 2026-11-16 scored the rooms the agent had been inside
    above the rooms it had never entered** (41 % against 35 %); the text of
    2026-11-17 the other way round (39 % against 54 %), and its numbers on
    entered rooms fall with the effort spent (40 actions or more inside:
    19 % on average, against 37 %).
  - **Say it in the affirmative.** Under A -- "the search goes on, so it
    has not been found yet" -- 45 of the 103 zeros the 3B wrote (44 %) gave
    the target's not having been found, or the lack of evidence, as their
    reason ("bedroom not checked, no toilet found there yet"), against 18 %
    under the text of 2026-11-16, and a fifth of the replies put every room
    at 0. With "so the target is in a place the robot has not seen yet" (C)
    it was 17 of 93 (18 %), and the numbers follow the effort spent again.
    The number of zeros itself did not fall with this sentence (92 under B,
    93 under C).
  - **Ask about every node, by count.** A reply that leaves nodes out
    scores exactly one node, the last one listed -- every such reply, under
    every wording (56 of 56 over the nine texts of the table). A closing
    question in the singular got 11 of them in a hundred (A); "for each of
    the N nodes above ... score all N", 5 (B).
  - **Name what the number is about.** "In the part of that room not seen
    yet" took the zeros on never-entered rooms from a third of their lines
    to a quarter, raised their mean from 41 % to 54 %, and took the replies
    with every node at 0 from 13 % to 8 % (D).
  - **How far to trust these.** Each is one comparison of two texts on the
    same hundred queries. A reply is repeatable (a cold start for every
    call), but what one sentence does to the others is not separable: the
    two rewordings of a single sentence in the last row moved the zeros on
    never-entered rooms by eight points and the effort correlation from
    -0.11 to -0.35. Differences of that size between two candidates are
    not findings.
  - **A quarter of the never-entered rooms are still put at 0 by the model**
    (mostly an unclassified room, "no evidence"). The code's rule is what
    holds there: such a zero is read as not given (`keep_unentered`) and
    the loop's floor for a never-entered room then applies.
  - **The 3B's use of effort is weak and uneven.** On the recorded queries
    a room mapped whole and searched for thirty actions or more averaged
    14 % (five of seven at 0); on the hand-written scenarios, asked cold,
    the same text put all eleven such rooms at 50 % or more. The loop's
    discount of an entered room is the part that does not depend on it.

  *The paired check* (2026-11-17). The tree of 2026-11-16 against this one
  -- prompt, room line and reply post-processing together, and nothing
  else -- on the same 90 single-storey HM3D v1 val episodes (19 scenes,
  sampled once with seed 17): ground-truth perception,
  `qwen2.5:3b-instruct` on both routes on the GPU, each arm from a frozen
  copy of its tree, every LLM call a cold start (`LLM_EXCLUSIVE`) so that
  the arms differ by the change alone -- the four pilot episodes of each
  arm came out identical when the arm was run again. The MP3D half (90
  episodes of the single-storey set) was still running when this was
  written.

  | HM3D, 90 pairs | 2026-11-16 | 2026-11-17 | difference (95 % interval) |
  |---|---|---|---|
  | SR | 86.7 % (78) | 87.8 % (79) | +1.1 points (-4.4, +6.7); 4 gained, 3 lost |
  | SPL | 0.431 | 0.460 | +0.029 (-0.015, +0.072) |
  | an oracle call, without the model load | 3.11 s (median 2.72) | 3.13 s (median 2.82) | prompt 1,331 -> 973 tokens, reply 135 -> 142 |
  | oracle repair calls an episode | 0.24 | 0.04 | -0.20 (-0.31, -0.09) |
  | room choices that are a room never entered | 70 % (227 of 322) | 77 % (273 of 354) | +0.51 an episode (+0.12, +0.93) |
  | returns to a room already searched (20+ actions inside), an episode | 2.81 | 2.80 | -0.01 (-0.63, +0.64) |
  | never-entered rooms offered to RPT\* | 66 % | 99 % | |

  - **SR and SPL did not move significantly**; 34 of the 90 episodes ran
    identically in both arms. A call takes as long as before: the prompt
    is a quarter shorter, and writing the reply is what costs.
  - **What changed is which rooms RPT\* is offered, and the code did
    most of it.** The 3B put a never-entered room at 0 in 36 % of its lines
    under BOTH texts (327 of 906; 273 of 748). The old code then left such
    a room out of the order; `keep_unentered` keeps it (273 raised), so
    the agent chooses never-entered rooms more often. It does not walk
    back into searched rooms any less.
  - **The check measures the three changes together.** Two arms cannot
    say how much is the wording and how much the rule.

  **The two discounts on a searched room, measured (same check, the arm of
  2026-11-17).** For a room entered and left, the model's own number fell
  from 48 % (under ten actions inside) to 36 % (forty or more); the loop's
  factor over the same range goes from 0.44 to 0.08. The model's share is
  small and the loop's decides -- including for the room the prompt leaves
  to the model, a large one searched for long with much left: of the 19
  lines with three or more frontiers, twenty or more actions inside and
  50 % or more from the model, RPT\* received 0.10 on average and 6 were
  put under the 0.05 threshold. Laid over the dataset's goal positions
  (analysis only -- the agent never sees them;
  `map_journal.npz` holds the partition of every step): of the 132
  entered-and-left lines the model put at 50 % or more, 12 were the room
  that held the target (9 %, where 7 % of all entered-and-left lines
  are): the 3B does not know which searched room deserves another look.
  The loop put 7 of the 132 under the threshold, none of them the target's
  room (of the 6 above, none; 7 of the 19 were the target's room and all
  7 stayed in the order), and in none of the 11 failed episodes had it
  hidden the target's room against the model -- at their last query 5
  did not have the target's room on the map yet, in 3 it was a
  never-entered room the agent did not reach in time, and in 3 the scan
  ledger had finished it. Neither discount was changed.
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
