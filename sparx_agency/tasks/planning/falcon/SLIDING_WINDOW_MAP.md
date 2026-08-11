# Sliding-window voxel map — scanning an area larger than memory

**Status: proposal, nothing implemented.** This is the integration plan for keeping the
dense voxel map only around the aircraft and letting FALCON's own coarse layer carry
everything else, so that peak memory stops growing with the size of the area scanned.

The idea comes from WildOS (arXiv 2602.19308), which never stores dense geometry outside
its depth horizon and keeps a sparse graph for everything beyond it. The finding of this
document is that **FALCON already has the graph** — so the work is not to import one, it is
to stop paying for voxels FALCON has already stopped reading.

---

## Verdict

**Yes, this is possible, and FALCON is unusually well set up for it.** Four of its design
decisions already assume the fine map is only useful nearby:

| what FALCON already does | where |
|---|---|
| Beyond `hybrid_search_radius` (10 m), every cost evaluation uses the connectivity graph, **not voxels** | `hierarchical_grid.cpp:2001,2111,2290,2328` |
| Space decomposition only runs on cells intersecting the map-update box, i.e. within sensor range | `hierarchical_grid.cpp`, Alg. 1, p.7 |
| A frontier is only re-checked if its box overlaps the update box | `frontier_finder.cpp:81,92,99` |
| The executed trajectory is truncated to **7 m** before the B-spline runs, so ESDF is only needed that far | `exploration_manager.cpp:1024-1031` |

So the coarse layer is already authoritative for long-range decisions. Only **two** places
still reach into far voxels, and both are listed below with a fix.

The honest caveat is not algorithmic, it is logistical: this is a change to FALCON's own
C++, which we consume as a patched upstream clone. See [Where the code goes](#where-the-code-goes).

---

## Before building any of this — check you have the problem

**At house scale you almost certainly do not.** The arithmetic below says a very large house
costs 26–200 MB at 0.2 m resolution. If a house-sized mission is running out of memory, the
voxel map is not sized to the house, and the fix is config, not code.

Two independent knobs, in the map YAML, and they are easy to get wrong together:

| knob | controls | trap |
|---|---|---|
| `map_min_*` / `map_max_*` | **the allocation** — `map_size_ = map_max_ - map_min_` (`map_server.cpp:48`), and the array is `map_size_idx_.prod()` | a generous "just in case" extent is paid for in full on tick one, whether or not you ever fly there |
| `box_min_*` / `box_max_*` | **the resolution** — under 4000 m³ picks 0.1 m, over it picks 0.2 m (`map_server.cpp:40-44`) | crossing that threshold changes memory by **8×** |

The failure mode this produces is non-obvious: **shrinking your task box can make memory
eight times larger.** A house with a 3 600 m³ task box drops under the threshold into 0.1 m
resolution; if the map extent was left at a comfortable 100 × 100 × 20 m, that is 200 M
voxels — **8.8 GB** — to map a building occupying two per cent of it.

| bounding box | at 0.2 m | at 0.1 m |
|---|---|---|
| 30 × 20 × 8 m | 26 MB | 212 MB |
| 60 × 40 × 12 m | 159 MB | 1.27 GB |
| 100 × 60 × 15 m | 497 MB | 3.97 GB |

**Also: `resolution_fine` in `voxel_mapping.yaml` is dead.** The node reads
`/voxel_mapping/resolutionf_fine` — note the transposed `f` — at `map_server.cpp:12`, while
the YAML key is `resolution_fine`. The lookup always falls back to its hard-coded 0.1 default.
Today the default happens to equal the YAML value so nothing is visibly wrong, but anyone who
tries to solve a memory problem by editing that line will find it does nothing.
`resolution_coarse` is spelled correctly and does work.

---

## Why the current design cannot scale

The map is a **dense flat array allocated once at startup** — `data.resize(map_size_idx_.prod())`
(`map_base_inl.h:5`), addressed row-major by `indexToAddress` (`map_base_inl.h:53`). There is
no octree, no hashing and no window. Six full-map-sized arrays exist:

| array | type | bytes / voxel |
|---|---|---|
| occupancy `data` | `OccupancyVoxel` (one `enum class`) | 4 |
| TSDF `data` | `TSDFVoxel` (two `double`) | 16 |
| ESDF `data` | `ESDFVoxel` (one `double`) | 8 |
| ESDF `tmp_buffer1_` | scratch, full size | 8 |
| ESDF `tmp_buffer2_` | scratch, full size | 8 |
| `frontier_flag_` | `vector<bool>` | 0.125 |
| | | **≈ 44 B/voxel** |

`FloatingPoint` is `double` (`exploration_types.h:23`), which is where most of this goes.
Resolution is 0.2 m for any box over 4000 m³ (`voxel_mapping.yaml:3-4`), i.e. 125 voxels/m³,
i.e. **5.5 kB per cubic metre of bounding box** — occupied, free or never visited alike,
because the array is allocated over the whole box on the first tick.

| bounding box | voxels | RAM |
|---|---|---|
| Power Plant (the paper's largest sim map, 16×29×19 m) | 1.1 M | 49 MB |
| 100 × 100 × 20 m | 25 M | **1.1 GB** |
| 200 × 200 × 20 m | 100 M | **4.4 GB** |
| 500 × 500 × 30 m | 938 M | **41 GB** |

On a Jetson Orin NX 16 GB — shared between CPU and GPU — the wall arrives somewhere around
a 200 m square, before ROS, the depth model, the bridge and our own nodes have taken their
share. Note the failure is at **startup allocation**, not gradual: the map does not grow as
you fly, it is already as large as it will ever be on tick one. So the symptom is a launch
that dies or thrashes, not a slow leak.

A window of 40 × 40 × 20 m costs **177 MB**, and 60 × 60 × 20 m costs **397 MB** — constant,
however far the aircraft flies. That is the whole prize: **O(area) becomes O(1)**.

---

## What FALCON keeps in its coarse layer already

`GridCell` (`hierarchical_grid.h:25-70`) is already the compact node you described. Per 5 m
cell it holds, with no voxels involved:

- `STATE { ACTIVE, EXPLORED, UNKNWON }` — a cell already knows it is finished
- `center_`, `center_free_`, `centers_free_`, `centers_free_active_`, `centers_unknown_`,
  `centers_unknown_active_` — the zone centres, which are the graph's vertices
- `unknown_num_`, `free_num_`, `frontier_num_` — the counts the coverage tour is built from
- `frontier_ids_`, `frontier_viewpoints_`, `frontier_yaws_` (and their per-centre `_mc_`
  variants) — **the frontier viewpoints and their yaws, already stored away from the voxels**
- `connectivity_matrixs_` — which centres inside the cell reach which
- `nearby_cells_ids_`, `bbox_min_/max_`

And `ConnectivityGraph` holds one node per zone with edges carrying a cost and a type
(FREE / UNKNOWN / PORTAL). `searchConnectivityGraphBFS` returns **node ids**, not positions.

This is exactly the WildOS navigation graph in all but name: nodes for reachable regions,
edges for traversability, frontier nodes carrying what the planner needs. It just happens to
be kept *alongside* a dense map instead of *instead of* one.

**But it is not free, and it does not scale forever.** `uniform_grid_.resize(...)`
(`hierarchical_grid.cpp:15`) allocates every cell of the whole bounding box up front, exactly
like the voxel map — 5 m cells instead of 0.2 m voxels, so 15 625× cheaper per unit volume,
but still **O(area) and still pre-allocated**. Several per-iteration passes also sweep all
cells rather than the active ones (`hierarchical_grid.cpp:327,385,497`). At 500 × 500 × 30 m
that is 60 000 cells: the memory is negligible, the per-iteration sweep cost is unmeasured.
See [The second wall](#the-second-wall).

**One free win hiding here.** `ConnectivityEdge` stores `std::vector<Position> path_` — the
full dense A\* path for every edge. The only accessor for it, `getFullConnectivityGraphPath`,
**has zero callers**; visualisation uses the endpoints-only `getFullConnectivityGraph`
(`hierarchical_grid.cpp:2967`, inside marker-building code). At 0.2 m resolution a 5–10 m
edge is 25–50 positions × 24 B, so across a large map this is the coarse layer's dominant
cost and it is dead weight. Delete it and the graph becomes genuinely cheap. *(The 3.6 GB
figure you would otherwise reach at 500 m scale is an estimate, not measured.)*

---

## Everything that still touches fine voxels

| # | site | distance | verdict |
|---|---|---|---|
| 1 | CCL space decomposition | update box only (≤ sensor range) | safe |
| 2 | Connectivity-graph `restrictedA*` | one or two 5 m cells around the update box | safe |
| 3 | Frontier search / `isFrontierChanged` | gated on overlap with the update box | safe |
| 4 | `sampleViewpoints` | only for newly-found frontiers (`computeFrontiersToVisit`, `frontier_finder.cpp:566`) | safe |
| 5 | Cost matrices `C_cp`, `C_sop` | voxel A\* under 10 m, graph BFS beyond | safe |
| 6 | B-spline / ESDF | path truncated to 7 m first | safe |
| 7 | `frontier_flag_` | full-map `vector<bool>`, addressed by the map's own `toadr` | **follows the window automatically once (A) lands** |
| 8 | **`planTrajToView` A\* to the next viewpoint** | **unbounded** | **must be fixed — see (C1)** |
| 9 | `PathCostEvaluator::searchPath` straight-line raycast | runs before the distance test at some call sites | **must be clamped — see (C2)** |

Only rows 8 and 9 are real work. That is what makes this proposal cheap.

Worth noting that row 8 is also a **paper-versus-code gap**: §V-C (p.10) says "Voxel-based
A\* search is employed for nearby targets, whereas graph A\* search is applied on the free
subgraph G_f for distant targets", but `planTrajToView` (`exploration_manager.cpp:1005`) is a
plain voxel A\* regardless of distance. Fixing it makes the code match its own paper, and it
gets faster, independent of any memory concern.

---

## The design

### A. Ring-buffer the fine map

Keep one flat array, sized to the window, and change only how an index becomes an address:

```
addr = mod(idx - window_origin, window_size)      # toroidal
```

This is the standard circular local map (Voxblox, ewok, the Fast-Planner line all use it).
Two properties matter here:

- **Sliding costs no data movement.** Advancing the origin by one slab invalidates exactly
  that slab; everything else keeps its address.
- **Every consumer already goes through `positionToAddress` / `indexToAddress`**
  (`map_base_inl.h:42-60`). Changing those two functions plus adding an `isInWindow()` guard
  reaches the whole codebase without editing the planner. This is what makes "touch the
  planner as little as possible" achievable rather than aspirational.

Add an eviction callback fired for each slab as it leaves.

### B. Freeze the coarse layer behind the window

- When a cell leaves the window, mark it `STATE::EXPLORED` and stop re-decomposing it. This
  is already the behaviour — only cells intersecting the update box are rebuilt — so it is a
  guard, not new logic.
- Keep its zone centres, frontier viewpoints and yaws. Already voxel-free.
- Drop `ConnectivityEdge::path_` entirely (dead, see above).

### C. Close the two leaks

**C1 — `planTrajToView`.** Since the result is truncated to 7 m anyway, the long half of that
A\* is thrown away. Replace with: graph A\* on `G_f` for the coarse route → take the first
graph node still inside the window → voxel A\* to that node only. Same output, bounded input,
matches the paper.

**C2 — `searchPath`'s straight-line raycast.** Audit its call sites, clamp the ray to the
window, and fall back to the graph cost when the target is outside.

### D. Offload, do not discard

For an area scan **the map is the deliverable**, so eviction must mean *write out*, not
*throw away*. Stream each evicted slab as a compressed occupancy block (occupancy alone is
4 of the 44 bytes and compresses hard — RLE or an octree on write). Over the bridge to the
host, or to disk.

### E. A coarse global layer, for re-entry

The one behaviour that genuinely changes: if the aircraft returns to a region it evicted,
the fine map there is empty. FALCON's own long-tail behaviour — going back at the end to
finish corners — makes this certain, not hypothetical.

Answer: keep a **global occupancy-only grid at 1 m**, alongside the window. At 500 × 500 × 30 m
that is 7.5 M voxels × 4 B = **30 MB**, i.e. free. It cannot support ESDF or a B-spline, and
it does not need to: nothing outside 7 m ever asks for those. What it does is stop the
aircraft flying into a wall it already mapped while the fine window refills from the sensor.

This is the one piece genuinely worth copying from WildOS in structure as well as spirit —
its planner keeps exactly this, an `UnexploredSpaceMap` at 1 m resolution marking explored
cells, and computes goal distances through the unexplored part of it
(`graphnav_planner/src/planner.cpp:63-78`).

---

## The second wall

The plan above moves the memory wall; it does not remove it. Phases 1–4 make the **fine**
layer O(1), but the coarse layer stays O(area) and pre-allocated, so a large enough area
eventually hits the same problem one or two orders of magnitude further out. Roughly: at 5 m
cells the coarse layer costs about what the voxel map costs at 15 625× the area, so a box
that would have needed 4 GB of voxels needs a few hundred kB of cells. That is a long way
off — but it is the same shape of bug, and it is worth knowing it is there rather than
discovering it later.

**This is where WildOS's design genuinely beats FALCON's**, and it is the one place where
adopting its structure rather than its discipline would pay. WildOS does not use a fixed grid
at all: it samples candidate nodes and rejects any that fall inside an existing node's free
radius (`Algorithm 3`, p.11), which places nodes densely in clutter and sparsely in open
ground. Its graph is O(explored), not O(bounding box), and there is nothing to pre-allocate.

Two WildOS per-node quantities would come with that and have no FALCON equivalent:

- **Free radius** — distance to the nearest obstacle or unknown cell, capped. This is a
  *sparsely sampled ESDF*: the compact way to keep clearance information for regions whose
  voxels are gone. FALCON's zone centres carry no clearance at all.
- **Explored radius** — how far around a node has actually been observed. This is a much
  better eviction criterion than FALCON's per-cell `EXPLORED` flag, because it is continuous
  and does not quantise to 5 m.

Not recommended for the first pass: the coverage tour and the ordering problem are both
defined over cells and zones, so replacing the fixed grid means touching the planner
properly — the opposite of this document's goal. Revisit if the coarse layer measures badly.

## What we are and are not taking from WildOS

**Taking, and load-bearing:** the coarse explored/unexplored grid at ~1 m as the re-entry
safety net (§E). Without it the aircraft returns to an evicted region blind. FALCON has no
equivalent; WildOS's planner keeps exactly this.

**Taking, as discipline:** the rule that dense geometry is never stored outside the depth
horizon, and the framing of that horizon as a named, first-class system parameter rather than
something implied by four unrelated constants. FALCON has every mechanism needed to obey that
rule and does not obey it — nothing in its design says where the boundary is. That framing is
what turns "FALCON has a graph" into "FALCON can drop the voxels", and it is the actual
contribution of WildOS to this piece of work.

**Held in reserve:** adaptive node sampling, free radius, explored radius — see
[The second wall](#the-second-wall).

**Not taking:** the navigation graph as a data structure. FALCON's connectivity graph is the
same idea, better integrated with its planner, and already load-bearing; a second one would
be duplicate state to keep consistent. ExploRFM, the visual frontiers, the scoring function
and the triangulation are all irrelevant here — they belong to a different combination.

---

## Phasing

| phase | change | effort | risk | win |
|---|---|---|---|---|
| 0 | Measure: instrument RSS against box size, confirm 44 B/voxel empirically | hours | none | the numbers above are read from source, not measured |
| 1a | Shrink the two ESDF scratch buffers to the local update box | ~20 lines | very low | **−36 % memory, no behaviour change** |
| 1b | Delete the dead `ConnectivityEdge::path_` | ~10 lines | very low | removes the coarse layer's dominant cost |
| 2 | Ring-buffer `positionToAddress` / `indexToAddress` + `isInWindow()` | moderate | medium | the main prize |
| 3 | Fix `planTrajToView` to hybrid (C1), clamp `searchPath` (C2) | small | medium | closes the leaks; also a speed-up |
| 4 | Eviction hook + slab offload + 1 m global occupancy | moderate | medium | makes the map a deliverable again |

**Phase 1a alone cuts a third of the memory for a day's work and changes no behaviour.** Do
that first regardless of whether the rest proceeds — it needs no decision from anyone.

Window sizing constraint, from the numbers above: the radius must exceed
`hybrid_search_radius` (10 m), `radius_far` (7 m) and sensor range (5 m). So ~15 m is the
floor; a 40 m box (177 MB) is tight and a 60 m box (397 MB) is comfortable.

---

## Risks, in the order they will bite

1. **Small maps silently exercise the wrong path.** `hierarchical_grid.cpp:2472` sets
   `hybrid_search_radius_ = infinity` when the box is under 1000 m³ — every search becomes
   voxel A\*. Any test on a small map will therefore not test the windowed behaviour at all.
   Tests must force a large box or override the parameter.
2. **Re-entry.** The long tail is FALCON's known behaviour and it is exactly the case the
   window is worst at. Phase 4's global layer is not optional cover for it.
3. **`frontier_flag_` sizing.** It is a full-map `vector<bool>` indexed through the same
   address function, so it should follow the window for free — but it is allocated separately
   in `frontier_finder.cpp:12` and must be re-sized to match, or the two will disagree
   silently.
4. **Zone ids are `cell_id * 10 + center_id`** with a hard `CHECK` at ten centres per cell
   (`connectivity_graph.h:86-88`). Not affected by windowing, but it is a latent abort in any
   scene that puts more than ten zones in one 5 m cell, and a large outdoor area is more
   likely to than an office.
5. **Coverage/completeness reporting.** FALCON reports coverage from the voxel map. Once
   voxels are evicted, that statistic has to come from the offloaded slabs or the coarse
   layer instead, or it will silently under-report.

---

## Where the code goes

This is the awkward part and it should be decided before any work starts.

Everything in A–C is **C++ inside upstream FALCON** — `voxel_mapping/`,
`exploration_preprocessing/`, `exploration_manager/`. We consume FALCON as a patched clone
built in Docker (`Dockerfile` clones `HKUST-Aerial-Robotics/FALCON`, then `patches/` applies
fixes). Today those patches are small `sed` scripts for one-line fixes. A ring-buffered map
is not that: it is a fork-sized change and should be a proper `git format-patch` series
against the upstream tag, or a maintained fork we clone instead.

It affects **both** deployments, and differently:

- `tasks/planning/falcon_pegasus/` runs FALCON complete, so it gets the full benefit and is
  where this must be validated.
- `tasks/planning/falcon/` uses FALCON only as a voxel mapper — our own chain plans off the
  BEV — so it gets the memory saving but none of the planner-side concerns. Our BEV grid
  has its own separate scaling question, not addressed here.

Anything ROS-free and reusable — a slab serialiser, the 1 m global occupancy — could sit in
`core/mapping/costmap/` and be shared, subject to the usual Noetic constraint if it ends up
on a FALCON import path (Python 3.8, numpy 1.17 API, no scipy).

---

## How to validate

- `./stub/check.sh <run>` in `falcon_pegasus/` flies the whole mission against the real
  FALCON stack with no Isaac Sim and no GPU in about four minutes. This is the regression
  loop; run it after every phase.
- `pytest sparx_agency/tasks/planning/falcon` (1390 passed, 2 skipped) for our own side.
- A soak on a deliberately oversized bounding box, watching RSS: the point of the change is
  that the curve goes flat, and that is the acceptance test.
- Compare exploration time and coverage before and after on the same run. The window must
  not change the route; if it does, something in the table of nine sites above was wrong.
