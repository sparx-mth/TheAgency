# FALCON source patches (falcon_sjtu)

The FALCON planner is upstream C++, cloned and compiled into the
`falcon-ros-custom:v1` image. Anything we change *inside FALCON itself* must be
reproducible from a clean image build — otherwise a rebuild or a fresh container
silently drops it. Each such change is a patch here, applied by the image build.

## Where the build actually applies them

The image is built by **`sjtu_project/falcon_docker/Dockerfile`** (an external
repo), which `git clone`s FALCON (`ros1-noetic`) and then runs a series of
self-verifying `fix_falcon_*.sh` scripts before `catkin_make`. Our patch is wired
in there the same way:

- `sjtu_project/falcon_docker/falcon_deadend_guard.patch` — the patch (authoritative copy).
- `sjtu_project/falcon_docker/fix_falcon_deadend_guard.sh` — `git apply`s it and self-verifies.
- `Dockerfile` step "6a-quinquies" — `COPY`s both in and runs the script.

The copy in this directory is a **mirror for visibility from TheAgency**; the
build reads the one in `falcon_docker/`. Keep them in sync if you change it.

## falcon_deadend_guard.patch — escape unreachable dead ends

Touches three upstream files (and none the other `fix_falcon_*.sh` scripts do):

- `exploration_manager/src/exploration_manager.cpp`
- `exploration_preprocessing/include/exploration_preprocessing/frontier_finder.h`
- `exploration_preprocessing/src/frontier_finder.cpp`

**Why.** The depth camera is blind inside ~0.95 m (its near clip), so an obstacle
the aircraft is about to hit never enters the map. FALCON's A* then routes a
"clear" path straight through it and re-selects the same blocked viewpoint every
cycle — FALCON only marks a frontier *dormant* when it has no viewpoint at all,
never when the viewpoint is merely unreachable. The result is a permanent stall
(coverage frozen, `next_pos ... same as current`, or the aircraft wandering a
1–2 m pocket) that previously only a crash broke, by wiping the map.

**What it adds.**
- `FrontierFinder::addBlockedRegion(pos)` + a `blocked_regions_` list;
  `computeFrontiersToVisit()` retires to dormant any cluster whose average is
  within `/frontier_finder/blocked_region_radius` (default 2.5 m; set in our
  `adapter/launch/exploration.launch`) of a blocked point.
- A no-progress guard in `planExploreMotionHGrid()`: if the aircraft's whole
  excursion stays under 2 m for 25 s while still not at the chosen viewpoint,
  that viewpoint is handed to `addBlockedRegion()` and the coverage tour moves on.

## Rebuilding vs. the running image

The running `falcon-ros-custom:v1` also has this compiled in (it was
`docker commit`ed live), so the current `run_falcon_sjtu.sh` workflow needs no
rebuild. The patch mechanism above is what makes it survive an image **rebuild**
(`docker build sjtu_project/falcon_docker`) or a fresh FALCON clone.

## Regenerating the patch

If you change FALCON's C++ again in the running container:

```bash
docker exec falcon-sjtu bash -lc \
  'cd /catkin_ws/src/FALCON && git --no-pager diff -- <changed files>' \
  > sparx_agency/tasks/planning/falcon_sjtu/patches/falcon_deadend_guard.patch
cp sparx_agency/tasks/planning/falcon_sjtu/patches/falcon_deadend_guard.patch \
   ~/GIT/sjtu_project/falcon_docker/falcon_deadend_guard.patch
```

The fix script self-verifies with a **single-line** sentinel (`confined to <2 m`)
because the guard's full log line is split across two C string literals — grep a
whole-phrase sentinel and it will spuriously miss.

## Not FALCON code (already reproducible, no patch needed)

The flight tuning lives in our adapter, mounted at runtime from this repo, so it
is already reproducible: `adapter/launch/exploration.launch` (inflation 0.25,
frontier clearances, `blocked_region_radius`), `adapter/launch/bspline_follower.launch`
(speed cap 0.6, recovery params), and `adapter/scripts/bspline_follower_node.py`
(re-survey, hold-on-silence, 22° attitude reflex).

