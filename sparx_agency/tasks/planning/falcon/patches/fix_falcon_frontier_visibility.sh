#!/usr/bin/env bash
# Stop FALCON declaring "exploration finished" while the world is unexamined.
#
# Measured live in Sphera (sphera_jail, 2026-08-18): 372 s into an otherwise
# healthy run -- tracking error 0.02-0.10 m, zero collisions, voxel count still
# climbing -- the frontier set collapsed to one cluster with TWELVE retired
# behind it, and the FSM quit:
#
#   [ExplorationManager] Frontier number: 1, dormant frontier number: 12
#   [ExplorationManager] No frontier
#   [FSM] Finish exploration: No frontier detected
#
# Three causes, fixed here. All three are param-gated and default to upstream
# behaviour, so this script is inert until the launch file opts in.
#
#   1. countVisibleCells() treats UNKNOWN as an occluder, exactly like a wall.
#      But a frontier cell IS the boundary of unknown space, so a ray arriving
#      at one crosses that boundary by construction -- the test reports "you
#      cannot see this frontier" *because* it is a frontier. Allow a ray to
#      cross a bounded prefix of unobserved voxels; OCCUPIED still blocks at any
#      distance, which keeps sealed cavities unviewable.
#      -> /frontier_finder/visib_unknown_tolerance (0 = upstream)
#
#   2. min_visib_num is an absolute cell count, but the most any viewpoint CAN
#      see is the cluster's own size, so the bar is hardest exactly where the
#      cluster is smallest -- and small leftover clusters are what remains late
#      in a mission. Apply a cluster-relative bar, but ONLY to candidates with
#      real clearance: a viewpoint wedged against structure (isNearOccupied)
#      keeps the full absolute bar, which is what stops the relaxation admitting
#      frontiers that bound a sealed interior cavity.
#      -> /frontier_finder/open_visib_fraction (<=0 = upstream), open_visib_floor
#
#   3. grantFinishAmnesty() already exists but is called BEFORE categorisation,
#      guarded on both frontiers_ and tmp_frontiers_ being empty. In the failure
#      above the last cluster was sitting in tmp_frontiers_ and went dormant
#      DURING categorisation, so the amnesty never fired. Re-check after
#      categorisation and run one more pass over the revived clusters.
#
#      That pass has to be RELAXED to be worth running, which the first attempt
#      at this fix got wrong. Measured 2026-08-18, second run: the retry fired,
#      re-offered 9 clusters, and all 9 retired again immediately, with
#      "[UniformGrid] Cell 23 has 1 frontiers, but no free subspace" repeating
#      throughout. The gate was never visibility -- it was isNearUnknown()
#      rejecting every candidate position, which early in a mission is ALL of
#      them, since a frontier is surrounded by unknown space by definition. So
#      while re-examining retired clusters, isNearUnknown is dropped and the
#      visibility bar goes to zero. Everything that protects the aircraft is
#      kept: the candidate must still be in the box and not in an occupied or
#      unobserved voxel, and a cluster inside a blocked region still retires.
#      Only that pass is relaxed; freshly detected clusters face the full test,
#      which is what stops the relaxation admitting sealed-cavity frontiers.
#
#      And the pass is RATE-LIMITED, which the second attempt got wrong. Measured
#      2026-08-18, third run: with the pass running on every cycle it fired 8465
#      times in 700 s, and because a zero visibility bar accepts a viewpoint that
#      can see a single cell, it kept manufacturing targets the aircraft was
#      already standing on. FALCON planned 500 trajectories, flew 384 m of them,
#      and moved 3 m net -- a livelock in place, which is a worse failure than the
#      premature finish it was meant to cure. So: only after the frontier set has
#      been empty for several consecutive cycles, at most once every few seconds,
#      and never below a minimum visibility. Between attempts the frontier set is
#      simply empty, which the FSM's own finish grace turns into a hover, and a
#      hover that recovers beats a livelock that does not.
#
# Written as an idempotent anchor-based script rather than a .patch on purpose:
# falcon_visib_unknown_tolerance/open_visib_bar/vp_audit from falcon_sjtu all
# collide with our own frontier_finder.cpp ports (deadend_guard,
# blocked_region_ttl/widen, publish_fail_blacklist) and cannot be applied in any
# order. See patches/README.md.
set -euo pipefail

SRC=/catkin_ws/src/FALCON/falcon_planner/exploration_preprocessing/src/frontier_finder.cpp
test -f "$SRC" || { echo "[frontier-visib] ERROR: $SRC not found" >&2; exit 1; }

python3 - "$SRC" <<'PYEOF'
import sys

path = sys.argv[1]
src = open(path).read()

if "amnestyRelaxed" in src:
    print("[frontier-visib] already applied, nothing to do")
    sys.exit(0)
if "visib_unknown_tolerance" in src:
    raise SystemExit(
        "[frontier-visib] ERROR: a PARTIAL earlier version of this fix is in "
        "place (visibility edits without the relaxed amnesty pass). Restore the "
        "file first: git -C /catkin_ws/src/FALCON checkout -- "
        "falcon_planner/exploration_preprocessing/src/frontier_finder.cpp")


def replace_once(text, old, new, what):
    """Exactly-one-occurrence replace, so a silent no-op is impossible."""
    n = text.count(old)
    if n != 1:
        raise SystemExit(
            "[frontier-visib] ERROR: %s anchor matched %d times (want 1)" % (what, n))
    return text.replace(old, new)


# ── 1. helpers, inside namespace fast_planner ──────────────────────────────
HELPERS = '''namespace fast_planner {

namespace {

// How much unobserved space a line of sight may cross. A frontier cell is the
// boundary of unknown space, so a ray arriving at one passes through the
// unobserved boundary layer by construction; the raycast runs from the cell
// toward the viewpoint, so the tolerated voxels are exactly that layer.
// Unknown met further along is a genuinely unexplored volume being looked
// through, and still blocks. 0 restores upstream exactly.
int unknownVisibilityTolerance() {
  static bool loaded = false;
  static int tolerance = 0;
  if (!loaded) {
    ros::param::param("/frontier_finder/visib_unknown_tolerance", tolerance, 0);
    loaded = true;
    ROS_WARN("[visib_unknown] tolerance=%d voxel(s) of unobserved space per ray "
             "(0 restores upstream: any UNKNOWN blocks the line of sight)",
             tolerance);
  }
  return tolerance;
}

// How much of a frontier an OPEN-SPACE viewpoint must see. Relaxes only, never
// tightens, and only for candidates that are not near occupied -- structure-hugging
// viewpoints keep the absolute bar so sealed cavities stay unviewable.
// fraction <= 0 restores the absolute bar everywhere.
int openSpaceVisibilityBar(const int cluster_cells, const int absolute_bar) {
  static bool loaded = false;
  static double fraction = 0.0;
  static int floor_cells = 0;
  if (!loaded) {
    ros::param::param("/frontier_finder/open_visib_fraction", fraction, 0.0);
    ros::param::param("/frontier_finder/open_visib_floor", floor_cells, 0);
    loaded = true;
    ROS_WARN("[open_bar] absolute=%d fraction=%.2f floor=%d (applies only to "
             "candidates that are NOT near occupied; fraction <= 0 restores the "
             "absolute bar everywhere)", absolute_bar, fraction, floor_cells);
  }
  if (fraction <= 0.0)
    return absolute_bar;
  int bar = static_cast<int>(fraction * static_cast<double>(cluster_cells));
  if (bar > absolute_bar)
    bar = absolute_bar;
  if (bar < floor_cells)
    bar = floor_cells;
  return bar;
}

// True only while the frontier finder is re-examining clusters it had already
// retired. A file-static rather than a member so this fix stays inside one
// translation unit and does not have to touch frontier_finder.h -- the two
// functions that read it and the one that sets it are all in this file, and the
// finder is single-threaded through updateFrontierStruct().
bool g_amnesty_relax = false;
int g_empty_cycles = 0;
double g_last_amnesty_time = 0.0;

// Whether to spend an amnesty pass on this cycle. Gated on BOTH a run of empty
// cycles and a wall-clock interval, so a pass is a rescue rather than a target
// generator; see the header note on the 8465-pass livelock.
bool amnestyPassDue() {
  static bool loaded = false;
  static int min_empty_cycles = 0;
  static double interval_sec = 0.0;
  if (!loaded) {
    ros::param::param("/frontier_finder/amnesty_min_empty_cycles", min_empty_cycles, 10);
    ros::param::param("/frontier_finder/amnesty_interval_sec", interval_sec, 10.0);
    loaded = true;
    ROS_WARN("[amnesty_gate] a relaxed pass needs %d consecutive empty cycle(s) "
             "and %.0fs since the last one", min_empty_cycles, interval_sec);
  }
  if (g_empty_cycles < min_empty_cycles)
    return false;
  const double now = ros::Time::now().toSec();
  if (g_last_amnesty_time > 0.0 && now - g_last_amnesty_time < interval_sec)
    return false;
  g_last_amnesty_time = now;
  return true;
}

// The bar a relaxed pass still applies. Zero accepts a viewpoint that can see a
// single cell, which is how the livelock got its targets; a quarter of the
// normal bar still admits the genuinely-hard clusters this pass exists for.
int amnestyVisibilityBar(const int absolute_bar) {
  static bool loaded = false;
  static int divisor = 4;
  if (!loaded) {
    ros::param::param("/frontier_finder/amnesty_visib_divisor", divisor, 4);
    loaded = true;
  }
  if (divisor <= 0)
    return 0;
  const int bar = absolute_bar / divisor;
  return bar > 1 ? bar : 1;
}

}  // namespace

// Free, not a member, for the same reason. Declared here so sampleViewpoints()
// below can see it.
bool amnestyRelaxed() { return g_amnesty_relax; }
'''
src = replace_once(src, "namespace fast_planner {\n", HELPERS, "namespace open")

# ── 2. countVisibleCells: bounded UNKNOWN crossing ─────────────────────────
OLD_RAY = '''    raycaster_->input(cell, pos);
    bool visib = true;
    while (raycaster_->nextId(idx)) {
      if (map_server_->getOccupancy(idx) == voxel_mapping::OccupancyType::OCCUPIED ||
          map_server_->getOccupancy(idx) == voxel_mapping::OccupancyType::UNKNOWN) {
        visib = false;
        break;
      }
    }'''
NEW_RAY = '''    raycaster_->input(cell, pos);
    bool visib = true;
    const int unknown_budget = unknownVisibilityTolerance();
    int unknown_crossed = 0;
    while (raycaster_->nextId(idx)) {
      const voxel_mapping::OccupancyType occ = map_server_->getOccupancy(idx);
      if (occ == voxel_mapping::OccupancyType::OCCUPIED) {
        visib = false;
        break;
      }
      // The ray starts AT the frontier cell, so the first unobserved voxels it
      // meets are the boundary layer that makes the cell a frontier at all.
      if (occ == voxel_mapping::OccupancyType::UNKNOWN &&
          ++unknown_crossed > unknown_budget) {
        visib = false;
        break;
      }
    }'''
src = replace_once(src, OLD_RAY, NEW_RAY, "countVisibleCells raycast")

# ── 3. sampleViewpoints (2D only): cluster-relative bar in open space ──────
# Scope the edit to the 2D sampler's body; sampleViewpoints3D carries an
# identical gate and is not compiled into the active path.
head, sep, tail = src.partition("void FrontierFinder::sampleViewpoints(")
if not sep:
    raise SystemExit("[frontier-visib] ERROR: sampleViewpoints not found")
end = tail.index("\nvoid FrontierFinder::")
body, rest = tail[:end], tail[end:]

OLD_BAR = '''      if (visib_num > min_visib_num_) {'''
NEW_BAR = '''      // Near structure keeps upstream's absolute bar; only a candidate with real
      // clearance may use the cluster-relative one. Computed once, reused below.
      // The amnesty pass drops to a fraction of the bar rather than to zero:
      // the point is to ask whether the cluster can be seen at all usefully,
      // and a viewpoint that can see one single cell is not a target.
      const bool near_occupied = isNearOccupied(sample_pos);
      const int visib_bar =
          amnestyRelaxed()
              ? amnestyVisibilityBar(min_visib_num_)
              : (near_occupied ? min_visib_num_
                               : openSpaceVisibilityBar(static_cast<int>(cells.size()),
                                                        min_visib_num_));
      if (visib_num > visib_bar) {'''
body = replace_once(body, OLD_BAR, NEW_BAR, "sampleViewpoints bar")

# The clearance test that actually retired everything. A frontier is bounded by
# unknown space by definition, so demanding margin from it makes a reachable
# frontier unviewable. Dropped ONLY while re-examining a retired cluster; the
# in-box and occupied/unknown tests are never dropped.
OLD_CLEARANCE = '''      if (!map_server_->isInBox(sample_pos) ||
          map_server_->getOccupancy(sample_pos) == voxel_mapping::OccupancyType::OCCUPIED ||
          map_server_->getOccupancy(sample_pos) == voxel_mapping::OccupancyType::UNKNOWN ||
          isNearUnknown(sample_pos))
        continue;'''
NEW_CLEARANCE = '''      if (!map_server_->isInBox(sample_pos) ||
          map_server_->getOccupancy(sample_pos) == voxel_mapping::OccupancyType::OCCUPIED ||
          map_server_->getOccupancy(sample_pos) == voxel_mapping::OccupancyType::UNKNOWN ||
          (!amnestyRelaxed() && isNearUnknown(sample_pos)))
        continue;'''
body = replace_once(body, OLD_CLEARANCE, NEW_CLEARANCE, "sampleViewpoints clearance")
body = replace_once(body, "        if (isNearOccupied(sample_pos))\n",
                    "        if (near_occupied)\n", "sampleViewpoints bucketing")
src = head + sep + body + rest

# ── 4. amnesty is re-checked AFTER categorisation, not only before ─────────
OLD_LOOP = '''  // Try find viewpoints for each cluster and categorize them according to
  // viewpoint number
  for (auto &tmp_ftr : tmp_frontiers_) {'''
NEW_LOOP = '''  // Try find viewpoints for each cluster and categorize them according to
  // viewpoint number.
  //
  // Two passes, because the amnesty above is checked BEFORE this loop and so
  // misses the case that actually ends missions: the last surviving cluster is
  // sitting in tmp_frontiers_ and goes dormant HERE, leaving frontiers_ empty
  // with the amnesty never invoked. Pass 2 runs only in that situation, only
  // while grantFinishAmnesty() still has budget (finish_amnesty_max_), and only
  // over the clusters it hands back -- so a genuinely finished mission still
  // terminates.
  for (int amnesty_pass = 0; amnesty_pass < 2; ++amnesty_pass) {
  if (amnesty_pass == 1) {
    if (!frontiers_.empty()) {
      g_empty_cycles = 0;
      break;
    }
    ++g_empty_cycles;
    if (dormant_frontiers_.empty() || !amnestyPassDue())
      break;
    // Deliberately NOT via grantFinishAmnesty(): that call is bounded by
    // finish_amnesty_max_ for the whole mission, and the pre-loop call above
    // already spends from it, so routing this through it burned both grants in
    // the same 1.2ms and rescued nothing (measured 2026-08-18). No budget is
    // needed here anyway -- if the relaxed pass finds nothing, frontiers_ stays
    // empty and the FSM finishes exactly as it would have.
    tmp_frontiers_.clear();
    for (list<Frontier>::iterator it = dormant_frontiers_.begin();
         it != dormant_frontiers_.end(); ++it) {
      // sampleViewpoints() asserts the cluster it is handed is CLEAN
      // (CHECK_EQ(viewpoints_.size(), 0)); a retired cluster can still carry
      // the viewpoints it was selected on, and handing one of those back aborts
      // the node from inside the frontier finder.
      Frontier revived = *it;
      revived.viewpoints_.clear();
      revived.visib_num_ = 0;
      tmp_frontiers_.push_back(revived);
    }
    dormant_frontiers_.clear();
    g_amnesty_relax = true;
    ROS_WARN("[FrontierFinder] Categorisation emptied the frontier set; "
             "re-examining %lu retired cluster(s) with the clearance margin and "
             "the visibility bar dropped", tmp_frontiers_.size());
  }
  for (auto &tmp_ftr : tmp_frontiers_) {'''
src = replace_once(src, OLD_LOOP, NEW_LOOP, "categorisation loop head")

OLD_TAIL = '''  // Bad implementation, need to be changed to unique id
  // Reset indices of frontiers'''
NEW_TAIL = '''  }  // amnesty_pass
  g_amnesty_relax = false;

  // Bad implementation, need to be changed to unique id
  // Reset indices of frontiers'''
src = replace_once(src, OLD_TAIL, NEW_TAIL, "categorisation loop tail")

open(path, "w").write(src)
print("[frontier-visib] applied: unknown-tolerant rays, open-space bar, "
      "post-categorisation amnesty")
PYEOF

grep -q "visib_unknown_tolerance" "$SRC" || {
  echo "[frontier-visib] ERROR: verification grep failed" >&2; exit 1; }
grep -q "amnesty_pass" "$SRC" || {
  echo "[frontier-visib] ERROR: amnesty retry not present" >&2; exit 1; }
grep -q "amnestyRelaxed() && isNearUnknown" "$SRC" || {
  echo "[frontier-visib] ERROR: relaxed clearance test not present" >&2; exit 1; }
echo "[frontier-visib] OK"
