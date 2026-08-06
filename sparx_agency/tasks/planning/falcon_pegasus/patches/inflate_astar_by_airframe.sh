#!/bin/bash
# Tell FALCON's A* how big the drone is.
#
# FALCON reasons about the aircraft as a point in the one place it matters most.
# `pathfinding/src/astar.cpp` tests a SINGLE voxel per node:
#
#     if (getOccupancyGrid()->getVoxel(nbr_pos).value == OCCUPIED || ... UNKNOWN)
#       continue;
#
# with no radius and no parameter for one. So A* will happily route a 0.7 m
# aircraft through a 0.3 m gap between a desk and a partition. The B-spline
# optimiser downstream then cannot find a feasible curve along that route --
# `bspline_opt/safe_distance` wants 0.7 m of clearance that does not exist -- and
# the FSM loops PLAN_TRAJ -> PUB_TRAJ -> PLAN_TRAJ with `Collision detected on
# the trajectory before publishing` until the aircraft's stall watchdog gives up.
# Measured: one flight wedged at (-20.1, -6.7) after 163 s, mapping 288 m3 of a
# 2400 m3 building.
#
# This adds the missing dilation, and three decisions in it are worth stating.
#
# XY ONLY. The rotor disc is horizontal; the airframe is 0.7 m across and about
# 0.15 m tall. Inflating in z as well would eat most of the 1.2 m exploration
# band and stop the aircraft flying over a desk it clears easily.
#
# THE ROUTE PLANNER ONLY -- the two-argument `search()`, which is what
# `ExplorationManager::planExploreMotionHGrid` calls to route to the next
# viewpoint. The five-argument overload is used for connectivity and tour-cost
# queries between hierarchical grid cells, and inflating those changes which
# cells FALCON believes are reachable at all. That is a much larger behavioural
# change with its own failure mode ("Cell N has frontiers, but no free
# subspace"), and it is not what "the drone is 0.7 m wide" implies. A route that
# cannot be found is already a handled outcome: the FSM logs it, keeps the
# frontier, and tries again.
#
# THE START IS EXEMPT, within `start_clearance`. Inflation prevents the aircraft
# entering a pocket it cannot fit through; it does nothing to get one out, and
# an aircraft already inside the skirt of a wall would find no valid start node
# and never plan again. Suppressing the skirt near the start means a drone that
# has drifted against something can still plan its way off it -- the same rule
# `core/planning/planners/astar` already applies for the XTEND.
#
# Both distances are ROS parameters defaulting to ZERO, so this patch is inert
# until `nav` sets them. That keeps the image usable for a baseline run and, more
# usefully, means the radius can be tuned from the launch file without another
# twenty-minute rebuild.
set -e

HEADER=/catkin_ws/src/FALCON/falcon_planner/pathfinding/include/pathfinding/astar.h
SOURCE=/catkin_ws/src/FALCON/falcon_planner/pathfinding/src/astar.cpp
[ -f "$HEADER" ] || { echo "astar.h not found at $HEADER"; exit 1; }
[ -f "$SOURCE" ] || { echo "astar.cpp not found at $SOURCE"; exit 1; }

python3 - "$HEADER" "$SOURCE" <<'PYEOF'
import sys

header_path, source_path = sys.argv[1], sys.argv[2]


def replace_once(text, needle, replacement, what):
    """Substitute exactly one occurrence, or fail loudly.

    A patch that silently matches nothing produces an image that looks patched
    and behaves like upstream, which is the worst of both.
    """
    count = text.count(needle)
    if count != 1:
        raise SystemExit("patch: expected exactly 1 occurrence of %s, found %d" % (what, count))
    return text.replace(needle, replacement)


# ── header: two config fields, and the helpers that use them ────────────────
header = open(header_path).read()

header = replace_once(
    header,
    "    bool verbose_;\n",
    "    bool verbose_;\n"
    "\n"
    "    // How far to grow obstacles, in metres, in the horizontal plane only.\n"
    "    // Set to the aircraft's radius. 0 disables and restores upstream\n"
    "    // point-robot behaviour.\n"
    "    double inflate_radius_;\n"
    "    // Radius around the search start within which the inflation above is\n"
    "    // suppressed, so an aircraft already touching a wall can still plan\n"
    "    // its way off it instead of finding no valid start node.\n"
    "    double start_clearance_;\n",
    "Config::verbose_")

header = replace_once(
    header,
    '      std::cout << "Verbose: " << verbose_ << std::endl;\n',
    '      std::cout << "Verbose: " << verbose_ << std::endl;\n'
    '      std::cout << "Inflate radius: " << inflate_radius_ << std::endl;\n'
    '      std::cout << "Start clearance: " << start_clearance_ << std::endl;\n',
    "Config::print")

header = replace_once(
    header,
    "  double getDiagHeu(const Eigen::Vector3d &x1, const Eigen::Vector3d &x2);\n",
    "  double getDiagHeu(const Eigen::Vector3d &x1, const Eigen::Vector3d &x2);\n"
    "\n"
    "  // Airframe-aware occupancy. See inflate_radius_.\n"
    "  void buildInflateOffsets();\n"
    "  bool isBlockedInflated(const Eigen::Vector3d &pos, const Eigen::Vector3d &start);\n"
    "  std::vector<std::pair<double, double>> inflate_offsets_;\n",
    "Astar private helpers")

open(header_path, "w").write(header)


# ── source: read the parameters, build the stencil, use it ──────────────────
source = open(source_path).read()

source = replace_once(
    source,
    '  nh.param("/astar/verbose", config_.verbose_, false);\n',
    '  nh.param("/astar/verbose", config_.verbose_, false);\n'
    '  nh.param("/astar/inflate_radius", config_.inflate_radius_, 0.0);\n'
    '  nh.param("/astar/start_clearance", config_.start_clearance_, 0.0);\n',
    "Astar::init parameters")

# The stencil asks the map for its resolution, so it can only be built once
# map_server_ has been assigned -- which happens further down init(), not with
# the parameter reads above. Building it early dereferences an unassigned
# pointer and kills exploration_node before it plans anything.
source = replace_once(
    source,
    "  config_.map_origin_ = map_server_->getOrigin();\n",
    "  config_.map_origin_ = map_server_->getOrigin();\n"
    "  buildInflateOffsets();\n",
    "Astar::init map_server_ assignment")

# The stencil, built once. Sampling in POSITION space rather than voxel indices
# keeps this independent of how the map addresses itself, and the step is tied
# to the map's own resolution so no voxel inside the radius can be stepped over.
helpers = '''
void Astar::buildInflateOffsets() {
  inflate_offsets_.clear();
  if (config_.inflate_radius_ <= 0.0)
    return;

  const double resolution = map_server_ ? map_server_->getResolution() : 0.0;
  if (resolution <= 0.0) {
    ROS_ERROR("[Astar] inflate_radius is %.2f m but the map has no usable resolution "
              "(%.3f); leaving A* as a point-robot planner rather than building a "
              "stencil of unbounded size",
              config_.inflate_radius_, resolution);
    return;
  }
  // Half a voxel: fine enough that a sample lands in every voxel the disc
  // touches, coarse enough that the stencil stays small.
  const double step = std::max(resolution * 0.5, 1e-3);
  const int extent = (int)std::ceil(config_.inflate_radius_ / step);
  const double radius_sq = config_.inflate_radius_ * config_.inflate_radius_;

  std::vector<std::pair<double, std::pair<double, double>>> ranked;
  for (int ix = -extent; ix <= extent; ++ix) {
    for (int iy = -extent; iy <= extent; ++iy) {
      if (ix == 0 && iy == 0)
        continue;
      const double ox = ix * step, oy = iy * step;
      const double distance_sq = ox * ox + oy * oy;
      if (distance_sq > radius_sq)
        continue;
      ranked.push_back(std::make_pair(distance_sq, std::make_pair(ox, oy)));
    }
  }
  // Farthest first. A node beside a wall is rejected by an outer sample, so
  // ordering outward-in makes the common rejection the fastest one.
  std::sort(ranked.begin(), ranked.end(),
            [](const std::pair<double, std::pair<double, double>> &a,
               const std::pair<double, std::pair<double, double>> &b) {
              return a.first > b.first;
            });
  for (const auto &entry : ranked)
    inflate_offsets_.push_back(entry.second);

  ROS_WARN("[Astar] obstacles inflated by %.2f m in XY (%zu samples per node, "
           "map resolution %.2f m); inflation suppressed within %.2f m of the start",
           config_.inflate_radius_, inflate_offsets_.size(), resolution,
           config_.start_clearance_);
}

bool Astar::isBlockedInflated(const Eigen::Vector3d &pos, const Eigen::Vector3d &start) {
  auto blocked = [&](const Eigen::Vector3d &p) {
    const voxel_mapping::OccupancyType value =
        map_server_->getOccupancyGrid()->getVoxel(p).value;
    return value == voxel_mapping::OccupancyType::OCCUPIED ||
           value == voxel_mapping::OccupancyType::UNKNOWN;
  };

  if (blocked(pos))
    return true;
  if (inflate_offsets_.empty())
    return false;
  // An aircraft that has drifted against a wall is inside its own skirt. Keep
  // the skirt off near the start or it can never plan away from it.
  if (config_.start_clearance_ > 0.0 &&
      (pos - start).head<2>().norm() < config_.start_clearance_)
    return false;

  for (const std::pair<double, double> &offset : inflate_offsets_) {
    if (blocked(Eigen::Vector3d(pos.x() + offset.first, pos.y() + offset.second, pos.z())))
      return true;
  }
  return false;
}

'''

source = replace_once(
    source,
    "int Astar::search(const Eigen::Vector3d &start_pt, const Eigen::Vector3d &end_pt) {\n",
    helpers.lstrip("\n") +
    "int Astar::search(const Eigen::Vector3d &start_pt, const Eigen::Vector3d &end_pt) {\n",
    "Astar::search (2-arg) definition")

# Only the two-argument search's own body. Sliced out first, because the same
# OCCUPIED-or-UNKNOWN test appears verbatim in searchBBox and the three
# searchUnknown* variants, and a plain replace() walks straight into them --
# which it did, silently, on the first attempt.
begin = source.index(
    "int Astar::search(const Eigen::Vector3d &start_pt, const Eigen::Vector3d &end_pt) {")
end = source.index("\nint Astar::searchBBox(", begin)
body = source[begin:end]

replaced = 0
while True:
    marker = "map_server_->getOccupancyGrid()->getVoxel("
    at = body.find(marker)
    if at < 0:
        break
    # The test spans two getVoxel calls joined by `||`; find the variable from
    # the first and rewrite the whole condition, however it happens to be
    # indented.
    variable = body[at + len(marker):body.index(")", at + len(marker))]
    condition_start = body.rindex("if (", 0, at)
    condition_end = body.index(") {", body.index("UNKNOWN", at))
    indent = " " * (condition_start - body.rindex("\n", 0, condition_start) - 1)
    body = (body[:condition_start]
            + "if (isBlockedInflated(%s, start_pt)" % variable
            + body[condition_end:])
    del indent
    replaced += 1

if replaced != 3:
    raise SystemExit("patch: expected 3 occupancy tests in Astar::search(start, end), "
                     "rewrote %d" % replaced)
source = source[:begin] + body + source[end:]

# std::sort and std::max, which upstream did not need here.
if "#include <algorithm>" not in source:
    lines = source.split("\n")
    first_include = next(i for i, line in enumerate(lines) if line.startswith("#include"))
    lines.insert(first_include, "#include <algorithm>")
    source = "\n".join(lines)

open(source_path, "w").write(source)
print("patched: A* inflates obstacles by /astar/inflate_radius in XY")
PYEOF
