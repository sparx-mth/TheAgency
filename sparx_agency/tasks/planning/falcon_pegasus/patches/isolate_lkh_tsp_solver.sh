#!/bin/bash
# Stop FALCON's vendored LKH TSP solver taking the whole planner down with it.
#
# `ExplorationManager::solveTSP` calls `solveTSPLKH()` in-process. LKH is
# third-party C written to be a one-shot program -- read a file, solve, exit --
# and FALCON calls it thousands of times in one process instead. Its global
# state does not survive that, and on some coverage-tour instances it segfaults
# deep inside its own Lin-Kernighan search:
#
#     #2  solveTSPLKH(char const*)
#     #1  Between_SL / Best
#     #0  SIGSEGV (Address not mapped to object [0x1000001ae])
#
# It is the single biggest reason no Isaac Sim flight finishes. Observed at
# 43 s, 133 s and 163 s into otherwise healthy flights, and twice more before
# that. And it is invisible from the aircraft: `traj_server` is a separate
# process that outlives the planner and keeps republishing the last
# trajectory's final point, so commands keep arriving at 100 Hz, tracking error
# reads centimetres, and only the trajectory id stops moving.
#
# THE FIX IS ISOLATION, NOT REPAIR. Nothing here can debug LKH's internals, and
# it does not matter: `solveTSP` already talks to it entirely through FILES --
# it writes `<name>.tsp`, calls the solver, and reads `<name>.txt`. So the
# solver can run in a forked child. If the child dies, the parent is untouched
# and writes a fallback tour into the file the reader is about to open, in the
# same TSPLIB format, so every line of parsing and every id convention
# (`skip_first_`, `skip_last_`, `result_id_offset_`) stays exactly as upstream
# wrote it.
#
# THE FALLBACK IS A GREEDY NEAREST-NEIGHBOUR TOUR. Worse than LKH's, and the
# right thing anyway: a coverage tour that visits the same cells in a slightly
# worse order costs seconds of flight, while a dead planner costs the flight.
# It runs only when LKH has already failed.
#
# The child is also given a deadline. `fork()` from a process with ROS's
# background threads can, in principle, leave the child deadlocked on a malloc
# lock held at fork time; LKH normally answers in milliseconds, so anything past
# the timeout is killed and treated as a failure rather than hanging the FSM
# forever.
set -e

MANAGER=/catkin_ws/src/FALCON/falcon_planner/exploration_manager/src/exploration_manager.cpp
[ -f "$MANAGER" ] || { echo "exploration_manager.cpp not found at $MANAGER"; exit 1; }

python3 - "$MANAGER" <<'PYEOF'
import sys

path = sys.argv[1]
source = open(path).read()

call = """  // Call LKH TSP solver
  solveTSPLKH((ep_->tsp_dir_ + "/" + config.problem_name_ + ".par").c_str());
"""

replacement = """  // Call LKH TSP solver -- in a CHILD PROCESS, so that when it segfaults it
  // takes only itself with it. See patches/isolate_lkh_tsp_solver.sh.
  const std::string tsp_par = ep_->tsp_dir_ + "/" + config.problem_name_ + ".par";
  const std::string tsp_out = ep_->tsp_dir_ + "/" + config.problem_name_ + ".txt";
  if (!runLKHIsolated(tsp_par, tsp_out)) {
    ROS_ERROR("[ExplorationManager] the LKH solver died on '%s' (%d cities); "
              "falling back to a greedy nearest-neighbour tour",
              config.problem_name_.c_str(), config.dimension_);
    writeGreedyTour(cost_matrix, config.dimension_, config.skip_last_, tsp_out);
  }
"""

# RE-RUNNABLE. An image built by an earlier run of this script has no bare
# solveTSPLKH call site left to replace, and refusing to run against it means
# the only way to correct a bug in the code below is a rebuild from the base
# image -- an hour, on a machine whose compiler segfaults at random. So a source
# that is already patched is brought UP TO DATE in place instead.
already = "runLKHIsolated(tsp_par, tsp_out)" in source
if already:
    # The greedy tour's length must be accumulated per edge in hundredths. See
    # the long comment at the accumulator for why: getting this wrong aborts
    # the node on a glog CHECK_NEAR, in the one path meant to keep it alive.
    corrections = [
        ("  double total = 0.0;\n", "  long long total_units = 0;\n"),
        ("    total += best_cost;\n",
         "    total_units += (long long)(best_cost * 100.0);\n"),
        ("    total += cost_matrix(current, depot);\n",
         "    total_units += (long long)(cost_matrix(current, depot) * 100.0);\n"),
        ('fout << "COMMENT : Length = " << (long long)(total * 100.0) << "\\n";',
         'fout << "COMMENT : Length = " << total_units << "\\n";'),
    ]
    applied = 0
    for old, new in corrections:
        if old in source:
            source = source.replace(old, new)
            applied += 1
    if applied == 0:
        print("patch: already applied and already correct, nothing to do")
    else:
        open(path, "w").write(source)
        print("patch: corrected the greedy tour length convention (%d edits)" % applied)
    raise SystemExit(0)

if source.count(call) != 1:
    raise SystemExit("patch: expected exactly 1 solveTSPLKH call site, found %d"
                     % source.count(call))
source = source.replace(call, replacement)

# The two helpers, placed immediately before solveTSP so they are defined at the
# point of use without touching the header.
helpers = '''namespace {

/// Run LKH in a forked child. True if it finished; false if it died or hung.
///
/// The child does nothing but solve and exit, and exits with _exit() so that no
/// atexit handler, stream flush or ROS teardown runs twice.
bool runLKHIsolatedImpl(const std::string &par_file, double timeout_s) {
  ::fflush(NULL);
  const pid_t pid = ::fork();
  if (pid < 0) {
    ROS_ERROR("[ExplorationManager] fork() failed before the TSP solve: %s",
              strerror(errno));
    return false;
  }
  if (pid == 0) {
    solveTSPLKH(par_file.c_str());
    ::_exit(0);
  }

  // Poll rather than block, so a child that wedges cannot wedge the FSM.
  const ros::WallTime deadline = ros::WallTime::now() + ros::WallDuration(timeout_s);
  int status = 0;
  while (ros::WallTime::now() < deadline) {
    const pid_t done = ::waitpid(pid, &status, WNOHANG);
    if (done == pid)
      return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (done < 0)
      return false;
    ::usleep(200);
  }
  ROS_ERROR("[ExplorationManager] the TSP solver did not answer within %.1f s; killing it",
            timeout_s);
  ::kill(pid, SIGKILL);
  ::waitpid(pid, &status, 0);
  return false;
}

/// Write a greedy nearest-neighbour tour where LKH would have written its own.
///
/// Same TSPLIB shape the reader in solveTSP expects, so none of its parsing or
/// id conventions change: a "COMMENT : Length = N" line it may scan for, a
/// TOUR_SECTION, one 1-based city per line, terminated by -1.
void writeGreedyTourImpl(const Eigen::MatrixXd &cost_matrix, int dimension,
                         bool skip_last, const std::string &out_file) {
  std::vector<bool> visited(dimension, false);
  std::vector<int> tour;
  tour.reserve(dimension);

  int current = 0;                       // city 1: always the current state
  visited[0] = true;
  tour.push_back(0);
  // With skip_last_ the reader STOPS at city `dimension`, which upstream uses
  // as a virtual depot and LKH always leaves at the end of the tour. A greedy
  // walk has no such guarantee, and dropping that city mid-walk would silently
  // truncate the tour -- handing FALCON a handful of frontiers instead of all
  // of them, in the one code path that only runs when things have already gone
  // wrong. So it is held back and appended.
  const int depot = dimension - 1;
  if (skip_last && dimension > 1)
    visited[depot] = true;

  // Accumulated in HUNDREDTHS, per edge, and never as a sum of doubles.
  //
  // This is not a style choice, it is the file format. solveTSP writes the
  // cost matrix to LKH as per-edge truncated integers, so the Cost LKH reports
  // back is sum_i trunc(100 * c_i). The caller re-derives exactly that, edge by
  // edge, and then asserts the two agree:
  //
  //     CHECK_NEAR(cost, grid_tour2_cost_sum, 1e-4)   exploration_manager.cpp:246
  //
  // glog's CHECK_NEAR is fatal. Summing raw doubles and truncating once gives
  // trunc(100 * sum_i c_i) instead, which differs by up to a hundredth per edge
  // -- measured at 0.04 on a ten-city tour, four hundred times the tolerance,
  // and over the tolerance on 400 of 400 random ten-city instances. So the
  // fallback added to keep exploration_node alive when LKH dies was aborting
  // it instead: SIGABRT, immediately after "the LKH solver died on
  // 'coverage_path'", with traj_server still republishing the last endpoint so
  // the aircraft looked healthy while the planner was gone.
  long long total_units = 0;
  for (int step = 1; step < dimension; ++step) {
    int best = -1;
    double best_cost = std::numeric_limits<double>::max();
    for (int candidate = 0; candidate < dimension; ++candidate) {
      if (visited[candidate])
        continue;
      const double cost = cost_matrix(current, candidate);
      if (cost < best_cost) {
        best_cost = cost;
        best = candidate;
      }
    }
    if (best < 0)
      break;
    visited[best] = true;
    tour.push_back(best);
    total_units += (long long)(best_cost * 100.0);
    current = best;
  }
  if (skip_last && dimension > 1) {
    total_units += (long long)(cost_matrix(current, depot) * 100.0);
    tour.push_back(depot);
  }

  std::ofstream fout(out_file);
  fout << "COMMENT : Length = " << total_units << "\\n";
  fout << "TOUR_SECTION\\n";
  for (size_t i = 0; i < tour.size(); ++i)
    fout << (tour[i] + 1) << "\\n";     // TSPLIB cities are 1-based
  fout << "-1\\n";
  fout.close();
}

}  // namespace

bool ExplorationManager::runLKHIsolated(const std::string &par_file,
                                        const std::string &out_file) {
  (void)out_file;
  return runLKHIsolatedImpl(par_file, 5.0);
}

void ExplorationManager::writeGreedyTour(const Eigen::MatrixXd &cost_matrix, int dimension,
                                         bool skip_last, const std::string &out_file) {
  writeGreedyTourImpl(cost_matrix, dimension, skip_last, out_file);
}

'''

anchor = "void ExplorationManager::solveTSP(const Eigen::MatrixXd &cost_matrix,"
if source.count(anchor) != 1:
    raise SystemExit("patch: expected exactly 1 solveTSP definition, found %d"
                     % source.count(anchor))
source = source.replace(anchor, helpers + anchor)

# Headers for fork/waitpid/kill. Appended AFTER the existing include block, never
# before it -- inserting ahead of a PCL or Eigen header is how a previous patch
# in this directory made gcc 9 die with an internal compiler error.
needed = ['#include <unistd.h>', '#include <sys/wait.h>', '#include <signal.h>',
          '#include <cerrno>', '#include <cstring>', '#include <limits>',
          '#include <fstream>', '#include <vector>', '#include <string>']
missing = [h for h in needed if h not in source]
if missing:
    lines = source.split("\n")
    last_include = max(i for i, line in enumerate(lines) if line.startswith("#include"))
    for offset, header in enumerate(missing):
        lines.insert(last_include + 1 + offset, header)
    source = "\n".join(lines)

open(path, "w").write(source)
print("patched: LKH runs in a child process with a greedy fallback")
PYEOF

HEADER=/catkin_ws/src/FALCON/falcon_planner/exploration_manager/include/exploration_manager/exploration_manager.h
[ -f "$HEADER" ] || { echo "exploration_manager.h not found at $HEADER"; exit 1; }

python3 - "$HEADER" <<'PYEOF'
import sys

path = sys.argv[1]
header = open(path).read()

anchor = "  vector<int> dijkstra(vector<vector<double>> &graph, int start, int end);\n"
addition = anchor + """
  // Run the vendored LKH solver in a forked child, so its segfaults are its
  // own. True if it completed. See patches/isolate_lkh_tsp_solver.sh.
  bool runLKHIsolated(const std::string &par_file, const std::string &out_file);
  // Write a greedy nearest-neighbour tour in LKH's own output format, for when
  // it did not complete.
  void writeGreedyTour(const Eigen::MatrixXd &cost_matrix, int dimension,
                       bool skip_last, const std::string &out_file);
"""

# Re-runnable, like the source patch above. Declaring the two helpers a second
# time is not harmless -- C++ rejects the redeclaration outright ("cannot be
# overloaded with" itself) and the build fails on a header that was already
# correct.
if "runLKHIsolated" in header:
    print("patched: exploration_manager.h already declares the isolated solver")
else:
    if header.count(anchor) != 1:
        raise SystemExit("patch: expected exactly 1 dijkstra declaration, found %d"
                         % header.count(anchor))
    open(path, "w").write(header.replace(anchor, addition))
    print("patched: exploration_manager.h declares the isolated solver")
PYEOF
