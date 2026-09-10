# ObjectNav benchmark harness — run it, score it, log it, compare it

Drives any headless agent through any `ObjNavEnv`, scores every benchmark with
the same code, and writes results outside the repository. The contracts it
drives live in [`core/planning/objnav`](../../../core/planning/objnav/).

```bash
# the whole loop on a fake building, with a privileged oracle -- run this first
.venv/bin/python -m sparx_agency.tasks.planning.objnav_benchmark.smoke --quick
```

It exits 1, and says why, unless the oracle scores SR 100% with no agent errors
— so it can gate a scripted sweep. Results land in
`~/objnav_benchmark/<benchmark>/<split>/<UTC stamp>/`.

## What it answers, and what it refuses to answer

**It answers:** on this benchmark split, with ground-truth RGB-D and a perfect
pose, how often does the agent find the object (SR), how efficiently (SPL,
SoftSPL), how close does it end (DTG) — with confidence intervals, broken down
by category and scene, and set against the numbers the papers report.

**It refuses** to produce a number it cannot defend. Every failure mode below
yields a *plausible* figure, so each one stops the run instead:

- a simulator whose realised motion disagrees with the episode's
  `DiscreteActionSpec`, checked after every action: another turn, tilt or step
  (AI2-THOR built without `rotateStepDegrees=30`, habitat-sim without the
  benchmark config), an unconverted Y-up frame, or a mirrored handedness (see
  *The motion check*);
- an environment that keeps running after STOP or past its step budget, ends
  early, changes camera mid-episode, serves an episode id twice, serves an
  episode of another benchmark or split than the run's, or reports a step count
  that disagrees with the actions it was sent;
- episode metadata whose keys name privileged information (goal, shortest,
  geodesic, distance_to, path_to, view_point/viewpoint, target_position,
  object_position) — a tripwire for the known leaks, not a guarantee;
- a path length the environment reports that disagrees with the one recomputed
  from the poses the agent observed — a unit or accounting bug in an adapter
  (centimetres, a dropped axis, `p` accounted in 2-D or skipping a step). A
  rotated or mirrored frame keeps every chord's length; that is the motion
  check's to catch;
- a simulator's own SPL, SoftSPL, success or DTG that disagrees with ours;
- a success without STOP on a benchmark whose protocol requires STOP;
- an agent decision the benchmark cannot execute (`AgentContractError`). Under
  `on_agent_error="record"` it is logged as an agent error and scored as a
  failure, and so is agent diagnostics that cannot be stored (kept as
  `episode_info_error` beside a score that stands). Our own
  `ObjNavInternalError` and `HarnessError` always stop the run: an invariant of
  the infrastructure breaking must never be charged to the method;
- a second writer on a results directory, a resume under a different
  configuration, an episode logged twice, a finish beside rows the logger did
  not write, a row from another schema version.

## The motion check

After every action the runner compares the pose before and after with the
episode's action spec (`kinematics.KinematicTolerance`, defaults sized for
Habitat's partial no-sliding steps and stairs, and for AI2-THOR's actuation
noise):

| action | realised motion allowed |
|---|---|
| TURN_LEFT / TURN_RIGHT | no translation (2 cm); yaw change within 3° of ±turn angle |
| LOOK_UP / LOOK_DOWN | no translation or yaw change; pitch change 0 (refused at a limit) or the signed tilt, within 0.5° |
| MOVE_FORWARD | planar step at most the forward step + 3 cm, within 10° of the heading; climb at most max(0.2 m, the planar step) |
| STOP | nothing moves |

`kinematics=None` disables it — only for a simulator whose motion has been
verified some other way.

## How an episode is scored

One scoring function for every benchmark, from the quantities the environment
measured (`EpisodeMeasurement`): `l` the shortest path to the nearest goal, `d0`
and `dT` the distance to goal at the start and the end, `p` the distance
travelled. Definitions follow Anderson et al. 2018 and habitat-lab's `nav.py`:

| metric | per episode | notes |
|---|---|---|
| Success | the benchmark's own judgement | Habitat: STOP within 0.1 m geodesic of a goal view point; RoboTHOR: STOP with the object visible within 1 m |
| SPL | `S · l / max(l, p)` | failures score 0 |
| SoftSPL | `max(0, 1 − dT/d0) · l / max(l, p)` | Habitat's definition; not gated on success or STOP, so a failure contributes the progress it made; RoboTHOR has no official one |
| DTG | `dT` | `inf` (no goal reachable from the end) is counted separately, never averaged |

`p` sums the 3-D straight-line displacement between successive base positions
from the reset position on, as Habitat does: turns, tilts and STOP add nothing,
and a blocked step adds only what it really moved. Where the official code has
no answer, the convention is stated in `scoring.py`: both `l` and `p` zero gives
a ratio of 1 (as RoboTHOR's evaluators do; Habitat divides by zero), and `d0`
zero gives a soft success of 1 only when `dT` is zero too.

A run's figures are **means over all episodes**, failures included, as the
leaderboards compare them. Each carries an interval: Wilson for SR, a seeded
percentile bootstrap for SPL. Two variants of our method on the same episodes
are compared episode by episode — exact McNemar on success, a sign-flip
permutation test on SPL (exact enumeration decides ties exactly) — because
pairing is what lets a few hundred episodes separate them.

## The results directory

| file | written | holds |
|---|---|---|
| `run.json` | at start and at finish; a resume enters itself only once it logs or finishes, so a refused resume leaves it untouched | git revision (with `-dirty`), argv, Python, platform, the full run configuration, every resume |
| `episodes.jsonl` | one line per episode, as it finishes | an `EpisodeRecord`: identity, measurement, score, action counts, the agent's diagnostics |
| `summary.json` | at finish | the `BenchmarkSummary` |
| `run.lock` | held while a logger writes | one writer at a time; released by `finish()`, `close()` or process exit |

Appending per episode means a crash loses one episode, not the sweep, and a run
can be resumed: the completed episode ids are skipped, and the configuration
must match exactly or the resume is refused. A truncated last line — a run
killed mid-write — is reported with its line number, never skipped. Use
`MetricsLogger` as a context manager, so a run that raises releases its lock.

## Comparing with the papers

`ReportedResult` holds one number a paper printed, as a fraction, with the table
it came from; `comparison_table` sets our summary under them as a Markdown
table and refuses a reported result from another benchmark or split. The
numbers themselves are recorded on each benchmark branch, beside the adapter
whose protocol they were measured under — the papers do not all share one
(camera 0.88 m in ApexNav, 0.90 m in SG-Nav; OSG evaluates a 400-episode subset
of HM3D v0.1).

## The fake environment is a test rig, not a benchmark

`fake_env/` is a 2.5-D building on a grid with ray-cast depth, an atomic-block
collision model and grid geodesics. With `OracleSearchPolicy` — which reads the
goal region straight from the environment and routes with a clearance penalty
— it exercises the entire loop in seconds: environment contract, motion check,
headless agent, action converter, scoring, logging. The oracle is privileged by
construction: an upper bound for checking the pipeline, never a baseline. On
the demo building it reaches SR 1.0 and SPL 0.94 and is never blocked; a random
walk scores 0.

## Files

| file | what it does |
|---|---|
| `errors.py` | `HarnessError`, `ScoringError`, `ResultsError` |
| `checks.py` | the shared validation predicates: `is_real`, `is_fraction`, `is_length`, `is_count`, `is_name` |
| `records.py` | `EpisodeScore`, `EpisodeRecord`, and the row schema |
| `summaries.py` | `GroupSummary`, `BenchmarkSummary`, `PairedComparison`, `ReportedResult` |
| `scoring.py` | SPL, SoftSPL, 3-D path length, and `score_episode` with its protocol checks |
| `cross_checks.py` | the path-length and native-metric cross-checks |
| `stats.py` | Wilson, bootstrap, McNemar, paired permutation — standard library only (not `statistics.py`, which would shadow the standard library) |
| `aggregate.py` | per-run, per-category and per-scene summaries; paired comparisons |
| `comparison.py` | the SOTA comparison table |
| `results_io.py` | the results layout, the git revision, strict JSON, atomic writes, the episode-file reader |
| `run_info.py` | what `run.json` records about how a run was produced |
| `run_lock.py` | one writer per results directory |
| `run_records.py` | what one results directory may hold: one experiment, each episode once |
| `logger.py` | `MetricsLogger`: its lifecycle, the lock, `episodes.jsonl`, `summary.json`, the deferred resume entry |
| `kinematics.py` | the motion check: `KinematicTolerance` and `check_motion` |
| `env_contract.py` | the `ObjNavEnv` contract checks at reset, every step and the measurement |
| `agent_contract.py` | the agent contract and the raise-or-record policy for agent errors |
| `run_plan.py` | what is settled before the first episode: options, participants, the episode list (served ids unique), a resumed logger |
| `runner.py` | the loop: `run_episode`, `run_benchmark` |
| `fake_env/` | the test rig: `world.py` and `ascii_map.py` (the building), `raster.py` (navigable and goal regions, voxels), `geodesics.py` (the legal moves and geodesic fields), `episodes.py` (episode specs and the measurement), `rendering.py` (ray-cast depth), `motion.py` (the atomic forward block), `env.py` (`FakeObjNavEnv`), `oracle_policy.py` and `oracle_aim.py` (the privileged oracle), `labels.py` |
| `smoke.py` | the whole loop on the fake building, from the command line, as a gate |
| `tests/` | the harness's own tests, including a Python 3.8 scan of the harness and of the voxel-camera stub the fake renders with; `corridor.py` and `fake_rooms.py` are shared helpers |

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/tasks/planning/objnav_benchmark -q
```

## Not in here, on purpose

- **No simulator.** Environments come from the benchmark branches.
- **No SOTA numbers yet.** Each benchmark branch records the ones it compares
  against, with their sources and protocol notes.
- **No per-step trace.** Each step's reasons come back to the caller of
  `act()`; only the agent's per-episode `episode_info()` is stored.
- **No plots.** Aggregation returns data; presentation is a separate concern.
