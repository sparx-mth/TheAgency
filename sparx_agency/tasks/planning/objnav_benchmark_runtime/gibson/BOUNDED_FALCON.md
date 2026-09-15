# Bounded FALCON for Gibson ObjectNav

## Identity and scope

This is a **FALCON 2D/2.5D ObjectNav adaptation**, not the unchanged aerial
planner, a detector, or a renamed frontier sweep. Gibson `cc57809f` is the base.
The completed detector commit `5c465f97` was applied without committing; its
YOLO-World **X-v2** default and selectable pretrained LLMDet are preserved.
The divergent shared-infrastructure branch `60d91714` was not merged wholesale.
No other simulator worktree is changed. The old explorer remains the default
until broader independent validation supports changing it.

### Sources actually inspected

* Zhang, Chen, Feng, Zhou and Shen, **FALCON: Fast Autonomous Aerial Exploration
  using Coverage Path Guidance**, IEEE T-RO, vol. 41, pp. 1365–1385,
  DOI [10.1109/TRO.2024.3522148](https://doi.org/10.1109/TRO.2024.3522148).
  Method reference: [arXiv 2407.00577v2](https://arxiv.org/abs/2407.00577v2),
  16 May 2025. The local PDF extraction has 20 pages; its complete text, Alg. 1,
  method, evaluation, ablations and references were read, not just the abstract.
* [Official source](https://github.com/HKUST-Aerial-Robotics/FALCON/tree/312eb4d32c6c7af1a482f94a0a204aa2bb150cca),
  `ros1-noetic`, revision `312eb4d32c6c7af1a482f94a0a204aa2bb150cca`.
  Local reference: `~/papers/falcon/{paper.pdf,text,code}`. This is the pinned
  inspected revision, not a claim about the latest remote tip.
* Supplementary experiments: [authors' video](https://youtu.be/BGH5T2kPbWw),
  cited on pp. 1 and 18. The available editor tools do not render images/video:
  the video and extracted figure images were **not visually verified** in this
  session. No experimental numbers or equations are inferred from flattened
  figure/table text. The implementation correspondence below is checked against
  executable source. No separate textual supplement was located.

Upstream has no top-level LICENSE at the inspected revision; planner package
manifests say `TODO`. Consequently no upstream C++/solver code is vendored or
relicensed here. This is an independently written implementation of the paper's
optimization problems using existing SPARX geometry and scipy. Direct binary
reuse also requires ROS1, a voxel MapServer, aerial trajectories and actuation;
simply fixing trajectory altitude would not meet the ground-robot contract.

### Precedent and comparisons, without inventing one

FALCON p. 3 Table I and p. 5 §II-B distinguish prior **2D** coverage-guided
exploration: Kan, Teng & Karydis, *Online Exploration and Coverage Planning in
Unknown Obstacle-Cluttered Environments* (RA-L 2020, 5(4):5969–5976), and Zhao
et al., *TDLE: 2-D LiDAR Exploration with Hierarchical Planning Using Regional
Division* (CASE 2023). These precede FALCON and are **not FALCON adaptations**.
Its own p. 11 Table II labels several environments 2.5D, but still evaluates a
flying robot. Official `hierarchical_grid.cpp:1529` contains `getCCLCenters2D`,
which slices at a hardcoded 1 m; it is not a body-height-clearance ObjectNav port.

Relevant ObjectNav methods reviewed in this repository are
[SG-Nav](https://arxiv.org/abs/2410.08189), §3.3 pp. 5–6 (LLM frontier reasoning and
graph-based re-perception), [ApexNav](https://arxiv.org/abs/2504.14478), §IV-B/C
(adaptive semantic/geometric exploration and target verification), and
[OSG Navigator](https://arxiv.org/abs/2508.04678), §7 (Gibson/HM3D evaluation).
They motivate the separation of exploration, reasoning and verification, not
an established Gibson FALCON comparison. The existing
[protocol audit](PROTOCOL_AND_TUNING.md) documents their different evaluation
settings. No published planar **FALCON ObjectNav** adaptation was verified.
Additional web queries for that specific precedent returned search interstitials;
this limited search does not prove none exists. No such precedent is claimed.

## Component mapping

Paths in the source column are under official `falcon_planner/` at the pin above.

| Component | Treatment and correspondence |
|---|---|
| Occupancy/TSDF/ESDF | **Replaced:** retain observed metric-depth robot-height log-odds map and physical inflation. No GT map/room data. Drone voxel integration is not needed for planar motion. |
| Incremental frontier maintenance | **Adapted:** IV-C, p. 7; `exploration_preprocessing/src/frontier_finder.cpp:53–134`. Update the free/unknown boundary on changed observations; vectorized CCL, overlap identity, persistent geometric visited/rejected viewpoints. No frozen initial list. |
| Frontier splitting | **Retained in 2D:** `frontier_finder.cpp:442–522`, PCA splits until bounded cluster radius. |
| Viewpoints and information | **Retained/adapted:** IV-C pp. 7–8; `frontier_finder.cpp:736–889,1109–1133`. Ring samples, mean facing, visible-frontier test, unknown-ray gain qualification, top alternatives. Actual camera intrinsics/depth/pitch, known-column occlusion; not omnidirectional sensing. |
| Connectivity-aware decomposition | **Retained in 2D:** IV-A p. 6; `hierarchical_grid.cpp:527–622,1214,1529`. Uniform coarse cells, separate safe-free and unknown CCL zones, centres rectified into the correct component. Cached by geometry, not room IDs. |
| Connectivity/portal graph | **Retained/adapted:** IV-B p. 7 Alg. 1; `hierarchical_grid.cpp:623–790`. Restricted one/two-cell paths; free/free, unknown/unknown and same-cell portal edges. Exclude isolated unknown components. Native index/arithmetic limits are not copied. |
| Coverage path | **Retained:** V-A p. 8; `hierarchical_grid.cpp:1966–2158`, `exploration_manager.cpp:154–240`. Open ATSP through active-free viewpoint centres **and** unknown-zone centres. First destination is active-free. Coverage is confined to the current scope. |
| CP-guided local ordering | **Retained:** V-B p. 9; `exploration_manager.cpp:284–613`. Remove current/first CP zones; insert their frontier representatives with hard precedence on the remaining CP sequence. Execute the viewpoint prefix, defer those ordered beyond the next CP element. |
| Local viewpoint refinement | **Retained:** V-C pp. 9–10; `exploration_manager.cpp:789–927,1428–1552`. Layered shortest-path optimization over alternatives, including the next guidance element as a terminal cost where available. |
| LKH and SOP solvers | **Replaced, objectives retained:** exact subset DP for small domains, deterministic beam for larger ones. `optimal`/`beam`/infeasible/deadline provenance is recorded. No claim of LKH-equivalent performance or exactness for beam results. |
| Motion cost | **Adapted:** V-A/B pp. 8–9; `pathfinding/src/path_cost_evaluator.cpp:13–77`. Serial forward-distance/step plus path yaw changes/turn, not UAV max(position,yaw), acceleration or vertical cost. Costs are estimates, actual actions govern budgets. |
| Hybrid path search | **Adapted:** near grid Dijkstra (zero-heuristic A*), far zone-graph shortest paths with endpoint offsets. Reuses SPARX compact passability graph; diagonal corner cuts removed. Hypothetical penalized unknown routes cannot become executable routes. |
| B-spline/quadrotor trajectory | **Replaced:** V-C p. 10; `exploration_manager.cpp:980–1040`. Free-space path shortening, unchanged discrete converter, footprint/unknown checks of the next forward sweep, bounded recovery. No flight/altitude clamping, teleportation, extra sensing or enlarged actions. |
| Replanning | **Adapted:** p. 11 §VII-A; `exploration_fsm.cpp:495–545,855–869`. Retain observation/frontier and safety events, but do not replace a safe committed destination every 3 s. Complete/invalid/blocked viewpoints authorize replanning. |

Important paper/code differences: released unknown penalty is 2.0; viewpoint
cutoff uses mean **plus** configured z-score (shipped zero). The paper describes
mean minus z-score. This port follows the zero-cutoff source operating point,
keeps equal-gain ties explicitly, and records all parameters. It does not copy
the upstream finite failed-path magic numbers or relax failed refinement edges.

Existing XTEND deployment clones the branch without a revision pin in
`tasks/planning/falcon/Dockerfile:179`. Its patches change cost assertions, SOP
timeout assertions, depth allocation, map resolution, system-info parsing,
visualization/log cadence and build compatibility. Pegasus additionally isolates
LKH and patches index/bounds, airframe inflation and measured-state replanning;
SJTU has further blockage/visibility/room-confinement patches. None are evidence
that Gibson previously ran FALCON, and none of those deployments was modified.

## State machine and budgets

`BurstMachine` is simulator-neutral. `FalconObjectNav` composes it with the
unchanged `RptStarRoomSolver`, room classifier/oracle and target-evidence logic:

**initial/selected room → local exploration → room reasoning → room selection
→ committed A*/WA* transit → local exploration**, with verification and recovery
interrupts. Zero-motion transitions happen in a bounded loop, not idle actions.

Default parameters are in `core/planning/exploration/falcon/params.py` and appear
in every run configuration. Important defaults:

* 48 actions per exploration burst, including its turns, failed movements,
  recovery and verification while the burst is active; **500 episode actions**
  from the unchanged Gibson protocol. STOP is also an action.
* 24 actions per target interrupt; after rejection/loss resume the same burst
  token/allowance and saved route. Budget expiry during verification does not
  suppress the bounded target reaction; subsequent verification is global-only.
* 90 transit actions; 3 recovery actions per attempt, 4 failed local goals;
  sustained <0.05 m² proxy gain over 14 exploration decisions ends a burst.
* Separate 1.5 s cooperative planning deadline, 18 sampled frontier clusters per
  transaction, 18,000 passable-grid-node cap, 64 order-node cap (decomposition
  capped at four times that), exact DP up to 10 nodes then beam width 128. Native
  scipy calls are capacity-bounded and checked before/after, not preempted. Deferred
  frontiers remain observed, not labelled searched. Deadline/cap failures are
  visible, never an implicit frontier-backend run.
* Radius 4 m anchored local envelope; coarse cell size 4 m. Initially use a
  provisional observed region without requiring any room boundary or full scan.
  As inferred boundaries arrive, match the anchored overlap and exclude other
  observed rooms. The envelope never follows doorway jitter or expands each step.
* Up to 3 overlapping-region bursts; prefer fresh reachable rooms. A useful,
  budget-expired partial region can be reconsidered by RPT* after reasoning.
  A partial/budget verdict is **not** searched-completely or target-free.

Room IDs never own the budget. Splits inherit overlapping geometric visit
history; merges inherit all overlapping visits. The active burst remains the
same token. RPT* changes the destination only at a decision boundary, not when
new objects change scores mid-transit. Transit arrival needs a reachable goal
inside the observed room and debounced entry. A blocked centroid is never
required. A significant observed floor-height change discards incompatible map,
room, viewpoint and route geometry, but retains the global action ledger.

Room classification retains the existing object/class/stability thresholds.
Geometry updates accumulate evidence with model queries disabled; reasoning at
the burst boundary enables the existing classifier and probability oracle.
Only an actual partition change resets stability evidence. Repeated calls on
one observation do not count as multiple evidence updates, and starting a new
burst is not itself a partition reset.

View headings are quantized to the existing turn lattice; already-observed
zero-motion headings are not emitted as informative views. An unusable initial
view can trigger a small charged recovery, not a mandatory panorama.

Routes remain committed through turns and unrelated map changes. Safety,
cross-track error, positional stagnation, scope invalidation, actual blocked
motion and arrival invalidate them. No movement is credited merely because a
new path exists. A target interrupt saves, rather than discards, the exploration
route; after rejection it is rechecked against current geometry before reuse.
Budget-expired unfinished viewpoints and route cursors are stored separately
and can resume on an authorized overlapping-region revisit. Only explicit
inactive phases shift route-watchdog clocks, never ordinary replanning or
failed movement. The final safety veto checks the **actual converter action**,
including after cursor restoration and through recording proxies.
An unsafe transit/verification proposal invalidates its route and commits to
charged recovery turns before retrying, rather than immediately undoing the
veto turn. Repeated unsafe transit proposals terminate after the configured
failure cap; repeated unsafe target approaches reject the candidate. Recovery
time remains inside the original target-interrupt action allowance.
The existing bounded LLM HTTP retry/schema repair remains. Failure is recorded
and terminates rather than silently substituting uniform reasoning. Unreachable
transit and action-limit failures terminate visibly; there is no hanging wait.

The mapper's only added free evidence is the robot's **current measured
footprint**, never a future goal; occupied cells are retained. Both explorer
selections use that same map. Unknowns constrain physical motion. The separate
CP graph may reason hypothetically through unknowns with a penalty, precisely
as guidance, never as an executable path. Clearance remains observed-map based:
depth discretization and unobserved obstacles can still cause real collisions.

## Reproducible development run

Use the existing Habitat environment for rendering, the existing lightweight
environment for tests, and the completed detector's separate model environment.
No ROS/native FALCON build, new model weights or new dependency installs are needed.

1. Inspect `nvidia-smi`, CPU/RAM and running processes. Do not change another
   service's vocabulary. Use a dedicated CPU detector; the LLM must also be CPU
   or remote while Habitat owns the rendering GPU.
2. Obtain the vocabulary with `gibson.run --print-vocabulary`. Start the completed
   detector with that exact order, `--backend yolo_world --model
   "$HOME/models/objnav/yolov8x-worldv2.pt" --device cpu --conf 0.05
   --torch-threads 4 --host 127.0.0.1 --port 18095`; see the
   [detector runbook](../../../mapping/scene_graph/serve/README.md).
3. Run the existing CLI with `--explorer falcon --detector-backend yolo_world
   --detector-url http://127.0.0.1:18095 --scene Collierville --limit 1`, the
   documented episode/scene directories, `--allow-sim-version-mismatch`, and a
   **new** `--output` directory. `--record` is optional; no video is required for
   action/coverage/latency diagnostics. `--explorer frontier` selects the baseline.
4. A policy JSON may contain `falcon` parameter overrides. The explorer flag
   cannot silently disagree with `local_exploration` in that file.

Paired comparison (repository root, same existing CPU services throughout):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$HOME/miniconda3/envs/habitat/bin/python" \
  -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.compare_explorers \
  --output "$HOME/objnav_benchmark/gibson/falcon-paired-dev" \
  --scenes Collierville Corozal --seed 0 \
  --detector-url http://127.0.0.1:18095 --detector-backend yolo_world \
  --episodes-dir "$HOME/datasets/objectnav/gibson/objectnav/gibson/v1.1/val" \
  --scenes-dir "$HOME/datasets/gibson/scenes" --allow-sim-version-mismatch
```

The runner checks equal source/model/LLM/data/action identities, uses fresh
sequential processes and refuses a FALCON result with no executable CP/SOP
plan. Outputs: `COMPARISON.md`, `comparison.json`, `coverage.csv`, and original
per-explorer/scenario run configurations, scores and console logs. Native
collisions and first success-region entry are in the evaluator-only
`evaluation_diagnostics.jsonl` sidecar, not policy observations or primary
native-score fields. First region entry differs from a verified target STOP;
terminal actions are reported separately. Stagnation wall time is the decision
computation preceding unchanged pose/coverage, excluding renderer/wait time.
Same sensor
protocol is controlled; post-action frames necessarily differ between policies.
This compares the preserved baseline against the **integrated hierarchy**, not
an isolated explorer-only ablation: reasoning cadence and additional ground
safety orchestration also differ by design. The same completed detector,
LLM configuration, observation/action protocol, starts and global cap are used.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests \
  sparx_agency/core/planning/objnav \
  sparx_agency/core/planning/exploration/tests \
  sparx_agency/tasks/mapping/scene_graph/tests -q
```

See `FALCON_RESULTS.md` for measured results and outstanding limitations. Only
previously inspected development starts are used here; do not label them held
out or tune the remaining published validation split using their scores.
Gibson/SemExp v1.1 scoring, ZSON meaning and Habitat 0.2.4 vs reference 0.1.5
caveats remain as documented in `PROTOCOL_AND_TUNING.md`.



