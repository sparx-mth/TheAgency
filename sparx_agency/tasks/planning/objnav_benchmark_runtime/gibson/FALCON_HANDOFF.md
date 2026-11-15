# FALCON/Gibson handoff — 2026-09-15

## What is implemented

- Gibson `cc57809f` plus the completed detector commit `5c465f97` applied
  **without a commit**. YOLO-World X-v2 is preserved; LLMDet and explicit S/L
  rollback remain available. Other simulator worktrees and drone deployments
  were not edited.
- Selectable `--explorer frontier` (default) / `--explorer falcon`.
- ROS-free FALCON planar adaptation: live frontier/PCA maintenance, camera-FoV
  viewpoint qualification, incremental free/unknown connectivity and portals,
  active-free-plus-unknown coverage ordering, hard-precedence SOP, layered view
  refinement and free-space routes. Exact/beam/cap/deadline status is recorded.
- An explicit action-clock hierarchy, provisional anchored local regions,
  split/merge-aware history, reachable interior room-entry goals, partial-room
  revisits, persisted unfinished local routes, and bounded target/recovery
  interrupts. Actual emitted actions are charged; Gibson remains at 500.
- Existing RPT*, classifier, probability oracle and target-evidence methods are
  retained. Classification evidence accumulates without per-frame LLM calls;
  burst reasoning no longer fabricates a partition reset.
- Final discrete-action safety vetoes work through direct and recording-proxy
  agents. Unsafe transit/verification proposals enter bounded recovery rather
  than an immediate turn/undo cycle.
- Recorded coverage/action curves, latency tails, revisit/turn/stagnation data,
  stage allocation, and evaluator-only native collisions/first-region-entry.

## Validation and artifacts

**2,242 tests passed**, including the existing runtime/harness, ObjectNav,
exploration/RPT*, A*, mapping/topology and scene-graph detector suites.
Python 3.8 syntax and ROS/model-free imports were checked for the core port.
Actual Gibson episodes were run end to end, not just a synthetic renderer.
All 1,000 published starts pass preflight; no full held-out performance run was
performed. Final results: frontier 2/2 successes, FALCON 1/2, with the detailed
Corozal improvement and Collierville regression reported without splicing runs.

Final same-source development comparison:
`~/objnav_benchmark/gibson/falcon-dev-20260915/paired-validated/`.
See [measured results](FALCON_RESULTS.md) and
[sourced design/runbook](BOUNDED_FALCON.md) for exact commands, source identity,
metrics, previous attempts and limitations. The first episode in each tested
scene was already inspected; none is described as held out.

## Important limits / next work

This is **not** an unchanged FALCON reproduction, a proven ObjectNav improvement,
or a full frozen validation. Keep the frontier default. Small development gains
in SPL or proxy coverage cannot outweigh a success-rate regression or establish
SOTA. More reliable observed-room entry, partial revisit eligibility and lower
planning tail cost need further independent development evaluation.

Unknown costs exist only in hypothetical coverage guidance. Actual movement
requires observed free space and footprint clearance. Sparse/noisy depth and
room partitions can still reject useful motion. Viewpoint rejection is
geometric and conservative. Global multi-floor memory is not implemented; floor
resets invalidate geometry and can double-count a revisited floor in the area
proxy. Cross-burst route restoration is unit-tested; the small real pair need
not exercise a restored pending route.

Native scipy work is capacity-bounded with cooperative deadline checks, not a
preemptive hard real-time service. Exact DP is only claimed for unpruned tours.
The source repository's missing explicit planner licence is why upstream C++
was not vendored. The full paper text and relevant source were inspected, but
supplementary video/figure images could not be visually verified with the
available tools. No published planar FALCON ObjectNav precedent was verified.

## Workspace notes

No commit, push, destructive reset, environment install, model download or
other simulator merge was performed. `.run/` was pre-existing and is unrelated.
The original CPU detector services on 18092–18094 and CPU Ollama were preserved.
A dedicated X-v2 service on 18095 was used for comparison; only session-owned
processes may be stopped. That owned service was stopped after validation; the
pre-existing services remain running. Changes are unstaged for review, and
the branch tip is unchanged. Results and test logs are outside the repository.

The IDE occasionally supplied a stale partial-buffer file view. Affected files
were reconstructed coherently and validation used actual saved files in fresh
Python processes. Do not trust an old editor preview over the recorded source
fingerprint and on-disk tests. Interpreter/index warnings about numpy, scipy,
pytest or Habitat are distinct from the validated runtime environments.


