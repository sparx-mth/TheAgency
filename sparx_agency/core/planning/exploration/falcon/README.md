# Bounded FALCON planar adaptation
# Bounded FALCON planar adaptation

Simulator-neutral host-side exploration algorithms; no ROS, detector, LLM,
Habitat, CUDA or simulator geometry is imported here. The parent exploration
facade deliberately does not import the scipy modules. `params.py` and
`bursts.py` are lightweight; the complete planner requires the existing numpy
and scipy installation. Python syntax is 3.8 compatible, but the legacy Noetic
image is not provisioned with scipy and is not the target runtime.

This independently written port preserves FALCON's connectivity-aware
free/unknown decomposition, portal graph, active-free **plus unknown** coverage
path, precedence-constrained local order and layered viewpoint refinement.
Unknown guidance and executable free-space paths are separate. The ordering
solvers and drone trajectory layer are deliberately replaced, not relabelled.

- `params.py`: validated settings and exact official paper/source revision.
- `frontiers.py`: live boundaries, PCA clusters and camera-frustum viewpoints.
- `connectivity.py`: incremental tile/edge caches and hybrid guidance costs.
- `ordering.py`: open ATSP/SOP DP/beam, explicit deadlines and layered refinement.
- `travel.py`: cached no-corner-cut shortest paths and serial ground-action costs.
- `planner.py`: complete CP/SOP/refinement/free-route planning transaction.
- `bursts.py`: action-clock hierarchy and interruption-safe budget ledger.

The caller supplies only an **observed** occupancy grid, motion cost field,
anchored local scope, measured pose, camera intrinsics/range/pitch and public
forward/turn geometry. It must call `observe` on incoming maps, revalidate a
committed route, and charge actual actions. Physical embodiment and safe scopes
are adapter responsibilities, never inferred from a benchmark name.

The Gibson adapter adds measured-footprint clearance, final-action vetoes,
room entry/revisit history, route pause/resume, accumulated room reasoning and
the existing RPT* and A*/WA* implementations. It does not use this local CP to
order the building's rooms.

See [sourced component mapping and run instructions](../../../../tasks/planning/objnav_benchmark_runtime/gibson/BOUNDED_FALCON.md).
Tests live with the benchmark rig under `objnav_benchmark_runtime/tests/test_falcon*.py`.
The native ROS FALCON deployments are unchanged. No upstream source is vendored;
licensing and exact algorithm differences are documented in the mapping.

