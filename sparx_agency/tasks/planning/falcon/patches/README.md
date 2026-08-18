# FALCON source patches (Rooster/Sphera)

Upstream FALCON (`HKUST-Aerial-Robotics/FALCON`, branch `ros1-noetic`) is cloned
and compiled into the `falcon-ros:noetic`/`falcon-ros:jetson` image by
**this repo's own `../Dockerfile`** — unlike the `falcon_sjtu` deployment,
this build IS reproducible from source: every fix below is either a
`COPY`+`RUN`-applied `.sh` script or a `git apply`-applied `.patch` file
wired directly into the Dockerfile, so `docker build` alone reproduces the
whole set. There is no docker-commit lineage to lose here.

## Applied as `git apply` patches (ported from `falcon_sjtu/patches/`, 2026-08-17)

Applied in step "6-sparx" of the Dockerfile, immediately after the FALCON
clone and before any sed-based fix below — each is verified with a grep for
a sentinel string, and the build fails rather than shipping an unpatched
planner if any patch doesn't apply clean.

- **`falcon_hgrid_clamp.patch`** — clamps `UniformGrid` grid-cell lookups
  into the box before indexing (the aircraft sits below the box floor on
  every takeoff, and an unclamped negative/out-of-range index segfaults or
  bad_allocs the planner at mission start), and floors every A*/BFS
  cost-matrix edge so the FATAL `CHECK_GT(cost, 1e-4)` can't abort on a
  drone-sitting-on-a-grid-center edge case. **Supersedes and replaces**
  `fix_falcon_cost_check.sh` (removed from the Dockerfile) — it carries the
  same cost floors plus the grid-index clamp that script never had.
- **`falcon_slow_traj_rescale.patch`** — fixes the FSM's "trajectory
  averaging under 0.5 m/s gets rescaled to 2.0 m/s" logic. Two real bugs:
  (1) a non-turning leg has `avg_yaw_vel == 0`, so the original
  `1.57/avg_yaw_vel` is an infinity that `std::min` does not propagate
  correctly, producing a non-finite rescale ratio that later **aborts
  `exploration_node` from inside a visualization call**, wiping the
  (non-decaying) voxel map; (2) even when finite, the hardcoded 2.0 m/s/1.57
  rad/s targets are a 4-13x *compression* for an aircraft actually
  configured to cruise under 0.5 m/s — which Rooster is
  (`explore_fixed_vx=0.2`, `explore_fixed_vy=0.1` in `nav_stack.launch`,
  both under the 0.5 m/s trigger). Every Rooster exploration trajectory was
  hitting this rescale before this patch. Targets/trigger/clamp are now
  `ros::param`-tunable (`/fsm/slow_traj_target_vel`, `_target_yaw`,
  `_trigger_vel`, `_ratio_min`, `_ratio_max`).
- **`falcon_replan_from_pose.patch`** — FALCON computes each replan's start
  state from its own previous trajectory (evaluated `replan_duration_`
  ahead, clamped to the trajectory's end), never from odometry. Once the
  aircraft stops tracking that curve (our depth brakes/retreats/map-freeze
  gates make this common), every later plan starts from a "ghost" position
  that never returns — a compounding tracking-lag bug, not something the
  follower/velocity-servo layer downstream can correct on its own. Falls
  back to the real measured pose once predicted-vs-actual drift exceeds
  `/fsm/replan_from_pose_drift` (default 1.5 m).
- **`falcon_raycast_out_of_map.patch`** — the TSDF DDA raycaster's stepped
  voxel index can land one step past the map edge even though the ray's
  endpoints were clamped in metres; neither the TSDF nor occupancy voxel
  writer bounds-checks the address it's given (one writes the array
  directly, the other CHECK-aborts). Skips the voxel instead.

See `sparx_agency/tasks/planning/falcon_sjtu/patches/README.md` for the full
derivation, measurements and exact crash logs behind each of the four fixes
above — they were proven live on the SJTU deployment before being ported
here unmodified.

## Applied as `.sh` scripts (pre-existing, applied at their own Dockerfile steps)

- `fix_falcon_system_info.sh` — replaces `printSystemInfo()` so a missing/
  non-numeric `nvidia-smi` output can't `std::stol()`-abort the node at
  startup before ROS logging exists.
- `fix_falcon_visgrid_cadence.sh` — drops free/unknown occupancy-grid
  visualization publishes to every 10th cycle (was every 0.5s, cost scales
  with explored volume) so mapping progress doesn't slow the whole mission.
- `fix_falcon_depth_overflow.sh` — fixes a floor-vs-ceil sizing mismatch in
  `MapServer::depthToPointcloud` that heap-corrupts when image dimensions
  aren't divisible by the decimation skip.
- `fix_falcon_map_resolution.sh` — fixes the misspelled `resolutionf_fine`
  rosparam key (fine voxel resolution was unsettable), the box-volume-based
  resolution choice that can silently 8x memory below a 4000 m³ threshold,
  and makes the chosen grid size log unconditionally.
- `fix_falcon_fsm_logspam.sh` — demotes two per-callback (~90-130Hz)
  `LOG(INFO)` lines in `exploration_fsm.cpp` to `VLOG(99)` so they don't
  bury every other log once `GLOG_logtostderr=1` is set. Targets specific
  log-line markers via regex, not line numbers — does not overlap with
  either `exploration_fsm.cpp` patch above (those touch the replan-start
  and slow-trajectory-rescale blocks, not the four per-callback markers).
- `fix_falcon_so3_gencfg.sh` — CMake configure-time fix for
  `so3_disturbance_generator`'s missing `dynamic_reconfigure` dependency
  (WITH_SIM=1/x86 builds only).
- `ignore_cuda_pkgs.sh` — `CATKIN_IGNORE`s CUDA/sim-only packages on the
  Jetson (`WITH_SIM=0`) build.
- `fix_falcon_sop.sh` — raises the SOP/TSP solver's hardcoded
  `CHECK_LE(sop_time, 1.0)` fatal timeout to 10s; applied post-`catkin_make`
  (step 8b) since it self-skips if already applied and only needs an
  incremental rebuild.

## `fix_falcon_frontier_visibility.sh` — the frontier tests that ended a healthy run

Applied as a script, not a `.patch`, and that is deliberate. It carries the
behaviour of falcon_sjtu's `falcon_visib_unknown_tolerance` +
`falcon_open_visib_bar` plus a fix of our own to the amnesty call site.

Measured live 2026-08-18 in `sphera_jail`: 372 s into a run with 0.02–0.10 m
tracking error, zero collisions and a still-growing voxel map, the frontier set
collapsed to one cluster with twelve retired behind it and the FSM declared
`Finish exploration: No frontier detected`. Three tests did the retiring:

1. `countVisibleCells()` treats `UNKNOWN` as an occluder, but a frontier cell IS
   the boundary of unknown space, so a ray reaching one crosses that boundary by
   construction. Now a bounded prefix of unobserved voxels may be crossed
   (`/frontier_finder/visib_unknown_tolerance`, 2); `OCCUPIED` still blocks at
   any distance, which keeps sealed cavities unviewable.
2. `min_visib_num` is absolute, so it bites hardest on the smallest clusters —
   what is left late in a mission. A cluster-relative bar now applies, but only
   to viewpoints that are NOT `isNearOccupied`
   (`open_visib_fraction` 0.5, `open_visib_floor` 4). The near-occupied
   exclusion is load-bearing: relaxing globally regressed falcon_sjtu's
   warehouse into a pinned abort by admitting frontiers bounding sealed crate
   interiors.
3. `grantFinishAmnesty()` was already ported and already wired, but is checked
   BEFORE categorisation and guarded on `frontiers_` and `tmp_frontiers_` both
   being empty — so it misses the case that actually ends missions, where the
   last cluster is in `tmp_frontiers_` and goes dormant during categorisation.
   A second categorisation pass now re-checks it afterwards, bounded by
   `finish_amnesty_max_` as before.

All three default to upstream behaviour in the source; `nav_stack.launch` sets
the params that switch them on. The script is idempotent (it no-ops if
`visib_unknown_tolerance` is already present) and fails loudly if any anchor
does not match exactly once.

## Not ported from `falcon_sjtu`, and why

`falcon_vp_audit`, `falcon_visib_unknown_tolerance`, `falcon_open_visib_bar`,
`falcon_finish_amnesty_gate` — all four target the same regions of
`frontier_finder.cpp` that our own ports (`falcon_deadend_guard`,
`falcon_blocked_region_ttl`, `falcon_blocked_region_widen`,
`falcon_publish_fail_blacklist`) already rewrote, and fail `git apply` and
`git apply --3way` in **any** order. Verified against the live tree
2026-08-18: each fails at a different hunk of the same file. The two that
matter for exploration coverage are carried by
`fix_falcon_frontier_visibility.sh` above instead; `falcon_vp_audit` is a
diagnostic counter set we do without.

`fix_falcon_sop.sh` is superseded by `falcon_deadend_guard` in falcon_sjtu but
NOT applied that way here — Rooster's simpler `fix_falcon_sop.sh` sed is still
in place and untouched by the port.

Still unported: `falcon_simtime`, `falcon_sjtu_session` (session/bookkeeping,
not behaviour). Also not ported: `falcon_keep_out_boxes`/`falcon_astar_inflate`,
which exist only inside falcon_sjtu's `falcon-ros-custom:v1` docker-commit image
with no `.patch` file at all — see `[[project_falcon_patch_porting_gap]]` (repo
memory) before assuming that image is disposable.
