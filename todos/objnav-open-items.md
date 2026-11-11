# ObjectNav open items (2026-10-19)

Nothing below is decided. Discuss with the team first. Evidence paths are under `~/objnav_benchmark/`.

## Decided, done (commit 2564d0c2)
- `nearest_frontier` visit tries the next frontier and releases `route_failed`, not `exhausted`.
- Verification refuses a LOOK that reverses the last one more than `max_tilt_reversals` (1).
- Neither changed any of the 15 Gibson episodes except Darden/000000 (`val_nf` vs `val_nf2`).

## To decide
1. **Escape mode cannot retrace the agent's own path** (HM3D train, `hm3d_train2`, XYyR54sxe6b/038807).
   Boxed in for 415 of 500 steps; the way back is blocked on the inflated map. Idea: treat cells the agent
   already stood on as traversable in `escape`. Needs a failing test first.
2. **Mapped walls too thick.** On the same path the clearance to mapped obstacles is 0.10-0.14 m, below the
   0.18 m body radius. Cause unknown (depth noise, door/cabinet?). Check the map at step 85, not the final one.
3. **High-confidence false positives accepted** (Gibson `val_nf2` Darden/000000, steps 131-133: "bed" with
   `radius_m` 0.15-0.17, box 220 px wide, STOP at 2.31 m). A height rule was rejected (low seats, carpet).
   A per-class minimum radius as a `suspect` penalty (bed/sofa, about 0.25 m under 1.2 m range) separated 4 real
   beds (0.30-1.00) from the false one on this tiny sample; not approved, not implemented.
4. **Room-visit budget counts only SEARCH actions** (`RoomSearchLoop.charge`). Transit, peeks and recovery are not
   charged, so a visit can outlast `explore_visit_steps` (60). Matches "stuck in a room too long"; not verified.
5. **Revisits still do a 360 scan** under `nearest_frontier` (option 1 chosen for now). Revisit data after a run.
6. **Gibson/Collierville/000000 detours:** a 50-step transit moved 0.3 m (goal shifted three times) and 99 steps of
   peeks. Read the loop event log (`oracle`, `semantic_replan`) to see why the goal moved.
7. **Collierville R0 labelled "bathroom" at 0.80** while holding sofa, desk, plants. Does it skew the visit order?
8. **Warm-up repeats on each new storey**, not only at t=0. Pseudocode shows t=0 only; second pass is deliberately
   NOT added to the pseudocode.

## HM3D (code written, not committed)
- New: `objnav_benchmark_runtime/hm3d/` (protocol, dataset, env, run), `tests/test_hm3d.py` (7 pass),
  `HabitatRGBDSimulator.geodesic_distance`. Success = STOP within 0.1 m of a goal view point (habitat-lab config;
  the STOP requirement is from memory of `Success`, not read).
- Develop on **train** only; keep HM3D val for the final report. 8 train scenes already extracted
  (`~/datasets/scene_datasets_train_explore/hm3d/train`): HkseAnWCgqk XYyR54sxe6b gmuS7Wgsbrx ixTj1aTMup2
  oEPjPNSPmzL qk9eeNeR4vw vLpv2VX547B zepmXAdrpjR.
- First run (3 episodes, XYyR54sxe6b): toilet DTG 0.15 m (SR 0, threshold 0.1), plant stuck (item 1),
  plant 040422 STOP at step 6 from 5.7 m (**not looked at yet**).
- Unverified: scoring vs a published number; episodes name `hm3d_v0.2`, local folders are unversioned.

## Housekeeping
- Full Gibson val run: about 51 h on this laptop (1.28 s/step, 15-episode sample); use `launch_detached.sh`
  with its supervisor, after the open items above. Nadav's val result (explore visit, SR 0.792, SPL 0.464) is the baseline.
- `test_monitor` fails without our changes; the other 13 failures seen while the run was active were GPU-busy.
- Ollama's runner uses the GPU (2.3 GB) although `CUDA_VISIBLE_DEVICES=-1` was set; `OLLAMA_NUM_GPU=0` untested.
- Untracked, not ours: `.agent/`, `src/diffusion-policy/`.
