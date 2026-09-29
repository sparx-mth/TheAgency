# Handoff — HM3D ObjectNav benchmark, 2026-09-29

Branch: `feat/objnav-habitat-hm3d-daphna` (tracks `origin/feat/objnav-habitat-hm3d-nadav`).
Everything below is committed on this branch unless marked otherwise.

## The actual goal

Moshe (email, "Research alternatives" thread) asked for a larger, more diverse
ObjectNav evaluation than the 15-building Gibson pilot before drawing
conclusions. Nadav's `objnav_benchmark_runtime/hm3d/` package is that harness —
your team's real method (`sparx-rpt-llm-host-sweep`: scene graph → LLM room
prior → RPT\* room order → weighted A\*), run on HM3D-v2 (36 scenes, 1000
episodes). This session got that harness fully working end to end, found and
fixed two crash bugs, and is mid-way through a third, more interesting one
(an exploration bug, not a crash).

Kiril's later email proposes connecting RPT\*'s problem to a classical OR
literature ("expanding search" / Alpern & Lidbetter 2013, and a 2026 follow-up
on search with unreliable predictions) as a sharper novelty angle than "a
better planner" — worth a read if picking the research-framing thread back up
rather than the engineering one.

## Environment (all local to this machine, not portable)

- `hm3d_bench` → renamed to **`objnav-habitat`** conda env (python 3.9,
  `habitat-sim==0.2.4` — required exactly, not 0.2.1).
- **`detector_bench`** conda env — YOLO-World-X (`yolov8x-worldv2.pt` in
  `~/models/objnav/`), GPU (`cuda:0`), served on `127.0.0.1:18092`.
- **Ollama**, dedicated instance on port **11435** (not the system default
  11434), model `qwen2.5:3b-instruct`, models dir `~/models/objnav/ollama`.
- Data: `~/datasets/objectnav/hm3d/v2/val/` (episodes) +
  `~/datasets/scene_datasets/hm3d_v0.2/val/` (scenes, 36/36 present).
  v1 data was also set up earlier (`hm3d/val`, `objectnav/hm3d/v1`) but not
  used since — v2 is the active split.
- Standard run env vars:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate objnav-habitat
  cd ~/GIT/TheAgency
  export LLM_BACKEND=ollama LLM_BASE_URL=http://127.0.0.1:11435 LLM_MODEL=qwen2.5:3b-instruct LLM_TIMEOUT_S=120
  export HM3D_EPISODES_DIR="$HOME/datasets/objectnav/hm3d/v2/val"
  export HM3D_SCENES_DIR="$HOME/datasets/scene_datasets"
  ```
  Detector and Ollama must already be running in their own terminals before
  any `hm3d.run`/`hm3d.campaign` invocation. `--allow-shared-gpu` is required
  (desktop compositor overhead alone exceeds the harness's 512MiB gate).

## Fixes made this session (all committed)

1. **`b59dbffc`** — `reset()` teleported to the *raw* published height with no
   navmesh check; the first real action of an episode was the first time
   habitat-sim's native collision test reconciled that height against the
   recomputed (`navmesh="agent"`) navmesh, and the harness's kinematics
   checker read that reconciliation as an illegal move during a turn.
   Fixed by having `reset()` ask `pathfinder.snap_point()` for the real
   height and place the agent there directly — one working height per
   episode, decided once (Nadav's framing: "2.5-D").

2. **`7e081c2b`** — Same class of bug, different action: `MOVE_FORWARD`'s
   climb tolerance assumes at most a 45° slope; short real stairs (2-3 steps
   to a sunken room) can exceed that on one action, and never register as a
   ground-truth stair connector at all (`floor_levels()` only recognizes
   storeys ≥1.5m apart, so a ~0.4m level change gets absorbed into the main
   floor's bucket). Fixed by checking, after every step, whether the landing
   pose is confirmed on the real navmesh (`snap_point` again) — if so, trust
   it over the naive slope-cap tolerance. Required also patching
   `RecordingEnv` (the `--record` wrapper), which doesn't forward arbitrary
   new env methods — the first attempt at this fix silently did nothing
   under `--record`, which is how every run in this session was invoked.

3. **`ec88f935`** — **Partial**, see below.

Both scenes that crashed on these bugs (`00810-CrMo8WxCyVb`, `00800-TEEsavR23oF`,
`00827-BAbdmeyTvMZ`, `00869-MHPLjHsuG27`) now run clean; the always-good scene
(`00877-4ok3usBNeis`) reproduces bit-identical results, confirming neither fix
changes actual search behavior.

## In progress: the exploration-freeze bug (NOT solved)

Found by inspecting a recording the user flagged by hand:
`stairfix_test_00800_v2/recordings/d6a925c60b5b` (episode
`00800-TEEsavR23oF/000000`). The agent froze — zero net position change —
for **373 of the episode's 500 steps** (plus an earlier 82-step freeze). This
single failure mode plausibly explains a large share of the low SPL/high
step-count numbers seen everywhere this session (the campaign's aggregate
`mean_steps` was 372.7 — suspiciously close to 373).

**Two distinct sub-mechanisms found inside that one freeze, confirmed by
reading `steps.jsonl`'s `command.info`/`decision.info` fields directly:**

**(a) Turn oscillation, steps ~134-153.** Two frontier candidates roughly
symmetric on either side of the agent scored as tied; the goal flip-flopped
between them every tick (`route_replaced: "goal_changed"` every step,
`heading_error_rad` alternating sign), so the agent kept re-turning to face
whichever won that tick's ranking, never finishing a turn, never starting to
walk. Root cause: `CommittedRoute.clear()` resets its 30-step no-progress
watchdog (`_motion_xy = None`) for `"goal_changed"`, but *not* for
`route_obstructed`/`off_route`/`forward_blocked` — so a goal that changes
every tick resets the very watchdog meant to catch "stuck", every tick,
before it can ever reach 30. **Fixed in `ec88f935`** — added `"goal_changed"`
to the exclusion list. Verified in isolation (`_motion_xy` now survives a
`clear("goal_changed")` call) and confirmed this specific 20-step
oscillation window no longer reproduces in principle — but see below.

**(b) A second, larger freeze, steps ~156-499 — UNFIXED, dominates the
episode.** `route=committed_safe_route` → immediately
`replaced=forward_blocked` on the very next tick, repeating for the rest of
the episode. `action=MOVE_FORWARD` every tick, `distance_to_goal_m`
completely static (e.g. exactly `4.681` for 5+ consecutive logged ticks).
Re-ran the exact same episode after the (a) fix: **byte-identical freeze
windows, zero change** — (a)'s fix is correct but doesn't touch this second
mechanism, which is doing essentially all of the damage (343 of the 373
frozen steps).

**Where I got to on root-causing (b), not yet confirmed:**
- `notify_blocked()` (`rpt_policy.py:113`) fires on a collided `MOVE_FORWARD`,
  calls `route_memory.clear("forward_blocked")` (already excluded from
  resetting `_motion_xy`, both before and after this session's fix) **and
  separately** does `self._route, self._goal = None, None` directly —
  wiping the policy's own goal state outside of `route_memory` entirely.
- `reusable()` (the only place the no-progress timeout is actually checked)
  **is** called every tick via `_navigate()` (`rpt_policy.py:234`), with
  whatever `goal` was just selected — so it's not simply "never called",
  contradicting my first hypothesis.
- Not yet checked: whether `_motion_step` (as opposed to `_motion_xy`) is
  being reset by something else in this specific path — e.g. `_reset_floor()`
  calls `route_memory.clear("floor_reset")`, and `"floor_reset"` is *not* in
  the exclusion list. Step 154 of this exact freeze shows
  `replaced="ground-truth floor decision"` right before this second freeze
  begins — i.e. the multi-floor coordinator did something right at the
  boundary between sub-freeze (a) and (b). Worth checking whether the
  agent is stuck trying to approach/traverse a stair connector it can't
  actually reach, and whether floor-related clears are repeatedly re-arming
  (or re-wiping) state in a way that keeps defeating the watchdog.
- **Next concrete step**: instrument or re-read `notify_blocked()`'s full
  body and whatever calls `_navigate()` around a blocked step, to see
  exactly what `goal`/`kind` gets passed to `reusable()` on the tick right
  after a block — confirm whether `_motion_step` truly survives, and if not,
  find what's resetting it.

## Background runs

Nothing is currently running — the last background campaign
(`deep_5scenes_fixed`, 5 scenes × ~28 episodes) was stopped deliberately to
prioritize this bug. Partial results exist at
`~/objnav_benchmark/hm3d_v2/deep_5scenes_fixed/` for whichever scenes
finished before it was stopped (check for `index.html` per scene folder).
**Do not resume it as-is** — it predates fix (a) and entirely lacks a fix for
(b), so its numbers aren't representative of the current code.

## Recommended order of work next session

1. Finish root-causing (b) — the bigger freeze — using the concrete next
   step above.
2. Re-verify both (a) and (b) are fixed on `00800-TEEsavR23oF/000000`
   specifically (the reproduction case), then spot-check 2-3 other scenes.
3. Only then re-run the deep 5-scene (or full 36-scene) campaign — numbers
   before (b) is fixed aren't worth collecting.
4. Consider whether this exploration-freeze class of bug is worth reporting
   to Nadav now (it's shared `methods/` code, used by Gibson too) even before
   fully solved — the turn-oscillation half alone (fix `ec88f935`) is a real,
   self-contained, already-fixed defect worth landing independently of (b).
