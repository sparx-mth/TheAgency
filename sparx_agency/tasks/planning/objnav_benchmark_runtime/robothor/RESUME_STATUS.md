# RoboTHOR status — 2026-09-15

Branch `claude/zson-robothore-benchmark-380820`, cut from the infrastructure
tip `60d91714`. The Gibson adapter was the template; this is the first adapter
to drive the shared `run_evaluation` rather than its own loop.

## What is done

The adapter is complete and runs end to end against the **real** reference
simulator on the **real** published episodes. No published-split result has
been produced, and none should be quoted from anything below.

- **1,660 tests pass** across the runtime, the shared ObjectNav layer and
  harness, object mapping, topology, A\* and planner geometry. None of them
  needs ai2thor, a GPU or the dataset.
- All **1,800 published validation episodes load and validate**, including the
  check that every shipped `shortest_path_length` equals the length of its own
  corners. Counts confirm the published shape exactly: 15 scenes × 120, 12
  categories × 150, and **108 episodes (6.0%) with `shortest_path_length == 0`**.
- A **live build check** (`thor.smoke`) passes with zero problems.
- A **three-episode plumbing run** and a **one-episode run of the real method**
  both completed, scored, and wrote a report and a comparison table.

## What the live build confirmed, and what it corrected

`thor/smoke.py` measures what could otherwise only be read about. Against the
pinned build `bad5bc2b…` driven by ai2thor 5.0.0:

| measured | result |
|---|---|
| All six actions vs `check_motion` | **ok** — the ENU conversion is right against the real simulator, which is the only place a mirrored or transposed frame can be seen |
| Horizon clamp | **±30.0°** — the pinned build's, not ai2thor 5.0.0's −30/+60. Confirms which build is loaded |
| Depth scale | **1.003 m per unit** over six measured moves — the modern client decodes the 2021 build's three-channel packed depth correctly |
| Camera height | **0.8688 m**, and this **corrected the protocol**: the declared value had been 0.901 − 0.0312 when it should be `origin_height_m` − 0.0312. A 1 mm error the check caught |
| Far sentinel | Pixels beyond `max_depth_m` clip to `+inf` (9% of a frame looking down a corridor) |

The live `GetShortestPath` also reproduced a published episode's
`shortest_path_length` **bit for bit** (3.9881372734590386), which is the
strongest available evidence that the installed build is the one the episodes
were generated on.

## The finding that matters most, before any sweep

The one real-method episode ran 500 steps with **zero `LOOK_UP`/`LOOK_DOWN` and
zero `STOP`**, 263 `TURN_LEFT` against 61 `TURN_RIGHT`, and ended 1.25 m from
the goal.

The cause is specific and confirmed by reading the code:
**`methods/rpt_policy.py` never sets `NavigationCommand.camera_pitch`** — the
string "pitch" does not occur in it. The action converter therefore never emits
a LOOK, and the camera stays wherever the episode started.

On Gibson that was harmless twice over: its action spec has no LOOK actions at
all, and Habitat's episodes start with a level camera. **On RoboTHOR every
published episode starts at `initial_horizon = 30` — fully pitched down at the
clamp.** So the method currently searches for AlarmClocks, Mugs, Apples and
SprayBottles, which sit on tables and counters, with the camera aimed at the
floor one to two metres ahead, for all 500 steps.

This is a method change, not an adapter bug, so it has not been made here:
`rpt_policy` is shared with the Gibson branch, and a pitch command would be
refused outright by a spec without LOOK actions. The fix belongs in the policy,
gated on `episode.action_spec.has_camera_tilt`, and recorded in
`configuration()` so a run says whether it was active.

**Do not spend a 1,800-episode sweep before deciding this.** The zero-STOP
behaviour alone guarantees 0% success under RoboTHOR's STOP-required rule.

## Second-order concerns, in order

1. **Turn bias.** 263 left versus 61 right turns suggests the search is
   spinning rather than sweeping. Measure it over more episodes before tuning.
2. **Detection of small objects.** RoboTHOR's goals are small where Gibson's
   were furniture. `AlarmClock` accepts `"clock"` and `BasketBall` accepts
   `"sports ball"` because without them those 300 episodes have no recall;
   both are a false-positive surface and a false accept is a false STOP.
3. **Runtime.** One 500-step episode took **377 s** with the CPU detector. At
   that rate 1,800 episodes is roughly 190 hours single-process; shard it
   (`--shards/--shard-index`) or shorten it, and budget accordingly.
4. **`map_size_m`** is 30 here rather than the 80 m default, and
   `DEFAULT_SEGMENTATION`'s room sizes were chosen for a hospital, not a
   three-room apartment. `DEFAULT_SEGMENTATION` is not injectable through
   `RPTSearchPolicy.__init__`, so changing it needs a shared-code change.

## Machine state

- New conda env **`ai2thor`**: Python 3.10, ai2thor 5.0.0, numpy 1.26.4,
  opencv-python-headless 4.11.0.86 (5.x needs numpy≥2), scipy, scikit-image,
  scikit-fmm, networkx. `pip check` is clean. **Nothing was installed into
  `habitat`, `navdp` or the repo venv.**
- Unity build `thor-Linux64-bad5bc2b…` in `~/.ai2thor/releases/` (1.3 GiB
  extracted). SHA-256 verified against the publisher's `.sha256`.
- Episodes in `~/datasets/objectnav/robothor/{train,val,test}/episodes/`.
- A **third** CPU detector service was started on port **18094** with this
  benchmark's vocabulary (ports 18092 and 18093 still carry Gibson's and were
  left untouched). Stop it with `pkill -f "port 18094"` when done.
- Results: `~/objnav_benchmark/robothor/{plumbing,first-episode}/`. Both are
  development runs, not benchmark results.

## Running it

See [README.md](README.md). The environment variables that matter, and that the
preflight refuses to run without:

```bash
export DISPLAY=:1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
export ROBOTHOR_EPISODES_DIR="$HOME/datasets/objectnav/robothor"
export LLM_BACKEND=ollama LLM_BASE_URL=http://127.0.0.1:11434 LLM_MODEL=qwen2.5:3b-instruct
```

Without the two `__NV_*` variables the run renders on the integrated GPU and
nothing says so — which is why the preflight refuses to proceed.
