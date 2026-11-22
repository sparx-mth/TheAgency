# frame_log — every RGB frame of an episode, under the pose it was captured from

The method's visual memory, and the first piece of the VLM pipeline analysed in
[`docs/papers/vlm-search-scoring-20261119`](../../../../../docs/papers/vlm-search-scoring-20261119/README.md):
a decision that values places from images -- the past frames that look at a
frontier cluster, handed to one VLM as a batch -- will read those frames from
here. **Today the log only keeps.** Nothing reads it back during an episode
yet, and it changes no decision.

```text
 ObjNavObservation ─► EpisodeFrameLog.add()                  whoever is shown the frame feeds the log
     rgb  ─► FrameCodec.encode ─► bytes ─► FrameStore.put ─┬─► DiskFrameStore    <root>/<episode>/step_000137.jpg + frames.jsonl
     step ─► the frame's name                              └─► MemoryFrameStore  the same bytes in a dict, dropped per episode
     pose ─► FrameRecord ─► records["step_000137.jpg"]         the lookup dictionary: name -> step and pose, always in RAM

 EpisodeFrameLog.records        which frames exist, and where each was taken
 EpisodeFrameLog.rgb(name)      what a frame shows: the same pixels from either store
```

| module | holds |
|---|---|
| `interfaces.py` | the seam: `FrameRecord` (name, step, capture pose), the `FrameStore` protocol, `FrameLogError`, the naming rule (`frame_name`) |
| `codec.py` | `FrameCodec`: an RGB frame to the bytes of an image file and back -- JPEG at quality 95 by default, PNG when it must be lossless |
| `disk_store.py` | `DiskFrameStore` (one folder per episode), `episode_directory_name`, and `read_frame_log` for reading a folder back afterwards |
| `memory_store.py` | `MemoryFrameStore`: one episode's encoded frames in RAM |
| `episode_frame_log.py` | `EpisodeFrameLog`: `begin(episode)`, `add(observation)`, `records`, `rgb(name)`; `from_env()` reads `OBJNAV_FRAME_LOG_DIR` |

## Turning it on

For a benchmark run, name a directory -- one per run; inside the run's own
output directory is fine:

```bash
OBJNAV_FRAME_LOG_DIR=~/objnav_benchmark/frames/<run> python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.run ...
```

The one policy builder (`tasks/planning/objnav_benchmark_runtime/policy_factory.py`)
reads the variable for every dataset entrypoint and hands the log to
`RPTSearchPolicy(..., frame_log=...)`. The variable is inherited by a
campaign's child processes, so it switches the log on for a whole campaign;
each episode has its own folder, so the processes never write one file. The
run's frozen configuration gains `method.frame_log` (where the frames went,
the image format) **only when the variable is set**, so a run recorded
without it still resumes. Only `--agent rpt` keeps a log: with another agent
the variable is refused rather than ignored.

In code, the log is one more thing the policy is built with, beside its
detector and its LLM:

```python
log = EpisodeFrameLog(MemoryFrameStore())            # or DiskFrameStore(root); FrameCodec("png") for lossless
policy = RPTSearchPolicy(detector, llm, settings, frame_log=log)
```

## What a disk store leaves

```text
<root>/
  pLe4wQe7qrG_000000-3e0a50b8038d/     the episode id made readable, then 12 hex digits of its SHA-256
    episode.json                       schema, episode / scene / benchmark / split, the target, the image format, the camera, the pose convention
    frames.jsonl                       one line per frame, in capture order
    step_000000.jpg                    the frame the episode started with
    step_000001.jpg                    the frame after the first action
    ...
```

A line of `frames.jsonl` is the lookup entry of one image -- its file name, the
step and the exact capture pose:

```json
{"name": "step_000242.jpg", "step": 242, "pose": {"x": -1.8153514862060547, "y": 0.8287400007247925, "z": 0.07521793246269226, "yaw": 2.8785765549399125, "camera_pitch": 0.5235988527333357}}
```

The pose is the observation's `AgentPose` as it is: `x, y, z` in metres in
world ENU (`z` is the floor under the agent; the camera sits
`CameraSpec.height_m` above it), `yaw` in radians counter-clockwise from world
`+x`, `camera_pitch` in radians, positive looking down. It is the frame the
occupancy map, the frontiers and the RPT\* instance's node positions are in,
so a place on the map and a frame's pose can be compared directly, and
`episode.json` carries the camera that turns a pose into what the frame sees
(image size, pinhole intrinsics, mount height, depth range). The twelve
hex digits are the ones the runtime's recorder names `recordings/<key>/` with:
an episode's frames and its recording are found by one key.

`read_frame_log(directory)` returns `(header, records)` for anything that
looks at an episode afterwards; an image is then
`FrameCodec().decode((directory / name).read_bytes())` -- or any image viewer.

## The rules

1. **Every frame a decision is taken on, once.** The policy feeds the log at
   the top of each decision, before perception, the map or the search read
   the frame, so whatever consults the log during a decision finds the
   current frame in it. An episode of N actions logs N frames, steps 0 to
   N-1. The observation the last action returns is the one frame no policy
   is ever shown, and the only one not logged.
2. **The pose is the frame's own.** The record is built from the observation
   in the same call that stores its image; there is no second source for
   either.
3. **A step is logged once, in increasing order.** A repeat or an older step
   raises: it would replace an image under a pose that may differ.
4. **It fails loudly, as infrastructure.** `FrameLogError` is an
   `ObjNavInternalError`: a frame that cannot be encoded or stored stops a
   benchmark run instead of being scored as a failed episode of the method,
   and no record exists without its image.
5. **An episode logged again replaces itself.** A run resumed after a crash
   re-runs the episode it died in, so a disk store first removes what the
   earlier attempt left in that episode's folder -- the header, the index,
   files named like frames, a write left unfinished; nothing else, never a
   directory. Finished episodes are not touched. Two runs sharing one root
   overwrite each other's episodes, hence one root per run.
6. **Not a setting of the method.** `RPTSettings` and `configuration()` do not
   know the log; without one the policy behaves, and is recorded, exactly as
   before.

## Swapping the store

A store is handed bytes under a name and asked for the bytes of a name
(`FrameStore`: `begin`, `put`, `get`, `describe`); it never sees a pixel. The
frame is encoded once, above the store, so the disk folder and the RAM
dictionary hold byte-identical images, the lookup dictionary has the same
keys, and `rgb(name)` returns the same array from either. Moving a run from
disk to RAM is the one argument of `EpisodeFrameLog(...)`; another store
(shared memory, a database) is four methods.

Cost, measured on one MP3D episode on 2026-11-22 (243 frames of 640 x 480):
111 kB a frame at the median as JPEG quality 95 and 27 MB for the episode, in
either store; about half at quality 85; about 490 kB a frame as PNG; under
2 ms to encode or decode a frame. Extrapolated from that one episode, a full
split of two thousand episodes on disk is tens of gigabytes, which is what
the memory store is for.

## Not here yet, on purpose

Which frames are worth keeping (the log keeps all of them), which frames
stand for a frontier cluster, the VLM client and its recorded answers, and
depth. Each is a consumer or a filter of this log, not part of it.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core/planning/object_search/frame_log -q
```

The log on its own (`tests/`: the codec, both stores against one contract, the
episode log); the policy feeding it (`../tests/test_policy_frame_log.py`);
and a whole run through the real runner on the harness's fake building, the
policy builder's variable and the failure's attribution
(`tasks/planning/objnav_benchmark_runtime/tests/test_frame_log_run.py`).
