# sjtu_internvla_n1 — InternVLA-N1 flying the SJTU Gazebo drone

InternVLA-N1 controls the SJTU warehouse drone in Gazebo, the **NavDP way**: the
policy answers with a body-frame trajectory, we anchor it in the world and fly it
as a route, and a separate follower tracks it into the drone's one control input.
Nothing here is N1-specific below the wire; swap the policy node for the NavDP one
and the follower would not notice.

**Everything runs on the CPU except the InternVLA-N1 network, which owns the GPU
(~8 GB) alone.** That is the whole design constraint. Gazebo has no GPU physics,
the mapper is Gazebo's own depth, and both ROS2 nodes are pure numpy with
`CUDA_VISIBLE_DEVICES=""`. The only thing on the card is the model server.

## The chain

```
Gazebo — SJTU no_roof_small_warehouse (Docker, ROS2 Humble, domain 20, CPU)
  /simple_drone/front/image_raw        (RGB 600×600)
  /simple_drone/front_depth/.../image_raw (depth 32FC1 m)
  /simple_drone/odom                   (pose + body twist)
        │
        ▼   (host ROS2, CPU, CUDA hidden)
  n1_policy_node ──HTTP──►  InternVLA-N1 server  (GPU, conda `internnav`, ~8 GB)
   • InternVLAN1Policy.step(obs, LanguageGoal) → body-frame trajectory (T,≥2 FLU)
   • PlanCommitExecutor: anchor at the capture pose, commit ~half, re-infer
   • publishes /simple_drone/n1/trajectory   (nav_msgs/Path, world frame)
        │
        ▼   (host ROS2, CPU)
  trajectory_follower_node
   • PurePursuitTracker3D pursues the path → world-frame velocity
   • rotate world→body, clamp with the SJTU velocity adapter
   • publishes /simple_drone/cmd_vel   (geometry_msgs/Twist, body FLU)
```

This is FALCON's `navdp_click_node → path → waypoint_follower_node` split, in
ROS2, with no ROS1 bridge because N1 is already ROS2.

## Where the trajectory actually comes from

InternVLA-N1's System 1 predicts a fan of candidate trajectories per call, which
`core/.../internvla_n1/trt/postprocess.mean_path` (upstream: `vln_utils.py::
traj_to_actions(..., use_discrate_action=False)`) integrates into one body-frame
XY curve — the *exact* shape NavDP emits. **That continuous curve is what this
stack flies.** It is `S1Output.trajectory`, carried to the HTTP response by the
`trajectory` patch in `tasks/planning/vlas/internvla_n1/upstream/` (see the README
there); `InternVLAN1Policy` reads it and prefers it over the discrete action.

Stock InternVLA-N1 discretizes that curve into a single VLN-CE action
(STOP / FORWARD / TURN_LEFT / TURN_RIGHT) and returns only that — which is why
the patch exists. **Apply the upstream patch and restart the model server**, or
the server emits no `trajectory` field and the policy falls back to the discrete
action (below).

On the occasional **pure-S2 step** — an in-place turn or a look-down, where the
model genuinely emits no curve — there is no continuous trajectory, so the policy
renders that one discrete action as a short followable body step
(`geometry.trajectory_from_action`): a forward action advances 0.25 m, a turn
places a waypoint one step ahead bent by 15°. The follower keeps moving and the
next S1 step returns to flying the curve.

## Layering (nothing here breaks it)

| layer | what | where |
|---|---|---|
| policy | wire contract + the body-frame trajectory | `core/planning/vlas/internvla_n1/` (`policy.py`, `geometry.py`, registered `"internvla_n1"`) |
| task | the ROS2 nodes, the follower glue | this package |
| robot | topics, camera intrinsics, actuation limits | `robots/SJTU/` + the binding YAML |

The policy never names SJTU and the SJTU robot layer never names N1; they meet in
`robots/SJTU/config/vla/internvla_n1.yaml`, which both nodes read via their
`config_file` parameter. The follower reuses the platform's own
`robots/SJTU/adapters/velocity_command.py` for the body-twist clamp, so the sign
and saturation logic lives once, next to the drone.

## Running it

```bash
export SJTU_PROJECT_DIR=~/GIT/sjtu_project   # the external sim checkout
export DISPLAY=:1                            # Gazebo Classic needs an X display

# one command: GPU preflight → Gazebo warehouse (CPU) → wait for N1 server (GPU)
#            → the two CPU nodes → takeoff → instruction
sparx_agency/tasks/planning/sjtu_internvla_n1/scripts/run_sjtu_n1.sh \
    no_roof_small_warehouse "go to the far shelves and stop"
```

The script **refuses to start unless the GPU is empty** (`check_gpu_free.py
--require-empty`), gives the card to N1, and pins everything else off it. It does
not vendor the simulator or the model server:

* **Gazebo** comes up via `robots/SJTU/setup/bringup_world.sh` (Docker, CPU). Set
  `START_SIM=0` to manage the world yourself in another terminal.
* **The InternVLA-N1 server** must be runnable on the GPU. The script waits for it
  at `127.0.0.1:8087`; start it yourself (conda env `internnav`) or hand the
  script `N1_SERVER_CMD='<command>'` to start it. `~/GIT/InternNav` is not assumed.

Verify the split at any time:

```bash
python3 sparx_agency/tasks/planning/sjtu_internvla_n1/scripts/check_gpu_free.py \
        --allow internnav --allow python     # only N1 may hold the card
nvidia-smi                                    # everything else is CPU
```

Redirect the drone mid-flight:

```bash
ros2 topic pub --once /simple_drone/navigation/instruction std_msgs/msg/String \
    "{data: 'turn around and go back to the door'}"
```

### Running the nodes alone

With the world and the server already up, and `ROS_DOMAIN_ID`/`RMW_IMPLEMENTATION`
matching the sim (domain 20, `rmw_cyclonedds_cpp`):

```bash
ros2 launch sparx_agency/tasks/planning/sjtu_internvla_n1/launch/sjtu_internvla_n1.launch.py
```

## Recording a run

```bash
# hospital world, the exploration order, recording on. Ctrl-C to stop, or set a duration.
sparx_agency/tasks/planning/sjtu_internvla_n1/scripts/record_run.sh
RECORD_SECONDS=180 sparx_agency/tasks/planning/sjtu_internvla_n1/scripts/record_run.sh \
    hospital "Explore the entire hospital, enter all the rooms, reach every area at least once"
```

It produces two artifacts and prints the measured S1/S2 FPS at the end:

* **an MP4** — a two-panel video: the **drone camera on the left** (with the
  instruction, the action, the System-2 pixel goal and the System-1/System-2 FPS
  drawn on) and **N1's route top-down on the right** (the committed route in
  yellow, the speculative tail in orange, the trail it has flown in green). The
  top-down view is the honest way to show a *drone's* route — an aircraft flies
  at camera height, so a ground path projects to the horizon in first person and
  only reads clearly from above. Written with OpenCV's `mp4v` encoder, so no
  system `ffmpeg` is needed.
* **a rosbag** — every relevant topic, for replay and offline analysis.

The rendering is ROS-free (`recording.py`) and testable; see the format without
any of the stack:

```bash
python -m sparx_agency.tasks.planning.sjtu_internvla_n1.scripts.demo_recording \
    --output /tmp/sjtu_n1/demo.mp4 --seconds 12
```

`record:=true output:=<path>` on the launch adds the recorder to a manual bring-up.

## FPS: System 1 and System 2

The two systems run at very different rates, and the split is the point of the
design — System 1 is the small, fast trajectory policy; System 2 is the 7B VLM.

**Live:** the `trajectory` patch to the InternNav agent also times each system
(`s1_ms`/`s2_ms` in the response). The policy node turns them into a smoothed
rate, publishes them on `/simple_drone/n1/info`, draws them on the recorded
video, and logs a line every 5 s:

```
[n1_policy_node] N1 FPS  System1=22.8 Hz  System2=1.4 Hz  (action=MOVE_FORWARD)
```

**Measured (this machine, `~/trt/internnav/REPORT.md`, RTX 5070 Laptop, sm_120):**

| | before | after (TensorRT S1) |
|---|---:|---:|
| **System 1 alone** | 6.77 Hz (p50 147.6 ms) | **22.99 Hz** (p50 43.4 ms) |
| **Whole dual-system pipeline** | 1.36 Hz (734.5 ms/decision) | **1.41 Hz** (707.9 ms/decision) |

System 2 is **~98.5% of every decision's time** and, because it is autoregressive
(a Qwen2.5-VL-7B behind a KV cache) and 16.6 GB at bf16, it is deliberately *not*
TensorRT-converted — so making System 1 3.4× faster moved the whole pipeline only
1.04× (the Amdahl ceiling with System 1 free is 1.05×). Cadence: System 2 fires
once every `sys2_max_forward_step` (8) System-1 steps, so per control decision
System 1 runs ~0.25× and System 2 ~0.125×.

> The live number reflects whatever the running server actually uses. If it
> serves the torch System 1, expect ~6–7 Hz; the 22.99 Hz is the TensorRT S1 from
> the optimization workspace. Either way the pipeline is System-2-bound at ~1.4 Hz.

## Configuration

One file: `robots/SJTU/config/vla/internvla_n1.yaml`. The knobs that matter:

* `server.host` / `server.port` — where the N1 model server is.
* `camera.*` — the SJTU front pinhole (600×600, fx=fy=390.64). Passed to the
  server so it projects its pixel goal correctly; a wrong intrinsic is this
  platform's most expensive bug.
* `commit.*` — how much of each prediction to fly before re-inferring (NavDP's
  plan-commitment discipline). Kept short for the 0.25 m action-fallback steps.
* `follower.cruise_speed` / `follower.target_altitude_m` / `follower.max_*` — the
  pursuit speed, the altitude held after takeoff, and the SJTU airframe clamps
  (well under its 2 m/s ceiling).

## Topics

| topic | type | dir | note |
|---|---|---|---|
| `/simple_drone/front/image_raw` | `sensor_msgs/Image` | in | RGB 600×600 |
| `/simple_drone/front_depth/depth/image_raw` | `sensor_msgs/Image` | in | 32FC1 metres |
| `/simple_drone/odom` | `nav_msgs/Odometry` | in | pose + body twist (the feedback source) |
| `/simple_drone/navigation/instruction` | `std_msgs/String` | in | the language goal |
| `/simple_drone/n1/trajectory` | `nav_msgs/Path` | out | the committed route (world), what the follower flies |
| `/simple_drone/n1/trajectory_full` | `nav_msgs/Path` | out | the whole prediction, for RViz |
| `/simple_drone/cmd_vel` | `geometry_msgs/Twist` | out | body FLU — the drone's only control input |

## Tests

ROS-free, in the plain `.venv`:

```bash
pytest sparx_agency/core/planning/vlas/internvla_n1 \
       sparx_agency/tasks/planning/sjtu_internvla_n1
```

The policy translation, the trajectory shaping and the path→trajectory timing are
all unit-tested without ROS or Gazebo; the ROS2 nodes are thin wiring over them.

## Known limitations

* **The upstream patch must be applied and the server restarted.** Without it the
  server emits no `trajectory` field and the policy falls back to the coarse
  discrete action — see "Where the trajectory actually comes from" and
  `tasks/planning/vlas/internvla_n1/upstream/README.md`.
* **DDS interop.** The sim is Humble/CycloneDDS on domain 20; the host is Jazzy.
  The nodes are launched with `rmw_cyclonedds_cpp` and domain 20 to match — a
  mismatch drops all traffic silently. `ros-jazzy-rmw-cyclonedds-cpp` must be
  installed on the host.
* **Takeoff.** The run script commands takeoff and holds a cruise altitude; there
  is no landing sequence — stop the nodes and the drone holds its last command
  (the SJTU plugin has no failsafe from outside).
* **No obstacle reflexes.** Unlike `falcon_sjtu`, this stack has no depth brake or
  clearance envelope yet — it flies N1's trajectory as given. Keep `cruise_speed`
  conservative in cluttered worlds.

