# Rooster R1 / Sphera bindings for the VLA policies

Every file here maps one policy in `core/planning/vlas/` onto **this robot**:
which topics carry its RGB / goal / feedback, and how its output becomes Rooster
`ManualControl` axes. Nothing here is policy logic, and nothing in
`core/planning/vlas/` names a robot — that separation is the whole point.

| file | policy | goal modality | notes |
|---|---|---|---|
| `rooster_r1_internnav.yaml` | InternVLA-N1 | language | discrete VLN-CE action → `action_mapping` table |
| `rooster_r1_omnivla.yaml`   | OmniVLA      | language / pose / image | `z` axis is **tilt**, `-1000` brakes |
| `rooster_r1_nomad.yaml`     | NoMaD        | (exploration) | `z` axis is **thrust**: cruise 400 / turn 300 |

All three run the drone in **GROUND_ROLL** (`requested_flight_mode: 1`) — it
drives rather than flies. The shared actuation transport (arming, the KeepAlive
heartbeat, the hold-then-stop `ManualControl` latch) lives once in
[`../../adapters/rooster_manual_control.py`](../../adapters/rooster_manual_control.py);
only the axis *meaning* differs per policy, which is why the scale tables are
here rather than in the adapter.

## Running a policy on this robot

```bash
# inside the Sphera container, with the model server already up
python3 -m internnav_bridge.bridge_node \
    --config <repo>/sparx_agency/robots/ROBOTICAN/config/vla/rooster_r1_internnav.yaml
```

The bridge nodes themselves live with their policy, under
`tasks/planning/vlas/<policy>/ros2/`.

## Adding another robot

Copy the closest YAML here into `robots/<PLATFORM>/config/vla/`, retarget the
topics, and — only if the actuation protocol differs from Rooster `ManualControl`
— add a `<platform>_<protocol>.py` adapter next to
`robots/ROBOTICAN/adapters/rooster_manual_control.py`. The policy package is not
touched.

## Known issues in these configs (pre-existing, carried over verbatim)

* `outputs.state`, `outputs.waypoint.enabled`,
  `outputs.keep_alive.requested_flight_mode` and
  `outputs.keep_alive.command_reboot` are **read by nothing** — the code
  subscribes `/{id}/fcu/state` and hardcodes the flight mode / reboot flag.
* `rooster_r1_internnav.yaml` disables depth (`inputs.depth.enabled: false`), so
  InternVLA-N1 runs RGB-only with a zeroed depth channel. If you enable it, note
  the upstream agent expects depth **normalised 0..1 over 10 m** while the bridge
  forwards raw metres — that conversion does not exist yet.
* OmniVLA's PD controller hardcodes `DT = 1/3 s`, which must match
  `bridge.inference_rate: 3.0`. Nothing enforces the pairing; changing the rate
  silently mis-scales every velocity.
