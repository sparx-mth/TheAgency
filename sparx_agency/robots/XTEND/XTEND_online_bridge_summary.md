# XTEND Online Control / UI / Stream — Summary README

This README summarizes the XTEND control refactor and the intended runtime architecture.

The main goal was to avoid multiple scripts talking directly to the drone at the same time, and to move toward a clean ROS-based command flow where the UI, planner, and test publishers all send commands into one online bridge.

---

## 1. Final architecture

### One node owns the XTEND WebSocket

Only **one** process should connect to the XTEND WebSocket and send `VIRTUAL_CONTROLLER` commands to the drone.

That node is:

```bash
online_nav_bridge_publisher.py
```

It should be the single owner of:

```text
WebSocket -> XTEND drone command API
```

Do **not** run these together if they all send controller commands:

```text
online_nav_bridge_capture.py
online_nav_bridge_publisher.py
xtend_direct_ui_controller.py
xtend_twist_controller.py
```

Only one of them should control the drone at a time.

---

## 2. ROS command flow

The intended flow is:

```text
UI / planner / script
        |
        v
ROS 2 command topics
        |
        v
online_nav_bridge_publisher.py
        |
        v
XTEND WebSocket VIRTUAL_CONTROLLER
        |
        v
Drone
```

The UI should be a **ROS publisher only**.

The online bridge should be the only node that translates commands into XTEND axes/buttons.

---

## 3. Main topics

### Manual command topic

```text
/drone/cmd_nav
```

Type:

```text
std_msgs/msg/String
```

Payload is JSON.

Examples:

```json
{"action": "arm", "value": 0}
{"action": "takeoff", "value": 0}
{"action": "forward", "value": 400}
{"action": "turn_right", "value": 1000}
{"action": "stop", "value": 0}
{"action": "land", "value": 0}
{"action": "disarm", "value": 0}
```

### Image topic

The online publisher also publishes cropped XTEND FPV frames to:

```text
/xtend/image_raw
```

Type:

```text
sensor_msgs/msg/Image
```

The current crop is:

```text
source: 720x420
crop:   504x280
x offset = 108
y offset = 70
```

---

## 4. Important behavior change: hold-style movement

The old code used duration-style methods from `automation.py`, for example:

```python
await self.move_forward(duration=duration, value=thrust)
```

That pattern means:

```text
send forward for duration
then set axis back to zero
```

This was not what we wanted for online control.

The new bridge behavior should be **hold-style**:

```text
FORWARD command     -> set forward axis and keep sending it
TURN_RIGHT command  -> set yaw axis and keep sending it
STOP command        -> zero all axes
LAND + DISARM         -> stop first, then land + disarm
```

This works because `ControllerAutomation.send_message()` continuously sends the current `self.send_command` at the configured frequency. The command state is not sent only once.

So:

```python
self.send_command["axes"][2] = 400
```

means the bridge keeps sending forward at each WebSocket tick until another command changes it.

---

## 5. Manual UI behavior

The UI should no longer open a WebSocket. It should only publish ROS commands to `/drone/cmd_nav`.

Manual sequence:

```text
ARM
TAKEOFF
set Forward thrust = 500
FORWARD
wait manually
STOP
set Turn thrust = 700 or 1000
TURN RIGHT
wait manually until about 90 degrees
STOP
LAND
DISARM
```

Forward and turning should be separate:

```text
FORWARD -> STOP -> TURN RIGHT -> STOP
```

Do not combine forward and right unless intentionally testing an arc.

---

## 6. UI command schema

The updated UI should send:

```json
{"action": "forward", "value": 500}
```

not:

```json
{"action": "forward", "value": 2000}
```

In the new hold-style bridge:

```text
value = thrust
```

not duration.

Recommended UI buttons:

```text
ARM
TAKEOFF
FORWARD
TURN LEFT
TURN RIGHT
STOP
LAND
DISARM
```

Recommended UI fields:

```text
Forward thrust
Turn thrust
```

The UI may show a local timer, but the authoritative log should be in the bridge.

---

## 7. Action timing logs

We added action timing concepts:

```text
active_action
active_action_start_t
action_log
```

The bridge should log hold-style actions such as:

```text
forward_500
turn_right_700
turn_left_700
```

An action starts when the bridge receives a hold command:

```text
forward
turn_right
turn_left
```

An action ends when the bridge receives:

```text
stop
land
disarm
shutdown
or another hold command
```

Example action summary:

```text
00: forward_500 12.177s reason=stop
01: turn_right_700 1.421s reason=stop
```

The bridge writes:

```text
xtend_actions_YYYYmmdd_HHMMSS.csv
```

Suggested columns:

```text
index
action
start_t
end_t
duration_sec
reason
```

---

## 8. Telemetry logs

The bridge should log telemetry from `ROBOT_STATUS`.

Suggested telemetry CSV:

```text
xtend_telemetry_YYYYmmdd_HHMMSS.csv
```

Suggested columns:

```text
time_sec
iso_time
robot_uid
x
y
z
bearing_raw
active_action
axes_0_lateral
axes_1_vertical
axes_2_forward
axes_3_yaw
axes_4_marker_vertical
```

This lets us compare:

```text
commanded action duration
actual local_telemetry x/y/z movement
bearing change during turns
axes being sent at each telemetry sample
```

This is useful for calibrating:

```text
how long forward_500 moves about 4 meters
how long turn_right_700 turns about 90 degrees
```

---

## 9. Stream / capture

We discussed two options:

### Option A: online bridge publishes image stream

Use:

```bash
online_nav_bridge_publisher.py
```

It publishes cropped FPV frames to:

```text
/xtend/image_raw
```

This is the preferred online ROS style.

### Option B: online bridge saves image files

Use or refactor:

```bash
online_nav_bridge_capture.py
```

It saves JPG + JSON sidecars.

Long term, `capture` and `publisher` should share a base class to avoid duplicated command/telemetry/logging code.

---

## 10. Current duplication problem

`online_nav_bridge_capture.py` and `online_nav_bridge_publisher.py` have duplicated logic:

```text
/drone/cmd_nav subscriber
command queue
hold-style command parsing
set_axes / stop_motion
action timer
telemetry logging
WebSocket send/receive
shutdown cleanup
```

The only real difference should be video output:

```text
capture version   -> saves JPG + JSON
publisher version -> publishes /xtend/image_raw
```

---

## 11. Proposed clean refactor

Create a shared base file:

```bash
sparx_agency/robots/XTEND/xtend_online_bridge_base.py
```

With class:

```python
class OnlineXtendBridgeBase(ControllerAutomation):
```

It should contain:

```text
ROS /drone/cmd_nav subscriber
async command queue
hold-style movement
set_axes()
stop_motion()
start_action_timer()
end_action_timer()
save_action_log_csv()
telemetry CSV logging
receive_message() override with telemetry logging
dynamic_executor()
clean shutdown
```

Then have two small subclasses:

```bash
online_nav_bridge_capture.py
```

```python
class OnlineNavBridgeCapture(OnlineXtendBridgeBase):
    # only implements save-frame capture loop
```

```bash
online_nav_bridge_publisher.py
```

```python
class OnlineNavBridgePublisher(OnlineXtendBridgeBase):
    # only implements /xtend/image_raw publishing loop
```

That removes the duplicated command/control code.

---

## 12. Shutdown behavior

The bridge should have a `try/finally` in `run_bridge()`.

On shutdown:

```text
stop motion
end active action with reason="shutdown"
cancel asyncio tasks
save action CSV
close telemetry CSV
stop RTSP grabber if used
destroy ROS node
rclpy.shutdown()
```

This matters because otherwise the last active command may continue to be sent until the WebSocket task exits.

---

## 13. Run order

### Terminal 1: run the online bridge

Publisher version:

```bash
cd ~/GIT/TheAgency
source /opt/ros/humble/setup.bash
source theagency_venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

python3 sparx_agency/demos/Demo_No4-XTEND_MapRoom/online_nav_bridge_publisher.py
```

### Terminal 2: run the UI

```bash
cd ~/GIT/TheAgency
source /opt/ros/humble/setup.bash
source theagency_venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

python3 sparx_agency/robots/XTEND/ui.py
```

### Terminal 3: verify image stream

```bash
ros2 topic hz /xtend/image_raw
```

Optional:

```bash
ros2 topic echo /xtend/image_raw --once
```

---

## 14. Example CLI commands without UI

Forward hold:

```bash
ros2 topic pub --once /drone/cmd_nav std_msgs/msg/String \
"{data: '{"action":"forward", "value":500}'}"
```

Stop:

```bash
ros2 topic pub --once /drone/cmd_nav std_msgs/msg/String \
"{data: '{"action":"stop", "value":0}'}"
```

Turn right:

```bash
ros2 topic pub --once /drone/cmd_nav std_msgs/msg/String \
"{data: '{"action":"turn_right", "value":700}'}"
```

Land:

```bash
ros2 topic pub --once /drone/cmd_nav std_msgs/msg/String \
"{data: '{"action":"land", "value":0}'}"
```

Disarm:

```bash
ros2 topic pub --once /drone/cmd_nav std_msgs/msg/String \
"{data: '{"action":"disarm", "value":0}'}"
```

---

## 15. Planner integration

The planner can publish to either:

```text
/drone/cmd_nav
```

for discrete commands like:

```json
{"action": "forward", "value": 500}
{"action": "stop"}
{"action": "turn_right", "value": 700}
```

or later:

```text
/cmd_vel
```

for continuous `geometry_msgs/Twist`.

For now, since the UI and manual commands are command-style, use `/drone/cmd_nav` as the shared command interface.

Later we can extend the same online bridge to also subscribe to `/cmd_vel`.

---

## 16. Safety rules

- Only one WebSocket command bridge should run at a time.
- Do not run direct UI and online bridge together.
- Start with low thrust values.
- Use STOP before LAND when testing manually.
- Do not change `automation.py` unless needed; keep it as the known-working low-level API.
- Put shared behavior into `xtend_online_bridge_base.py` instead of duplicating code between capture and publisher.
