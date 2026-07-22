# XTEND Environment and Run Commands

This README covers the XTEND scripts currently used for video, telemetry, capture, DA3 depth, cropping, and trajectory checks.

Repository root assumed in the commands:

```bash
/home/user1/GIT/TheAgency
```

Package root:

```bash
/home/user1/GIT/TheAgency/sparx_agency
```

---

## 1. Environment

### Option A — venv inside `TheAgency`

```bash
cd /home/user1/GIT/TheAgency
python3 -m venv --system-site-packages theagency_venv
source theagency_venv/bin/activate

python -m pip install --upgrade pip wheel
pip install "setuptools<80" "numpy<2"
```

Run scripts with:

```bash
cd /home/user1/GIT/TheAgency
source theagency_venv/bin/activate
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH
```

### Option B — venv inside `sparx_agency`

You showed this interpreter:

```bash
/home/user1/GIT/TheAgency/sparx_agency/theagency_venv/bin/python3.12
```

If using that venv, still make sure `PYTHONPATH` points to the parent of `sparx_agency`:

```bash
cd /home/user1/GIT/TheAgency
source /home/user1/GIT/TheAgency/sparx_agency/theagency_venv/bin/activate
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH
```

For ROS-dependent scripts:

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH
```

## 1.1 PyCharm environments

Use different PyCharm environment variables on Jetson and PC. They are not interchangeable.

### Jetson remote interpreter — ROS 2 Humble / Python 3.10

Typical remote interpreter:

```text
/home/user/GIT/TheAgency/theagency_venv/bin/python3.10
```

Working directory:

```text
/home/user/GIT/TheAgency
```

Environment variables:

```text
LD_LIBRARY_PATH=/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/aarch64-linux-gnu:/opt/ros/humble/lib:/usr/local/cuda/lib64:
PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/home/user/GIT/TheAgency
PYTHONUNBUFFERED=1
```

Use this environment for:
- live XTEND capture on Jetson
- ROS 2 Humble scripts
- TensorRT / CUDA / PyCUDA / DA3 on Jetson
- RTSP capture from the drone

Note: `/home/user/GIT/TheAgency` is the important package path. You usually do **not** need `/home/user/GIT/TheAgency/sparx_agency` in `PYTHONPATH`, because imports like `import sparx_agency` need the parent folder. Keeping both usually works, but if imports behave strangely, keep only the parent path.

### PC interpreter — ROS 2 Jazzy / Python 3.12

Working directory:

```text
/home/user1/GIT/TheAgency
```

Environment variables:

```text
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/jazzy/lib
PYTHONPATH=/usr/lib/python3.12/dist-packages:/opt/ros/jazzy/lib/python3.12/site-packages:/home/user1/GIT/TheAgency
PYTHONUNBUFFERED=1
```

Use this environment for:
- PC-side scripts
- plotting / offline checks
- ROS 2 Jazzy nodes
- analysis tools that do not require Jetson CUDA/TensorRT

Path difference to remember:

```text
Jetson: /home/user/GIT/TheAgency
PC:     /home/user1/GIT/TheAgency
```

For Python 3.12 venvs, keep the matching ROS Jazzy Python 3.12 paths. Do not mix ROS Humble Python 3.10 paths into a Python 3.12 PC interpreter.

---

## 2. Common XTEND values

WebSocket:

```bash
--host 192.0.0.15
--port 8000
```

RTSP:

```bash
--rtsp-uri rtsp://192.0.0.15:8510/active_drone_fpv
```

Robot UID:

```bash
--robot-uid drndfb3eeb1
```

Capture/data root:

```bash
/home/user/jetson-containers/data
```

---

## 3. Script commands

## 3.1 `tasks/mapping/take_xtend_da3_frames.py`

Purpose: capture XTEND RTSP RGB frames and DA3 depth frames **without movement**. It also listens to XTEND telemetry and stores bearing in `metadata.csv`.

It creates:

```text
<output-dir>/xtend_da3_take_<YYYYmmdd_HHMMSS>/
├── rgb/
├── depth_npy/
├── depth_vis/
└── metadata.csv
```

Run with your Python 3.12 venv:

```bash
cd /home/user1/GIT/TheAgency
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH

/home/user1/GIT/TheAgency/sparx_agency/theagency_venv/bin/python3.12   /home/user1/GIT/TheAgency/sparx_agency/tasks/mapping/take_xtend_da3_frames.py
```

Recommended explicit run:

```bash
cd /home/user1/GIT/TheAgency
source /home/user1/GIT/TheAgency/sparx_agency/theagency_venv/bin/activate
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH

python3 sparx_agency/tasks/mapping/take_xtend_da3_frames.py   --rtsp-uri rtsp://192.0.0.15:8510/active_drone_fpv   --xtend-host 192.0.0.15   --xtend-port 8000   --robot-uid drndfb3eeb1   --engine-path /home/user1/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine   --config-yaml /home/user1/depth_anything_ws/src/ros2-depth-anything-v3-trt/camera_info_example.yaml   --output-dir /home/user1/Documents/xtend_da3_takes   --capture-hz 5   --duration-sec 30   --max-depth-m 7.0
```

Useful options:

```bash
--capture-hz 5
--duration-sec 30
--max-frames 100
--xtend-raw-dump-seconds 5
```

---

## 3.2 `demos/Demo_No4-XTEND_MapRoom/online_nav_bridge_capture.py`

Purpose: bridge ROS 2 navigation commands from `/drone/cmd_nav` to XTEND WebSocket commands, while continuously capturing RTSP frames with JSON sidecars.

The ROS command topic is:

```text
/drone/cmd_nav
```

Expected message type:

```text
std_msgs/msg/String
```

Expected JSON payload examples:

```json
{"action": "arm"}
```

```json
{"action": "takeoff"}
```
in ms 4000 = 4 sec
```json
{"action": "forward", "value": 4000}
```

3000 = 3 sec
```json
{"action": "rotate_left", "value": 3000}
```

Run:

```bash
cd /home/user1/GIT/TheAgency
source /opt/ros/humble/setup.bash
source theagency_venv/bin/activate
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH

python3 sparx_agency/demos/Demo_No4_XTEND_MapRoom/online_nav_bridge_capture.py
```

Publish commands:

```bash
ros2 topic pub --once /drone/cmd_nav std_msgs/msg/String "{data: '{"action":"arm"}'}"
ros2 topic pub --once /drone/cmd_nav std_msgs/msg/String "{data: '{"action":"takeoff"}'}"
ros2 topic pub --once /drone/cmd_nav std_msgs/msg/String "{data: '{"action":"rotate_left", "value":3000}'}"
ros2 topic pub --once /drone/cmd_nav std_msgs/msg/String "{data: '{"action":"land"}'}"
```

Default values currently hardcoded in the script:

```text
host=192.0.0.15
port=8000
frequency=30.0
robot_uid=drndfb3eeb1
rtsp_uri=rtsp://192.0.0.15:8510/active_drone_fpv
out_dir=./captures
drone_id=42B
capture_interval_sec=0.5
```

---

## 3.3 `demos/Demo_No4-XTEND_MapRoom/xtend_dome_main.py`

Purpose: XTEND dome/room capture demo.

Recommended debugging run:

```bash
cd /home/user1/GIT/TheAgency
source theagency_venv/bin/activate
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH

python3 sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_dome_main.py   --host 192.0.0.15   --port 8000   --robot-uid drndfb3eeb1   --drone-id 42B   --out-dir /home/user/jetson-containers/data   --capture-hz 1   --yaw-bucket-deg 0
```

Recommended after movement is stable:

```bash
python3 sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_dome_main.py   --host 192.0.0.15   --port 8000   --robot-uid drndfb3eeb1   --drone-id R1   --out-dir /home/user/jetson-containers/data   --capture-every-ms 500   --yaw-bucket-deg 0
```

Expected output:

```text
/home/user/jetson-containers/data/42B/<session_time>/
├── 42B_YYYYmmdd_HHMMSS.jpg
├── 42B_YYYYmmdd_HHMMSS.json
└── ...
```

Latest symlink:

```bash
rm -f /home/user/jetson-containers/data/42B/latest
ln -s /home/user/jetson-containers/data/42B/<session_time>       /home/user/jetson-containers/data/42B/latest
```

---

## 3.4 `robots/XTEND/get_xtend_probe.py`

Purpose: combined XTEND RTSP and WebSocket telemetry probe. It can show video, print telemetry schema, and dump raw `ROBOT_STATUS` messages.

Run:

```bash
cd /home/user1/GIT/TheAgency
source theagency_venv/bin/activate
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH

python3 sparx_agency/robots/XTEND/get_xtend_probe.py   --host 192.0.0.15   --port 8000   --robot-uid drndfb3eeb1   --rtsp-uri rtsp://192.0.0.15:8510/active_drone_fpv   --mode both   --frequency-hz 10   --raw-dump-seconds 5   --robot-status-only
```

Show video:

```bash
python3 sparx_agency/robots/XTEND/get_xtend_probe.py   --host 192.0.0.15   --port 8000   --robot-uid drndfb3eeb1   --rtsp-uri rtsp://192.0.0.15:8510/active_drone_fpv   --show-video
```

Useful options:

```bash
--mode send|listen|both
--frequency-hz 10
--raw-dump-seconds 5
--robot-status-only
--show-video
--rtsp-latency-ms 0
```

---

## 3.5 `robots/XTEND/xtend_rtsp_image_publisher.py`

Purpose: publish XTEND RTSP video into ROS 2 as `sensor_msgs/msg/Image`.

Default topic:

```text
/xtend/image_raw
```

Default frame ID:

```text
xtend_camera
```

Run:

```bash
cd /home/user1/GIT/TheAgency
source /opt/ros/humble/setup.bash
source theagency_venv/bin/activate
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH

python3 sparx_agency/robots/XTEND/xtend_rtsp_image_publisher.py   --rtsp-uri rtsp://192.0.0.15:8510/active_drone_fpv   --image-topic /xtend/image_raw   --frame-id xtend_camera   --publish-hz 3   --backend ffmpeg
```

Alternative GStreamer backend:

```bash
python3 sparx_agency/robots/XTEND/xtend_rtsp_image_publisher.py   --rtsp-uri rtsp://192.0.0.15:8510/active_drone_fpv   --image-topic /xtend/image_raw   --frame-id xtend_camera   --publish-hz 10   --backend gstreamer
```

Check topic:

```bash
ros2 topic list | grep xtend
ros2 topic hz /xtend/image_raw
ros2 topic echo /xtend/image_raw --once
```

---

## 3.6 `robots/XTEND/create_xtend_da3_depth_from_images.py`

Purpose: run DA3 TensorRT depth from an image folder.

Run on original 720x420 images:

```bash
cd /home/user1/GIT/TheAgency
source theagency_venv/bin/activate
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH

python3 sparx_agency/robots/XTEND/create_xtend_da3_depth_from_images.py   --input-images-dir /home/user1/Documents/2026_05_04___11_15_19_700_65sec_4m   --output-dir /home/user1/Documents/depth_2026_05_04___11_15_19_700_65sec_4m   --engine-path /home/user1/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine   --calib-yaml /home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml   --max-depth-m 7.0
```

Run on cropped 504x280 images:

```bash
python3 sparx_agency/robots/XTEND/create_xtend_da3_depth_from_images.py   --input-images-dir /home/user1/Documents/2026_05_04___11_15_19_700_65sec_4m_crop_504x280   --output-dir /home/user1/Documents/depth_2026_05_04___11_15_19_700_65sec_4m_crop_504x280   --engine-path /home/user1/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine   --calib-yaml /home/user1/Documents/camera_xtend_crop_504_280.yaml   --no-rectify   --max-depth-m 7.0
```

Important:
- If images are 504x280, the YAML must also say 504x280.
- Do not use the original 720x420 YAML with cropped images.
- Avoid `--resize-to-calib` if the goal is to avoid resizing/interpolation.

---

## 3.7 `robots/XTEND/crop_xtend_rgb_json_to_depth_size.py`

Purpose: crop paired JPG/JSON frames to match DA3 output size, usually `504x280`.

Run crop only:

```bash
cd /home/user1/GIT/TheAgency
source theagency_venv/bin/activate
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH

python3 sparx_agency/robots/XTEND/crop_xtend_rgb_json_to_depth_size.py   --input-dir /home/user1/Documents/2026_05_04___11_15_19_700_65sec_4m   --output-dir /home/user1/Documents/2026_05_04___11_15_19_700_65sec_4m_crop_504x280   --target-width 504   --target-height 280
```

Run crop and write matching cropped calibration YAML:

```bash
python3 sparx_agency/robots/XTEND/crop_xtend_rgb_json_to_depth_size.py   --input-dir /home/user1/Documents/2026_05_04___11_15_19_700_65sec_4m   --output-dir /home/user1/Documents/2026_05_04___11_15_19_700_65sec_4m_crop_504x280   --target-width 504   --target-height 280   --calib-yaml /home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml   --output-calib-yaml /home/user1/Documents/camera_xtend_crop_504_280.yaml
```

For 720x420 to 504x280 center crop:
- crop x offset = 108
- crop y offset = 70
- `fx/fy` stay unchanged
- `cx_new = cx_old - 108`
- `cy_new = cy_old - 70`

---

## 3.8 `robots/XTEND/plot_xtend_trajectory.py`

Purpose: read all JSON sidecars in a folder and draw the x/y/z trajectory.

Run:

```bash
cd /home/user1/GIT/TheAgency
source theagency_venv/bin/activate
export PYTHONPATH=/home/user1/GIT/TheAgency:$PYTHONPATH

python3 sparx_agency/robots/XTEND/plot_xtend_trajectory.py   /home/user1/Documents/2026_05_04___11_15_19_700_65sec_4m
```

Save plot:

```bash
python3 sparx_agency/robots/XTEND/plot_xtend_trajectory.py   /home/user1/Documents/2026_05_04___11_15_19_700_65sec_4m   --save /home/user1/Documents/xtend_trajectory.png
```

Show frame index labels:

```bash
python3 sparx_agency/robots/XTEND/plot_xtend_trajectory.py   /home/user1/Documents/2026_05_04___11_15_19_700_65sec_4m   --show-index
```

---

## 4. Recommended workflows

### Workflow A: static DA3 capture without movement

```bash
python3 sparx_agency/tasks/mapping/take_xtend_da3_frames.py   --capture-hz 5   --duration-sec 30   --output-dir /home/user1/Documents/xtend_da3_takes
```

### Workflow B: dome capture then offline DA3

```bash
python3 sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_dome_main.py   --drone-id 42B   --out-dir /home/user/jetson-containers/data   --capture-hz 1   --yaw-bucket-deg 0
```

```bash
python3 sparx_agency/robots/XTEND/plot_xtend_trajectory.py   /home/user/jetson-containers/data/42B/latest   --show-index
```

```bash
python3 sparx_agency/robots/XTEND/crop_xtend_rgb_json_to_depth_size.py   --input-dir /home/user/jetson-containers/data/42B/latest   --output-dir /home/user/jetson-containers/data/42B/latest_crop_504x280   --target-width 504   --target-height 280   --calib-yaml /home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml   --output-calib-yaml /home/user/jetson-containers/data/42B/camera_xtend_crop_504_280.yaml
```

```bash
python3 sparx_agency/robots/XTEND/create_xtend_da3_depth_from_images.py   --input-images-dir /home/user/jetson-containers/data/42B/latest_crop_504x280   --output-dir /home/user/jetson-containers/data/42B/latest_depth_da3   --engine-path /home/user1/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine   --calib-yaml /home/user/jetson-containers/data/42B/camera_xtend_crop_504_280.yaml   --no-rectify   --max-depth-m 7.0
```

---

## 5. Safety / debugging notes

- Do not change `robots/XTEND/automation.py` if that file already works with XTEND.
- Keep dome movement close to the original `automation.py` scenario style.
- Start with `--capture-hz 1 --yaw-bucket-deg 0`.
- Avoid printing every telemetry packet; throttle debug prints.
- Confirm `robot_uid` matches the actual robot block in `ROBOT_STATUS`.
- RTSP working does not prove movement commands are accepted; video and movement use different paths.
- If drone does not move, first test basic `automation.py` movement independently before adding capture.
