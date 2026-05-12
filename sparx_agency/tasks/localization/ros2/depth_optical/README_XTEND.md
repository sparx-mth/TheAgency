# Visual Odometry & Live Depth Pipeline

This repository contains the complete pipeline for estimating camera velocity and pose using RGB images, live depth computation (via Depth Anything V3), and Optical Flow. 

The pipeline consists of 5 main components running concurrently:
1.  **RGB File Publisher:** Streams static RGB frames as ROS 2 topics.
2.  **Depth Processor Node:** Computes metric depth in real-time using TensorRT.
3.  **Flow Depth Velocity Node:** Calculates linear velocity using Lucas-Kanade Optical Flow and Weighted Least Squares (WLS).
4.  **Velocity Integrator:** Integrates the velocity to estimate the absolute pose and path distance.
5.  **TF Publisher:** Maintains the static transform tree for the camera frame.

---

## 🚀 Execution Guide


to run you can run the ros launch file- 


```bash
cd ~/GIT/TheAgency
ros2 launch sparx_agency/tasks/localization/ros2/depth_optical/optical_depth_pipeline.launch.py
```


OR -to run the full pipeline, open **5 separate terminals**. 

### Terminal 1: Flow Depth Velocity Node
Calculates the velocity based on RGB and computed depth topics.
```bash
cd ~/GIT/TheAgency
source .venv/bin/activate

python3 -m sparx_agency.tasks.localization.ros2.depth_optical.flow_depth_velocity_node \
  --ros-args \
  -p use_sim_time:=false \
  -p show_debug:=true \
  -p csv_filename:="/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/zone_velocities_log.csv" \
  -p image_topic:=/xtend/rgb \
  -p depth_topic:=/xtend/depth_m \
  -p camera_config_yaml:=/home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml \
  -p depth_scale:=0.8
```

###   Terminal 2: Velocity Integrator
Accumulates velocity messages into an absolute position (odom frame).
```bash
cd ~/GIT/TheAgency
source .venv/bin/activate

python3 -m sparx_agency.tasks.localization.ros2.depth_optical.velocity_integrator \
  --ros-args \
  -p use_sim_time:=false \
  -p target_frame:=odom \
  -p init_from_gt:=false
```

###  Terminal 3: Static Transform (TF2)

Publishes the static transform between the local odometry frame and the camera link.
```bash
cd ~/GIT/TheAgency
source .venv/bin/activate

ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom xtend_camera
```

### Terminal 4: Live Depth Processor (TensorRT)
Note: This node requires the dedicated Depth Anything V3 virtual environment.

```bash
cd ~/GIT/TheAgency
source ~/depth_anything_ws/src/ros2-depth-anything-v3-trt/da3_venv/bin/activate

python3 -m sparx_agency.tasks.mapping.ros2.depth_processor_node \
  --ros-args \
  -p use_sim_time:=false \
  -p engine_path:=$HOME/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine \
  -p config_yaml:=$HOME/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml \
  -p rgb_topic:=/xtend/rgb \
  -p pub_depth_topic:=/xtend/depth_m \
  -p pub_debug_topic:=/xtend/depth_vis
```

### Terminal 5: RGB Data Publisher
Feeds the dataset images and CameraInfo into the pipeline. Run this last so the other nodes don't miss the initial frames.

```bash

cd ~/GIT/TheAgency
source .venv/bin/activate

python3 -m sparx_agency.tasks.localization.common.publish_rgb_from_files \
  --rgb-dir ~/Documents/xtend_da3_takes/xtend_rectified_depth_take_003_20260429_160647/rgb_rectified \
  --publish-hz 10.0
```


🛠️ Important Notes

    NumPy Version: Ensure that your standard .venv utilizes numpy<2.0 to avoid compatibility issues with ROS 2 cv_bridge.

    Camera Intrinsics: The Depth Processor strictly relies on the camera_xtend_ros_calib_720_420.yaml to convert relative DA3 outputs into accurate metric scales.


## 🧠 Algorithm Overview: Visual Odometry

The `FlowDepthVelocityNode` estimates the camera's 3D linear velocity using a robust Visual Odometry pipeline. Here is how it works under the hood:

1. **Data Synchronization:** RGB frames and depth maps are temporally synchronized using `message_filters` to ensure accurate pairing.
2. **Depth Smoothing:** An Exponential Moving Average (EMA) filter is applied to the incoming depth maps to reduce temporal noise and stabilize depth readings.
3. **Feature Tracking (Optical Flow):** The Lucas-Kanade algorithm detects and tracks distinct visual features (corners) between consecutive frames, calculating their 2D pixel displacement ($du, dv$).
4. **3D Velocity Estimation (WLS):** The algorithm translates 2D pixel motion into 3D camera velocity ($v_x, v_y, v_z$) using the pinhole camera model equations and the corresponding depth values. It solves this overdetermined system using **Weighted Least Squares (WLS)**, assigning higher weights to features near the image center to minimize the impact of lens distortion and rotation.
5. **Signal Filtering:** Finally, a low-pass filter and a deadband threshold are applied to the estimated velocity. This eliminates micro-vibrations and prevents accumulation of drift (zero-velocity update) when the drone is stationary.