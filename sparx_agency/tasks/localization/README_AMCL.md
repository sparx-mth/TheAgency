# Vision-Based AMCL Localization Pipeline (XTEND)

This repository contains a real-time, vision-based localization pipeline for the XTEND robot. It integrates **Optical Flow** for continuous odometry and **Depth Anything V3** with a custom **Dense Grid-Based Markov Localization (AMCL)** algorithm.

### Key Features:
* **Pre-computed Ray-Cast LUT (Look-Up Table):** To avoid heavy trigonometric calculations at runtime, the system relies on a 4D tensor LUT computed offline. This allows the sensor model to evaluate expected depths for 64 beams across 32 headings in `O(1)` time.
* **Local Window Optimization:** Instead of searching the entire global map, the algorithm extracts a local sliding window (e.g., 5x5 meters) around the Optical Flow's predicted pose. 
* **Bayesian Fusion:** * **Prior (Motion Model):** A Gaussian belief distribution is generated around the raw visual odometry prediction.
  * **Likelihood (Sensor Model):** The real-time depth array from *Depth Anything V3* is compared against the theoretical depths in the LUT.
  * The final pose is derived via `argmax` over the fused belief matrix, snapping the robot to the most geometrically logical position while preserving the smoothness of the optical flow.

---

## Running the Pipeline

To run the full localization pipeline, open separate terminal windows and execute the following nodes in order.

### 1. Depth Processor Node (Depth Anything V3)
Generates metric depth maps from RGB images using a TensorRT engine.
```bash
cd ~/GIT/TheAgency
source ~/depth_anything_ws/src/ros2-depth-anything-v3-trt/da3_venv/bin/activate

python3 -m sparx_agency.tasks.mapping.ros2.depth_processor_node_debug \
  --ros-args \
  -p use_sim_time:=false \
  -p engine_path:=$HOME/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine \
  -p config_yaml:=$HOME/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml \
  -p rgb_topic:=/xtend/rgb \
  -p pub_depth_topic:=/xtend/depth_m \
  -p pub_debug_topic:=/xtend/depth_vis
```

### 2.Optical Flow Velocity Node
```bash
python3 -m sparx_agency.tasks.localization.ros2.depth_optical.flow_depth_velocity_node \
  --ros-args \
  -p use_sim_time:=true \
  -p show_debug:=true \
  -p csv_filename:="$HOME/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/zone_velocities_log.csv" \
  -p image_topic:=/xtend/rgb \
  -p depth_topic:=/xtend/depth_m \
  -p camera_config_yaml:=$HOME/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml \
  -p depth_scale:=0.9
```

### 3. Velocity Integrator Node
```bash
python3 -m sparx_agency.tasks.localization.ros2.depth_optical.velocity_integrator_debug \
  --ros-args \
  -p use_sim_time:=true \
  -p target_frame:=odom \
  -p init_from_gt:=false
  ```

### 4. Static TF Publisher
```bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom xtend_camera
  ```

### 5. AMCL Pose Estimator
```bash
python3 -m sparx_agency.tasks.amcl_pose
```

### 6. Image Publisher (Data Source)

```bash
python3 -m sparx_agency.tasks.localization.common.publish_rgb_from_files_all \
  --rgb-dir ~/Documents/xtend_da3_takes/2026_05_04___11_15_19_700_65sec_4m_ \
  --publish-hz 10.0
```


