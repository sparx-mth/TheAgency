# AprilTag Localization & Triangulation Pipeline

This repository provides a robust localization pipeline for cameras and drones, utilizing AprilTag detection to estimate 6-DOF poses within a known world frame. The system leverages pre-defined tag poses (via YAML) and camera intrinsics to perform real-time triangulation.

---

## 🚀 Running the Pipeline

Ensure your environment is set up and your configuration files (YAMLs) are correctly pointing to your local paths.



### Option 1: Live ROS Mode (Real-time Drone Feed)
This mode subscribes to a live ROS image topic. The pipeline will process frames as they arrive and publish the estimated PoseStamped message.

```bash
python3 -m sparx_agency.tasks.localization.apriltag_triangulation_node \
  --tag_map_path $HOME/GIT/TheAgency/sparx_agency/tasks/localization/config/tag_map_path_ALL.yaml \
  --camera_calib_path $HOME/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml \
  --tag_size_m 0.13 \
  --source ros
  --min_margin 15.0
```

Default Topic: /xtend/rgb

Output: The calculated pose is published to /xtend/april_tag_pose (geometry_msgs/PoseStamped).


### Option 2: Offline Mode (Processing from Image Directory)
Use this mode to process a sequence of saved images. Useful for debugging and offline evaluation.

```bash
python3 -m sparx_agency.tasks.localization.apriltag_triangulation_node \
  --tag_map_path $HOME/GIT/TheAgency/sparx_agency/tasks/localization/config/tag_map_path_ALL.yaml \
  --camera_calib_path $HOME/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml \
  --tag_size_m 0.13 \
  --image_dir $HOME/Pictures/xtend_da3_take_20260527_124147
```


### Live Visualization Tools
Run these in a separate terminal while the localization node is active to visualize the trajectory.
#### 2D Trajectory & Yaw
```bash
python3 -m sparx_agency.tasks.localization.plot_apriltag_pose_live_2D
```

#### 3D Spatial Trajectory
```bash
python3 -m sparx_agency.tasks.localization.plot_apriltag_pose_live
```