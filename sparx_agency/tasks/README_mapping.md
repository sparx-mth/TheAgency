## 📄 README: Mapping Task (SJTU + ROS 2 Bag)
### Overview
This task (`create_map_from_video.py`) runs the probabilistic mapping pipeline using a monocular video feed and odometry, specifically tuned for the SJTU drone environment.

---
### 📂 Dataset / ROS 2 Bag Requirements
To run this task with a recorded bag, ensure the bag contains the following topics:

| Topic                            | Type                    | Required For            |
|----------------------------------|-------------------------|-------------------------|
| `/simple_drone/front/image_raw ` | `sensor_msgs/Image`     | Depth Estimation (RGB)  |
| `/simple_drone/front/camera_info`| `sensor_msgs/CameraInfo`| Camera Intrinsics       |
| `/simple_drone/odom`             | `nav_msgs/Odometry`     | Robot Localization      |
| `/tf & /tf_static`               | `tf2_msgs/TFMessage`    | Transforms              |

---
## 🚀 How to Run
### 1. Play the Bag
Open a terminal and play your recorded data. For example:
```bash 
ros2 bag play ~/Videos/run_1767702755/run_004_bag/ --clock --rate 0.1
```
Important: Use --clock to synchronize the mapping node with the recorded time.

### 2. Run the Mapping Task
In a new terminal, launch the task script. Ensure your PYTHONPATH includes the sparx_agency root.
```bash
cd ~/GIT/TheAgency/
python3 -m sparx_agency.tasks.create_map_from_video --ros-args -p use_sim_time:=true
```
---
## 🛠️ Configured Parameters for SJTU
The task initializes the ProbabilisticGridConfig with the following SJTU-specific defaults:

1. Resolution: 0.3m
2. Obstacle Height: 0.9m to 2.5m (Ignores the floor and drone shadow).
3. Max Range: 15.0m (for Depth-Anything-V2).

---
## 📊 Visualization
Launch RViz2: rviz2
1. Set Fixed Frame to map (or simple_drone/odom).
2. Add an OccupancyGrid display for topic /occupancy_grid.
3. Add a PointCloud2 display for topic /debug/cloud_global.