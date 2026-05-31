# Localization README Knowledge Retention Document

This document summarizes the main README files under:

```text
/GIT/TheAgency/sparx_agency/tasks/localization/
```

---

## README Index

| README File                          | Main Topic                               | Purpose                                                                                                      |
| ------------------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `README_apriltag.md`                          | AprilTag Localization & Triangulation    | Full 6-DOF camera/drone pose estimation using known AprilTag world poses                                     |
| `README_azimuth_apriltag.md`         | AprilTag Azimuth Estimation              | Single-image absolute camera heading estimation using AprilTag orientation                                   |
| `README_AMCL.md`                     | Vision-Based AMCL Localization           | Fusion of Optical Flow odometry, Depth Anything V3, and grid-based Markov localization                       |
| `ros2/depth_optical/README.md`       | Depth + Optical Flow Localization        | ROS2 visual odometry pipeline using Depth Anything, optical flow, velocity integration, and GT evaluation    |
| `ros2/depth_optical/README_XTEND.md` | XTEND Live Depth + Optical Flow Pipeline | Real-time XTEND pipeline using RGB input, Depth Anything V3, optical flow velocity, pose integration, and TF |

---

# 1. AprilTag Localization & Triangulation Pipeline

**Path:**

```text
/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/README_apriltag.md
```

## What this README contains

This README describes the main AprilTag localization pipeline.
The system detects AprilTags in RGB images and estimates the camera or drone pose in a known world frame. It uses:

* AprilTag detection
* Known tag positions from a YAML file
* Camera calibration parameters
* `solvePnP` / pose estimation logic
* ROS publishing of the estimated pose

## Main output

The node publishes the estimated camera pose to:
```text
/xtend/april_tag_pose
```

Message type:
```text
geometry_msgs/PoseStamped
```


# 2. AprilTag-Based Camera Azimuth Estimation

**Path:**

```text
/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/README_azimuth_apriltag.md
```

## What this README contains

This README explains a lightweight AprilTag-based method for estimating only the camera heading, also called azimuth or yaw.
Unlike the full triangulation pipeline, this module does not estimate full position. It only estimates:

```text
camera absolute azimuth / heading in world coordinates
```

The computation is single-shot:

```text
Image → AprilTag detection → solvePnP → relative yaw → absolute azimuth
```

## Main idea

The system assumes that the orientation of each AprilTag in the world is already known.

Example YAML:

```yaml
tags:
  10: 0
  11: 90
  12: 180
  13: 270
```

The detector finds a known tag in the image, estimates the relative angle between the camera and the tag, and then combines it with the known tag/wall orientation.

## Main output

The output is a single azimuth angle in degrees:

```text
0–360 degrees
```

Example:
```text
123.456789
```

---

# 3. Vision-Based AMCL Localization Pipeline

**Path:**

```text
/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/README_AMCL.md
```

## What this README contains

This README describes the full vision-based AMCL localization pipeline for the XTEND robot.

The system combines:

* Optical Flow for continuous odometry
* Depth Anything V3 for metric depth estimation
* A custom Dense Grid-Based Markov Localization / AMCL algorithm
* A pre-computed ray-casting LUT
* Bayesian fusion between motion prior and depth-based sensor likelihood


### Pre-computed LUT

The system uses a 4D tensor LUT to avoid expensive runtime ray-casting.

The LUT stores expected depths for:

* 64 beams
* 32 headings
* multiple map cells

---

# 4. Depth + Optical Flow Localization Pipeline

**Path:**

```text
/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/README.md
```

## What this README contains

This README describes the ROS2 depth + optical flow localization pipeline.

The system estimates monocular visual odometry by combining:

* Depth Anything V2 or V3
* Optical Flow
* Depth-scaled motion estimation
* Velocity integration
* Evaluation against ground truth pose

## Main pipeline

```text
RGB Image
↓
Depth Anything
↓
Depth Map
↓
Optical Flow + Depth
↓
Metric Velocity
↓
Pose Integration
↓
GT Pose Error / RMS Drift
```

## Main ROS topics

### Input topics

```text
/simple_drone/front/image_raw
/simple_drone/front/camera_info
/simple_drone/odom
/clock
```

### Output topics

```text
/depth_anything/depth
/depth_anything/depth_vis
/flow_depth/velocity
```
---


# 5. XTEND Visual Odometry & Live Depth Pipeline

**Path:**

```text
/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/README_XTEND.md
```

## What this README contains

This README describes the live XTEND version of the visual odometry pipeline.

It estimates camera velocity and pose using:

* RGB image publishing
* Live metric depth from Depth Anything V3
* Lucas-Kanade Optical Flow
* Weighted Least Squares velocity estimation
* Velocity integration
* Static TF publishing

## Main components

The pipeline has five main components:

1. RGB File Publisher
2. Depth Processor Node
3. Flow Depth Velocity Node
4. Velocity Integrator
5. Static TF Publisher

## Main use cases

Use this README when you want to:

* Run the live XTEND visual odometry pipeline
* Use `/xtend/rgb` as the RGB input topic
* Generate `/xtend/depth_m` using Depth Anything V3
* Estimate velocity from optical flow and depth
* Integrate velocity into a pose estimate
* Run all components manually or using a launch file

## Recommended launch command

```bash
cd ~/GIT/TheAgency
ros2 launch sparx_agency/tasks/localization/ros2/depth_optical/optical_depth_pipeline.launch.py
```

## Algorithm summary

The `FlowDepthVelocityNode` estimates linear velocity using:

1. Synchronization between RGB and depth frames
2. EMA smoothing on depth maps
3. Lucas-Kanade Optical Flow for feature tracking
4. Pinhole camera model projection from 2D pixel flow to 3D velocity
5. Weighted Least Squares to estimate robust motion
6. Low-pass filtering and deadband threshold to reduce drift


---



# Overall Project Structure

The localization folder contains several complementary localization approaches:

1. **AprilTag triangulation**
   Used when known visual markers are available and full pose estimation is required.

2. **AprilTag azimuth estimation**
   Used when only heading correction is needed from a known tag orientation.

3. **Depth + Optical Flow visual odometry**
   Used for continuous motion estimation from monocular RGB and learned depth.

4. **XTEND live visual odometry**
   A real-time version of the depth + optical flow pipeline adapted to XTEND topics and camera calibration.

5. **AMCL map correction**
   Used to reduce optical-flow drift by comparing live depth observations to expected map geometry.

Together, these README files document the main localization tools used in the XTEND vision-based localization project.
