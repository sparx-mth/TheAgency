#  ROS2 Workspace Setup & Usage Guide

This repository provides tools and instructions for running the RQS IAI ROS2 workspace, configuring ROS Domain IDs, scanning DDS domains, streaming video from drones into ROS2 topics, and launching RViz2 inside a Docker container.

---

## 📌 ROS Domain ID Structure

Each system component operates on a different **ROS_DOMAIN_ID**:

| Component | ROS_DOMAIN_ID |
|----------|----------------|
| Sphera – Remote Control | **9** |
| Sphera – Backend Docker | **5** |
| Drone – Backend Docker | **2** |

To change the domain according to the task, edit:

```
rqs_iai_ws/src/docker-compose.yml
```

Example:

```yaml
environment:
  - ROS_DOMAIN_ID=5
```

---

## 🚀 Starting the Main Docker Environment

From inside the workspace:

```bash
cd ~/rqs_iai_ws/src
docker compose up it
```

Attach to the running container:

```bash
docker attach it
```

---

## 🔍 ROS2 Domain Scanner

A Python tool for scanning all ROS2 Domain IDs (0–255) and detecting active DDS networks.

**Location:**

```
/home/rooster/workspace/src/examples/src/ros2_domain_scanner.py
```

**Run:**

```bash
python3 ros2_domain_scanner.py
```

---

## 🎥 Multi-Drone Video Streaming

This script sends video streams from drones into ROS2 topics.

**Location:**

```
/home/rooster/workspace/src/examples/src/multi_video_stream.py
```

**Run:**

```bash
python3 multi_video_stream.py
```

---

## 🟦 Running RViz2 in Docker

Run inside:

```
~/rqs_iai_ws/src
```

**Command:**

```bash
docker run --rm -it   --net=host   -e DISPLAY=$DISPLAY   -e QT_X11_NO_MITSHM=1   -e LIBGL_ALWAYS_SOFTWARE=1   -e ROS_DOMAIN_ID=5   -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp   -e CYCLONEDDS_URI=file:///etc/cyclonedds.xml   -v /tmp/.X11-unix:/tmp/.X11-unix:rw   -v $(pwd)/cyclonedds.xml:/etc/cyclonedds.xml:ro   osrf/ros:foxy-desktop   bash
```

---

## 📦 Setup Inside the RViz Container

Install CycloneDDS:

```bash
apt update && apt install -y ros-foxy-rmw-cyclonedds-cpp
```

Source ROS:

```bash
source /opt/ros/foxy/setup.bash
```

Install video tools (optional):

```bash
pip3 install opencv-python
apt install -y ros-foxy-cv-bridge
```

Run RViz2:

```bash
rviz2
```

---

## 📁 Repository Structure

```
src/
 ├── docker-compose.yml
 ├── examples/
 │    ├── src/
 │    │    ├── ros2_domain_scanner.py
 │    │    ├── multi_video_stream.py
 │    │    └── ...
 └── ...
```