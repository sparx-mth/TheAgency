# AprilTag-Based Camera Azimuth Estimation (OpenCV, No ROS)

## High-level idea

**This system estimates the camera’s absolute azimuth (heading) in the world from a single image, by observing an AprilTag whose orientation in the world is known.**

The system performs a **single-shot computation**:
> **Image → Azimuth (degrees)**

---

## What does this code compute?

This code computes:

> **The absolute azimuth (yaw) of the camera in world coordinates (0–360°)**

based on:
- A detected AprilTag in the image
- The known orientation of that tag in the world

Only **orientation (yaw / azimuth)** is computed — not position, not mapping.


## Step-by-step explanation

### 1️⃣ What is known in advance (Ground Truth)

#### a) Tag orientation in the world

Defined in a YAML configuration file:

```yaml
tags:
  10: 0
  11: 90
  12: 180
  13: 270
```


#### b) Physical size of the tag

Provided as a command-line argument:

```bash
--tag_size_m 0.08
```

Meaning:
- The real-world edge length of the square AprilTag is known (meters)


#### c) Camera calibration

Provided via a standard camera calibration YAML file.

The intrinsic matrix:

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

Describes:
- Pixel-to-angle projection
- Optical center
- Field-of-view geometry

Distortion coefficients (`D`) are supported but optional.


### 2️⃣ What is measured from the image

Using the AprilTag detector (`pupil-apriltags`):

```python
corners_2d = [
  (u1, v1),
  (u2, v2),
  (u3, v3),
  (u4, v4)
]
```

These are the four detected tag corners in **pixel coordinates**.

Only tags that appear in the YAML configuration are considered valid.

---

### 3️⃣ What `solvePnP` computes (core geometry)

The question:

> Which 3D pose of a square produces exactly these pixel projections?

`cv2.solvePnP` computes:

#### Translation vector

```python
tvec = [tx, ty, tz]
```

In the **camera optical frame**:
- `tx`: left / right offset
- `tz`: forward distance

Only `tx` and `tz` are needed for azimuth.


### 4️⃣ Relative yaw computation

```python
relative_yaw = atan2(-tx, tz)
```

This is the **relative horizontal angle** between camera and tag.

---

### 5️⃣ Absolute azimuth estimation

```python
camera_azimuth = wall_azimuth + relative_yaw
```

Because the wall orientation of the tag is known, the camera’s absolute orientation can be recovered.

The result is normalized to **0–360 degrees**.

---

## Final summary

**By detecting an AprilTag with known world orientation and measuring its relative angle in a single image, the system infers the camera’s absolute azimuth.**

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install opencv-python pupil-apriltags pyyaml
```

---

## Running the system (single image)

```bash
python3 -m sparx_agency.tasks.localization.opencv.tag_azimuth_node \
  --tag_config_path sparx_agency/tasks/localization/config/tags_azimuth.yaml \
  --camera_calib_path sparx_agency/tasks/localization/config/front_camera_calib.yaml \
  --tag_size_m 0.08 \
  --image path/to/image.png
```

### Output

```text
123.456789
```

---

## Failure behavior

The program **fails fast** and exits if:
- Tag configuration is missing or empty
- Camera calibration is invalid
- AprilTag detector cannot be created
- No known tag is detected in the image
- Pose estimation fails

This makes it suitable for **service-style request/response systems**.