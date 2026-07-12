# Rooster Development Environment Setup

Follow these steps to set up the development environment.

---

## Step 1: Create workspace directory

```bash
mkdir -p $HOME/rqs_iai_ws/src

cp -r <media>/rqs_iai/* $HOME/rqs_iai_ws/src

```
Create $HOME/rqs7-private-parameters
```bash
mkdir $HOME/rqs7-private-parameters
```

## Step 2: Edit docker-compose.yml
Set environment variables
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ROS_DOMAIN_ID=9

```
## Step 4: Build ROS2 workspace in docker container
```bash
docker compose up build_ws
```

## Step 5: Start interactive container
```bash
docker compose up it
```
Open new termainal and attach to it container

```bash
docker attach it
```
## Step 6: Create & Run sphera scenario with rooster

## Step 7: Test your environment
```bash
$ ros2 topic list
```
`**Expected output:**`
/parameter_events
/rosout
rooster@pavel-pc:~/rqs7_ws$ ros2 topic list
/R1/azimuth_altitude
/R1/calibration_feedback
/R1/diagnostics
/R1/fcu/actuator_control
/R1/fcu/battery
/R1/fcu/command/attitude
/R1/fcu/command/attitude_rate
/R1/fcu/command/body/velocity
/R1/fcu/command/global/position
/R1/fcu/command/local/position
/R1/fcu/command/local/velocity
/R1/fcu/command_long
/R1/fcu/gps_input
/R1/fcu/manual_control
/R1/fcu/obstacles
/R1/fcu/ranger
/R1/fcu/set_distance_sensor
/R1/fcu/state
/R1/fcu/status_text
/R1/fcu/vision_position_estimate
/R1/gcs_keep_alive
/R1/io_driver/leds_data
/R1/keep_alive
/R1/manual_control
/R1/progress_feedback
/R1/sphera/set_state
/R1/sphera/state
/R1/state
/R1/statistics
/R1/statistics_extended
/R1/video_handler/video_cleanup_status
/R1/video_handler/video_status
/Rooster_1/tf
/parameter_events
/rosout
...

```

---

## Cage Bar Removal — Research Log

The Rooster drone has a physical cage surrounding the fisheye camera. The cage includes permanent side arcs (always visible) and a motorised horizontal bar that sweeps up and down during flight. Both corrupt DA3METRIC-LARGE depth estimates: the bar pixels produce near-depth artefacts and, when the bar is large, bias DA3's global metric scale.

### Approaches tried (concept-level)

**1. Depth-space detection + depth interpolation**
Detect bar rows in the DA3 output by thresholding near-depth pixel fraction per row, then interpolate bar rows from neighbouring depth values.
Result: failed. The depth-row fraction was inversely correlated with bar presence — frames with the cage fully blocking the view had high near-fraction; bar-visible frames had low fraction. Cannot distinguish bar from scene in depth space alone.

**2. RGB dark blob detection (connected components) + position filter + TELEA inpainting**
Find large dark connected components touching frame edges (cage position prior), inpaint with cv2.INPAINT_TELEA.
Result: failed. Scene objects (chair, gun, box) near the cage merged into cage blobs and were incorrectly inpainted away.

**3. Static mask + dynamic connected components + TELEA inpainting**
Separate permanent cage (static mask built from calibration frames, pixels dark in ≥80% of frames) from the moving bar (connected components with separate filter). Inpaint both with TELEA.
Result: failed. When the moving bar did not connect through the bottom arc it split into narrow components below the width threshold. Object-merging problem persisted.

**4. Static mask + horizontal morphological opening + full-row TELEA inpainting**
Static mask for permanent cage. Moving bar detected via morphological open with a wide horizontal kernel (keeps only long horizontal dark runs) + row-fraction threshold → mask entire rows → inpaint everything with TELEA.
Result: partially worked for small bars (<10% row coverage). Failed for large bars (>12%): TELEA propagates from region edges and smears wall+floor texture into the bar region; DA3 then interprets the smear as wrong scene context and its global metric scale collapses by 0.5–0.7 m.

**5. Static mask TELEA + dynamic bar rows → vertical row blending**
Same detection as approach 4, but instead of TELEA for bar rows, each bar row is replaced by a linear blend of the nearest non-bar rows above and below (preserves real room texture rather than smearing).
Result: improved RGB quality, but the global scale collapse for large bars persisted. Even with natural-looking row fills, DA3's context-dependent scale estimation was already corrupted by the bar for frames where bar coverage exceeded ~12%.

**6. RGB-guided depth repair (current)**
Run DA3 on the unmodified original frame (no RGB pre-processing). Use the RGB-based bar detector (morphological opening + row-fraction threshold) to locate bar rows in the depth output. Interpolate those specific rows from neighbouring non-bar depth rows.
Rationale: DA3's global scale is set by the full original scene; only the bar-row pixels in the resulting depth map are patched. Avoids introducing any synthetic texture that could corrupt DA3's global context.
Status: see `bar_inpainter.py` → `repair_depth()`.
