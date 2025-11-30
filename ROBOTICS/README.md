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



