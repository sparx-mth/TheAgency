# TheAgency

1. There are 2 docker images to run - for Isaac Sim and for Dev. They are ready to pull from:
2. RUN `export ISAAC_ROS_WS=/path/to/TheAgency/isaac_ros-dev/`
3. Clone isaac_ros_visual_slam: `mkdir -p ${ISAAC_ROS_WS}/isaac_ros_assets && cd ${ISAAC_ROS_WS}/isaac_ros_assets && git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git `
4. Run Isaac Sim using `scripts/run_docker_isaac_sim_ros.sh` and lauch the simulation using `runapp` (alias).
5. Run the dev container: 
<!-- The following is from https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/tutorial_isaac_sim.html -->
6. Launch Visual Slam on tmux: `ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_isaac_sim.launch.py`
7. Launch rviz with Visual Slam config: `rviz2 -d $(ros2 pkg prefix isaac_ros_visual_slam --share)/rviz/isaac_sim.cfg.rviz`
8. Start moving the robot: `ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"`
9. If everything works fine, there is an option to save the map and load it, look at the tutorial link above (8.). 

========================

## Documnetation for building the dockers:
### 2.1 Isaac Sim (Following https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html)
2.1.1. Login to nvidia docker-hub: 
    Go to https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
    Sign-in
    On right-top corner, click your username -> setup -> API keys -> Generate API Key
2.1.2 Pull the base image from docker-hub
        $ docker login nvcr.io
        Username: $oauthtoken
        Password: 
        $ docker pull nvcr.io/nvidia/isaac-sim:4.0.0
2.1.3 Build the second layer in order to add ROS bridge using `scripts/build_docker_nvidia_isaac_ros.sh`. That builds docker/Dockerfile-nvidia-isaac-ros.

### 2.2 Isaac ROS Dev Docker:
2.2.1. Set up the developer environment - https://nvidia-isaac-ros.github.io/getting_started/dev_env_setup.html
2.2.2 Clone isaac_ros_common under ${ISAAC_ROS_WS}/src: `cd ${ISAAC_ROS_WS}/src && git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git`
2.2.3. Fix the Dev Container according to https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/pull/136/files
2.2.4. Build the dev-container: `cd ${ISAAC_ROS_WS}/src/isaac_ros_common/scripts && ./run_dev.sh`
<!-- The following is from https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html#quickstart -->
2.2.5. Clone visual slam:
    `cd ${ISAAC_ROS_WS}/isaac_ros_assets && git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git `
2.2.6. Run the container and build visual slam:
    `./run_dev.sh -d $ISAAC_ROS_WS/isaac_ros_assets/isaac_ros_visual_slam && cd /workspaces/isaac_ros-dev && rosdep install --from-paths ${ISAAC_ROS_WS}/isaac_ros_visual_slam --ignore-src -y && cd ${ISAAC_ROS_WS}/ &&  colcon build --symlink-install --packages-up-to isaac_ros_visual_slam`


