# TheAgency

* https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html

$ docker login nvcr.io
Username: $oauthtoken
Password: 

$ docker pull nvcr.io/nvidia/isaac-sim:4.0.0

========================

For ROS repos, there is an Isaac ROS Dev Docker:
Exectue installation according to this order:
1. Login to nvidia docker-hub: 
    Go to https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
    Sign-in
    On right-top corner, click your username -> setup -> API keys -> Generate API Key
2. Set up the developer environment - https://nvidia-isaac-ros.github.io/getting_started/dev_env_setup.html
% 3. Follow this link to 'Download Quickstart Assets', dont build yet - https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html#quickstart
3. Fix the Dev Container according to https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/pull/136/files
4. Build the dev-container: cd isaac_ros_common/scripts && ./run_dev.sh
% 4. https://nvidia-isaac-ros.github.io/concepts/docker_devenv/index.html#development-environment
5. cd ${ISAAC_ROS_WS}/isaac_ros_assets && git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git 
6. Then ./run_dev.sh -d $ISAAC_ROS_WS/isaac_ros_assets/isaac_ros_visual_slam
7. Inside the docker:   
            /workspaces/isaac_ros-dev$  rosdep install --from-paths ${ISAAC_ROS_WS}/isaac_ros_visual_slam --ignore-src -y
                                        cd ${ISAAC_ROS_WS}/ &&    colcon build --symlink-install --packages-up-to isaac_ros_visual_slam


