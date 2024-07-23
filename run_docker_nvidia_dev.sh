#!/bin/bash

command="docker run -it --rm --privileged --network host \\
    -v /tmp/.X11-unix:/tmp/.X11-unix \\
    -v /home/oded/.Xauthority:/home/admin/.Xauthority:rw \\
    -e DISPLAY \\
    -e NVIDIA_VISIBLE_DEVICES=all \\
    -e NVIDIA_DRIVER_CAPABILITIES=all \\
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/usr/local/share/middleware_profiles/rtps_udp_profile.xml \\
    -e ROS_DOMAIN_ID \\
    -e USER \\
    -e ISAAC_ROS_WS=/workspaces/isaac_ros-dev \\
    -v /home/oded/GIT/TheAgency/isaac_ros-dev/:/workspaces/isaac_ros-dev \\
    -v /etc/localtime:/etc/localtime:ro \\
    --name isaac_ros_dev-x86_64-container \\
    --runtime nvidia \\
    --user=admin \\
    --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \\
    --workdir /workspaces/isaac_ros-dev \\
    isaac_ros_dev-x86_64 /bin/bash"

# Print the command with variables expanded
echo "Executing command:"
eval echo "$command"

# Execute the command
eval "$command"
