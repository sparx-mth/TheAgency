# ==============================================================================
# BASE DEVELOPMENT LAYER (NVIDIA CUDA Runtime + Development Drivers)
# ==============================================================================
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
ENV GZ_SIM_RESOURCE_PATH=/usr/share/gz/models
# Install foundational tools and configure ROS 2 repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    curl \
    wget \
    sudo \
    nano \
    ca-certificates \
    gnupg2 \
    lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | tee /etc/apt/sources.list.d/ros2.list \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# X11 & QT5 GUI DEPENDENCIES (Required for Gazebo Ignition GUI)
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxcursor1 \
    libxrandr2 \
    libxinerama1 \
    libxi6 \
    libxtst6 \
    libxss1 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon0 \
    libdbus-1-3 \
    libegl1 \
    libxcb-cursor0 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*   

# ==============================================================================
# ROS 2 & AUTONOMY SIMULATION PACKAGE STACK
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-ros-gz \
    ros-humble-ros-gz-sim \
    ros-humble-rmw-cyclonedds-cpp \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# Install isolated Python toolchain dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-numpy \
    python3-opencv \
    python3-yaml \
    python3-requests \
    python3-pil \
    python3-matplotlib \
    python3-scipy \
    python3-pandas \
    python3-networkx \
    python3-sympy \
    python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# USER & WORKSPACE CONFIGURATION LAYER 
# ==============================================================================
ARG USERNAME=user1
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ==============================================================================
# INTERACTIVE BASHRC CONFIGURATION
# ==============================================================================
RUN echo '\n\
# ROS 2 Environment Setup\n\
source /opt/ros/humble/setup.bash\n\
[ -f /home/ws/install/setup.bash ] && source /home/ws/install/setup.bash\n\
\n\
# Autonomous Systems Runtime Parameters\n\
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp\n\
export GZ_SIM_RESOURCE_PATH="/usr/share/gz/models:/home/ws/install/simple_drone/share/simple_drone/models"\n\
export LD_LIBRARY_PATH="/opt/ros/humble/lib:$LD_LIBRARY_PATH"\n\   
' >> /home/$USERNAME/.bashrc \
    && echo '\n\
# ROS 2 Environment Setup\n\
source /opt/ros/humble/setup.bash\n\
[ -f /home/ws/install/setup.bash ] && source /home/ws/install/setup.bash\n\
\n\
# Autonomous Systems Runtime Parameters\n\
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp\n\
export GZ_SIM_RESOURCE_PATH="/usr/share/gz/models:/home/ws/install/simple_drone/share/simple_drone/models"\n\
export LD_LIBRARY_PATH="/opt/ros/humble/lib:$LD_LIBRARY_PATH"\n\   
' >> /root/.bashrc

WORKDIR /home/ws