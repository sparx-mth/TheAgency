# Use an official OSRF ROS image that already has OpenGL/GLX capabilities for rendering
FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV TURTLEBOT3_MODEL=burger
ENV CMAKE_PREFIX_PATH="/opt/open3d/lib/cmake/Open3D/"
ENV GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:/opt/ros/humble/lib
ENV GAZEBO_NO_AUDIO=1
# Create a non-root user named vscode with sudo privileges
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 1. Install system tools and fundamental build dependencies
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
    git \
    nano \
    wget \
    build-essential \
    python3-colcon-common-extensions \
    python3-osrf-pycommon \
    python3-rosdep \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 2. Install specific third-party dependencies required by HKUST FALCON
RUN apt-get update && apt-get install -y \
    linux-tools-generic \
    linux-tools-common \
    libarmadillo-dev \
    libgoogle-glog-dev \
    libdw-dev \
    libc++1 \
    libc++abi1 \
    ros-humble-geographic-msgs \
    ros-humble-pcl-conversions \
    ros-humble-octomap-ros \
    ros-humble-octomap-rviz-plugins \
    ros-humble-turtlebot3-description \
    ros-humble-gazebo-plugins \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-simulations \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# 3. Download and unpack Open3D C++ Package with modern C++11 ABI matching ROS Humble
RUN mkdir -p /opt/open3d && \
    chmod -R +r /opt/open3d && \
    wget https://github.com/isl-org/Open3D/releases/download/v0.19.0/open3d-devel-linux-x86_64-cxx11-abi-cuda-0.19.0.tar.xz -O /tmp/open3d.tar.xz && \
    tar -xvf /tmp/open3d.tar.xz -C /opt/open3d/ --strip-components=1 && \
    rm /tmp/open3d.tar.xz

RUN git clone --depth 1 --branch v2.7.1 https://github.com/stevengj/nlopt.git \
    && cd nlopt \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && cd ../.. && rm -rf nlopt


# Create workspace directory
# RUN mkdir -p /turtlebot3_ws/src
# WORKDIR /turtlebot3_ws/src

# # Copy your local workspace source code into the image
# # Ensure your .dockerignore file excludes 'build' and 'install' folders to save space
# COPY . /turtlebot3_ws/src/

# # Install dependencies and build
# WORKDIR /turtlebot3_ws
# RUN source /opt/ros/humble/setup.bash && \
#     rosdep install --from-paths src --ignore-src -r -y && \
#     colcon build --symlink-install

# # Set environment variables for the runtime
# ENV GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/turtlebot3_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/models
# ENV COLCON_PREFIX_PATH=/turtlebot3_ws/install:$COLCON_PREFIX_PATH

# Source the workspace in the entrypoint
RUN echo "source /turtlebot3_ws/install/setup.bash" >> ~/.bashrc   

# Fix workspace ownership for the non-root user
RUN mkdir -p /workspace && chown -R vscode:vscode /workspace

# Source setup on system-wide terminal startup (works for any user)
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc && \
    echo "if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi" >> /etc/bash.bashrc && \
    echo "if [ -f /workspace/venv/bin/activate ]; then source /workspace/venv/bin/activate; fi" >> /etc/bash.bashrc

# for profiling
RUN ln -s /usr/lib/linux-tools/*-generic/perf /usr/bin/perf || true
ENV CMAKE_BUILD_TYPE=RelWithDebInfo