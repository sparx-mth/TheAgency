# syntax=docker/dockerfile:1
ARG ROS_DISTRO=humble
ARG TARGETARCH=amd64

# --- Stage 1: Base PC (AMD64) ---
FROM ros:${ROS_DISTRO} AS base-pc
ARG ROS_DISTRO
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv git curl wget \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-cyclonedds \
    ros-${ROS_DISTRO}-ros-gz \
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    ignition-fortress \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 2: Base Jetson (ARM64) ---
FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1 AS base-jetson
ARG ROS_DISTRO
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv git curl wget software-properties-common \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y ros-humble-desktop ros-humble-ros-gz ros-humble-cyclonedds \
    && rm -rf /var/lib/apt/lists/*

# --- Select Base ---
FROM base-pc AS base-amd64
FROM base-jetson AS base-arm64
FROM base-${TARGETARCH} AS base

# --- Stage 3: Heavy Deps (Cached Layer) ---
FROM base AS cacher
ARG ROS_DISTRO
# Add NVIDIA CUDA repo for TensorRT
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y \
    python3-numpy \
    python3-opencv \
    tensorrt \
    python3-libnvinfer \
    libnvinfer-plugin10 \
    python3-yaml python3-requests python3-pil \
    python3-matplotlib python3-scipy python3-pandas \
    python3-sklearn python3-numba python3-networkx \
    python3-sympy python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*

# Copy reqs to install pip deps in cache layer
COPY sparx_agency/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# --- Stage 4: Final Image ---
FROM base
ARG ROS_DISTRO
ARG USERNAME=user1
ARG USER_UID=1000
ARG USER_GID=1000

# Create User
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt-get update && apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Copy cached deps from Stage 3
COPY --from=cacher /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=cacher /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=cacher /usr/bin /usr/bin
COPY --from=cacher /opt/ros /opt/ros

USER $USERNAME
WORKDIR /home/ws
ENV SHELL=/bin/bash
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_DOMAIN_ID=0
ENV ROS_LOCALHOST_ONLY=1
CMD ["/bin/bash"]   