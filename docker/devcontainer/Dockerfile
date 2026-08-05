# ============================================================
# Dev container for the project's own code (not FALCON, not the ROS1 bridge --
# those have their own containers under sparx_agency/tasks/planning/falcon/).
#
# Covers the perception/mapping/ROBOTICAN stack: ROS2 Humble, TensorRT (DA3
# depth), GStreamer (Rooster video), and the vendor Sphera message packages,
# built for Humble from source (see stage_vendor_msgs.sh).
#
# BASE_IMAGE default matches THIS machine's validated stack: driver reports
# CUDA 13.0 (`nvidia-smi`), and requirements/devices/office_pc.txt --
# synced via `pip freeze` from this machine's actual working venv -- shows
# tensorrt was installed as a cu13 wheel (comment: "cu13 sub-dep is
# deprecated on PyPI; install separately"). If your machine's driver reports
# a different CUDA version, or your existing DA3 engine was built with a
# different TensorRT, override BASE_IMAGE and the pip tensorrt version below
# to match -- engines are locked to the exact build that produced them (see
# tasks/common/model_registry/).
#
# ARG ROS_DISTRO=humble on both PC and the eventual Jetson build, so the same
# Dockerfile serves both -- only BASE_IMAGE changes (an L4T base on Jetson).
# ============================================================
ARG BASE_IMAGE=nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG ROS_DISTRO=humble
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

# 1. Base tools + the ROS 2 apt repo (this base image is plain CUDA/Ubuntu,
#    not an OSRF ros image, so the repo isn't preconfigured).
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl gnupg2 lsb-release software-properties-common ca-certificates \
      git python3-pip python3-tk \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
       http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) main" \
       > /etc/apt/sources.list.d/ros2.list \
    && rm -rf /var/lib/apt/lists/*

# 2. ROS 2 Humble + the packages items 1-8 actually need: cv_bridge (depth
#    node), tf2_ros (static/dynamic TF), octomap-server, rviz2 (offline
#    replay verification), rmw_cyclonedds_cpp (matches Sphera's DDS config),
#    colcon (to build the vendor interfaces below).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-ros-base \
      ros-${ROS_DISTRO}-cv-bridge \
      ros-${ROS_DISTRO}-tf2-ros \
      ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
      ros-${ROS_DISTRO}-octomap-server \
      ros-${ROS_DISTRO}-rviz2 \
      python3-colcon-common-extensions \
      python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# 3. GStreamer + PyGObject -- mandatory for the Rooster video path
#    (rooster_frame_dir_publisher.py, ui.py), not optional: a version
#    mismatch here is a hard SIGSEGV, not a Python exception (see
#    run_rooster_frame_dir_publisher.sh's warning). Notably absent from
#    every requirements/*.txt today -- this image is what actually documents
#    the dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
      gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

# 4. Plain system-level Python deps -- installed via apt, not pip. These
#    have no CUDA/numpy-ABI coupling and no version pin anything else in this
#    project depends on, so the distro package is simplest and gets security
#    updates through the normal apt channel instead of a frozen pip pin.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-yaml \
      python3-requests \
      python3-pil \
      python3-websockets \
    && rm -rf /var/lib/apt/lists/*

# 5. Python deps that stay on pip, pinned to match
#    requirements/devices/office_pc.txt (this machine's own validated venv),
#    NOT requirements.freeze.txt (pip-installs a Jazzy-era rclpy and a
#    different CUDA wheel set that would fight both the apt ROS install and
#    this image's own CUDA/TensorRT). No venv inside the container -- system
#    Python 3.10 with ROS supplying rclpy; the host's venv/ is 3.12 and would
#    be an ABI mismatch here.
#
#    Kept on pip on purpose, not moved to apt:
#      - numpy / opencv-python: exact versions matched to this project's
#        validated combination; Ubuntu 22.04's apt numpy/opencv are several
#        minor versions behind and untested against this pipeline.
#      - pupil-apriltags: no Debian/Ubuntu package exists.
#      - onnx / pycuda: CUDA-toolkit-adjacent, kept version-matched to the
#        pinned CUDA/TensorRT rather than whatever apt's system CUDA build was.
#      - tensorrt: apt's TensorRT packages in the CUDA network repo are
#        version-suffixed per CUDA release (e.g. "+cuda12.9") and have had
#        unresolved dependency conflicts on fresh CUDA 13 base images
#        (NVIDIA/TensorRT#4545); the pip wheel via pypi.nvidia.com is the
#        approach already validated working on this exact host (see the
#        comment in office_pc.txt). Re-evaluate apt once that settles.
RUN pip3 install --no-cache-dir \
      numpy==1.26.4 \
      opencv-python==4.11.0.86 \
      pupil-apriltags==1.0.4.post11 \
      onnx==1.21.0 \
      pycuda==2026.1 \
    && pip3 install --no-cache-dir tensorrt --extra-index-url https://pypi.nvidia.com

# 6. Vendor ROBOTICAN/Sphera message packages, built for this image's
#    ROS_DISTRO from the .msg/.srv sources staged by stage_vendor_msgs.sh
#    (run that script before `docker build`, or this COPY fails on purpose).
#    sphera_common_interfaces is NOT included: it has no .msg source
#    anywhere, only a prebuilt Foxy binary inside Sphera's own image -- code
#    needing it (rooster_ground_truth_localization.py) keeps running inside
#    Sphera's `it` container via docker exec. See docker/devcontainer/README.md.
COPY vendor/rooster_interfaces /opt/rooster_ws/src/rooster_interfaces
RUN . /opt/ros/${ROS_DISTRO}/setup.bash \
    && cd /opt/rooster_ws \
    && rosdep init 2>/dev/null || true \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src -r -y \
    && colcon build --symlink-install

# 7. Container user matching the host's UID/GID (user1:1000:1000), so the
#    repo -- mounted at the SAME absolute path as the host -- stays writable
#    and file ownership doesn't drift. The run_*.sh wrappers hardcode
#    /home/$USER/GIT/TheAgency, so $USER/$HOME must resolve identically.
ARG USERNAME=user1
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && usermod -aG video,dialout ${USERNAME}

COPY --chown=${USERNAME}:${USERNAME} entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER ${USERNAME}
WORKDIR /home/${USERNAME}/GIT/TheAgency
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
