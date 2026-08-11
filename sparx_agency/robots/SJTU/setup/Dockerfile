# sjtu_drone_sparx:humble -- the SJTU simulator image, plus CycloneDDS.
#
# Build:
#   docker build -t sjtu_drone_sparx:humble sparx_agency/robots/SJTU/setup
#
# This is a two-layer image on purpose. The base, sjtu_drone_nadav:humble_ros2,
# is 3.7 GB of ROS 2 Humble + Gazebo Classic 11 + the simulator's build deps and
# takes the better part of an hour to produce; nothing here needs it rebuilt.
# All this adds is the DDS implementation the ROS 1 bridge can actually talk to.
#
# What it is fixing. The external repo's sjtu_drone/run.sh apt-installs
# ros-humble-rmw-cyclonedds-cpp on EVERY launch and `exit 1`s if the install
# fails, so bringing the world up needs working apt mirrors -- on a machine with
# no network, or when the ROS package server is having a bad day, the simulator
# does not start at all. Baking the package in makes bring-up offline-clean and
# saves the ~20 s apt round trip each run.
#
# Why CycloneDDS rather than the default Fast DDS: see cyclonedds_no_shm.xml,
# which also explains why shared memory has to be off.
FROM sjtu_drone_nadav:humble_ros2

LABEL org.opencontainers.image.title="sjtu_drone_sparx"
LABEL org.opencontainers.image.description="SJTU Gazebo simulator with CycloneDDS for ROS1 bridge interop"

# --no-install-recommends keeps this to the rmw layer and its libddsc runtime
# (~2 MB) instead of pulling a second copy of the ROS desktop metapackages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

COPY cyclonedds_no_shm.xml /etc/cyclonedds/no_shm.xml

# Defaults, not a lock: `docker run -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
# still overrides them, which is what bringup_world.sh --rmw fastrtps does. The
# point of setting them here is that a bare `docker run` on this image, or a
# `docker exec` into a running one, lands on the same DDS as the bridge instead
# of silently falling back to Fast DDS and seeing nothing.
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=file:///etc/cyclonedds/no_shm.xml
