cmake_minimum_required(VERSION 3.0.2)
project(falcon_pegasus)
find_package(catkin REQUIRED COMPONENTS
  rospy geometry_msgs nav_msgs sensor_msgs std_msgs quadrotor_msgs tf2_ros)
catkin_package()

# The two rospy nodes. Everything they compute lives outside this package:
# the wire protocol and the frame renderer are in the sparx_agency repo, which
# run_falcon_pegasus.sh mounts read-only on PYTHONPATH, so a host edit takes
# effect without rebuilding this image.
#
# cloud_reader.py is a helper the recorder IMPORTS rather than a node, so it is
# deliberately not listed here -- but it must still be mounted alongside the
# nodes, or the recorder dies on ImportError inside the container.
catkin_install_python(PROGRAMS
  scripts/pegasus_bridge_node.py
  scripts/map_recorder_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
install(DIRECTORY runs/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/runs OPTIONAL)
