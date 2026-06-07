cmake_minimum_required(VERSION 3.0.2)
project(falcon_adapter)
find_package(catkin REQUIRED COMPONENTS rospy geometry_msgs nav_msgs sensor_msgs std_msgs tf quadrotor_msgs)
catkin_package()

# FALCON adapter nodes (rospy). They import the ROS-free algorithms from the
# mounted sparx_agency repo (core.mapping.bev, core.planning.planners.astar,
# core.localization) and the sibling cloud_utils.py; run_falcon.sh mounts the
# repo on PYTHONPATH. cloud_utils.py is a helper module (imported, not run) so
# it is intentionally not listed here.
catkin_install_python(PROGRAMS
  scripts/bev_publisher_node.py
  scripts/mapping_sync_node.py
  scripts/bev_click_goal_node.py
  scripts/astar_planner_node.py
  scripts/waypoint_follower_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)