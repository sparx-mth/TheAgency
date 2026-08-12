cmake_minimum_required(VERSION 3.0.2)
project(falcon_adapter)
find_package(catkin REQUIRED COMPONENTS rospy geometry_msgs nav_msgs sensor_msgs std_msgs tf quadrotor_msgs)
catkin_package()

# FALCON adapter nodes (rospy). They import the ROS-free algorithms from the
# mounted sparx_agency repo (core.mapping.*, core.planning.*, core.localization)
# and the robot-specific bits (robots.XTEND.demo_mode); run_falcon.sh mounts the
# repo on PYTHONPATH. cloud_utils.py, pure_pursuit_follower.py, thinking.py and
# thought_journal.py are helper modules (imported by the nodes, not run) so they
# are intentionally not listed here -- run_falcon.sh bind-mounts them alongside
# the nodes.
catkin_install_python(PROGRAMS
  scripts/falcon_adapter_node.py
  scripts/sensor_gate_node.py
  scripts/bev_publisher_node.py
  scripts/mapping_sync_node.py
  scripts/bev_click_goal_node.py
  scripts/astar_planner_node.py
  scripts/navdp_click_node.py
  scripts/combination_planner_node.py
  scripts/astar_navdp_fallback_node.py
  scripts/hybrid_planner_node.py
  scripts/path_corrector_node.py
  scripts/trajectory_simplifier_node.py
  scripts/waypoint_follower_node.py
  scripts/pose_adapter_node.py
  scripts/sim_adapter_node.py
  scripts/mission_director_node.py
  scripts/cmd_vel_gate_node.py
  scripts/lost_localization_node.py
  scripts/falcon_exploration_follower_node.py
  scripts/depth_debug.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)