cmake_minimum_required(VERSION 3.5)
project(fcu_driver_interfaces)

# Default to C99
if (NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif ()

if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ros_common_interfaces REQUIRED)


# declare the message files to generate code for
set(msg_files
        "msg/LocalVelocityCommand.msg"
        "msg/BodyVelocityCommand.msg"
        "msg/UAVState.msg"
        "msg/StatusText.msg"
        "msg/GlobalWPCommand.msg"
        "msg/ActuatorControl.msg"
        "msg/CommandLong.msg"
        "msg/ManualControl.msg"
        "msg/VisionPositionEstimate.msg"
        "msg/DistanceSensor.msg"
        "msg/LocalPositionCommand.msg"
        "msg/GPS.msg"
        "msg/Battery.msg"
        "msg/FencePolygon.msg"
        "msg/ThrusterCommand.msg"
        "msg/AttitudeCommand.msg"
        )
# declare the services files to generate code for
set(srv_files
        "srv/Takeoff.srv"
        "srv/SetFCUParamFloat.srv"
        "srv/SetFCUParamsFloat.srv"
        "srv/SetFCUParamInt.srv"
        "srv/SetFCUParamsInt.srv"
        "srv/GetFCUParamFloat.srv"
        "srv/GetFCUParamInt.srv"
        "srv/SetFlightMode.srv"
        "srv/GetLogFilesList.srv"
        "srv/DownloadLogFile.srv"
        "srv/GetGeographicData.srv"
        "srv/SetGeoFence.srv"
        "srv/SetRallyPoints.srv"
        )
# declare the actions files to generate code for
set(action_files
        "action/Calibration.action"
        )
rosidl_generate_interfaces(${PROJECT_NAME}
        ${msg_files}
        ${srv_files}
        ${action_files}
        DEPENDENCIES std_msgs geometry_msgs ros_common_interfaces
        )

if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # uncomment the line when a copyright and license is not present in all source files
    #set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # uncomment the line when this package is not in a git repo
    #set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

ament_package()
