cmake_minimum_required(VERSION 3.5)
project(rooster_handler_interfaces)

# Default to C++14
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

find_package(ament_cmake REQUIRED)

## find the ROS message code generators
#find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ros_common_interfaces REQUIRED)

# declare the message files to generate code for
set(msg_files
        "msg/RoosterState.msg"
        "msg/RoostersState.msg"
        "msg/StatusText.msg"        
        "msg/KeepAlive.msg"
        )
set(srv_files
    # "srv/SetRoosterFlightMode.srv"
    "srv/RoosterAction.srv"
    )

rosidl_generate_interfaces(${PROJECT_NAME}
        ${msg_files}
        ${srv_files}
        DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
        )

#ament_export_dependencies(rosidl_default_runtime)

ament_package()
