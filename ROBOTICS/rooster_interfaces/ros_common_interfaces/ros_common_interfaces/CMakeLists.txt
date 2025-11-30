cmake_minimum_required(VERSION 3.5)
project(ros_common_interfaces)

# Default to C99
if (NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif ()

# Default to C++14
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)

# declare the message files to generate code for
set(msg_files
        "msg/LLA.msg"
        "msg/LLAStamped.msg"
        "msg/Float32Stamped.msg"
        )

set(srv_files
        "srv/SaveParameters.srv"
        )

set(action_files
        "action/FileTransfer.action"
        )

rosidl_generate_interfaces(${PROJECT_NAME}
        ${msg_files}
        ${srv_files}
        ${action_files}
        DEPENDENCIES std_msgs
        )

ament_package()
