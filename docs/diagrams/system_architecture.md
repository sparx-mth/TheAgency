# TheAgency — System Architecture

Paste the Mermaid block into **[mermaid.live](https://mermaid.live)** or into **Draw.io** via
*Arrange → Insert → Advanced → Mermaid* to export as SVG / PNG.

---

```mermaid
flowchart TD

%% ══════════════════════════════════════════════════════════
%% SENSOR / DATA INPUT
%% Online and Offline are interchangeable — same topics out.
%% Depth sits here because it can be replaced by a 3D camera.
%% ══════════════════════════════════════════════════════════
subgraph SENSOR["Sensor / Data Input"]

    subgraph RGB_SRC["RGB Source  ·  online or offline"]
        ONLINE(["🟢 Online\nonline_nav_bridge_dir_publisher\nXTEND WebSocket + RTSP"])
        OFFLINE(["🔵 Offline\noffline_frame_dir_publisher\nextend_da3_take session"])
    end

    subgraph DEPTH_SRC["Depth  ·  RGB + model  or  3D camera"]
        D_MODEL(["DA3 TRT\ndepth_processor_node\nRGB → metric depth"])
        D_NPY(["Pre-computed .npy\nJetson-saved"])
        D_3D(["3D Camera\n— future —"])
    end

end

%% ══════════════════════════════════════════════════════════
%% LOCALIZATION
%% ══════════════════════════════════════════════════════════
subgraph LOC["Localization  ·  core/localization/"]
    L_AT(["AprilTag\nTriangulation\napriltag_triangulation_node"])
    L_OF(["Optical Flow\n+ Depth velocity\nvelocity_integrator"])
    L_AMCL(["AMCL\nParticle Filter\namcl_provider"])
    L_OUT(["localization_node\n/xtend/localization\nmap → xtend_camera TF"])
    L_AT & L_OF & L_AMCL --> L_OUT
end

%% ══════════════════════════════════════════════════════════
%% MAPPING
%% ══════════════════════════════════════════════════════════
subgraph MAP["Mapping  ·  core/mapping/"]
    M_PC["depth_to_pointcloud_node\n/xtend/pointcloud"]
    M_2D["occupancy_from_pointcloud_node\n2D log-odds OccupancyGrid\nIntegratedMap + Numba raycasting"]
    M_3D["octomap_server_node\n3D voxel map"]
    M_PC --> M_2D & M_3D
end

%% ══════════════════════════════════════════════════════════
%% PLANNING & EXECUTION
%% ══════════════════════════════════════════════════════════
subgraph PLAN["Planning & Execution  ·  core/planning/"]
    P_PLNR["Path Planner\nRRT* · BIT*"]
    P_SMTH["Smoother\nHermite · MinSnap"]
    P_TRKR["Tracker\nPID / MPC"]
    P_DEMO["Demo Manager\nxtend_demo_manager\nidle · fly_straight · turning\nvisual_servoing · finish"]
    P_PLNR --> P_SMTH --> P_TRKR --> P_DEMO
end

%% ══════════════════════════════════════════════════════════
%% XTEND ROBOT ADAPTER
%% ══════════════════════════════════════════════════════════
subgraph XTEND_ADAPTER["XTEND Adapter  ·  robots/XTEND/"]

    subgraph XTEND_ROS2["ROS2 Humble — Jetson Orin AGX"]
        A_TWIST["xtend_twist_converter\nextend_twist_to_cmd_nav.py\n/cmd_vel → /xtend/cmd_nav JSON"]
        A_BRIDGE["xtend_bridge\nonline_nav_bridge_dir_publisher\nWebSocket owner · bearing publisher"]
    end

    subgraph XTEND_DOCKER["Docker Stack — ROS1 Noetic  (Jetson)"]
        D_ROSCORE["roscore"]
        D_ROS1BR["ros1_bridge\nrun_bridge.sh\nROS1 ↔ ROS2  /cmd_vel"]
        D_FALCON["falcon container\nfalcon_adapter\nreal_drone.launch"]
        D_ROSCORE --> D_ROS1BR --> D_FALCON
    end

end

%% ══════════════════════════════════════════════════════════
%% OTHER ROBOT ADAPTERS
%% ══════════════════════════════════════════════════════════
subgraph OTHER_ROBOTS["Other Robot Adapters  ·  robots/"]
    R_RB["ROBOTICAN\ncmd_vel → ROS2"]
    R_SJ["SJTU / Gazebo\ncmd_vel → ROS2"]
end

%% ══════════════════════════════════════════════════════════
%% XTEND DRONE (hardware)
%% ══════════════════════════════════════════════════════════
subgraph DRONE["XTEND Drone"]
    DR_WS["WebSocket onboard\ntelemetry · bearing · IMU"]
    DR_FC["Flight Controller\nARM · TAKEOFF · NAV\nLAND · DISARM"]
end

%% ══════════════════════════════════════════════════════════
%% DATA FLOW
%% ══════════════════════════════════════════════════════════

%% RGB source → depth + localization
ONLINE  -->|"rgb_frame_path · bearing"| D_MODEL
OFFLINE -->|"rgb_frame_path · bearing"| D_MODEL
ONLINE  -->|"rgb_frame_path · bearing"| L_AT & L_OF
OFFLINE -->|"rgb_frame_path · bearing"| L_AT & L_OF

%% Depth → mapping + localization
D_MODEL & D_NPY & D_3D -->|"depth_frame_path\n/xtend/pointcloud"| M_PC
D_MODEL & D_NPY & D_3D -->|"depth_frame_path"| L_OF & L_AMCL

%% Localization → mapping (TF) + planning
L_OUT -->|"TF · /xtend/localization"| MAP
L_OUT -->|"/xtend/localization"| PLAN

%% Map → planning
M_2D & M_3D -->|"/xtend/occupancy_grid"| PLAN

%% Planning → adapters
P_DEMO -->|"/cmd_vel  ROS2"| A_TWIST
P_DEMO -->|"/cmd_vel  ROS2"| D_ROS1BR
P_DEMO -->|"/cmd_vel"| OTHER_ROBOTS

%% XTEND adapter → drone
A_TWIST -->|"/xtend/cmd_nav  JSON"| A_BRIDGE
A_BRIDGE -->|"WebSocket JSON"| DR_WS
D_FALCON -->|"flight commands"| DR_FC

%% Drone → online bridge (telemetry back)
DR_WS -->|"bearing · IMU\ntelemetry"| ONLINE
```