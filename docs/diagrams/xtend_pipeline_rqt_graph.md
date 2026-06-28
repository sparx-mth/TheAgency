# XTEND Pipeline — rqt_graph Style Flowchart

Source: `sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_pipeline_launcher_ui_auto_with_manual_modes.py`

Paste the Mermaid block into **[mermaid.live](https://mermaid.live)** or into **Draw.io** via
*Arrange → Insert → Advanced → Mermaid* to export as SVG/PNG.

---

```mermaid
graph LR

%% ─────────────────────────── PC BASE STATION ───────────────────────────
subgraph PC["PC — Base Station  (ROS Jazzy · ROS_DOMAIN_ID=5)"]
    UI(["XtendPipelineLauncher\nTkinter UI\nJetson SSH: 192.0.0.89"])
    PCUI["xtend_pc_ui\nui.py\n[item 10]"]
end

%% ─────────────────────────── JETSON ORIN AGX ───────────────────────────
subgraph Jetson["Jetson Orin AGX — 192.0.0.89  (ROS Humble · ROS_DOMAIN_ID=5)"]

    subgraph AutoOrch["AUTO Orchestrator  ·  tmux: xtend_auto_launch"]
        Auto(["AUTO pipeline script\n① Bridge  ② Depth  ③ Twist-conv  ④ Demo-mgr\n→ ARM → TAKEOFF → sleep 30s\n→ ⑥ AprilTag  →  ⑧ Falcon\n→ publish demo_mode fly_straight"])
    end

    subgraph AutoOffline["AUTO Offline  ·  tmux: xtend_auto_offline"]
        OffAuto(["OFFLINE pipeline script\n① Replay  ② Depth  ③ AprilTag"])
    end

    subgraph CoreSessions["Core tmux Sessions  (Jetson)"]
        direction TB
        Bridge["xtend_bridge  [1]\nonline_nav_bridge_dir_publisher.py\n· 504×294 resize frames → /tmp/xtend_frames\n· owns XTEND WebSocket"]

        OffReplay["xtend_offline_replay  [1-ALT]\noffline_frame_dir_publisher.py\nreads /tmp/xtend_capture @ 10 Hz loop"]

        Depth["xtend_depth  [2]\ndepth_processor_node.py\nDA3METRIC-LARGE FP16 504×294\nfocal scale ÷300 · clip 0.2–8 m\nsaves .npy → /tmp/xtend_depth"]

        TwistConv["xtend_twist_converter  [3]\nextend_twist_to_cmd_nav.py\nlinear.x=0.3 → fwd thrust 400\nmax 600 · timeout 1.5 s"]

        DemoMgr["xtend_demo_manager  [4]\nextend_drone_demo_manager.py\nstates: idle · fly_straight\nturning · visual_servoing · finish\ndisarm-delay 8 s"]

        AprilTag["xtend_april_tag  [6]\napriltag_triangulation_node\ntag36h11 · size 0.13 m\nnew_map.yaml · 504×294 calib"]
    end

    subgraph OptSessions["Optional tmux Sessions"]
        TwistReplay["xtend_twist_replayer  [5]\ntwist_replayer.py\nJSONL log → /cmd_vel"]
    end

    subgraph DockerStack["Docker Stack — ROS1 Noetic"]
        ROSCore["roscore\ncontainer"]
        RosBridge["ros1_bridge\ncontainer\nrun_bridge.sh\nROS1 ↔ ROS2"]
        Falcon["falcon\ncontainer  [12]\nfalcon_adapter\nreal_drone.launch\nmap:=office"]
        BEV["planner_bev_goal  [15]\nbev_click_goal.py\n(optional)"]
        RViz["planner_rviz  [14]\nrviz.launch\n(optional)"]
    end

end

%% ─────────────────────────── DRONE HARDWARE ───────────────────────────
subgraph DroneHW["XTEND Drone"]
    WS["XTEND WebSocket\n(onboard)"]
    Drone(["Flight Controller\nARM · TAKEOFF · NAV\nLAND · DISARM"])
end

%% ═══════════════════════════════════════════════
%% UI ACTIONS  (all over SSH)
%% ═══════════════════════════════════════════════
UI -->|"SSH → spawn tmux:\nextend_auto_launch"| Auto
UI -->|"SSH → spawn tmux:\nextend_auto_offline"| OffAuto
UI -->|"SSH → spawn tmux\nper LaunchItem"| CoreSessions
UI -->|"SSH: ros2 topic pub\n/xtend/demo_mode_request\n{mode, source, reason}"| DemoMgr

%% ═══════════════════════════════════════════════
%% AUTO ORCHESTRATOR  →  tmux session spawning
%% ═══════════════════════════════════════════════
Auto -->|"start_tmux ①\nwait /xtend/rgb_frame_path"| Bridge
Auto -->|"start_tmux ②\nwait /xtend/depth_m"| Depth
Auto -->|"start_tmux ③\nsleep 2 s"| TwistConv
Auto -->|"start_tmux ④\nwait /xtend/demo_mode"| DemoMgr
Auto -->|"ros2 topic pub --once\n/xtend/cmd_nav  ARM"| Bridge
Auto -->|"ros2 topic pub --once\n/xtend/cmd_nav  TAKEOFF\nsleep 30 s"| Bridge
Auto -->|"start_tmux ⑥\nwait /xtend/april_tag_pose"| AprilTag
Auto -->|"docker exec falcon ⑧\nroslaunch falcon_adapter\nreal_drone.launch"| Falcon
Auto -->|"step 9: ros2 pub\n/xtend/demo_mode_request\nfly_straight"| DemoMgr

OffAuto -->|"start_tmux ①"| OffReplay
OffAuto -->|"start_tmux ②"| Depth
OffAuto -->|"start_tmux ③"| AprilTag

%% ═══════════════════════════════════════════════
%% DRONE → BRIDGE  (inbound telemetry + video)
%% ═══════════════════════════════════════════════
Drone -->|"telemetry: IMU, bearing\nonboard sensors"| WS
WS -->|"RTSP video stream\nraw frames"| Bridge

%% ═══════════════════════════════════════════════
%% RGB FRAME PATH CHAIN
%% ═══════════════════════════════════════════════
Bridge -->|"/xtend/rgb_frame_path\nstd_msgs/String  (file path)"| Depth
Bridge -->|"/xtend/rgb_frame_path"| AprilTag
Bridge -->|"/xtend/bearing\n/xtend/local_telemetry"| DemoMgr

OffReplay -.->|"/xtend/rgb_frame_path\n[offline mode]"| Depth
OffReplay -.->|"/xtend/rgb_frame_path"| AprilTag

%% ═══════════════════════════════════════════════
%% DEPTH OUTPUTS
%% ═══════════════════════════════════════════════
Depth -->|"/xtend/depth_frame_path\npath to .npy"| RosBridge
Depth -->|"/xtend/depth_m" | RosBridge
%% ═══════════════════════════════════════════════
%% CMD_NAV FLOW  →  drone
%% ═══════════════════════════════════════════════
PCUI -->|"/cmd_vel\ngeometry_msgs/Twist"| TwistConv
TwistReplay -.->|"/cmd_vel Twist\n[optional]"| TwistConv
TwistConv -->|"/xtend/cmd_nav\nJSON String\n{action, value}"| Bridge
DemoMgr -->|"/xtend/cmd_nav\nARM · LAND · DISARM\n{action, value}"| Bridge
Bridge -->|"WebSocket JSON\n/xtend/cmd_nav forward"| WS
WS -->|"flight commands"| Drone

%% ═══════════════════════════════════════════════
%% DEMO MODE STATE MACHINE
%% ═══════════════════════════════════════════════
DemoMgr -->|"/xtend/demo_mode\nstd_msgs/String\nidle·fly_straight·finish…"| Falcon

%% ═══════════════════════════════════════════════
%% APRILTAG  →  PLANNER
%% ═══════════════════════════════════════════════
AprilTag -->|"/xtend/april_tag_pose\ngeometry_msgs/PoseStamped"| RosBridge

%% ═══════════════════════════════════════════════
%% PLANNER  →  TWIST (via ROS1↔ROS2 bridge)
%% ═══════════════════════════════════════════════
Falcon -->|"/cmd_vel  (ROS1 Noetic)"| RosBridge
RosBridge -->|"/cmd_vel  (ROS2 Humble)"| TwistConv
ROSCore --- RosBridge
Falcon --- BEV
Falcon --- RViz
RosBridge ---> |"/depth_m " | Falcon
RosBridge ---> |"/xtend/april_tag_pose\ngeometry_msgs/PoseStamped" | Falcon
```

---

## Legend

| Style | Meaning |
|---|---|
| Solid `-->` | Primary, always-on data flow |
| Dashed `-.->` | Optional / offline-mode path |
| `---` | Same-container grouping (no topic) |

## AUTO startup sequence

```
UI ──SSH──► xtend_auto_launch tmux
               │
               ①  xtend_bridge          waits /xtend/rgb_frame_path
               ②  xtend_depth           waits /xtend/depth_m
               ③  xtend_twist_converter (sleep 2 s)
               ④  xtend_demo_manager    waits /xtend/demo_mode
               │
               ├── ros2 pub /xtend/cmd_nav  ARM
               ├── ros2 pub /xtend/cmd_nav  TAKEOFF
               └── sleep 30 s  (stabilisation)
               │
               ⑥  xtend_april_tag       waits /xtend/april_tag_pose
               ⑧  docker exec falcon  → roslaunch falcon_adapter real_drone.launch
               └── ros2 pub /xtend/demo_mode_request  fly_straight
```

## Key drone parameter chain

```
/cmd_vel (Twist)
  → xtend_twist_converter
    → /xtend/cmd_nav  {action, value}   (linear.x=0.3 → fwd thrust 400, max 600)
      → online_nav_bridge_dir_publisher
        → XTEND WebSocket JSON
          → Flight Controller
```