# TheAgency — System Architecture

Paste the Mermaid block into **[mermaid.live](https://mermaid.live)** or into **Draw.io** via
*Arrange → Insert → Advanced → Mermaid* to export as SVG / PNG.

---

```mermaid
flowchart TD

    SRC["Source\nOnline — RTSP + WebSocket\nOffline — recorded session"]

    DEPTH["Depth\nDA3 TRT · pre-computed .npy\nRGB → metric depth image"]

    LOC["Localization\nAprilTag · Optical Flow · AMCL"]

    MAP["Mapping / Perception\nPointcloud · 2D occupancy · 3D octomap"]

    PLAN["Planning\nRRT* · BIT* · Smoother · Tracker\nDemo Manager"]

    CTRL["Drone Control\ncmd_nav WebSocket\nROS1 bridge · Falcon adapter"]

    DRONE(["Drone\nXTEND · ROBOTICAN · SJTU"])

    SRC     -->|"rgb · bearing"| DEPTH
    SRC     -->|"rgb · bearing"| LOC
    DEPTH   -->|"depth"| LOC
    DEPTH   -->|"depth · pointcloud"| MAP
    LOC     -->|"pose · TF"| MAP
    LOC     -->|"pose"| PLAN
    MAP     -->|"occupancy"| PLAN
    PLAN    -->|"cmd_vel"| CTRL
    CTRL    --> DRONE
    DRONE   -->|"telemetry"| SRC
```