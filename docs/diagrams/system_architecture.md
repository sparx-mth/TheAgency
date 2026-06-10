# TheAgency — System Architecture

Paste the Mermaid block into **[mermaid.live](https://mermaid.live)** or into **Draw.io** via
*Arrange → Insert → Advanced → Mermaid* to export as SVG / PNG.

---

```mermaid
flowchart LR

    DEMO["Demo Manager\n─────────────\norchestrates mission\nTakeoff · Land · request plan"]

    BRIDGE["Online Bridge\n─────────────\nRTSP capture → rgb frames\nWebSocket ↔ Drone\nbearing from telemetry\ncmd_nav → Drone"]

    OFFLINE["Offline Source\nrecorded session\nrgb · bearing from CSV"]

    subgraph PERCEPTION["Perception"]
        DEPTH["Depth\nDA3 TRT · pre-computed .npy\nRGB → metric depth image"]
        LOC["Localization\nAprilTag · Optical Flow · AMCL"]
    end

    FALCON["Falcon  ROS1\n─────────────\n2D / 3D Occupancy\nRRT planner\n→ /cmd_vel Twist"]

    DRONE(["Drone\nXTEND · ROBOTICAN · SJTU"])

    BRIDGE  -->|"rgb · bearing"| PERCEPTION
    OFFLINE -->|"rgb · bearing"| PERCEPTION
    DEPTH   -->|"depth image"| FALCON
    LOC     -->|"pose"| FALCON
    FALCON  -->|"cmd_vel"| BRIDGE
    DRONE   -->|"telemetry"| BRIDGE
    BRIDGE  -->|"WebSocket"| DRONE

    DEMO -.->|"request plan"| FALCON
    DEMO  -->|"ARM · TAKEOFF · LAND"| BRIDGE
    DEMO -.->|"monitor"| PERCEPTION
```