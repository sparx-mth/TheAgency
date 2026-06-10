# TheAgency — System Architecture

Paste the Mermaid block into **[mermaid.live](https://mermaid.live)** or into **Draw.io** via
*Arrange → Insert → Advanced → Mermaid* to export as SVG / PNG.

---

```mermaid
flowchart LR

    DEMO["Demo Manager\n─────────────\norchestrates mission\nTakeoff · Land\nrequest plan"]

    subgraph PERCEPTION["Perception"]
        SRC["Source\nOnline — RTSP + WebSocket\nOffline — recorded session"]
        DEPTH["Depth\nDA3 TRT · pre-computed .npy\nRGB → metric depth image"]
        LOC["Localization\nAprilTag · Optical Flow · AMCL"]
    end

    FALCON["Falcon  ROS1\n─────────────\n2D / 3D Occupancy\nRRT planner\n→ /cmd_vel Twist"]

    BRIDGE["Online Bridge\n─────────────\nWebSocket  cmd_nav\n(single drone connection)"]

    DRONE(["Drone\nXTEND · ROBOTICAN · SJTU"])

    SRC    -->|"rgb · bearing"| DEPTH & LOC
    DEPTH  -->|"depth image"| FALCON
    LOC    -->|"pose"| FALCON
    FALCON -->|"cmd_vel"| BRIDGE
    BRIDGE --> DRONE
    DRONE  -->|"telemetry · bearing"| SRC

    DEMO -.->|"request plan"| FALCON
    DEMO -->|"ARM · TAKEOFF · LAND"| BRIDGE
    DEMO -.->|"monitor"| PERCEPTION
```