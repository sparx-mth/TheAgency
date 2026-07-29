# Demo 4 — Full Pipeline

---

## PART A — Map the Room

```mermaid
flowchart TD

%% ── DRONE ───────────────────────────────────────────
D1(["XTEND Drone 1\n192.0.0.15"])

%% ── P1: DIR PUBLISHER ───────────────────────────────
subgraph P1["① Dir Publisher   robots/XTEND/online_nav_bridge_dir_publisher.py"]
    P1A["RTSP decode → BGR\npreprocess → 504×294\nwrite frame_N.tmp → rename .jpg  (atomic)\nkeep rolling 30 frames"]
    P1B["publish /xtend/rgb_frame_path\nformat: path  sec  nanosec"]
    P1C["recv ROBOT_STATUS\npublish /xtend/bearing  Float32\npublish /xtend/local_telemetry  Odometry"]
    P1D["sub /xtend/cmd_nav  String JSON\n→ arm / takeoff / land / rotate / stop\n→ WebSocket VIRTUAL_CONTROLLER"]
end

D1 -->|"RTSP H.264"| P1A
D1 <-->|"WebSocket"| P1C
P1A --> P1B
P1B --> TMP1

TMP1["/tmp/xtend_frames/\nframe_N.jpg  rolling 30"]

%% ── P2: DEPTH PROCESSOR ─────────────────────────────
subgraph P2["② Depth Processor   tasks/mapping/ros2/depth_processor_node.py"]
    P2A["cv2.imread(path)\nDA3TensorRTModel.infer_all()\n→ metric depth via LUT  0.2–8.0 m\nwrite frame_N.tmp → rename .npy  (atomic)"]
    P2B["publish /xtend/depth_frame_path\npublish /xtend/depth_m  32FC1"]
end

P1B -->|"/xtend/rgb_frame_path"| P2A
TMP1 -->|"read JPEG"| P2A
P2A --> P2B
P2A --> TMP2

TMP2["/tmp/xtend_depth/\nframe_N.npy  float32 metres"]

%% ── P3: LOCALIZATION ────────────────────────────────
subgraph P3["③ Live Localization   demos/Demo_No4/online_rgbd_localization_node.py"]
    P3A["read JPEG + NPY from /tmp dirs\n① AprilTag detect → solvePnP → world_T_cam\n② RGB-D odom delta  (Open3D)\n③ depth scale EMA from tag Z\npublish /xtend/pose  PoseStamped"]
end

P1B -->|"/xtend/rgb_frame_path"| P3A
P2B -->|"/xtend/depth_frame_path"| P3A
TMP1 -->|"read JPEG"| P3A
TMP2 -->|"read NPY"| P3A

%% ── P4: FLIGHT ORCHESTRATOR ─────────────────────────
subgraph P4["④ Flight Orchestrator   demos/Demo_No4/xtend_dome_main.py"]
    P4A["arm → takeoff"]
    P4B["rotate_degrees(360°)\n4×90° chunks\nread /xtend/pose for start_yaw\npub rotate_left → /xtend/cmd_nav\naccumulate angle_step()  (wrapped diff)\nstop when chunk done  30 s timeout"]
    P4C["_capture_loop()\ntrigger: 0.5 s  OR  30° yaw bucket\ncopy frame_N.jpg  → R2_ts.jpg\ncopy frame_N.npy  → R2_ts.npy\nwrite R2_ts.json  pose, vlm_text:null, nanoowl:null"]
    P4D["land → disarm\n_update_latest_symlink()\ncaptures/latest → session_dir"]
    P4A --> P4B --> P4C --> P4D
end

P3A -->|"/xtend/pose"| P4B
P3A -->|"pose → json sidecar"| P4C
TMP1 -->|"copy jpg"| P4C
TMP2 -->|"copy npy"| P4C
P4B -->|"/xtend/cmd_nav"| P1D
P4D -->|"/xtend/cmd_nav land/disarm"| P1D

SESSION["captures/2026_06_24_12_05/\nR2_ts.jpg  R2_ts.npy  R2_ts.json\n× 44 frames\n\ncaptures/latest/ → symlink"]
P4C --> SESSION
P4D --> SESSION

%% ── P5: COMM MANAGER  (NanoLLM_VILA_and_OWL) ────────
subgraph P5["⑤ Inference   GIT/NanoLLM_VILA_and_OWL/comm_manager_vllm.py"]
    P5A["watch captures/latest/*.jpg\nfor each image:"]
    P5B["call_vlm()\nPOST :8080/v1/chat/completions\n→ A chair, A box, A gun..."]
    P5C["caption_to_owl_prompts()\n→ chair, box, gun, ..."]
    P5D["_post_nanoowl_multipart()\nPOST :5060/infer  image + prompts\n→ bbox, label, score per object"]
    P5E["write json.nanoowl = prompts + result\nwrite json.entries = vlm_text\nrender *_ann.jpg"]
    P5A --> P5B --> P5C --> P5D --> P5E

    VD["vLLM Docker\nQwen3-VL-4B :8080"]
    OD["NanoOWL Docker\nowl_patch32.engine :5060"]
    P5B <-->|"HTTP"| VD
    P5D <-->|"HTTP"| OD
end

SESSION -->|"latest/*.jpg"| P5A
P5E --> SESSION

%% ── P6: ROOM MAPPER ─────────────────────────────────
subgraph P6["⑥ Room Mapper   demos/Demo_No4/room_mapper/run_room_mapper.py"]
    P6A["iter_frames(data_dir)\n→ FrameRecord list  (lazy jpg+npy+json)"]
    P6B["per frame:\nPoseFuser.update(bgr, depth_m)\n→ world_T_cam, depth_scale\nupdate_grid_from_depth()\n→ LogOddsGridCostmap"]
    P6C["load_labels_from_session()\nparse json.nanoowl.result.detections\nfilter min_score 0.25"]
    P6D["place_objects()  bbox → p_world via depth\nsnap_objects_to_free_space()\ncluster_objects(radius 2 m)\nflag_beyond_tags / outside_map\nfilter suspicious"]
    P6E["render_map()\n→ room_map.png\n→ room_map.json  label, x, y, z per object"]
    P6A --> P6B --> P6D
    P6A --> P6C --> P6D --> P6E
    P6B --> P6E
end

SESSION -->|"jpg + npy + json"| P6A

ROOMMAP["room_map.png\nroom_map.json\nlabel, world_x, world_y, world_z"]
P6E --> ROOMMAP

%% ── STYLES ───────────────────────────────────────────
classDef drone  fill:#1e2a40,stroke:#4a9eff,color:#cce0ff
classDef tmp_   fill:#1a1a2e,stroke:#9999ee,color:#ddddff
classDef node_  fill:#1a2a1a,stroke:#55cc55,color:#ccffcc
classDef infer_ fill:#2a1a3a,stroke:#bb66ff,color:#eeddff
classDef disk_  fill:#2e2a14,stroke:#ddbb44,color:#fff8cc
classDef out_   fill:#2a1a12,stroke:#ff8833,color:#ffddcc

class D1 drone
class TMP1,TMP2 tmp_
class P1A,P1B,P1C,P1D,P2A,P2B,P3A,P4A,P4B,P4C,P4D,P6A,P6B,P6C,P6D,P6E node_
class P5A,P5B,P5C,P5D,P5E,VD,OD infer_
class SESSION disk_
class ROOMMAP out_
```

---

## PART B — Autonomous Navigation & Landing

```mermaid
flowchart TD

ROOMMAP2["room_map.json\nlabel, world_x, world_y, world_z"]

%% ── TARGET SELECTION ────────────────────────────────
subgraph TS["① Target Selection"]
    TS1["read room_map.json\npick landing target object\ne.g. label='box' with highest confidence\n→ target world_x, world_y"]
end

ROOMMAP2 --> TS1

%% ── DRONE 2 ──────────────────────────────────────────
D2(["XTEND Drone 2\nnav drone"])

%% ── N1: DIR PUBLISHER ───────────────────────────────
subgraph N1["② Dir Publisher   (same as Part A)"]
    N1A["RTSP → /tmp/xtend_frames/  frame_N.jpg\npublish /xtend/rgb_frame_path\npublish /xtend/bearing\nsub /xtend/cmd_nav → WebSocket"]
end

D2 <-->|"RTSP + WebSocket"| N1A
N1A --> NTMP1
NTMP1["/tmp/xtend_frames/  frame_N.jpg"]

%% ── N2: DEPTH ───────────────────────────────────────
subgraph N2["③ Depth Processor   (same as Part A)"]
    N2A["read JPEG → DA3 TRT\n→ /tmp/xtend_depth/  frame_N.npy\npublish /xtend/depth_frame_path\npublish /xtend/depth_m  32FC1"]
end

N1A -->|"/xtend/rgb_frame_path"| N2A
NTMP1 -->|"read JPEG"| N2A
N2A --> NTMP2
NTMP2["/tmp/xtend_depth/  frame_N.npy"]

%% ── N3: LOCALIZATION ────────────────────────────────
subgraph N3["④ RGB Localization   tasks/localization/apriltag_triangulation_node.py"]
    N3A["read JPEG from /xtend/rgb_frame_path\ndetect tag36h11 → solvePnP\nestimate_camera_pose_from_tags()\nEMA smooth  α=0.1\npublish /xtend/april_tag_pose  PoseStamped"]
end

N1A -->|"/xtend/rgb_frame_path"| N3A
NTMP1 -->|"read JPEG"| N3A

%% ── N4: POTENTIAL FIELD ─────────────────────────────
subgraph N4["⑤ Potential Field   core/mapping/costmap/"]
    N4A["PotentialMapper.update(point_cloud)\n─────────────────\n/xtend/depth_m → backproject → occupancy EMA  α=0.30\nPotentialFieldLayer.compute_from_prob_grid()\n  distanceTransform → Gaussian σ=0.6 m → U_rep\n  U_att = parabola toward target goal\n  U_total = 3.5·U_rep + 1.0·U_att\n  gradient descent → fwd+left vector\npublish /local_nav_vector  Vector3Stamped\npublish /map_local  OccupancyGrid"]
end

N2A -->|"/xtend/depth_m"| N4A
N3A -->|"/xtend/april_tag_pose  set_goal"| N4A
TS1 -->|"target world_x, world_y"| N4A

%% ── N5: PLANNER ─────────────────────────────────────
subgraph N5["⑥ Planner  (Falcon or InterNAV)"]
    N5F["Falcon\ntasks/planning/falcon/\n─────────────────\ndepth → voxel map\n→ /falcon/bev_2d  OccupancyGrid\nA* path plan\n→ /cmd_vel  Twist"]
    N5I["InternVLA-N1\ntasks/planning/vlas/internvla_n1/\n─────────────────\nRGB + instruction → InternVLA-N1\n→ &lt;ns&gt;/navigation/action  discrete\n→ &lt;ns&gt;/navigation/waypoint  (x,y)"]
end

N4A -->|"/local_nav_vector  /map_local"| N5F
N1A -->|"/xtend/rgb_frame_path"| N5I
N2A -->|"/xtend/depth_frame_path"| N5F
TS1 -->|"nav instruction + target label"| N5I

%% ── N6: CMD ROUTING ─────────────────────────────────
subgraph N6["⑦ Command Routing"]
    N6A["TwistToCmdNavConverter\nrobots/XTEND/adapters/\n─────────────────\nlinear.x  → forward/backward  value\nangular.z → rotate_left/right  value\n0.3 m/s → 400,  0.65 rad/s → 1000"]
    N6B["XtendDroneDemoManager\ndemos/Demo_No4/xtend_drone_demo_manager.py\n─────────────────\nmodes: FLY_STRAIGHT / TURNING\n       VISUAL_SERVOING / FINISH\nroute /cmd_vel → /xtend/cmd_nav\nFINISH: stop → land → disarm"]
    N6A --> N6B
end

N5F -->|"/cmd_vel  Twist"| N6A
N5I -->|"/cmd_vel  Twist"| N6A
N4A -->|"/local_nav_vector → Twist"| N6A

N6B -->|"/xtend/cmd_nav  String JSON"| N1A

%% ── LANDING ─────────────────────────────────────────
LAND["Demo mode = FINISH\nstop → land → disarm\nDrone 2 landed on target object"]
N6B -->|"FINISH sequence"| LAND

%% ── STYLES ───────────────────────────────────────────
classDef drone  fill:#1e2a40,stroke:#4a9eff,color:#cce0ff
classDef tmp_   fill:#1a1a2e,stroke:#9999ee,color:#ddddff
classDef node_  fill:#1a2a1a,stroke:#55cc55,color:#ccffcc
classDef plan_  fill:#2a2a12,stroke:#ddcc33,color:#fffacc
classDef route_ fill:#2a1a12,stroke:#ff8833,color:#ffddcc
classDef out_   fill:#12261a,stroke:#44ff88,color:#ccffdd

class D2 drone
class NTMP1,NTMP2 tmp_
class N1A,N2A,N3A,N4A node_
class N5F,N5I plan_
class N6A,N6B,TS1 route_
class LAND out_
```

---

## Data transport summary

| Signal | Type | Producer → Consumer |
|---|---|---|
| `/xtend/rgb_frame_path` | `String` `"{path} sec nsec"` | Dir Publisher → Depth, Localization, InternNAV |
| `/xtend/depth_frame_path` | `String` `"{path} sec nsec"` | Depth Processor → Localization, Falcon |
| `/tmp/xtend_frames/frame_N.jpg` | JPEG file | Dir Publisher → all readers |
| `/tmp/xtend_depth/frame_N.npy` | NPY float32 | Depth Processor → all readers |
| `/xtend/depth_m` | `Image` 32FC1 | Depth Processor → PotentialMapper |
| `/xtend/bearing` | `Float32` rad | Dir Publisher → Localization |
| `/xtend/pose` | `PoseStamped` | Live Localization → Dome Orchestrator |
| `/xtend/april_tag_pose` | `PoseStamped` | AprilTag Node → PotentialMapper goal |
| `/local_nav_vector` | `Vector3Stamped` | PotentialMapper → cmd routing |
| `/falcon/bev_2d` | `OccupancyGrid` | Falcon BEV → A\* planner |
| `/cmd_vel` | `Twist` | Planner → TwistToCmdNavConverter |
| `/xtend/cmd_nav` | `String` JSON | Demo Manager → Dir Publisher → WebSocket |
