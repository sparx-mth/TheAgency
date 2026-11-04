# 2D occupancy-grid room segmentation and doorway detection — literature survey

Reviewed 2026-10-11 on branch `feat/objnav-habitat-gibson-nadav`. **Literature only: no code was
changed and nothing below is implemented.** Three verification threads (classical, learned,
doorway/ObjectNav) read primary texts and repositories; items marked **[U]** could not be verified
this session and should be treated as hearsay. Detailed per-paper ledgers from the threads sit
in the session workspace (`~/.copilot/session-state/589bd69f-5e24-4174-9c91-78bd26102efa/files/`), not in the repo.

## 0. What we fly today, so the comparison has a baseline

Read from `core/mapping/topology/` and `LESSONS.md`, not from memory.

| Stage | Flown implementation | Literature ancestor |
|---|---|---|
| Free-mask healing | 3×3 binary closing + opening (`heal_free_mask`) | standard |
| Room partition | EDT → `peak_local_max` seeds (2.0 m separation, 0.6 m clearance) → `skimage.watershed` → merge basins by **absolute dynamics** 0.5 m → drop < 150 cells (`room_watershed.py`, `room_merge.py`) | Diosi 2005 / Bormann-dist watershed; merge is a "dynamics/prominence" test akin to MAORIS and BWEM but in metres, not ratios |
| Door source | **Exogenous**: surveyed YAML list (`discover_doors` only confirms a *listed* door has been seen), YOLO RGB-D door boxes with jamb-band depth and a 0.35–1.5 m width gate (`methods/doors.py`), and `trail_thresholds` — clearance dips ≤ 0.55 m along the drone's own track with ≥ 0.30 m rise on both sides within 1.5 m | Pronobis & Jensfelt 2012 (perceived doors as place property); `trail_thresholds` is a one-dimensional Thrun critical point along the trajectory, i.e. Tourani 2026's "traversal evidence" |
| Door → partition | disk carve, snapped to the nearest medial-axis **choke** within 0.9 m (half-width 0.25–0.8 m), plain 1.6 m disk when the snapped disk does not sever (`door_carve_mask`, `_severs`) | Thrun 1998 critical point; MORE's oriented-Gaussian cut is the sibling `room_separation.py` (untested in flight) |
| Persistence | `RoomRegistry` IoU 1:1 matching, 10-tick memory | no published equivalent (see §7) |
| Unflown sibling | `voronoi.py` + `room_separation.py`: boundary cost field → Voronoi skeleton → Gaussian door field → cut edges | MoMa-LLM / MORE (see §9 for the attribution error) |

**Nothing in the flown stack extracts a doorway from the grid itself.** The watershed's ridges *are*
the openings, but they are never emitted as door nodes with a position, normal and width; room
adjacency is computed from label borders after the fact. The recorded failures (LESSONS 2026-09-01,
2026-10-05 ×2) all stem from that: a missing or misplaced exogenous door, a choke under-measured near
the frontier because clearance there is distance-to-UNKNOWN (47 % of snapped disks non-severing, one
door toggling 44 times), a merge threshold that becomes a knife-edge the moment the door list is
removed (35 → 29 → 22 → 9 rooms at 0.20 / 0.30 / 0.50 m), and portals up to 24.75 m wide that no disk
can sever.

## 1. Taxonomy — every classical doorway detector is one of four signals

| # | Signal | Definition | Representative work | Output |
|---|---|---|---|---|
| S1 | **GVD critical point** | local minimum of clearance along a Voronoi/medial-axis edge, cut by the *critical line* to its two basis obstacle pixels | Thrun & Bücken 1996 / Thrun 1998; Beeson 2005 gateways; Wurm 2008; Friedman 2007; Bormann 2016 alg. 3 | point + normal (the critical line) + width (2 × clearance) |
| S2 | **Distance-transform saddle** | ridge between two basins of the EDT, kept if the saddle is "significantly" lower than both peaks | Diosi 2005; Bormann alg. 2; MAORIS 2018; HOV-SG 2024; ReasonNavi 2026; BWEM / Perkins 2010 (game AI) | ridge line; centre = max-clearance ridge cell, ends = min-clearance cells; width = 2 × saddle |
| S3 | **Persistent split under erosion / dilation** | a connected component that survives a sweep of structuring-element radius δ | Fabrizi & Saffiotti 2000; Buschka 2002; Bormann alg. 1; Area Graph 2019 (α-shape); Hydra 2022/2024; S-Graphs+ 2023 | opening = where the component broke; width ≈ 2δ at the break; no normal unless derived |
| S4 | **Gap in a wall line** | an under-covered edge in a line arrangement of dominant-direction walls | ROSE 2021 + ROSE² 2022; Luperto 2019–2022 layout prediction; Gómez 2019; Ericson & Jensfelt 2024; Tourani 2026 | segment on the wall plane: centre, normal = wall normal, width = gap length |

Learned methods (§2.2) either regress S1/S4-style door instances directly (Hiller 2019, QueSTMaps
2024) or label GVD nodes/patches as "doorway" (Friedman 2007 VRF, Goeddel & Olson 2016, Pronobis &
Zheng TopoNets 2019).

## 2. State-of-the-art survey

### 2.1 Classical methods

| Method (venue) | Signal | Doorway mechanism | Robustness reported | Cost | Online? | Code / licence |
|---|---|---|---|---|---|---|
| **Bormann et al.** "Room segmentation: survey, implementation, analysis" (ICRA 2016, DOI 10.1109/ICRA.2016.7487234) | S1–S3 + AdaBoost features | Voronoi: GVG point with exactly two closest obstacles, clearance local-min within `neighborhood = 280/clearance_px`, suppress points closer than 0.5×clearance keeping the larger basis-vector angle ("large angle occurs at doors"); 5 merge heuristics (merge rooms < 12.5 m², one neighbour, < 75 % wall) | 20 maps ± furniture. Furnished recall/precision: morph 84.6/90.5, dist 76.1/88.4, **Voronoi 86.6/94.5 (lowest variance, over-segments corridors)**, features 85.1/87.1 (most stable count, merges neighbours) | morph 1.6 s, dist 1.8 s, Voronoi 13 s, features 269 s (1 core, 0.05 m maps) | **No** — "assume a complete grid map"; unknown must be pre-mapped | `ipa320/ipa_coverage_planning` C++/ROS1; package.xml: LGPL academic, commercial by arrangement with Fraunhofer IPA |
| **Thrun & Bücken 1996 / Thrun 1998** (AAAI 96; AIJ 99) | S1 | critical point = Voronoi point whose clearance is a local min in ε-neighbourhood; critical line to basis points | 67 regions on one office map — raw criterion fires at every furniture gap (Wurm) | — | no | none |
| **Beeson, Jong, Kuipers** "Towards autonomous topological place detection using the EVG" (ICRA 2005) | S1 | Extended Voronoi Graph caps clearance at M so the skeleton hugs walls in open areas; **Reduced EVG = union of shortest paths between exits** prunes spurious branches; gateway = constriction (clearance local-min) on a branch leaving a junction's place neighbourhood, length 2·r_m normal to the branch | Flags: limited FOV "can create gateways … that may not exist in a fully explored LPM"; GVG "sensitive to range-sensing noise" | real-time on a scrolling local grid (no numbers) | **Yes** | none |
| **Wurm, Stachniss, Burgard** (IROS 2008) | S1 | Thrun critical points, pruned to: (1) degree-2 node, (2) with a degree-3 neighbour, (3) **not leading from known into unknown** | "reliably separate rooms and corridor segments"; raw Thrun "a lot of false positives in cluttered environments" | — | **Yes**, on partially explored maps | none |
| **Friedman, Pasula, Fox** "Voronoi Random Fields" (IJCAI 2007) | S1 + CRF | GVG nodes labelled {room, doorway, hallway, junction}; key feature: **size of the minimal loop through the node** — separates doorways from furniture gaps ("small loops caused by tables") | 91.7 % vs 85.2 % AdaBoost-only (4 Radish maps, LOO) | training ~2 min, inference < 1 min/map | no | re-implemented as `ipa_room_segmentation` alg. 5 (`doorway_points`), needs labelled training maps |
| **MAORIS** Mielle, Magnusson, Lilienthal (ICRA 2018, arXiv 1709.09899) | S2 | free-space "ripples": DT → disc-painted image → regions; absorb a region sharing > 40 % contour with one neighbour; merge if abs(Δ) ≤ 0.3·max; **door = ripple minimum significantly below both neighbours** | MCC **0.98** vs Voronoi 0.65 vs DuDe 0.70 on 20 *unfurnished* Bormann maps; parameter-stability curves; **"only adapted to environments without furniture"**; Area Graph measured it at 0.47 on cluttered/incomplete maps | mean 43 s (8.8 s without the two largest); Rust port claims 0.5 s **[U]** | no | `MalcolmMielle/MAORIS` C++ GPL-3; Rust port on Codeberg |
| **Area Graph** Hou, Yuan, Schwertfeger (ICAR 2019, arXiv 1910.01019); later osmAG 2023 | S3 (+VD) | CGAL outlier removal → **α-shape with α from robot width removes furniture** → Voronoi → drop near-wall edges → dead-end removal ×3 → α-shape on free space with disk **W = W_door + 0.1 m** (must satisfy W_door < W < W_corridor; 1.25 m default); passage = Voronoi edge crossing the α-boundary | MCC on cluttered/**incomplete** subset: Bormann-Voronoi 0.42, MAORIS 0.47, AG 0.61–0.67; "incomplete rooms as areas split to the unknown region"; fails on very wide doors and heavy clutter | not reported | no | `STAR-Center/areaGraph` C++ GPL-3 (CGAL/Qt4, last push 2020) |
| **ROSE / ROSE²** Kucner et al. (ICRA 2021); Luperto, Kucner, Tassi, Magnusson, Amigoni (RA-L 2022, arXiv 2203.03519) | S4 | ROSE: 2D DFT → dominant wall directions → declutter. ROSE²: Hough segments → wall clusters → representative lines (collinear merge if gap < a doorway width) → arrangement; edge weight = wall coverage, drop < 0.1; rooms = DBSCAN over faces; **Voronoi connectivity check splits rooms whose nodes disconnect**. No explicit door node — a door is a low-coverage edge | 10 real cluttered maps IoU **73.3** vs morph 51.2 / dist 54.7 / Voronoi 28.7; Bormann furnished P/R 93.5/91.0; **partial maps: IoU ≈ 80 flat across coverage 0.2–1.0** — the only published segmentation-vs-coverage curve; glass walls cap it at IoU 38 | not reported in RA-L; AAMAS 2021 layout node "few seconds → < 10 s" | "can be used online" (claimed) | `goldleaf3i/declutter-reconstruct` Python **MIT**; `aislabunimi/ROSE2` ROS1 GPL-3; ROSE `tkucner/rose` GPL-3 |
| **DuDe** Fermin-Leon, Neira, Castellanos (ICRA 2017; TIGRE ECMR 2017) | contour decomposition | grid → polygons → dual-space decomposition, concavity threshold 3 m; **incremental: decomposes only the difference image** | over-segments corridors; MCC 0.70 (MAORIS) | 95 % of maps < 1 s | **Yes** | `lfermin77/Incremental_DuDe_ROS` C++ ROS Indigo, no licence |
| **GRID-FAST** Fredriksson, Saradagi, Nikolakopoulos (JIRS 2024, arXiv 2406.11635) | scan-line gaps | multi-direction scan-line intersection detection emitting **explicit openings**; filters `min_gap_size` (robot-passable), `unknown_cells_filter`, `object_filter_max_steps`; node classes intersection / pathway / dead-end / pathway-to-unexplored | handles unknown natively; up to 92 % fewer graph nodes than next-best; no room-IoU benchmark | online on `nav_msgs/OccupancyGrid` | **Yes, ROS 2** | `LTU-RAI/GRID-FAST` C++ GPL-3, "under development" |
| **BWEM / BWTA2 / Perkins** (game-AI terrain analysis: Perkins AIIDE 2010; Uriarte & Ontañón AIIDE 2016; BWEM 2015) | S2 / S1 | priority-flood watershed on DT; **merge iff saddle/peak ≥ 0.90** or basin tiny; choke centre = max-clearance ridge cell, ends = min-clearance; Perkins: merge if choke radius > 90 % of the smaller / 85 % of the larger region radius, 97 % agreement with human chokes | mature engineering, not robotics-evaluated | ms | batch per map | BWEM-community (public), BWTA2 LGPL-3 |
| Joo 2010 "virtual door detection" (CEC); Choi 2011 (Auton. Robots); Kleiner 2017 (IROS); Kim 2022 (RA-L); Qin 2013 incremental GVD; Warberg 2024 line-based rooms + GP-EDF; Fleer 2017 | various | — | records only **[U]** | | | |

### 2.2 Learned methods

| Method (venue) | Input | Doors? | Reported quality | Cost / deployability | Code |
|---|---|---|---|---|---|
| **Hiller, Qiu, Particke, Hofmann, Thielecke** "Learning topometric semantic maps from occupancy grids" (IROS 2019, arXiv 2001.03676) | occupancy grid rescaled to 0.05 m, 64×64 sliding window stride 8 | **Yes** — 4-conv binary CNN flags door patches, U-Net segments the door, min-bounding-rectangle instances (centre/width/orientation), **width prior 0.7–1.6 m**; validated by "close all doors, reopen one, segment count must drop" | sim-trained; real-patch recall 0.63 → > 0.9 with structured noise augmentation; door recall-IoU 0.87; **not tested on partial maps** | tiny nets but O(area/64) windows; batch | repo has the simulator only — **no weights, no inference code** |
| **Nikdel, Chen, Vaughan** (CRV 2020, arXiv 1903.03669) | two 244×244 ego-centric 16 m local maps | opening classes (open-room / closed-room / corridor), frame-to-frame tracked | Husky trials; only grid-door detector evaluated *during* exploration | **140 Hz GTX 1080 Ti, 12 Hz CPU** | none |
| **Goeddel & Olson** (IROS 2016) | local patch CNN {corridor, doorway, room} | doorway as a class | **15–23 % on doorways — negative result** | — | none |
| **TopoNets / GraphSPN / DGSM** Zheng & Pronobis (IROS 2019; AAAI 2018; IROS 2017) | local laser patches, SPN + topological graph | doorway as a place class | office-scale real data | real-time exact inference claimed | released |
| **QueSTMaps** Mehan et al. (IROS 2024 oral, arXiv 2404.06442) | 3-channel BEV (density + ceiling slice + floor slice) of a complete scan | **Yes** — Mask R-CNN instances for rooms *and* "transition regions" | AP@0.5 Structured3D 88.0 doors / 98.9 rooms; **MP3D 60.9 / 88.5 vs RoomFormer 24.9 / 64.4** | ResNet-50 Mask R-CNN on full-scene BEV; batch; GPU | not found |
| **RoomFormer** Yue et al. (CVPR 2023) | point-density image | door/window polylines (F1 ≈ 81 on S3D) | **door AP collapses 76.9 → 24.9 S3D→MP3D** (QueSTMaps) | 10 ms/scene TITAN RTX | MIT |
| HEAT (CVPR 2022), PolyDiffuse (NeurIPS 2023), PolyRoom (ECCV 2024), SLIBO-Net (NeurIPS 2023) **[U input]** | density images of complete scans | no doors (HEAT, PolyDiffuse, PolyRoom) | — | GPU | mixed |
| Raster-to-Vector (ICCV 2017), DeepFloorplan (ICCV 2019), CubiCasa5K (SCIA 2019) | floor-plan *drawings* | yes | — | — | MIT / GPL-3 / custom |
| **Image-space door detectors**: DoorDetect (Arduengo 2021, mAP 45); DeepDoors2 (Ramôa 2021); **Antonazzi, Luperto, Basilico, Borghese** ECMR 2023 + JFR 2025 (DETR / YOLOv5, env-specific fine-tune AP 34→72; **YOLOv5 20 fps on Jetson TX2**) | RGB(-D) frames | door *status*; in Antonazzi, door *locations are assumed known on the prior map* | — | Jetson-proven | `aislabunimi/door-detection-long-term` (no licence) |
| **Learn and Link** Molina, Kumar, Srivastava (ICRA 2020) | top-down obstacle image | "critical regions" for sampling, not doors | — | batch | — |
| **Ericson & Jensfelt** "Beyond the frontier" (RA-L 2024, arXiv 2406.09160) | partial occupancy grid | transformer predicts wall segments beyond the frontier (KTH floor plans); **doors appear as gaps** | — | GPU | lericson.se/floorist |

### 2.3 Room graphs inside ObjectNav and scene-graph systems

| System | Partitions rooms? | Algorithm | Doors | Recompute / cost |
|---|---|---|---|---|
| **Hydra** Hughes, Chang, Carlone (RSS 2022; IJRR 2024) | yes, on the sparse-GVD "places" layer | prune places by clearance at **δ ∈ [0.45, 1.2] m**; RSS: median component count over the sweep; IJRR: Betti-0 persistence with plateau ratio α = 0.5; flood-fill the rest by decreasing clearance | implicit — the min-clearance place on a room–room edge | every update, **4–19 ms**; needs the 3D ESDF (voxblox-bound) |
| **HOV-SG** Werby et al. (RSS 2024) | yes, BEV | wall histogram → threshold → dilate → EDF → seeds → `cv2.watershed` → merge small | none | offline after mapping; region P/R 84.1/83.6 vs Hydra 86.2/77.6 (8 HM3DSem scenes, manual GT) |
| **MoMa-LLM** (RA-L 2024) / **MORE** Mohammadi et al. (2025, arXiv 2505.03035) | yes, BEV Voronoi | oriented Gaussians sized by the *perceived* door bbox → line integral along graph edges → cut → components; **sub-rooms merged once the region is fully explored** | perceived door objects | per high-level step, cheap — this is our `room_separation.py` |
| **Clio** (RA-L 2024) | yes | information-bottleneck clustering of Hydra places by CLIP | none | GPU |
| **OccuSG** (2026 preprint) | yes, 2D occupancy | incremental DuDe decomposition, region tracking by IoU | implicit | 0.5 Hz maintenance |
| **ReasonNavi** (2026 preprint) | yes | EDT + Otsu seeds + watershed + small-room merge, OpenCV only | implicit | per step |
| **VoroNav** Wu et al. (ICML 2024) | **no** | skeletonize the free map each step → Reduced Voronoi Graph; rooms are LLM *text* | none | per step; no code |
| **SG-Nav** (NeurIPS 2024) / **UniGoal** (CVPR 2025) | perceptual room nodes | open-vocab 3D instance segmenter prompted with room names | none | GPU heavy; room-node ablation costs 0.7 % SR |
| **OSG Navigator** (2024/25) | perceptual | VQA place + detector doors + LLM association; **connector nodes P 0.85 / R 0.80** (graph metric) | yes, perceived | LLM loop |
| SayNav (ICAPS 2024) | from GT segmentation | — | door = room–room edge | sim only |
| ESC, L3MVN, StructNav, OpenFMNav, TriHelper, GAMap, InstructNav, Mem2Ego, ApexNav, TSGM, TopoNav, SemExp, PONI, VLFM | **none** | frontier / value maps | none | — |

**Observation:** every system that runs online chose a classical S2/S3 partition as recently as 2026;
no ObjectNav paper ships a learned grid-door detector. Where doors exist as nodes they come from
perception (MORE, OSG), exactly our YOLO path.

## 3. Doorway and opening extraction — focused comparison

What each signal hands a planner, and the parameters the literature has converged on:

| | S1 GVD critical point | S2 DT saddle | S3 erosion persistence | S4 wall-line gap |
|---|---|---|---|---|
| Door position | point on the skeleton | ridge cell of max clearance | break locus (needs a second pass) | midpoint of the gap |
| Door normal | critical line direction (basis-point pair) | ridge tangent ⟂ | not given | wall normal — **best defined** |
| Door width | 2 × clearance | 2 × saddle clearance | ≈ 2δ at break | gap length — **exact, independent of clearance** |
| Published parameters | width band **0.7–1.6 m** (Hiller), **0.8–1.2 m single / 1.6–2.4 m double** (Gómez 2019); suppress neighbours within 0.5× clearance keeping the larger basis angle (Bormann); degree-2 ∧ junction neighbour ∧ ¬(known→unknown) (Wurm); minimal-loop size (Friedman) | merge if **saddle/peak ≥ 0.90** (BWEM) or **choke > 90 % / 85 % of region radii** (Perkins) or abs(Δ) ≤ 0.3·max (MAORIS); ripple absorbed at > 40 % shared contour; min basin 64 cells / 1.5–2 m² | δ ∈ [0.45, 1.2] m, keep the most persistent component count, α = 0.5 (Hydra); W = W_door + 0.1 m (Area Graph) | drop lines < 10 % covered (ROSE²); collinear-merge gap < one doorway; vertical-Hough confirm (Gómez); traversal crossing a wall plane (Tourani) |
| Clutter | poor raw; good with Friedman's loop test or a declutter pre-pass | medium (furniture makes extra peaks) | poor (furniture = extra constrictions) | **good** — walls, not free space |
| Partial maps / unknown | Wurm's known→unknown rule and Beeson's exit pruning handle it explicitly; otherwise jittery | boundary to unknown becomes a false ridge unless unknown is excluded from the EDT source | Hydra/Area Graph treat unknown as boundary — works, but splits at frontiers | ROSE² flat IoU 0.2→1.0 coverage; Luperto predicts missing walls |
| Wide portals (our 24.75 m case) | no (clearance never dips) | no | no (door ≥ corridor width breaks Hydra/AG) | **yes** — a gap is a gap |
| Uniformly narrow corridor | over-segments at every clearance ripple (Bormann, MAORIS Fig. 2d) | same; our `threshold_rise` is the standard fix (rise on both sides) | Hydra collapses corridors into one region | fine |
| Compute | ms (EDT + skeleton we already compute) | ms | ms per δ × ~10 δ | seconds (Hough + DBSCAN); DFT declutter extra |
| Online in the literature | Beeson, Wurm, Qin 2013 | HOV-SG offline, ReasonNavi per step | Hydra incremental | ROSE² claimed, not timed |

## 4. Comparative analysis for downstream ObjectNav

Criteria: (R) robustness on noisy, partial, monocular-BEV grids; (C) computational overhead on a
GPU-contended Jetson; (N) fit for ObjectNav — needs explicit door nodes (centre, normal, width), stable
room identities, and tolerance to a room first seen only through its doorway.

| Candidate | R | C | N | Verdict |
|---|---|---|---|---|
| Flown watershed + absolute dynamics + exogenous doors | medium — depends entirely on door supply; knife-edge without it | very low | medium — doors are not derived, IDs churn at the frontier | baseline |
| **S1 critical points with the full filter set** (Wurm + Friedman loop + Bormann suppression + width band) on our medial axis | medium-high once filtered; needs unknown excluded from clearance | very low (reuses the skeleton we compute) | **high** — the only classical signal producing point + normal + width from the grid alone | strongest short-term candidate |
| **S2 with relative ratios** (BWEM 0.90, Perkins 90/85 %) instead of absolute 0.5 m dynamics | likely more scale-invariant across room sizes; unmeasured on our maps | zero extra | medium | cheap A/B on `live_bev_hospital.npz` and the 15 recorded floors |
| **S3 Hydra persistence** on the skeleton | high on room *count* stability; weak on wide doors, open plan | low (≈ 10 EDT thresholds + CC) | medium-high — directly attacks the 44-toggle churn | complementary to S1 |
| **S4 ROSE² structure-first** as a low-rate pre-filter | **highest** on clutter and partial maps (only method with coverage curves); fails on glass/perception artefacts | seconds — run at 0.2–0.5 Hz, not per action | high — gives wall lines, hence wide-portal gaps and well-defined normals | best medium-term candidate; MIT Python |
| Area Graph α-shape | good on incomplete maps; one global W cannot cover 0.8 m doors and 24 m portals | low-medium (CGAL) | medium | parameter choice is the blocker |
| MAORIS | excellent on clean complete maps only | 43 s batch | low | not for exploration-time BEV |
| GRID-FAST | explicit openings, unknown-aware, ROS 2 | low | medium — graph nodes, not rooms | worth reading the opening filter |
| Hiller / Nikdel learned patch detector | unknown on our BEV; needs our own labelled data and sim→real augmentation | small nets, TensorRT-trivial — but no weights exist | medium | not now |
| QueSTMaps / RoomFormer | door AP 25–61 on real scans, complete scenes only | ResNet-50 Mask R-CNN on the contended GPU | low | no |
| Perceptual doors (Antonazzi, OSG, MORE) | status and confirmation, not discovery; Jetson-proven at 20 fps | GPU (shared with YOLO we already run) | medium — we have this path | keep as evidence, not as the sole source |

**Does the literature show deep learning beating classical geometry on door detection?** No. No paper
benchmarks a learned grid-door detector against GVD critical points or watershed saddles with
door-level metrics on the same maps; Goeddel & Olson's patch CNN scored 15–23 % on doorways; the one
cross-domain measurement (RoomFormer, S3D → MP3D) cuts door AP by two thirds; and every online system
through 2026 partitions classically. Learned components earn their keep on door *status* from images
and on predicting walls beyond the frontier (Ericson & Jensfelt), not on discovering doors in the grid.

## 5. The literature mapped onto our recorded failures

| LESSONS entry | What failed | What the field does |
|---|---|---|
| 2026-10-05 churn: choke clearance near the frontier is distance-to-UNKNOWN → 47 % non-severing disks, one door toggling 44 times | unknown cells act as obstacles in the EDT | Wurm 2008 rule 3 drops critical points on branches leading known→unknown; Beeson 2005 REVG keeps only paths between *exits*; Area Graph and GRID-FAST treat unknown as a distinct class; Tourani 2026 **defers** a passage hypothesis until evidence accumulates; Hydra's persistence picks the component count that survives a δ sweep rather than one threshold |
| 2026-10-05 Hanson: YOLO door 0.6 m off its doorway severed a bedroom | a disk at a wrong point | MORE sizes an **oriented Gaussian** from the door bbox and orientation rather than a disk; Hiller emits an oriented min-bounding rectangle; snapping to the nearest S1 critical point (what we did) is Thrun's definition; Pronobis & Jensfelt tolerate FP/FN doors through non-monotonic inference |
| Allensville hallway read as a doorway (0.30–0.50 m clearance for forty actions) | a corridor is a uniform clearance minimum | Friedman's minimal-loop feature; Beeson's EVG clearance cap M; both-side rise (our `threshold_rise`) is MAORIS's ripple test; Bormann documents Voronoi corridor over-segmentation as the family's known defect |
| `room_merge`: dynamics threshold is a plateau *with* the door list, a knife-edge *without* it | absolute-metre merge criterion | BWEM and Perkins merge on **ratios** (saddle/peak ≥ 0.90; choke > 90 %/85 % of region radii), MAORIS on relative abs(Δ) ≤ 0.3·max — scale-free across room sizes |
| 2026-09-01 hospital: 24.75 m portals, outdoor space wraps the building | clearance-based signals cannot sever a wide opening; no interior boundary | only S4 (wall-line gap) represents a wide portal; Area Graph's outer α-shape defines the building boundary; Bormann requires unknown pre-mapped for the same reason |
| Opening nodes: a room glimpsed through its door hangs off the hallway | partition over observed free space only | MORE merges sub-rooms **only once the region is fully explored**; Luperto 2019–2021 predicts the partially observed room's layout; Ericson & Jensfelt predict walls (and door gaps) beyond the frontier |
| Room IDs renumbered R23 → R26 → R28 → R52 | per-tick recomputation | **no paper measures room/door identity stability across incremental updates**; OccuSG 2026 tracks regions by IoU like `RoomRegistry`; DuDe and Qin 2013 update only the changed region |

## 6. Benchmarks and datasets available for an evaluation

| Dataset | Room GT | Door GT | Fit for our grids |
|---|---|---|---|
| Bormann 20 maps ± furniture ± trashbins (`ipa_room_segmentation/common/files/test_maps/`) | yes | no (doorway label only in 3 VRF training maps) | laser-quality, complete; good for regression against published numbers |
| ROSE² `declutter-reconstruct/MAPFILES` incl. exploration snapshots (FR79, Cartesium, GMapping) | yes | no | **the only public partial-map series with GT** |
| MAORIS robot maps + 25 sketches (Zenodo 10.5281/zenodo.892062) | yes | no | clean maps |
| QueSTMaps MP3D doorway annotations | yes | **yes (manual)** | if obtainable; BEV from complete scans |
| HouseExpo (35k SUNCG layouts, MIT; IROS 2020) | category + bbox | no — doors cut into openings, not stored | training-scale only; SUNCG provenance caveat |
| CubiCasa5K, Structured3D, KTH floor plans | yes | yes | drawings / synthetic; domain gap |
| HM3D-Sem, Gibson | none / tiny | no | HOV-SG hand-labelled 8 scenes |
| **Ours**: `topology/tests/fixtures/live_bev_hospital.npz`, 15 recorded ObjectNav floors, `floor_maps.npz` + `trajectory.csv` replays, `hospital_doors.yaml` (35 doors, 11 with width) | GT 20 rooms + 7 corridors (hospital) | partial | the only data in the correct domain |

There is **no public robot-map dataset with door ground truth** apart from the QueSTMaps MP3D
extension; door-level evaluation will have to be built on our own replays.

## 7. Gaps the literature leaves open

1. Temporal stability of room and door identities under incremental mapping is unmeasured anywhere;
   `RoomRegistry` with memory ticks has no published counterpart to compare against.
2. No method is evaluated on monocular-depth BEV: walls are thick and bleed, floor coverage is
   patchy, and frontier edges look like openings. Every benchmark map is laser-built.
3. No learned-vs-classical door-detection comparison exists on shared maps.
4. ROSE², the strongest on partial maps, reports no runtime and emits no door nodes.
5. Wide portals (archways, open plan) are a declared failure mode of S1–S3 and are simply not
   addressed in S2/S3 papers.

## 8. Candidate shortlist for discussion (nothing implemented)

Ranked by expected gain per unit of overhead on the flown pipeline. These are reading notes, not a plan.

1. **Derive doors from the grid with S1 + the full published filter set** on the medial axis we
   already compute: width band, Bormann neighbour suppression keeping the larger basis angle,
   Wurm's degree/junction/known→unknown rules, Friedman's minimal-loop test. Output centre, normal,
   width — which `door_carve_mask` and `room_separation.py` can both consume — and let exogenous
   doors become *evidence* on those candidates instead of cuts of their own.
2. **Exclude unknown from the clearance source** (or classify skeleton branches into unknown as
   exits, Beeson/Wurm) before any choke measurement; this is the literature's answer to the 47 %
   non-severing disks.
3. **A/B relative merge ratios** (BWEM 0.90, Perkins 90/85 %) against absolute 0.5 m dynamics on the
   existing fixture and replays; zero new code paths, one parameter semantics change if it wins.
4. **Hydra-style δ-persistence** for the room count instead of a single `peak_local_max` setting;
   milliseconds, targets the churn directly.
5. **ROSE² as a 0.2–0.5 Hz structure pre-filter** (MIT, Python, scipy allowed on this host-owned
   path) feeding wall lines to the above; the one approach validated on partial maps, and the only
   one that represents a 24 m portal.
6. Defer any learned grid-door detector until 1–5 are measured; if revisited, the Hiller/Nikdel
   ego-patch pattern trained on *our* BEV artefacts is the only deployable shape, and it has no
   public weights.

## 9. Corrections to our own documentation found on the way

- `core/mapping/topology/README.md`, `voronoi.py` and `room_separation.py` attribute the Gaussian
  door-field pipeline to "Werby et al. (MORE, 2025)". arXiv 2505.03035 (MORE) is **Mohammadi,
  Honerkamp, Büchner, Cassinelli, Welschehold, Despinoy, Gilitschenski, Valada (2025)**; Werby et al.
  is HOV-SG (RSS 2024), which uses a watershed, not a door field. The lineage is MoMa-LLM (RA-L 2024)
  → MORE (2025). Not changed here — documentation fix for a later commit.
- `room_watershed.py`'s rationale ("a room is a closed area bounded by doors") matches Pronobis &
  Jensfelt 2012 and Hiller 2019 exactly; worth citing when that docstring is next touched.

## References (primary sources verified unless marked)

- Bormann, Jordan, Li, Hampp, Hägele. Room segmentation: Survey, implementation, and analysis. ICRA 2016. DOI 10.1109/ICRA.2016.7487234. Code: github.com/ipa320/ipa_coverage_planning.
- Thrun, Bücken. Integrating grid-based and topological maps for mobile robot navigation. AAAI 1996. Thrun. Learning metric-topological maps for indoor mobile robot navigation. AIJ 99, 1998. DOI 10.1016/S0004-3702(97)00078-7.
- Beeson, Jong, Kuipers. Towards autonomous topological place detection using the extended Voronoi graph. ICRA 2005. DOI 10.1109/ROBOT.2005.1570793.
- Wurm, Stachniss, Burgard. Coordinated multi-robot exploration using a segmentation of the environment. IROS 2008.
- Friedman, Pasula, Fox. Voronoi random fields. IJCAI 2007.
- Fabrizi, Saffiotti. Extracting topology-based maps from gridmaps. ICRA 2000. DOI 10.1109/ROBOT.2000.846479 **[U body]**. Buschka, Saffiotti. A virtual sensor for room detection. IROS 2002. DOI 10.1109/IRDS.2002.1041463 **[U body]**. Diosi, Taylor, Kleeman. ICRA 2005. DOI 10.1109/ROBOT.2005.1570263 **[U body]**.
- Mielle, Magnusson, Lilienthal. MAORIS. ICRA 2018. arXiv 1709.09899. Code: github.com/MalcolmMielle/MAORIS (GPL-3).
- Hou, Yuan, Schwertfeger. Area Graph. ICAR 2019. arXiv 1910.01019. Code: github.com/STAR-Center/areaGraph (GPL-3). osmAG: arXiv 2309.04791.
- Kucner et al. ROSE. ICRA 2021. arXiv 2004.08794. Luperto, Kucner, Tassi, Magnusson, Amigoni. ROSE². RA-L 2022. arXiv 2203.03519. Code: github.com/goldleaf3i/declutter-reconstruct (MIT); github.com/aislabunimi/ROSE2 (GPL-3).
- Luperto, Fochetta, Amigoni. Exploration of indoor environments through predicting the layout of partially observed rooms. AAMAS 2021. arXiv 2004.06967.
- Fermin-Leon, Neira, Castellanos. Incremental contour-based topological segmentation (DuDe). ICRA 2017. DOI 10.1109/ICRA.2017.7989297.
- Fredriksson, Saradagi, Nikolakopoulos. GRID-FAST. JIRS 2024. arXiv 2406.11635. Code: github.com/LTU-RAI/GRID-FAST (GPL-3).
- Perkins. Terrain analysis in real-time strategy games. AIIDE 2010. DOI 10.1609/aiide.v6i1.12405. Uriarte, Ontañón. BWTA2. AIIDE 2016. BWEM: github.com/N00byEdge/BWEM-community.
- Hiller, Qiu, Particke, Hofmann, Thielecke. Learning topometric semantic maps from occupancy grids. IROS 2019. arXiv 2001.03676. Repo: github.com/LIKERobo/SemanticMapGeneration (simulator only).
- Nikdel, Chen, Vaughan. CRV 2020. arXiv 1903.03669. Goeddel, Olson. IROS 2016. DOI 10.1109/IROS.2016.7759589.
- Mehan et al. QueSTMaps. IROS 2024. arXiv 2404.06442. Yue et al. RoomFormer. CVPR 2023. arXiv 2211.15658. Chen, Qian, Furukawa. HEAT. CVPR 2022. arXiv 2111.15143.
- Gómez, Hernandez, Barber. Sensors 2019. DOI 10.3390/s19204595. Anguelov, Koller, Parker, Thrun. ICRA 2004. Pronobis, Jensfelt. ICRA 2012.
- Antonazzi, Luperto, Basilico, Borghese. ECMR 2023, arXiv 2203.03959; JFR 2025, arXiv 2401.17996, DOI 10.1002/rob.70084.
- Tourani et al. Passage-aware structural mapping for RGB-D VSLAM. 2026. arXiv 2604.24707. Bavle et al. S-Graphs+. RA-L 2023. arXiv 2212.11770.
- Hughes, Chang, Carlone. Hydra. RSS 2022, arXiv 2201.13360; IJRR 2024. Code: github.com/MIT-SPARK/Hydra (BSD-2). Maggio et al. Clio. RA-L 2024. arXiv 2404.13696.
- Werby et al. HOV-SG. RSS 2024. arXiv 2403.17846 (MIT). Mohammadi et al. MORE. 2025. arXiv 2505.03035. MoMa-LLM. RA-L 2024.
- Wu et al. VoroNav. ICML 2024. arXiv 2401.02695. Yin et al. SG-Nav. NeurIPS 2024. arXiv 2410.08189. Loo, Wu, Hsu. Open Scene Graphs. 2025. arXiv 2508.04678. OccuSG 2026 arXiv 2606.13727; ReasonNavi 2026 arXiv 2602.15864.
- Ericson, Jensfelt. Beyond the frontier. RA-L 2024. arXiv 2406.09160. Li et al. HouseExpo. IROS 2020. arXiv 1903.09845 (MIT). Aydemir, Jensfelt, Folkesson. KTH floor plans. IROS 2012.
