# VLM scoring for zero-shot object search — literature review and state of the art

Reviewed 2026-11-19 on branch `feat/objnav-benchmark-infra-nadav`. **Literature only: no code was changed
and nothing below is implemented.** The design consequences are in [INTEGRATION.md](INTEGRATION.md).

**How this was done.** 123 papers (2022 – November 2026) were read from their own text: 19 leading or
structurally closest systems directly, 102 through eight reader ledgers that follow one template (what
is scored, by what, how it reaches the planner, use of RGB beyond depth, perception, results, cost,
ablations), and two trained systems (Hydra-Nav, NaviScale) for their protocol and results only. The
comparison tables of the paper draft (`docs/paper/zson_paper.tex`, as of 2026-11-19) served as a
cross-check: every row they share with section 6 agrees. The ledgers, with every table reference, are
in the session workspace
(`~/.copilot/session-state/400afaf2-dfd3-4baa-9854-8f6eb6d37e80/files/ledger_*.md`); PDFs and extracted
text are under `~/papers/vlm-scoring-survey/`. Neither is in the repo.

**Conventions.** SR / SPL in percent. **[U]** = not verifiable from the paper's text. "Second-hand" = a
number one paper prints for another. HM3D **v0.1** (= v1) is the 2,000-episode, 20-scene val split we
run; **v0.2** (= v2) is the 1,000-episode, 36-scene split with corrected annotations and no cross-floor
goals; many papers say only "HM3D" and are assigned by the counts they state.

## 0. The baseline this is compared with

Watershed rooms on a depth-built occupancy grid → all-objects detector → text LLM labels each room →
text LLM gives each room a probability → RPT\* orders the rooms by expected time → A\*. Full val splits,
500 low-level actions, no navmesh in the agent. With the simulator's annotations as the detector
(aligned with the goal list), numbers of the paper draft: HM3D v0.1 **71.3 / 38.2** with the 14B text
model and 71.4 / 37.4 with the 3B one (83.7 SR when the goal is on the start storey, at 45.2 and 44.2
SPL); MP3D **65.6 / 29.1** (an earlier revision). With YOLO-World (an older revision): HM3D v0.1
**41.3 / 22.3**. 5.3 probability queries and 27.8 label queries per HM3D episode.

## 1. Reading published numbers: four protocol classes

Headline numbers in this field are not on one protocol, and the differences are larger than most of the
gaps between methods.

| class | what it means | examples |
|---|---|---|
| **S** standard | full val split, 500 low-level actions (0.25 m, 30°), learned perception, nothing fitted on benchmark data | VLFM, ApexNav, CogNav, TravExplorer, STRIVE, SysNav, PIGEON, RATE-Nav, MarvisNav, DRIVE-Nav, AECNav, ConsistNav, RefBeliefNav |
| **W** WMNav-style | a budget of 40–100 *decisions*, each a metre-scale polar move, usually with a free six-view panorama | WMNav, SkillNav, MerNav, ReMemNav, HGWM, HTCE, CSG-Nav, DSCD-Nav [budget U] |
| **P** perception not learned or not stated | simulator semantics, or no detector named | our runs; SOR-Nav [U]; Mem2Ego, LFG, LocalNav (ground truth stated) |
| **T** uses benchmark data | a policy, selector, verifier, retrieval corpus or prior built from HM3D / MP3D training scenes or episodes | Uni-NaVid, Hydra-Nav, IntentNav, FiLM-Nav, FOM-Nav, LERR-Nav, NaviScale, TrajRAG, CLOSER, StopWatch; ASCENT's priors; EvolveNav's rule memory, which is built from the evaluated episodes and their outcomes |

Further deviations to check before quoting a row: a learned PointNav controller trained on HM3D
(VLFM and most of its descendants; a re-run of VLFM with a classical planner gives 50.9 / 23.6 instead
of 52.5 / 30.4); success radius (UniGoal 1.6 m, VLMnav 1.2 m, R2F 1.5 m; 0.2 m from a goal view point
instead of 0.1 m in ApexNav, ConsistNav and TravExplorer); simulator path planner for
motion (3D-Mem, R2F, NavProbe, FSUNav); subsets (3DGSNav's 80.0 / 51.8 "HM3D v1" has ablation rows that
cannot come from 2,000 episodes; OSG 400 episodes; ImagineNav 200; R2F 60); wall-clock budgets (RTNav);
and quoted rows that drift (ApexNav is quoted at 59.6 / 33.0 on v0.1 and re-run at 58.6 / 32.4 and
52.8 / 27.1; MVP-Nav reports 65.4 and is re-evaluated at 49.6).

## 2. Who uses a VLM to decide where to go, and what it scores

Seven families, by the unit that receives the score.

| family | what is scored, from what | typical model | representative systems | best standard-protocol result |
|---|---|---|---|---|
| **1. Whole frame → map** | one image–text similarity per frame, painted on the depth-visible view cone; read back at frontiers | BLIP-2 ITM, SigLIP2 / RADIO, CLIP | VLFM, ApexNav, AECNav, ASCENT, ConsistNav, RefBeliefNav, TravExplorer, SER-Nav, CLUE; patch level: GAMap, WTRP-Searcher; voxel level: BeliefMapNav | AECNav v0.2 84.7 / 45.3, MP3D 51.3 / 25.9; TravExplorer v0.1 70.0 / 37.2; RefBeliefNav v0.1 69.2 / 35.3 |
| **2. Heading of a panorama** | a 0–10 (or 0–4) score per direction from a generative VLM, ray-cast into a "curiosity" map | Gemini, Qwen-VL | WMNav, SkillNav, MerNav, ReMemNav, HGWM, CSG-Nav; one float per frame: SemNav; four action scores: HistVLM, TRN; belief field: Structure over Reasoning | class W only: SkillNav v0.1 69.7 / 39.3; SemNav (class S) 54.9 / 35.9 |
| **3. Marks on the egocentric frame** | lettered or numbered candidates (frontiers, graph nodes, points of interest, action arrows) drawn on the RGB image; the VLM returns a probability per mark, a pick, or letter logits | Gemini, GPT-4o, Qwen3-VL-8B, LLaVA | OpenFrontier, MarvisNav, ViTL, PIGEON, Explore-EQA, VLMnav, CPNav, Mem2Ego | PIGEON v0.1 67.1 / 35.8, MP3D 52.6 / 18.5; MarvisNav v0.2 81.2 / 42.5 |
| **4. One image per candidate** | a stored frame looking at each frontier or direction ("frontier snapshot", representative view); the VLM picks or rates | GPT-4o, Qwen2.5/3-VL | 3D-Mem, Prune-Then-Plan, SCOPE, HGR, DRIVE-Nav, RTNav, EvoMemNav; trained: FiLM-Nav, FOM-Nav, IntentNav | DRIVE-Nav v0.1 55.5 / 35.9, MP3D 41.8 / 22.6 |
| **5. Structured scene with rooms or regions** | rooms / regions / graph nodes described as text, with or without images; the model picks or rates a room, frontier or landmark | Gemini, GPT-4(V), Qwen | SysNav, STRIVE, RATE-Nav, CogNav, SG-Nav, UniGoal, GraphEQA, HFLEX-EQA, MSGNav, SOR-Nav, fast–slow ReAct, OSG | CogNav v0.1 72.5 / 26.2; STRIVE MP3D 52.3 / 23.1 |
| **6. Rendered top-down map** | the model looks at a map image with marks | Qwen2.5-VL-7B | TopV-Nav; SSMG-Nav (junction sub-graphs); IntentNav (trained) | TopV-Nav v0.1 52.0 / 28.6 |
| **7. No generative model online** | embeddings and detector evidence only | RADIO + SigLIP2 + SAM3; BLIP-2 + SAM3 | AECNav, ConsistNav, TravExplorer, R2F, ERA | AECNav, ConsistNav, TravExplorer (above) |

Notes that matter for the idea under study:

- **Closest prior art to "one VLM, frames in, scores out, a planner over the scores".** *SemNav*: GPT-4o
  returns one float per raw frame ("likelihood of finding X if I move in this direction"), fused into a
  VLFM-style map, read as a per-frontier success probability and fed to an expected-cost frontier
  planner, with a target-only detector and no rooms: 54.9 / 35.9, and 54.6 / 34.3 when the same scores
  are used greedily. *OpenFrontier*: a probability per marked visual frontier times a predicted
  information gain, divided by distance: v0.2 77.3 / 35.6, MP3D 40.7 / 17.8 (its frontier detector is
  trained on HM3D scans and its controller is VLFM's). *SysNav / STRIVE*: the VLM chooses the next room
  from a structured description (SysNav adds a representative image per room), classical coverage
  inside the room: v0.1 63.7 / 30.5 and 62.9 / 34.2, MP3D 50.7 / 18.1 and 52.3 / 23.1. *RATE-Nav*: a
  watershed region partition, and a VLM that judges from a region's key frames whether the target is
  there (high / uncertain / very low): v0.1 67.8 / 31.3, MP3D 50.3 / 20.6.
- **Nobody combines the two halves.** No surveyed system asks a generative VLM for a probability per
  node *and* optimises a visiting order over those probabilities; the readers of all eight ledgers
  report this independently. SemNav comes nearest: its score is asked per frame, not per node, is read
  back from a map as a frontier's success probability, and enters a learning-over-subgoals planner that
  picks the frontier of least expected cost.
- **Rooms without room segmentation.** Outside SysNav / STRIVE (wall detection and dilation), RATE-Nav
  (watershed) and ERA (clearance-seeded regions), "rooms" are model outputs: a scene classifier on the
  frontier's frame (ASCENT, fast–slow ReAct, LERR-Nav), a VLM's room type per landmark (CogNav), room
  names prompted to a segmenter (SG-Nav), BLIP-2 view labels associated by an LLM (OSG), or a room word
  in the scoring prompt (ApexNav, CLUE, HGWM).

## 3. How the score reaches the map and the planner

**Where it is stored.** (a) Every cell of the depth-visible view cone gets the frame's scalar (VLFM:
cos² angular confidence, no range term, cone capped at the 5 m depth limit in the released code;
frontier value = median within 0.5 m of the frontier). (b) Cells along each heading's rays, merged by
minimum so that a region once judged empty stays low (WMNav family). (c) On the candidate itself —
graph node, frontier, room — refreshed at each query (families 3–5). (d) Nowhere: mapless systems
re-decide from the current view (VLMnav, ImagineNav, DSCD-Nav).

**Fusion over time.** Keeping a memory of the score is worth far more than its exact rule: averaging
instead of overwriting +7.3 SR on HM3D, the confidence weighting on top +0.7 (VLFM); a choice with its
exploration state drawn on it 81.2 against 71.9 without memory and 73.6 with a text history
(MarvisNav); about 20 de-duplicated keyframes +11 SR, all frames +0 (ImagineNav++). Rules in use:
confidence-weighted mean (VLFM, ApexNav), minimum (WMNav), maximum (GAMap), per-cell Kalman filter with
range-growing variance (WTRP-Searcher), bounded log-odds with competing labels and misses (AECNav,
ApexNav), tempered log-domain belief with a leak and a revisit discount (Structure over Reasoning).

**Choosing the next goal.** Argmax of the score (VLFM, R2F, GAMap); a utility of score, gain and
distance (OpenFrontier, AECNav, RefBeliefNav: potential + 0.25 × information gain − 5 × path length;
without the gain term −11.3 SR, without revisitable waypoints −4.9); the model's own pick (families
3–5); or a **sequence**:

| system | objective and solver | sequence against greedy, same scores |
|---|---|---|
| WTRP-Searcher | weighted latency over viewpoints and mid-confidence detections; LKH heuristic, 10 ms | +1.3 SR, +5.9 SPL (multi-target HM3D, 236 episodes) |
| BeliefMapNav | expected distance over frontiers weighted by visible voxel belief; simulated annealing | +6.5 SR, +2.3 SPL (400 HM3D episodes) |
| SSMG-Nav | expected path length over softmax beliefs; 2-opt | +0.03 SPL (GOAT subset) |
| SemNav | expected cost with success / failure costs (learning-over-subgoals form) | +0.3 SR, +1.6 SPL |
| ApexNav | tour over high-score frontiers when cues are strong, nearest frontier otherwise | adaptive 76.2 / 38.0 against 72.9 / 36.1 pure semantic argmax (v0.2) |
| TravExplorer | open tour (asymmetric TSP, LKH) over confirmed targets, else the floor's frontiers, else stair landmarks; BLIP-2 values down-weight frontiers | not ablated; 70.0 / 37.2 on v0.1 with no language model |
| SOR-Nav [class P] | tour over frontiers plus an LLM "leave this region?" gate | nearest-frontier −5.8 SR, −10.7 SPL (v0.2) |
| AirHunt (aerial) | tour with value-gap precedence | 16.2 → 73.1 SR against greedy-by-value |
| SEEK (rooms from a floor plan) | room MDP with a per-room search cost and independent probabilities | SPL 0.96 against 0.74 greedy; below coverage when the prior is wrong |

All of these use heuristic solvers, the weighted-latency form without survival factors, and (SEEK
apart) no per-node search cost; RPT\* covers what they do and more.

**Readout and calibration.** Written numbers (WMNav 0–10, OpenFrontier and VoroNav probabilities, CPNav
confidences) or token probabilities of a forced short answer (Explore-EQA: letters on marked points
plus a yes/no for the whole view, so that views are comparable; ViTL: letters with an "explore
elsewhere" option; Prune-Then-Plan; CovNav; RTNav's verifier). Nobody measures whether a frontier score
behaves as a probability; the nearest things are SEEK's Brier score, SCOPE's reported calibration error
and Explore-EQA's remark that its relevance score stays below 0.5 in most *successful* episodes.
Calibration is used to *prune*, not to produce probabilities: Prune-Then-Plan compares each frontier's
score with the scores human-labelled bad frontiers receive and sends the agent to the nearest survivor
(39.5 against 37.8 for the VLM's pick and 36.9 for nearest-frontier); CPNav's conformal floor removes
low-score actions, calibrated on 500 simulator episodes with oracle paths [split U]; CovNav uses the
prediction-set size to decide when a slow reasoning pass is worth it, calibrated on MP3D-train images.

**When the model is asked, and what it costs.**

| system | calls per episode | note |
|---|---|---|
| ours today | 5.3 + 27.8 text calls | probabilities + room labels |
| MarvisNav | 4.1 | one per decision site (graph junction) |
| fast–slow ReAct | 5.0 deliberative, plus a per-detection gate | text first, at most two images on demand |
| PIGEON | 6.2 (real robot, 7B) | on reaching a point of interest |
| RefBeliefNav | 8.2 GPT-4o + BLIP-2 every frame | only when the two best candidates are close and poorly supported (37.2 without the trigger, with lower SR) |
| OpenFrontier | 22.8 (second-hand) | every six steps |
| WMNav | 54.5 (second-hand) | three roles per decision |
| InstructNav | 113.6 (second-hand) | |
| OSG | about 150 | |
| Structure over Reasoning | 290–340 | 3-token answers, 62 ms each |

Asking more is not better: deliberating at every frontier choice scores below not deliberating (64.8
against 65.35) while deliberating only when every frontier is far gives 68.75 (fast–slow ReAct); and
under a wall-clock budget heavy pipelines collapse (BeliefMapNav 65.5 → 27.3 on v0.1, while VLFM loses
only 1.3; RTNav).

## 4. RGB beyond the depth range

| mechanism | who | what is stored, where | evidence | usable without training? |
|---|---|---|---|---|
| **Ray frontiers** | RayFronts; R2F | features of pixels with no valid depth (R2F: ≥ 3.5 m), attached to the frontier the viewing ray passes, in 30° direction bins; frontier score = best bin; re-cast when the frontier moves | 2.2× the ruled-out unmapped volume of one-value-per-frontier baselines at 10–20 m range, about 4× that of one ray per image with no depth (outdoor, no planner); R2F 78.3 / 29.6 on 60 episodes with navmesh planning, no ablation | yes |
| **Far-view description per candidate** | VoroNav | a caption of the frame facing each graph neighbour, given to the LLM with the map-derived object list | +2.5 SR, against +1.3 for the object lists | yes |
| **Marks on the frame / frontier images** | OpenFrontier, MarvisNav, ViTL, PIGEON; 3D-Mem, SCOPE, HGR, RTNav | implicit: the VLM sees what lies behind the mark at any range; the answer is attached to the frontier's map position | see sections 2 and 7; in HGR the VLM's guess of the room type behind a frontier is right 68 % of the time, 26.5 % of predictions are refuted on arrival (two thirds of them mirrors and glass), and deleting refuted predictions beats decaying them (72.4 against 69.4 SR, GOAT subset) | yes (OpenFrontier's detector is not) |
| **Room word in the scoring prompt** | ApexNav, CLUE | "…a ⟨target⟩ or a ⟨room⟩ ahead", so a target room scores before the target is visible | CLUE: decisive for toilet (72.6 against 63.8), the weakest cue for chair, plant, tv | yes |
| **Image-space frontiers** | LRN, FrontierNet, WildOS | a trained decoder marks where one can continue beyond the map; handed to the planner as a goal at the map's edge | FrontierNet: 58.6 % against 44.1 % of volume at half budget | no: all trained (FrontierNet on HM3D) |
| **Bearing-only far targets** | EZREAL, WildOS | a direction, or a triangulated position with 42–49 % range error, used as an attractor and replaced once in range | memory on a graph matters more than the visual score (LRN without memory: intervention in 3 of 3 dead-end runs) | yes |
| **Predicted depth** | ERA, RANGER, MVP-Nav | the depth sensor replaced by a monocular or multi-view model | ERA: v0.1 55.2 / 27.4, MP3D 46.4 / 21.2 with no depth sensor | yes, with a revisable map |

What the evidence fixes:

- **The standard value map throws the far field away.** VLFM's scalar is computed from the whole image
  but painted only on cells depth has seen, so what is visible down a corridor is credited to the floor
  in front of the robot. Structure over Reasoning and RefBeliefNav gate evidence by depth visibility on
  purpose.
- **No one stores a generative VLM's score beyond the depth range**, and no beyond-range mechanism has a
  standard-protocol HM3D or MP3D result.
- **Identification degrades fast with range.** Three systems independently put about 3 m as the limit
  for confirming a target from RGB (CSG-Nav: a small plant 5/5 at 1 m, 1/5 at 5 m; a chair behind a
  doorway 0/5 at 5 m; RTNav and CLOSER use 3 m). Views taken from where the robot stands also separate
  directions worse than views about 2 m further on (ImagineNav: 43.0 against 55.0).
- **Predicted geometry must not be mapped as measured geometry.** With monocular depth of about 10 %
  relative error, VLFM falls from 68.2 to 10.8 SR because smeared depth edges close passages and delete
  frontiers; a revisable occupancy rule restores 50.7 and goals in known-free space 61.7. Only 9 % of
  the wrong obstacle cells come from beyond sensor range. Frontier-only goal sets run dry even with a
  depth sensor (8 % of v0.2 episodes for VLFM).

## 5. Models and their sensitivity

Cloud models in the leading systems: Gemini 2.0 / 2.5 / 3.5 Flash (STRIVE, SysNav, OpenFrontier,
MarvisNav, SkillNav, ReMemNav), GPT-4o / GPT-4V (PIGEON, CogNav, 3D-Mem, SemNav, RefBeliefNav),
Qwen-VL-Max / Plus (RATE-Nav, MerNav). Local models reported under the standard protocol: Qwen3-VL-8B
(DRIVE-Nav, EvoMemNav), Qwen2.5-7B text (ASCENT), Qwen3-8B text (CovNav), Qwen3-VL-2B (REST),
Qwen2.5-VL-7B (TopV-Nav, SR-Nav), LLaVA-NeXT-34B (ViTL).

| interface | model swap | effect |
|---|---|---|
| VLM picks the action (WMNav, HM3D v0.1) | Gemini 1.5 Pro → Qwen2.5-VL-7B → 3B | 58.1 → 46.1 → 29.7 SR |
| VLM debates the action (DSCD-Nav, v0.2) | Gemini 2.5 Flash → Qwen3-VL-8B → GPT-4.1-nano | 80.1 → 67.8 → 58.7 |
| VLM picks a frontier image (RTNav, v0.2, real time) | Qwen3.5-9B → Gemma4-12B → Qwen3.5-4B | 72.3 → 66.2 → 37.8 |
| LLM state machine (CogNav, v0.1) | GPT → Llama3.1-8B + CogVLM2 | 72.5 → 62.6 |
| scene graph + marked image (LocalNav, OVON, ground-truth semantics) | Gemini 3.1 Pro → a 4B model | 41.6 → 26.0 |
| VLM picks a lettered route choice (MarvisNav, v0.2) | Gemini 3.5 Flash → Gemma4-26B → Qwen3-VL-8B | 81.2 → 78.9 → 76.0 |
| VLM gives a probability per marked frontier (OpenFrontier, v0.2) | Gemini 2.5 Flash → Gemma-3-4b → LLaVA-1.6 | 77.3 → 76.9 → 74.2 |
| VLM decides in a mapped system (RoboAtlas, GOAT subset) | GPT-4o → Qwen2.5-VL-7B | 90.6 → 88.8 |
| text LLM picks a frontier after predicting failure for each (EvolveNav, v0.1) | Qwen2.5-7B → Qwen3-8B → Qwen3.5-9B | 67.0 → 67.3 → 67.2 |
| text LLM gives a probability per room, RPT\* orders (ours, v0.1, oracle perception) | Qwen2.5-14B → 3B | 71.3 → 71.4 SR, 38.2 → 37.4 SPL; 151 of 2,000 outcomes change |

The narrower the question and the more of the memory and planning sits outside the model, the less the
model's size matters. A 4B model can say whether a view looks like the goal, and nothing else: route
choice at chance, distance estimates constant, stop votes 31–47 % precise, grounding to the right
category but the wrong instance (Structure over Reasoning). An untrained 2B model selecting among
frontier images is below random (IntentNav), and a small LLM given a long map description fails until
the prompt is pruned (EfficientNav). Language models know which room an object belongs to (rank
agreement with real rooms 0.67–0.75) but not where in the room (14–32 % of the search-cost reduction a
real-layout prior gives).

## 6. State of the art on HM3D and MP3D

### 6.1 HM3D v0.1 (2,000 episodes) — our benchmark

| method (arXiv) | class | SR | SPL | decides where to go with | notes |
|---|---|---|---|---|---|
| CogNav (2412.10439) | S | **72.5** | 26.2 | text LLM state machine over a landmark graph; GPT-4V adds room types and relations | MP3D run on 1.8K episodes |
| TravExplorer (2605.19958) | S | 70.0 | **37.2** | BLIP-2 value map on traversable voxels + open tour over frontiers; no language model | climbs stairs; success distance 0.2 m; same-floor episodes 74.6 / 41.9, others 48.7 / 16.8 |
| RefBeliefNav (2610.15375) | S | 69.2 | 35.3 | BLIP-2 belief + uncertainty; frontiers and revisitable waypoints; GPT-4o on demand | 17 constants chosen on 1,000 HM3D episodes |
| fast–slow ReAct (2608.09816) | S | 68.75 | 34.0 | BLIP-2 value map; VLM deliberation only at impasses | PointNav trained on HM3D; hand-written category–room table; VLM not named [U] |
| RATE-Nav (2506.02354) | S | 67.8 | 31.3 | region watershed + VLM verdict per region | Qwen-VL-Max |
| PIGEON (2511.13207) | S | 67.1 | 35.8 | GPT-4o picks among numbered points of interest on raw frames | climbs stairs with a trained stair policy; PointNav low level |
| ZONDA (2607.21025) | S | 66.5 | 33.0 | embedding-similarity map over frontier blocks | version not named; multi-floor |
| ASCENT (2505.23019) | S\* | 65.4 | 33.5 | BLIP-2 value map; Qwen2.5-7B over scene-classifier and tag text for far frontiers | priors from benchmark training statistics; multi-floor |
| SysNav (2603.06914) | S | 63.7 | 30.5 | Gemini picks the next room; classical in-room coverage | |
| ConsistNav (2605.09869) | S | 63.2 | 34.8 | inherited BLIP-2 frontier map; candidate state machine | no LLM |
| STRIVE (2505.06729) | S | 62.9 | 34.2 | Gemini picks the next room from structured text | 72.7 / 38.2 on same-floor episodes |
| BeliefMapNav (2506.06487) | S | 61.4 | 30.6 | voxel belief + sequence over frontiers | re-run at 65.5 / 32.2 by RTNav |
| MarvisNav (2610.06510) | S | 60.5 | 36.2 | Gemini picks a lettered route choice with its state | |
| CovNav (2610.27677) | S | 60.4 | 33.1 | object–target yes/no from a local text LLM | |
| ApexNav (2504.14478) | S | 59.6 | 33.0 | BLIP-2 map, adaptive semantic / geometric | |
| WTRP-Searcher (2503.06937) | S? | 59.2 | 39.8 | CLIP map + weighted-latency tour | continuous control, budget not stated |
| DRIVE-Nav (2603.28691) | S | 55.5 | 35.9 | Qwen3-VL-8B picks among direction views | |
| ERA (2611.17730) | S | 55.2 | 27.4 | viewpoint utility; no depth sensor | |
| SemNav (2506.03516) | S | 54.9 | 35.9 | GPT-4o float per frame + expected-cost planner | |
| UniGoal / SG-Nav / L3MVN | S | 54.5 / 54.0 / 50.4 | 25.1 / 24.9 / 23.1 | LLM over a scene graph (the first two) or over frontier object lists | older reference rows; UniGoal's success radius is 1.6 m |
| VLFM (2312.03275) | S | 52.5 | 30.4 | BLIP-2 map, argmax frontier | PointNav trained on HM3D |
| DSCD-Nav (2601.21409) | W [U] | 75.6 | 40.1 | cloud VLM debate over polar actions | budget in a missing supplement |
| SkillNav (2607.15758) | W | 69.7 | 39.3 | six-heading curiosity map + written "skills" | 40 steps; problematic episodes filtered out |
| MerNav (2602.05467) | W | 68.0 | 36.9 | as WMNav, with memory and review | 100 steps |
| HGWM / ReMemNav / WMNav | W | 62.6 / 60.0 / 58.1 | 31.5 / 36.8 / 31.2 | six-heading scores | 40 steps |
| SOR-Nav (2609.34707) | P [U] | 66.1 | 36.9 | frontier tour + LLM "leave this region?" gate | no detector named; "RGB, depth and semantic observations" from the simulator |
| ASCENT with an ideal detector | P | 70.3 | 41.0 | as ASCENT | +4.9 SR over its learned detector; same-floor episodes 74.9 / 45.1 |
| L3MVN / HRO with ground-truth segmentation | P | 66.4 / 64.9 | 36.4 / 35.8 | LLM frontier scores | +12 SR each over learned segmentation |
| VLFM with an ideal detector (run by ASCENT's authors) | P | 62.4 | 40.0 | as VLFM | +9.9 SR |
| **ours**, oracle semantics | P | 71.3 | 38.2 | LLM probability per room + RPT\* | 14B text model; 71.4 / 37.4 with the 3B; start-storey goals 83.7 / 45.2; no storey change |
| **ours**, YOLO-World | S | 41.3 | 22.3 | same | older revision; 24 % wrong STOPs |
| EvolveNav (2606.18235) | T | 67.3 | 33.9 | Qwen3-8B (text) picks a frontier after predicting a failure risk for each | rules distilled from earlier evaluated episodes; 64.7 / 32.5 without them and without the risk step |
| LERR-Nav / IntentNav / NaviScale / TrajRAG / FiLM-Nav / FOM-Nav | T | 72.3 / 70.5 / 64.3 / 62.5 / 61.7 / 57.3 | 35.6 / 34.6 / 34.8 / 33.9 / 37.3 / 40.8 | trained selectors, map predictors or corpora | |

Best standard-protocol SR: CogNav (72.5, with the lowest SPL of the group), then TravExplorer (70.0).
Best standard-protocol SPL: TravExplorer (37.2), which uses no language model, then MarvisNav,
DRIVE-Nav, SemNav, PIGEON and RefBeliefNav at 35–36 (WTRP-Searcher's 39.8 is not on the discrete
protocol). Nothing training-free and standard is above 73 / 38. Among the results with simulator
semantics ours has the highest SR (71.3–71.4 against 70.3 for ASCENT with an ideal detector, a margin
inside our confidence interval) and not the highest SPL (38.2 against 41.0 and 40.0); our
learned-detector run is far below the standard-protocol group.

### 6.2 MP3D (2,195 episodes) — our benchmark

| method | class | SR | SPL | notes |
|---|---|---|---|---|
| PIGEON | S | **52.6** | 18.5 | GPT-4o; stairs |
| STRIVE | S | 52.3 | 23.1 | |
| AECNav (2608.10817) | S | 51.3 | **25.9** | no generative model online; SigLIP2 + SAM3; PointNav from VLFM |
| SysNav | S | 50.7 | 18.1 | |
| ConsistNav | S | 50.6 | 25.7 | no LLM |
| RATE-Nav | S | 50.3 | 20.6 | |
| RefBeliefNav | S | 49.1 | 18.5 | |
| TravExplorer | S | 48.8 | 21.1 | no language model; climbs stairs; success distance 0.2 m |
| ZONDA | S | 48.2 | 21.5 | multi-floor |
| fast–slow ReAct | S | 47.3 | 18.0 | |
| CogNav | S | 46.6 | 16.1 | 1.8K episodes, 20 categories |
| ERA | S | 46.4 | 21.2 | no depth sensor |
| ASCENT | S\* | 44.5 | 15.5 | |
| MarvisNav / DRIVE-Nav / OpenFrontier | S | 42.1 / 41.8 / 40.7 | 20.4 / 22.6 / 17.8 | |
| ApexNav / VLFM | S | 39.2 / 36.4 | 17.8 / 17.5 | |
| UniGoal / SG-Nav / BeliefMapNav | S | 41.0 / 40.2 / 37.3 | 16.4 / 16.0 / 17.6 | older reference rows |
| MerNav / ReMemNav / SkillNav / DSCD-Nav / HGWM / WMNav | W | 50.8 / 49.8 / 48.3 / 47.8 / 46.8 / 45.4 | 19.5 / 24.3 / 25.5 / 24.2 / 17.9 / 17.2 | |
| FSUNav (2604.03139) | S? | 48.8 | 26.9 | simulator's planner for motion |
| SOR-Nav | P [U] | 61.8 | 38.5 | see 6.1 |
| ASCENT / VLFM with an ideal detector | P | 58.6 / 55.3 | 30.3 / 29.6 | +14.1 and +18.9 SR over their learned detectors |
| **ours**, oracle semantics | P | 65.6 | 29.1 | an earlier revision; start-storey goals 77.4 / 34.3; no storey change |
| Hydra-Nav / IntentNav / StopWatch / EvolveNav / LERR-Nav / NaviScale | T | 64.0 / 53.8 / 49.9 / 49.0 / 47.9 / 43.1 | 29.6 / 23.1 / – / 19.1 / 18.7 / 16.8 | Hydra-Nav is a fine-tuned navigation VLM; EvolveNav: see 6.1 |

Best standard-protocol SR is a three-way tie within the noise of 2,195 episodes (PIGEON, STRIVE, AECNav
at 51–53); best SPL is AECNav and ConsistNav at about 26 (FSUNav's 26.9 is obtained with the
simulator's planner) — the two leaders in efficiency use no generative model at all, as TravExplorer
does not on HM3D v0.1.

### 6.3 HM3D v0.2 (1,000 episodes), for reference

AECNav 84.7 / 45.3; ConsistNav 84.2 / 41.2; MarvisNav 81.2 / 42.5; SysNav 80.8 / 37.2; STRIVE and PIGEON
79.6 / 38.7 and 38.1; CovNav 77.9 / 40.2; OpenFrontier 77.3 / 35.6; ApexNav 76.2 / 38.0; MSGNav
74.1 / 33.4; DRIVE-Nav 72.4 / 41.3; ERA 70.1 / 30.6. Class W: SkillNav 75.9 / 43.2, MerNav 74.8 / 37.6.
Class P [U]: SOR-Nav 87.6 / 52.7. Class T: StopWatch 83.0 SR, IntentNav 82.2 / 38.5, TrajRAG 78.1 / 40.2,
FiLM-Nav 77.0 / 41.3, Uni-NaVid 73.7 / 37.1 (second-hand). v0.2 has no cross-floor goals and cleaner
annotations, which is why standard-protocol methods gain 11–21 SR over v0.1. Four papers print an
"HM3D" result without saying which episodes: OpenFrontier (placed above because MarvisNav lists it
under v0.2), FSUNav 76.2 / 40.5, Schrödinger's Navigator 68.2 / 31.1 and the trained Hydra-Nav
84.8 / 41.1.

### 6.4 How the leaders decide where to go, what they use, what we could take

| method | where to go | VLM / LLM | detector and verification | what we could take |
|---|---|---|---|---|
| **CogNav** | an LLM moves between five search states and picks a landmark (a Voronoi node near frontiers or objects) from text; room type per landmark from GPT-4V on marked images every 10 steps | GPT-4V + a text LLM; about 15 state queries and 23 room-type queries per episode | OpenSEED; two verification states | the states *candidate verification* and *target confirmation* are worth +15 SR, *contextual search* +2.4: spend the VLM on confirming, not only on guiding |
| **TravExplorer** | confirmed target instances first; otherwise an open tour (asymmetric TSP) over the current floor's frontiers, with a BLIP-2 value map (VLFM's rule on traversable voxels) lowering frontiers of little value; stair landmarks only when the floor's frontiers are exhausted | none generative; BLIP-2 image–text matching | SAM 3 with a probabilistic instance map: a cluster in view but not detected receives zero-confidence evidence | the best SPL on v0.1 comes from a tour over frontiers with no language model; leave the floor only when it is exhausted; false stops are 22 % of all its episodes and the largest failure group, searches that run out of budget 6 % |
| **RefBeliefNav** | argmax of target potential + information gain − path cost over frontiers **and past waypoints** | BLIP-2 every frame; GPT-4o reliability score when the top two are close | Grounding-DINO + MobileSAM | revisitable waypoints as nodes; ask the large model only when the decision is close and poorly supported |
| **fast–slow ReAct** | BLIP-2 value map every step; a VLM is woken at four impasses (stall, all frontiers far, frontiers exhausted, floor change), reads a room-type grid and stored keyframes, text first | one VLM [U], 5 calls | detector ensemble; a per-detection VLM gate vetoes 16.7 % of candidates | a keyframe store indexed by map cell; deliberate only where a mistake is expensive; every failure falls back to geometry; a montage of old frames to find a missed opening |
| **RATE-Nav** | frontier exploration, with a VLM that decides from a region's key frames when the region is not worth more | Qwen-VL-Max, Llama-Vision 11B | YOLO-World / GLIP; VLM re-perception (+7.5 SR) | *rule a region out from its images* — the cheaper visit; region partition by watershed (+9.9 SR in their ablation) |
| **PIGEON** | at sparse decision points the VLM picks among numbered frontier, suspect-object and stair marks on raw frames, with coverage-selected memory frames; may answer "not enough information" | GPT-4o (zero-shot row) | YOLOv7 / OWLv2 + MobileSAM; suspect objects are checked by the VLM | the suspect-object check is worth 10 SR, the frontier choice 0.6–1.3; maximum-coverage selection of memory frames; abstention |
| **SysNav / STRIVE** | rooms from wall detection; the VLM picks the next room, may leave a room early; coverage planning inside | Gemini 2.5 / 2.0 Flash | YOLO-World + SAM2.1 / MM-Grounding-DINO + SAM; context check and one re-check from a better viewpoint (+7.5 SR) | the room as the model's decision unit: +4.6 SR, +3.8 SPL over viewpoint-level decisions at a third of the tokens; a representative image per room |
| **AECNav** | frontier utility = semantic value + information gain along the path − cost | none online (SigLIP2 on a shared encoder) | SAM3; log-odds per object cluster with look-alike classes and misses as negative evidence (+12.8 SR) | competing labels and "expected but not seen" for the target; gain counted along the path, not only at the destination |
| **ConsistNav** | inherited frontier map | none | seven-state commitment machine with per-candidate negative evidence and a multi-cue stop gate | +8.0 SR (v0.2) and +11.4 (MP3D) from execution discipline alone |
| **MarvisNav** | at graph junctions the VLM picks a lettered route choice drawn on the frame, coloured by exploration state | Gemini 3.5 Flash; 76.0 / 38.3 with Qwen3-VL-8B | RF-DETR + SAM3; the VLM's "target may be visible" lowers the detector threshold | state on the mark; decide only at junctions; depth-consistent projection of candidates and arrows where a route leaves the image. False positives are 58–65 % of its failures |
| **DRIVE-Nav** | frontiers grouped into persistent directions by path bearing; the VLM picks among one representative view per direction, inspected only within the forward 240° | Qwen3-VL-8B (local), a few calls per episode | SAM3; cross-frame VLM verdict (sure / uncertain / no) over at most three frames | stable direction identities; a fully local stack under the standard protocol |
| **OpenFrontier** | probability per marked visual frontier × predicted gain ÷ distance | Gemini 2.5 Flash; within 3 SR with LLaVA-1.6 | SAM3; one VLM check at a viewpoint in front of the candidate | the per-mark probability; recent frames in the query move a frontier from 0.65 to 0.10 once its room has been seen |
| **SkillNav / MerNav / WMNav** (W) | six-heading scores into a curiosity map; rule-based corrections written onto the map | Gemini 3 Flash, Qwen3-VL-Plus | a goal VLM that must confirm twice | corrections belong on the map, not in the prompt |
| **SOR-Nav** (P) | a frontier tour, and an LLM asked only whether the current region contradicts the target | text LLM | object clusters need three observations and a fresh look before STOP | an asymmetric question ("is there a reason to leave?") instead of a ranking |

## 7. Where the points come from

| component | measured effect | source |
|---|---|---|
| semantic choice of where to explore | +0.6 to +4 SR, more on SPL; larger in multi-goal and question-answering settings (3D-Mem: +5.4 SR on GOAT-Bench, +11 SPL on A-EQA) | ApexNav, PIGEON, CogNav, VoroNav, L3MVN, SemNav, fast–slow ReAct, DRIVE-Nav; 3D-Mem; our flat-`p` ablation (+3.3 SR, n.s.) |
| images in the prompt instead of graph text only | 63.5 against 13.6 % success in question answering | GraphEQA |
| a room-level cue | +0.2 to +2.8 SR on average; decisive for toilet, harmful as the only cue for chair, plant, tv | CLUE, HRO, RPV-SemNav, BeliefMapNav, SG-Nav |
| ordering instead of greedy | +2 to +6 SPL, sometimes SR | section 3 |
| memory of the score / state on the choice | +7 to +11 SR | VLFM, MarvisNav, ImagineNav++ |
| marks instead of whole frames | +17 SR (numbered arrows); frontier region 70.3 against 63.8 whole frame ≈ no image | VLMnav; FOM-Nav |
| negative evidence from what was seen | +5.3 SR | BeliefMapNav |
| candidates beyond frontiers | +4.9 SR; frontier-only sets run dry in 8–23 % of episodes | RefBeliefNav; ERA |
| target verification / evidence fusion | +4 to +25 SR | SG-Nav, STRIVE, RATE-Nav, PIGEON, AECNav, CogNav, ApexNav, ZONDA |
| execution and commitment | +8 to +11 SR | ConsistNav |
| perfect semantics | +5 to +26 SR (ASCENT +4.9 on v0.1 and +14.1 on MP3D, VLFM +9.9 and +18.9, L3MVN and HRO +12, VoroNav +26); the language model then adds +0.3 SR | ASCENT, L3MVN, HRO, VoroNav |
| handling other floors | +6.5 to +8.7 SR on v0.1; VLFM scores 64.6 on same-floor and 0.4 on cross-floor episodes, TravExplorer 74.6 and 48.7 | PIGEON, ZONDA; LERR-Nav's split; TravExplorer |
| a larger language model for the same decision | none: 67.0 / 67.3 / 67.2 SR with 7B / 8B / 9B; 71.4 / 71.3 with 3B / 14B | EvolveNav; ours |

On these benchmarks the search prior is the small term and target confirmation the large one; half of
the remaining failures of the best systems are stops on the wrong object, part of them real but
unannotated instances (TriHelper relabels 38 % of its detection failures as simulator misjudgements,
SemNav about 25 % of its failures).

## 8. What to take, in order

1. **Ask for a probability per discrete, marked candidate and keep RPT\*.** The published per-candidate
   outputs (OpenFrontier, VoroNav, TopV-Nav, ViTL's letter logits) are already the list RPT\* consumes,
   and every same-score comparison favours ordering over greedy.
2. **Put the far view on the candidate, not on the floor in front of the robot.** Score a frontier or a
   glimpsed room from the frame that looks through it and store the score on that node (VoroNav,
   OpenFrontier, MarvisNav, ray frontiers).
3. **Rule places out from their images.** A region-level verdict from key frames (RATE-Nav), history
   frames beside the candidate (OpenFrontier), negative evidence by visibility (BeliefMapNav).
4. **Keep the question narrow and the structure outside the model**, call it at loop points only, and
   let it abstain (Structure over Reasoning, fast–slow ReAct, PIGEON, REST).
5. **Use the same VLM to confirm the target** before lock and before STOP, with look-alike classes as
   competing evidence (STRIVE, PIGEON, DRIVE-Nav, AECNav).
6. **Never restrict the node set to frontiers**; add regions still to inspect and revisitable waypoints
   (RefBeliefNav, ERA).
7. **Evaluate the scorer offline first** — rank, Brier score, ruled-out area — before closing the loop
   (SEEK, RPV-SemNav, RayFronts).

## 9. Not verified

- SOR-Nav's perception in simulation (no detector is named; the simulator is said to provide semantic
  observations) and therefore the class of its three rows.
- DSCD-Nav's step budget and episode counts (supplement not in the PDF); fast–slow ReAct's VLM; the HM3D
  version behind several rows that state only counts (RefBeliefNav, ZONDA, SemNav, WTRP-Searcher) or
  nothing (OpenFrontier, FSUNav, Schrödinger's Navigator, Hydra-Nav).
- How TravExplorer's value map enters its tour: the printed cost has only path length and a constant
  per frontier type.
- Whether quoted rows were re-run (ViTL's VLFM row); the per-episode call counts of WMNav, OpenFrontier
  and InstructNav are other papers' measurements.
- Training-set overlap of the monocular depth and feature models with HM3D / MP3D / Gibson scenes is
  not discussed by the papers that use them.
- Figures were read only where a reader rendered the page; numbers that exist only as plots are not
  reported.
