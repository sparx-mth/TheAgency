# One VLM in place of two LLMs and the all-objects detector — analysis, three pipelines, a test plan

Written 2026-11-19 on branch `feat/objnav-benchmark-infra-nadav`. **Analysis only: no code was changed and
nothing below is implemented.** The literature behind every claim is in [SUMMARY.md](SUMMARY.md); numbers
marked *ours* were computed for this note from the recorded runs under `runs/` and are post-hoc readings,
not new experiments.

Section 9 reads the draft pseudocode `docs/zson_vlm_pseudocode.tex`, which appeared in the working tree
during the review, against this analysis.

The part under discussion is the middle of the pipeline: today *room segmentation → all-objects detector →
LLM #1 (room label) → LLM #2 (per-room probability)*; proposed *RGB frames → one VLM → scores*. Input on
one side is RGB + depth + pose, output on the other is what RPT\* consumes: a set of nodes, each with an
entry point, a probability `p` of finding the target there (independent, not normalised, below 1), a
search cost `τ`, and travel costs between entry points. Everything outside that span (occupancy mapping,
RPT\*, A\*, the closing) is assumed unchanged.

## 1. What the numbers say before any design

### 1.1 Where the search spends its budget (*ours*)

The split first. In the HM3D v1 val run of the current code with oracle perception and the 14B model
(`runs/hm3d_v1_gt_3b_vs_14b/stage2_14b`, 71.3 SR / 38.2 SPL), 348 of the 2,000 episodes have their goal
on another storey; the agent stays on its start storey and wins 12 % of them. They are 305 of the 575
failures, and no scorer of start-storey nodes touches them (stair handling is worth 6.5 to 8.7 SR to
PIGEON and ZONDA on this split). What follows is about the other 1,652 episodes (83.7 / 45.2), grouped
by how many probability queries the episode issued:

| probability queries in the episode | episodes | SR | SPL | mean actions |
|---|---|---|---|---|
| 0 (no decision was asked for) | 262 | 99.2 | 77.6 | 45 |
| 1–2 | 507 | 98.4 | 63.4 | 98 |
| 3–5 | 423 | 95.3 | 38.1 | 211 |
| 6–10 | 308 | 61.0 | 17.3 | 389 |
| 11 or more | 152 | 21.1 | 4.3 | 486 |

Three readings. (a) Close to half the episodes (769) are decided within two decisions and are nearly
always won (98.7 %): no scorer can improve them. (b) A decision costs about 40 actions (a room visit),
so 500 actions buy ten to twelve visits; the 460 episodes that need more than five decisions hold 240 of
the 270 same-storey failures. (c) In the 167 same-storey failures where the target was never seen (the
`limit_never_seen` tag of `hm3d/campaign_breakdown.py`, the count the paper draft reports), the agent
ends a median 7.5 m from the goal after starting a median 8.2 m from it, and 41 % end farther away than
they began, after a median of eleven probability queries: the budget went to the wrong part of the
house, not to a near miss. So the place where a better scorer can matter is **long searches**, and it
can matter in two ways: a better *order*, and a *cheaper visit* (deciding about a room without spending
40 actions in it).

What a decision buys is mostly a walk. In the paper draft's budget table for these runs the transit
to the chosen room takes 46–48 % of all actions and the frontier walk inside it 10 %; four visits in
ten walk to no frontier at all because the way in had resolved them, and 13 % of the transits end
early because the room was seen on the way and nothing was left to look at. The agent mostly inspects a
room while approaching it. A judgement about a room made from where the agent stands therefore saves
the expensive half of a visit, and that is the judgement the far field can supply (section 2.4).

The picture does not depend on the run. The 3B run of the same code (`stage1_3b`, 71.4 / 37.4) has 497
long episodes holding 240 of its 270 same-storey failures (161 of the 270 never-seen); the full run of
the earlier revision (`runs/gt_oracle_2000ep_14b_20261111T114322Z`, 71.0 / 39.8) has 441 holding 235 of
272.

A decision offers few nodes: mean 2.9, median 3, a single node in 23 % of the 10,773 recorded decisions,
at most 5 in 90 % and at most 6 in 95 %. A VLM prompt with two images per node therefore carries 6
images typically, 10 at the 90th percentile and 12 at the 95th. An episode issues 5.3 probability
queries on average (median 4) and 27.8 room-label queries.

### 1.2 What the semantic prior is worth

Our own ablation (`docs/progress/entries/046`, 180 single-storey MP3D episodes, oracle perception): the
3B oracle against a flat `p = 0.15` for every node is **+3.3 SR points [−0.0, +6.7] and +1.5 SPL points**,
not significant; 131 of 180 episodes are identical. The full HM3D split says the same from another side
(paper draft, Results): replacing the 3B model by the 14B one in both queries changes the outcome of 151
episodes — 77 lost, 74 gained — and leaves SR where it was (71.4 against 71.3), with +0.8 SPL
[0.0, 1.5]. A model nearly five times larger moves individual episodes and not the total, and the last
revision of the probability prompt moved SR by 1.1 points on each of two 90-episode sets, not
significant either. Every published ablation that isolates "semantic choice of where to explore" from
the rest lands in the same range:

| system | what was removed | effect |
|---|---|---|
| ApexNav (HM3D v0.2) | all semantics: nearest frontier only | −2.4 SR, −1.6 SPL |
| PIGEON (HM3D v0.1 / v0.2) | the VLM's frontier selection | −0.6 / −1.3 SR |
| CogNav (HM3D v0.1) | the "contextual search" state | −2.4 SR |
| VoroNav (HM3D, ground-truth semantics) | the LLM | −0.3 SR, −3.1 SPL |
| L3MVN (HM3D) | the LLM score | −2.4 SR |
| Fast–slow ReAct (HM3D v0.1) | deliberation over far frontiers | −3.4 SR [1.7, 5.1] |
| DRIVE-Nav (HM3D-OVON, 49 categories) | the VLM's choice among direction views | −4.2 SR, −5.1 SPL |

The prior is worth a few points, more on SPL than on SR, more in large or open-vocabulary settings, and a
wrong confident prior is worse than none (SEEK: below plain room coverage whenever the true room was
given under 0.5). **Replacing the two LLMs by a better scorer should be expected to move SR by single
digits at most**; what it can change structurally is cost, robustness and the far field.

### 1.3 What dominates once the detector is real (*ours* and published)

The same method with YOLO-World in place of the simulator's annotations
(`runs/hm3d_v1_val_j8p2_20261026`, HM3D v1 val, an older revision, so not a controlled pair) scores
**41.3 SR / 22.3 SPL against 71.3 / 38.2**. Wrong STOPs are 24.2 % of all episodes (484; median 6.3 m from
the goal) against 0.5 % with the oracle (10 episodes); by category 61 % of plant episodes, 49 % of
tv_monitor, 44 % of sofa. The literature agrees to the point of monotony: false-positive stops are
58–65 % of MarvisNav's failures on all three benchmarks; removing target verification costs 7–13 SR in
STRIVE, PIGEON, RATE-Nav and AECNav, 15–21 in ApexNav, 25 in ZONDA; an execution/verification layer
alone adds 8–11 SR in ConsistNav. Part of this is a benchmark ceiling that no verifier removes (real
but unannotated instances on HM3D v0.1), which the oracle hides because it reports only the listed goal
instances.

### 1.4 Consequences for the design

1. Measure the headroom before building: how much would a *perfect* `p` give on the long-search episodes?
   (Section 6, step 0.) If the answer is small, the search prior is not where the points are.
2. Judge the VLM pipeline on SPL, actions-to-first-sighting and calls, on the long-search subset, not on
   overall SR, which is saturated by the easy half.
3. Use the VLM where the evidence is strongest: to decide about a place *without going there* (far-field
   preview, rule-out from the doorway) and, under a learned detector, to verify the target. The second is
   outside the span being replaced, but the same model provides it and it decides whether any search
   improvement will be visible at all with real perception.
4. Keep the ordering in RPT\*. No published system asks a VLM for a probability per node and
   optimises a visiting order exactly (SemNav, the nearest, reads per-frame scores from a map into an
   expected-cost frontier planner); the ones that order heuristically gain over greedy on the same scores
   (WTRP-Searcher +5.9 SPL, BeliefMapNav +6.5 SR on 400 episodes, SemNav +1.6 SPL over a per-frame
   GPT-4o score, AirHunt 16 → 73 SR in flight), and the best standard-protocol SPL on HM3D v0.1 belongs
   to a tour over frontiers with no language model at all (TravExplorer, 70.0 / 37.2). This is the part
   of the current method worth keeping whatever feeds it.

## 2. Thinking

### 2.1 Per frame, per frontier, per cluster or per room?

RPT\* treats a node as an **independent search event**: a place the agent can go to (an entry point),
that costs something to search, and whose "the target is found here" outcome does not overlap another
node's. That criterion, not the VLM's convenience, selects the unit.

| unit | entry point | search cost | independent of the other nodes? | verdict |
|---|---|---|---|---|
| frame | none: a frame is an observation, not a place | none | no | evidence, never a node; must be attached to places (pipeline C) |
| raw frontier | yes | weak | no: several frontiers open onto the same unseen room and each would be credited with it | too many, unstable, double counting |
| frontier group ("portal": frontiers that lead to the same unseen region) | yes | estimable | approximately, if grouped by what lies behind and not by image or metric proximity | the natural node for space that is **not mapped yet** |
| room / region of mapped floor | yes | yes (frontiers left, floor not inspected) | yes, by construction of a partition | the natural node for space that **is mapped** |
| map cell | no | no | — | storage for evidence (pipeline C), aggregated into nodes afterwards |

So the answer to "per what" is: **score the units the planner can treat as separate search events — a
region of mapped floor, and a distinct region behind the map's edge — and treat frames as the evidence
those scores are read from.** The evidence in the literature points the same way: room-level VLM
decisions beat viewpoint-level ones at a third of the tokens (STRIVE: 79.6 / 38.7 against 75.0 / 34.9);
asking the model at every frontier is worse than not asking (fast–slow ReAct: 64.8 against 65.35 for the
plain value map and 68.75 when asked only about far frontiers); a whole-frame feature for a frontier is no
better than no image, while the image region where the frontier projects is (FOM-Nav: 63.8, 63.7, 70.3).

### 2.2 From a VLM score to `p` and `τ`

- **Shape.** RPT\* wants independent values that do not sum to one. A written percentage per node (what
  LLM #2 returns today, and what VoroNav, TopV-Nav, OpenFrontier and SEEK return) fits directly. A single
  pick (InstructNav, VLMnav, MarvisNav) discards exactly what RPT\* uses. A softmax over option letters
  (ViTL) sums to one and needs its "elsewhere" option to be kept as residual mass.
- **Readout.** Two options, to be compared offline: a written integer percentage after a short
  justification (today's contract, works with constrained JSON), or the token probability of a yes/no
  answer per node (ViTL, CovNav, RTNav's verifier), which is cheaper, needs no parsing and avoids the
  round-number clustering of written values, but costs one forward pass per node. Scores that come from
  different images or calls are not on one scale (Explore-EQA multiplies each mark's probability by a
  view-level "is anything here worth exploring" for that reason, and ViTL does the same); asking for all
  nodes of a decision in one call, as the oracle does today, avoids the problem.
- **Calibration.** Nobody in the surveyed work checks that a VLM score behaves as a probability (the one
  proper scoring rule is SEEK's Brier score). For RPT\* the order depends on the absolute size of `p`
  (the survival factor), so two cheap guards matter more than fine calibration: a cap below 1, and a
  monotone map fitted once on scenes outside the evaluation split. Conformal methods (CPNav,
  Prune-Then-Plan) give a pruning threshold, not a probability, and they are fitted on data from the
  benchmark's own scenes (500 simulator episodes with oracle paths, split not stated; 5,500
  human-labelled frontiers), which a zero-shot claim cannot rest on. Their lesson still
  transfers: a VLM is more reliable at *rejecting* candidates than at ranking them (prune with the VLM,
  then go to the nearest survivor: 39.5 against 37.8 for the VLM's own pick and 36.9 for nearest
  frontier in Prune-Then-Plan), so coarse levels — ruled out, neutral, likely — may serve RPT\* better
  than fine differences between percentages.
- **Abstention.** The model must be able to say "cannot tell from these images" (PIGEON's zero token,
  REST's none-of-the-above); the node then keeps a neutral prior instead of a hallucinated number.
- **Search cost.** For a mapped region `τ` stays what it is (frontiers left). For a region behind the
  map's edge the size is unknown; the VLM can give a coarse class (closet / room / large open area /
  corridor), and geometry can bound it (how many depth rays left the map there without hitting anything).
  OpenFrontier multiplies the VLM's probability by a predicted information gain for the same reason.
- **Negative evidence.** What remains in a node is the prior times what has not been ruled out. Today
  that is the mapped-floor share and the effort discount. A VLM that looks at a room's frames can rule
  out more ("no bed in what is shown", RATE-Nav's three levels), but only for targets large enough to be
  recognised at that range (section 2.4).

### 2.3 The scene graph and the room segmentation are two different things

Today's "scene graph" bundles a **semantic layer** (object landmarks of many classes, room labels with
strength and hysteresis, glimpsed objects on exits) and a **geometric layer** (the clearance watershed,
room identity, frontiers per room, mapped floor, the finished-room ledger). They do different jobs:

| function today | provided by | still needed with a VLM? |
|---|---|---|
| a text interface for a model that cannot see | semantic layer | **no** — this is what the VLM replaces |
| nodes with separate search events for RPT\* | geometric layer | yes, in some form (rooms or portals + zones) |
| progress bookkeeping: mapped share, effort, "finished" | geometric layer + sight ledger | yes; the per-cell sight record already exists and is not tied to room ids |
| commitment: a visit holds a room for up to 60 actions | geometric layer | yes, at whatever node level is used |
| an entry point and a search cost per node | geometric layer | yes |

So the question splits. **Dropping the semantic layer is the proposal itself and loses little**: the
labels exist only to be read by LLM #2, and in the literature explicit room nodes add 0.7 SR (SG-Nav) and
a room cue adds 0.2–2.8 SR on average (CLUE, HRO, RPV-SemNav), concentrated in room-specific targets such
as toilet. What goes with it, and must be replaced or consciously dropped: the non-target classes that
the geometry heuristics read (the list of classes that cast no frontier shadow, the support-surface
list, glances cued by non-target boxes) and the door boxes that force cuts in the watershed.

**Dropping the geometric partition is a separate, riskier step.** Gains: no partition instability
(renumbering, slivers, fragments, the merge threshold, the glimpsed room that names the hallway — the
Ranchester failure recorded in `core/planning/object_search/rooms/opening_nodes.py`), no door
detector, fewer hand-set rules to disclose (the five finished-room rules). Losses: the unit that
makes "this place is done" well defined; the independence RPT\* assumes; a stable identity for
commitment.

The published results on *our two benchmarks* do not settle it. Systems that carry rooms or regions
(SysNav, STRIVE, RATE-Nav, CogNav, fast–slow ReAct) reach 63–72 SR on HM3D v0.1 and 47–52 on MP3D.
Region-free systems reach the same band when they add something else: TravExplorer 70.0 and 48.8 with
a tour over frontiers, stair traversal and no language model; RefBeliefNav 69.2 and 49.1 with
revisitable waypoints; PIGEON 67.1 and 52.6 with GPT-4o, stairs and a verification stage that its own
ablation credits with ten points; ZONDA 66.5 and 48.2 with stairs and multi-view verification. The
image-frontier systems MarvisNav, DRIVE-Nav and OpenFrontier sit at 55–61 and 41–42, although they lead
on HM3D v0.2. The direct measurements favour a region level where one was tested: RATE-Nav attributes
+9.9 SR to its region watershed; STRIVE's room-level decisions beat viewpoint-level ones by 4.6 SR;
ERA finds that goal sets made of frontiers alone run dry (8 % of HM3D v0.2 episodes for VLFM, 23 % for
its own frontier-only variant, with sensor depth); and RefBeliefNav loses 4.9 SR without revisitable
waypoints. None of this proves that rooms are necessary; it says that if they go, something must take
over "regions of mapped floor still to inspect" (pipeline B's zones).

### 2.4 Depth to 5 m, RGB beyond it

What the camera shows past the depth limit can be used in five ways, in decreasing order of confidence:

1. **Preview of unmapped space through an opening.** The room beyond a doorway is in the image long
   before it is in the map. Today it is not a node until it is mapped, and when exits were nodes
   (`exits_as_nodes`, deprecated 2026-10-21) the text model had nothing to value them by and chased every
   gap. A VLM values the opening by what is visible through it. Closest precedents: VoroNav's far-sight
   captions per candidate (+2.5 SR, more than the map-derived object lists), OpenFrontier and MarvisNav
   marking frontiers on the RGB frame, RayFronts/R2F anchoring beyond-range evidence at the frontier the
   viewing ray leaves the map through. Such a preview is a hypothesis, not a fact: GPT-4o's guess of the
   room type behind a frontier is right 68 % of the time and 26.5 % of its predictions are refuted on
   arrival; deleting a refuted prediction works better than decaying it (HGR: 72.4 against 69.4 SR on
   GOAT-Bench).
2. **Rule-out at range.** A living room seen from its doorway at 6–8 m contains no bed; today it is
   finished from outside only if 80 % of its cells were in view within 4 m. For large targets a VLM (or
   the detector's silence) can lower that room's remaining probability and its `τ` without a visit. This
   is the "cheaper visit" of section 1.1.
3. **Spot at range.** The target itself, far and without depth. The current system already keeps a
   bearing-only guess and has been burnt by it (one remembered guess per target, capped pursuit). A VLM
   can confirm or reject the glimpse before 50 actions are spent; a far estimate should stay an attractor
   that is overwritten by the first in-range view, never a terminal goal (WildOS, ERA, EZREAL).
4. **Layout cues.** Corridor continuing, dead end, closet, stairs, window, mirror, glass: these set `τ`
   and remove frontiers that are not passages (specular and transparent surfaces are 67 % of the wrong
   frontier predictions corrected in HGR).
5. **Speculative geometry.** Monocular depth aligned to the valid 0.5–5 m band can extend range
   estimates, but only for *costs and gains* (how far is that doorway, how much floor lies beyond), never
   for occupancy. ERA's measurements set the limits: about 10 % relative error, VLFM collapses from 68 to
   11 SR when predicted depth is integrated as if it were sensor depth, and the damage comes from depth
   edges rather than from the far range. `core/mapping/depth/` already holds Depth Anything V2 and V3;
   whichever model is used must be checked for HM3D / MP3D / Gibson scenes in its training set.

Two limits. **Identification from RGB degrades fast with range**: three independent systems put about
3 m as the limit for confirming a target (CSG-Nav: a small plant recognised 5/5 at 1 m and 1/5 at 5 m, a
chair behind a doorway 0/5 at 5 m; RTNav and CLOSER use 3 m), so uses 2 and 3 are for large targets and
room-level cues, and a far answer is a hint to be verified. And **the far field must not write into the
trusted map**: every system that works keeps it as evidence attached to a frontier or a direction, and
hands the local planner a goal at the map's edge (LRN).

### 2.5 "A detector only for the target"

With YOLO-World the vocabulary size barely changes the cost of a frame, so the saving is not compute: it
is the 27.8 label calls per episode, the room-type vocabulary, and the dependence of every probability
on noisy context boxes. With oracle perception the context objects are ideal, so the current chain is at
its best there and a VLM can only match it; the comparison that matters is under a learned detector
(section 6). Two additions keep the spirit of "target only" and are worth their cost: look-alike classes
as competing labels for the target (ApexNav, AECNav: the confuser term), and a VLM check of each target
candidate with its surroundings and a second view before the lock or the STOP.

### 2.6 What to ask the VLM, when, and what it costs

Design rules that the evidence supports:

- **Narrow questions, structure outside the model.** With a 4B model, "does the goal lie this way" per
  view is useful, while route choice is at chance, distances are constant and stop votes are 31–47 %
  precise; thirteen attempts to hand the model more responsibility gained nothing (Structure over
  Reasoning). How much the model's size matters depends on what it is asked. Where it chooses the
  motion itself the loss is large (WMNav: 58.1 with Gemini 1.5 Pro, 46.1 with Qwen2.5-VL-7B, 29.7 with
  the 3B); where it rates or picks among marked candidates on top of a geometric stack the loss is
  small down to about 8B (OpenFrontier: 77.3 with Gemini, 74.2 with LLaVA-1.6; MarvisNav: 81.2 with
  Gemini 3.5 Flash, 76.0 with Qwen3-VL-8B) and then falls off a cliff (RTNav: 72.3 with a 9B, 37.8 with
  a 4B).
- **Marks, not whole frames; memory on the choice, not in a transcript.** Numbered marks on the image
  (VLMnav +17.2 SR); exploration state drawn on the candidate (MarvisNav: 81.2 against 71.9 with no
  memory and 73.6 with a text history); a few de-duplicated keyframes rather than the recent history
  (ImagineNav++ +11 SR; naive history −1.4 to −6.4 in VLMnav; text + top-down map worse than no memory
  in WMNav).
- **Ask rarely, at decisions that are expensive to undo.** Loop points as today (mean 5.3 per episode,
  median 4). MarvisNav makes 4.1 calls per episode, PIGEON 6.2, fast–slow ReAct 5.0; WMNav makes 54.
- **Local, open weights, deterministic.** A 7–9B VLM is the published floor for selecting among marked
  candidates. On the 24 GB card it should fit beside the renderers, but not for free: the 3B text model
  that now gives the same result as the 14B takes 3.5 GB and leaves room for twenty scenes in parallel,
  the 14B took 15–18 GB and left room for eight (entry 047, paper draft), and a VLM with image tokens
  will sit nearer the second (to be measured). On the 8 GB laptop it does not fit beside anything, so
  full runs belong on the workstation. Answers must be recorded and replayable as the LLM's are today,
  keyed by prompt and image hashes.

## 3. Three pipelines

Common to all three: occupancy from depth within 0.5–5 m, frontiers and the per-cell sight record as
today; a detector with the target's prompts only; a **keyframe store** (pose-tagged thumbnails kept when
a frame adds new footprint, with the cells each one covers and a mask of its pixels without valid
depth); one VLM service called at loop points; RPT\* and A\* downstream.

### Pipeline A — rooms stay, the VLM's eyes replace the labels (plus portal nodes)

**How it works**
1. Map, frontiers, sight record, target detector, keyframe store as above.
2. The clearance watershed partitions the mapped floor into regions as today, but a region carries no
   label and no objects: only its geometry facts (mapped share, frontiers left, effort) and up to three
   keyframes chosen to cover the most of it; for a region never entered, the frame that looks into it.
3. For every exit whose far side is visible in a stored frame but is not yet floor on the map, a
   **portal node** is created at the exit, with that frame and a mark at the exit's projected position.
4. At a loop point one call: the images, each tied to a node id; one line of facts per node; the target's
   name. The reply gives, per node, a justification of a few words and a percentage that the target is
   in the part of the node not yet seen; for a portal also what lies beyond (kind of space or "not a
   passage") and a coarse size.
5. The reply passes through today's checks and exploration-state adjustment unchanged; a portal's value
   is zeroed when it is not a passage and its `τ` is set from the size class.
6. RPT\* orders rooms, portals and target landmarks; the agent commits to the first. A room visit is
   today's (transit, frontier walk); a portal visit is the existing peek (walk to the threshold, look
   in). When a portal is consumed, the prediction is compared with what was found and logged.

**What the VLM scores, and how it reaches RPT\*:** a node (region or portal); the percentage becomes
`p(v)` exactly where LLM #2's number enters today.
**RGB beyond depth:** a region's keyframes show its far side, so the VLM can rule a room in or out from
its doorway; portal nodes exist *because* RGB sees what depth does not; the reply may flag the target as
visible in an image, which lowers the detector's threshold for that view (MarvisNav) or creates a
bearing-only node.
**Stays:** occupancy, watershed and room identity, the finished-room ledger, RPT\*, A\*, visits, closing.
**Goes:** context classes, object landmarks other than the target, LLM #1 with its relabelling rules and
its 27.8 calls per episode, the room-type vocabulary, the text oracle. Door prompts are the one open
choice: keeping them costs nothing in the detector and protects the partition; dropping them is the
strict reading of "target only" and should be measured separately.
**Takes from:** SysNav and STRIVE (the room as the VLM's decision unit, a representative image per room,
room-level over viewpoint-level); RATE-Nav (the model judges a region from its key frames at three
levels); OpenFrontier (history frames move a frontier's probability from 0.65 to 0.10); MarvisNav
(state belongs on the choice); fast–slow ReAct (map-indexed keyframes, fail-open calls); PIGEON
(maximum-coverage frame selection, an "insufficient information" answer); VoroNav (far-sight evidence
per candidate); HGR (verify and retract what was predicted behind a frontier).
**Advantages:** the smallest change that tests the idea; a clean paired comparison because only the
oracle changes; RPT\*'s instance and its independence assumption untouched; the room as the model's
unit is supported where it was tested inside one system (STRIVE, RATE-Nav); removes about 28 of 33
model calls per episode; no longer hostage to context detections.
**Disadvantages and risks:** inherits the watershed's instabilities; a region's frames may not show its
unseen part; multi-image reasoning errors and loss of resolution on contact sheets; under oracle
perception the current labels are already ideal, so parity is the realistic outcome there; the prior's
measured value is small, so the gain has to come from portals and rule-out, which are the new, unproven
parts; and it keeps the unit whose transits take nearly half of our actions (the published room-level
systems also have the lower SPLs, 26–34 on HM3D v0.1 against 35–37 for the best systems that decide at
frontiers, a comparison confounded by success rate and perception).

### Pipeline B — a portal graph: no scene graph, no room segmentation

**How it works**
1. Map, frontiers, sight record, target detector, keyframe store. No rooms, no doors.
2. Frontier cells are grouped into **portals** by connectivity on the free-space graph (never across a
   wall), and portals whose outward rays converge on the same unseen pocket are merged into one node
   with several entry cells. Identities are tracked over time by overlap of members.
3. **Zones** take over what the room ledger does: connected patches of mapped free floor that the sight
   record has not yet seen within detection range, above a minimum area.
4. For each portal the store supplies the frame that looks most squarely through it, with the portal
   inside the image and not occluded by nearer geometry; for each zone, the frame that covers most of
   it. A portal without a usable frame keeps a neutral prior and gets a cheap glance scheduled.
5. At a loop point one call with set-of-marks images: a letter at each portal's projected position,
   coloured by state (new, seen before, partly explored), zones outlined, and a few de-duplicated
   frames of the floor already searched next to each. Per portal: what lies beyond (room kind,
   corridor, dead end, not a passage), a percentage that the target is beyond it and not yet seen, a
   size class. Per zone: a percentage that the target is in the outlined, uninspected area.
6. Nodes for RPT\* are merged portals, zones and target landmarks; `p` is the percentage times passage
   validity after the calibration map; `τ` comes from the size class or the open-ray gain; travel costs
   come from the grid. A portal visit is a peek, a zone visit a bounded sweep, with today's commitment.
7. On arrival the prediction is compared with what is seen; a refuted prediction is deleted rather than
   decayed (HGR) and the pair is logged for the calibration map.

**What the VLM scores, and how it reaches RPT\*:** a portal or a zone; the percentage is `p(v)`.
**RGB beyond depth:** this pipeline is built on it. A mark sits on the edge of the mapped floor and the
question is about the image content beyond that edge.
**Stays:** occupancy, frontiers, the sight record, RPT\*, A\*, peek and sweep visits, closing.
**Goes:** the watershed, room identity and registry, the finished-room rules, the door detector, labels,
landmarks, both LLMs.
**Takes from:** OpenFrontier (a probability per marked visual frontier, times a gain; robust to the
choice of VLM); MarvisNav (state-coloured marks, sparse decision sites, depth-consistent projection,
arrows where a route leaves the image); DRIVE-Nav (persistent directions with a representative view,
run with a local Qwen3-VL-8B: +4.2 SR, +5.1 SPL on HM3D-OVON); PIGEON (points of interest that carry
their raw observations); SOR-Nav (frontier clusters by graph connectivity, tour ordering); 3D-Mem, SCOPE
and HGR (frontier snapshots; hypotheses about what lies behind, verified on arrival); RayFronts and R2F
(far evidence anchored at the frontier by direction and re-cast when the frontier moves); WildOS and LRN
(the far field as an attractor, the goal handed over at the map's edge); ERA and RefBeliefNav (never
only frontiers: known-free viewpoints and revisitable waypoints, −4.9 SR without them); FOM-Nav and
VLMnav (localise and number the candidate in the image).
**Advantages:** no partition to go wrong; fewest components; the glimpsed-room problem disappears, since
a glimpse is a portal with a picture; decisions are taken at the threshold or from afar, which is where
the systems that mark frontiers on the image get their SPL (MarvisNav 36.2, DRIVE-Nav 35.9, PIGEON 35.8
on HM3D v0.1) and where our transits cost; the room rules and their thresholds leave the paper.
**Disadvantages and risks:** correlated portals break RPT\*'s independence wherever the merge is wrong;
portal identities churn as the map grows, which means more re-solves and more calls; "this place is
done" has to be rebuilt as zones, new bookkeeping to design and to disclose; in SR, MarvisNav, DRIVE-Nav
and OpenFrontier trail the region-based systems on HM3D v0.1 and MP3D (PIGEON, with GPT-4o, stairs and
a verification stage, does not); selection among marks is unreliable below about 8B (RTNav's 4B: 37.8;
an untrained 2B scored below random in IntentNav).

### Pipeline C — a belief field from per-frame scores: no rooms, nodes derived from the field

**How it works**
1. Map, frontiers, sight record, target detector, keyframe store. No rooms.
2. On each keyframe (new-area frames only, every 5–10 actions) the VLM answers one narrow question per
   image sector, or per lettered frontier in view plus "elsewhere": does the target lie this way,
   visible or farther on? The score is read from the token probabilities and centred, so that a low
   answer counts against that direction.
3. Each score is written to the map along the sector's rays: onto observed free cells inside the depth
   range, and, along rays that left the map without hitting anything, onto unknown cells out to a far
   limit with a weight that decays with range. Far evidence is also filed under the frontier group the
   ray exits through, so that it survives when the map grows.
4. Evidence is fused per cell (tempered accumulation with a leak and a discount for repeated looks from
   the same place); cells seen within detection range without a detection leave the support.
5. At a loop point nodes are frontier groups and zones as in B; every unknown or uninspected cell is
   assigned to exactly one node, the one whose entry is nearest by path, so no mass is counted twice.
   `p(v)` is one minus the exponential of the belief mass in the node's region, `τ(v)` grows with the
   region's area.
6. RPT\* and A\* as usual.

**What the VLM scores, and how it reaches RPT\*:** a (frame, sector) pair; scores become cell evidence,
and a node's `p` is the mass of its region.
**RGB beyond depth:** the ray extension is the mechanism — RayFronts and R2F with a VLM score in place
of an embedding.
**Stays:** occupancy, frontiers, the sight record, RPT\*, A\*, closing. **Goes:** rooms, graph, both
LLMs, doors.
**Takes from:** VLFM (the value map; averaging instead of overwriting is worth +7.3 SR); ViTL (letter
logits with an "elsewhere" option, times a frame-level yes-probability); Structure over Reasoning (one
narrow question to a small VLM, tempered belief, revisit discount); SemNav (a GPT-4o float per frame
feeding an expected-cost planner); WTRP-Searcher and BeliefMapNav (sequence planning over map-derived
scores; visibility as negative evidence, +5.3 SR); RayFronts and R2F; AECNav and ApexNav (log-odds
evidence, misses as negative evidence); RefBeliefNav (potential plus information gain minus cost).
**Advantages:** no identities to track, since the field persists and nodes are recomputed; `p` and `τ`
follow from one quantity with independence by construction; many weak looks average out; narrow
questions suit a small model (62 ms per three-token answer in Structure over Reasoning); extends to
several targets; the map itself is the most proven mechanism in the field, since value maps filled by
an embedding score hold the best standard-protocol SPL on HM3D v0.1 and MP3D (TravExplorer, AECNav,
ConsistNav).
**Disadvantages and risks:** the most calls by far; what is new here is exactly what is unproven: a
generative model's score in the map in place of an embedding (ViTL +1.7 SR over VLFM; SemNav
54.9 / 35.9), the cast into unknown space, and nodes derived from the field; evidence cast into unknown
space smears when the geometry is wrong; several new constants (decay, leak, the mass scale) with no
principled value; the reason for a decision is no longer a sentence one can read.

## 4. Side by side

| | A: rooms + VLM (+ portals) | B: portal graph | C: belief field |
|---|---|---|---|
| scene graph / room segmentation | semantic layer gone, geometric partition kept | both gone | both gone |
| what the VLM scores | region, portal | portal, zone | frame sector or lettered frontier |
| VLM calls per episode (estimate) | 5–6, 6–12 images each | 6–12 | 20–50 narrow ones |
| `p` for RPT\* | written percentage per node | percentage × passage validity | belief mass of the node's region |
| independence of nodes | as today | depends on the portal merge | by construction |
| far-field use | region frames, portals, far-target flag | the core mechanism | rays past the depth limit |
| new bookkeeping | keyframe store, portals | + portal identity, zones | + field, far rays, partition |
| hand-set rules removed | label rules, vocabulary | + all room rules | + all room rules |
| main risk | gains limited to the new parts | independence and churn | cost, smeared evidence |
| published support on HM3D v0.1 / MP3D | the room unit wins inside one system (STRIVE, RATE-Nav); room-level systems reach 63–72 / 47–52 SR with the lower SPLs | marks at frontiers: PIGEON 67.1 / 52.6, MarvisNav 60.5 / 42.1, with SPL at 36 | the map is the most proven mechanism (TravExplorer 70.0 / 48.8 with an embedding score); a generative score in it is not (SemNav 54.9) |

## 5. Which to try first

**Pipeline A, in two increments (regions scored from images, then portal nodes), after a headroom
measurement.** Reasons:

1. It isolates the question that was asked. Only the source of `p` changes; the RPT\* instance, the
   visits, the ledger and the closing stay, so a paired run attributes any difference to the VLM.
2. The literature does not settle whether a region level is needed, so it should not be dropped in
   the same step in which the scorer changes. Of the ten standard-protocol, training-free systems at
   or above 63 SR on HM3D v0.1, five carry a room or region cue (CogNav, fast–slow ReAct, RATE-Nav,
   ASCENT, SysNav) and five do not (TravExplorer, RefBeliefNav, PIGEON, ZONDA, ConsistNav); the second
   five get there through verification, stair handling, revisitable candidates and execution
   discipline, not through a search prior. With the rooms kept, a null result of the first experiment
   says something about the VLM and nothing about the partition.
3. It removes the larger cost first: the label calls are 84 % of today's model calls.
4. Portal nodes add the far field without a new node theory, reusing the peek visit and the retired
   `exits_as_nodes` path, now with a picture to value each exit by.
5. It builds what B and C need anyway (keyframe store, VLM service with recorded answers, the offline
   decision benchmark, the calibration map). B then becomes one switch on the same harness — nodes are
   portals and zones instead of regions — and the question "do we need the room segmentation" gets an
   experimental answer rather than an argument. C is worth building only if B shows that the limit is
   the views available per node, or for the real-time case on a drone where a small model asked narrow
   questions is the only thing that fits.

What this order does not claim is that rooms are the better end state, and the published numbers do
not decide that for path efficiency either. The best standard-protocol SPLs on HM3D v0.1 belong to
systems that decide at frontiers (TravExplorer 37.2, MarvisNav 36.2, PIGEON 35.8) and the room-level
systems are at 26–34; but the ratio of SPL to SR, which takes the success rate out, is 0.53 for
TravExplorer and PIGEON, 0.54 for STRIVE and for our oracle run, and 0.60 for MarvisNav at a lower SR,
and inside one system the room unit was the more efficient one (STRIVE, +3.8 SPL). The case for
portals rests on our own budget, in which the transits between rooms take nearly half of the actions.
The second increment of A is B's mechanism inside the current structure and the first place where
that case can be tested; if it holds there, B is the destination. The draft read in section 9 starts
from that destination; the decision replay of step 0 can compare the two starting points on the same
recorded decisions before either is built.

## 6. How to test it against the current method

**Step 0 — headroom and decision quality, no closed loop.**
- *Ceiling.* On the 180-episode MP3D set and the 90-episode HM3D set, run the current agent with an
  analysis-only prior that puts high `p` on the node leading to the nearest goal and low `p` elsewhere.
  With flat-`p` and the LLM this gives three points on one axis; the distance from the LLM to the ceiling
  is the most any scorer can win. Such a prior is privileged, lives in the benchmark runtime and never
  in `core/`, and is non-publishable like `--perception ground-truth`.
- *Decision replay.* The recorded runs hold every oracle decision (about 10,800 in each HM3D run) with
  its nodes, and an episode replays exactly from its recorded answers. Replay, capture the frames at each
  decision, build each candidate prompt, and score every variant on: the rank it gives the node that
  the shortest path to the nearest goal enters first (for decisions with two or more nodes); the
  Brier score of `p` against "a goal instance is in this node"; the regret in expected time of the
  order RPT\* returns. For the far field, the share of unmapped area
  a preview rules out without losing the target (RayFronts' measure). This costs minutes per variant and
  is where prompts, image layouts, the readout and the calibration map are chosen. The unit can be
  varied in the same replay: a region with its keyframes, a marked frontier cluster, or a source frame
  per cluster as in the draft of section 9, the frontier clusters being recomputed from the replayed
  map.
- *Hygiene.* Do the prompt work on scenes outside the val splits (HM3D train scenes or Gibson) so that
  the final number is a held-out one — which also retires the "developed on the evaluated episodes"
  caveat for this component.

**Step 1 — closed loop, paired, fixed sets.** Same seed, closing and perception in every arm. Arms:
flat-`p`; the current LLM chain; A without portals; A with portals; later B. Sets: the two fixed sets
above plus the same-storey never-seen failures and an equal number of sampled successes, so that a
regression shows. Metrics: SR, SPL, SoftSPL, DTG and steps as today, and three that isolate the search:
actions until a goal instance is first in view within detection range (computed after the fact), the
share of episodes with a sighting inside the budget, and the same on the long-search subset (more than
five decisions in the baseline). Also model calls, tokens and wall time per episode. Statistics as in
entry 046: McNemar on SR flips, paired bootstrap on SPL. The share of outcomes that flip between two
arms is known from two paired runs: 10 of 180 between flat-`p` and the LLM (entry 046) and 151 of 2,000
between the 3B and the 14B model, that is 6–8 %. At 8 %, and for 80 % power at the 5 % level, 270
paired episodes detect a difference of about 5 SR points, 600 about 3 and the full split about 2. On
the fixed sets SR can therefore show only a large effect, and the search-only metrics, which are not
binary, carry this step.

**Step 2 — full splits, two perception regimes.** (i) Oracle target, where the current chain also gets
oracle context objects and the VLM arm gets RGB: this asks whether eyes can match ideal labels.
(ii) A learned detector for both, target-only plus VLM verification in the new arm: this is the
comparison a reader will care about and the one in which the new arm should win, because context
detections no longer feed the probabilities and wrong STOPs are attacked directly. Report both beside
the published standard-protocol rows of [SUMMARY.md](SUMMARY.md), section 6.

**What would count as success.** No loss of SR on the easy half; on the long-search subset fewer actions
to first sighting and higher SPL; at most a fifth of today's model calls; and, under the learned
detector, a wrong-STOP rate well below today's 24 %.

## 7. Where it would attach (for later; nothing is built)

- The model seam is `core/planning/object_search/interfaces.py` (`RoomReasoner`): a sibling contract that
  also takes images, with the HTTP client in the benchmark runtime's services as today.
- Nodes and prompts live in `core/mapping/topology/search_node_oracle.py` and `search_node_prompts.py`;
  the test that keeps object and room names out of a prompt applies to the VLM prompt as well.
- Portal nodes and the peek visit exist under `core/planning/object_search/`: `rooms/opening_nodes.py`
  (`exits_as_nodes`) and `exploration/doorway_peek.py`; the unseen floor beyond an exit is already
  measured there. Keyframe footprints can reuse the ray casting of `world_model/sightlines.py`.
- The flat baseline exists (`LoopSettings.flat_p`); recorded answers and replay exist for text and would
  need image hashes in the key (`core/mapping/topology/llm_answer_cache.py`).
- Nothing privileged may enter `core/`: the ceiling prior of step 0 belongs beside the ground-truth
  perception source in `tasks/planning/objnav_benchmark_runtime/`.

## 8. Open questions

- How large is the headroom? Everything in section 5 is conditional on step 0 showing some.
- Written percentage or token probability, and how stable is either across re-solves of the same node?
- How many images per node before a 7–9B model degrades, and contact sheet or separate images?
- Does a rule-out from the doorway hold for the benchmark's small targets, or only for bed, sofa, toilet?
- Can portals be merged reliably enough for RPT\*'s independence, or should correlated portals be given
  to the solver as one node with several entries from the start?
- Which monocular depth model, if any, is clean of the benchmark scenes?
- Under a learned detector on HM3D v0.1, how many of the wrong STOPs are real but unannotated instances,
  i.e. what is the ceiling a verifier can reach?

## 9. The draft pseudocode read against this note

A draft, *VLM-Guided Object Search* (`docs/zson_vlm_pseudocode.tex`), appeared in the working tree
while this note was being written (untracked; this review did not write or edit it). It was still being
edited, so the reading below is of its state at 14:55 on 2026-11-19 and is keyed to the steps of its
loop, not to its note numbers. The draft scores **frames**: after a warm-up turn, each round selects the
most informative recent images, a VLM gives one score per image, every frontier cluster takes the score
of the image it was created from, old clusters keep their stored scores, and RPT\* orders the clusters
with Dijkstra distances; the agent walks to the first with A\*, visits the nearest frontier, and
repeats. A detector for the target runs on every frame. No rooms, no scene graph.

In the terms of section 3 this is pipeline B's node set with pipeline C's scoring unit. Its nearest
published relatives are SemNav (one float per frame, read as a frontier's probability, into an
expected-cost planner: 54.9 / 35.9, and 54.6 / 34.3 when the same scores are used greedily) and the
frontier-snapshot systems (3D-Mem, HGR, SCOPE: each frontier keeps the image it was seen in).

| step of the draft | what the evidence says | amendment worth considering |
|---|---|---|
| warm-up turn, once at the start | consistent with today (the warm-up finds the target in 11 % of the HM3D episodes) | none |
| the VLM is asked once per round, when a destination is planned | supported: 4–6 calls per episode in MarvisNav, fast–slow ReAct and PIGEON; asking at every frontier choice is worse than not asking (64.8 against 65.35) | none; expect more rounds than today's 5.3 queries per episode, since a frontier cluster is a smaller step than a room |
| "most informative images from the recent steps" | frames chosen for new coverage and de-duplicated help (+11 SR with about 20 keyframes, +0 with all frames, ImagineNav++; maximum-coverage selection in PIGEON) | select per live cluster, not per buffer: an image that no cluster was created from cannot change any score. For each cluster, the stored frame that looks most squarely through it |
| one score per image | a whole-frame input for a frontier is no better than no image (63.8 against 63.7), while the image region where the frontier projects is (70.3, FOM-Nav); numbered marks are worth +17 SR (VLMnav). One score per frame cannot tell apart two clusters created from the same frame | draw a mark per cluster on the frame and ask for a score per mark: still one call, and a `p` per node (OpenFrontier, MarvisNav) |
| context = what has been seen so far | a transcript is the weak form of memory: 73.6 with a text history against 81.2 with the state drawn on the choice (MarvisNav); a naive history costs 1.4–6.4 SR (VLMnav) | a state on each candidate (new, seen before, partly explored) and a few frames of the floor already searched beside it (OpenFrontier: 0.65 → 0.10) |
| the score goes to the cluster created from the image | the creating frame is the first sighting, usually the worst view (far, oblique, at the image border); clusters split, merge and move as the map grows | replace the stored frame when a better view exists; track cluster identity by overlap (RayFronts re-casts when a frontier moves, DRIVE-Nav keeps persistent directions) |
| old clusters keep their stored scores and are not scored again | a stored score is stale once the space behind the cluster has been partly seen; scores from different calls are not on one scale (the reason Explore-EQA and ViTL normalise across views); for a prediction refuted on arrival, deleting beats decaying (HGR) | re-score the live clusters in the same call with what has been seen since; a cluster with no usable frame gets a neutral prior (today's flat 0.15) |
| `RPT*(F, scores, d)` | RPT\* assumes independent search events with a probability below 1 and a search cost each. Clusters that open onto the same unseen space are one event counted twice; a frame score is not a probability; and without a search cost a closet and a wing of the house cost the same | merge clusters by what lies behind them (section 2.1; the solver's limit of twelve nodes needs it too); a cap and a map from score to `p` fitted outside the evaluation split; `τ` from a size class or from the depth rays that left the map through the cluster (section 2.2) |
| follow the path to the first cluster | today 13 % of the transits end because the place was seen on the way, and 13 % fail | release rules: resolved on the way, unreachable, a bound on actions |
| one walk to the nearest frontier on every arrival | consistent with today, where most of the looking is done on the way in (four visits in ten walk to no frontier) | none |
| the detector on every frame, then target closing | wrong stops are the dominant failure with a real detector: 24 % of the episodes in our YOLO-World run, 58–65 % of MarvisNav's failures, 22 % of TravExplorer's episodes | a VLM check of the candidate with its surroundings and a second view before the lock and the STOP (7–13 SR in STRIVE, PIGEON, RATE-Nav, AECNav) |
| (not in the draft) what happens when no frontier is left | goal sets made of frontiers alone run dry in 8–23 % of episodes (ERA); −4.9 SR without revisitable waypoints (RefBeliefNav). With depth to 5 m and detection to 4 m, floor can be mapped without having been inspected | zones (mapped floor not yet seen within detection range) and revisits as nodes |

The heaviest rows are the fourth and the eighth: what is scored, and what RPT\* is given. With those
two amended the draft is pipeline B. It does not change the test plan, it sharpens it: the decision
replay of step 0 can score, on the same recorded decisions, the draft's variant (a frame score per
creating image), a mark per cluster, and a region with its keyframes. That is a measured answer to "per
frame, per cluster or per room" before any closed-loop run.
