# One VLM instead of two LLMs and the all-objects detector — conclusions

Written 2026-11-19 on branch `feat/objnav-benchmark-infra-nadav`. **Thinking and literature only:
nothing is implemented.** This file is the short version, meant to be read on its own. The evidence is
in the two long documents beside it:

- [SUMMARY.md](SUMMARY.md) — the literature review (123 papers) and the state of the art on HM3D and MP3D.
- [INTEGRATION.md](INTEGRATION.md) — the analysis of our own runs, the design reasoning, the three
  pipelines in full, the test plan, and a reading of the draft pseudocode.

Numbers are SR / SPL in percent. "Ours" are post-hoc readings of recorded runs, not new experiments.

## 1. The answer in one page

1. **The idea is sound, but expect path efficiency and cost to move, not the success rate.** Every
   published ablation, and two of our own, put the value of "a semantic choice of where to go" at a few
   SR points. A better scorer cannot win much there. It can win by judging a place *without walking to
   it*, which is exactly what RGB beyond the depth range offers.
2. **Keep RPT\*.** No published system asks a VLM for a probability per node and then orders the nodes
   exactly. Those that order heuristically beat greedy selection on the same scores. This is the part
   of the method worth keeping whatever feeds it.
3. **Score places, not frames.** RPT\* needs nodes that are separate search events: a region of mapped
   floor, or a distinct region behind the map's edge. Frames are the evidence the score is read from.
4. **The scene graph and the room segmentation are two different things.** The semantic layer (object
   landmarks, room labels) exists only to give a blind model something to read; the VLM replaces it and
   little is lost. The geometric partition (which floor belongs to which room, what is finished) is a
   separate and riskier thing to drop.
5. **Try Pipeline A first** (rooms kept, VLM scores regions from keyframes, plus "portal" nodes for
   space seen through openings), after a cheap offline headroom measurement. Pipeline B (no rooms, no
   scene graph) is then a switch on the same harness.
6. **With a real detector the search prior is not the bottleneck; wrong stops are.** The same VLM
   should verify the target before the lock and before STOP.

## 2. What our own runs say

Current code, HM3D v0.1 val, oracle perception: **71.3 / 38.2** with the 14B text model and 71.4 / 37.4
with the 3B. MP3D: 65.6 / 29.1 (an earlier revision).

**The goal's storey comes first.** 348 of 2,000 episodes have the goal on another storey; the agent
stays on its own and wins 12 % of them. They are 305 of the 575 failures, and no scorer touches them.

**On the start storey (1,652 episodes, 83.7 / 45.2), failures sit in the long searches:**

| probability queries in the episode | episodes | SR | SPL | mean actions |
|---|---|---|---|---|
| 0 | 262 | 99.2 | 77.6 | 45 |
| 1–2 | 507 | 98.4 | 63.4 | 98 |
| 3–5 | 423 | 95.3 | 38.1 | 211 |
| 6–10 | 308 | 61.0 | 17.3 | 389 |
| 11 or more | 152 | 21.1 | 4.3 | 486 |

- Close to half the episodes are settled within two decisions and nearly always won. Nothing to gain.
- The 460 episodes with more than five decisions hold 240 of the 270 same-storey failures.
- In the 167 failures where the target was never seen, the agent ends a median 7.5 m from the goal after
  starting 8.2 m from it, and 41 % end farther away than they began. The budget went to the wrong part
  of the house.
- A decision costs about 40 actions. Transits between rooms take 46–48 % of all actions; the walk
  inside the room only 10 %. The agent mostly inspects a room while approaching it.

**The prior is worth little, and a bigger model does not help:**

- LLM against a flat `p = 0.15`: +3.3 SR [−0.0, +6.7], +1.5 SPL, not significant (180 MP3D episodes).
- 3B against 14B on the full split: 151 outcomes change, 77 one way and 74 the other; +0.8 SPL.

**With a real detector the picture changes.** YOLO-World (an older revision): 41.3 / 22.3. Wrong STOPs
are 24 % of episodes (484), against 0.5 % with the oracle; 61 % of plant episodes, 49 % of tv_monitor.

**Practical facts for a VLM prompt.** A decision offers 3 nodes at the median, at most 5 in 90 % of
cases. An episode issues 5.3 probability queries and 27.8 room-label queries, so the labels are 84 % of
today's model calls.

## 3. What the literature says

**Read published numbers by protocol.** The differences between protocols are larger than most gaps
between methods: standard (500 low-level actions, full split, learned perception); WMNav-style (40–100
metre-scale decisions); simulator semantics (ours); and methods that use benchmark data.

**Who uses a VLM to decide where to go** — seven families by what receives the score: a whole frame
painted on a map (VLFM and descendants); headings of a panorama (WMNav family); marks drawn on the frame
(OpenFrontier, MarvisNav, PIGEON); one image per candidate (3D-Mem, DRIVE-Nav); rooms or regions
(SysNav, STRIVE, RATE-Nav, CogNav); a rendered top-down map; and no generative model at all (AECNav,
ConsistNav, TravExplorer).

**State of the art, standard protocol, training-free:**

| benchmark | best SR | best SPL |
|---|---|---|
| HM3D v0.1 | CogNav 72.5 / 26.2; TravExplorer 70.0 / 37.2; RefBeliefNav 69.2 / 35.3 | TravExplorer 37.2 (no language model); MarvisNav 36.2; PIGEON 35.8 |
| MP3D | PIGEON 52.6, STRIVE 52.3, AECNav 51.3 (a tie) | AECNav 25.9, ConsistNav 25.7 (no generative model) |
| HM3D v0.2 | AECNav 84.7 / 45.3; ConsistNav 84.2 / 41.2; MarvisNav 81.2 / 42.5 | |

Among results with simulator semantics, ours has the highest SR on HM3D v0.1 (ASCENT with an ideal
detector: 70.3 / 41.0) but not the highest SPL.

**Where the points come from, across all ablations:**

| component | measured effect |
|---|---|
| semantic choice of where to explore | +0.6 to +4 SR |
| ordering instead of greedy, same scores | +2 to +6 SPL |
| memory of the score, or state drawn on the choice | +7 to +11 SR |
| marks instead of whole frames | up to +17 SR |
| target verification and evidence fusion | +4 to +25 SR |
| execution and commitment discipline | +8 to +11 SR |
| handling other floors | +6.5 to +8.7 SR on HM3D v0.1 |

**The closest prior art to the idea:**

- *SemNav* — GPT-4o gives one float per frame, read as a frontier's probability, into an expected-cost
  planner. 54.9 / 35.9; the planner adds +1.6 SPL over greedy.
- *OpenFrontier* — a probability per frontier marked on the image, times a predicted gain.
- *SysNav / STRIVE* — the VLM picks the next room; room-level decisions beat viewpoint-level ones by
  4.6 SR at a third of the tokens.
- *RATE-Nav* — a watershed region partition and a VLM that rules a region out from its key frames.
- *MarvisNav* — the VLM picks a lettered route choice coloured by exploration state; 4.1 calls per episode.

**RGB beyond the depth range:**

- The standard value map throws the far field away: the score is computed from the whole image but
  painted only where depth has seen.
- Nobody stores a generative VLM's score beyond the depth range. This is open ground.
- What works: evidence attached to the frontier a viewing ray leaves through (RayFronts, R2F); a
  description of the far view per candidate (VoroNav, +2.5 SR); marks on the frame.
- Limits: confirming a target from RGB is unreliable beyond about 3 m. A guess of the room behind a
  frontier is right 68 % of the time, and deleting a refuted guess beats decaying it. Predicted depth
  must never be mapped as measured (VLFM falls from 68.2 to 10.8 SR when it is).

**Models.** The narrower the question and the more structure sits outside the model, the less its size
matters. Picking among marked candidates holds down to about 8B (MarvisNav 81.2 → 76.0, OpenFrontier
77.3 → 74.2) and collapses at 4B (RTNav 72.3 → 37.8). When the model chooses the motion itself the loss
is large (WMNav 58.1 → 46.1 → 29.7).

## 4. The design reasoning

**Per frame, frontier, cluster or room?** A frame is not a place. Raw frontiers are too many and count
one unseen room several times. The right units are a region of mapped floor, and a group of frontiers
that leads to the same unseen space (a "portal").

**From score to RPT\*.** A written percentage per node fits directly. A single pick throws away what
RPT\* uses. Ask for all nodes in one call so the scores share a scale. Cap `p` below 1, let the model
abstain, and fit one monotone map on scenes outside the evaluation split. A VLM rejects candidates more
reliably than it ranks them, so coarse levels may serve better than fine percentages.

**Five uses of the far field, most to least trustworthy:** preview an unmapped room through an opening;
rule a room out from its doorway; spot the target at range (a hint to verify, never a goal); layout
cues (dead end, closet, mirror, glass); speculative geometry for costs only, never occupancy.

**Target-only detector.** The saving is not compute. It is the label calls, the room vocabulary, and
probabilities that no longer depend on noisy context boxes. Worth adding: look-alike classes as
competing labels, and a VLM check of each candidate.

**How to ask.** Narrow questions, marks rather than whole frames, memory drawn on the choice rather
than kept as a transcript, and only at decisions that are expensive to undo.

## 5. The three pipelines

All take RGB + depth and hand RPT\* a set of nodes with `p`, `τ` and travel costs. All keep occupancy
from depth, frontiers, a target-only detector, RPT\* and A\*, and add a store of pose-tagged keyframes.

| | A: rooms + VLM + portals | B: portal graph | C: belief field |
|---|---|---|---|
| what the VLM scores | a region from its keyframes; a portal from the frame looking through it | a marked frontier group; an uninspected zone | a frame sector or lettered frontier |
| how it reaches RPT\* | percentage per node, where LLM #2's number enters today | percentage × "is a passage" | belief mass of the node's region |
| RGB beyond depth | rule a room out from its doorway; portals; far-target flag | the core mechanism | scores cast along rays past the depth limit |
| scene graph / rooms | semantic layer gone, partition kept | both gone | both gone |
| VLM calls per episode | 5–6 | 6–12 | 20–50 narrow ones |
| takes from | SysNav, STRIVE, RATE-Nav, VoroNav | OpenFrontier, MarvisNav, PIGEON, HGR | VLFM, SemNav, RayFronts |
| main advantage | smallest change; clean paired test | no partition to go wrong; decisions from afar | no identities to track; suits a small model |
| main risk | gain limited to the new far-field parts | correlated portals break independence; node churn | cost; smeared evidence; new constants |

## 6. Recommendation

**Pipeline A, in two increments, after a headroom measurement.**

1. Only the source of `p` changes, so a paired run attributes any difference to the VLM.
2. The literature does not settle whether a region level is needed: of the ten leading systems on HM3D
   v0.1, five carry rooms or regions and five do not. So do not drop it in the same step as the scorer.
3. It removes the larger cost first (the label calls).
4. Portals add the far field using the existing peek visit.
5. It builds what B needs anyway. B then becomes one switch.

This does not claim that rooms are the better end state. Our transits between rooms take nearly half
the actions; if portals show a gain in SPL inside A, B is the destination.

## 7. How to test it

**Step 0 — offline, no closed loop.**
- *Ceiling:* run the current agent with a privileged prior that points at the node leading to the goal.
  The distance from the LLM to that ceiling is the most any scorer can win. If it is small, stop here.
- *Decision replay:* the recorded runs hold about 10,800 decisions each. Replay them, capture the
  frames, and score each prompt variant on the rank of the correct node, the Brier score of `p`, and the
  area a preview rules out. The unit (frame, cluster, room) can be compared in the same replay.
- Do the prompt work on scenes outside the val splits.

**Step 1 — closed loop, paired, fixed sets.** Arms: flat `p`, the current LLM chain, A without portals,
A with portals, later B. Judge on actions to first sighting and SPL in the long-search subset; SR alone
needs the full split to show less than about 5 points.

**Step 2 — full splits, two regimes.** Oracle target (can eyes match ideal labels?) and a learned
detector with VLM verification (the comparison a reader cares about).

**Success:** no loss on the easy half; fewer actions to first sighting and higher SPL in long searches;
at most a fifth of today's model calls; a wrong-STOP rate well below 24 % with a learned detector.

## 8. The draft pseudocode

`docs/zson_vlm_pseudocode.tex` appeared in the working tree during this work; it is not part of this
commit and was not edited. It scores frames and gives each frontier cluster the score of the image it
was created from. That is Pipeline B's nodes with a per-frame score. Two amendments matter most:

- **Score a mark per cluster, not the frame.** One score per frame cannot separate two clusters born in
  the same frame, and a whole-frame input is no better than no image.
- **Give RPT\* a well-posed instance.** Merge clusters that open onto the same unseen space, map the
  score to a probability below 1, and add a search cost per node.

Also worth adding: zones and revisits when no frontier is left, and a VLM check before STOP.
INTEGRATION.md, section 9, goes through it step by step.

## 9. Not verified

- SOR-Nav's perception source in simulation, and DSCD-Nav's step budget.
- The HM3D version behind a few rows that state only counts, or nothing.
- Whether a 7–9B VLM fits beside the renderers on the 24 GB card.
- The call counts of WMNav and OpenFrontier are MarvisNav's measurements, not their own.
- The papers and the per-paper reading ledgers are outside the repo (`~/papers/vlm-scoring-survey/`
  and the session workspace).
