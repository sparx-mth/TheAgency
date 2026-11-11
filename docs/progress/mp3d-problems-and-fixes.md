# MP3D: every problem found and every fix made (2026-11-05 to 2026-11-10)

**Branch:** `feat/objnav-habitat-mp3d-nadav`. **Companion entries:** `entries/043` (the full-split run), `044` (the
closing), `045` (the room graph, the oracle's priors), `046` (the fixed set, the ablation, the single-storey fixes).
This document is the explanation those entries do not give room for: for each problem, the episode that showed it,
the root cause, how the code behaved before and after, why the new behaviour is better, what the change measured,
and whether it is a fix of the method or a workaround for the 3B model or for ground-truth perception.

**How to read the flags.**

| flag | meaning |
|---|---|
| **FIX** | a mechanism of the method was wrong and is now right; the change would stand under any LLM and any detector |
| **PARTLY** | the mechanism is real, but the change was sized or motivated by the 3B model's or the GT sensor's behaviour; re-examine when either changes |
| **WORKAROUND** | compensates for the 3B model or for GT perception; not a fix of the method, kept behind a flag, expected to be dropped or revisited when the model or the detector changes |
| **INFRA** | measurement or harness; no effect on what the method does |

**How to read the numbers.** Three kinds of evidence appear, of very different weight:

1. Entries 044-045 were checked on **15 hand-picked episodes with live LLM calls**: the same code flipped episodes
   between two consecutive runs (045: x8F5xyUWy9e/000038 S -> F, 2azQ1b91cZZ/000134 S -> S with 230 fewer steps),
   so their outcome columns are illustrations, not measurements. Those fixes are in commit `96cd374c`, which is the
   **baseline** of the fixed set, so the set does not measure them separately.
2. Entry 046's fixed set: **180 same-storey episodes in 8 flat houses**, every arm run from a frozen source tree with
   a replayable LLM answer store (a re-run of unchanged code is bit-identical, 14/14). Paired comparisons: dSR with a
   bootstrap 95 % interval, McNemar's exact p, and the list of flipped episodes, each attributed to a fix by the first
   action at which its recording diverges from the baseline's. The **minimum detectable effect on 180 episodes is
   about 5 points**; every per-fix effect below is smaller than that and is read from the flips, not from SR.
3. The recordings themselves (`steps.jsonl`, `trajectory.csv`, the maps), read with `single_storey_failures.py`.

---

## 0. The headline table

Arms on the 180-episode set. "filter" is the ground-truth on-object filter (section D1); "fixes" are the four
single-storey method fixes (B4, B5, C1, E6-E7).

| arm | perception | method | SR | SPL | dSR vs baseline [95 %] | p | flips +/- |
|---|---|---|---|---|---|---|---|
| baseline | GT, filter off | `96cd374c` | 62.8 % | 27.6 % | -- | -- | -- |
| flat-p | GT, filter off | oracle p = 0.15 everywhere | 59.4 % | 26.1 % | -3.3 [-6.7, +0.0] | 0.11 | 2 / 8 |
| fixes | GT, filter off | B4 C1 E6 E7 | 64.4 % | 27.0 % | +1.7 [-2.8, +6.1] | 0.63 | 10 / 7 |
| gt-base | GT, **filter on** | `96cd374c` + fixes OFF by config | 77.2 % | 32.8 % | +14.4 [+7.8, +21.1] | <0.001 | 33 / 7 |
| gt-fixes v1 | GT, filter on | B4 B5(map-based) C1 E6 E7 | 78.3 % | 31.4 % | +15.6 [+8.9, +22.2] | <0.001 | 35 / 7 |
| gt-fixes v3 | GT, filter on | B4 B5(sighting's plane, every plane) C1 E6 E7 | 77.2 % | 29.9 % | +14.4 [+7.8, +21.1] | <0.001 | 33 / 7 |
| **gt-fixes v4** | GT, filter on | B4 **B5(sighting's plane, raised planes only)** C1 E6 E7 | 77.8 % | 30.4 % | +15.0 [+8.3, +21.7] | <0.01 | 34 / 7 |
| gt-dining (stopped at 52) | GT, filter on | E6 alone | 65.4 % vs gt-base 71.2 % on the same 52; SPL 25.3 vs 28.1; +1 / -4 |
| gt-shape (stopped at 53) | GT, filter on | E7 alone | 67.9 % vs 73.6 % on the same 53; SPL 25.7 vs 28.8; +2 / -5 |
| **gt-fixes v5** | GT, filter on | B4 B5 **B6** C1; **E6 E7 off** | V5_ROW |

Paired under identical perception (filter on, each against gt-base): v1 **+1.1 pts [-4.4, +6.1], p 0.84, 13 / 11
flips**; v3 **+0.0 [-5.0, +5.0], p 1.0, 11 / 11, SPL -2.9 [-5.0, -0.9]**; v4 **+0.6 pts [-4.4, +5.6], p 1.00, 12 / 11 flips; SPL -2.5 pts [-4.6, -0.4]**.

Per-category SR (successes / n) on the same 180 episodes:

| category | n | baseline | gt-base | gt-fixes v1 | gt-fixes v4 |
|---|---|---|---|---|---|
| table | 37 | 81.1 | 97.3 | 97.3 | 97.3 |
| chair | 25 | 36.0 | 68.0 | 64.0 | 64.0 |
| picture | 21 | 61.9 | 71.4 | 76.2 | 76.2 |
| cushion | 20 | 85.0 | 90.0 | 95.0 | 95.0 |
| cabinet | 14 | 57.1 | 92.9 | 92.9 | 92.9 |
| seating | 11 | 90.9 | 90.9 | 81.8 | 90.9 |
| chest_of_drawers | 8 | 50.0 | 50.0 | 50.0 | 50.0 |
| plant | 7 | 71.4 | 71.4 | 85.7 | 85.7 |
| bed | 6 | 50.0 | 66.7 | 33.3 | 33.3 |
| towel | 6 | 33.3 | 50.0 | 66.7 | 50.0 |
| shower | 5 | 80.0 | 60.0 | 80.0 | 80.0 |
| sofa | 5 | 20.0 | 40.0 | 40.0 | 40.0 |
| counter | 4 | 25.0 | 75.0 | 50.0 | 50.0 |
| fireplace | 3 | 100 | 100 | 100 | 100.0 |
| stool | 3 | 66.7 | 100 | 100 | 100.0 |
| sink | 2 | 0 | 0 | 50.0 | 0.0 |
| clothes, gym_equipment, toilet | 1 each | 0 / 0 / 100 | 0 / 0 / 0 | 0 / 100 / 0 | 0 / 100 / 0 |

What the table says, in one paragraph: **the ground-truth sensor's phantoms were the largest single cause of
failure** (32 of the baseline's 67), and removing them is worth 14 points -- which is a statement about the
benchmark's "perfect perception", not about the method. The method's own fixes are worth a few points at most on this
set, each one visible in the episodes it was written for and invisible in SR. The oracle's numbers are worth about
three points over a flat prior.

---

## A. Measurement (entry 046) -- INFRA

### A1. Results were not comparable between runs

**Problem.** Entry 044's 24 re-runs of campaign failures succeeded in 10 of them with unmodified code. Every
per-episode claim about a fix was confounded with the model's run-to-run variation.

**Root cause.** Three sources: the 3B model's live answers vary even at temperature 0 and a fixed seed (Ollama's
prefix cache and batching, `LESSONS.md` 2026-10-31); a scene worker imports the repository at its start, so a run that
takes hours beside an edited tree mixes two versions of the method; and the renderer differs between machines.

**Before / after.** Before: campaigns on whatever episodes, live model, source tree as it was. After:
`mp3d/sets/single_storey_flat_v1.json` -- 180 episodes drawn (seed 17, uniform) from the 1,281 val episodes whose
start, nearest goal view point and shortest path stay on one storey in a house that is not split-level
(`single_storey_set.py`; x8F5xyUWy9e, X7HyMhZNoso, EU6Fwq7SyZv excluded by the user's instruction);
`run_single_storey.sh <name>` rsyncs the tracked tree into `~/GIT/TheAgency_snapshots/<name>/` and runs from there;
every LLM answer is recorded under a key that includes the episode (`LLM_ANSWERS_RECORD`), and later arms replay the
earlier stores (`REPLAY=`), so two arms differ only where a change makes the method ask something new.

**Measured.** A replay of the baseline's own code on 14 episodes: 14/14 bit-identical trajectories, 0 live calls. The
flat-p arm shared 131 of 180 trajectories with the baseline exactly.

### A2. A worker that lost its main thread stalled the campaign for an hour

**Problem.** Entry 044: Z6MFQCViBuw/000005's worker twice lost its main thread mid-episode and lingered with 32 live
CUDA threads; `campaign.py` waited on it. In the baseline run, 2azQ1b91cZZ/000118 stalled the same way for 30 min.

**Fix.** `hm3d/scene_pool.run_concurrently(stall_s=...)` watches each worker's `live.json` / `progress.json` /
`episodes.jsonl` / console log; a worker whose outputs do not change for `--stall-minutes` (30) is killed with its
session and recorded as exit -9, and the same `--output` resumes it. The resumed episode succeeded at step 124: the
stall was the harness's.

### A3. Two runs recorded different answers to one question into one store

**Problem.** The flat-p arm, launched into the shared store while the baseline's stalled scene was being resumed, had
five scenes refuse to start: "LLM answer shards disagree on key ...".

**Root cause.** A process loads the store once at its start; both runs held QUCTc6BB5sX at the same time; the second
asked a question live two minutes after the first had recorded it, and the warm server answered differently.

**Fix.** `llm_answer_cache --resolve-conflicts <dir> [--keep earliest|latest]` keeps one answer and moves the rest to a
sidecar; the rule since: a concurrent run records into its own directory and replays the others. The gt-base and
gt-fixes arms, run side by side, each had their own store; the v2 arm could replay only one of them for the same
reason. `LESSONS.md` 2026-11-10 (two entries).

### A4. The analysis tools

`single_storey_report.py` (paired SR/SPL, bootstrap intervals, McNemar, flips, per-category and per-scene) and
`single_storey_failures.py` (a mechanism tag per failure from the recording and the published goals; the GT sensor's
phantom ids counted apart; a near-miss STOP read against the goal's `.house` box as *beside*, *in the band* or
*far*; the labeller tally). Both standard-library.

---

## B. The target closing

### B1. An arrived standoff a step outside the limit spun the agent off the target (entry 044) -- FIX

**Problem.** zsNo4HB9uLZ/000078 (cushion among cushions on a sofa): the route to the standoff completed with the
cushion's near edge at 1.05-1.30 m; the completed route's idle result was executed as a TURN, three turns swung the
agent 90 degrees off the cushion, and the lock was lost as "no safe path to the target 1.1 m away".

**Root cause.** The standoff ring is drawn about the memory's centroid and the executor ends within its tolerance of
the ring point, so the near edge of a small object reads 1.05-1.30 m where the route completes -- the common case, not
the exception. The code treated an arrival outside the limit as "not there yet" and fell through to CLOSE, which had
nothing to do but execute the idle TURN.

**Before / after.** Before: arrival + out of range -> CLOSE -> idle TURN. After: an arrived standoff with the fresh
range at most one forward step beyond the limit enters the inspection, whose step rule takes the one MOVE_FORWARD that
puts the next frame in range (`arrived_step_out_inspections`). The flag is read after the route's own command, since
the action the route completes on is the one that sets it.

**Measured.** 000078: 211 -> 48 steps. In the 15-episode check the rule never fired elsewhere (the next rule changed
the trajectories before it was needed). In the baseline of the fixed set it is on, unmeasured separately.

### B2. The re-sighting was measured on the neighbouring instance (entry 044) -- FIX (PARTLY in origin)

**Problem.** The same episode: with several cushions in frame the range was measured on the neighbour and the centring
turns chased a sliver at the frame's edge.

**Root cause.** The re-sighting was the most confident box. Under GT perception every instance has confidence 1.0, so
`max` took whichever the detector listed first. Under a real detector the most confident box is also not necessarily
the one the lock is on.

**After.** `_best_projected`: the projection nearest the 3-D memory, a whole box before a sliver; before any memory,
confidence then area. **Flag:** the trigger was a GT artefact (confidence ties), but the rule -- the re-sighting is the
box nearest the memory -- is the right one under any detector. 14 reselections over the 15 episodes, each at the
exact action where the fixed trajectory parted from the baseline's; zsNo4HB9uLZ/000001 (a sliver cabinet) went from a
STOP 1.87 m off to a success.

### B3. The inspection resumed its approach without bound, and on side-clipped boxes (entry 044) -- FIX

**Problem.** INSPECT / CLOSE looping. **After:** the inspection resumes the approach at most
`max_inspection_resumptions` (3) times per lock, never on a box clipped at the frame's side (its near range is the
visible, farther part's). Counters `resumption_cap_hits`, `clipped_resumptions_withheld`.

### B4. A raised target was STOPped on from too far (entry 046) -- FIX, model corrected

**Problem.** 2azQ1b91cZZ/000025: a wall cabinet, underside 0.87 m up, STOPped with the memory 0.86 m away; DTG 0.22,
SR 0. TbHJrupSAjP/000156: a picture, DTG 0.14. Seven of the baseline's nine near-misses were raised targets.

**Root cause (as finally understood).** MP3D's view points are made by habitat's dataset generator
(`habitat/datasets/object_nav/create/object_nav_generator.py`, read in the forks that kept it): a grid of candidates
at the object's centre height, kept within 1 m of the object's oriented box *in that plane*, dropped vertically onto
the navmesh, and kept only where the object's pixels show with the camera turned to its centre at three pitches. For
a raised object the camera, 0.88 m up, must look up at it, and the far candidates fail the pixel test. Over the set's
1,020 goals: the farthest view point of an object on the floor is 1.02 m from its box (median), of one raised 0.3 m or
more 0.84 m (pictures 0.81, wall cabinets 0.72). The overshoot's "within 0.90 m planar" test was written for floor
objects.

**Before / after.** Before: after the STOP decision, one step, a second only while the memory's planar range exceeds
0.90 m, two at most. After: when the lock's lowest measured point is at least `elevation_min_m` (0.30) above the base,
the within-test reads `hypot(planar, lift)` and up to `stop_overshoot_elevated_max_steps` (3) steps are taken, so the
agent stands closer under a raised object. An object on the floor reads exactly as before.

**Why better, and a caveat.** The behaviour is right -- stand closer under raised things -- and the two episodes it
was written for succeed with no regression. But the first explanation written into the code ("the evaluator measures
1 m from the base in 3-D") was wrong; `hypot(planar, lift)` is a heuristic whose amount happens to grow with height,
not the evaluator's rule. The comment was corrected (`LESSONS.md` 2026-11-10); the heuristic stays because it is
bounded (`stop_overshoot_min_range_m` 0.45) and measured, and the mechanism it only half-addressed is B5.

**Measured.** Filter off: +2 / -0 (000025, 000156). Filter on: both are also among gt-fixes' gains over gt-base.

### B5. A thin wall-mounted target was STOPped on from beside it (entry 046) -- FIX, in two versions

**Problem.** 2azQ1b91cZZ/000157, QUCTc6BB5sX/000055, zsNo4HB9uLZ/000045, oLBMNvg9in8/000083 (pictures, a towel): the
agent ended 0.2-0.4 m beyond the object's width along its wall, 45-70 degrees off its normal, inside the 1 m band and
0.1-0.5 m from the nearest view point. 8 of the baseline's 9 near-misses stood *beside* their object; every picture
success of the same run stood within 0.28 m of the object's width.

**Root cause.** The same generator rule: a 2 cm picture shows no pixels from beside itself, so its view points lie
only in front -- within 0.26 m of its width along the wall (median over 78 pictures; 0.6-0.9 m for every other
category). The closing's standoff ring starts from the agent's own bearing (`target_path.TargetApproachPath`,
offsets 0, +-30, +-60, +-90, 180), so a picture first seen at a grazing angle bought a grazing standoff, and a fresh
in-range sighting on the way STOPped the agent wherever it stood ("fresh in-range sighting ... alignment not
required").

**Version 1 (run as gt-fixes v1) -- and why it was wrong.** The "front" was read off the occupancy map: for a target
raised 0.3 m or more with something occupied within 0.3 m of its memory, the direction from the memory to the centroid
of the free cells within 0.8 m. Three defects, all visible in the recordings of the v1 arm:

* **Biased toward the agent.** During an approach the free cells around the memory are the ones the agent has seen,
  i.e. on the agent's side: in zsNo4HB9uLZ/000045 the "front" came out 5 degrees from the agent's bearing and the
  standoff was the identical point (the run is step-for-step the baseline's).
* **Unstable.** The free centroid moves as the map fills, so every memory refinement re-selected a different standoff:
  pLe4wQe7qrG/000152 (seating) re-selected 31 times in one lock, oscillating between two points 0.9 m apart, and was
  released as "no safe path"; the baseline succeeded at action 233. Closing time spent occluded rose from 3,477 to
  4,427 actions over the arm.
* **No fallback.** When no frontal ring point had a route the lock was released instead of trying the old ring:
  2azQ1b91cZZ/000157, the picture the fix was written for, lost its lock at action 54 as "no safe path 1.2 m away".

The v1 arm still gained from it where the map happened to be known around the object (QUCTc6BB5sX/000055 towel,
QUCTc6BB5sX/000137 plant, TbHJrupSAjP/000156 picture via the pursuit's walk-back, 2azQ1b91cZZ/000025): closing-attributed
flips +4 / -1 (000152) against gt-base.

**Version 2 (the sighting's plane; run ungated as gt-fixes v3, gated as gt-fixes v4).** The normal is in the
sighting itself. `perception/face_normal.py`: the
world-frame points of every valid depth pixel on the object's mask (the box's central quarter without a mask) are
fitted as a plane (PCA of their covariance); when they are planar (standard deviation along the smallest axis under
5 cm), broad (over 5 cm along the second axis -- a one-pixel sliver fixes no plane) and standing (the normal's
vertical component under 0.35 -- a table top is a plane with nothing to stand in front of), the planar unit normal
toward the camera is recorded on the projection row (`face_normal_xy`). A chair, a sofa, a bed, a plant are not
planes and get none. The closing's normal is the mean of its sightings' normals, trusted while they agree (a poor
first fit -- a towel folded over its rail -- is outvoted; `face_normal_conflicts` counts the dissenters); the
pursuit's walk-back remembers it with the released target. Four behaviours follow:

1. **The ring is tried about the normal first**, with every bearing but the one into the wall (every 20 degrees),
   at the standoff radius and then at `frontal_inner_radii_m` (0.65, 0.45 m) -- a plane has nothing in front of it
   to collide with, and a picture in a 0.7 m corridor has its view points nearer than the ring -- and about the
   agent's bearing as before when no frontal point has a route (`_ring_bearings`, `frontal`). A goal chosen from the
   agent's ring before the lock had a normal is re-selected once when the normal arrives (`frontal_reselections`).
2. **A side-on in-range STOP waits.** When the agent stands more than `frontal_off_normal_deg` (35) off the normal and
   a frontal standoff with a route is not yet reached, an in-range frame does not enter the inspection; the route is
   walked instead, at most `frontal_max_deferrals` (12) actions per lock, after which the STOP rules apply as before
   (`frontal_deferrals`, `frontal_deferral_caps`). The angle is read from the memory, which lies on the near part of
   a grazed object (up to half its width toward the agent), hence a threshold tighter than the view points' own
   50-degree spread.
3. **No map reading.** The normal does not move with the map, so the standoff does not churn.
4. **The whole mask, not the near surface.** The first cut of v2 fitted the plane on the near-surface component the
   centroid is measured on, and 14 of the first 15 locks had a "normal" -- the front of a sofa's seat or a chair's
   backrest is a plane within the depth band -- at a cost of 50-170 actions on a sofa and a table. Fitted on the
   whole mask, the sofa, the table and the bench read as no plane; a chair seen squarely from behind still does
   (its backrest and rear legs are one plane), at a bounded cost of a few deferred actions.

**Why better.** The direction comes from the object's own surface, is the same under any detector that yields a box
or a mask with depth, and generalises past pictures: a cabinet's front, a shower screen, a towel on a rail are
standing planes too; a sofa is not and is left alone. Physically it is also what a robot should do -- stand where
the thing can be seen.

**What it cannot do.** It needs routable floor in front of the object. 2azQ1b91cZZ/000157's picture hangs in a
0.7 m diagonal corridor: the frontal rings at 0.85, 0.65 and 0.45 m land on or inside the facing wall's inflation,
no frontal point has a route, and the agent's ring is used -- the STOP is beside the picture as before (its view
points are a 0.3 m band the planner cannot target). QUCTc6BB5sX/000055's towel hangs over a vanity: the same.
Probed on six target episodes before the full arm: zsNo4HB9uLZ/000045 (picture) F -> S, the two above unchanged, a
chair +10 actions, a sofa +1.

**Gating (v4).** Run on every standing plane (v3), the rule fired on 90 of the 180 locks -- tables (their apron
seen from afar), chairs (a backrest from behind), cushions (a slab), floor cabinets -- and over those 90 it gained
8 and lost 3 episodes at a cost of 1,358 actions and 1.7 SPL points (the deferrals and the detours to a frontal
point of objects whose view points are all round). The mechanism is the raised plane's -- a picture, a wall cabinet,
a towel -- so the rule applies only when the lock's lowest point stands `frontal_min_elevation_m` (0.6) or more
above the base (`TargetClosing.frontal_normal`); a plane on the floor keeps the agent's ring.

**Measured.** v1: above. v3 (every plane): against gt-base +0.0 pts [-5.0, +5.0], 11 / 11 flips, SPL -2.9
[-5.0, -0.9]; closing-attributed flips +4 (2azQ1b91cZZ/000025, QUCTc6BB5sX/000137, TbHJrupSAjP/000156,
zsNo4HB9uLZ/000045 -- the picture the rule was written for) / -1 (zsNo4HB9uLZ/000023, a picture whose frontal
standoff lay beyond a barrier). v4 (raised planes): the rule touched 17 locks (12 pictures, 3 seats, a cabinet, a plant), with 14 deferred actions in all; over them +3 / -1 (gains 2azQ1b91cZZ/000025, TbHJrupSAjP/000156, zsNo4HB9uLZ/000045; loss zsNo4HB9uLZ/000023) at a SPL cost of 0.02 points -- the step cost of v3 is gone. Against gt-base the whole arm is +0.6 pts [-4.4, +5.6], 12 / 11 flips, SPL -2.5 [-4.6, -0.4]; the SPL loss is the labeller's (section E6-E7: -2.1 of the -4.4 SPL-points summed over 58 episodes whose first divergence is a transit choice).

### B6. The pursuit chased a glimpse for most of the episode (found in gt-fixes v4) -- FIX

**Problem.** Z6MFQCViBuw/000030 (chair): 446 of 500 actions "walking back to the released chair"; seven more
episodes of the same house spent 275-453 actions the same way. Eight of the ten failures tagged "wedged" in the
v4 arm, 20 % of all that remained.

**Root cause.** The house is built round an atrium. Chairs on the far side of the void are in the RGB frame with
no depth support (beyond the camera's 5 m, or across the drop). Such a box starts a *bearing approach* (walk toward
it until a frame has depth); when that is given up ("not moving", "lost"), `TargetPursuit.remember_bearing` keeps a
point `max_depth + 1 m` along the bearing as a target to walk back to -- a rule written for a television seen
from beyond 5 m down a corridor (00853/000065, locked with depth at action 487). Here every new glimpse of the same
chair made a *new* remembered point, each pursued for up to `max_walk_actions` (120), and the pursuit's bounds
(`max_attempts` 3, `max_age_actions` 300) are per target, not per episode: six glimpses, seven pursuits, 446
actions, `arrived` 0.

**Before / after.** `PursuitSettings.bearing_merge_m` (1.5): a bearing target placed within it of a remembered one
of the same label and floor is the same object -- its guess is updated, its attempts kept. `bearing_max_walk_actions`
(48): a guess along a line is walked for 48 actions, not 120. `max_total_walk_actions` (150): all pursuits of an
episode together walk at most this; spent, released targets are left to the search. Counters `merged`, `walked`,
`budget_spent` in the pursuit's stats.

**Why better.** The pursuit was built for a measured target behind a wall the map has since opened; a bearing is a
hypothesis, and a hypothesis that fails to verify in 48 actions should not be bought again at the next glimpse.
The bound on the whole episode's pursuit time keeps the search in charge of the budget. Under a real detector the
same loop arises wherever a target is seen across a drop or through a window.

**Measured.** V5_B6_MEASURED

---

## C. Navigation

### C1. A squeeze between furniture was treated as a doorway, and never mapped (entry 046) -- FIX

**Problem.** 2azQ1b91cZZ/000058 (table): six refused MOVE_FORWARDs from one spot at five headings in 20 actions, the
closing released a table in plain view 3 m away as "went nowhere", then locked and lost it 26 more times; 500 steps,
DTG 4.46. Three baseline failures were wedged this way; with the filter on (episodes run longer) twelve.

**Root cause.** When the simulator refuses a step, `navigation/scrape_guard.py` decides whether the refusal is a
*doorway scrape* -- the body brushing a jamb in a narrow channel -- in which case it is kept **off the map** so the
doorway does not get walled shut. The channel test was geometry-only: any narrow passage on the route counted. A
squeeze between a sofa and a wall was "a doorway"; the refusals that would have mapped the sofa were excused; the
planner kept routing through it; the closing's stall clock ran out.

**Before / after.** Before: a refusal in any narrow channel is excused. After (`ScrapeSettings.doorways_only`, True):
a channel must carry a confirmed door landmark (`channel.door_id`); otherwise the refusal is written as an obstacle at
once (a cell 0.25 m ahead), and a second within 0.5 m leaves a sticky bar A* routes around. Counter `no_door`.

**Measured.** Filter off: 000058 succeeds at 183, Z6MFQCViBuw/000090 at 333; no regression. Filter on, against
gt-base: +3 (Z6MFQCViBuw/000098 500 -> 119, QUCTc6BB5sX/000065, 2azQ1b91cZZ/000108) / -1 to -2 (Z6MFQCViBuw/000117
now wedged elsewhere after 25 written refusals changed its frontier choice; Z6MFQCViBuw/000183 uncertain between this
and B5). Stall releases over the arm 23 -> 15. Wedges remain the second-largest failure class (section G).

---

## D. Ground-truth perception

### D1. The semantic sensor paints instance ids where no object stands (entry 046) -- WORKAROUND for GT, by design

**Problem.** 32 of the baseline's 67 failures were STOPs on a lock whose 3-D memory stood more than 1.5 m from every
goal object of the class: 23 "slivers" under 10 cm of visible height (a chair 1.4 cm tall at 1 m), 9 whole boxes
(71 x 94 px chairs, curtains and vases painted across the wall of Z6MFQCViBuw's staircase, 3-7 m from the chairs).
136 of the baseline's 475 locks (29 %) were such phantoms. Verified visually in Z6MFQCViBuw/000035.

**Root cause.** MP3D's semantic mesh leaks instance ids through walls and stair risers; a rendered pixel carries the
id of a surface that is not at that pixel's depth. Nothing in the method can know this: the lock is on the sensor's
word.

**Before / after.** Before: every pixel of an instance id was that instance. After (`ground_truth/perception.py`,
`--ground-truth-on-object-m` 0.25, **on by default**): each instance's pixels are back-projected with the frame's depth
and tested against the instance's annotated box inflated by 0.25 m; fewer than 8 readings or under 50 % on the box ->
the instance is not reported; a partial fit -> the box and mask are trimmed to the on-box pixels. `GroundTruthSource`
publishes depth, pose and camera for it; `HabitatRGBDSimulator.semantic_boxes` the boxes.

**Flag.** Entirely a perception-layer change for a sensor that will be replaced; nothing in the navigation reads it
(the user's decision). A closing-level "sliver gate" written the same day for the same symptom was removed: it would
have put a GT-specific rule into the method.

**Measured.** gt-base against baseline: **+14.4 pts [+7.8, +21.1]**, p < 0.001, 33 / 7 flips; wrong STOPs within 1 m
48 -> 10; DTG 2.58 -> 1.79 m; phantom-tagged failures 32 -> 0. The 7 regressions are episodes that now run to the
step limit where before a phantom STOP happened to land near a goal, or whose route changed.

### D2. The sliver minimum-size rule (task step 4) -- not applied

Proposed: a box under about 10 px on its short side, or 10 cm of visible height and 5 cm of width, is no observation.
Measured on the baseline's locks it separates badly (rejects 38 of 339 true far sightings, admits 58 of 136
phantoms). Superseded by D1 on the user's instruction; not applied.

---

## E. The oracle and the room labeller

### E1. The oracle had no prior for fifteen MP3D categories (entry 045) -- PARTLY

**Problem.** "A table is in the bathroom", "a stool in the hallway", "office equipment in a garage" for gym equipment:
the 3B model's `home` step had no prior for the categories HM3D lacks.

**After.** `room_priors.TARGET_HOMES` / `IMPLAUSIBLE_ROOMS` / `HOME_OBJECTS` cover all 21 categories; the oracle's
prompt carries a `WHERE IT LIVES:` line (`target_home_line`) and STEP 1 is told to start from it.

**Flag.** The line is model-agnostic (the same briefing for any model), and a briefing about where things live is a
legitimate part of the contract; but a 14B model knows where a stool lives, and the line was written because the 3B
did not. Keep it, expect its effect to shrink. **Measured:** with the line the model follows the briefing for the
categories it had no prior for (counter -> kitchen in 7 of 7 calls; chest of drawers -> bedroom in 7 of 7); what it
still gets wrong is its own reading (a cushion -> "a chair is most likely in a dining room", twice in six). No SR
measurement.

### E2. Slivers of 0.2-0.8 m2 were offered as rooms (entry 045) -- FIX

**Problem.** Watershed cuttings at a doorway or along a wall were nodes: offered in 40 % of the oracle's calls on the
24 re-run episodes, the model's first choice in one in eight (EU6Fwq7SyZv/000118: 90 tiny nodes offered, 30 chosen).

**Root cause.** The fragment rule exempted a room with a door, and a home object of the target credited to a fragment
(a dining table standing across the cut) kept it a node.

**After.** `LoopSettings.tiny_room_m2` (1 m2): a room under it with no live frontier is a fragment, door or no door;
the `home_object_kept` exception does not apply to it. **Measured (12 episodes):** tiny nodes offered 168 -> 2, chosen
60 -> 1.

### E3. "No unexplored frontier left" was often false (entry 045) -- FIX

**Problem.** The HOUSE line told the model there was no frontier when the agent was merely boxed in; the phrase flipped
back in 167 of 266 campaign episodes that read it. The line was also built before the labels were refreshed, so the
first call read "rooms found: none yet" over a node list that said `type=bathroom`.

**After.** `house_context.frontier_state` tri-state (reachable / only slivers / unreachable from here / none), read
off the rooms' own frontier counts over the whole map, not the reachable inventory; `refresh_labels` before the line.
**Measured (15 episodes):** flips about 1.2 per episode -> about 0.2.

### E4. The oracle was called in bursts (entry 045) -- FIX (cost), PARTLY (motivation)

**Problem.** QUCTc6BB5sX/000069 asked 27 times in 230 actions inside a 5 m patch, on consecutive actions over the
same two nodes; 183 campaign episodes had six or more calls in 25 actions.

**Root cause.** The mid-walk reconsideration fired on a change of *any* room's object-class set -- under GT perception a
new class is credited to some room on nearly every action -- and in the SELECT state a renumbered room id was "a node
without a probability" and bought a call at once.

**After.** `discovery.semantic_signature` / `semantic_change`: a clue is a room on or off the offer, a room's type, or
a home-object kind new to the storey; geometry alone waits for the loop point or `reconsider_min_actions` (8); a
release for unreachable / blocked / timeout re-selects on the standing numbers instead of re-asking. **Measured (12
episodes):** 340 calls -> 89, bursts 83 -> 26; SR 5/12 -> 4/12 (noise). **Flag:** the bookkeeping was wrong whatever
the model; the urgency came from the 3B's call latency being the throughput limit.

### E5. Room labels swung on the same objects (entry 045) -- PARTLY

**Problem.** `dining_room -> hallway -> dining_room` on `chair, lamp` / `chair` / `chair, lamp`; 66 label changes over
24 episodes; each swing a clue and an oracle call.

**Root cause.** Two halves: the watershed moves a boundary and a room loses or regains one kind of object on most
actions (geometry, real); and the 3B labeller answers the same evidence differently on a count change or the periodic
refresh (the model).

**After.** `RoomLabelSettings.relabel_hysteresis`: a differing verdict without a new kind of evidence is held until it
repeats; a kind *lost* is held for `refresh_steps` (50) actions. **Flag:** the geometry half is a fix; the hold on
repeated identical evidence is a workaround for the model's inconsistency that a stronger model would not need, and
costs nothing when the model is consistent. **Measured (12 episodes):** churn 43 -> 21.

### E6. `table` flooded `dining_room` (entry 046) -- WORKAROUND (3B and the benchmark's label)

**Problem.** 225 `dining_room` adoptions over the baseline's 180 episodes, 23 on `table` alone, 22 on `chair + table`,
13 on `cabinet + chair + table`; 32 episodes ended with two or more dining rooms, and the oracle was briefed with them.

**Root cause.** The benchmark's `table` is any table -- a desk, a nightstand, a side table, a coffee table -- and the
labeller's prompt says "a table with chairs and no kitchen fittings is a dining room"; the 3B made a dining room of a
table alone, of a table with a picture, of a table with one chair.

**Before / after.** Before: the model's verdict stood. After (`RoomLabelSettings.dining_min_chairs`, 3): a
`dining_room` verdict with fewer than three chairs in the room's signature is read as `unknown` at the same
confidence; 0 disables.

**Flag.** A hard rule over the model's verdict, sized to this model's reading of this prompt. A 14B may not need it;
the user's later instruction (do not tune around the 3B) came after it was written. It is behind a flag and the
measurement says it is **not improving the search**: the briefing is cleaner, the room order reshuffles.

**Measured.** Filter off: adoptions 225 -> 122 (602 verdicts downgraded), episodes with two or more dining rooms
32 -> 16; filter on: 242 -> 102, 37 -> 18. Attributable flips (E6 and E7 together, every one at a *transit* node
choice): filter off +5 / -7 (3 gains and 3 losses were GT phantoms met or avoided on the new route; the method's own
+3 / -4); filter on (v4 arm) +5 / -6, and the 58 episodes whose first divergence is a transit choice walk 1,674 more
actions and lose 2.1 SPL-points between them -- the whole SPL loss of the arm. Alone, each rule trended the same
way on the first 52-53 episodes of its own arm (E6: 65.4 % against gt-base's 71.2 % on the same episodes, +1 / -4,
708 more actions; E7: 67.9 % against 73.6 %, +2 / -5, 485 more actions), and the user stopped both arms and the
labeller work: **both rules are off by default since 2026-11-11** (`dining_min_chairs` 0, `shape_rule` False),
kept behind their flags. The model change is the fix for what they addressed.

### E7. The labeller was asked about rooms with no objects (entry 046) -- PARTLY

**Problem.** 257 classifier calls over the baseline on rooms whose signature was empty (the "shape question"), answered
hallway 140, foyer 56, unknown 35, storage_closet 18, living_room 8 -- differently for the same prompt in different
runs.

**After.** `RoomLabelSettings.shape_rule` (True): a room with no classifiable objects that the geometry admits as a
passage is `hallway` by rule, weak (`shape_rule_confidence` 0.5, below the strong threshold), relabelled at once when
an object appears; no call.

**Flag.** Motivated by the 3B's randomness, but the question itself has no evidence in it under any model -- a rule
that costs no call and is as weak as the model's answer was is the honest reading. **Measured:** classifier calls
2,753 -> 2,291 (filter off), 3,177 -> 2,508 (filter on); verdicts on empty rooms 257 -> 0 and 322 -> 0; no SR effect
separable from E6.

### E8. What the oracle's numbers are worth (entry 046, the flat-p ablation)

`LoopSettings.flat_p` 0.15 (the median of the model's own numbers) for every node, everything else unchanged: SR
59.4 % against 62.8 %, dSR -3.3 [-6.7, +0.0], p 0.11, 131 of 180 trajectories identical, flips 2 / 8. The 3B oracle's
probabilities are worth about three points on this set over no oracle at all. Left as it is, by the user's decision:
the model is to be swapped, not tuned around.

---

## F. Things written and taken out again

* The closing-level sliver gate (D1): a GT rule in the method; removed the same day.
* The map-based "front" (B5 v1): biased, unstable, no fallback; replaced by the sighting's plane.
* The "3-D from the base" explanation of the view-point rule (B4): wrong; the behaviour kept, the comment corrected,
  the lesson recorded.

---

## G. What remains on a single storey (the method's failures; GT artefacts excluded)

From gt-fixes v4 (filter on, every fix on, the frontal rule gated), 180 episodes, 40 failures, all
the method's:

| mechanism | n | what it is |
|---|---|---|
| never seen in 500 actions | 15 | 7 never came within 6 m of a same-level goal (the room order never reached its room: 2azQ1b91cZZ/000128, /000142, /000188, /000196, QUCTc6BB5sX/000192, Z6MFQCViBuw/000120, /000197); 3 came within 4-6 m (2azQ1b91cZZ/000178, QUCTc6BB5sX/000054, TbHJrupSAjP/000162); 5 had the goal inside the camera's horizontal field within 4 m on 11-69 frames yet never in the detector's output -- behind a wall of the next room or outside the pitched camera's vertical field (oLBMNvg9in8/000199: a towel 0.75 m away, 71 degrees off the heading, then 5 frames of a border-clipped box that cannot start a takeover; zsNo4HB9uLZ/000022, 2azQ1b91cZZ/000173, TbHJrupSAjP/000060, /000163) |
| bearing-pursuit loop ("wedged") | 8 | all in Z6MFQCViBuw, the atrium house: chairs and chests on the far side of the void are seen in RGB without depth, the bearing approach is given up, the pursuit remembers a point 6 m along the bearing and walks back toward it for up to 120 actions, and every new glimpse adds a new point -- 275-453 of 500 actions "walking back to the released chair" (/000007, /000011, /000024, /000030, /000060, /000069, /000093, /000117). Fixed after this arm (B6); measured in gt-fixes v5 |
| wedged in clutter | 2 | Z6MFQCViBuw/000090 and /000183: 300+ actions in `target_close_occluded`, the lock released at the 160-action bound and re-taken; the floor between the agent and the target is drawn shut |
| locks released, step limit | 8 | 4 bearing approaches given up on a far target that was never measured (QUCTc6BB5sX/000035, /000122, TbHJrupSAjP/000138, Z6MFQCViBuw/000199); the escape loop (zsNo4HB9uLZ/000153, below); a lock released as "no safe path" in clutter (2azQ1b91cZZ/000109, QUCTc6BB5sX/000184); a sink locked at action 480 (TbHJrupSAjP/000190, DTG 0.01: out of actions) |
| near-miss STOP | 6 | 3 beside a planar target whose front has no routable floor (2azQ1b91cZZ/000157: a 0.7 m corridor; QUCTc6BB5sX/000055: a vanity under the towel; QUCTc6BB5sX/000163); 3 in the band, in front, off every view point (Z6MFQCViBuw/000121, oLBMNvg9in8/000083, zsNo4HB9uLZ/000188) |
| seen through a barrier | 1 | zsNo4HB9uLZ/000023: a picture seen across a half-wall; the frontal standoff lay beyond it |

Mechanisms behind them, from the recordings:

* **Search order (7 + 3).** The method's own residual on a single storey: the oracle's room order, with the 3B
  model's near-flat probabilities (E8), did not reach the target's room in 500 actions. This is the layer the user
  has reserved for the model change.
* **Coverage (5).** The target's room was entered but the camera never held the object: the policy walks a room's
  centreline pitched 30 degrees down and looks at frontiers, not along the walls; a towel at 0.75 m beside the
  path, a bed through a doorway the agent passed, a chest in the corner. A room sweep on entry (a turn with the
  camera level) is the obvious candidate and is not written here.
* **The bearing-pursuit loop (8).** B6, written after this arm.
* **Clutter (4).** A target seen, locked and unreachable on the map as drawn: the route to every standoff crosses
  cells the map holds occupied (furniture under or beside the target, a door drawn shut), the lock is released and
  re-taken. The scrape guard (C1) mapped these cells faithfully; what is missing is a way to approach an object
  whose floor the map denies -- a question for the map's inflation, not the closing.
* **Standing spot (6 + 1).** The frontal rule (B5) needs routable floor in front of the object; three pictures and
  towels have none within the ring's radii. Three more stand in front, within the band, and still miss every view
  point (furniture under a picture leaves one 0.1 m strip; a navmesh detour makes 0.21 m into 0.51 m).

Known single-episode mechanisms, documented and not fixed:

* **The map closes the door behind the agent -> escape loop.** zsNo4HB9uLZ/000153 (sink): inside a 2 x 2.5 m
  bathroom the doorway is drawn 0.2-0.3 m wide, the inflated map boxes the agent in, and the exploration fallback's
  `escape` spends 394 of 500 actions turning in a 1 x 1.3 m area. Deterministic across every arm (394 / 376 / 394
  actions). The only other episode with 100+ escape actions (oLBMNvg9in8/000129, 219) succeeded.
* **A real instance that is not a goal.** Z6MFQCViBuw/000098 (chair): a chair the dataset dropped (no view point
  reaches it), 3.4 m from every goal chair; approached, refused, released and walked back to for 500 actions. Any
  detector sees it; the method cannot know it does not count. (Succeeded in gt-fixes v1 by another route.)
* **In the band, off every view point.** Z6MFQCViBuw/000121 (picture): frontal, 0.69 m from the box, but the goal's
  view points are one 0.1 m strip (furniture under the picture). oLBMNvg9in8/000083 (picture): 0.21 m from a view
  point in a straight line, 0.51 m by the navmesh. QUCTc6BB5sX/000184 (counter): the first overshoot step blocked
  0.94 m from the box, whose view points end at 0.76.

---

## H. Summary by flag

| change | flag | measured effect (best evidence) |
|---|---|---|
| A1-A4 fixed set, frozen snapshot, replay store, watchdog, store repair, tools | INFRA | bit-identical re-runs; the study |
| B1 arrived-step-out inspection | FIX | 000078 211 -> 48 steps (15-episode check) |
| B2 re-sighting nearest the memory | FIX (GT-triggered) | 14 reselections = 14 partings; a sliver case converted |
| B3 resumption cap, no side-clipped resumption | FIX | no separate measurement |
| B4 elevation overshoot | FIX, model corrected | +2 / -0 |
| B5 frontal standoff v1 | FIX in aim, defective | +4 / -1, with churn and lost locks |
| B5 frontal standoff v3 (sighting's plane, every plane) | FIX, over-applied | +4 / -1 attributable; SPL -1.7 over the 90 locks it touched |
| B5 frontal standoff v4 (sighting's plane, raised planes only) | FIX | +3 / -1 over the 17 locks it touched; SPL -0.02 |
| B6 pursuit bounds (merged bearing guesses, 48-action bearing walks, 150 actions per episode) | FIX | V5_B6_SHORT |
| C1 doorway-only scrape guard | FIX | +2 / -0 (filter off), +3 / -1..2 (filter on); stall releases 23 -> 15 |
| D1 GT on-object filter | WORKAROUND (GT, perception layer) | +14.4 pts, p < 0.001 |
| D2 sliver minimum size | not applied | -- |
| E1 MP3D priors, WHERE IT LIVES | PARTLY (3B) | the model follows the briefing; no SR measurement |
| E2 tiny fragments not nodes | FIX | tiny offers 168 -> 2 |
| E3 frontier tri-state, HOUSE line order | FIX | false "no frontier" 1.2 -> 0.2 per episode |
| E4 reconsideration spacing, unasked releases | FIX / PARTLY | calls 340 -> 89 on 12 episodes |
| E5 label hysteresis | PARTLY (3B inconsistency + real geometry churn) | churn 43 -> 21 |
| E6 dining needs three chairs | WORKAROUND (3B, benchmark label); **off by default** | adoptions 225 -> 122; alone 65.4 % vs 71.2 % on 52 episodes, +1 / -4 |
| E7 empty room is a hallway by rule | PARTLY; **off by default** | calls -17 %; alone 67.9 % vs 73.6 % on 53 episodes, +2 / -5 |
| E8 flat-p ablation | finding | the oracle is worth +3.3 pts |

**Recommendations when the model changes to a 7B/14B:** E6 and E7 are already off (the user's decision of
2026-11-11, after the measurement above); re-run the fixed set with `relabel_hysteresis: false` to see whether E5
still earns its place; re-run flat-p to see what the larger oracle is worth. **When the detector changes to YOLO:** D1 is inert (the HM3D/MP3D envs
publish depth and boxes only under GT); B2, B4, B5 and C1 apply unchanged -- B5 needs depth on the detection's pixels,
which the projection already requires.
