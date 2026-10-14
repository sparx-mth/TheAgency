# 023 - The end-to-end review before the 5x3 benchmark, and the benchmark

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** review complete; the same-storey 5x3 campaign flown (`runs/zson-benchmark-5x3-20261005`: SR 0.467, SPL 0.272, DTG 3.61 m); 1891 regressions pass across the runtime, exploration, topology and objnav packages (1869 before), three of them for fixes applied after the run
**Roadmap item:** ObjectNav exploration efficiency (010) and room-search loop (011); follows 022

## The ask
Walk the whole method once more, end to end -- loop, prompt, the model's answers, the room
partition, the closing, the stairs, the harness -- fix only what is a bug or plainly not what the
method intends, and then fly five buildings x three episodes, a different target and start per
episode, and report SR, SPL and DTG.

## How the review was done
The decision chain (room-search loop, policy, scene graph, scans, labels, priors, openings, oracle,
solver instance, supervisor) was read directly, against the Ranchester couch recording. Five
read-only reviewers took the separable layers in parallel, each told to confirm against the
recordings: target closing / perception / camera / glances; stairs and the floor atlas; the
exploration fallback, sight ledger and frontiers; the Gibson harness and its metrics; the room
segmentation chain. Each reported findings with severity, evidence and a minimal fix; the fixes
were applied by hand with a regression each. The 14B reasoning model was probed on a synthetic
node set before and after the prompt changes.

## What was wrong, and what changed (all small)
**Room numbers churned** (R23 -> R26 -> R28 -> R52 for one room): a door snapped to a choke whose disk
did not sever the floor had no separating power, so the room behind it merged into the hallway on
the ticks the snap found a choke and split off under a new number on the ticks it did not; and the
registry retired a pid after one absent tick. Non-severing disks fall back to the plain barring
disk; the registry remembers vanished masks for ten updates. Replay of the recording: 36 pids for
10 rooms (26 deaths) -> 11 for 11 (3).

**The oracle's reading of the storey**: given "(this one at +3.1 m)" the 14B called an upper storey
a ground floor and kept the couch's living room "here"; given a sink glimpsed through a gap it gave
the gap 0.20 for a toilet ("a bathroom, no toilet"). The context line now names the storey's rank
in words, STEP 2 reads it, rule 1 covers an opening whose glimpse names the home type, the example
carries a toilet counter-example, and `parse_home_here` reads negations as `missing`. After the
change: toilet -> `missing`, sink gap 0.60 first; couch upstairs -> `elsewhere`, stairs 0.70.

**Labels**: one cabinet made a "kitchen" at 0.9 on an upper storey and the storey summary told the
oracle a kitchen was found. Generic objects alone classify nothing; weak labels read `kitchen?` in
the summary; "Living Room" is `living_room`, not `unknown`.

**Finished rooms**: a sticky `fragment`/`seen_through` verdict on a sliver could finish the room
that grew out of it under the same number; a room grown past 1.5x is re-judged.

**Target closing**: a fresh off-centre sighting at a pitch other than the predicted one LOOKed up
and down for the whole inspection budget (the converter tilts before it faces); an inspection
entered on a filtered estimate while every fresh frame measured the surface out of range released
a target in plain view as "saw nothing"; the landmark map was frozen during a takeover so the
documented map release could never fire; a glance in force resumed stale after a release; cue
glances turned toward boxes perception had placed on another storey.

**Fallback**: a blind-radius demotion of the goal in force made the fallback alternate between two
exits for ever (never invoked in the recent flights, confirmed in replay); the relocation target
flipped between the two ends of a hall; floor-wide frontier routes could walk down a seen flight.

**Stairs**: a storey first reached by a traversal took its height from poses on the eased last
treads (0.17 m high in Hanson), so no staircase seen up there became a portal; the fallback rule's
"searched out" verdict hid the stairs node from the oracle for 100 actions.

**Evaluator**: success was gated on the goal region's median height; Klickitat's chair samples lie
on two levels 0.6 m apart, and a STOP beside the lower chair would have scored 0 four centimetres
from the region. The gate is the nearest sample's height. The report writes its metrics before it
raises on a missing video.

**Generation**: `--start-storey same` puts the starts on the annotated storey, so a STOP at an
annotated instance scores (the cross-floor starts of 022 scored 0 by construction at real upstairs
instances); a building with fewer categories than episodes is ineligible.

Deferred, noted in the reviews: a sticky choke per door (the non-severing disk was the cause); the
door-pair annulus at the detected cell rather than the cut (unconfirmed); the terrain raster's
per-action cost in ground-truth mode; `success` not requiring STOP (the SemExp convention, documented).

## The benchmark
`runs/zson-benchmark-5x3-20261005`: Newfields (couch, potted plant, bed), Allensville (toilet, couch,
bed), Ranchester (potted plant, toilet, chair), Leonardo (tv, couch, toilet), Klickitat (couch,
potted plant, chair); seed 23; starts 4-12 m geodesic from the goal region on the annotated storey,
0.35 m clearance, 2 m apart; node oracle on `qwen2.5:14b-instruct`, room classifier on the 3B, both
on the CPU container; YOLO-World detector on the CPU; 500-action cap; success = inside the 1 m dilated
region of an annotated instance at the final position (STOP not required, SemExp); SPL = S * l /
max(l, p) with 3D path length; DTG = 3D navmesh geodesic from the final position to the nearest
region sample.

### Results (`runs/zson-benchmark-5x3-20261005/campaign/RESULTS.md`, `index.html` for the videos)

| Episodes | SR | SR 95% CI | SPL | SPL 95% CI | DTG (m) | SoftSPL | Mean actions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | **0.467** (7/15) | 0.25-0.70 | **0.272** | 0.11-0.46 | **3.61** | 0.332 | 258 |

| Episode | Target | SR | SPL | DTG m | Actions | What happened |
|---|---|---:|---:|---:|---:|---|
| Newfields/0 | couch | 1 | 0.775 | 0.00 | 90 | living room read as the home, found on the way |
| Newfields/1 | potted plant | 0 | 0 | 7.48 | 500 | walked down an UNSEEN staircase on a loop transit (action 183), searched the basement; back up only at 434 |
| Newfields/2 | bed | 0 | 0 | 5.46 | 2 | spawned at a kitchen island: counter front `bed 0.67/0.34` at 0.57 m, STOP at action 2 |
| Allensville/0 | toilet | 1 | 0.148 | 0.00 | 299 | found after a long peek sequence |
| Allensville/1 | couch | 1 | 0.824 | 0.00 | 132 | |
| Allensville/2 | bed | 0 | 0 | 2.18 | 115 | counter front `bed 0.70/0.62` from 0.55 m (six frames, its own landmark); the real bed in view 2 m behind; before that four unverified takeovers on one far spot (31/47/63/79) |
| Ranchester/0 | potted plant | 1 | 0.695 | 0.00 | 59 | |
| Ranchester/1 | toilet | 0 | 0 | 8.62 | 500 | a 12-room ground floor, 22 openings peeked one by one; five oracle calls in five actions at 111-115 |
| Ranchester/2 | chair | 1 | 0.631 | 0.00 | 62 | |
| Leonardo/0 | tv | 1 | 0.781 | 0.00 | 63 | |
| Leonardo/1 | couch | 0 | 0 | 4.53 | 319 | locked on the real sofa from 4.6 m; 160 actions of MOVE_FORWARD into an obstacle the map does not show; closing bound -> agent error |
| Leonardo/2 | toilet | 0 | 0 | 4.16 | 500 | wedged from ~150: collision slides of 5-10 cm reset the blocked clock; 350 actions of jitter |
| Klickitat/0 | couch | 0 | 0 | 14.52 | 500 | split-level: stepped down a half-level (z -0.34 -> -0.82), the floor band anchored to the storey saw neither floor nor obstacle; 361 idle turns |
| Klickitat/1 | potted plant | 0 | 0 | 7.23 | 500 | split-level again: no nodes, 319 relocation actions on the lower level |
| Klickitat/2 | chair | 1 | 0.225 | 0.00 | 228 | |

By category: chair 2/2, tv 1/1, couch 2/4, potted plant 1/3, toilet 1/3, bed 0/2. By building:
Allensville and Ranchester 2/3, Newfields, Leonardo and Klickitat 1/3.

Every success was a STOP at an annotated instance; every failure is one of five mechanisms, none of them
the reasoning: (1) a detector false positive at terminal range on a counter front (2 episodes -- the
STOP rule takes two frames of a target-class box at 0.5-0.6 m, and the map had nothing else there to
outvote it); (2) a wedge against geometry the map does not show (2 episodes: the closing's approach and
the loop's transit); (3) an unseen staircase walked down on a loop transit (1); (4) the single-height
floor band in a split-level house (2, both Klickitat); (5) budget on a large floor (2). The oracle's
verdicts were sensible throughout (`found` on the storeys with the home type, `elsewhere` on the
Ranchester couch's upper storey in the earlier campaign); the room partition held its numbers.

Wall time: 3 h 05 min for the 15 episodes (plus the aborted first launch); the 14B's ~40 s per call
dominates the long episodes (Leonardo/2: 58 min, 34 oracle calls).

## After the benchmark (not in it)
Three of the five mechanisms have small fixes, applied after the run with regressions (1891 pass):
- the closing releases a lock whose approach went nowhere -- twenty CLOSE actions inside a 20 cm
  circle -- with a rejection instead of pushing on to the 160-action bound (`approach_stall_actions`,
  `stall_releases`);
- the search's blocked clock is reset by a forward step's worth of displacement (60 % of the step),
  not by a collision slide, so the BLOCKED verdict fires in a wedge;
- an opening released BLOCKED, UNREACHABLE or TRANSIT_TIMEOUT is retired for the storey instead of
  being re-chosen by the supervisor's escape hatch (one oracle call per seven actions in the wedge).
Not fixed: the false near STOP (the detector's word at two frames; a VLM look at the candidate, or a
requirement that a STOP candidate was first seen from farther away, is the user's design call); the
unseen staircase under a loop transit (the stair-blocked planning world covers only SEEN stairs, and
only on fallback routes); the split-level floor band (a per-cell height band, a mapping change).

## The first launch, aborted after one episode
Newfields/000000 (couch): STOP at action 251 on brown leather bar seating read as `sofa 0.65-0.78`
from 0.94 m, the real couch in view 3.7 m behind it (DTG 3.66). Two things in that episode:
the storey line of the first version ("1 above the lowest (an upper floor)") had the 14B say
`elsewhere` -- Newfields' lowest storey is a basement, this storey IS the ground floor, and a
living room was found on it -- so the stairs up were valued 0.30 and the openings 0.10; and the
STOP itself is the detector's word taken at two frames. The campaign was aborted (three of the
five buildings have a sub-level), the line rephrased to the rank from both ends with no floor
name, STEP 2 told that the rooms decide and what each rank usually is, and the 14B probed on the
Newfields-like case (`found`, living room 0.60 first), the upstairs couch (`elsewhere`, stairs 0.70)
and a basement toilet (`missing`, stairs 0.55) before the relaunch. The aborted output is kept in
`campaign-aborted-storey-line/`.

## Open
- A STOP on two consecutive frames of a target-class box at terminal range is only as good as the
  detector: bar seating as a sofa (Newfields), a settee as a sofa (Hanson), a ride-on horse as a
  chair. Any stricter rule (more frames, the map's vote, a VLM look) is a design change for later.
- The 14B's answers are better than the 3B's but ~40 s each on this CPU; ~10 calls an episode.
