# `plan_commit` — fly the prediction you have

A learned navigation policy is trained and evaluated **one frame at a time**:
show it an RGB-D frame and a goal, score the trajectory it returns, move to the
next frame. Nothing in that loop ever asks whether the trajectory was *flown*.

Deployed naively, the same policy is asked again every frame, and the aircraft
inherits a loop that offline evaluation never tested. At 3 Hz and 1 m/s it
covers 0.33 m before the plan it is flying is discarded and replaced. A 24-point
NavDP prediction spans about 4.8 m, so the aircraft executes roughly the first
**7 %** of every plan and nothing else. The route shape the policy predicted —
the part that goes round the shelf — never happens. What does happen is that
whatever small bias lives in the first segment is applied continuously, and that
is the one thing that compounds.

This package makes the aircraft keep a promise instead.

## The rule

1. Ask the policy once. Anchor its body-frame answer at the pose the frame was
   captured from → a world-frame route with a start, an end and an arc length.
2. **Commit to the first half of it.** 16 waypoints → commit through waypoint 8.
3. Fly that route with pure pursuit, so the carrot advances as the aircraft
   does and the *shape* gets flown, not just the first segment. Two details of
   that carrot are load-bearing and both were got wrong first:
   * The lookahead is measured **along the route**, not as a radius around the
     aircraft. On a U-turn tighter than the lookahead, a radius finds the
     return leg lying beside the aircraft and hands that back — bearing 168°,
     the whole turn skipped.
   * The heading is the route's **tangent**, not the bearing to the carrot. A
     chord cuts the corner: on a circular arc it subtends half the angle the
     tangent does, so an aircraft flown on it is permanently half a corner
     behind, and its camera frames somewhere it is not going — which then
     becomes the observation the next inference is made from. The tangent is
     also what the expert labels encode as NavDP's yaw channel
     (`to_navdp_label` takes `atan2` of each route step), so it is the heading
     the policy was trained against.
4. When the commitment is behind the aircraft, ask again — from where it now is,
   looking at what it can now see.

Half is not arbitrary. A learned trajectory is most trustworthy near the
observation it came from: the near end is metres the camera actually saw, the
far end is extrapolation, and by the time the aircraft would reach it the world
has had several seconds to change. Committing to the near half buys real
progress at the horizon where the prediction is worth the most, and re-asks
long before the far end would be reached.

**The carrot rides the whole prediction, not just the committed part**, so the
last `lookahead_m` of a commitment is steered by the speculative tail. That is
deliberate: a lookahead that stopped at the commit point would shrink to nothing
as the aircraft approached it and decelerate into every leg boundary. The tail
is aimed at, never promised, and the next inference replaces it well before the
aircraft could arrive there.

## The three escape hatches, and the rate floor

A commitment that cannot be broken is a trap, so three things can end one early:

| guard | what it catches |
|---|---|
| `min_commit_m` | a prediction that barely moves — a near-stop, not a route. Ask again rather than crawl. Never set it to `0`: a degenerate plan would be "flown" instantly and every tick would re-infer, which is the original bug. |
| `max_commit_s` | a commitment that is taking too long: yawing in place, blocked, fighting wind. A stale observation stops being worth flying. |
| `max_deviation_m` | a route the aircraft is no longer tracking. If it is metres off the plan, the plan is not what is being flown and should not be what decides when to re-plan. |

`min_period_s` is the fourth knob and it works the other way: it **suppresses** a
replan reason that arrives too soon after the last request, so it can only make
a commitment last longer, never end it sooner. It is the backstop that stops a
fast server reintroducing per-frame inference through one of the three above —
so lowering it to unstick a stuck aircraft is exactly the wrong move.

`arrive_radius_m` is the one knob that ends a commitment *successfully* rather
than early. Arc length alone misses the corner-cutting case: the aircraft passes
inside the commit point and its projected arc never quite reaches the end, so it
would fly on past a commitment it has finished. Being near the commit point is
half the test and **not sufficient on its own** — arrival also requires the
commitment to be all but flown in arc terms. Otherwise a long route whose commit
*point* happens to lie near the aircraft — a loop, or a corridor the policy
enters and reverses out of inside the committed half — is "arrived at" from a
standing start, having flown nothing, and every tick re-infers. That is the
original bug wearing its third hat, and proximity plus progress is what closes
it.

Progress only ever moves forward, twice over. The projection is refused any
segment behind a cursor the executor only advances, and the arc it returns is
also kept as a high-water mark. Nearest-point projection is not monotonic where
a route doubles back on itself: a hairpin whose return leg passes a few
centimetres from the outbound one reads as *already finished* from a standing
start, and a commitment that can finish before it is flown is the original bug
wearing a different hat.

## Using it

```python
from sparx_agency.core.planning.vlas.common.plan_commit import (
    CommitSpec, PlanCommitExecutor,
)

executor = PlanCommitExecutor(CommitSpec(fraction=0.5, lookahead_m=1.2))

# once per control step
tick = executor.tick(x, y, now_s)
if tick.replan_reason is not None:              # None means "keep flying"
    executor.mark_attempt(now_s)                # costs a period even if it fails
    result = policy.step(rgb, depth, goal)
    if result is not None:
        executor.commit(trajectory, pose_at_capture, now_s)
        tick = executor.tick(x, y, now_s)       # the new plan's carrot, not the old one's
if tick.target is not None:                     # None only before the first plan
    fly_at(tick.target)
```

Three details that are easy to get wrong:

* **Tick again after committing.** The tick that triggered the inference carries
  the *replaced* plan's carrot, and `None` on the very first commit. Steering at
  it for one step is a per-leg twitch that will be blamed on the policy.

* **Anchor with the pose the frame was captured at**, not the live pose. They
  differ by one inference latency, and anchoring with the live pose bakes that
  latency into the route as a translation — the plan is laid down ahead of where
  the policy actually looked.
* **`mark_attempt` is separate from `commit`** so that a dropped inference still
  costs a period. Otherwise a server that is down is hammered every tick, and
  the aircraft keeps flying a commitment it should have replaced.

## Who uses it

* `tasks/planning/vlas/navdp/finetune/world_goal/fly_navdp.py` — the closed-loop
  PEGASUS comparison, trained against untrained.
* `tasks/planning/falcon/adapter/scripts/navdp_click_node.py` — the real
  aircraft, so what flies in the simulator is what flies outdoors.

Nothing here knows which policy produced the trajectory, which is the point:
FlowNav and anything after it get the same behaviour for free.
