"""When the caller is told to re-ask the policy, and when it is made to wait.

``PlanCommitExecutor._gate`` is a small piece of code carrying two obligations
that pull against each other, and both are multi-tick claims the single-guard
tests next door cannot express:

* **Nothing is lost to the rate floor.** A reason raised while the floor is
  suppressing it is *held*, not dropped. The case that matters is a reason which
  is true only while suppressed -- blown wide inside the period and back on the
  route before it ends -- because it exists on no later tick that could re-derive
  it. Dropping it strands the aircraft on a route it has already lost.
* **Nothing is told once and forgotten.** A held reason survives being returned
  and is cleared only by :meth:`~PlanCommitExecutor.mark_attempt`, because the
  caller that ticks is not always the caller that can act: ``fly_navdp`` ticks
  every physics step and asks the policy only on render steps.

Between them sits the floor itself, whose job is to stop a dead policy server
being hammered at the control rate.

These were found by adversarial review of the executor; the contracts they pin
are stated in ``_gate``'s own docstring but were previously exercised only one
tick at a time, which passes against an implementation that drops a suppressed
reason instead of holding it.
"""
import numpy as np

from sparx_agency.core.planning.vlas.common.plan_commit.executor import (
    OFF_ROUTE,
    CommitSpec,
    PlanCommitExecutor,
)


def straight_ahead(count=24, step=0.2):
    """A body-frame prediction running straight forward, NavDP's shape and span."""
    return np.stack([np.arange(1, count + 1) * step, np.zeros(count)], axis=1)


def standing_commitment(**overrides):
    """An executor with one commitment already anchored at the origin."""
    engine = PlanCommitExecutor(CommitSpec(**overrides))
    engine.mark_attempt(0.0)
    engine.commit(straight_ahead(), (0.0, 0.0, 0.0), 0.0)
    return engine


def test_off_route_raised_only_while_suppressed_is_delivered_once_the_floor_ends():
    """A reason true *only* inside the suppressed window still reaches the caller.

    The aircraft is blown 3 m wide inside the period and is back on the route
    before it ends, so on every tick after the floor elapses the deviation test
    passes and no reason can be re-derived. If the gate drops rather than holds,
    the excursion is never reported and the commitment keeps standing.
    """
    engine = standing_commitment(min_period_s=1.0, max_deviation_m=2.0)

    assert engine.tick(0.2, 3.0, 0.1).replan_reason is None, "inside the floor"
    assert engine.tick(0.2, 0.0, 0.2).replan_reason is None, "back on route, held"
    assert engine.tick(0.2, 0.0, 1.05).replan_reason == OFF_ROUTE


def test_a_held_reason_is_cleared_by_asking_the_policy_not_by_being_returned():
    """The reason persists across ticks until the policy is actually asked.

    ``fly_navdp`` ticks at the physics rate and can only act on a render step, so
    a reason cleared by the act of returning it would be delivered on a tick the
    caller can do nothing with, and never again.
    """
    engine = standing_commitment(min_period_s=0.05, max_deviation_m=2.0)

    assert engine.tick(0.2, 3.0, 1.0).replan_reason == OFF_ROUTE
    assert engine.tick(0.2, 0.0, 1.1).replan_reason == OFF_ROUTE, "still standing"
    assert engine.tick(0.2, 0.0, 1.2).replan_reason == OFF_ROUTE, "still standing"

    engine.mark_attempt(1.2)
    assert engine.tick(0.2, 0.0, 1.3).replan_reason is None


def test_asking_the_policy_does_not_silence_a_condition_that_still_holds():
    """Clearing the held reason must not latch the condition off.

    The guard against the previous test's contract: an executor that cleared on
    ``mark_attempt`` and then refused to raise again would leave an aircraft that
    is demonstrably still off its route flying it in silence.
    """
    engine = standing_commitment(min_period_s=0.05, max_deviation_m=2.0)

    assert engine.tick(0.2, 3.0, 1.0).replan_reason == OFF_ROUTE
    engine.mark_attempt(1.0)                       # asked, and nothing improved
    assert engine.tick(0.2, 3.0, 1.1).replan_reason == OFF_ROUTE


def test_a_policy_that_never_commits_is_asked_no_faster_than_the_rate_floor():
    """A dead server is asked once per ``min_period_s``, not once per control step.

    Ticked 200 times across ten simulated seconds against a one-second floor. The
    bound is the span over the period, plus one for the ask at t=0; an executor
    that only ever recorded its first attempt would tell the caller on nearly
    every tick instead.
    """
    engine = PlanCommitExecutor(CommitSpec(min_period_s=1.0))
    told, now = 0, 0.0
    for _ in range(200):                           # 200 x 0.05 s = a 10 s span
        if engine.tick(0.0, 0.0, now).replan_reason is not None:
            told += 1
            engine.mark_attempt(now)               # server is down: no commit
        now += 0.05

    assert engine.commitments == 0
    assert told <= 11, "10 s against a 1 s floor, plus the ask at t=0"
