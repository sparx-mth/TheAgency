"""Notice when the policy has decided the task is over, and say so.

**The failure this exists for, measured.** Over six supervised flights in the
SJTU hospital, STOP was 45-73% of every answer InternVLA-N1 gave. Those STOPs
were not spread out. They came in runs, and the runs were bimodal with nothing
in between::

    streak length:  1   4   6   34  40  49  51  66  66  69  82  406
                    ^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    recoverable     never recovered without help

Ninety-nine per cent of all STOP answers lived inside a run of five or more.
The worst single run was **406 consecutive STOPs over 50.1 minutes -- 56% of a
90-minute flight, during which coverage moved by 0.0 points.** The aircraft was
motionless, the camera saw the same frame for fifty minutes, and every fresh
instruction the supervisor published was ignored.

That gap in the histogram is the whole design. Between six and thirty-four
there is nothing: either the policy answers STOP a handful of times and carries
on, or it has stopped listening and will not start again by itself.

**Five, and it is not the midpoint of the gap.** Five is below the longest run
that recovered unaided, so it will sometimes interrupt a policy that was about
to carry on. That is the cheap mistake and it is chosen deliberately: a
needless restart costs one HTTP POST and one episode of accumulated history
that was about to be discarded anyway, while a missed latch costs what the
measurement above says it costs -- fifty minutes and half a flight. Nothing is
gained by waiting for seven to be sure, and a great deal is risked. The
threshold is the caller's if a flight ever shows otherwise.

**Why a restart is the answer, and not a better prompt.** The instruction is
sent with every step, so the policy was being told something new and answering
STOP anyway. What it accumulates instead is state: a frame history inside the
policy, a cached pixel goal, a step counter. Fifty minutes of identical frames
is a history that reads as "you have been here, doing nothing, forever", and
the model draws the obvious conclusion. The agent's own ``reset()`` clears all
of it -- history, pixel goal, both systems' input and output -- without
reloading the checkpoint, which is why the recovery is cheap enough to spend on
a suspicion.

This class only *counts*. It does not know what a reset is, or that there is a
server. That keeps it ROS-free, dependency-free and testable, and it is the
node's business to decide what to do when it is told.

Python 3.8, no numpy, no ROS.
"""
from __future__ import annotations


class StopLatch:
    """Counts consecutive STOPs, and says when the policy has stopped listening.

    Args:
        after: How many STOPs in a row before the policy is treated as latched.
            Five, from the measured gap between recoverable runs (up to six)
            and unrecoverable ones (thirty-four and up).

    Raises:
        ValueError: ``after`` is less than one, which would fire on the first
            STOP of every arrival and restart the agent constantly.
    """

    def __init__(self, after=5):
        # type: (int) -> None
        after = int(after)
        if after < 1:
            raise ValueError("after must be at least 1, got %d" % after)
        self.after = after
        self._run = 0
        self._latches = 0

    @property
    def run(self):
        # type: () -> int
        """How many STOPs in a row have been seen."""
        return self._run

    @property
    def latches(self):
        # type: () -> int
        """How many times the policy has been declared latched this flight."""
        return self._latches

    def record(self, stopped):
        # type: (bool) -> bool
        """Note one answer; True when the policy should be restarted.

        Args:
            stopped: Whether this answer was a STOP. Anything the aircraft can
                act on -- a turn, a step, a curve -- is False, and clears the
                run: a policy that has just moved is plainly still listening.

        Returns:
            True on the answer that reaches the threshold, and only that one.
            The run is cleared at the same time, so a policy that goes on
            refusing earns its next restart the same way it earned the first
            rather than one on every subsequent frame.
        """
        if not stopped:
            self._run = 0
            return False
        self._run += 1
        if self._run < self.after:
            return False
        self._run = 0
        self._latches += 1
        return True

    def clear(self):
        # type: () -> None
        """Forget the current run without counting a latch.

        For the caller that has just changed the aircraft's situation itself --
        backed it away from a wall, say. The next STOP after that is the first
        of a new run, because it is an answer to a genuinely different view.
        """
        self._run = 0
