"""Producer-side gate that keeps narrated thoughts from spamming the log.

A node narrates from inside its control loop, where the same conclusion is
re-reached every tick: a follower running at 20 Hz would emit "Flying forward to
waypoint 3" twenty times a second and bury every other thought in the operator's
view. The gate turns that per-tick conclusion into an EDGE -- it passes a thought
only when the narration actually changes.

Thoughts are gated per ``key``, where a key names one narration SLOT: an
independent story the node is telling. A follower's motion slot ("aligning" ->
"flying" -> "reached") must not cancel its localization slot ("no localization"),
so they use different keys and are gated independently.

A slot may also re-narrate on a timer (``repeat_after_s``): an unchanged
"No localization" is worth restating every few seconds so the operator knows it
is still true and the log is not simply stalled. Left unset, a slot narrates
purely on change.

This module is deliberately ROS-free and Python 3.8 compatible so it can be shared
by the ROS1 adapter nodes (which import ``core`` under Python 3.8). The caller
passes its own clock in, so it works identically under a ROS clock, a simulated
clock, or a test's fake one.
"""

from typing import Dict, Optional, Tuple


class ThoughtGate:
    """Passes a narrated thought only when it is worth emitting.

    Args:
        repeat_after_s: Default seconds after which an UNCHANGED thought is
            re-narrated. ``None`` narrates purely on change.

    Example:
        >>> gate = ThoughtGate()
        >>> gate.should_emit("motion", "Stopping to turn", now=0.0)
        True
        >>> gate.should_emit("motion", "Stopping to turn", now=0.1)
        False
        >>> gate.should_emit("motion", "Flying forward", now=0.2)
        True
    """

    def __init__(self, repeat_after_s: Optional[float] = None) -> None:
        if repeat_after_s is not None and repeat_after_s <= 0.0:
            raise ValueError("repeat_after_s must be positive or None, got %r"
                             % (repeat_after_s,))
        self._repeat_after_s = repeat_after_s
        self._last = {}  # type: Dict[str, Tuple[str, float]]

    def should_emit(self, key: str, text: str, now: float,
                    repeat_after_s: Optional[float] = None) -> bool:
        """Report whether ``text`` should be narrated on slot ``key`` at ``now``.

        Records the decision, so a caller must act on a ``True`` -- asking twice
        for the same tick answers ``False`` the second time.

        Args:
            key: Names the narration slot; slots are gated independently.
            text: The narration line being considered.
            now: The caller's current time, in seconds.
            repeat_after_s: Overrides the instance default for this call.

        Returns:
            True when the text changed, when the slot is new, or when the slot's
            repeat interval has elapsed; False otherwise.
        """
        every = self._repeat_after_s if repeat_after_s is None else repeat_after_s
        if every is not None and every <= 0.0:
            # Held to the same contract as the constructor: a non-positive
            # interval would silently turn the gate off for this slot, which
            # reads as "the gate is broken" rather than "I passed a bad value".
            raise ValueError("repeat_after_s must be positive or None, got %r"
                             % (every,))
        prev = self._last.get(key)
        if prev is not None and prev[0] == text:
            if every is None or (now - prev[1]) < every:
                return False
        self._last[key] = (text, float(now))
        return True

    def reset(self, key: Optional[str] = None) -> None:
        """Forget a slot's history, so its next thought narrates again.

        Call this when a slot's story restarts (a new goal, a new mission leg):
        the drone re-narrating "Aligning to waypoint 1" for a *fresh* route is
        news, even though it repeats what it said for the last one.

        Args:
            key: The slot to forget; ``None`` forgets every slot.
        """
        if key is None:
            self._last.clear()
        else:
            self._last.pop(key, None)
