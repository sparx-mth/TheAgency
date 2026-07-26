"""Consumer-side rolling log of the drone's thoughts.

Holds the last N thoughts a consumer received, newest last, for rendering as the
drone's train of thought -- the BEV viewer draws it under the map. Bounded, so a
long flight cannot grow it without limit.

Consecutive repeats of the same line are COLLAPSED into one entry with a count
("Stopping to turn (x3)") rather than appended. A repeated line means one
ongoing state, and letting it push the rest of the reasoning off the top would
cost the operator exactly the context the log exists to give. Repeats that are
interleaved with other thoughts are genuine separate events and stay separate.

This module is deliberately ROS-free and Python 3.8 compatible so it can be shared
by the ROS1 adapter nodes (which import ``core`` under Python 3.8).
"""

from collections import deque
from typing import Deque, List, Optional

from sparx_agency.core.common.thought_message import Thought


class ThoughtEntry(object):
    """One line of the log: a thought and how many times it repeated in a row.

    Attributes:
        thought: The thought, refreshed to the most recent repeat's stamp.
        count: How many consecutive times this line was thought (1 = once).
    """

    __slots__ = ("thought", "count")

    def __init__(self, thought: Thought, count: int = 1) -> None:
        self.thought = thought
        self.count = count

    def display_text(self) -> str:
        """The line as rendered, with a repeat marker when it repeated."""
        if self.count > 1:
            return "%s (x%d)" % (self.thought.text, self.count)
        return self.thought.text

    def __repr__(self) -> str:
        return "ThoughtEntry(%r, count=%d)" % (self.thought.text, self.count)


class ThoughtLog(object):
    """A bounded, repeat-collapsing log of thoughts, oldest first.

    Args:
        capacity: How many entries to keep; older ones fall off the front.

    Example:
        >>> log = ThoughtLog(capacity=4)
        >>> from sparx_agency.core.common.thought_message import Thought
        >>> t = Thought(0.0, "Stopping to turn", "nav", "info", "follower")
        >>> log.add(t)
        True
        >>> log.add(t._replace(stamp=0.5))    # same line -> collapsed
        False
        >>> [e.display_text() for e in log.entries()]
        ['Stopping to turn (x2)']
    """

    def __init__(self, capacity: int = 12) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1, got %r" % (capacity,))
        self._capacity = capacity
        self._entries = deque(maxlen=capacity)  # type: Deque[ThoughtEntry]

    def add(self, thought: Thought) -> bool:
        """Append a thought, or collapse it into the newest entry if it repeats.

        Args:
            thought: The thought to record.

        Returns:
            True if it became a new entry, False if it was collapsed as a repeat
            of the newest entry.
        """
        if self._entries:
            newest = self._entries[-1]
            if (newest.thought.text == thought.text
                    and newest.thought.source == thought.source):
                newest.count += 1
                newest.thought = thought      # keep the latest stamp/level
                return False
        self._entries.append(ThoughtEntry(thought))
        return True

    def entries(self, limit: Optional[int] = None) -> List[ThoughtEntry]:
        """The log oldest-first, optionally only the newest ``limit`` entries.

        Args:
            limit: Return at most this many of the NEWEST entries; ``None``
                returns all of them.

        Returns:
            The entries, oldest first. The list is a copy, but the entries are
            live -- a later repeat mutates one already handed out.
        """
        items = list(self._entries)
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be >= 0, got %r" % (limit,))
            # Slice from the END, never `items[len(items) - limit:]`: that start
            # index goes NEGATIVE once limit exceeds the log's length, and Python
            # re-reads it from the end -- quietly returning the newest few instead
            # of everything. The viewer asks for exactly `capacity` lines, so it
            # sits in that case on every frame until the log first fills.
            items = items[-limit:] if limit else []
        return items

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
