#!/usr/bin/env python3
"""
thought_journal.py -- persist the nav stack's narrated thoughts to a flight log.

Helper module (imported by ``thought_logger_node``, not run as a node). Every
node in the stack already narrates WHY it did something onto ``/nav/thinking``
(see ``thinking.py``), but that stream is live-only: the BEV viewer draws it in a
rolling window and it is gone. After a flight, the one question worth asking is
"what was the planner thinking when it did that?", and by then the window has
scrolled and the terminal is closed.

This writes the same stream to a file, one line per thought::

    14:32:07.412  WARN   astar_planner[plan]   No route at the preferred 0.40m
                                               standoff, so I am flying a 0.25m
                                               squeeze

Design notes:

  * **Flushed per line.** A flight log that is lost when the node is killed (or
    the battery goes) records exactly the flights you most want to read back, so
    durability beats throughput here. The stream is edge-triggered upstream, so
    the write rate is a handful of lines per second at worst.
  * **Size-capped.** On an embedded target an unbounded log is a way to fill the
    root filesystem mid-mission. Past the cap the journal stops writing and says
    so, rather than truncating silently or taking the disk down.
  * **ROS-free**, so it is testable without a ROS environment and reusable by any
    consumer of the thought stream.

Python 3.8 compatible: the FALCON ROS1/Noetic adapter runs these scripts under
3.8 (see ``tasks/planning/falcon/run_falcon.sh``).
"""
import os
import time

#: Written once at the top of a new journal so absolute time is recoverable.
_HEADER = "# falcon thought journal -- opened %s\n"
#: Written on the first thought, so the relative ROS column has a stated origin.
_ORIGIN = "# ros t0 = %.6f (the +s column is relative to this)\n"


#: Env var naming the log directory. run_falcon.sh sets it to a HOST directory
#: bind-mounted into the container -- without that, a journal written to the
#: container's own filesystem dies with the (--rm) container, which is precisely
#: the flight you wanted to read back.
LOG_DIR_ENV = "FALCON_LOG_DIR"


def default_journal_path(root=None, now=None):
    """Build a timestamped journal path under ``root``.

    Args:
        root: Directory to write into. Defaults to ``$FALCON_LOG_DIR`` and then
            to ``~/.ros/falcon``; both are derived at runtime rather than
            hardcoded, so the same code works on the host, inside the FALCON
            container and on the Jetson.
        now: ``time.struct_time`` to stamp the filename with (defaults to now).

    Returns:
        Absolute path of the form ``<root>/thoughts_YYYYmmdd_HHMMSS.log``.
    """
    base = (root or os.environ.get(LOG_DIR_ENV)
            or os.path.join(os.path.expanduser("~"), ".ros", "falcon"))
    stamp = time.strftime("%Y%m%d_%H%M%S", now or time.localtime())
    return os.path.join(base, "thoughts_%s.log" % stamp)


class ThoughtJournal(object):
    """Append-only, size-capped, line-flushed sink for narrated thoughts.

    Args:
        path: File to append to. Parent directories are created.
        max_bytes: Stop writing past this size (0 = unlimited). The cap is
            checked before each write, so the file may exceed it by one line.
        wall_clock: Callable returning epoch seconds, for the human-readable
            timestamp. Injected for tests.

    Raises:
        IOError / OSError: If the path cannot be opened. Journalling is opt-in,
            so a caller that asked for it deserves to hear that it failed rather
            than fly on believing a log is being written.
    """

    def __init__(self, path, max_bytes=8 * 1024 * 1024, wall_clock=None):
        self.path = str(path)
        self.max_bytes = int(max_bytes)
        self._wall_clock = wall_clock or time.time
        self._written = 0
        self._lines = 0
        self._capped = False
        self._t0 = None          # ROS stamp of the first thought (relative origin)
        parent = os.path.dirname(self.path)
        if parent:
            try:
                os.makedirs(parent)
            except OSError:
                if not os.path.isdir(parent):
                    raise
        self._fh = open(self.path, "a")
        self._emit(_HEADER % time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime(self._wall_clock())))

    @property
    def lines(self):
        """Thoughts written so far (excludes the header and the cap notice)."""
        return self._lines

    @property
    def capped(self):
        """True once ``max_bytes`` was reached and writing stopped."""
        return self._capped

    def write(self, thought):
        """Append one thought. Returns True if it was written.

        Args:
            thought: A ``core.common.thought_message.Thought`` (any object with
                ``stamp``, ``text``, ``category``, ``level``, ``source``).
        """
        if self._fh is None or self._capped:
            return False
        if self.max_bytes > 0 and self._written >= self.max_bytes:
            self._capped = True
            self._emit("# journal capped at %d bytes after %d thoughts; "
                       "raise ~journal_max_bytes to keep recording\n"
                       % (self.max_bytes, self._lines))
            return False
        if self._t0 is None:
            self._t0 = float(thought.stamp)
            self._emit(_ORIGIN % self._t0)
        self._emit(self.format(thought, self._wall_clock(),
                               float(thought.stamp) - self._t0))
        self._lines += 1
        return True

    @staticmethod
    def format(thought, wall_s, rel_s):
        """Render one thought as a log line.

        Two clocks, because they answer different questions. The wall clock
        leads: that is what an operator correlates against a video, a note or
        another process's log. The relative ROS stamp follows: that is what
        lines the thought up with a bag, and reading down the column shows the
        CADENCE of the stack's decisions -- which is usually the tell when a
        planner is thrashing.
        """
        clock = time.strftime("%H:%M:%S", time.localtime(wall_s))
        millis = int((wall_s - int(wall_s)) * 1000.0)
        return "%s.%03d  %+8.2f  %-5s  %s[%s]  %s\n" % (
            clock, millis, rel_s, str(thought.level).upper(), thought.source,
            thought.category, thought.text)

    def close(self):
        """Write a footer and close. Safe to call more than once."""
        if self._fh is None:
            return
        self._emit("# closed after %d thoughts\n" % self._lines)
        try:
            self._fh.close()
        finally:
            self._fh = None

    def _emit(self, line):
        """Write and flush one raw line, tracking the size against the cap."""
        self._fh.write(line)
        self._fh.flush()
        self._written += len(line)
