#!/usr/bin/env python3
"""
thought_logger_node.py -- record the nav stack's reasoning to a flight log.

Subscribes to the one shared narration topic every node publishes to
(``/nav/thinking``; see ``thinking.py``) and appends each thought to a file via
:class:`thought_journal.ThoughtJournal`. That turns "what was the planner
thinking when it did that?" from a question you had to be watching the BEV
window to answer into one you can answer after landing.

It is a SEPARATE node rather than a file handle inside the planner on purpose:

  * the narration topic already carries every node's reasoning -- planner,
    follower, mapping sync, recovery -- and reconstructing a flight needs all of
    them interleaved on one timeline, not the planner's half;
  * no narrating node has to know that anything is being recorded;
  * a journal that cannot open its file takes down the logger, not the flight.

Filters let you narrow the record without touching any other node, e.g. just the
planner's reasoning::

    <param name="sources"   value="astar_planner" />
    <param name="min_level" value="warn" />

  in   ~thinking_topic (std_msgs/String, JSON per core.common.thought_message)
       /nav/thinking            -- shared by every narrating node
  ~journal_path      (str, '')   file to append to; '' = ~/.ros/falcon/thoughts_<stamp>.log
  ~journal_max_bytes (int, 8MB)  stop writing past this size; 0 = unlimited
  ~categories        (str, '')   comma-separated allow-list; '' = every category
  ~sources           (str, '')   comma-separated node allow-list; '' = every node
  ~min_level         (str, info) info | warn | error
"""
import rospy
from std_msgs.msg import String

from sparx_agency.core.common.thought_message import LEVELS, parse_thought_message

from thinking import THINKING_TOPIC
from thought_journal import ThoughtJournal, default_journal_path


def _csv_set(raw):
    """Parse a comma-separated allow-list; empty means 'allow everything'."""
    return set(p.strip() for p in str(raw or "").split(",") if p.strip())


class ThoughtLoggerNode(object):
    """Persists the shared thought stream to a file, with optional filters."""

    def __init__(self):
        G = rospy.get_param
        self.topic = G("~thinking_topic", THINKING_TOPIC)
        self.categories = _csv_set(G("~categories", ""))
        self.sources = _csv_set(G("~sources", ""))

        level = str(G("~min_level", "info")).lower()
        if level not in LEVELS:
            raise ValueError(
                "~min_level %r is not one of %s -- a typo here would silently "
                "record nothing, which is worse than refusing to start"
                % (level, ", ".join(LEVELS)))
        self.min_level = LEVELS.index(level)

        path = str(G("~journal_path", "")).strip() or default_journal_path()
        self.journal = ThoughtJournal(
            path, max_bytes=int(G("~journal_max_bytes", 8 * 1024 * 1024)))
        rospy.on_shutdown(self._close)

        self._dropped = 0
        rospy.Subscriber(self.topic, String, self._cb, queue_size=100)
        rospy.loginfo("thought_logger: recording %s -> %s (min_level=%s%s%s)",
                      self.topic, self.journal.path, level,
                      ", categories=%s" % ",".join(sorted(self.categories))
                      if self.categories else "",
                      ", sources=%s" % ",".join(sorted(self.sources))
                      if self.sources else "")

    def _cb(self, msg):
        """Parse, filter and journal one narrated thought."""
        try:
            thought = parse_thought_message(msg.data,
                                            default_stamp=rospy.Time.now().to_sec())
        except ValueError as e:
            # A malformed thought is a bug in the narrating node, not a reason to
            # stop recording the rest of the flight.
            self._dropped += 1
            rospy.logwarn_throttle(
                10.0, "thought_logger: dropping unparseable thought (%d so far): %s",
                self._dropped, e)
            return
        if self._allowed(thought):
            self.journal.write(thought)

    def _allowed(self, thought):
        """True if ``thought`` passes the level / category / source filters."""
        if LEVELS.index(thought.level) < self.min_level:
            return False
        if self.categories and thought.category not in self.categories:
            return False
        return not self.sources or thought.source in self.sources

    def _close(self):
        """Footer + close on shutdown, and say where the log ended up."""
        self.journal.close()
        rospy.loginfo("thought_logger: wrote %d thoughts to %s",
                      self.journal.lines, self.journal.path)


def main():
    rospy.init_node("thought_logger")
    ThoughtLoggerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
