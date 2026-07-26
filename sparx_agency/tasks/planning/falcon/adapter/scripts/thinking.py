#!/usr/bin/env python3
"""
thinking.py -- narrate a node's decisions onto the shared /nav/thinking topic.

Helper module (imported by the adapter nodes, not run as a node). Gives every
node in the nav stack one line to say WHY it just did something::

    self.thinker = Thinker("waypoint_follower")
    ...
    self.thinker.say("Stopping to turn toward waypoint %d" % n)
    self.thinker.say("No localization -- holding", category="sensor",
                     level="warn", repeat_after_s=5.0)

The BEV viewer subscribes to the one topic and draws the thoughts as a rolling
log under the map, so an operator reads the drone's reasoning in one place
instead of correlating a dozen terminals of ``rospy.loginfo``.

WHY A NODE CAN NARRATE FROM ITS CONTROL LOOP
    ``say`` is EDGE-TRIGGERED by construction: a :class:`ThoughtGate` drops a
    line whose text has not changed since the last call on the same ``key``, so
    a follower running at 5 Hz that calls ``say("Flying forward to waypoint 3")``
    every tick emits exactly once -- and emits again the moment the text becomes
    "waypoint 4". Callers need no ``_prev_state`` bookkeeping of their own.

    ``key`` names a narration SLOT -- one story the node is telling -- and slots
    are gated independently, so a node's motion narration ("aligning" -> "flying"
    -> "reached") never suppresses its sensor narration ("no localization"). It
    defaults to ``category``, which is the right slot for most nodes; pass an
    explicit key only for two independent stories within one category.

    A state that persists is worth restating occasionally so the operator knows
    it is still true and the log has not simply stalled: pass ``repeat_after_s``.

    Narrate DECISIONS, not telemetry. A line that changes every tick (a distance,
    a counter) defeats the gate and buries the reasoning -- that belongs in the
    viewer's HUD, not the log.

TRANSPORT
    Deliberately NOT latched. The log is a chronology: a late-joining viewer
    replaying a latched backlog would show old thoughts as if they were being
    thought now. A viewer that starts late has simply missed them.

PERSISTENCE
    The topic is live-only: the viewer draws it in a rolling window and it is
    gone. With ``/thinking/log_path`` set, each node ALSO appends its thoughts
    to that file, so "what was the planner thinking when it did that?" survives
    the flight. Written here, in-process, rather than by a node subscribing to
    the topic: that would cost a whole extra Python process on the Orin to do
    what one ``write()`` does in the node that already formatted the line. Every
    node appends to the same file; see ``thought_journal.py`` for why that is
    safe. The path is a GLOBAL param, not ``~``, so one setting covers all
    twelve narrating nodes and they agree on one file per run.

  out  ~thinking_topic (std_msgs/String, JSON per core.common.thought_message)
       /nav/thinking            -- shared by every narrating node
  ~thinking       (bool, true)  -- false silences this node's narration
  ~thinking_echo  (bool, true)  -- also mirror each thought to rosout
  /thinking/log_path (str, '')  -- append every thought here; '' = no file
"""
import rospy
from std_msgs.msg import String

from sparx_agency.core.common.thought_gate import ThoughtGate
from sparx_agency.core.common.thought_message import Thought, encode_thought

from thought_journal import ThoughtJournal

#: Global (not ``~``) param naming the shared flight log; '' disables the file.
LOG_PATH_PARAM = "/thinking/log_path"

#: The one topic every narrating node publishes to and the BEV viewer reads.
THINKING_TOPIC = "/nav/thinking"


class Thinker(object):
    """Narrates one node's reasoning onto the shared thinking topic.

    Args:
        source: The narrating node's name, shown against each line.
        topic: Topic to publish on; defaults to the ``~thinking_topic`` rosparam,
            itself defaulting to :data:`THINKING_TOPIC`.
        repeat_after_s: Default seconds after which an unchanged thought is
            re-narrated. ``None`` narrates purely on change.
        queue_size: Publisher queue depth. Sized for a burst of transitions
            (several nodes reacting to one replan) without dropping lines.
    """

    def __init__(self, source, topic=None, repeat_after_s=None, queue_size=20):
        self.source = str(source)
        self.topic = topic or rospy.get_param("~thinking_topic", THINKING_TOPIC)
        # An empty topic means "off", matching the viewer's ~thinking_topic:=''.
        # Without this it would reach rospy.Publisher("") and take the node down on
        # an obscure ROS name error, for what reads like a request to stay quiet.
        self.enabled = bool(rospy.get_param("~thinking", True)) and bool(self.topic)
        self._echo = bool(rospy.get_param("~thinking_echo", True))
        self._gate = ThoughtGate(repeat_after_s=repeat_after_s)
        # Not latched -- see the module docstring.
        self._pub = (rospy.Publisher(self.topic, String, queue_size=queue_size)
                     if self.enabled else None)
        # Shared flight log. A journal that cannot be opened must not take a
        # flight node down over a diagnostic, so fall back to narrating live and
        # say why -- loudly enough that "the log is empty" is never a mystery.
        self._journal = None
        log_path = str(rospy.get_param(LOG_PATH_PARAM, "") or "").strip()
        if log_path:
            try:
                self._journal = ThoughtJournal(log_path)
            except (IOError, OSError) as e:
                rospy.logerr("%s: cannot open the thought log %s (%s); "
                             "narrating live only", self.source, log_path, e)
        if self.enabled:
            rospy.loginfo("%s: thinking out loud on %s%s", self.source, self.topic,
                          " (+ log %s)" % log_path if self._journal else "")

    def say(self, text, category="nav", level="info", key=None,
            repeat_after_s=None):
        """Narrate ``text``, unless it repeats what this slot last said.

        Safe to call every control tick; see the module docstring.

        Args:
            text: The human-readable, first-person narration line.
            category: Narrating subsystem; one of
                ``core.common.thought_message.CATEGORIES``.
            level: Severity; one of ``core.common.thought_message.LEVELS``.
            key: Narration slot; defaults to ``category``.
            repeat_after_s: Overrides the instance default for this call.

        Returns:
            True if the thought was published, False if the gate dropped it as
            an unchanged repeat or narration is disabled.

        Raises:
            ValueError: If ``text`` is empty or ``category``/``level`` is
                unknown -- a mislabelled thought is a bug, not something to
                quietly mis-colour in the operator's log.
        """
        if self._pub is None and self._journal is None:
            return False
        now = rospy.Time.now().to_sec()
        # Encode (and so validate) BEFORE consulting the gate. should_emit records
        # the text it was asked about, so gating first would let a mislabelled
        # thought raise on its first tick and then be swallowed as a "repeat" on
        # every tick after -- the loudest bug in the file would fire once and go
        # quiet. Validating first makes it raise every time, until it is fixed.
        payload = encode_thought(text, now, category=category, level=level,
                                 source=self.source)
        if not self._gate.should_emit(key or category, text, now, repeat_after_s):
            return False
        if self._journal is not None:
            self._journal.write(Thought(stamp=now, text=text.strip(),
                                        category=category, level=level,
                                        source=self.source))
        if self._pub is not None:
            self._pub.publish(String(data=payload))
        if self._echo:
            log = rospy.logwarn if level in ("warn", "error") else rospy.loginfo
            log("%s: %s", self.source, text)
        return True

    def forget(self, key=None):
        """Let a slot narrate its next thought even if it repeats the last one.

        Call this when a slot's story restarts -- a new goal, a new route, a new
        mission leg. "Aligning to waypoint 1" for a FRESH route is news, even
        though it repeats what was said for the previous one.

        Args:
            key: The slot to forget; ``None`` forgets every slot.
        """
        self._gate.reset(key)
