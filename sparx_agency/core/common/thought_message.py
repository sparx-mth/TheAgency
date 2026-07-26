"""Codec for the drone's "thinking" message.

Every node in the nav stack narrates its own decisions -- why it is turning, what
it is planning around, which object it is homing on, why it gave up -- as a plain
``std_msgs/String`` carrying JSON on ONE shared topic (``/nav/thinking``). The BEV
viewer collects them into a chronological log drawn under the map, so an operator
reads the drone's reasoning live instead of correlating a dozen terminals of
``rospy.loginfo``::

    {"stamp": 1780843795.329, "text": "Aligning to waypoint 3 (x=1.0, y=2.0)",
     "category": "nav", "level": "info", "source": "waypoint_follower"}

``stamp`` is when the thought was FORMED (the publisher's clock), not when it was
delivered, so the viewer can order a burst from several nodes correctly.
``category`` says which subsystem is speaking -- it colours the line and lets a
consumer filter one subsystem; ``level`` marks a thought that reports trouble
(``warn``) or a hard stop (``error``). ``source`` is the narrating node.

One shared topic rather than one per node keeps the ordering global and the
viewer's subscription trivial: a newly narrating node needs no viewer change.

Encoding is strict -- an unknown category/level raises rather than silently
reaching the operator mislabelled. Parsing is strict for the same reason; a
consumer that must survive a malformed publisher catches ``ValueError`` and
counts the drop, rather than the codec guessing what was meant.

This module is deliberately ROS-free and Python 3.8 compatible so it can be shared
by the ROS1 adapter nodes (which import ``core`` under Python 3.8) and the ROS2
sidecar alike.
"""

import json
from typing import NamedTuple, Optional

#: Which subsystem a thought comes from. Consumers colour/filter by this, so the
#: set is closed: a typo must fail loudly at the publisher, not paint an
#: uncoloured line in the operator's log.
CATEGORIES = ("nav", "plan", "object", "sensor", "map", "mission")

#: How much the operator should care. ``info`` narrates normal progress, ``warn``
#: reports trouble the drone is handling (lost object, replanning, stale sensor),
#: ``error`` reports a decision it could not resolve (no route, giving up).
LEVELS = ("info", "warn", "error")


class Thought(NamedTuple):
    """One narrated decision from a node in the nav stack.

    Attributes:
        stamp: When the thought was formed, floating-point seconds.
        text: The human-readable, first-person narration line.
        category: The narrating subsystem; one of :data:`CATEGORIES`.
        level: Severity; one of :data:`LEVELS`.
        source: Name of the node that formed the thought.
    """

    stamp: float
    text: str
    category: str
    level: str
    source: str


def encode_thought(text: str, stamp: float, category: str = "nav",
                   level: str = "info", source: str = "") -> str:
    """Serialize one thought to the JSON wire format.

    Args:
        text: The human-readable narration line.
        stamp: When the thought was formed, in seconds.
        category: The narrating subsystem; one of :data:`CATEGORIES`.
        level: Severity; one of :data:`LEVELS`.
        source: Name of the narrating node.

    Returns:
        The JSON payload for a ``std_msgs/String``.

    Raises:
        ValueError: If ``text`` is empty, or ``category``/``level`` is not a
            known value.
    """
    clean = (text or "").strip()
    if not clean:
        raise ValueError("thought text is empty")
    if category not in CATEGORIES:
        raise ValueError("unknown thought category %r (expected one of %s)"
                         % (category, ", ".join(CATEGORIES)))
    if level not in LEVELS:
        raise ValueError("unknown thought level %r (expected one of %s)"
                         % (level, ", ".join(LEVELS)))
    return json.dumps({"stamp": float(stamp), "text": clean,
                       "category": category, "level": level,
                       "source": str(source)})


def parse_thought_message(data: str,
                          default_stamp: Optional[float] = None) -> Thought:
    """Parse a thought JSON message.

    Args:
        data: The raw string payload of the message.
        default_stamp: Stamp to assume when the payload omits ``stamp`` (e.g. the
            consumer's receive time).

    Returns:
        The parsed thought.

    Raises:
        ValueError: If the payload is not a JSON object, carries an empty
            ``text``, has an unknown ``category``/``level``, or omits ``stamp``
            with no ``default_stamp`` supplied.
    """
    try:
        payload = json.loads(data)
    except ValueError as e:
        raise ValueError("thought message is not valid JSON: %s" % e)
    if not isinstance(payload, dict):
        raise ValueError("thought message must be a JSON object, got %r"
                         % type(payload).__name__)

    # Not str(): a JSON null/number/object would coerce to a real-looking line
    # ("None", "0", "{'a': 1}") and reach the operator as if the drone thought it.
    raw_text = payload.get("text", "")
    if not isinstance(raw_text, str):
        raise ValueError("thought message 'text' must be a string, got %r"
                         % type(raw_text).__name__)
    text = raw_text.strip()
    if not text:
        raise ValueError("thought message is missing 'text'")

    if "stamp" in payload:
        try:
            stamp = float(payload["stamp"])
        except (TypeError, ValueError):
            raise ValueError("thought message field 'stamp' is malformed: %r"
                             % (payload["stamp"],))
    elif default_stamp is None:
        raise ValueError("thought message is missing 'stamp' and no default "
                         "was given")
    else:
        stamp = float(default_stamp)

    category = str(payload.get("category", "nav"))
    if category not in CATEGORIES:
        raise ValueError("unknown thought category %r (expected one of %s)"
                         % (category, ", ".join(CATEGORIES)))
    level = str(payload.get("level", "info"))
    if level not in LEVELS:
        raise ValueError("unknown thought level %r (expected one of %s)"
                         % (level, ", ".join(LEVELS)))

    source = payload.get("source", "")
    if not isinstance(source, str):
        raise ValueError("thought message 'source' must be a string, got %r"
                         % type(source).__name__)
    return Thought(stamp=stamp, text=text, category=category, level=level,
                   source=source)
