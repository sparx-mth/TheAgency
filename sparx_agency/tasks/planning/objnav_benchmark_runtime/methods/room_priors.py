"""Where a target category cannot be: the room types a search need not enter.

The node oracle is asked the same question in prose and answers in
percent, but a model that gives a bedroom 3% for a sofa still leaves the
bedroom in the order, and after the living rooms are spent the order gets
to it. This table is the hard version of the rule the prompt states: a
room whose identified type is listed here for the target is not a node at
all -- not valued, not ordered, not entered -- whatever its frontier count.
Only types the classifier can emit are listed; ``unknown`` is never excluded.

The lists are exclusions, not permissions, so a type missing from a list
stays searchable: a sofa in a hallway is rare, but the hallway leads on.

Two guards keep the table from ruling out a room the partition got wrong
(since 2026-10-05). A **weak** label -- one kind of object -- never
excludes: it is a guess the oracle sees as ``type=living_room?`` and
values itself. And a room holding a **home object** of the target
(:data:`HOME_OBJECTS`: a sink or a shower where a toilet lives) is never
excluded whatever its label: the Hanson recording of 2026-10-04 merged a
bathroom's sink into a "living room" of sofas and ruled the room out for
the toilet standing a metre from the sink.
"""
from __future__ import annotations

#: Prompt aliases collapse to one key, as in the room labeller.
_ALIASES = {"couch": "sofa", "tv": "television", "tv monitor": "television", "plant": "potted plant"}

#: Target -> room types it is not worth searching for it.
IMPLAUSIBLE_ROOMS = {
    "sofa": frozenset(("bedroom", "bathroom", "kitchen", "laundry_room", "storage_closet")),
    "bed": frozenset(("kitchen", "bathroom", "living_room", "dining_room", "office", "hallway",
                      "laundry_room", "storage_closet", "lobby", "waiting_area", "reception", "exam_room")),
    "toilet": frozenset(("kitchen", "bedroom", "living_room", "dining_room", "office", "hallway",
                         "laundry_room", "storage_closet", "lobby", "waiting_area", "reception", "exam_room")),
    "television": frozenset(("bathroom", "laundry_room", "storage_closet", "hallway")),
    "chair": frozenset(("bathroom", "laundry_room", "storage_closet")),
    "potted plant": frozenset(("laundry_room", "storage_closet")),
}


#: Target -> object classes that mark the room the target lives in. A room holding one
#: is never ruled out for that target, whatever type its other objects gave it.
HOME_OBJECTS = {
    "toilet": frozenset(("sink", "shower", "bathtub", "toilet")),
    "sofa": frozenset(("television", "sofa", "coffee table")),
    "bed": frozenset(("bed", "wardrobe", "nightstand", "dresser")),
    "television": frozenset(("television", "sofa")),
    "chair": frozenset(("chair", "desk", "dining table")),
    "potted plant": frozenset(("potted plant",)),
}


def home_object(target, class_name):
    """Whether an object of ``class_name`` marks the room ``target`` lives in."""
    key = target_key(target)
    if key is None or not class_name:
        return False
    name = _ALIASES.get(str(class_name).strip().lower(), str(class_name).strip().lower())
    return name in HOME_OBJECTS.get(key, frozenset())


def ruled_out(target, label, strength, objects):
    """Why a room is not worth searching for ``target``, or None.

    ``type:<label>`` when the room's identified type cannot hold the target
    (:func:`implausible_room`), the label is **strong** (two or more kinds
    of object agree on it), and no object of the target's own class and no
    :func:`home_object` of the target stands in the room. A weak label, a
    seen target or a home object leave the room a node; the oracle values
    it on its evidence.
    """
    if not implausible_room(target, label):
        return None
    if str(strength or "").lower() != "strong":
        return None
    accepts = getattr(target, "accepts", None)
    for name in objects:
        if callable(accepts) and accepts(name):
            return None
        if home_object(target, name):
            return None
    return "type:%s" % label


def target_key(target):
    """The table key for a :class:`TargetLabels` (or a plain category string), or None."""
    names = []
    if isinstance(target, str):
        names.append(target)
    else:
        for attribute in ("query", "category"):
            value = getattr(target, attribute, None)
            if value:
                names.append(str(value))
        names.extend(str(v) for v in getattr(target, "accept", ()) or ())
    for name in names:
        key = _ALIASES.get(name.strip().lower(), name.strip().lower())
        if key in IMPLAUSIBLE_ROOMS:
            return key
    return None


def implausible_room(target, label):
    """Whether a room of type ``label`` is not worth searching for ``target``.

    Never true for ``unknown``, an empty label, or a target the table does
    not know.
    """
    if not label or str(label).strip().lower() in ("", "unknown"):
        return False
    key = target_key(target)
    if key is None:
        return False
    return str(label).strip().lower() in IMPLAUSIBLE_ROOMS[key]

