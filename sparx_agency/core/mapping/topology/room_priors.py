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


#: Object class -> the one room type it names on its own. A bed is a bedroom and a toilet
#: a bathroom with no second kind of object needed: the label a classifier gives a room
#: holding one is STRONG when it agrees with this table (``room_labels.RevisableRoomLabels``),
#: so the type prior may rule the room out at once -- the Hanson recording of 2026-10-05
#: walked into a bedroom it had seen the bed of through the door (actions 270-289), because
#: one kind of object was by rule a weak label. A chair, a cabinet, a desk or a plant names
#: nothing on its own and is not listed. The merge guards of :func:`ruled_out` still hold:
#: a home object of the target, or the target itself, keeps the room a node.
SIGNATURE_OBJECTS = {
    "bed": "bedroom",
    "toilet": "bathroom",
    "shower": "bathroom",
    "bathtub": "bathroom",
    "oven": "kitchen",
    "stove": "kitchen",
    "refrigerator": "kitchen",
}


#: Object classes that stand in every kind of room and name none: two of them together are
#: not a STRONG label (``room_labels.RevisableRoomLabels``). The Ranchester couch search of
#: 2026-10-05 had a cabinet and a potted plant make an upstairs room a "living_room" at 0.95,
#: strong, which put a living room on the storey summary the node oracle reads and wobbled its
#: verdict on where living rooms are; under the type prior the same label would have ruled the
#: room out for a bed or a toilet. A chair, a desk, a table, a sofa or a television is not
#: generic: with a second kind they do describe a room.
GENERIC_OBJECTS = frozenset(("cabinet", "potted plant", "book", "vase", "clock", "cup", "bottle"))


def generic_object(class_name):
    """Whether ``class_name`` is an object that stands in any room and names none (:data:`GENERIC_OBJECTS`)."""
    if not class_name:
        return False
    name = _ALIASES.get(str(class_name).strip().lower(), str(class_name).strip().lower())
    return name in GENERIC_OBJECTS


def signature_type(class_name):
    """The room type ``class_name`` names on its own (:data:`SIGNATURE_OBJECTS`), or None."""
    if not class_name:
        return None
    name = _ALIASES.get(str(class_name).strip().lower(), str(class_name).strip().lower())
    return SIGNATURE_OBJECTS.get(name)


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

