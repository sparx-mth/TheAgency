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

