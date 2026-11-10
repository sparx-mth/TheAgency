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

#: Prompt aliases collapse to one key, as in the room labeller. For OBJECT CLASS names (the detector's
#: labels): a class the home-object and generic-object tables read.
_ALIASES = {"couch": "sofa", "tv": "television", "tv monitor": "television", "monitor": "television",
            "armchair": "chair", "plant": "potted plant"}

#: For TARGET names only (:func:`target_key`): the category, query and accept labels of the MP3D, HM3D and
#: Gibson label tables, folded onto the table keys below. Kept apart from :data:`_ALIASES`, which canonicalises
#: object classes: ``coffee table`` is a home object of a sofa in its own right and must not read as ``table``.
_TARGET_ALIASES = dict(_ALIASES, **{
    "computer monitor": "television", "tv_monitor": "television", "houseplant": "potted plant",
    "desk": "table", "dining table": "table", "coffee table": "table", "side table": "table",
    "painting": "picture", "framed picture": "picture", "wall art": "picture",
    "kitchen cabinet": "cabinet", "cupboard": "cabinet", "pillow": "cushion", "throw pillow": "cushion",
    "dresser": "chest of drawers", "drawers": "chest of drawers", "nightstand": "chest of drawers",
    "chest_of_drawers": "chest of drawers", "bathroom sink": "sink", "kitchen sink": "sink", "washbasin": "sink",
    "bar stool": "stool", "step stool": "stool", "footstool": "stool", "bath towel": "towel", "hand towel": "towel",
    "shower stall": "shower", "shower cabin": "shower", "bath tub": "bathtub", "bath": "bathtub",
    "countertop": "counter", "kitchen counter": "counter", "chimney breast": "fireplace",
    "gym_equipment": "gym equipment", "exercise equipment": "gym equipment", "treadmill": "gym equipment",
    "exercise bike": "gym equipment", "weight bench": "gym equipment", "bench": "seating", "church pew": "seating",
    "pew": "seating", "clothing": "clothes", "laundry": "clothes", "hanging clothes": "clothes"})

#: Target -> the room types it lives in, most likely first, and the types it is rare in -- the prior the
#: node oracle is briefed with (:func:`target_home_line`), covering every MP3D goal category (the HM3D and
#: Gibson categories are among them). Until 2026-11-09 only the six HM3D categories had any prior, and
#: the 3B oracle answered "a table is in the bathroom" and "none of the known rooms are a counter" for
#: the rest (the MP3D ground-truth run, 14,457 oracle calls).
TARGET_HOMES = {
    "chair": (("dining_room", "kitchen", "office", "living_room", "bedroom"), ("bathroom", "hallway")),
    "table": (("dining_room", "kitchen", "living_room", "office", "bedroom"), ("bathroom", "hallway")),
    "picture": (("living_room", "hallway", "bedroom", "dining_room", "office", "foyer"),
                ("bathroom", "storage_closet", "laundry_room")),
    "cabinet": (("kitchen", "bathroom", "dining_room", "bedroom", "living_room", "laundry_room", "storage_closet"),
                ("hallway",)),
    "cushion": (("living_room", "bedroom"), ("kitchen", "bathroom", "hallway")),
    "sofa": (("living_room", "office", "bedroom"), ("kitchen", "bathroom", "hallway")),
    "bed": (("bedroom",), ("living_room", "kitchen", "bathroom", "dining_room", "hallway")),
    "chest of drawers": (("bedroom", "hallway", "living_room"), ("kitchen", "bathroom", "dining_room")),
    "potted plant": (("living_room", "hallway", "foyer", "dining_room", "kitchen", "balcony"),
                     ("storage_closet", "laundry_room")),
    "sink": (("kitchen", "bathroom", "laundry_room"), ("bedroom", "living_room", "dining_room", "hallway")),
    "toilet": (("bathroom",), ("kitchen", "bedroom", "living_room", "dining_room", "hallway")),
    "stool": (("kitchen", "bathroom", "living_room", "bedroom"), ("hallway", "storage_closet")),
    "towel": (("bathroom", "kitchen", "laundry_room", "bedroom"), ("living_room", "dining_room", "hallway")),
    "television": (("living_room", "bedroom", "office"), ("kitchen", "bathroom", "hallway")),
    "shower": (("bathroom",), ("kitchen", "bedroom", "living_room", "dining_room", "hallway")),
    "bathtub": (("bathroom",), ("kitchen", "bedroom", "living_room", "dining_room", "hallway")),
    "counter": (("kitchen", "bathroom", "laundry_room"), ("bedroom", "living_room", "dining_room", "hallway")),
    "fireplace": (("living_room", "dining_room", "bedroom", "office"), ("kitchen", "bathroom", "hallway")),
    "gym equipment": (("office", "bedroom", "garage", "storage_closet", "living_room"),
                      ("kitchen", "bathroom", "dining_room", "hallway")),
    "seating": (("living_room", "dining_room", "foyer", "hallway", "kitchen"), ("bathroom", "storage_closet")),
    "clothes": (("bedroom", "storage_closet", "laundry_room", "bathroom"), ("kitchen", "dining_room", "living_room")),
}

#: Target -> room types it is not worth searching for it. Only the hard cases: a type that is
#: implausible but not impossible (a sofa in a dining room, a chair in a hallway) is left to the
#: oracle, which reads :data:`TARGET_HOMES` instead.
IMPLAUSIBLE_ROOMS = {
    "sofa": frozenset(("bedroom", "bathroom", "kitchen", "laundry_room", "storage_closet")),
    "bed": frozenset(("kitchen", "bathroom", "living_room", "dining_room", "office", "hallway",
                      "laundry_room", "storage_closet", "lobby", "waiting_area", "reception", "exam_room")),
    "toilet": frozenset(("kitchen", "bedroom", "living_room", "dining_room", "office", "hallway",
                         "laundry_room", "storage_closet", "lobby", "waiting_area", "reception", "exam_room")),
    "television": frozenset(("bathroom", "laundry_room", "storage_closet", "hallway")),
    "chair": frozenset(("bathroom", "laundry_room", "storage_closet")),
    "potted plant": frozenset(("laundry_room", "storage_closet")),
    "table": frozenset(("bathroom", "laundry_room", "storage_closet")),
    "picture": frozenset(("laundry_room", "storage_closet", "garage")),
    "cabinet": frozenset(()),
    "cushion": frozenset(("bathroom", "kitchen", "laundry_room", "storage_closet", "garage")),
    "chest of drawers": frozenset(("kitchen", "bathroom", "dining_room", "laundry_room", "garage")),
    "sink": frozenset(("bedroom", "living_room", "dining_room", "office", "hallway", "foyer")),
    "stool": frozenset(("laundry_room", "storage_closet")),
    "towel": frozenset(("living_room", "dining_room", "office", "hallway", "foyer", "storage_closet", "garage")),
    "shower": frozenset(("kitchen", "bedroom", "living_room", "dining_room", "office", "hallway", "foyer",
                         "laundry_room", "storage_closet", "garage")),
    "bathtub": frozenset(("kitchen", "bedroom", "living_room", "dining_room", "office", "hallway", "foyer",
                          "laundry_room", "storage_closet", "garage")),
    "counter": frozenset(("bedroom", "living_room", "dining_room", "office", "hallway", "foyer", "storage_closet",
                          "garage")),
    "fireplace": frozenset(("kitchen", "bathroom", "laundry_room", "storage_closet", "garage", "hallway")),
    "gym equipment": frozenset(("kitchen", "bathroom", "dining_room", "laundry_room", "hallway", "foyer")),
    "seating": frozenset(("bathroom", "laundry_room", "storage_closet")),
    "clothes": frozenset(("kitchen", "dining_room", "living_room", "office", "hallway", "foyer", "garage")),
}


#: Target -> object classes that mark the room the target lives in. A room holding one
#: is never ruled out for that target, whatever type its other objects gave it.
HOME_OBJECTS = {
    "toilet": frozenset(("sink", "shower", "bathtub", "toilet", "towel", "mirror")),
    "sofa": frozenset(("television", "sofa", "coffee table", "cushion", "fireplace", "seating")),
    "bed": frozenset(("bed", "wardrobe", "nightstand", "dresser", "chest of drawers", "cushion", "clothes")),
    "television": frozenset(("television", "sofa", "fireplace", "cushion")),
    "chair": frozenset(("chair", "desk", "dining table", "table", "stool", "seating")),
    "potted plant": frozenset(("potted plant",)),
    "table": frozenset(("table", "chair", "stool", "seating", "sofa")),
    "picture": frozenset(("picture",)),
    "cabinet": frozenset(("cabinet", "counter", "sink", "stove", "oven", "refrigerator", "dishwasher", "microwave",
                          "mirror", "toilet")),
    "cushion": frozenset(("cushion", "sofa", "bed", "seating", "chair")),
    "chest of drawers": frozenset(("chest of drawers", "bed", "wardrobe", "mirror", "lamp")),
    "sink": frozenset(("sink", "toilet", "shower", "bathtub", "towel", "mirror", "counter", "stove", "oven",
                       "refrigerator", "dishwasher")),
    "stool": frozenset(("stool", "counter", "table")),
    "towel": frozenset(("towel", "sink", "shower", "bathtub", "toilet", "mirror")),
    "shower": frozenset(("shower", "toilet", "sink", "bathtub", "towel")),
    "bathtub": frozenset(("bathtub", "toilet", "sink", "shower", "towel")),
    "counter": frozenset(("counter", "sink", "stove", "oven", "refrigerator", "dishwasher", "microwave", "stool",
                          "cabinet", "mirror", "toilet")),
    "fireplace": frozenset(("fireplace", "sofa", "television", "seating")),
    "gym equipment": frozenset(("gym equipment",)),
    "seating": frozenset(("seating", "table", "chair", "cushion", "sofa")),
    "clothes": frozenset(("clothes", "wardrobe", "bed", "washing machine", "chest of drawers", "towel")),
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
        key = _TARGET_ALIASES.get(name.strip().lower(), name.strip().lower())
        if key in TARGET_HOMES or key in IMPLAUSIBLE_ROOMS:
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
    return str(label).strip().lower() in IMPLAUSIBLE_ROOMS.get(key, frozenset())


def _room_words(labels):
    """``bedroom, living room`` for a tuple of classifier labels (the underscores read badly in prose)."""
    return ", ".join(str(label).replace("_", " ") for label in labels)


def target_home_line(target):
    """One prompt line on where ``target`` normally lives, from :data:`TARGET_HOMES`, or an empty string.

    ``bathroom, kitchen, laundry room (most likely first); rare in: bedroom, living room,
    dining room, hallway`` -- the node oracle's step 1, given rather than guessed, for
    every category of the three benchmarks; the prompt labels it ``WHERE IT LIVES``.
    """
    key = target_key(target)
    homes = TARGET_HOMES.get(key) if key is not None else None
    if not homes:
        return ""
    likely, rare = homes
    line = "%s (most likely first)" % _room_words(likely)
    if rare:
        line += "; rare in: %s" % _room_words(rare)
    return line

