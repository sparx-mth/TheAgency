"""MP3D ObjectNav v1 vocabulary: the 21 goal categories of the 2021 Habitat Challenge.

Categories are spelled as the published dataset spells them, underscores and
all (``chest_of_drawers``, ``tv_monitor``, ``gym_equipment``): the episode
files carry ``object_category`` verbatim, and
:meth:`~sparx_agency.core.planning.objnav.labels.table_mapper.TableLabelMapper.target_labels`
looks the category up without normalising it. The dataset loader cross-checks
this tuple against the release's own ``category_to_task_category_id``, so a
different release fails at load instead of scoring 21 categories against 20.

The ``query`` of each row is what a person (and the LLM) would call the object,
which is not the dataset's spelling. The accept sets are deliberately narrow
and mutually exclusive: MP3D has ``chair``, ``stool`` and ``seating`` as three
separate goals, and a label accepted by two of them is a guaranteed false STOP
for one. No fuzzy or LLM-based target matching is used in evaluation.
"""
from __future__ import annotations

from sparx_agency.core.planning.objnav.labels.table_mapper import TableLabelMapper
from sparx_agency.core.planning.objnav.types.target import LabelSpec

#: The published goal categories, in the release's task-category-id order.
CATEGORIES = (
    "chair", "table", "picture", "cabinet", "cushion", "sofa", "bed",
    "chest_of_drawers", "plant", "sink", "toilet", "stool", "towel",
    "tv_monitor", "shower", "bathtub", "counter", "fireplace",
    "gym_equipment", "seating", "clothes",
)

#: category -> (LLM/query name, detector prompts, extra accepted labels).
_TABLE = (
    ("chair", "chair", ("chair",), ()),
    # mpcat40 has no "desk": MP3D annotates desks under "table", so "desk"
    # must be a prompt of this row. As an accept-only label it would be
    # unreachable -- vocabulary() asks the detector for prompts only.
    ("table", "table", ("table", "desk"), ("dining table", "coffee table", "side table")),
    ("picture", "picture", ("picture", "painting"), ("framed picture", "wall art")),
    ("cabinet", "cabinet", ("cabinet",), ("kitchen cabinet", "cupboard")),
    ("cushion", "cushion", ("cushion", "pillow"), ("throw pillow",)),
    ("sofa", "sofa", ("sofa", "couch"), ()),
    ("bed", "bed", ("bed",), ()),
    ("chest_of_drawers", "chest of drawers", ("chest of drawers", "dresser"),
     ("drawers", "nightstand")),
    ("plant", "potted plant", ("potted plant", "plant"), ("houseplant",)),
    ("sink", "sink", ("sink",), ("bathroom sink", "kitchen sink", "washbasin")),
    ("toilet", "toilet", ("toilet",), ()),
    ("stool", "stool", ("stool",), ("bar stool", "step stool", "footstool")),
    ("towel", "towel", ("towel",), ("bath towel", "hand towel")),
    ("tv_monitor", "tv", ("tv", "television"), ("tv monitor", "monitor", "computer monitor")),
    ("shower", "shower", ("shower",), ("shower stall", "shower cabin")),
    ("bathtub", "bathtub", ("bathtub",), ("bath tub", "bath")),
    ("counter", "counter", ("counter", "countertop"), ("kitchen counter",)),
    ("fireplace", "fireplace", ("fireplace",), ("chimney breast",)),
    ("gym_equipment", "gym equipment", ("gym equipment", "exercise equipment"),
     ("treadmill", "exercise bike", "weight bench")),
    ("seating", "seating", ("seating", "bench"), ("church pew", "pew")),
    ("clothes", "clothes", ("clothes", "clothing"), ("laundry", "hanging clothes")),
)

#: Prompts that are not goals but tell one MP3D room from another, plus the
#: door prompts the depth-backed door detector needs.
CONTEXT_VOCABULARY = (
    "door", "doorway", "open doorway", "door frame",
    "refrigerator", "oven", "microwave", "stove", "dishwasher",
    "washing machine", "bookshelf", "book", "lamp", "mirror",
    "curtain", "clock", "vase", "cup", "bottle", "rug", "stairs",
    "wardrobe", "computer", "keyboard",
)


def mp3d_label_mapper() -> TableLabelMapper:
    """Build the 21-category MP3D table and its scene-independent context vocabulary."""
    table = {}
    for category, query, prompts, accept in _TABLE:
        table[category] = LabelSpec(query, prompts, accept)
    if tuple(table) != CATEGORIES:
        raise AssertionError("MP3D table and CATEGORIES disagree")
    return TableLabelMapper("mp3d", table, context_vocabulary=CONTEXT_VOCABULARY)
