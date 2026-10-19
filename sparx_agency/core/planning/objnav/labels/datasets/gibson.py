"""Gibson ObjectNav v1.1 vocabulary (SemExp ``constants.py``).

Category indices are dataset indices, NOT COCO ids or semantic instance ids.
Only the first six categories are navigation goals; context objects cannot
trigger STOP. No fuzzy or LLM-based target matching is used in evaluation.
"""
from __future__ import annotations

from sparx_agency.core.planning.objnav.labels.table_mapper import TableLabelMapper
from sparx_agency.core.planning.objnav.types.target import LabelSpec

CATEGORIES = ("chair", "couch", "potted plant", "bed", "toilet", "tv")


def gibson_label_mapper() -> TableLabelMapper:
    """Build the six-category table and a scene-independent context vocabulary."""
    prompts = (
        ("chair",), ("couch", "sofa"), ("potted plant",),
        ("bed",), ("toilet",), ("tv", "television"),
    )
    table = {category: LabelSpec(category, names, names)
             for category, names in zip(CATEGORIES, prompts)}
    # Context classes: room cues for the LLM (sink, oven, shower ...), doors for the
    # partition, and -- since 2026-10-05 -- DISTRACTORS: an open-vocabulary detector
    # scores every prompt against the crop and lands on the nearest one it was given,
    # so a child's ride-on horse read as "chair" at 0.9 (Hanson/000001) when no toy
    # was in the vocabulary. A distractor class is never a navigation goal and never
    # names a room; it only gives the wrong match somewhere else to go. "bathtub" is a
    # home object of the toilet (room_priors.HOME_OBJECTS) that the vocabulary lacked.
    return TableLabelMapper(
        "gibson", table,
        context_vocabulary=("dining table", "oven", "sink", "refrigerator",
                            "book", "clock", "vase", "cup", "bottle",
                            "door", "doorway", "open doorway", "door frame",
                            "cabinet", "desk", "shower", "bathtub", "stairs", "staircase",
                            "toy", "rocking horse", "stuffed animal"))
