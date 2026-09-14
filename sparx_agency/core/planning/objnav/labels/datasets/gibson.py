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
    return TableLabelMapper(
        "gibson", table,
        context_vocabulary=("dining table", "oven", "sink", "refrigerator",
                            "book", "clock", "vase", "cup", "bottle",
                            "door", "doorway", "open doorway", "door frame",
                            "cabinet", "desk", "shower"))
