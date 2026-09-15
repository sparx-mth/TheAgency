"""HM3D ObjectNav vocabulary (the six challenge categories, v1 and v2).

The keys are the dataset's own ``object_category`` spellings, verbatim, as they
appear in ``objectnav_hm3d`` episode rows: ``chair``, ``bed``, ``plant``,
``toilet``, ``tv_monitor``, ``sofa``. The tuple order below is the dataset's
category index order (chair 0 ... sofa 5), which is what a row's numeric goal
index means -- it is **not** a COCO id and not a semantic instance id.

The same six categories are used by HM3D-v1 (2022 challenge, HM3D-Semantics
v0.1) and HM3D-v2 (HM3D-Semantics v0.2), so one table serves both; only the
episode data and the scene release differ.

Detector prompts are deliberately narrow. Matching is exact membership after
``normalize_label``, so an extra generous synonym is not a free recall
improvement: it is a false STOP waiting to happen. ``sofa`` is prompted as
``couch`` because that is the COCO/YOLO-World spelling, and ``tv_monitor`` as
``tv``; neither word is allowed to belong to a second row.

**No row carries accept-only labels**, and that is deliberate rather than an
omission. ``LabelSpec.accept`` exists for a closed-vocabulary detector that
emits its own synonyms unbidden. This benchmark runs an open-vocabulary one that
is configured with exactly ``TableLabelMapper.vocabulary()`` — the prompts and
the context words, never the accept-only labels — and
``methods/perception.py`` refuses any detection whose class is outside that
vocabulary. An accept-only synonym here would therefore be unreachable at best,
and at worst would abort the episode it was meant to rescue. A word we want
credit for has to be a prompt.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from sparx_agency.core.planning.objnav.labels.table_mapper import TableLabelMapper
from sparx_agency.core.planning.objnav.types.target import LabelSpec

#: Dataset ``object_category`` spellings, in dataset category-index order.
CATEGORIES = ("chair", "bed", "plant", "toilet", "tv_monitor", "sofa")

#: What each category is called when it is put to a language model.
QUERIES = {"chair": "chair", "bed": "bed", "plant": "potted plant",
           "toilet": "toilet", "tv_monitor": "tv", "sofa": "sofa"}

#: Detector prompts per category, primary first. Tuples, never sets: the order
#: reaches the open-vocabulary detector verbatim and must not move between a
#: run and its resume.
PROMPTS = {
    "chair": ("chair",),
    "bed": ("bed",),
    "plant": ("potted plant",),
    "toilet": ("toilet",),
    "tv_monitor": ("tv", "television"),
    "sofa": ("couch", "sofa"),
}

#: Non-goal prompts that give the room classifier and the door finder something
#: to see. A method choice, not part of the benchmark's vocabulary; the four
#: door prompts are the ones ``methods/doors.py`` looks for.
CONTEXT_VOCABULARY = (
    "dining table", "oven", "sink", "refrigerator", "microwave",
    "book", "clock", "vase", "cup", "bottle", "laptop",
    "door", "doorway", "open doorway", "door frame",
    "cabinet", "desk", "shower", "bathtub", "stairs", "washing machine",
)


def hm3d_label_mapper() -> TableLabelMapper:
    """The six-category table shared by HM3D-v1 and HM3D-v2.

    Returns:
        A :class:`TableLabelMapper` named ``"hm3d"``, keyed by the dataset's
        own category spellings, with a scene-independent context vocabulary.
    """
    table = {category: LabelSpec(QUERIES[category], PROMPTS[category])
             for category in CATEGORIES}
    return TableLabelMapper("hm3d", table,
                            context_vocabulary=CONTEXT_VOCABULARY)
