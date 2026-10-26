"""HM3D-Semantics raw category names -> the method's detector vocabulary, for the ground-truth perception.

HM3D-Semantics names every annotated instance in the annotator's own words --
``"chair"``, ``"office chair"``, ``"tv stand"``, ``"wall"`` -- some 1,600
distinct strings over the release. The method's detector speaks the label
table's vocabulary (:func:`~sparx_agency.core.planning.objnav.labels.datasets.hm3d.hm3d_label_mapper`:
six goal categories' prompts plus the context words that tell rooms apart),
and a detection outside that vocabulary is refused by the perception cycle. So
the oracle has to translate, and it does so in two tiers:

1. **The goal categories come from the publisher, not from this table.** The
   episode files list, per scene and category, the instance ids that *are* its
   goals (``HM3DDataset.goal_instances``) -- the publisher's own reading of the
   annotations, and exactly what the success radius is measured to. Where a
   scene lists a category, those ids get the category's primary prompt and no
   other instance does, however it is named: a ``"chair"`` the publisher left
   out of the chair goals would end a chair episode with a false STOP if the
   oracle called it one.
2. **Everything else is this table**: the context vocabulary (doors, stairs,
   kitchen and bathroom fixtures, the distractors) and, in a scene whose
   episodes never ask for a category, that category's instances by name.

The table is exact-match on the normalised raw name. Deliberately narrow: a
raw name that is not listed is not emitted, because the vocabulary is closed
and an ideal detector prompted with ``"dining table"`` has no word for a
nightstand. Judgement calls are marked below; the table's SHA-256 goes into
the frozen run configuration so a changed table is a changed experiment.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, FrozenSet, Mapping, Optional

from sparx_agency.core.common.label_match import normalize_label
from sparx_agency.core.planning.objnav.labels.datasets.hm3d import CATEGORIES, PROMPTS

#: Raw HM3D-Semantics category (normalised) -> detector vocabulary word.
RAW_TO_VOCABULARY: Dict[str, str] = {
    # -- goal categories: tier 2 only (a scene without goals of the category) ---------------------
    "chair": "chair", "dining chair": "chair", "office chair": "chair", "desk chair": "chair",
    "kitchen chair": "chair", "folding chair": "chair", "rocking chair": "chair", "high chair": "chair",
    "armchair": "armchair", "arm chair": "armchair", "sofa chair": "armchair", "lounge chair": "armchair",
    "recliner": "armchair",
    "bed": "bed", "double bed": "bed", "single bed": "bed", "bunk bed": "bed", "king bed": "bed", "queen bed": "bed",
    "plant": "potted plant", "potted plant": "potted plant", "houseplant": "potted plant", "house plant": "potted plant",
    "indoor plant": "potted plant", "decorative plant": "potted plant",
    "toilet": "toilet", "toilet bowl": "toilet",
    "tv": "tv", "television": "television", "tv monitor": "tv", "tv screen": "tv", "flat screen tv": "tv",
    "flatscreen tv": "tv", "monitor": "monitor", "computer monitor": "monitor",
    "sofa": "couch", "couch": "couch", "loveseat": "couch", "love seat": "couch", "sectional sofa": "couch",
    "l shaped sofa": "couch", "l-shaped sofa": "couch", "sofa bed": "couch",
    # -- context vocabulary: the room classifier's and the door finder's words --------------------
    # "table" alone is a judgement call: HM3D's bare "table" is most often a dining or kitchen table,
    # and YOLO-World's "dining table" prompt (COCO's table class) fires on them; coffee and side
    # tables are named as such in the annotations and are not listed.
    "table": "dining table", "dining table": "dining table", "kitchen table": "dining table",
    "oven": "oven", "stove": "oven", "stove top": "oven", "stovetop": "oven", "cooktop": "oven",
    "kitchen stove": "oven", "range": "oven",
    "sink": "sink", "kitchen sink": "sink", "bathroom sink": "sink", "bath sink": "sink", "wash basin": "sink",
    "washbasin": "sink", "basin": "sink",
    "refrigerator": "refrigerator", "fridge": "refrigerator", "mini fridge": "refrigerator", "freezer": "refrigerator",
    "book": "book", "books": "book",
    "clock": "clock", "wall clock": "clock", "alarm clock": "clock",
    "vase": "vase",
    "cup": "cup", "mug": "cup",
    "bottle": "bottle", "water bottle": "bottle",
    "door": "door", "front door": "door", "sliding door": "door", "glass door": "door", "closet door": "door",
    "double door": "door", "garage door": "door", "open door": "door", "wooden door": "door",
    "doorway": "doorway", "door way": "doorway",
    "door frame": "door frame", "doorframe": "door frame",
    "cabinet": "cabinet", "kitchen cabinet": "cabinet", "cupboard": "cabinet", "wardrobe": "cabinet",
    "bathroom cabinet": "cabinet", "bath cabinet": "cabinet", "storage cabinet": "cabinet", "dresser": "cabinet",
    "chest of drawers": "cabinet",
    "desk": "desk", "office desk": "desk", "computer desk": "desk", "writing desk": "desk",
    "shower": "shower", "shower stall": "shower", "shower cabin": "shower",
    "bathtub": "bathtub", "bath tub": "bathtub", "tub": "bathtub", "bath": "bathtub",
    "stairs": "stairs", "stair": "stairs", "steps": "stairs", "stairway": "staircase", "staircase": "staircase",
    "toy": "toy", "toys": "toy", "doll": "toy",
    "rocking horse": "rocking horse",
    "stuffed animal": "stuffed animal", "teddy bear": "stuffed animal", "plush toy": "stuffed animal",
    "stool": "stool", "bar stool": "bar stool", "barstool": "bar stool",
}

#: The label a goal category's instances carry: its primary detector prompt.
GOAL_LABELS: Dict[str, str] = {category: PROMPTS[category][0] for category in CATEGORIES}

#: The vocabulary words that belong to a goal category (every prompt), so tier 2 can tell a
#: goal-category name from a context word and defer to the publisher where it has spoken.
PROMPT_CATEGORY: Dict[str, str] = {prompt: category for category in CATEGORIES for prompt in PROMPTS[category]}


def vocabulary_label(raw_name: str) -> Optional[str]:
    """The detector vocabulary word for one raw HM3D-Semantics name, or ``None`` when the method has no word for it."""
    return RAW_TO_VOCABULARY.get(normalize_label(str(raw_name)))


def table_sha256() -> str:
    """Identity of the mapping above, for the frozen run configuration."""
    body = json.dumps(RAW_TO_VOCABULARY, sort_keys=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def instance_labels(semantic_objects: Mapping[int, str],
                    goal_instances: Mapping[str, FrozenSet[int]],
                    vocabulary) -> Dict[int, str]:
    """``{instance id: vocabulary word}`` for one scene: the publisher's goals first, the name table second.

    Args:
        semantic_objects: ``{instance id: raw category}`` as the simulator read
            the scene's annotations.
        goal_instances: ``{category: ids}`` the publisher lists as that
            category's goals in this scene (``HM3DDataset.goal_instances``).
        vocabulary: The label table's ``vocabulary()``; a word outside it is
            never emitted, whatever the table says.

    Returns:
        Only the instances the oracle will report. An instance named for a
        goal category the publisher has listed in this scene, but not among
        its ids, is dropped (tier 1 rules, see the module docstring).

    Raises:
        ValueError: A listed goal id is not an annotated instance of the
            scene: the id spaces differ, and an oracle blind to the target
            would score like a failure of the method.
    """
    words = frozenset(vocabulary)
    labels: Dict[int, str] = {}
    listed = {category: frozenset(int(i) for i in ids) for category, ids in goal_instances.items()}
    for category, ids in listed.items():
        missing = sorted(i for i in ids if i not in semantic_objects)
        if missing:
            raise ValueError(
                "The publisher's %s goals name instance ids %s that the scene's semantic annotations do not "
                "contain (%d annotated instances); the goal object_id and the semantic sensor's ids are not the "
                "same id space here, and the oracle would never see the target"
                % (category, missing[:5], len(semantic_objects)))
        label = GOAL_LABELS[category]
        if label in words:
            for instance in ids:
                labels[instance] = label
    for instance, raw in semantic_objects.items():
        instance = int(instance)
        if instance in labels:
            continue
        word = vocabulary_label(raw)
        if word is None or word not in words:
            continue
        category = PROMPT_CATEGORY.get(word)
        if category is not None and category in listed:
            continue                      # the publisher has spoken for this category in this scene
        labels[instance] = word
    return labels
