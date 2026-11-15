"""Matterport3D annotations -> the method's detector vocabulary, for the ground-truth perception.

Every MP3D object carries two names, both read by habitat-sim from the scene's
``<scene>.house`` file (its ``C`` lines): the annotator's **raw** category
(``"armchair"``, ``"fridge"``, row ``index`` of Matterport's
``metadata/category_mapping.tsv``) and the **mpcat40** class that table maps it
to (``chair``, ``appliances``; Matterport's ``metadata/mpcat40.tsv``). The 21
ObjectNav goal categories *are* mpcat40 classes: ``objectnav_mp3d_v1`` names
each goal by its mpcat40 name and lists the class index in
``category_to_mp3d_category_id``, and those indices are mpcat40.tsv's
(:data:`GOAL_MPCAT40_INDEX`; the dataset loader refuses a release that
disagrees). So the goal side of the table is an identity on the mpcat40 name,
not a hand-written synonym list. Its tiers mirror ``hm3d_categories``:

1. **The publisher's goal ids** (``MP3DDataset.goal_instances``): the
   ``object_id`` of every goal row of a category the scene's episodes ask
   for. They get the category's primary detector prompt (the MP3D label
   table's first prompt: ``"chest of drawers"``, ``"tv"``), and no other
   instance does. An instance of that mpcat40 class the publisher left out is
   **not reported** -- it would end an episode with a false STOP.
2. **A goal class the scene has no episodes for**: its instances by mpcat40
   name, under the same primary prompt.
3. **Context words** (doors, stairs, kitchen and bathroom fixtures): never an
   instance whose mpcat40 class is a goal. The raw name decides first
   (:data:`RAW_TO_CONTEXT`, every spelling ``category_mapping.tsv`` gives the
   word, typos included), then the mpcat40 class (:data:`MPCAT40_TO_CONTEXT`)
   for the classes that are one context word each.

Instance id 0 is never reported: habitat-sim renders 0 where nothing is
annotated, so MP3D's object 0 (a real object) cannot be told from background.
No published val or val_mini goal is object 0.

The table's SHA-256 goes into the frozen run configuration, so a changed table
is a changed experiment.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, namedtuple
from typing import Dict, FrozenSet, Mapping

from sparx_agency.core.common.label_match import normalize_label
from sparx_agency.core.planning.objnav.labels.datasets.mp3d import CATEGORIES, mp3d_label_mapper

#: One annotated MP3D instance: the ``.house`` raw category and its mpcat40 class.
MP3DAnnotation = namedtuple("MP3DAnnotation", ("raw", "mpcat40"))

#: mpcat40.tsv's index of each goal class -- what ``category_to_mp3d_category_id`` must say.
GOAL_MPCAT40_INDEX: Dict[str, int] = {
    "chair": 3, "table": 5, "picture": 6, "cabinet": 7, "cushion": 8, "sofa": 10, "bed": 11,
    "chest_of_drawers": 13, "plant": 14, "sink": 15, "toilet": 18, "stool": 19, "towel": 20,
    "tv_monitor": 22, "shower": 23, "bathtub": 25, "counter": 26, "fireplace": 27,
    "gym_equipment": 33, "seating": 34, "clothes": 38,
}

#: mpcat40 classes that are one context word each (mpcat40.tsv names).
MPCAT40_TO_CONTEXT: Dict[str, str] = {"door": "door", "stairs": "stairs", "mirror": "mirror", "curtain": "curtain"}

#: Raw ``.house`` category (normalised) -> context word: every raw spelling whose cleaned
#: ``category`` in category_mapping.tsv is the word. None of their mpcat40 classes is a goal.
RAW_TO_CONTEXT: Dict[str, str] = {
    # mpcat40 door, split the way the door finder's prompts are
    "doorway": "doorway",
    "door frame": "door frame", "doorframe": "door frame", "big door frame": "door frame", "dooframe": "door frame",
    "door frame 3": "door frame", "door frame fireplace wall": "door frame", "door frame wall entry": "door frame",
    "door side": "door frame", "doorfra,e": "door frame", "doorgrame": "door frame", "doorr frame": "door frame",
    # mpcat40 stairs
    "staircase": "staircase", "stair case": "staircase",
    # mpcat40 appliances
    "refrigerator": "refrigerator", "fridge": "refrigerator", "frige": "refrigerator",
    "oven": "oven", "microwave": "microwave", "stove": "stove", "range": "stove", "dishwasher": "dishwasher",
    "washing machine": "washing machine", "laundry machine": "washing machine",
    # mpcat40 shelving, objects, lighting, floor, furniture
    "bookshelf": "bookshelf", "bookcase": "bookshelf", "book stand": "bookshelf",
    "book": "book", "books": "book",
    "lamp": "lamp", "lamps": "lamp", "lamp base": "lamp", "lamb": "lamp",
    "clock": "clock", "vase": "vase", "cup": "cup", "cups": "cup",
    "bottle": "bottle", "bottles": "bottle", "while bottles": "bottle",
    "rug": "rug", "wardrobe": "wardrobe", "alamari": "wardrobe",
    "computer": "computer", "keyboard": "keyboard",
}


def _goal_labels() -> Dict[str, str]:
    mapper = mp3d_label_mapper()
    return {category: mapper.target_labels(category).detector_prompts[0] for category in CATEGORIES}


#: The label a goal class's instances carry: its primary detector prompt.
GOAL_LABELS: Dict[str, str] = _goal_labels()

if set(GOAL_MPCAT40_INDEX) != set(CATEGORIES):
    raise AssertionError("GOAL_MPCAT40_INDEX and the MP3D label table name different categories")


def check_release(category_to_mp3d_category_id: Mapping[str, int]) -> None:
    """Refuse a release whose goal classes are not mpcat40.tsv's.

    Raises:
        ValueError: A category's index differs from :data:`GOAL_MPCAT40_INDEX`.
    """
    wrong = {c: v for c, v in category_to_mp3d_category_id.items() if GOAL_MPCAT40_INDEX.get(c) != int(v)}
    if wrong:
        raise ValueError("category_to_mp3d_category_id disagrees with mpcat40.tsv for %r (expected %r): the "
                         "goal classes are not the mpcat40 classes the oracle labels"
                         % (wrong, {c: GOAL_MPCAT40_INDEX.get(c) for c in wrong}))


def context_label(annotation: MP3DAnnotation):
    """The context word for one non-goal instance, or ``None``."""
    word = RAW_TO_CONTEXT.get(normalize_label(str(annotation.raw)))
    return word if word is not None else MPCAT40_TO_CONTEXT.get(str(annotation.mpcat40))


def table_sha256() -> str:
    """Identity of the tables above, for the frozen run configuration."""
    body = json.dumps({"goal_mpcat40": GOAL_MPCAT40_INDEX, "goal_labels": GOAL_LABELS,
                       "mpcat40_context": MPCAT40_TO_CONTEXT, "raw_context": RAW_TO_CONTEXT},
                      sort_keys=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def instance_labels(semantic_objects: Mapping[int, MP3DAnnotation],
                    goal_instances: Mapping[str, FrozenSet[int]],
                    vocabulary) -> Dict[int, str]:
    """``{instance id: vocabulary word}`` for one scene, by the three tiers of the module docstring.

    Args:
        semantic_objects: ``{instance id: MP3DAnnotation}`` as habitat-sim read the ``.house`` file.
        goal_instances: ``{category: ids}`` of the scene's goal rows (``MP3DDataset.goal_instances``).
        vocabulary: The MP3D label table's ``vocabulary()``; nothing outside it is emitted.

    Raises:
        ValueError: A listed goal id is not an annotated instance, or its
            mpcat40 class is not its category: the episodes and the
            annotations disagree, and the oracle would never see the target.
    """
    words = frozenset(vocabulary)
    labels: Dict[int, str] = {}
    for category, ids in goal_instances.items():
        for instance in sorted(int(i) for i in ids):
            annotation = semantic_objects.get(instance)
            if annotation is None:
                raise ValueError(
                    "The publisher's %s goal %d is not an annotated instance of the scene (%d annotated); the "
                    "goal object_id and the semantic sensor's ids are not the same id space here"
                    % (category, instance, len(semantic_objects)))
            if annotation.mpcat40 != category:
                raise ValueError("The publisher's %s goal %d is annotated as mpcat40 %r (raw %r)"
                                 % (category, instance, annotation.mpcat40, annotation.raw))
            if GOAL_LABELS[category] in words:
                labels[instance] = GOAL_LABELS[category]
    for instance, annotation in semantic_objects.items():
        instance = int(instance)
        if instance in labels or instance <= 0:
            continue
        if annotation.mpcat40 in GOAL_LABELS:
            if annotation.mpcat40 in goal_instances:
                continue                  # the publisher has spoken for this category in this scene
            word = GOAL_LABELS[annotation.mpcat40]
        else:
            word = context_label(annotation)
        if word is not None and word in words:
            labels[instance] = word
    return labels


def unlabelled_account(semantic_objects: Mapping[int, MP3DAnnotation], labels: Mapping[int, str],
                       goal_instances: Mapping[str, FrozenSet[int]]):
    """``(deferred, unmapped)``: goal-class instances left out by the publisher, and the raw names with no word."""
    unmapped, deferred = Counter(), Counter()
    for instance, annotation in semantic_objects.items():
        if int(instance) in labels:
            continue
        if annotation.mpcat40 in goal_instances:
            deferred[annotation.mpcat40] += 1
        else:
            unmapped[str(annotation.raw)] += 1
    return deferred, unmapped


class MP3DCategoryTable:
    """The MP3D table behind the :class:`~.source.GroundTruthSource` seam (see ``hm3d_categories.HM3DCategoryTable``)."""

    name = "mp3d"
    annotations = "Matterport3D .house annotations, mpcat40 goal classes"
    context_labels = "Matterport3D raw .house names, then mpcat40 classes, via ground_truth/mp3d_categories.py"

    @staticmethod
    def instance_labels(semantic_objects, goal_instances, vocabulary):
        return instance_labels(semantic_objects, goal_instances, vocabulary)

    @staticmethod
    def unlabelled_account(semantic_objects, labels, goal_instances):
        return unlabelled_account(semantic_objects, labels, goal_instances)

    @staticmethod
    def sha256():
        return table_sha256()


#: The table every MP3D oracle reads.
MP3D_TABLE = MP3DCategoryTable()
