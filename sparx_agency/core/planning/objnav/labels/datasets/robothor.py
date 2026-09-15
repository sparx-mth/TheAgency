"""RoboTHOR ObjectNav vocabulary: the challenge's 12 target `objectType` strings.

The categories are AI2-THOR ``objectType`` identifiers, verbatim, as they
appear in the episode files and in ``obj["objectId"].split("|")[0]`` -- which
is how ``robothor_challenge/challenge.py`` finds the goal object. They are
CamelCase with no spaces, and :func:`~sparx_agency.core.common.label_match.normalize_label`
deliberately does not split CamelCase, so ``"HousePlant"`` normalises to
``"houseplant"`` and the readable ``"house plant"`` a detector is prompted
with has to be written out here. Two spellings are easy to get wrong and are
worth checking against the dataset rather than the prose: ``BasketBall`` has a
capital second B, and the challenge README's prose list ("Alarm Clock",
"Spray Bottle") is *not* the API spelling.

The upstream RoboTHOR documentation lists a 13th type, ``RemoteControl``. It
is not a challenge target and is not a goal here; the shipped val episodes use
exactly these 12, 150 episodes each.

Accept sets are deliberately narrow. A false accept ends the episode with a
false STOP, and RoboTHOR scenes contain near neighbours of several goals --
``Cup`` beside ``Mug``, ``Bottle`` beside ``SprayBottle``, ``Box`` beside
``GarbageCan`` -- so the obvious short synonym is usually the dangerous one.
Two exceptions are taken knowingly, because without them a whole category's
recall collapses to nothing and 150 episodes are lost outright: ``"clock"``
counts as ``AlarmClock`` and ``"sports ball"`` as ``BasketBall``, since those
are the words a COCO-trained or COCO-biased open-vocabulary detector actually
emits. Neither has a competing RoboTHOR goal, but both are a genuine
false-positive surface and are recorded as such in the adapter's README.

The context vocabulary is what tells RoboTHOR's apartment rooms apart, plus
the door prompts the depth-backed door detector needs. None of it can trigger
a STOP: only a category's own accept set can.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from sparx_agency.core.planning.objnav.labels.table_mapper import TableLabelMapper
from sparx_agency.core.planning.objnav.types.target import LabelSpec

#: The 12 challenge target ``objectType`` strings, in the order the RoboTHOR
#: challenge lists them. Results are reported per category under these keys.
CATEGORIES = (
    "AlarmClock", "Apple", "BaseballBat", "BasketBall", "Bowl", "GarbageCan",
    "HousePlant", "Laptop", "Mug", "SprayBottle", "Television", "Vase",
)

#: ``category -> (LLM query, detector prompts, extra accepted labels)``.
_TABLE = {
    "AlarmClock": ("alarm clock", ("alarm clock",), ("clock", "digital clock")),
    "Apple": ("apple", ("apple",), ()),
    "BaseballBat": ("baseball bat", ("baseball bat",), ()),
    "BasketBall": ("basketball", ("basketball",), ("basket ball", "sports ball")),
    "Bowl": ("bowl", ("bowl",), ("mixing bowl",)),
    "GarbageCan": ("garbage can", ("garbage can", "trash can"),
                   ("trash bin", "waste bin", "wastebasket", "waste basket")),
    "HousePlant": ("house plant", ("house plant", "potted plant"),
                   ("houseplant", "plant in a pot")),
    "Laptop": ("laptop", ("laptop",), ("laptop computer", "notebook computer")),
    "Mug": ("mug", ("mug",), ("coffee mug",)),
    "SprayBottle": ("spray bottle", ("spray bottle",), ("spray can",)),
    "Television": ("tv", ("television", "tv"), ("tv monitor", "flat screen tv")),
    "Vase": ("vase", ("vase",), ("flower vase",)),
}

#: Objects that separate an apartment's rooms, plus the door prompts
#: :mod:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors`
#: needs. Never a goal, so never a STOP.
CONTEXT_VOCABULARY = (
    "bed", "sofa", "couch", "armchair", "desk", "office chair", "chair",
    "dining table", "coffee table", "side table", "shelf", "bookshelf",
    "dresser", "wardrobe", "cabinet", "drawer",
    "sink", "toilet", "bathtub", "shower", "towel", "mirror",
    "refrigerator", "microwave", "oven", "stove", "kettle", "toaster",
    "book", "pillow", "blanket", "lamp", "floor lamp", "painting", "window",
    "door", "doorway", "open doorway", "door frame",
)


def robothor_label_mapper() -> TableLabelMapper:
    """Build the 12-category RoboTHOR table and its apartment context vocabulary."""
    table = {category: LabelSpec(query, prompts, accept)
             for category, (query, prompts, accept) in _TABLE.items()}
    return TableLabelMapper("robothor", table,
                            context_vocabulary=CONTEXT_VOCABULARY)
