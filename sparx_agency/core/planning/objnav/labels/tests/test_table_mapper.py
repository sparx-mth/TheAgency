"""What a dataset's label table hands the policy, and what it refuses to build.

The table is where a benchmark's words become our detector's words, and every
way it can be wrong is silent at run time: an ambiguous accept label is a
false STOP, a respelled category searches for nothing, a set-valued vocabulary
reorders the detector's prompts between runs. Each test pins one of them.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.objnav.errors import (
    ObjNavError,
    UnknownCategoryError,
)
from sparx_agency.core.planning.objnav.interfaces.label_mapper import LabelMapper
from sparx_agency.core.planning.objnav.labels.table_mapper import TableLabelMapper
from sparx_agency.core.planning.objnav.types.target import LabelSpec, TargetLabels

HM3D_ORDER = ("chair", "bed", "plant", "toilet", "tv_monitor", "sofa")


def hm3d_like_table():
    """The six HM3D ObjectNav categories, written the way a person would."""
    return {
        "chair": LabelSpec("chair", ("chair",)),
        "bed": LabelSpec("bed", ("bed",)),
        "plant": LabelSpec("Potted plant",
                           ("potted plant", "Potted_Plant", "plant"),
                           accept=("houseplant",)),
        "toilet": LabelSpec("toilet", ("toilet",)),
        "tv_monitor": LabelSpec("TV", ("tv", "TV_Monitor", "television"),
                                accept=("Computer_Monitor",)),
        "sofa": LabelSpec("sofa", ("sofa", "couch")),
    }


def hm3d_like(context=()):
    """A mapper over :func:`hm3d_like_table`."""
    return TableLabelMapper("hm3d_like", hm3d_like_table(),
                            context_vocabulary=context)


# -- what it hands out ----------------------------------------------------

def test_the_mapper_is_a_label_mapper_named_by_its_registry_key():
    """The agent type-checks against the ABC and the registry checks the name."""
    mapper = hm3d_like()
    assert isinstance(mapper, LabelMapper)
    assert mapper.name == "hm3d_like"


def test_categories_come_back_verbatim_in_table_order():
    """Results are reported under the dataset's own spelling, in its order."""
    assert hm3d_like().categories() == HM3D_ORDER


def test_every_category_round_trips_into_normalised_target_labels():
    """The policy must receive the normalised query, prompts and accept set."""
    mapper = hm3d_like()
    assert mapper.target_labels("tv_monitor") == TargetLabels(
        category="tv_monitor",
        query="tv",
        detector_prompts=("tv", "tv monitor", "television"),
        accept_labels=frozenset(
            {"tv", "tv monitor", "television", "computer monitor"}),
    )
    for category in HM3D_ORDER:
        target = mapper.target_labels(category)
        assert target.category == category
        assert mapper.covers(category)


def test_the_same_prebuilt_target_is_returned_on_every_call():
    """Targets are built once at construction, never per episode."""
    mapper = hm3d_like()
    assert mapper.target_labels("bed") is mapper.target_labels("bed")


def test_a_detector_spelling_of_the_category_is_accepted():
    """``TV_Monitor`` from a detector must count as the tv_monitor goal."""
    target = hm3d_like().target_labels("tv_monitor")
    assert target.accepts("TV_Monitor")
    assert target.accepts("  Television ")


def test_an_accept_only_label_counts_but_is_never_prompted():
    """A closed-vocabulary synonym is accepted without widening the prompts."""
    mapper = hm3d_like()
    assert mapper.target_labels("tv_monitor").accepts("computer monitor")
    assert "computer monitor" not in mapper.vocabulary()


def test_accepting_is_exact_never_fuzzy():
    """A fuzzy neighbour of the target is a false STOP, so it must not count."""
    mapper = hm3d_like()
    assert not mapper.target_labels("tv_monitor").accepts("tv stand")
    assert not mapper.target_labels("plant").accepts("plant pot")


def test_duplicate_prompts_are_dropped_keeping_the_first_position():
    """A repeated prompt wastes detector capacity and breaks TargetLabels."""
    target = hm3d_like().target_labels("plant")
    assert target.query == "potted plant"
    assert target.detector_prompts == ("potted plant", "plant")


def test_vocabulary_is_prompts_in_table_order_then_context_without_repeats():
    """A detector is prompted in a stable order, goals first, each label once."""
    mapper = hm3d_like(context=["Sink", "chair", "refrigerator", "sink"])
    assert mapper.vocabulary() == (
        "chair", "bed", "potted plant", "plant", "toilet",
        "tv", "tv monitor", "television", "sofa", "couch",
        "sink", "refrigerator",
    )


def test_changing_the_source_table_afterwards_does_not_change_the_mapper():
    """A mapper shared across episodes must not drift with its caller's dict."""
    table = hm3d_like_table()
    mapper = TableLabelMapper("hm3d_like", table)
    table["lamp"] = LabelSpec("lamp", ("lamp",))
    del table["bed"]
    assert mapper.categories() == HM3D_ORDER
    assert not mapper.covers("lamp")


def test_repr_names_the_mapper_and_its_categories():
    """A log line that prints the mapper must say which table it is."""
    text = repr(hm3d_like())
    assert "hm3d_like" in text
    assert "tv_monitor" in text


# -- lookups --------------------------------------------------------------

def test_an_unknown_category_raises_naming_the_mapper_and_its_categories():
    """A missing table row must fail at reset with enough to fix it."""
    with pytest.raises(UnknownCategoryError) as caught:
        hm3d_like().target_labels("fireplace")
    message = str(caught.value)
    assert "hm3d_like" in message
    assert "fireplace" in message
    for category in HM3D_ORDER:
        assert repr(category) in message


def test_a_respelled_category_is_not_normalised_into_a_hit():
    """An adapter that respells the category has a bug; the lookup reports it."""
    mapper = hm3d_like()
    assert not mapper.covers("TV_Monitor")
    with pytest.raises(UnknownCategoryError, match="dataset's own spelling"):
        mapper.target_labels("TV_Monitor")


def test_a_category_that_is_not_a_string_is_an_unknown_category():
    """An unhashable category must not escape as a bare TypeError."""
    with pytest.raises(UnknownCategoryError):
        hm3d_like().target_labels(["chair"])


# -- what it refuses to build ---------------------------------------------

def test_a_label_accepted_by_two_categories_is_refused_naming_both():
    """One detection counting as two goals is a guaranteed false STOP."""
    table = {
        "sofa": LabelSpec("sofa", ("sofa",), accept=("couch",)),
        "chair": LabelSpec("chair", ("chair",), accept=("Couch",)),
    }
    with pytest.raises(ObjNavError) as caught:
        TableLabelMapper("clash", table)
    message = str(caught.value)
    assert "'couch'" in message
    assert "'sofa'" in message and "'chair'" in message


def test_a_prompt_of_one_category_accepted_by_another_is_refused():
    """Prompts are accepted too, so they collide with another row's accept set."""
    table = {
        "chair": LabelSpec("chair", ("chair", "armchair")),
        "sofa": LabelSpec("sofa", ("sofa",), accept=("arm_chair", "Armchair")),
    }
    with pytest.raises(ObjNavError, match="'armchair' is accepted by both"):
        TableLabelMapper("clash", table)


def test_two_categories_that_normalise_alike_are_refused():
    """``tv_monitor`` and ``TV Monitor`` are one object typed twice."""
    table = {
        "tv_monitor": LabelSpec("tv", ("tv",)),
        "TV Monitor": LabelSpec("monitor", ("monitor",)),
    }
    with pytest.raises(ObjNavError, match="'tv monitor'"):
        TableLabelMapper("twice", table)


@pytest.mark.parametrize("context", [
    ["sink", "   "],
    ["sink", "__"],
    ["sink", 7],
    "sink",
    {"sink", "oven"},
    None,
])
def test_a_malformed_context_vocabulary_is_refused(context):
    """Blanks, a bare string's letters and set order never reach the detector."""
    with pytest.raises(ObjNavError):
        hm3d_like(context=context)


@pytest.mark.parametrize("name, table", [
    ("", {"bed": LabelSpec("bed", ("bed",))}),
    ("   ", {"bed": LabelSpec("bed", ("bed",))}),
    (None, {"bed": LabelSpec("bed", ("bed",))}),
    ("t", {}),
    ("t", [("bed", LabelSpec("bed", ("bed",)))]),
    ("t", {"": LabelSpec("bed", ("bed",))}),
    ("t", {"_ ": LabelSpec("bed", ("bed",))}),
    ("t", {3: LabelSpec("bed", ("bed",))}),
    ("t", {"bed": ("bed", ("bed",))}),
])
def test_a_malformed_table_is_refused(name, table):
    """A table that cannot be read unambiguously must fail before any episode."""
    with pytest.raises(ObjNavError):
        TableLabelMapper(name, table)
