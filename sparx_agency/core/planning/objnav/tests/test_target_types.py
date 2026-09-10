"""``LabelSpec`` and ``TargetLabels`` accept a person's table row and refuse any target the detector and the agent could disagree about.

A blank query asks the LLM about nothing, a bare string becomes one prompt per
character, and a prompt the target does not accept makes the detector find the
object and the agent ignore it. None of those crash anything downstream --
they move SR and SPL -- so each documented refusal is exercised here, with the
error class the docstring names, and the exact-match acceptance rule too.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.target import LabelSpec, TargetLabels


def target(**overrides):
    """HM3D's ``tv_monitor``, already normalised, with any field replaced."""
    fields = dict(category="tv_monitor", query="tv",
                  detector_prompts=("tv", "television"),
                  accept_labels=frozenset({"tv", "television", "tv monitor"}))
    fields.update(overrides)
    return TargetLabels(**fields)


def test_a_label_spec_keeps_its_words_verbatim_as_tuples():
    """A table row is written by a person in plain words; the mapper normalises it."""
    spec = LabelSpec("TV", ["TV Monitor", "television"])
    assert spec.query == "TV"
    assert spec.prompts == ("TV Monitor", "television")
    assert spec.accept == ()


@pytest.mark.parametrize("query", ["", "   ", "_", None])
def test_a_label_spec_refuses_a_blank_query(query):
    """The LLM's room probabilities come from the query; a blank one asks about nothing."""
    with pytest.raises(ObjNavError, match="query"):
        LabelSpec(query, ("chair",))


@pytest.mark.parametrize("prompts,message", [
    ((), "at least one"), ("chair", "bare string"),
    (("chair", " "), "non-blank"), (("chair", 3), "non-blank"),
], ids=["none", "bare-string", "blank-entry", "non-string"])
def test_a_label_spec_refuses_bad_prompts(prompts, message):
    """A bare string would become one prompt per character."""
    with pytest.raises(ObjNavError, match=message):
        LabelSpec("chair", prompts)


@pytest.mark.parametrize("accept,message", [
    ("seat", "bare string"), (("seat", ""), "non-blank"),
], ids=["bare-string", "blank-entry"])
def test_a_label_spec_refuses_bad_accept_labels(accept, message):
    """A blank accept label would normalise to the empty string and match nothing useful."""
    with pytest.raises(ObjNavError, match=message):
        LabelSpec("chair", ("chair",), accept)


@pytest.mark.parametrize("build", [
    lambda: LabelSpec("tv", {"tv", "television"}),
    lambda: LabelSpec("tv", ("tv",), accept={"monitor", "screen"}),
    lambda: LabelSpec("tv", (p for p in ("tv", "television"))),
    lambda: target(detector_prompts={"tv", "television"}),
], ids=["prompts-set", "accept-set", "prompts-generator",
        "detector-prompts-set"])
def test_prompts_and_accept_labels_in_an_unordered_collection_are_refused(build):
    """A set's order changes with PYTHONHASHSEED: a run and its resume would hand the detector another primary prompt."""
    with pytest.raises(ObjNavError, match="list or tuple"):
        build()


def test_accept_labels_may_come_in_any_collection_and_are_stored_as_a_frozenset():
    """Their order means nothing: membership is all a target asks of them."""
    labels = target(accept_labels={"tv", "television", "tv monitor"})
    assert labels.accept_labels == frozenset({"tv", "television", "tv monitor"})


def test_target_labels_accept_exactly_after_normalisation():
    """``TV_Monitor`` counts; ``monitor`` and ``tv screen`` do not."""
    labels = target()
    assert labels.accepts("TV_Monitor")
    assert labels.accepts("  Television ")
    assert not labels.accepts("monitor")
    assert not labels.accepts("tv screen")


def test_target_labels_do_not_accept_by_shared_token():
    """The fuzzy rule accepts ``car`` for ``car keys``; on a benchmark that is a false STOP."""
    labels = target(category="car_keys", query="car keys",
                    detector_prompts=("car keys",), accept_labels=("car keys",))
    assert labels.accepts("Car_Keys")
    assert not labels.accepts("car")
    assert not labels.accepts("keys")


def test_target_labels_keep_the_category_verbatim_and_store_a_frozenset():
    """Results are reported under the dataset's own word, underscores and all."""
    labels = target(accept_labels=["tv", "television"])
    assert labels.category == "tv_monitor"
    assert labels.detector_prompts == ("tv", "television")
    assert isinstance(labels.accept_labels, frozenset)
    assert labels.accept_labels == frozenset({"tv", "television"})


@pytest.mark.parametrize("overrides,message", [
    (dict(query="TV"), "normalised"),
    (dict(detector_prompts=("TV",), accept_labels=("TV",)), "normalised"),
    (dict(accept_labels=("tv", "television", "Tv_Monitor")), "normalised"),
    (dict(detector_prompts=("tv", "tv")), "repeats"),
    (dict(detector_prompts=()), "empty"),
    (dict(accept_labels=("tv",)), "not accept"),
    (dict(category=" "), "category"),
    (dict(category=None), "category"),
], ids=["query-not-normalised", "prompt-not-normalised", "accept-not-normalised",
        "repeated-prompt", "no-prompt", "prompt-not-accepted", "blank-category",
        "no-category"])
def test_malformed_target_labels_are_refused(overrides, message):
    """A prompt the target does not accept makes the detector find it and the agent ignore it."""
    with pytest.raises(ObjNavError, match=message):
        target(**overrides)
