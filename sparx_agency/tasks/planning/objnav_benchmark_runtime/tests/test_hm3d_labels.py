"""HM3D's six goal words, and what the perception stack is allowed to call them.

Every way this table can be wrong is quiet until it has already cost episodes.
A respelled category raises at episode one, because the mapper looks categories
up verbatim rather than guessing. A synonym that belongs to two rows is a false
STOP for whichever of the two an episode asks for. A prompt missing from the
vocabulary is never asked of the detector at all -- the HM3D run wires the
detector service with :meth:`vocabulary` and nothing else, so a dropped door
prompt makes the door finder silently never fire.

The expected spellings below are written out by hand on purpose: importing the
table's own constants to compare against would assert nothing.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

import sparx_agency
from sparx_agency.core.planning.objnav.errors import UnknownCategoryError
from sparx_agency.core.planning.objnav.labels.datasets.hm3d import (
    CATEGORIES,
    hm3d_label_mapper,
)
from sparx_agency.core.planning.objnav.labels.registry import (
    default_label_mapper_registry,
)
from sparx_agency.core.planning.objnav.labels.table_mapper import TableLabelMapper
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import (
    DOOR_LABELS,
)

REPO_ROOT = pathlib.Path(sparx_agency.__file__).resolve().parents[1]

#: HM3D's ``category_to_task_category_id``, as a tuple: a goal row's numeric
#: index is this tuple's index, so the order is data, not presentation.
PUBLISHED_ORDER = ("chair", "bed", "plant", "toilet", "tv_monitor", "sofa")

#: Per category: the LLM query, the detector prompts in order, and every
#: detector label that ends an episode.
EXPECTED = {
    "chair": ("chair", ("chair",), {"chair"}),
    "bed": ("bed", ("bed",), {"bed"}),
    "plant": ("potted plant", ("potted plant",), {"potted plant"}),
    "toilet": ("toilet", ("toilet",), {"toilet"}),
    "tv_monitor": ("tv", ("tv", "television"), {"tv", "television"}),
    "sofa": ("sofa", ("couch", "sofa"), {"couch", "sofa"}),
}


def mapper() -> TableLabelMapper:
    """A freshly built HM3D mapper."""
    return hm3d_label_mapper()


# -- the six categories ---------------------------------------------------

def test_the_categories_are_the_datasets_own_spellings_in_goal_index_order():
    """The spellings are the lookup keys and the order is the goal index.

    ``dataset.py`` rejects any ``object_category`` outside this tuple, and the
    mapper matches categories verbatim, so respelling one here (``"tv monitor"``,
    ``"couch"``) turns every episode of that category into a hard failure.
    """
    assert CATEGORIES == PUBLISHED_ORDER
    assert mapper().categories() == PUBLISHED_ORDER


@pytest.mark.parametrize("category", PUBLISHED_ORDER)
def test_every_category_the_dataset_can_publish_has_a_target(category):
    """An episode of any of the six must resolve before its first step."""
    target = mapper().target_labels(category)
    assert target.category == category


@pytest.mark.parametrize("respelling", [
    "tv monitor", "tv-monitor", "TV_Monitor", "couch", "potted plant", "sofa ",
])
def test_a_respelled_category_is_refused_rather_than_quietly_matched(respelling):
    """A near-miss spelling is an adapter bug; guessing it would hide the bug."""
    with pytest.raises(UnknownCategoryError):
        mapper().target_labels(respelling)


def test_a_near_miss_category_message_names_the_spelling_that_would_work():
    """The fix is one word, so the error must say which word."""
    with pytest.raises(UnknownCategoryError) as caught:
        mapper().target_labels("TV Monitor")
    assert "'tv_monitor'" in str(caught.value)


# -- what each target hands the perception stack --------------------------

@pytest.mark.parametrize("category", PUBLISHED_ORDER)
def test_each_query_is_the_phrase_an_llm_is_asked_about(category):
    """Room probabilities come from an LLM, which reads ``tv_monitor`` as noise."""
    assert mapper().target_labels(category).query == EXPECTED[category][0]


@pytest.mark.parametrize("category", PUBLISHED_ORDER)
def test_each_categorys_prompts_are_an_ordered_tuple_primary_first(category):
    """The detector service is configured with this list in this exact order.

    A set here would reorder between a run and the resume that continues it,
    and the service's health check compares the class list tuple-for-tuple.
    """
    prompts = mapper().target_labels(category).detector_prompts
    assert isinstance(prompts, tuple)
    assert prompts == EXPECTED[category][1]


@pytest.mark.parametrize("category", PUBLISHED_ORDER)
def test_each_categorys_accept_set_is_exactly_what_may_end_an_episode(category):
    """Accepting is STOPping: one extra synonym is a failed episode, not recall."""
    assert set(mapper().target_labels(category).accept_labels) == EXPECTED[category][2]


@pytest.mark.parametrize("category", PUBLISHED_ORDER)
def test_every_prompt_is_also_accepted(category):
    """Asking the detector for a label we then ignore wastes the only detection."""
    target = mapper().target_labels(category)
    assert all(target.accepts(prompt) for prompt in target.detector_prompts)


@pytest.mark.parametrize("category", PUBLISHED_ORDER)
def test_nothing_is_accepted_that_the_detector_is_never_asked_for(category):
    """An accept-only synonym is unreachable here, and worse than useless.

    ``vocabulary()`` deliberately omits accept-only labels, ``HttpDetector`` is
    configured with exactly that tuple, and ``methods/perception.py`` raises on
    any class outside it. So a label accepted but never prompted for can never
    arrive -- and a detector that did emit one would abort the episode instead
    of stopping on it. Whatever this table wants credit for has to be a prompt.
    """
    built = mapper()
    target = built.target_labels(category)
    assert set(target.accept_labels) == set(target.detector_prompts)
    assert set(target.accept_labels) <= set(built.vocabulary())


def test_accepting_is_case_and_separator_insensitive_but_never_fuzzy():
    """Detector spellings vary; ``sofa bed`` is still not a sofa."""
    sofa = mapper().target_labels("sofa")
    assert sofa.accepts("Couch") and sofa.accepts("SOFA")
    assert not sofa.accepts("sofa bed") and not sofa.accepts("chair")


def test_two_builds_of_the_table_prompt_the_detector_identically():
    """A resume re-imports the table; a different order is a different service."""
    assert mapper().vocabulary() == mapper().vocabulary()
    for category in PUBLISHED_ORDER:
        assert (mapper().target_labels(category).detector_prompts
                == mapper().target_labels(category).detector_prompts)


# -- no label belongs to two rows -----------------------------------------

def test_the_table_builds_at_all():
    """Construction is where an ambiguous or duplicated row is refused."""
    assert isinstance(mapper(), TableLabelMapper)


def test_no_detector_label_counts_as_two_categories():
    """A shared label makes one category's target a false STOP for the other."""
    built = mapper()
    owners = {}
    for category in PUBLISHED_ORDER:
        for label in built.target_labels(category).accept_labels:
            assert owners.setdefault(label, category) == category, label


@pytest.mark.parametrize("label,owner", [
    ("couch", "sofa"),
    ("sofa", "sofa"),
    ("tv", "tv_monitor"),
    ("television", "tv_monitor"),
])
def test_the_seating_and_screen_synonyms_belong_to_one_row_each(label, owner):
    """``couch`` and ``tv`` are the two words a second row would plausibly claim."""
    built = mapper()
    for category in PUBLISHED_ORDER:
        assert built.target_labels(category).accepts(label) == (category == owner)


# -- the vocabulary the detector is configured with -----------------------

def test_the_vocabulary_opens_with_every_goal_prompt_in_table_order():
    """Whatever the context words are, the six goals must be askable."""
    built = mapper()
    goal_prompts = [prompt for category in PUBLISHED_ORDER
                    for prompt in built.target_labels(category).detector_prompts]
    assert built.vocabulary()[:len(goal_prompts)] == tuple(goal_prompts)


def test_the_vocabulary_repeats_nothing():
    """The service rejects a class list that disagrees with the one it was given."""
    vocabulary = mapper().vocabulary()
    assert len(set(vocabulary)) == len(vocabulary)


@pytest.mark.parametrize("door_prompt", sorted(DOOR_LABELS))
def test_the_vocabulary_carries_every_prompt_the_door_finder_looks_for(door_prompt):
    """``ObservedDoors`` drops any class outside ``DOOR_LABELS``.

    The detector only ever emits what the vocabulary asked for, so a door
    prompt missing here is a door finder that never fires and never complains.
    """
    assert door_prompt in mapper().vocabulary()


def test_the_vocabulary_is_all_normalised():
    """Detections are matched by exact membership after normalisation."""
    from sparx_agency.core.common.label_match import normalize_label

    assert all(word == normalize_label(word) for word in mapper().vocabulary())


# -- the registry entry ---------------------------------------------------

def test_the_default_registry_offers_hm3d():
    """``--dataset hm3d`` is how a run picks this table; the name is the key."""
    assert "hm3d" in default_label_mapper_registry().names()


def test_the_hm3d_description_names_the_dataset_a_person_is_choosing():
    """The description is read instead of the table when picking a benchmark."""
    assert "HM3D" in default_label_mapper_registry().get("hm3d").description


def test_creating_hm3d_returns_the_six_category_table_under_that_name():
    """The registry refuses a mapper whose own name is not its key.

    Round-tripping through it is what proves the ``"hm3d"`` line builds the
    HM3D table and not a copy-pasted neighbour's, which would otherwise file
    another dataset's per-category numbers under HM3D.
    """
    built = default_label_mapper_registry().create("hm3d")
    assert built.name == "hm3d"
    assert built.categories() == PUBLISHED_ORDER


def test_reading_the_hm3d_entry_imports_no_table_until_it_is_created():
    """Listing the benchmarks must not pay for every benchmark's table."""
    code = (
        "import sys\n"
        "from sparx_agency.core.planning.objnav.labels.registry import "
        "default_label_mapper_registry\n"
        "module = 'sparx_agency.core.planning.objnav.labels.datasets.hm3d'\n"
        "registry = default_label_mapper_registry()\n"
        "registry.names(), registry.get('hm3d').description\n"
        "print(module in sys.modules)\n"
        "registry.create('hm3d')\n"
        "print(module in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(REPO_ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "True"]
