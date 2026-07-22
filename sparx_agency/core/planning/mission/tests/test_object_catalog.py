"""Unit tests for the ROS-free object catalog loader.

Covers loading the objects JSON list into :class:`ObjectGoal` records (label
normalisation, position parsing, unknown-key tolerance), the access helpers
(``labels`` / ``unique_labels`` / ``by_label`` with duplicate labels, indexing /
iteration), reproducible random selection, and the ``ValueError`` raised on every
malformed shape rather than a silent default.

The loader is pure, so each test drives it with an in-memory JSON string (and one
round-trips the real ``objects.json`` shipped next to the FALCON task).
"""
import json
import random
from pathlib import Path

import pytest

from sparx_agency.core.planning.mission.object_catalog import (
    ObjectCatalog,
    ObjectGoal,
)

# The catalog shipped with the FALCON task (…/sparx_agency/tasks/planning/falcon).
_REPO_OBJECTS = (Path(__file__).resolve().parents[4]
                 / "tasks" / "planning" / "falcon" / "objects.json")


def _entry(label, x, y, z=0.0, **extra):
    d = {"label": label, "position_m": {"x": x, "y": y, "z": z}}
    d.update(extra)
    return d


def _catalog(*entries):
    return ObjectCatalog.from_json(json.dumps(list(entries)))


# ── loading ───────────────────────────────────────────────────────────
def test_loads_entries_in_order():
    cat = _catalog(_entry("refrigerator", -0.98, -4.12, 0.48),
                   _entry("table", -1.23, -1.44, 0.24))
    assert len(cat) == 2
    assert cat[0] == ObjectGoal("refrigerator", -0.98, -4.12, 0.48)
    assert cat[1].label == "table"


def test_labels_are_normalised():
    cat = _catalog(_entry("  Refrigerator ", 1.0, 2.0))
    assert cat[0].label == "refrigerator"


def test_position_is_coerced_to_float():
    cat = _catalog(_entry("box", 1, 2, 3))          # ints in JSON
    obj = cat[0]
    assert (obj.x, obj.y, obj.z) == (1.0, 2.0, 3.0)
    assert isinstance(obj.x, float)


def test_extra_keys_are_ignored():
    # objects.json carries frame_idx / tag_ids / tag_confidence -- must be tolerated.
    cat = _catalog(_entry("bag", 0.5, -1.4, 0.2, frame_idx=29,
                          tag_ids=[4, 7], tag_confidence=0.265))
    assert cat[0] == ObjectGoal("bag", 0.5, -1.4, 0.2)


def test_iteration_yields_all_objects():
    cat = _catalog(_entry("a", 0, 0), _entry("b", 1, 1), _entry("c", 2, 2))
    assert [o.label for o in cat] == ["a", "b", "c"]


# ── access helpers ─────────────────────────────────────────────────────
def test_labels_keeps_duplicates_unique_labels_dedupes():
    cat = _catalog(_entry("chair", 0.3, -4.7), _entry("chair", 1.5, -2.7),
                   _entry("table", -1.2, -1.4))
    assert cat.labels() == ["chair", "chair", "table"]
    assert cat.unique_labels() == ["chair", "table"]


def test_by_label_returns_every_match():
    cat = _catalog(_entry("chair", 0.3, -4.7), _entry("chair", 1.5, -2.7),
                   _entry("table", -1.2, -1.4))
    chairs = cat.by_label("CHAIR")             # matching is normalised
    assert len(chairs) == 2
    assert {(round(c.x, 1), round(c.y, 1)) for c in chairs} == {(0.3, -4.7), (1.5, -2.7)}
    assert cat.by_label("sofa") == []


def test_caption_is_label_and_xy():
    assert ObjectGoal("chair", 1.551, -2.672, 0.5).caption() == "chair  (1.55, -2.67)"


# ── random selection ───────────────────────────────────────────────────
def test_random_is_reproducible_with_a_seeded_rng():
    cat = _catalog(_entry("a", 0, 0), _entry("b", 1, 1), _entry("c", 2, 2))
    a = cat.random(random.Random(1234))
    b = cat.random(random.Random(1234))
    assert a == b
    assert a in list(cat)


def test_random_on_empty_catalog_raises():
    with pytest.raises(IndexError):
        _catalog().random()


# ── malformed input ────────────────────────────────────────────────────
def test_non_json_raises():
    with pytest.raises(ValueError):
        ObjectCatalog.from_json("{not json")


def test_top_level_must_be_a_list():
    with pytest.raises(ValueError):
        ObjectCatalog.from_json('{"label": "x", "position_m": {"x": 0, "y": 0, "z": 0}}')


def test_entry_must_be_object():
    with pytest.raises(ValueError):
        ObjectCatalog.from_json('["chair"]')


def test_missing_label_raises():
    with pytest.raises(ValueError):
        ObjectCatalog.from_json('[{"position_m": {"x": 0, "y": 0, "z": 0}}]')


def test_missing_position_raises():
    with pytest.raises(ValueError):
        ObjectCatalog.from_json('[{"label": "chair"}]')


def test_missing_position_component_raises():
    with pytest.raises(ValueError):
        ObjectCatalog.from_json('[{"label": "chair", "position_m": {"x": 0, "y": 0}}]')


def test_non_numeric_position_raises():
    with pytest.raises(ValueError):
        ObjectCatalog.from_json(
            '[{"label": "chair", "position_m": {"x": "left", "y": 0, "z": 0}}]')


# ── the real shipped catalog loads ─────────────────────────────────────
@pytest.mark.skipif(not _REPO_OBJECTS.exists(),
                    reason="shipped objects.json not found next to the FALCON task")
def test_shipped_objects_json_loads():
    cat = ObjectCatalog.from_json_file(_REPO_OBJECTS)
    assert len(cat) >= 1
    assert "refrigerator" in cat.unique_labels()
    # objects.json has two 'chair' rows at different positions.
    assert len(cat.by_label("chair")) == 2
    for obj in cat:
        assert obj.label == obj.label.strip().lower()
