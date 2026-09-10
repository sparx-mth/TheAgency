"""The registry that picks a dataset's label table by name.

It is the one place a benchmark name turns into a table, so the tests pin the
mistakes that would score a run against the wrong dataset without failing:
a name registered twice, a factory that builds the wrong mapper, and a
default registry that imports every table just to list them.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

import sparx_agency
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.labels.registry import (
    LabelMapperFactory,
    LabelMapperRegistry,
    default_label_mapper_registry,
)
from sparx_agency.core.planning.objnav.labels.table_mapper import TableLabelMapper
from sparx_agency.core.planning.objnav.types.target import LabelSpec

REPO_ROOT = pathlib.Path(sparx_agency.__file__).resolve().parents[1]


def tiny_mapper(name="tiny"):
    """A one-row mapper named ``name``."""
    return TableLabelMapper(name, {"chair": LabelSpec("chair", ("chair",))})


def registry_with(*names):
    """A registry holding a tiny mapper under each of ``names``."""
    registry = LabelMapperRegistry()
    for name in names:
        registry.register(LabelMapperFactory(
            name=name, create=lambda name=name: tiny_mapper(name)))
    return registry


# -- lookups --------------------------------------------------------------

def test_a_registered_mapper_is_created_by_name():
    """The benchmark config names a table; the registry must build that one."""
    mapper = registry_with("tiny").create("tiny")
    assert isinstance(mapper, TableLabelMapper)
    assert mapper.name == "tiny"
    assert mapper.categories() == ("chair",)


def test_names_are_sorted_whatever_the_registration_order():
    """A CLI's list of choices must not depend on registration order."""
    assert registry_with("mp3d", "hm3d", "gibson").names() == [
        "gibson", "hm3d", "mp3d"]


def test_contains_answers_for_registered_names_only():
    """``in`` is how a CLI validates a choice; a non-string is simply absent."""
    registry = registry_with("hm3d")
    assert "hm3d" in registry
    assert "mp3d" not in registry
    assert ["hm3d"] not in registry


def test_registering_a_name_twice_is_refused():
    """A silent overwrite would make the scored table depend on import order."""
    registry = registry_with("hm3d")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(LabelMapperFactory(name="hm3d", create=tiny_mapper))


def test_an_unknown_name_raises_key_error_listing_what_is_available():
    """A typo in a config must say which names would have worked."""
    with pytest.raises(KeyError) as caught:
        registry_with("hm3d", "mp3d").create("hm3dv2")
    message = str(caught.value)
    assert "hm3dv2" in message
    assert "hm3d, mp3d" in message


def test_an_unknown_name_on_an_empty_registry_says_nothing_is_registered():
    """An empty listing must not read like a formatting bug."""
    with pytest.raises(KeyError, match="none registered"):
        LabelMapperRegistry().create("hm3d")


def test_get_returns_the_factory_with_its_description():
    """A person choosing a table reads its description without building it."""
    registry = LabelMapperRegistry()
    registry.register(LabelMapperFactory(
        name="tiny", create=tiny_mapper, description="one chair"))
    assert registry.get("tiny").description == "one chair"


# -- the factory ----------------------------------------------------------

def test_the_factory_runs_only_when_a_mapper_is_created():
    """Listing mappers must not build tables; each create builds a fresh one."""
    calls = []

    def build():
        calls.append(1)
        return tiny_mapper("counted")

    registry = LabelMapperRegistry()
    registry.register(LabelMapperFactory(name="counted", create=build))
    assert registry.names() == ["counted"] and calls == []
    first = registry.create("counted")
    second = registry.create("counted")
    assert len(calls) == 2
    assert first is not second


def test_a_factory_that_returns_something_else_is_refused():
    """A non-mapper would only fail later, inside the agent, far from the cause."""
    registry = LabelMapperRegistry()
    registry.register(LabelMapperFactory(name="odd", create=lambda: {"chair": 1}))
    with pytest.raises(ObjNavError, match="not a LabelMapper"):
        registry.create("odd")


def test_a_factory_whose_mapper_carries_another_name_is_refused():
    """A copy-pasted line building MP3D under ``hm3d`` must not score silently."""
    registry = LabelMapperRegistry()
    registry.register(LabelMapperFactory(
        name="hm3d", create=lambda: tiny_mapper("mp3d")))
    with pytest.raises(ObjNavError) as caught:
        registry.create("hm3d")
    assert "'hm3d'" in str(caught.value) and "'mp3d'" in str(caught.value)


@pytest.mark.parametrize("kwargs", [
    {"name": "", "create": tiny_mapper},
    {"name": "  ", "create": tiny_mapper},
    {"name": None, "create": tiny_mapper},
    {"name": "hm3d", "create": "tiny_mapper"},
    {"name": "hm3d", "create": tiny_mapper, "description": None},
])
def test_a_malformed_factory_is_refused(kwargs):
    """A factory is validated when it is written, not when it is first used."""
    with pytest.raises(ObjNavError):
        LabelMapperFactory(**kwargs)


def test_registering_something_that_is_not_a_factory_is_a_type_error():
    """A bare callable has no name to register it under."""
    with pytest.raises(TypeError):
        LabelMapperRegistry().register(tiny_mapper)


# -- the default registry -------------------------------------------------

def test_every_default_mapper_builds_under_its_own_name():
    """Each benchmark branch's registration line must build its own table."""
    registry = default_label_mapper_registry()
    assert isinstance(registry, LabelMapperRegistry)
    for name in registry.names():
        mapper = registry.create(name)
        assert mapper.name == name
        assert mapper.categories()


def test_each_default_registry_is_independent():
    """Registering on one caller's registry must not leak into another's."""
    first = default_label_mapper_registry()
    first.register(LabelMapperFactory(name="scratch", create=tiny_mapper))
    assert "scratch" not in default_label_mapper_registry()


def test_building_the_default_registry_imports_no_dataset_table():
    """The factories import their tables lazily, so listing stays free."""
    code = (
        "import sys\n"
        "from sparx_agency.core.planning.objnav.labels.registry import "
        "default_label_mapper_registry\n"
        "default_label_mapper_registry().names()\n"
        "prefix = 'sparx_agency.core.planning.objnav.labels.datasets.'\n"
        "print(sorted(m for m in sys.modules if m.startswith(prefix)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(REPO_ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
