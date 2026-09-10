"""Name -> label-mapper factory, so a benchmark picks its dataset table by name.

Copies the ``Factory`` + ``Registry`` + ``default_*_registry()`` idiom of
:mod:`sparx_agency.core.planning.trackers.registry` and
:mod:`sparx_agency.core.planning.vlas.registry`, including the VLA registry's
habit of importing the implementation *inside* the factory closure: listing
the available mappers must not import, let alone build, every benchmark's
table.

Two things are checked at :meth:`LabelMapperRegistry.create` rather than
trusted, because each would otherwise file results under the wrong dataset
without failing anything:

* the factory returned a :class:`LabelMapper` at all;
* the mapper's own ``name`` is the key it was created under. A copy-pasted
  factory line that builds the MP3D table under ``"hm3d"`` looks fine until
  the per-category numbers are wrong.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.interfaces.label_mapper import LabelMapper


@dataclass(frozen=True)
class LabelMapperFactory:
    """A named, lazily-built label mapper.

    Attributes:
        name: The registry key, e.g. ``"hm3d"``. The mapper it builds must
            carry the same ``name``.
        create: Zero-argument callable returning the :class:`LabelMapper`. It
            imports its dataset module inside the callable, not at module
            scope.
        description: One line for a person choosing a mapper, e.g.
            ``"HM3D ObjectNav, 6 categories"``.

    Raises:
        ObjNavError: On a blank name, a ``create`` that is not callable, or a
            description that is not a string.
    """

    name: str
    create: Callable[[], LabelMapper]
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ObjNavError(
                "LabelMapperFactory.name must be a non-blank string, got %r"
                % (self.name,))
        if not callable(self.create):
            raise ObjNavError(
                "LabelMapperFactory.create must be a zero-argument callable "
                "returning a LabelMapper, got %r" % (self.create,))
        if not isinstance(self.description, str):
            raise ObjNavError(
                "LabelMapperFactory.description must be a string, got %r"
                % (self.description,))


class LabelMapperRegistry:
    """Look up label mappers by name. Starts empty."""

    def __init__(self) -> None:
        self._factories: Dict[str, LabelMapperFactory] = {}

    def register(self, factory: LabelMapperFactory) -> None:
        """Add ``factory`` under its name. Does not call it.

        Args:
            factory: The mapper's factory.

        Raises:
            TypeError: ``factory`` is not a :class:`LabelMapperFactory`.
            ObjNavError: A mapper of that name is already registered (a
                :class:`ValueError`). Overwriting it silently would make which
                table a benchmark is scored against depend on registration
                order.
        """
        if not isinstance(factory, LabelMapperFactory):
            raise TypeError(
                "LabelMapperRegistry.register takes a LabelMapperFactory, got "
                "%r" % (factory,))
        if factory.name in self._factories:
            raise ObjNavError(
                "label mapper %r is already registered; pick another name or "
                "remove the duplicate registration" % (factory.name,))
        self._factories[factory.name] = factory

    def names(self) -> List[str]:
        """The registered names, sorted."""
        return sorted(self._factories)

    def get(self, name: str) -> LabelMapperFactory:
        """The factory registered under ``name``, e.g. to read its description.

        Raises:
            KeyError: No such mapper; the message lists what is available.
        """
        if name not in self:
            raise KeyError("Unknown label mapper %r. Available: %s"
                           % (name, ", ".join(self.names()) or "none registered"))
        return self._factories[name]

    def create(self, name: str) -> LabelMapper:
        """Build the mapper registered under ``name``. Calls its factory each time.

        Args:
            name: The registry key.

        Returns:
            A :class:`LabelMapper` whose ``name`` is ``name``.

        Raises:
            KeyError: No such mapper; the message lists what is available.
            ObjNavError: The factory returned something that is not a
                :class:`LabelMapper`, or a mapper carrying another name.
        """
        mapper = self.get(name).create()
        if not isinstance(mapper, LabelMapper):
            raise ObjNavError(
                "label mapper factory %r returned %r, which is not a "
                "LabelMapper" % (name, mapper))
        if mapper.name != name:
            raise ObjNavError(
                "label mapper factory %r built a mapper named %r; the registry "
                "key and the mapper's name must agree, or results are filed "
                "under the wrong dataset" % (name, mapper.name))
        return mapper

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._factories


def default_label_mapper_registry() -> LabelMapperRegistry:
    """The label mappers of the benchmarks this repo runs. Registers nothing yet.

    Each benchmark branch adds its dataset table as
    ``labels/datasets/<dataset>.py`` (a function returning a
    :class:`~sparx_agency.core.planning.objnav.labels.table_mapper.TableLabelMapper`)
    and one ``register(...)`` line here, with the factory importing its module
    lazily inside the closure::

        def _hm3d() -> LabelMapper:
            from sparx_agency.core.planning.objnav.labels.datasets.hm3d import (
                hm3d_label_mapper)
            return hm3d_label_mapper()

        registry.register(LabelMapperFactory(
            name="hm3d", create=_hm3d,
            description="HM3D ObjectNav, 6 categories"))

    Returns:
        A new registry, so registering on it never changes another caller's.
    """
    registry = LabelMapperRegistry()
    return registry
