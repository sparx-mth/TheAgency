"""The :class:`LabelMapper` contract: a dataset's categories, in our vocabulary.

One implementation per dataset vocabulary -- HM3D, MP3D, Gibson, RoboTHOR each
name their goals differently. A mapper is a table, not simulator code, so it
lives in ``core`` and is testable anywhere; the benchmark branches add theirs
under ``labels/datasets`` and register them by name.

An ABC because mappers are selected by name at runtime (the repo's rule for
swappable backends).

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import abc
from typing import Tuple


class LabelMapper(abc.ABC):
    """Translates one dataset's goal categories into :class:`TargetLabels`.

    Attributes:
        name: The registry key, e.g. ``"hm3d"``.
    """

    name = ""

    @abc.abstractmethod
    def categories(self) -> Tuple[str, ...]:
        """The dataset's goal categories, verbatim, in the dataset's order."""
        raise NotImplementedError

    @abc.abstractmethod
    def target_labels(self, category: str):
        """The goal ``category`` in our vision vocabulary.

        Args:
            category: A dataset category, verbatim.

        Returns:
            :class:`~sparx_agency.core.planning.objnav.types.TargetLabels`.

        Raises:
            UnknownCategoryError: ``category`` is not in :meth:`categories`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def vocabulary(self) -> Tuple[str, ...]:
        """Every detector prompt this mapper can ask for, normalised.

        The prompts of every category plus any context vocabulary (the
        objects that tell rooms apart), without repeats, in a stable order --
        what a detector must be able to name for this dataset.
        """
        raise NotImplementedError

    def covers(self, category: str) -> bool:
        """Whether ``category`` is one of this dataset's goals."""
        return category in self.categories()
