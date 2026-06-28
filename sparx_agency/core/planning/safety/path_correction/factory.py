"""Factory for path-correction strategies (the swap-by-name point).

A ROS node that corrects planner output picks its strategy by a single ``name``
string. To add a new strategy (e.g. an ESDF ridge follower): implement a sibling
:class:`PathCorrector`, then add one branch here -- the node does not change.

Python 3.8 compatible (the FALCON Noetic adapter imports core under 3.8).
"""
from __future__ import annotations

from typing import Optional

from .base import PathCorrector
from .esdf_corrector import EsdfCorrectorConfig, EsdfPathCorrector
from .potential_field_corrector import (
    PotentialFieldCorrectorConfig,
    PotentialFieldPathCorrector,
)

#: Names this factory can build (kept in sync with the branches below).
AVAILABLE_CORRECTORS = (PotentialFieldPathCorrector.name, EsdfPathCorrector.name)


def make_path_corrector(name: str, config: Optional[object] = None) -> PathCorrector:
    """Build a :class:`PathCorrector` by ``name``.

    Args:
        name: Strategy identifier (see :data:`AVAILABLE_CORRECTORS`).
        config: The strategy-specific config object: a
            :class:`PotentialFieldCorrectorConfig` for ``"potential_field"`` or an
            :class:`EsdfCorrectorConfig` for ``"esdf"`` (or None for defaults).

    Raises:
        ValueError: If ``name`` is not a known strategy (no silent fallback).
        TypeError: If ``config`` is the wrong type for the chosen strategy.
    """
    if name == PotentialFieldPathCorrector.name:
        if config is not None and not isinstance(config, PotentialFieldCorrectorConfig):
            raise TypeError(
                "potential_field corrector needs a PotentialFieldCorrectorConfig, "
                "got %r" % type(config).__name__)
        return PotentialFieldPathCorrector(config)
    if name == EsdfPathCorrector.name:
        if config is not None and not isinstance(config, EsdfCorrectorConfig):
            raise TypeError(
                "esdf corrector needs an EsdfCorrectorConfig, got %r"
                % type(config).__name__)
        return EsdfPathCorrector(config)
    raise ValueError(
        "unknown path corrector %r (available: %s)"
        % (name, ", ".join(AVAILABLE_CORRECTORS)))
