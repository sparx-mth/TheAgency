"""Localization: lean timestamp/transform tooling + concrete providers.

The concrete providers (AprilTag/OpticalFlow/AMCL) are imported LAZILY via PEP
562 ``__getattr__`` so that importing a lean submodule of this package -- e.g.
``temporal_transform_buffer`` (used by the FALCON ROS1 ``mapping_sync`` node) or
``dead_reckoning_noise`` (used by ``falcon_adapter``) -- does NOT pull in the
providers and their heavy/optional dependencies (e.g. ``pupil_apriltags``, which
is not installed in the FALCON adapter container).

The public names still resolve on first access, so
``from sparx_agency.core.localization import AprilTagLocalizationProvider``
continues to work for code that actually needs a provider.
"""
from .base import BaseLocalizationProvider, LocalizationEstimate

__all__ = [
    "BaseLocalizationProvider",
    "LocalizationEstimate",
    "AprilTagLocalizationProvider",
    "OpticalFlowLocalizationProvider",
    "AmclLocalizationProvider",
]

_LAZY_PROVIDERS = frozenset((
    "AprilTagLocalizationProvider",
    "OpticalFlowLocalizationProvider",
    "AmclLocalizationProvider",
))


def __getattr__(name):
    """Resolve provider names on first access (PEP 562); keep submodule imports
    free of the provider dependency chain."""
    if name in _LAZY_PROVIDERS:
        from . import providers
        return getattr(providers, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
