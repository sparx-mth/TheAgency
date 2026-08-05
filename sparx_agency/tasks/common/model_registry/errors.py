"""Exceptions raised by the model registry.

Kept as flat, purpose-named classes so a caller can catch exactly the failure
mode it cares about (e.g. retry on :class:`DownloadError` but not on
:class:`IncompatibleArtifactError`), per the project rule of raising errors
rather than silently falling back to a default.
"""
from __future__ import annotations


class ModelRegistryError(Exception):
    """Base class for every error this package raises."""


class ManifestError(ModelRegistryError):
    """The registry manifest is missing, unreadable, or references an unknown model."""


class ArtifactMissingError(ModelRegistryError):
    """No local, downloadable, or buildable artifact satisfies a requested key."""


class IncompatibleArtifactError(ModelRegistryError):
    """A found artifact exists but was built for a different GPU/arch/TRT build."""


class DownloadError(ModelRegistryError):
    """Fetching an artifact from a remote store failed or produced a bad file."""


class BuildError(ModelRegistryError):
    """Building an engine from ONNX failed, or the ONNX itself is unavailable."""
