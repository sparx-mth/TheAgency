"""Abstract interface for fetching a model artifact from a remote store.

A new backend (a different cloud provider, a lab NAS, ...) is one file
implementing this ABC plus one dispatch entry in :mod:`factory` -- nothing
else in the registry needs to change.
"""
from __future__ import annotations

import abc
from pathlib import Path


class ArtifactStore(abc.ABC):
    """One backend capable of fetching artifacts identified by URI."""

    scheme = ""

    @abc.abstractmethod
    def exists(self, uri: str) -> bool:
        """Return True if an artifact exists at ``uri``, without downloading it."""

    @abc.abstractmethod
    def download(self, uri: str, dest: Path) -> Path:
        """Download ``uri`` to ``dest`` (atomically) and return ``dest``."""

    def upload(self, src: Path, uri: str) -> None:
        """Upload ``src`` to ``uri``. Not every backend supports this."""
        raise NotImplementedError(f"{type(self).__name__} does not support upload")
