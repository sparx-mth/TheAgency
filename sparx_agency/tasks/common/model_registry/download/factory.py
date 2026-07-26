"""Dispatch a URI to the :class:`ArtifactStore` that can fetch it."""
from __future__ import annotations

from urllib.parse import urlsplit

from sparx_agency.tasks.common.model_registry.errors import DownloadError


def store_for(uri: str):
    """Return an :class:`ArtifactStore` instance for ``uri``'s scheme."""
    scheme = urlsplit(uri).scheme
    if scheme in ("http", "https"):
        from sparx_agency.tasks.common.model_registry.download.http import HttpArtifactStore
        return HttpArtifactStore()
    if scheme == "s3":
        from sparx_agency.tasks.common.model_registry.download.s3 import S3ArtifactStore
        return S3ArtifactStore()
    raise DownloadError(f"no artifact store registered for scheme {scheme!r} (uri={uri})")
