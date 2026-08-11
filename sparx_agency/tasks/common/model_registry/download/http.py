"""Stdlib-only HTTP(S) artifact store.

Deliberately dependency-free so it works even in the FALCON container's bare
Python 3.8: it covers presigned S3/GCS URLs and any plain file server without
needing that store's SDK.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from sparx_agency.tasks.common.model_registry.download.base import ArtifactStore
from sparx_agency.tasks.common.model_registry.download.verify import atomic_replace
from sparx_agency.tasks.common.model_registry.errors import DownloadError


class HttpArtifactStore(ArtifactStore):
    scheme = "https"

    def exists(self, uri: str) -> bool:
        try:
            req = urllib.request.Request(uri, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def download(self, uri: str, dest: Path) -> Path:
        tmp = dest.with_name(dest.name + ".part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(uri, timeout=30) as resp, open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        except (urllib.error.URLError, OSError) as e:
            tmp.unlink(missing_ok=True)
            raise DownloadError(f"failed to download {uri}: {e}") from e
        return atomic_replace(tmp, dest)
