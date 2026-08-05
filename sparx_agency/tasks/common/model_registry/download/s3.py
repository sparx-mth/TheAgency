"""S3 (or any S3-compatible endpoint, e.g. a self-hosted MinIO) artifact store.

``boto3`` is a lazy import behind an actionable error, and lives in the
optional ``models`` poetry group -- not the default install -- so importing
this module costs nothing until a caller actually needs to fetch from s3://.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from sparx_agency.tasks.common.model_registry.download.base import ArtifactStore
from sparx_agency.tasks.common.model_registry.download.verify import atomic_replace
from sparx_agency.tasks.common.model_registry.errors import DownloadError


class S3ArtifactStore(ArtifactStore):
    scheme = "s3"

    def __init__(self, client=None):
        self._client = client

    def _boto3_client(self):
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError as e:
            raise DownloadError(
                "s3:// artifacts need boto3 -- install with "
                "`poetry install --with models`") from e
        self._client = boto3.client("s3")
        return self._client

    @staticmethod
    def _split(uri: str):
        parts = urlsplit(uri)
        return parts.netloc, parts.path.lstrip("/")

    def exists(self, uri: str) -> bool:
        bucket, key = self._split(uri)
        try:
            self._boto3_client().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:  # noqa: BLE001 -- any client/network error means "not found"
            return False

    def download(self, uri: str, dest: Path) -> Path:
        bucket, key = self._split(uri)
        tmp = dest.with_name(dest.name + ".part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._boto3_client().download_file(bucket, key, str(tmp))
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise DownloadError(f"failed to download {uri}: {e}") from e
        return atomic_replace(tmp, dest)
