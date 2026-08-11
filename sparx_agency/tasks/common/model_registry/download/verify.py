"""Checksum + atomic-write + locking helpers shared by every ArtifactStore.

A truncated TensorRT plan can partially deserialize, which is a worse failure
than a clean download error -- so every store writes to a ``.part`` file and
renames only after both a size and (when known) a sha256 check pass, under a
lock so two processes fetching the same artifact at once don't race.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
from pathlib import Path
from typing import Optional

from sparx_agency.tasks.common.model_registry.errors import DownloadError


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, *, expected_sha256: Optional[str] = None,
           expected_bytes: Optional[int] = None) -> None:
    if expected_bytes is not None:
        actual = path.stat().st_size
        if actual != expected_bytes:
            raise DownloadError(
                f"{path}: size {actual} != expected {expected_bytes} (truncated download?)")
    if expected_sha256 is not None:
        actual = sha256_of(path)
        if actual != expected_sha256:
            raise DownloadError(f"{path}: sha256 {actual} != expected {expected_sha256}")


def atomic_replace(tmp_path: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, dest)
    return dest


@contextlib.contextmanager
def locked(dest: Path):
    """Hold an exclusive lock on ``<dest>.lock`` for the duration of a fetch."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(dest) + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
