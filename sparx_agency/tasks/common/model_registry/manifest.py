"""Load and query ``configs/model_registry.json``.

The manifest is committed to git and lists *where artifacts come from*
(known-good local paths, download sources, checksums) -- never credentials,
and never the artifacts themselves (those are gitignored, per
``.gitignore``'s "downloaded/built on the target, never committed").
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from sparx_agency.tasks.common.model_registry.errors import ManifestError

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / "configs" / "model_registry.json"


def load_manifest(path: Optional[Path] = None) -> dict:
    """Load the manifest JSON from ``path`` (default: the committed one)."""
    p = Path(path) if path else DEFAULT_MANIFEST_PATH
    if not p.exists():
        raise ManifestError(f"model registry manifest not found: {p}")
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ManifestError(f"malformed manifest {p}: {e}") from e


def get_model(manifest: dict, model_id: str) -> dict:
    """Return the manifest entry for ``model_id``, or raise :class:`ManifestError`."""
    models = manifest.get("models", {})
    if model_id not in models:
        raise ManifestError(f"unknown model_id {model_id!r}; known: {sorted(models)}")
    return models[model_id]


def find_variant(model_entry: dict, *, precision: str, height, width, role) -> Optional[dict]:
    """Find the variant matching this exact precision/resolution/role, if any."""
    for variant in model_entry.get("variants", []):
        if (variant.get("precision") == precision
                and variant.get("height") == height
                and variant.get("width") == width
                and variant.get("role") == role):
            return variant
    return None


def search_roots(manifest: dict, repo_root: Path) -> List[Path]:
    """Other tasks' committed engine directories, treated as read-only search roots."""
    return [repo_root / r for r in manifest.get("search_roots", [])]


def legacy_paths(variant: dict) -> List[Path]:
    """Absolute paths (``~`` expanded) where this variant might already exist."""
    return [Path(p).expanduser() for p in variant.get("legacy_paths", [])]


def resolve_engine_uris(manifest: dict, variant: dict, *, target_tag: str, stem: str) -> List[str]:
    """Format ``engine_sources`` templates with the configured store + this key.

    Skips (rather than raises for) any source whose bucket env var isn't set,
    so a machine with no cloud credentials configured just finds no download
    sources instead of crashing -- the resolver treats that the same as "no
    prebuilt artifact available" and falls through to build/fail.
    """
    stores = manifest.get("defaults", {}).get("stores", {})
    primary = stores.get("primary", {})
    bucket = os.environ.get(primary.get("bucket_env", ""), "")
    prefix = primary.get("prefix", "")
    uris = []
    for template in variant.get("engine_sources", []):
        if "{bucket}" in template and not bucket:
            continue
        uris.append(template.format(bucket=bucket, prefix=prefix, target_tag=target_tag, stem=stem))
    return uris


def max_download_mb(manifest: dict) -> float:
    return manifest.get("defaults", {}).get("max_download_mb", 2048)
