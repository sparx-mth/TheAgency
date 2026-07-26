"""Answer "where is the engine for this model on this machine?"

Resolves a :class:`~sparx_agency.tasks.common.model_registry.key.ModelKey` to
a local file path by trying, in order: the local cache, extra search paths,
other tasks' committed engine directories, known legacy locations, a
download from the configured store, and finally an explicit build. It never
wraps inference -- two DA3 TensorRT wrappers already exist in ``core.mapping``
-- this module only resolves paths.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sparx_agency.tasks.common.hardware.detect import detect as detect_hardware
from sparx_agency.tasks.common.model_registry import manifest as manifest_mod
from sparx_agency.tasks.common.model_registry.download import factory as store_factory
from sparx_agency.tasks.common.model_registry.download import verify as verify_mod
from sparx_agency.tasks.common.model_registry.errors import ArtifactMissingError
from sparx_agency.tasks.common.model_registry.key import ModelKey, parse_resolution
from sparx_agency.tasks.common.model_registry.paths import REPO_ROOT
from sparx_agency.tasks.common.model_registry.paths import cache_root as resolve_cache_root
from sparx_agency.tasks.common.model_registry.paths import search_path_dirs
from sparx_agency.tasks.common.model_registry.sidecar import (
    build_sidecar, read_sidecar, validate_sidecar, write_sidecar,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedArtifact:
    """The result of a successful :func:`resolve` call."""

    path: Path
    key: ModelKey
    origin: str  # "local" | "legacy" | "download" | "build"
    sidecar: Optional[dict]
    target_tag: str

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "")


def _strict() -> bool:
    return _env_bool("SPARX_MODEL_STRICT", False)


def _search_local(key: ModelKey, root: Path, manifest_data: dict, profile
                  ) -> "tuple[Optional[ResolvedArtifact], list]":
    """Look in the cache, SPARX_MODEL_PATH, and other tasks' engine trees."""
    searched = []
    candidates = [root] + search_path_dirs() + manifest_mod.search_roots(manifest_data, REPO_ROOT)
    for base in candidates:
        candidate = base / key.relpath()
        searched.append(candidate)
        if candidate.exists():
            sidecar = read_sidecar(candidate)
            validate_sidecar(sidecar, profile=profile, strict=_strict())
            return ResolvedArtifact(candidate, key, "local", sidecar, key.target_tag), searched
    return None, searched


def _search_legacy(key: ModelKey, variant: Optional[dict]) -> "tuple[Optional[Path], list]":
    if not variant:
        return None, []
    for legacy in manifest_mod.legacy_paths(variant):
        if legacy.exists():
            return legacy, [legacy]
    return None, manifest_mod.legacy_paths(variant)


def _accept_legacy(legacy: Path, key: ModelKey, profile) -> ResolvedArtifact:
    sidecar = read_sidecar(legacy)
    if sidecar is None:
        if _strict():
            raise ArtifactMissingError(
                f"{legacy} exists but has no sidecar and SPARX_MODEL_STRICT=1")
        logger.warning("using legacy artifact with no version-lock sidecar: %s", legacy)
    else:
        validate_sidecar(sidecar, profile=profile, strict=_strict())
    return ResolvedArtifact(legacy, key, "legacy", sidecar, key.target_tag)


def _try_download(key: ModelKey, variant: Optional[dict], manifest_data: dict, profile,
                   root: Path, allow_download: Optional[bool], stores: Optional[dict]
                  ) -> "tuple[Optional[ResolvedArtifact], Optional[str]]":
    allowed = allow_download if allow_download is not None \
        else _env_bool("SPARX_MODEL_ALLOW_DOWNLOAD", True)
    if not allowed:
        return None, "download disabled"
    if not variant:
        return None, "model has no registered variant for this key"

    record = (variant.get("artifacts") or {}).get(key.target_tag)
    if record is None:
        return None, f"no published artifact for target_tag={key.target_tag!r}"
    if record.get("arch") and record["arch"] != profile.arch:
        return None, f"published artifact is arch={record['arch']}, this machine is {profile.arch}"

    size_mb = record.get("bytes", 0) / (1 << 20)
    cap_mb = manifest_mod.max_download_mb(manifest_data)
    if size_mb > cap_mb:
        return None, f"{size_mb:.0f} MB exceeds SPARX_MODEL_MAX_DOWNLOAD_MB={cap_mb}"

    uris = manifest_mod.resolve_engine_uris(manifest_data, variant,
                                           target_tag=key.target_tag, stem=key.stem())
    if not uris:
        return None, "no store configured (bucket env var unset) or no engine_sources"

    dest = root / key.relpath()
    last_err = None
    with verify_mod.locked(dest):
        if dest.exists():  # another process finished the download while we waited
            return ResolvedArtifact(dest, key, "download", read_sidecar(dest),
                                    key.target_tag), None
        for uri in uris:
            store = (stores or {}).get(uri) or store_factory.store_for(uri)
            try:
                store.download(uri, dest)
                verify_mod.verify(dest, expected_sha256=record.get("sha256"),
                                  expected_bytes=record.get("bytes"))
            except Exception as e:  # noqa: BLE001 -- try the next source
                last_err = e
                continue
            sidecar = build_sidecar(key, profile, origin="downloaded",
                                    trt_version=record.get("trt_version"),
                                    engine_sha256=record.get("sha256"))
            write_sidecar(dest, sidecar)
            return ResolvedArtifact(dest, key, "download", sidecar, key.target_tag), None
    return None, f"all download sources failed: {last_err}"


def _try_build(key: ModelKey, model_entry: dict, profile, root: Path,
                cache_root_override) -> "tuple[Optional[ResolvedArtifact], Optional[str]]":
    if profile.sm is None:
        return None, ("no CUDA-capable GPU detected on this machine -- use "
                      "DepthAnythingV2DepthModel for CPU/no-GPU inference instead")
    from sparx_agency.tasks.common.model_registry.build import onnx_source, trt_build
    try:
        onnx_path = onnx_source.ensure_onnx(key, model_entry, cache_root_override)
        dest = root / key.relpath()
        trt_build.build(key, onnx_path, dest, profile)
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    return ResolvedArtifact(dest, key, "build", read_sidecar(dest), key.target_tag), None


def resolve(model_id: str, *, precision: str = "fp16", resolution=None, role: str = None,
            device: str = "gpu", target_tag: str = None, allow_download: bool = None,
            allow_build: bool = False, manifest=None, cache_root=None,
            stores: dict = None) -> ResolvedArtifact:
    """Resolve one model artifact. Raises :class:`ArtifactMissingError` if it
    cannot be found locally, downloaded, or (if ``allow_build``) built."""
    profile = detect_hardware()
    tag = target_tag or profile.target_tag

    manifest_data = manifest if isinstance(manifest, dict) else manifest_mod.load_manifest(manifest)
    model_entry = manifest_mod.get_model(manifest_data, model_id)

    roles = model_entry.get("roles") or [None]
    if role is None:
        if len(roles) != 1:
            raise ValueError(f"model {model_id!r} has multiple roles {roles} -- pass role= explicitly")
        role = roles[0]

    h, w = parse_resolution(resolution) if resolution is not None else (None, None)
    mult = model_entry.get("resolution_multiple")
    if mult and resolution is not None and (h % mult or w % mult):
        raise ValueError(f"{model_id} resolution must be a multiple of {mult}, got {h}x{w}")

    key = ModelKey(model_id=model_id, precision=precision, height=h, width=w,
                  role=role, device=device, target_tag=tag)
    root = resolve_cache_root(cache_root)
    variant = manifest_mod.find_variant(model_entry, precision=precision, height=h, width=w, role=role)

    found, searched = _search_local(key, root, manifest_data, profile)
    if found:
        return found

    legacy, legacy_searched = _search_legacy(key, variant)
    searched += legacy_searched
    if legacy:
        return _accept_legacy(legacy, key, profile)

    downloaded, download_reason = _try_download(key, variant, manifest_data, profile,
                                                root, allow_download, stores)
    if downloaded:
        return downloaded

    if allow_build:
        built, build_reason = _try_build(key, model_entry, profile, root, cache_root)
        if built:
            return built
    else:
        build_reason = "allow_build=False (call ensure() or pass allow_build=True to build explicitly)"

    raise ArtifactMissingError(
        f"could not resolve {key.stem()} for target_tag={tag!r} "
        f"(sm={profile.sm}, arch={profile.arch}).\n"
        f"Searched: {[str(s) for s in searched]}\n"
        f"Download skipped: {download_reason}\n"
        f"Build skipped: {build_reason}")


def ensure(model_id: str, **kwargs) -> ResolvedArtifact:
    """``resolve()`` with building allowed -- the explicit, opt-in path."""
    kwargs.setdefault("allow_build", True)
    return resolve(model_id, **kwargs)
