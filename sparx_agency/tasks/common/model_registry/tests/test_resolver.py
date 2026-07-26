"""Resolver tests. Local/legacy resolution uses the real committed manifest
(this is the registry's acceptance gate: find the DA3 engine already on disk,
no network). Download uses a FakeStore so no test ever touches the network."""
import hashlib
import platform
from pathlib import Path

import pytest

from sparx_agency.tasks.common.model_registry.download.base import ArtifactStore
from sparx_agency.tasks.common.model_registry.download.verify import atomic_replace
from sparx_agency.tasks.common.model_registry.errors import ArtifactMissingError
from sparx_agency.tasks.common.model_registry.resolver import resolve

_LEGACY_DA3_ENGINE = Path(
    "~/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/"
    "DA3METRIC-LARGE/DA3METRIC-LARGE_fp16_546x364.engine").expanduser()


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Every test gets its own cache root and no inherited search path."""
    monkeypatch.setenv("SPARX_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("SPARX_MODEL_PATH", raising=False)
    monkeypatch.delenv("SPARX_MODEL_STRICT", raising=False)
    monkeypatch.delenv("SPARX_MODEL_BUCKET", raising=False)


@pytest.mark.skipif(not _LEGACY_DA3_ENGINE.exists(),
                    reason="DA3 engine not present on this machine")
def test_resolves_existing_da3_engine_via_legacy_path_no_network():
    artifact = resolve("da3_metric_large", role="depth_only", precision="fp16",
                       resolution="546x364", allow_download=False, allow_build=False)
    assert artifact.origin == "legacy"
    assert artifact.path == _LEGACY_DA3_ENGINE


def test_missing_role_raises_value_error_for_multi_role_model():
    manifest = {"models": {"m": {"roles": ["a", "b"], "variants": []}}}
    with pytest.raises(ValueError):
        resolve("m", manifest=manifest, allow_download=False, allow_build=False)


def test_no_artifact_anywhere_raises_with_reasons():
    manifest = {"models": {"m": {"roles": ["only"], "variants": []}}}
    with pytest.raises(ArtifactMissingError) as exc_info:
        resolve("m", role="only", manifest=manifest, allow_download=False, allow_build=False)
    msg = str(exc_info.value)
    assert "Download skipped" in msg
    assert "Build skipped" in msg


class _FakeStore(ArtifactStore):
    scheme = "mem"

    def __init__(self, content: bytes):
        self.content = content

    def exists(self, uri: str) -> bool:
        return True

    def download(self, uri: str, dest: Path) -> Path:
        tmp = dest.with_name(dest.name + ".part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(self.content)
        return atomic_replace(tmp, dest)


def test_downloads_from_fake_store_when_no_local_or_legacy_copy():
    content = b"pretend engine bytes"
    sha = hashlib.sha256(content).hexdigest()
    arch = platform.machine()
    uri = "mem://bucket/engine"
    manifest = {
        "models": {
            "m": {
                "roles": ["r"],
                "variants": [{
                    "precision": "fp16", "height": None, "width": None, "role": "r",
                    "engine_sources": [uri],
                    "artifacts": {"test_tag": {"sha256": sha, "bytes": len(content),
                                              "trt_version": "10.0.0", "arch": arch}},
                    "legacy_paths": [],
                }],
            },
        },
    }
    artifact = resolve("m", role="r", precision="fp16", target_tag="test_tag",
                       manifest=manifest, allow_download=True, allow_build=False,
                       stores={uri: _FakeStore(content)})
    assert artifact.origin == "download"
    assert artifact.path.read_bytes() == content
    assert artifact.sidecar["engine_sha256"] == sha


def test_download_disabled_falls_through_to_missing_error():
    manifest = {"models": {"m": {"roles": ["r"], "variants": [{
        "precision": "fp16", "height": None, "width": None, "role": "r",
        "engine_sources": ["mem://x"], "artifacts": {"test_tag": {"arch": platform.machine()}},
        "legacy_paths": [],
    }]}}}
    with pytest.raises(ArtifactMissingError) as exc_info:
        resolve("m", role="r", precision="fp16", target_tag="test_tag",
                manifest=manifest, allow_download=False, allow_build=False)
    assert "download disabled" in str(exc_info.value)
