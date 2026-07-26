"""Manifest-loading tests. Exercises the real committed manifest plus
synthetic ones for edge cases -- no filesystem writes, no network."""
import json

import pytest

from sparx_agency.tasks.common.model_registry import manifest as manifest_mod
from sparx_agency.tasks.common.model_registry.errors import ManifestError


def test_default_manifest_loads_and_has_da3():
    data = manifest_mod.load_manifest()
    entry = manifest_mod.get_model(data, "da3_metric_large")
    assert entry["roles"] == ["depth_only"]
    assert entry["resolution_multiple"] == 14


def test_unknown_model_raises():
    data = manifest_mod.load_manifest()
    with pytest.raises(ManifestError):
        manifest_mod.get_model(data, "no_such_model")


def test_missing_manifest_file_raises(tmp_path):
    with pytest.raises(ManifestError):
        manifest_mod.load_manifest(tmp_path / "does_not_exist.json")


def test_malformed_manifest_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(ManifestError):
        manifest_mod.load_manifest(bad)


def test_find_variant_matches_exact_key():
    data = manifest_mod.load_manifest()
    entry = manifest_mod.get_model(data, "da3_metric_large")
    variant = manifest_mod.find_variant(entry, precision="fp16", height=546, width=364,
                                        role="depth_only")
    assert variant is not None
    assert variant["legacy_paths"]


def test_find_variant_no_match_returns_none():
    data = manifest_mod.load_manifest()
    entry = manifest_mod.get_model(data, "da3_metric_large")
    assert manifest_mod.find_variant(entry, precision="fp16", height=1, width=1,
                                     role="depth_only") is None


def test_legacy_paths_expands_home():
    variant = {"legacy_paths": ["~/foo/bar.engine"]}
    paths = manifest_mod.legacy_paths(variant)
    assert len(paths) == 1
    assert "~" not in str(paths[0])


def test_resolve_engine_uris_skips_when_bucket_env_unset(monkeypatch):
    monkeypatch.delenv("SPARX_MODEL_BUCKET", raising=False)
    data = json.loads(json.dumps({
        "defaults": {"stores": {"primary": {"bucket_env": "SPARX_MODEL_BUCKET", "prefix": "p"}}},
    }))
    variant = {"engine_sources": ["s3://{bucket}/{prefix}/{target_tag}/{stem}.engine"]}
    uris = manifest_mod.resolve_engine_uris(data, variant, target_tag="orin_sm87", stem="x")
    assert uris == []


def test_resolve_engine_uris_formats_when_bucket_env_set(monkeypatch):
    monkeypatch.setenv("SPARX_MODEL_BUCKET", "my-bucket")
    data = {"defaults": {"stores": {"primary": {"bucket_env": "SPARX_MODEL_BUCKET", "prefix": "p"}}}}
    variant = {"engine_sources": ["s3://{bucket}/{prefix}/{target_tag}/{stem}.engine"]}
    uris = manifest_mod.resolve_engine_uris(data, variant, target_tag="orin_sm87", stem="x")
    assert uris == ["s3://my-bucket/p/orin_sm87/x.engine"]
