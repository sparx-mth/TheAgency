"""Cache-root resolution tests: env override, creation, and the
inside-the-repo guardrail. No network, no GPU."""
import pytest

from sparx_agency.tasks.common.model_registry.errors import ModelRegistryError
from sparx_agency.tasks.common.model_registry.paths import (
    REPO_ROOT, cache_root, search_path_dirs,
)


def test_cache_root_override_is_created(tmp_path):
    override = tmp_path / "cache"
    root = cache_root(override)
    assert root == override
    assert root.is_dir()


def test_cache_root_rejects_path_inside_repo():
    with pytest.raises(ModelRegistryError):
        cache_root(REPO_ROOT / "some" / "cache")


def test_search_path_dirs_empty_by_default(monkeypatch):
    monkeypatch.delenv("SPARX_MODEL_PATH", raising=False)
    assert search_path_dirs() == []


def test_search_path_dirs_splits_colon_separated(monkeypatch, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("SPARX_MODEL_PATH", f"{a}:{b}")
    assert search_path_dirs() == [a, b]
