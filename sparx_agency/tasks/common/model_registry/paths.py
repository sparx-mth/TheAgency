"""Where the registry keeps its local cache, and how callers can extend the search.

The cache lives outside the repo on purpose: artifacts are 100s of MB, and the
repo gets bind-mounted read-only in places (e.g. FALCON's container), so a
cache rooted inside the repo would be unwritable there and would bloat every
checkout besides.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from sparx_agency.tasks.common.model_registry.errors import ModelRegistryError

# sparx_agency/tasks/common/model_registry/paths.py -> repo root is 4 levels up.
REPO_ROOT = Path(__file__).resolve().parents[4]

CACHE_ROOT_ENV = "SPARX_MODEL_CACHE"
SEARCH_PATH_ENV = "SPARX_MODEL_PATH"


def default_cache_root() -> Path:
    return Path.home() / ".cache" / "sparx_agency" / "models"


def cache_root(override: Optional[Path] = None) -> Path:
    """Resolve, create, and validate the writable local artifact cache.

    Raises if the resolved root would sit inside this repo's tree -- that
    tree can be mounted read-only (see FALCON's ``run_falcon.sh``), and these
    files don't belong in a git worktree regardless.
    """
    raw = override or os.environ.get(CACHE_ROOT_ENV)
    root = Path(raw).expanduser() if raw else default_cache_root()

    try:
        root.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise ModelRegistryError(
            f"{CACHE_ROOT_ENV}={root} resolves inside the repo ({REPO_ROOT}); "
            f"point it somewhere outside the checkout, e.g. ~/.cache/sparx_agency/models")

    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK):
        raise ModelRegistryError(f"model cache root is not writable: {root}")
    return root


def search_path_dirs() -> List[Path]:
    """Extra roots from ``SPARX_MODEL_PATH`` (colon-separated, PATH-style)."""
    raw = os.environ.get(SEARCH_PATH_ENV, "")
    return [Path(p).expanduser() for p in raw.split(":") if p]
