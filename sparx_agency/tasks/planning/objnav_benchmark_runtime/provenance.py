"""Simulator-independent source/selection provenance and configuration locks.

Adapters supply published episode identities and protocol/data/model metadata.
A lock proves configuration consistency, never lack of earlier tuning leakage.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark.results_io import strict_json


SCHEMA = "objnav_configuration_lock/1"


def source_fingerprint(package_root=None):
    """Hash actual Python sources, including dirty and untracked implementation."""
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parents[3]
    if not root.is_dir():
        raise ValueError("Source root must be a directory")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def select_episodes(ids, limit=None, shards=1, shard_index=0):
    """Preserve publisher order; shard before applying an explicit subset limit."""
    ids = tuple(ids)
    if not ids or any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Episode identities must be nonempty, unique strings")
    if type(shards) is not int or type(shard_index) is not int or shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("Need integer shards >= 1 and 0 <= shard_index < shards")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    selected = ids[shard_index::shards]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Selection contains no episodes")
    return selected


def _configuration(config):
    value = json.loads(strict_json(config, "evaluation configuration"))
    if not isinstance(value, dict):
        raise ValueError("Evaluation configuration must be a mapping")
    ids = value.get("selected_episode_ids", [])
    if list(select_episodes(ids)) != ids:
        raise ValueError("Configuration must contain its exact ordered episode selection")
    return value


def freeze_configuration(config, path, *, development_note):
    """Create a non-overwriting exact-config lock with an explicit tuning disclosure."""
    if not isinstance(development_note, str) or not development_note.strip():
        raise ValueError("Record prior development/tuning exposure explicitly")
    payload = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
               "configuration": _configuration(config), "development_note": development_note,
               "held_out_claim": False}
    with Path(path).expanduser().open("x", encoding="utf-8") as stream:
        stream.write(strict_json(payload, "configuration lock", indent=2) + "\n")
    return payload


def check_frozen_configuration(config, path, *, expected_episode_ids=None):
    """Reject drift; optionally require a publisher's full ordered selection.

    No benchmark name, scene list, episode count or success rule is hardcoded.
    An adapter making a full-split claim must supply the publisher's identities.
    """
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("Not an ObjectNav configuration lock")
    current = _configuration(config)
    if expected_episode_ids is not None:
        expected = list(select_episodes(expected_episode_ids))
        if current["selected_episode_ids"] != expected:
            raise ValueError("Selection differs from the required published episode identities")
    frozen = payload["configuration"]
    differences = sorted(k for k in set(current) | set(frozen) if current.get(k) != frozen.get(k)
                         or (k in current) != (k in frozen))
    if differences:
        raise ValueError("Frozen configuration changed: " + ", ".join(differences))
    return payload

