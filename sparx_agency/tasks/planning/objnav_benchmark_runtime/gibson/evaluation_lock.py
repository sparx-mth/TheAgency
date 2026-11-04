"""Explicit development/frozen-validation provenance, not a promise against overfitting."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import SCENES

LOCK_KEYS = ("protocol", "runtime", "source_sha256", "method", "seed", "kinematics", "dataset")


def freeze_configuration(config, path):
    """Create a non-overwriting lock before evaluating the full validation split.

    This records a commitment to configuration, not proof that validation has
    never influenced development. The five previously inspected scene starts
    remain contaminated development examples and must be disclosed.
    """
    path = Path(path).expanduser()
    payload = {"schema": "gibson_configuration_lock/1",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "configuration": {key: config[key] for key in LOCK_KEYS},
               "previous_validation_development": True}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    return payload


def check_frozen_configuration(config, path):
    """Reject configuration drift and subset results labelled as frozen full validation."""
    payload = json.loads(Path(path).expanduser().read_text())
    if payload.get("schema") != "gibson_configuration_lock/1":
        raise ValueError("Not a Gibson configuration lock")
    if not config.get("full_split") or len(config.get("selected_episode_ids", [])) != 1000:
        raise ValueError("Frozen validation requires the complete published 1,000-episode split")
    expected_ids = {"%s/%06d" % (scene, index) for scene in SCENES for index in range(200)}
    if set(config["selected_episode_ids"]) != expected_ids:
        raise ValueError("Frozen validation requires the exact published episode identities")
    expected = payload["configuration"]
    current = json.loads(json.dumps({key: config.get(key) for key in LOCK_KEYS}, allow_nan=False))
    differences = [key for key in LOCK_KEYS if current.get(key) != expected.get(key)]
    if differences:
        raise ValueError("Frozen configuration changed: " + ", ".join(differences))
    return payload


