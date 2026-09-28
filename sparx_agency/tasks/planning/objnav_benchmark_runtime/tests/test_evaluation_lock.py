"""Full-split provenance guards and upstream sentinel compatibility."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.evaluation_lock import freeze_configuration, check_frozen_configuration
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.distance import GibsonDistanceField
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import SCENES


def configuration():
    return {"protocol": {"name": "test"}, "runtime": {"numpy": "test"}, "source_sha256": "test-source",
            "method": {"setting": (1, 2)}, "seed": 0, "kinematics": {}, "dataset": {"hash": "data"},
            "full_split": True, "selected_episode_ids": ["%s/%06d" % (scene, i) for scene in SCENES for i in range(200)]}


def test_lock_accepts_same_json_configuration_and_refuses_drift(tmp_path):
    config = configuration()
    path = tmp_path / "lock.json"
    freeze_configuration(config, path)
    assert check_frozen_configuration(config, path)["previous_validation_development"]
    for key in ("method", "dataset", "source_sha256", "seed"):
        changed = dict(config, **{key: "changed"})
        with pytest.raises(ValueError, match="changed"):
            check_frozen_configuration(changed, path)
    with pytest.raises(FileExistsError):
        freeze_configuration(config, path)


def test_subset_cannot_be_called_frozen_full_validation(tmp_path):
    config = configuration()
    path = tmp_path / "lock.json"
    freeze_configuration(config, path)
    with pytest.raises(ValueError, match="complete"):
        check_frozen_configuration(dict(config, full_split=False), path)
    with pytest.raises(ValueError, match="identities"):
        check_frozen_configuration(dict(config, selected_episode_ids=["Collierville/000000"] * 1000), path)


def test_reference_sentinel_start_is_retained_not_silently_excluded():
    semantic = np.zeros((7, 80, 80), np.uint8)
    semantic[0, 5:25, 5:25] = 1
    semantic[0, 50:75, 50:75] = 1
    semantic[1, 10, 10] = 1
    field = GibsonDistanceField(semantic, (0, 0), 0)
    start = (3.0, 0, 3.0)
    assert field.start_uses_sentinel(start)
    assert np.isfinite(field.distance(start, start=True))
    assert field.distance(start, start=True) == field.distance(start)


