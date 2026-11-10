"""The fixed single-storey evaluation set: the rules that admit an episode, and a draw that is the same every time."""
from __future__ import annotations

import json

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import single_storey_set as sets


def survey_row(episode, category="chair", *, kind="floor", start_level="0", goal_level="0", rise="0.0", drop="0.0",
               goal_dz="0.0"):
    return {"episode": episode, "category": category, "start_kind": kind, "start_level": start_level,
            "goal_level": goal_level, "path_max_rise_m": rise, "path_max_drop_m": drop, "goal_dz_m": goal_dz}


def test_the_rules_admit_only_floor_starts_whose_goal_and_path_stay_on_the_storey():
    assert sets.qualifies(survey_row("Z6MFQCViBuw/000001"))
    assert not sets.qualifies(survey_row("x8F5xyUWy9e/000001")), "a split-level scene is out whatever the episode"
    assert not sets.qualifies(survey_row("Z6MFQCViBuw/000002", kind="stairs"))
    assert not sets.qualifies(survey_row("Z6MFQCViBuw/000003", kind="platform"))
    assert not sets.qualifies(survey_row("Z6MFQCViBuw/000004", goal_level="1"))
    assert not sets.qualifies(survey_row("Z6MFQCViBuw/000005", rise="0.45")), "the path climbs a step"
    assert not sets.qualifies(survey_row("Z6MFQCViBuw/000006", drop="0.3"))
    assert not sets.qualifies(survey_row("Z6MFQCViBuw/000007", goal_dz="-0.25"))
    assert not sets.qualifies(survey_row("Z6MFQCViBuw/000008", goal_level="")), "no path to any view point"
    assert sets.qualifies(survey_row("x8F5xyUWy9e/000009"), excluded_scenes=()), "the exclusion list is a rule, not a constant"


def test_the_draw_is_seeded_uniform_over_the_pool_and_sorted():
    rows = [survey_row("sceneA/%06d" % i, "table" if i % 2 else "bed") for i in range(40)]
    rows += [survey_row("sceneB/%06d" % i, "sofa") for i in range(10)]
    rows += [survey_row("sceneB/%06d" % i, "sofa", kind="stairs") for i in range(10, 30)]      # not in the pool
    first = sets.select(rows, n=12, seed=3)
    again = sets.select(rows, n=12, seed=3)
    assert first == again, "the same seed draws the same set"
    assert first["episodes"] == sorted(first["episodes"]) and len(set(first["episodes"])) == 12
    assert first["pool"]["episodes"] == 50 and first["pool"]["by_scene"] == {"sceneA": 40, "sceneB": 10}
    assert sum(first["selected"]["by_scene"].values()) == 12
    assert sets.select(rows, n=12, seed=4)["episodes"] != first["episodes"], "another seed is another draw"
    with pytest.raises(ValueError, match="pool holds 50"):
        sets.select(rows, n=51)


def test_the_shipped_set_is_what_its_rules_and_seed_draw_from_the_shipped_survey():
    """The list every comparison is made on must be reproducible from the survey, or it is just a list."""
    meta = json.loads(sets.DEFAULT_SET.with_suffix(".meta.json").read_text())
    rows = sets.read_levels(sets.DEFAULT_LEVELS)
    result = sets.select(rows, n=meta["sample"]["n"], seed=meta["sample"]["seed"],
                         excluded_scenes=tuple(meta["rules"]["excluded_scenes"]),
                         max_path_dz_m=meta["rules"]["max_path_dz_m"], max_goal_dz_m=meta["rules"]["max_goal_dz_m"])
    assert result["episodes"] == json.loads(sets.DEFAULT_SET.read_text())
    assert result["selected"] == meta["selected"] and result["pool"] == meta["pool"]
    assert len(result["episodes"]) == 180
    assert not any(e.split("/")[0] in sets.SPLIT_LEVEL_SCENES for e in result["episodes"])
