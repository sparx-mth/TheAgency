"""The launcher of a paired CPU check's arm: what a scene process is run with, and when it is started."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import cpu_stub_campaign
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.cpu_stub_campaign import (
    CHILD_ENV, DEFAULT_SCENE_GIB, HIDE_DRI, RUN_FLAGS, SCENE_GIB, STUB, missing_rows, scene_jobs, selected_ids, shards,
    tree_revision)

PLAN = {"pLe4wQe7qrG": ["pLe4wQe7qrG/000001", "pLe4wQe7qrG/000002", "pLe4wQe7qrG/000003"],
        "Z6MFQCViBuw": ["Z6MFQCViBuw/000090"]}


def jobs_for(tmp_path, **overrides):
    options = dict(tree=tmp_path / "tree", python=Path("/opt/habitat/bin/python"), episodes_dir=Path("/data/episodes"),
                   scenes_dir=Path("/data/scenes"))
    options.update(overrides)
    return scene_jobs(PLAN, tmp_path / "arm", **options)


def test_a_scene_process_runs_the_check_of_2026_11_15_and_nothing_else(tmp_path):
    """The flags and the environment are the check's own, never changed between arms: an arm run today is an arm of
    the same check only if the stand-in, the renderer, the perception, the explorer, the seed and the thread counts
    are the ones the stored arms were run with."""
    jobs, scene_of = jobs_for(tmp_path)
    assert [job.key for job in jobs] == ["pLe4wQe7qrG", "Z6MFQCViBuw"], "the longest list first"
    assert scene_of == {"pLe4wQe7qrG": "pLe4wQe7qrG", "Z6MFQCViBuw": "Z6MFQCViBuw"}
    job = jobs[0]
    command = job.command
    assert command[:len(HIDE_DRI)] == HIDE_DRI and "mount -t tmpfs none /dev/dri" in command[4], "no DRM device: llvmpipe"
    tail = command[len(HIDE_DRI):]
    assert tail[:6] == ["env", "--chdir=%s" % (tmp_path / "tree"), "/opt/habitat/bin/python", "-u", str(STUB), "mp3d"]
    assert STUB.name == "stub_llm.py" and STUB.is_file()
    assert tail[tail.index("--scenes") + 1] == "pLe4wQe7qrG"
    assert tail[tail.index("--output") + 1] == str(tmp_path / "arm" / "pLe4wQe7qrG")
    start = tail.index("--episode-ids") + 1
    assert tail[start:start + 3] == PLAN["pLe4wQe7qrG"]
    assert tuple(tail[start + 3:]) == RUN_FLAGS == ("--seed", "17", "--perception", "ground-truth", "--explorer",
                                                    "frontier", "--lean", "--software-render", "--validate-starts")
    assert job.env["PYTHONPATH"] == str(tmp_path / "tree") and job.env["CUDA_VISIBLE_DEVICES"] == ""
    assert {k: job.env[k] for k in CHILD_ENV} == CHILD_ENV == {
        "CUDA_VISIBLE_DEVICES": "", "LP_NUM_THREADS": "6", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    assert job.directory == tmp_path / "arm" / "pLe4wQe7qrG"
    assert job.log_path == tmp_path / "arm" / "pLe4wQe7qrG.console.log"


def test_extra_arguments_a_recording_and_shards_reach_the_command(tmp_path):
    jobs, scene_of = jobs_for(tmp_path, extra=["--policy-config", "/tmp/p.json"], record=True, shard=2)
    assert [job.key for job in jobs] == ["pLe4wQe7qrG/shard_0", "Z6MFQCViBuw", "pLe4wQe7qrG/shard_1"]
    assert scene_of["pLe4wQe7qrG/shard_1"] == "pLe4wQe7qrG"
    first = jobs[0].command
    assert first[-3:] == ["--record-log-only", "--policy-config", "/tmp/p.json"]
    assert first[first.index("--output") + 1] == str(tmp_path / "arm" / "pLe4wQe7qrG" / "shard_0")
    assert jobs[0].log_path.name == "pLe4wQe7qrG.shard_0.console.log"
    assert shards(list(range(5)), 2) == [[0, 1], [2, 3], [4]] and shards([1, 2], 0) == [[1, 2]] == shards([1, 2], 5)


def test_a_subset_is_named_by_id_or_by_scene_and_an_unknown_name_is_refused():
    ids = [episode_id for scene in PLAN.values() for episode_id in scene]
    assert selected_ids(ids, None) == ids and selected_ids(ids, []) == ids
    assert selected_ids(ids, ["Z6MFQCViBuw", "pLe4wQe7qrG/000002"]) == ["pLe4wQe7qrG/000002", "Z6MFQCViBuw/000090"]
    with pytest.raises(SystemExit, match="does not hold: nowhere/000001"):
        selected_ids(ids, ["nowhere/000001"])
    # What a scene's process is budgeted at when it is started (``ram_gate.RamGate``): the largest scene's by default.
    assert SCENE_GIB["Z6MFQCViBuw"] == 6.2 and DEFAULT_SCENE_GIB >= max(SCENE_GIB.values()) - 0.3


def test_missing_rows_and_the_trees_revision(tmp_path):
    arm = tmp_path / "arm"
    (arm / "pLe4wQe7qrG" / "shard_0").mkdir(parents=True)
    (arm / "pLe4wQe7qrG" / "shard_0" / "episodes.jsonl").write_text(
        json.dumps({"episode_id": "pLe4wQe7qrG/000001"}) + "\n\n" + json.dumps({"episode_id": "pLe4wQe7qrG/000003"}) + "\n")
    assert missing_rows(PLAN, arm) == {"pLe4wQe7qrG": ["pLe4wQe7qrG/000002"], "Z6MFQCViBuw": ["Z6MFQCViBuw/000090"]}
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "SNAPSHOT_COMMIT.txt").write_text("f257a0ce\n")
    assert tree_revision(snapshot) == {"commit": "f257a0ce", "snapshot": True, "changes_bytes": None}
    (snapshot / "SNAPSHOT_CHANGES.diff").write_text("diff --git a/x b/x\n")
    assert tree_revision(snapshot)["changes_bytes"] == 19
    revision = tree_revision(cpu_stub_campaign.REPO)
    assert revision["snapshot"] is False and (revision["commit"] is None or len(revision["commit"]) == 40)
    assert tree_revision(tmp_path / "nowhere")["commit"] is None
