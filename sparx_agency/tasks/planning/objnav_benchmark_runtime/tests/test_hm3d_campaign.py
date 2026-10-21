"""The per-scene campaign, and the two ways it could lie about what it ran.

A campaign is an orchestrator, so its failure modes are not arithmetic: they are
saying a sweep succeeded when most of it crashed, and calling a handful of smoke
episodes a complete benchmark. Both would be read off ``campaign_summary.json``
and the exit status by whatever runs it next.

No simulator and no scene meshes: every child process is a fake, and the episode
rows are written by the harness's own logger.
"""
from __future__ import annotations

import json
import types

import pytest

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d import campaign, run
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import PUBLISHED_VAL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import HM3D_V2
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_hm3d_report import (
    record, run_config)

SCENES = ("00800-TEEsavR23oF", "00802-wcojb4TFT35")

EPISODES = "~/datasets/objectnav/hm3d/v2/val"


def write_scene(root, scene, episodes, *, finish=True):
    """One child's results directory, as ``run.py`` would leave it."""
    rows = [record("%s/%06d" % (scene, index), scene=scene, distance=0.05)
            for index in range(episodes)]
    directory = root / scene
    config = run_config(rows, full_split=False, scene_count=1)
    with MetricsLogger(directory, config, argv=["pytest"]) as logger:
        for row in rows:
            logger.log(row)
        if finish:
            logger.finish(summarise(rows))
    if not finish:
        (directory / "summary.json").unlink(missing_ok=True)
    return directory


def fake_runner(returncodes):
    """A child process that records its command and returns a scripted status."""
    seen = []

    def runner(command):
        seen.append(command)
        return types.SimpleNamespace(returncode=returncodes[len(seen) - 1])

    runner.seen = seen
    return runner


# -- the child commands ----------------------------------------------------

def test_every_child_command_is_one_run_py_actually_accepts():
    """The campaign builds argv by hand, so nothing but round-tripping it
    through ``run.py``'s own parser proves a flag was not renamed underneath."""
    command = campaign.scene_command(SCENES[0], "/tmp/out", version="v2",
                                     split="train", episodes_dir="E",
                                     scenes_dir="S", limit=3,
                                     extra=["--episodes-per-scene", "2"])
    parsed = run.parser().parse_args(command[command.index("--version"):])
    assert parsed.version == "v2" and parsed.split == "train"
    assert parsed.scenes == [SCENES[0]] and parsed.limit == 3
    assert parsed.episodes_per_scene == 2 and str(parsed.output) == "/tmp/out"


def test_the_split_reaches_the_children_and_the_scene_listing(tmp_path):
    """A development campaign reads its scene list from the development split's
    own episode files, so the split cannot be left to the pass-through args."""
    runner = fake_runner([0, 0])
    campaign.run_campaign(tmp_path, version="v2", split="train",
                          episodes_dir="E", scenes_dir="S", limit=1,
                          scenes=list(SCENES), runner=runner)
    for command in runner.seen:
        assert command[command.index("--split") + 1] == "train"
    assert json.loads((tmp_path / "campaign.json").read_text())["split"] == "train"


@pytest.mark.skipif(not (__import__("pathlib").Path(EPISODES).expanduser()
                         / "val.json.gz").is_file(),
                    reason="the published v2 validation episodes are not installed")
def test_listing_scenes_neither_parses_the_whole_split_nor_demands_a_complete_one():
    """Listing 145 train scene names by parsing seven million rows would cost a
    hundred seconds and several gigabytes, and asking the loader to certify a
    complete split would fail outright on any split but ``val``."""
    keys = campaign.scene_keys(EPISODES, HM3D_V2)
    assert len(keys) == PUBLISHED_VAL["v2"]["scenes"]
    assert all(key[:5].isdigit() and "-" in key for key in keys)
    # A development split lists its scenes just as readily.
    assert campaign.scene_keys(EPISODES, HM3D_V2.with_split("val")) == keys


# -- what the aggregate is allowed to call complete ------------------------

def test_a_scene_that_produced_one_episode_does_not_make_a_split_complete(tmp_path):
    """Every scene reporting a row is not every episode having been run. A smoke
    sweep, or a campaign where each scene died after one episode, would
    otherwise be written out as a complete benchmark with a headline SR."""
    for scene in SCENES:
        write_scene(tmp_path, scene, 1)
    report = campaign.aggregate(tmp_path, episodes_dir=EPISODES, version="v2")
    assert report["episodes"] == 2
    assert report["complete"] is False
    assert report["episodes_published"] == PUBLISHED_VAL["v2"]["episodes"]
    assert report["scenes_published"] == PUBLISHED_VAL["v2"]["scenes"]


def test_an_unfinished_scene_is_named_rather_than_averaged_away(tmp_path):
    """A scene killed mid-run still has rows; counting them silently would put a
    partial scene's score into the total with nothing saying so."""
    write_scene(tmp_path, SCENES[0], 2)
    write_scene(tmp_path, SCENES[1], 1, finish=False)
    report = campaign.aggregate(tmp_path, episodes_dir=EPISODES, version="v2")
    assert report["scenes_unfinished"] == [SCENES[1]]
    assert report["complete"] is False
    assert report["episodes"] == 3


def test_a_scene_with_no_rows_is_reported_missing_not_dropped(tmp_path):
    """A child that died before its first episode leaves an empty directory;
    leaving it out of the denominator would flatter every average."""
    write_scene(tmp_path, SCENES[0], 2)
    (tmp_path / SCENES[1]).mkdir()
    report = campaign.aggregate(tmp_path, episodes_dir=EPISODES, version="v2")
    assert report["scenes_without_rows"] == [SCENES[1]]
    assert report["scenes_with_rows"] == 1
    assert report["complete"] is False


def test_nothing_at_all_aggregates_to_nothing(tmp_path):
    """An aborted sweep must not produce a summary file to be misread later."""
    assert campaign.aggregate(tmp_path, episodes_dir=EPISODES, version="v2") is None
    assert not (tmp_path / "campaign_summary.json").exists()


# -- the exit status -------------------------------------------------------

def test_a_sweep_whose_scenes_crashed_does_not_exit_zero(tmp_path):
    """Whatever runs the campaign next reads the exit status, and a smoke run
    with ``--limit`` used to suppress every failure it had just printed.

    The crashed child here leaves no directory at all, which a scan of the
    results tree cannot see -- only the campaign manifest records that the
    scene was ever asked for.
    """
    def runner(command):
        scene = command[command.index("--scenes") + 1]
        if scene == SCENES[0]:
            write_scene(tmp_path, scene, 1)
            return types.SimpleNamespace(returncode=0)
        return types.SimpleNamespace(returncode=1)  # died before writing anything

    campaign.run_campaign(tmp_path, version="v2", episodes_dir=EPISODES,
                          scenes_dir="S", limit=1, scenes=list(SCENES),
                          runner=runner)
    status = campaign.main([str(tmp_path), "--aggregate-only", "--version", "v2",
                            "--episodes-dir", EPISODES, "--scenes-dir", "S"])
    assert status == 1
    report = json.loads((tmp_path / "campaign_summary.json").read_text())
    assert report["scenes_without_rows"] == [SCENES[1]]
    assert report["complete"] is False


def test_a_deliberate_subset_that_worked_exits_zero(tmp_path, monkeypatch):
    """A one-scene campaign is complete for what it asked for; failing it would
    make the exit status useless for exactly the runs people do most."""
    write_scene(tmp_path, SCENES[0], 1)
    monkeypatch.setattr(campaign, "run_campaign",
                        lambda *a, **k: [{"scene": SCENES[0], "returncode": 0,
                                          "status": "ran", "output": "x"}])
    status = campaign.main([str(tmp_path), "--version", "v2",
                            "--episodes-dir", EPISODES, "--scenes-dir", "S",
                            "--scenes", SCENES[0], "--limit", "1"])
    assert status == 0


# -- the geodesic self-test ------------------------------------------------

def test_shortest_paths_that_disagree_with_the_publisher_stop_the_run():
    """``info.geodesic_distance`` is the same quantity as l, so a disagreement
    is the wrong navmesh, not a difference of definition -- and every SPL would
    be wrong with it. It has to be an issue, not a number in the manifest."""
    good = {"episodes": 100, "within_1mm": 100, "median_abs_m": 0.0,
            "max_abs_m": 1e-9}
    assert run.geodesic_agreement_issue(good) is None
    assert run.geodesic_agreement_issue(None) is None

    bad = {"episodes": 100, "within_1mm": 3, "median_abs_m": 0.41,
           "max_abs_m": 2.2}
    complaint = run.geodesic_agreement_issue(bad)
    assert complaint is not None
    assert "3 of 100" in complaint and "navmesh" in complaint
    assert run.geodesic_agreement_issue(bad, allow=True) is None


def test_a_few_starts_may_sit_exactly_on_a_view_point():
    """Float32 and navmesh snapping put a start that is already on a view point
    a hair away from the publisher's zero, so the gate is not 100%."""
    nearly = {"episodes": 1000, "within_1mm": 995, "median_abs_m": 0.0,
              "max_abs_m": 4e-3}
    assert run.geodesic_agreement_issue(nearly) is None
    assert run.GEODESIC_AGREEMENT_FRACTION < 1.0
