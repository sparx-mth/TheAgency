"""The MP3D per-scene campaign and the assets tooling, without a simulator or Matterport3D.

A campaign is an orchestrator, so its failure modes are not arithmetic: they
are saying a sweep succeeded when most of it crashed, and calling a handful of
smoke episodes a complete benchmark. The assets tool's are reading a wrong
archive as the published one and calling a partial installation ready.
"""
from __future__ import annotations

import gzip
import json
import types
import zipfile

import pytest

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d import scene_pool
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import assets, campaign, run
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import MP3D_V1, PUBLISHED_VAL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_mp3d import (
    OTHER_SCENE, SCENE, row, write_release)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_mp3d_evaluation import (
    record, run_config)

SCENES = (SCENE, OTHER_SCENE)


def write_scene(root, dataset, scene, episodes, *, finish=True):
    """One child's results directory, as ``run.py`` would leave it."""
    rows = [record("%s/%06d" % (scene, index), scene=scene) for index in range(episodes)]
    directory = root / scene
    config = run_config(dataset, method="rpt", selected=[r.episode_id for r in rows])
    with MetricsLogger(directory, config, argv=["pytest"]) as logger:
        for r in rows:
            logger.log(r)
        if finish:
            logger.finish(summarise(rows))
    if not finish:
        (directory / "summary.json").unlink(missing_ok=True)
    return directory


def fake_runner(returncodes):
    seen = []

    def runner(command):
        seen.append(command)
        return types.SimpleNamespace(returncode=returncodes[len(seen) - 1])

    runner.seen = seen
    return runner


@pytest.fixture
def release(tmp_path):
    """Two scenes with two episodes each, and a published expectation of exactly that."""
    return write_release(tmp_path, MP3D_V1, {SCENE: [row(), row("bed")], OTHER_SCENE: [row("sofa"), row()]})


@pytest.fixture
def small_published_split(monkeypatch):
    monkeypatch.setitem(PUBLISHED_VAL, "v1", {"scenes": 2, "episodes": 4})


# -- the child commands ----------------------------------------------------

def test_every_child_command_is_one_run_py_actually_accepts():
    command = campaign.scene_command(SCENE, "/tmp/out", version="v1", split="train",
                                     episodes_dir="E", scenes_dir="S", limit=3,
                                     extra=["--episodes-per-scene", "2", "--lean"])
    parsed = run.parser().parse_args(command[command.index("--version"):])
    assert parsed.version == "v1" and parsed.split == "train"
    assert parsed.scenes == [SCENE] and parsed.limit == 3 and parsed.lean
    assert parsed.episodes_per_scene == 2 and str(parsed.output) == "/tmp/out"
    explicit = campaign.scene_command(SCENE, "/tmp/out", version="v1", episodes_dir="E",
                                      scenes_dir="S", episode_ids=[SCENE + "/000003"])
    assert run.parser().parse_args(explicit[explicit.index("--version"):]).episode_ids == [SCENE + "/000003"]


def test_the_scenes_come_from_the_split_and_each_runs_in_its_own_process(tmp_path, release):
    runner = fake_runner([0, 0])
    results = campaign.run_campaign(tmp_path / "root", version="v1", episodes_dir=release.episodes_dir,
                                    scenes_dir=release.scenes_dir, runner=runner, limit=1)
    assert [r["scene"] for r in results] == list(SCENES)
    assert all(r["returncode"] == 0 for r in results)
    for command, scene in zip(runner.seen, SCENES):
        assert command[command.index("--scenes") + 1] == scene
        assert command[command.index("--output") + 1].endswith(scene)
        assert command[command.index("--limit") + 1] == "1"
    manifest = json.loads((tmp_path / "root" / "campaign.json").read_text())
    assert manifest["scenes"] == 2 and manifest["version"] == "v1"
    assert campaign.scene_keys(release.episodes_dir, MP3D_V1) == SCENES


def test_the_smoke_cover_is_one_episode_per_scene_with_the_categories_balanced(tmp_path, release):
    runner = fake_runner([0, 0])
    campaign.run_campaign(tmp_path / "root", version="v1", episodes_dir=release.episodes_dir,
                          scenes_dir=release.scenes_dir, runner=runner, smoke=True)
    plan = json.loads((tmp_path / "root" / "smoke_plan.json").read_text())
    chosen = [command[command.index("--episode-ids") + 1] for command in runner.seen]
    assert len(chosen) == 2 and chosen[0].startswith(SCENE) and chosen[1].startswith(OTHER_SCENE)
    assert plan["version"] == "v1" and plan["split"] == "val"
    with pytest.raises(ValueError):
        campaign.run_campaign(tmp_path / "x", version="v1", episodes_dir=release.episodes_dir,
                              scenes_dir=release.scenes_dir, runner=runner, smoke=True, limit=1)


def test_a_resumed_root_skips_finished_scenes_and_continues_started_ones(tmp_path, release):
    root = tmp_path / "root"
    write_scene(root, release, SCENE, 2)
    write_scene(root, release, OTHER_SCENE, 1, finish=False)
    runner = fake_runner([0])
    results = campaign.run_campaign(root, version="v1", episodes_dir=release.episodes_dir,
                                    scenes_dir=release.scenes_dir, runner=runner, resume=True,
                                    extra=["--lean", "--resume-accepting-changes", "kinematics"])
    assert results[0]["status"] == "already finished"
    assert len(runner.seen) == 1 and "--resume" in runner.seen[0]
    assert runner.seen[0][runner.seen[0].index("--resume-accepting-changes") + 1] == "kinematics"
    with pytest.raises(ValueError, match="resume"):
        campaign.run_campaign(tmp_path / "fresh", version="v1", episodes_dir=release.episodes_dir,
                              scenes_dir=release.scenes_dir, runner=fake_runner([0, 0]),
                              extra=["--resume-accepting-changes", "kinematics"])


# -- the stall watchdog of the concurrent pool -------------------------------

class FakeChild:
    """A ``Popen`` look-alike: alive until ``finish`` is called or it is killed."""

    def __init__(self):
        self.pid = 4242
        self.returncode = None
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = scene_pool.STALLED_RETURNCODE

    def finish(self, code=0):
        self.returncode = code


def test_a_scene_worker_that_stops_writing_is_killed_and_recorded_as_stalled(tmp_path, monkeypatch):
    """A worker that lost its main thread never exits; the pool kills it after ``stall_s`` and moves on."""
    monkeypatch.setattr(scene_pool.time, "sleep", lambda s: None)
    child = FakeChild()
    kills = []

    def killpg(pid, sig):
        kills.append((pid, sig))
        child.kill()

    monkeypatch.setattr(scene_pool.os, "killpg", killpg)
    base = __import__("time").time()                 # the console log's real mtime must read as "just now"
    clock = iter(base + 100.0 * k for k in range(10_000))     # each poll is 100 s later
    monkeypatch.setattr(scene_pool.time, "time", lambda: next(clock))
    lines = []
    codes = scene_pool.run_concurrently([(SCENE, ["run"])], tmp_path, 2, stall_s=250.0, announce=lines.append,
                                        popen=lambda *a, **k: child)
    assert codes == {SCENE: scene_pool.STALLED_RETURNCODE}
    assert kills == [(4242, scene_pool.signal.SIGKILL)]
    assert any("STALLED" in line for line in lines)


def test_a_scene_worker_whose_heartbeat_moves_is_left_alone(tmp_path, monkeypatch):
    """The heartbeat is the scene directory's live.json / progress.json / episodes.jsonl, or the console log."""
    monkeypatch.setattr(scene_pool.time, "sleep", lambda s: None)
    child = FakeChild()
    monkeypatch.setattr(scene_pool.os, "killpg", lambda pid, sig: pytest.fail("a live worker was killed"))
    now = [__import__("time").time()]
    monkeypatch.setattr(scene_pool.time, "time", lambda: now[0])
    scene_dir = tmp_path / SCENE
    scene_dir.mkdir()
    polls = []

    def poll():
        polls.append(now[0])
        now[0] += 100.0
        heartbeat = scene_dir / "live.json"
        heartbeat.write_text("{}")
        scene_pool.os.utime(heartbeat, (now[0], now[0]))       # written on every action
        if len(polls) == 6:
            child.finish(0)
        return child.returncode

    child.poll = poll
    codes = scene_pool.run_concurrently([(SCENE, ["run"])], tmp_path, 2, stall_s=250.0, announce=lambda line: None,
                                        popen=lambda *a, **k: child)
    assert codes == {SCENE: 0} and len(polls) == 6
    assert scene_pool.scene_heartbeat(tmp_path, SCENE) == pytest.approx(now[0])
    with pytest.raises(ValueError, match="stall_s"):
        scene_pool.run_concurrently([(SCENE, ["run"])], tmp_path, 2, stall_s=-1.0)


def test_the_campaign_forwards_the_stall_minutes_to_the_pool(tmp_path, release, monkeypatch):
    seen = {}

    def run_concurrently(pending, root, workers, *, runner, stall_s):
        seen["stall_s"] = stall_s
        return {scene: 0 for scene, _ in pending}

    monkeypatch.setattr(campaign, "run_concurrently", run_concurrently)
    campaign.run_campaign(tmp_path / "root", version="v1", episodes_dir=release.episodes_dir,
                          scenes_dir=release.scenes_dir, runner=fake_runner([0, 0]), jobs=2, stall_s=1800.0)
    assert seen["stall_s"] == 1800.0


# -- what the aggregate is allowed to call complete ------------------------

def test_a_sweep_whose_scenes_crashed_does_not_exit_zero(tmp_path, release, monkeypatch, capsys):
    monkeypatch.setattr(campaign, "subprocess", types.SimpleNamespace(run=fake_runner([1, 1])))
    write_scene(tmp_path / "root", release, SCENE, 1, finish=False)
    code = campaign.main([str(tmp_path / "root"), "--episodes-dir", str(release.episodes_dir),
                          "--scenes-dir", str(release.scenes_dir)])
    assert code == 1
    assert "did not exit cleanly" in capsys.readouterr().out


def test_every_scene_reporting_rows_is_not_a_complete_split(tmp_path, release, small_published_split):
    for scene in SCENES:
        write_scene(tmp_path / "root", release, scene, 1)
    report = campaign.aggregate(tmp_path / "root", episodes_dir=release.episodes_dir, version="v1")
    assert report["episodes"] == 2 and report["scenes_with_rows"] == 2
    assert report["episodes_published"] == 4 and report["complete"] is False
    for scene in SCENES:
        write_scene(tmp_path / "full", release, scene, 2)
    report = campaign.aggregate(tmp_path / "full", episodes_dir=release.episodes_dir, version="v1")
    assert report["complete"] is True and report["scenes_expected"] == 2
    assert (tmp_path / "full" / "campaign_metrics.csv").read_text().count("\n") == 3


def test_an_unfinished_or_missing_scene_is_named_rather_than_averaged_away(tmp_path, release, small_published_split):
    root = tmp_path / "root"
    write_scene(root, release, SCENE, 2)
    write_scene(root, release, OTHER_SCENE, 2, finish=False)
    (root / "campaign.json").write_text(json.dumps({"runs": [{"scene": s} for s in SCENES + ("gone",)]}))
    report = campaign.aggregate(root, episodes_dir=release.episodes_dir, version="v1")
    assert report["scenes_unfinished"] == [OTHER_SCENE]
    assert report["scenes_without_rows"] == ["gone"]
    assert report["complete"] is False


# -- the assets tool ---------------------------------------------------------

def test_the_installation_check_reports_counts_against_the_published_split(release, tmp_path):
    report = assets.check_installation(release.episodes_dir, release.scenes_dir, MP3D_V1)
    assert report["episodes_found"] == 4 and report["scenes_found"] == 2
    assert report["episodes_published"] == 2195 and report["scenes_published"] == 11
    assert report["episodes_ready"] is False and report["scenes_ready"] is False
    assert report["scenes_complete"] == 2 and report["missing_count"] == 0
    assert any("found 4 episodes in 2 scenes" in p for p in report["problems"])
    (release.scenes_dir / "mp3d" / SCENE / (SCENE + ".navmesh")).unlink()
    again = assets.check_installation(release.episodes_dir, release.scenes_dir, MP3D_V1)
    assert again["missing_count"] == 1 and again["missing"][0].endswith(SCENE + ".navmesh")
    missing_root = assets.check_installation(release.episodes_dir, tmp_path / "nowhere", MP3D_V1)
    assert missing_root["scenes_ready"] is False and "does not exist" in missing_root["problems"][-1]


def test_the_installation_check_becomes_ready_when_the_counts_are_the_published_ones(release, small_published_split):
    report = assets.check_installation(release.episodes_dir, release.scenes_dir, MP3D_V1)
    assert report["episodes_ready"] is True and report["scenes_ready"] is True and not report["problems"]


def test_the_episode_archive_is_recognised_in_either_layout_and_nothing_else():
    flat = ["train/", "train/content/a.json.gz", "val/", "val/val.json.gz", "val/content/x.json.gz",
            "val_mini/", "val_mini/val_mini.json.gz"]
    assert assets._archive_layout(flat) == ("", {"val", "val_mini"})
    wrapped = ["objectnav_mp3d_v1/" + n for n in flat]
    assert assets._archive_layout(wrapped) == ("objectnav_mp3d_v1", {"val", "val_mini"})
    assert assets._archive_layout(["val/", "val/content/x.json.gz", "README"]) == ("", set())


def test_fetching_the_episodes_unpacks_every_split_and_is_idempotent(tmp_path):
    archive = tmp_path / "objectnav_mp3d_v1.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for split in ("train", "val"):
            zipped.writestr("%s/%s.json.gz" % (split, split), gzip.compress(b'{"episodes": []}'))
            zipped.writestr("%s/content/%s.json.gz" % (split, SCENE), gzip.compress(b'{"episodes": []}'))
    messages = []
    split_dir = assets.fetch_episodes(tmp_path / "data", MP3D_V1, url=archive.as_uri(), log=messages.append)
    assert split_dir == tmp_path / "data" / "objectnav" / "mp3d" / "v1" / "val"
    assert (split_dir / "val.json.gz").is_file() and (split_dir / "content" / (SCENE + ".json.gz")).is_file()
    assert (tmp_path / "data" / "objectnav" / "mp3d" / "v1" / "train" / "train.json.gz").is_file()
    assert not (tmp_path / "data" / "objectnav" / "mp3d" / "v1" / "objectnav_mp3d_v1.zip").exists()
    again = assets.fetch_episodes(tmp_path / "data", MP3D_V1, url=archive.as_uri(), log=messages.append)
    assert again == split_dir and messages[-1].startswith("episodes already present")


def test_an_archive_without_the_published_splits_is_refused(tmp_path):
    archive = tmp_path / "other.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("something/else.txt", "nope")
    with pytest.raises(RuntimeError, match="not the published archive"):
        assets.fetch_episodes(tmp_path / "data", MP3D_V1, url=archive.as_uri(), log=lambda m: None)


def test_the_instructions_name_the_public_episodes_and_the_licensed_scenes():
    text = assets.download_instructions(MP3D_V1, "/data")
    assert "--fetch-episodes" in text and "mp3d_habitat.zip" in text and "licensed" in text
    assert "/data/scene_datasets/mp3d/<scene>/" in text
