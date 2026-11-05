"""The sharded MP3D campaign, without a simulator, a GPU or Matterport3D.

Its failure modes are an orchestrator's: a shard plan that drops, repeats or
re-orders an episode; a pool that starts more work than it was allowed, or
starts it into a full card; a resumed shard sent to another LLM server than the
one its configuration froze; and a scene merged from shards that are not one
complete experiment. Every test runs the real planner, pool and merge with the
processes replaced.
"""
from __future__ import annotations

import json
import types

import pytest

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import (
    campaign, campaign_status, report, shard_plan, shard_pool, sharded_campaign)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import MP3D_V1, PUBLISHED_VAL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import select_episodes
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_mp3d import (
    OTHER_SCENE, SCENE, row, write_release)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_mp3d_evaluation import (
    record, run_config)

URL_A, URL_B = "http://127.0.0.1:11435", "http://127.0.0.1:11436"


@pytest.fixture
def release(tmp_path):
    """Five episodes in one scene, three in the other."""
    return write_release(tmp_path, MP3D_V1, {SCENE: [row()] * 5, OTHER_SCENE: [row("bed")] * 3})


@pytest.fixture
def small_published_split(monkeypatch):
    monkeypatch.setitem(PUBLISHED_VAL, "v1", {"scenes": 2, "episodes": 8})


def shard_config(dataset, scene, shards, index, ids, url=URL_A):
    """What ``mp3d.run`` freezes for one shard."""
    config = run_config(dataset, method="rpt", selected=ids)
    config.update(shards=shards, shard_index=index, scenes=[scene])
    config["method"] = dict(config["method"], llm={"base_url": url, "model": "qwen2.5:3b-instruct", "num_ctx": 8192},
                            llm_identity={"digest": "d1"})
    return config


def write_shard(root, dataset, scene, shards, index, *, url=URL_A, finish=True, rows=None):
    """One shard's results directory, as ``mp3d.run --shards K --shard-index k`` leaves it."""
    ids = list(select_episodes([i for i in dataset.episode_ids() if i.startswith(scene)], None, shards, index))
    directory = root / scene / ("shard_%d" % index)
    done = [record(i, scene=scene) for i in ids][:rows]
    with MetricsLogger(directory, shard_config(dataset, scene, shards, index, ids, url), argv=["pytest"]) as logger:
        for r in done:
            logger.log(r)
        if finish:
            logger.finish(summarise(done))
    return directory


class FakeProcess:
    def __init__(self, code=0, polls=0):
        self.code, self.polls, self.terminated = code, polls, False

    def poll(self):
        if self.polls > 0:
            self.polls -= 1
            return None
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


class OpenGate:
    last_reason = None

    def __init__(self):
        self.starts = 0

    def ready(self):
        return True

    def started(self):
        self.starts += 1


# -- the plan ---------------------------------------------------------------

def test_the_plan_covers_every_episode_once_in_the_children_own_selection():
    plan = shard_plan.plan_shards({"A": 23, "B": 5, "C": 10}, 10)
    assert [(j.scene, j.index) for j in plan] == [("A", 0), ("B", 0), ("C", 0), ("A", 1), ("A", 2)]
    for scene, total in (("A", 23), ("B", 5), ("C", 10)):
        jobs = [j for j in plan if j.scene == scene]
        ids = ["%s/%06d" % (scene, i) for i in range(total)]
        chosen = [select_episodes(ids, None, j.shards, j.index) for j in jobs]
        assert [len(c) for c in chosen] == [j.episodes for j in jobs]
        assert sorted(i for c in chosen for i in c) == ids
        assert all(list(c) == sorted(c) for c in chosen)          # the publisher's order within a shard
    assert max(j.episodes for j in plan) <= 10 and sum(j.episodes for j in plan) == 38
    with pytest.raises(ValueError):
        shard_plan.shard_count(5, 0)


def test_fresh_shards_are_dealt_over_the_servers_and_a_started_one_keeps_its_own(tmp_path, release):
    plan = shard_plan.plan_shards({SCENE: 5, OTHER_SCENE: 3}, 2)
    fresh = shard_plan.assign_llm_urls(plan, [URL_A, URL_B + "/"], tmp_path)
    assert [fresh[j.key] for j in plan] == [URL_A, URL_B] * 2 + [URL_A]
    write_shard(tmp_path, release, SCENE, 3, 0, url=URL_B, finish=False, rows=1)
    resumed = shard_plan.assign_llm_urls(plan, [URL_A, URL_B], tmp_path)
    assert resumed[SCENE + "/shard_0"] == URL_B
    with pytest.raises(ValueError, match="not among"):
        shard_plan.assign_llm_urls(plan, [URL_A], tmp_path)
    assert shard_plan.assign_llm_urls(plan, [], tmp_path) == {}


# -- the pool -----------------------------------------------------------------

def test_the_pool_never_exceeds_its_live_cap_and_a_zero_cap_drains_it(tmp_path):
    cap_file = tmp_path / "jobs"
    cap_file.write_text("2")
    alive, peak, launched, ended = [], [0], [], []

    def popen(command, stdout, stderr, env):
        process = FakeProcess(polls=2)
        alive.append(process)
        launched.append((command, env.get("LLM_BASE_URL")))
        peak[0] = max(peak[0], sum(p.code is None or p.polls > 0 for p in alive))
        if len(launched) == 3:
            cap_file.write_text("0")      # the operator drains the pool after the third start
        return process

    jobs = [shard_pool.PoolJob("s/shard_%d" % i, ["run", str(i)], tmp_path / "s" / ("%d.log" % i),
                               {"LLM_BASE_URL": URL_B}) for i in range(5)]
    codes, unstarted = shard_pool.run_pool(
        jobs, cap=lambda: shard_pool.read_cap(4, cap_file), gate=OpenGate(), popen=popen,
        on_exit=lambda job, code: ended.append(job.key), sleep=lambda s: None, log=lambda m: None)
    assert peak[0] <= 2
    assert sorted(codes) == ["s/shard_0", "s/shard_1", "s/shard_2"] and set(codes.values()) == {0}
    assert unstarted == ["s/shard_3", "s/shard_4"] and sorted(ended) == sorted(codes)
    assert all(url == URL_B for _, url in launched) and (tmp_path / "s" / "0.log").is_file()
    assert shard_pool.read_cap(7, tmp_path / "missing") == 7


def test_a_failing_pool_stops_the_shards_it_started(tmp_path):
    started = []

    def popen(command, stdout, stderr, env):
        started.append(FakeProcess(polls=10 ** 6))
        return started[-1]

    def sleep(_):
        raise KeyboardInterrupt

    jobs = [shard_pool.PoolJob("s/shard_0", ["run"], tmp_path / "0.log")]
    with pytest.raises(KeyboardInterrupt):
        shard_pool.run_pool(jobs, cap=lambda: 1, gate=OpenGate(), popen=popen, sleep=sleep, log=lambda m: None)
    assert started[0].terminated


def test_the_vram_gate_delays_starts_and_never_counts_more_than_the_card_holds():
    clock = [0.0]
    used = [10000]
    gate = shard_pool.VramGate(1400, 0.88, spacing_s=20, settle_s=120, query=lambda d: ("card", 24564, used[0]),
                               clock=lambda: clock[0])
    assert gate.ready()                                  # 10000 + 1400 <= 21616
    gate.started()
    clock[0] = 5.0
    assert not gate.ready() and "spacing" in gate.last_reason
    clock[0] = 25.0
    used[0] = 19000                                      # 19000 + 2 x 1400 > 21616: the fresh start is still budgeted
    assert not gate.ready() and "VRAM" in gate.last_reason
    clock[0] = 130.0                                     # settled: only what nvidia-smi shows, plus one
    assert gate.ready()
    gate_fail = shard_pool.VramGate(1400, query=lambda d: (_ for _ in ()).throw(RuntimeError("no card")))
    assert not gate_fail.ready() and "nvidia-smi" in gate_fail.last_reason
    with pytest.raises(ValueError):
        shard_pool.VramGate(1400, 1.5)


# -- the campaign ---------------------------------------------------------------

def shard_runner(release, calls):
    """A popen that 'runs' an mp3d.run shard by writing its finished results directory."""
    def popen(command, stdout, stderr, env):
        output = command[command.index("--output") + 1]
        scene = command[command.index("--scenes") + 1]
        shards = int(command[command.index("--shards") + 1])
        index = int(command[command.index("--shard-index") + 1])
        calls.append({"output": output, "env": env.get("LLM_BASE_URL"), "resume": "--resume" in command,
                      "command": command})
        from pathlib import Path
        root = Path(output).parent.parent
        if "--resume" not in command:
            write_shard(root, release, scene, shards, index, url=env.get("LLM_BASE_URL") or URL_A)
        else:
            directory = Path(output)
            info = json.loads((directory / "run.json").read_text())["config"]
            ids = info["selected_episode_ids"]
            with MetricsLogger(directory, info, resume=True, argv=["pytest"]) as logger:
                done = {r.episode_id for r in logger.records}
                for episode_id in ids:
                    if episode_id not in done:
                        logger.log(record(episode_id, scene=scene))
                logger.finish(summarise(list(logger.records)))
        return FakeProcess()
    return popen


def test_a_sharded_campaign_merges_each_scene_into_one_complete_result(tmp_path, release, small_published_split):
    root, calls = tmp_path / "root", []
    results = sharded_campaign.run_sharded_campaign(
        root, version="v1", episodes_dir=release.episodes_dir, scenes_dir=release.scenes_dir, shard_episodes=2,
        extra=["--lean", "--seed", "17"], jobs=3, llm_urls=[URL_A, URL_B], gate=OpenGate(),
        popen=shard_runner(release, calls), sleep=lambda s: None, log=lambda m: None)
    assert [r["status"] for r in results] == ["merged", "merged"] and all(r["returncode"] == 0 for r in results)
    assert len(calls) == 5 and {c["env"] for c in calls} == {URL_A, URL_B}
    for c in calls:
        assert c["command"][c["command"].index("--seed") + 1] == "17" and "--lean" in c["command"]
    rows = [json.loads(line)["episode_id"] for line in (root / SCENE / "episodes.jsonl").read_text().splitlines()]
    assert rows == ["%s/%06d" % (SCENE, i) for i in range(5)]
    for scene in (SCENE, OTHER_SCENE):
        merged = json.loads((root / scene / "run.json").read_text())["config"]
        used = sorted({c["env"] for c in calls if "/%s/" % scene in c["output"]})
        assert merged["llm_base_urls"] == used and merged["shards"] == 1
    plan = json.loads((root / "shard_plan.json").read_text())
    assert plan["episodes"] == 8 and len(plan["shards"]) == 5
    report_ = campaign.aggregate(root, episodes_dir=release.episodes_dir, version="v1")
    assert report_["complete"] is True and report_["episodes"] == 8
    assert {group["key"] for group in report_["by_category"]} == {"chair"}
    assert report_["apexnav_radius"]["success_distance_m"] == 0.2
    assert json.loads((root / "campaign.json").read_text())["sharded"] is True


def test_a_resumed_sharded_root_continues_started_shards_and_skips_finished_ones(tmp_path, release):
    root, calls = tmp_path / "root", []
    write_shard(root, release, SCENE, 3, 0, url=URL_B)                     # finished
    write_shard(root, release, SCENE, 3, 1, url=URL_B, finish=False, rows=1)  # started on server B
    with pytest.raises(ValueError, match="resume"):
        sharded_campaign.run_sharded_campaign(
            root, version="v1", episodes_dir=release.episodes_dir, scenes_dir=release.scenes_dir,
            shard_episodes=2, llm_urls=[URL_A, URL_B], gate=OpenGate(), popen=shard_runner(release, []),
            sleep=lambda s: None, log=lambda m: None)
    results = sharded_campaign.run_sharded_campaign(
        root, version="v1", episodes_dir=release.episodes_dir, scenes_dir=release.scenes_dir, shard_episodes=2,
        resume=True, extra=["--resume-accepting-changes", "kinematics"], jobs=2, llm_urls=[URL_A, URL_B],
        gate=OpenGate(), popen=shard_runner(release, calls), sleep=lambda s: None, log=lambda m: None)
    by_output = {c["output"].split("/root/")[1]: c for c in calls}
    assert SCENE + "/shard_0" not in by_output
    resumed = by_output[SCENE + "/shard_1"]
    assert resumed["resume"] and resumed["env"] == URL_B
    assert resumed["command"][resumed["command"].index("--resume-accepting-changes") + 1] == "kinematics"
    assert not by_output[SCENE + "/shard_2"]["resume"]
    assert "--resume-accepting-changes" not in by_output[SCENE + "/shard_2"]["command"]
    assert [r["status"] for r in results] == ["merged", "merged"]


def test_a_sharded_campaign_refuses_flags_that_choose_episodes(tmp_path, release):
    with pytest.raises(ValueError, match="chooses the episodes"):
        sharded_campaign.run_sharded_campaign(
            tmp_path / "root", version="v1", episodes_dir=release.episodes_dir, scenes_dir=release.scenes_dir,
            shard_episodes=2, extra=["--limit", "3"], gate=OpenGate(), popen=shard_runner(release, []),
            sleep=lambda s: None, log=lambda m: None)


# -- the merge -------------------------------------------------------------------

def test_shards_on_two_servers_merge_but_another_model_does_not(tmp_path, release):
    first = write_shard(tmp_path / "a", release, OTHER_SCENE, 2, 0, url=URL_A)
    second = write_shard(tmp_path / "a", release, OTHER_SCENE, 2, 1, url=URL_B)
    audit = report.merge_runs([first, second], tmp_path / "merged")
    assert audit["episodes_scored"] == 3
    other = tmp_path / "b" / OTHER_SCENE / "shard_1"
    ids = list(select_episodes([i for i in release.episode_ids() if i.startswith(OTHER_SCENE)], None, 2, 1))
    config = shard_config(release, OTHER_SCENE, 2, 1, ids, URL_B)
    config["method"]["llm_identity"] = {"digest": "another model"}
    with MetricsLogger(other, config, argv=["pytest"]) as logger:
        rows = [record(i, scene=OTHER_SCENE) for i in ids]
        for r in rows:
            logger.log(r)
        logger.finish(summarise(rows))
    with pytest.raises(ValueError, match="disagree"):
        report.merge_runs([first, other], tmp_path / "refused")


# -- the status -------------------------------------------------------------------

def test_the_status_folds_the_shards_and_reads_the_throughput_from_the_heartbeat(tmp_path, release, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    plan = shard_plan.plan_shards({SCENE: 5, OTHER_SCENE: 3}, 2)
    shard_plan.write_plan(root, plan, {}, 2)
    for directory, done in ((root / SCENE / "shard_0", 2), (root / SCENE / "shard_1", 1)):
        directory.mkdir(parents=True)
        (directory / "progress.json").write_text(json.dumps({"done": done, "overall": {"SR": 0.5, "SPL": 0.25}}))
    beats = [{"unix": 1000.0, "done": 0}, {"unix": 1000.0 + 1800, "done": 3}]
    (root / sharded_campaign.HEARTBEAT_FILE).write_text("\n".join(json.dumps(b) for b in beats) + "\n")
    monkeypatch.setattr(campaign_status, "live_outputs", lambda: {str((root / SCENE / "shard_1").resolve())})
    text = campaign_status.describe_campaign(root, now=1000.0 + 1800)
    assert "3/8 episodes" in text and "SR 0.500 SPL 0.250" in text
    assert "throughput 6.0 episodes/h" in text and "ETA 0.8 h" in text
    assert "1 running" in text
    assert campaign_status.describe_campaign(tmp_path / "unsharded") is None
