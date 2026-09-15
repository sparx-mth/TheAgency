"""HM3D's two published tables and the audit beside them: what a reader may conclude from a run.

Two numbers come out of one run because two papers disagree about one radius,
so the arithmetic relating them is what has to hold: the 0.2 m row is the same
episodes re-scored, never a second measurement, and re-scoring upward can only
add successes. The rest is transcription discipline -- a paper figure typed in
as a percent, a row filed under a benchmark key no run ever writes, a
400-episode subset printed as if it were the split -- and the audit that keeps
a subset, a crashed agent or a plumbing diagnostic from being quoted as a
benchmark result.

No simulator, no dataset: rows are built directly and a results directory is
written by the harness's own logger.
"""
from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from sparx_agency.core.planning.objnav.types.measurement import (
    TERMINATION_STEP_LIMIT,
    TERMINATION_STOP,
)
from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.comparison import comparison_table
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import (
    ACTION_NAMES,
    TERMINATION_AGENT_ERROR,
    EpisodeRecord,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import PUBLISHED_VAL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import (
    APEXNAV_SUCCESS_DISTANCE_M,
    HM3D_V1,
    HM3D_V2,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.report import (
    merge_runs,
    rescore,
    write_report,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.sota import OSG, reported_results

SCENES = ("00800-TEEsavR23oF", "00802-wcojb4TFT35")

#: Final distances spanning both radii and both of their boundaries, plus
#: habitat's infinity as the rows carry it.
DISTANCE_SPREAD = (None, 0.0, 0.05, 0.1, 0.15, 0.199, 0.2, 0.75, 4.0)


def record(episode_id, *, distance=0.05, agent="rpt", scene=SCENES[0],
           category="chair", shortest=2.0, travelled=8.0, stopped=True,
           error=None, protocol=HM3D_V2, soft_spl=0.42, wall_s=1.5):
    """One results row, scored by habitat's rule at the protocol's own radius.

    Built the way the runner leaves it, so re-scoring at that same radius has
    to reproduce it rather than merely resemble it.
    """
    counts = dict.fromkeys(ACTION_NAMES, 0)
    counts.update(MOVE_FORWARD=31, TURN_LEFT=8, STOP=1 if stopped else 0)
    success = bool(error is None and stopped and distance is not None
                   and distance < protocol.success_distance_m)
    ratio = 1.0 if max(shortest, travelled) <= 0 else shortest / max(shortest, travelled)
    termination = (TERMINATION_AGENT_ERROR if error is not None else
                   TERMINATION_STOP if stopped else TERMINATION_STEP_LIMIT)
    return EpisodeRecord(
        benchmark=protocol.benchmark, split=protocol.split, episode_id=episode_id,
        scene_id=scene, target_category=category, agent=agent, success=success,
        spl=float(success) * ratio, soft_spl=soft_spl, distance_to_goal_m=distance,
        path_length_m=travelled, observed_path_length_m=travelled,
        shortest_path_m=shortest, start_distance_to_goal_m=shortest,
        steps=sum(counts.values()),
        stop_called=termination != TERMINATION_STEP_LIMIT, termination=termination,
        action_counts=counts, wall_s=wall_s, agent_error=error)


def spread():
    """Every way an episode can end, at the radii that separate them."""
    rows = [record("dt%02d" % index, distance=value)
            for index, value in enumerate(DISTANCE_SPREAD)]
    rows.append(record("crashed", distance=0.0, error="RuntimeError: detector died"))
    rows.append(record("ran-out", distance=0.05, stopped=False))
    return rows


def complete_run_records(agent="rpt"):
    """Four episodes over two scenes: one success at 0.1 m, a second at 0.2 m."""
    return [record("ep%d" % index, agent=agent, scene=SCENES[index % 2],
                   category=("chair", "bed", "toilet", "sofa")[index],
                   distance=(0.05, 0.15, 3.0, None)[index])
            for index in range(4)]


def run_config(records, *, protocol=HM3D_V2, method="rpt", full_split=True,
               scene_count=len(SCENES), shards=1, shard_index=0, selected=None):
    """A run manifest shaped like the one ``hm3d.run.prepare`` assembles."""
    ids = [row.episode_id for row in records] if selected is None else list(selected)
    return {
        "protocol": asdict(protocol),
        "protocol_id": protocol.protocol_id,
        "runtime": {"habitat-sim": protocol.reference_habitat_sim_version},
        "reference_habitat_sim_version_match": True,
        "method": {"method": method, "publishable": method != "diagnostic-stop"},
        "seed": 0,
        "gpu_device": 0,
        "shards": shards,
        "shard_index": shard_index,
        "limit": None,
        "scenes": [],
        "starts_validated": True,
        "excluded_unreachable_episode_ids": [],
        "full_split": full_split,
        "dataset": {"release": "HM3D ObjectNav %s %s on %s"
                               % (protocol.dataset_version, protocol.split,
                                  protocol.scene_release),
                    "dataset_version": protocol.dataset_version,
                    "episode_count": len(ids),
                    "scene_count": scene_count,
                    "scene_assets_hashed": True},
        "selected_episode_ids": ids,
    }


def write_run(directory, records, **options):
    """A finished results directory, written through the harness's own logger."""
    config = run_config(records, **options)
    with MetricsLogger(directory, config, argv=["pytest"]) as logger:
        for row in records:
            logger.log(row)
        logger.finish(summarise(records))
    return config


@pytest.fixture
def small_published_split(monkeypatch):
    """Shrink the published v2 expectation so a complete run is four episodes.

    What is under test is the audit's completeness rule, not the size of the
    split; generating the real 1000 episodes would make this a benchmark run.
    """
    monkeypatch.setitem(PUBLISHED_VAL, "v2",
                        {"scenes": len(SCENES), "episodes": 4})


# -- rescore ---------------------------------------------------------------

def test_a_looser_success_radius_only_ever_adds_successes():
    """The 0.2 m table is an upper bound on the 0.1 m one by construction; a
    re-score that could take a success away would make the two incomparable."""
    rows = spread()
    succeeded = [{row.episode_id for row in rescore(rows, radius) if row.success}
                 for radius in (0.05, 0.1, 0.15, APEXNAV_SUCCESS_DISTANCE_M, 1.0)]
    for tighter, looser in zip(succeeded, succeeded[1:]):
        assert tighter <= looser
    stored = {row.episode_id for row in rows if row.success}
    assert stored < {row.episode_id
                     for row in rescore(rows, APEXNAV_SUCCESS_DISTANCE_M)
                     if row.success}


def test_no_radius_credits_an_unreachable_goal_or_a_crashed_agent():
    """A None distance is habitat's infinity, and an agent-error row already
    claims STOP, so only the crash guard keeps a wide radius off it."""
    rows = [record("unreachable", distance=None),
            record("crashed", distance=0.0, error="ValueError: boom")]
    for radius in (0.1, APEXNAV_SUCCESS_DISTANCE_M, 5.0):
        assert not any(row.success for row in rescore(rows, radius))
        assert all(row.spl == 0.0 for row in rescore(rows, radius))


def test_rescoring_at_the_protocol_radius_reproduces_the_rows_on_disk():
    """The environment and this module implement one rule, so the official
    radius is a fixed point; if it were not, the two tables would disagree
    about episodes nobody re-ran."""
    rows = spread()
    assert rescore(rows, HM3D_V2.success_distance_m) == rows


def test_a_gained_success_takes_the_published_spl_and_leaves_soft_spl_alone():
    """SPL is ``S * l / max(l, p)`` at whichever radius decided ``S``, while
    habitat-lab gates SoftSPL on neither success nor STOP: the radius does not
    enter it, and a SoftSPL that moved would be a second, unpublished metric."""
    rows = spread()
    loose = rescore(rows, APEXNAV_SUCCESS_DISTANCE_M)
    assert all(new.soft_spl == old.soft_spl for old, new in zip(rows, loose))
    gained = next(row for row in loose if row.episode_id == "dt04")
    assert gained.success and gained.spl == pytest.approx(2.0 / 8.0)
    assert not next(row for row in rows if row.episode_id == "dt04").success


def test_an_episode_that_started_on_its_goal_scores_one_instead_of_dividing_by_zero():
    """habitat-lab raises ZeroDivisionError when ``l == p == 0``; a report that
    died on the one episode whose start already was the goal would lose the
    whole run's table."""
    scored, = rescore([record("on-goal", distance=0.0, shortest=0.0, travelled=0.0)],
                      APEXNAV_SUCCESS_DISTANCE_M)
    assert scored.success and scored.spl == 1.0


# -- sota ------------------------------------------------------------------

@pytest.mark.parametrize("benchmark", ("hm3d_v1", "hm3d_v2"))
def test_every_published_row_is_a_fraction_that_names_its_paper_and_table(benchmark):
    """A transcribed number whose source is not recorded cannot be checked
    later, and a percent typed where a fraction belongs prints as 5250%."""
    rows = reported_results(benchmark)
    assert rows
    for row in rows:
        assert (row.benchmark, row.split) == (benchmark, "val")
        assert 0.0 < row.success_rate <= 1.0 and 0.0 < row.spl <= 1.0
        assert row.soft_spl is None or 0.0 < row.soft_spl <= 1.0
        assert "arXiv:" in row.source and "Table" in row.source
        assert row.notes.strip()
    assert len({row.method for row in rows}) == len(rows)


def test_osg_rows_are_off_by_default_and_declare_their_protocol_when_asked_for():
    """OSG's 400 self-sampled episodes are not the published split; printed in
    the default table they would read as a like-for-like row."""
    default = reported_results("hm3d_v1")
    assert not any(row.source == OSG for row in default)
    with_osg = reported_results("hm3d_v1", include_osg=True)
    assert with_osg[:len(default)] == default and len(with_osg) > len(default)
    for row in with_osg[len(default):]:
        assert row.source == OSG and row.distance_to_goal_m is not None
        assert "400" in row.notes and "NOT a like-for-like" in row.notes
    assert reported_results("hm3d_v2", include_osg=True) == reported_results("hm3d_v2")


def test_an_unknown_benchmark_names_the_ones_that_exist():
    """The caller who passes "hm3d" or an MP3D key has to be told what to pass."""
    with pytest.raises(KeyError, match="known: hm3d_v1, hm3d_v2"):
        reported_results("hm3d")


@pytest.mark.parametrize("protocol", (HM3D_V1, HM3D_V2))
def test_published_rows_are_filed_under_the_keys_a_run_files_its_results_under(protocol):
    """``comparison_table`` refuses a row from another benchmark or split, and
    it is only reached after the last episode: a key that disagreed would cost
    the whole run its table."""
    reported = reported_results(protocol.benchmark, protocol.split, include_osg=True)
    rows = [record("ep0", protocol=protocol),
            record("ep1", distance=3.0, protocol=protocol)]
    table = comparison_table(summarise(rows), reported)
    assert table.count("\n") == len(reported) + 2
    assert table.endswith("|") and "rpt (ours)" in table


def test_every_row_of_an_hm3d_table_is_an_hm3d_number():
    """A row whose own note says the paper reported no HM3D figure still prints
    SR and SPL cells in the HM3D table, which is where they get compared."""
    for benchmark in ("hm3d_v1", "hm3d_v2"):
        for row in reported_results(benchmark, include_osg=True):
            assert "hm3d not reported" not in row.notes.lower()


# -- write_report ----------------------------------------------------------

def test_a_finished_run_writes_both_tables_and_an_audit_of_what_was_scored(
        tmp_path, small_published_split):
    """The second table must be visibly the same episodes re-scored: its extra
    successes are counted, so a reader can see the 0.2 m row is not a re-run."""
    write_run(tmp_path / "run", complete_run_records())
    audit = write_report(tmp_path / "run")

    text = (tmp_path / "run" / "comparison.md").read_text()
    assert json.loads((tmp_path / "run" / "audit.json").read_text()) == audit
    assert text.startswith("# Full HM3D-v2 val validation")
    assert audit["full_split_completed"] is True
    assert audit["episodes_scored"] == audit["episodes_published"] == 4
    assert audit["scenes_loaded"] == audit["scenes_published"] == len(SCENES)
    assert audit["success_distance_m"] == HM3D_V2.success_distance_m
    assert audit["apexnav_radius_extra_successes"] == 1
    assert audit["apexnav_radius_success_rate"] == 0.5
    assert text.count("| Method |") == 2
    assert "rpt (ours)" in text and "rpt @0.2 m (ours)" in text
    assert "ApexNav" in text and "not a like-for-like" not in text.lower()


@pytest.mark.parametrize("flavour", ("subset", "missing_scene", "diagnostic", "crashed"))
def test_only_a_complete_clean_run_of_the_published_split_is_titled_a_benchmark_result(
        tmp_path, small_published_split, flavour):
    """Each of these prints a plausible success rate; the title and the audit
    are all that stand between one of them and a quoted benchmark number."""
    rows, options = complete_run_records(), {}
    if flavour == "subset":
        rows, options = rows[:2], {"full_split": False}
    elif flavour == "missing_scene":
        options = {"scene_count": 1}
    elif flavour == "diagnostic":
        rows = complete_run_records(agent="diagnostic-stop")
        options = {"method": "diagnostic-stop"}
    else:
        rows[2] = record("ep2", scene=SCENES[0], category="toilet", distance=1.0,
                         error="RuntimeError: detector died")
    write_run(tmp_path / "run", rows, **options)
    audit = write_report(tmp_path / "run")

    assert audit["full_split_completed"] is False
    assert (tmp_path / "run" / "comparison.md").read_text().startswith(
        "# NOT a full benchmark result")
    assert (audit["episodes_scored"] < 4) == (flavour == "subset")
    assert (audit["scenes_loaded"] == len(SCENES)) == (flavour != "missing_scene")
    assert (audit["agent_errors"] == 1) == (flavour == "crashed")


def test_a_run_recorded_under_another_protocol_is_refused_rather_than_rescored(tmp_path):
    """Every number in the report is arithmetic over rows scored at 0.1 m. A run
    already scored at ApexNav's radius would be re-scored to itself and printed
    as two independent tables saying the same thing."""
    loose = HM3D_V2.with_success_distance(APEXNAV_SUCCESS_DISTANCE_M)
    write_run(tmp_path / "run", [record("ep0", protocol=loose)], protocol=loose)
    with pytest.raises(ValueError, match="different HM3D protocol"):
        write_report(tmp_path / "run")


# -- merge_runs ------------------------------------------------------------

def test_merging_complete_disjoint_shards_scores_every_episode_exactly_once(
        tmp_path, small_published_split):
    """Sharding is how a long split is run in parallel, and the merged directory
    is the one that gets read: an episode dropped or doubled here is invisible
    in the table it produces."""
    rows = complete_run_records()
    shards = [tmp_path / "shard0", tmp_path / "shard1"]
    for index, directory in enumerate(shards):
        write_run(directory, rows[2 * index:2 * index + 2], full_split=False,
                  shards=2, shard_index=index)
    audit = merge_runs(shards, tmp_path / "merged")
    merged = (tmp_path / "merged" / "episodes.jsonl").read_text().splitlines()
    assert audit["episodes_scored"] == 4
    assert [json.loads(line)["episode_id"] for line in merged] == [
        "ep0", "ep1", "ep2", "ep3"]


def test_a_shard_that_is_not_complete_and_disjoint_is_refused(tmp_path,
                                                              small_published_split):
    """Two shards that overlap would count an episode twice and inflate n; one
    that is missing leaves a partial split looking like a whole one."""
    rows = complete_run_records()
    write_run(tmp_path / "shard0", rows[:2], full_split=False, shards=2, shard_index=0)
    write_run(tmp_path / "overlap", rows[1:3], full_split=False, shards=2, shard_index=1)
    with pytest.raises(ValueError, match="overlap"):
        merge_runs([tmp_path / "shard0", tmp_path / "overlap"], tmp_path / "merged")
    with pytest.raises(ValueError, match="Missing shards"):
        merge_runs([tmp_path / "shard0"], tmp_path / "alone")


def test_merged_complete_shards_are_reported_as_the_full_split(tmp_path,
                                                               small_published_split):
    """Every shard check -- no ``--limit``, every index present, no overlap --
    exists to establish that the merge *is* the published split. The audit then
    has to say so, or a sharded evaluation can never be quoted at all."""
    rows = complete_run_records()
    shards = [tmp_path / "shard0", tmp_path / "shard1"]
    for index, directory in enumerate(shards):
        write_run(directory, rows[2 * index:2 * index + 2], full_split=False,
                  shards=2, shard_index=index)
    assert merge_runs(shards, tmp_path / "merged")["full_split_completed"] is True


def test_a_development_split_is_never_printed_under_the_papers_table(tmp_path):
    """The compared papers report validation. A train row set under their table
    invites exactly the comparison it is not, and ``comparison_table`` would
    refuse the mismatched split anyway -- so the report says so instead."""
    train = HM3D_V2.with_split("train")
    rows = [record("ep%d" % i, protocol=train, distance=(0.05, 3.0)[i])
            for i in range(2)]
    write_run(tmp_path, rows, protocol=train, full_split=False)
    audit = write_report(tmp_path)
    assert audit["split"] == "train"
    assert audit["published_benchmark_split"] is False
    assert audit["full_split_completed"] is False
    assert audit["episodes_published"] is None
    text = (tmp_path / "comparison.md").read_text()
    # No transcribed paper row: no source citation, and none of the baselines.
    assert "arXiv:2504.14478" not in text and "arXiv:2410.08189" not in text
    assert "| SG-Nav" not in text and "| VLFM" not in text
    assert "no compared paper" in text and "train" in text
    # ApexNav is still named, but only as the owner of the 0.2 m radius, which
    # is a convention the re-score applies rather than a row to compare with.
    assert "ApexNav's 0.2 m success radius" in text
