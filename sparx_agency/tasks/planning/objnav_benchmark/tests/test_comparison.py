"""The table where a run meets the literature: its format, and its refusal to mix splits."""
from __future__ import annotations

import pytest

from sparx_agency.tasks.planning.objnav_benchmark.comparison import (
    comparison_table,
    reported_results_from_rows,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import HarnessError
from sparx_agency.tasks.planning.objnav_benchmark.summaries import (
    BenchmarkSummary,
    GroupSummary,
    ReportedResult,
)

EM_DASH = "—"


def summary(agent="headless/oracle", split="val", dtg=1.234, unreachable=0,
            agent_errors=0):
    """A run's summary with round numbers, built directly."""
    overall = GroupSummary(
        key="all", n_episodes=1000, success_rate=0.462,
        success_rate_ci=(0.431, 0.493), spl=0.251, spl_ci=(0.235, 0.268),
        soft_spl=0.301, distance_to_goal_m=dtg, n_unreachable_end=unreachable,
        mean_steps=120.0, mean_path_length_m=9.5)
    return BenchmarkSummary(
        benchmark="hm3d_v2", split=split, agent=agent, overall=overall,
        by_category=(), by_scene=(),
        terminations={"stop": 1000 - agent_errors, "step_limit": 0,
                      "agent_error": agent_errors},
        agent_errors=agent_errors, confidence=0.95)


def paper(method="Method A", benchmark="hm3d_v2", split="val", soft_spl=None,
          dtg=None, notes=""):
    """A reported result with made-up numbers and source."""
    return ReportedResult(
        method=method, benchmark=benchmark, split=split, success_rate=0.525,
        spl=0.304, source="Paper A, Table 1", soft_spl=soft_spl,
        distance_to_goal_m=dtg, notes=notes)


def lines_of(table):
    return table.split("\n")


def cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split(" | ")]


# -- comparison_table ------------------------------------------------------

def test_the_table_has_the_six_columns_in_order():
    """The column set is the one every ObjectNav paper table uses."""
    header, alignment = lines_of(comparison_table(summary(), []))[:2]
    assert header == "| Method | SR (%) | SPL (%) | SoftSPL (%) | DTG (m) | Source |"
    assert alignment == "| --- | ---: | ---: | ---: | ---: | --- |"


def test_reported_rows_keep_their_order_and_ours_comes_last():
    """Papers in the order the caller chose, then ours, so the eye lands on it last."""
    lines = lines_of(comparison_table(summary(), [paper("B"), paper("A")]))
    assert len(lines) == 5
    assert [cells(line)[0] for line in lines[2:]] == [
        "B", "A", "headless/oracle (ours)"]


def test_our_row_shows_the_success_rate_and_spl_with_their_intervals():
    """A gap narrower than the interval is not a result, so the interval is in the row."""
    ours = lines_of(comparison_table(summary(), []))[-1]
    assert ours == ("| headless/oracle (ours) | 46.2 [43.1, 49.3] | "
                    "25.1 [23.5, 26.8] | 30.1 | 1.23 | "
                    "this run: 1000 episodes, 95% intervals |")


def test_a_reported_row_prints_percent_and_metres():
    """Fractions in the records, percent in the table: the one place that multiplies by 100."""
    row = lines_of(comparison_table(
        summary(), [paper(soft_spl=0.35, dtg=2.5)]))[2]
    assert row == "| Method A | 52.5 | 30.4 | 35.0 | 2.50 | Paper A, Table 1 |"


def test_our_label_replaces_the_agent_name_but_keeps_the_ours_mark():
    """A readable label is fine; losing the (ours) mark is not."""
    ours = lines_of(comparison_table(summary(), [], our_label="VLM search"))[-1]
    assert cells(ours)[0] == "VLM search (ours)"


def test_values_a_paper_did_not_report_print_as_an_em_dash():
    """A missing value printed as 0 would read as a terrible result."""
    table = comparison_table(summary(dtg=None), [paper()])
    reported, ours = lines_of(table)[2:]
    assert cells(reported)[3:5] == [EM_DASH, EM_DASH]
    assert cells(ours)[4] == EM_DASH


def test_a_reported_result_for_another_split_is_refused():
    """Mixing splits in one table is exactly the error the table exists to catch."""
    with pytest.raises(HarnessError, match="hm3d_v2/test.*hm3d_v2/val"):
        comparison_table(summary(split="val"), [paper(split="test")])


def test_a_reported_result_for_another_benchmark_is_refused():
    """An HM3D v1 figure beside a v2 run compares different episodes."""
    with pytest.raises(HarnessError, match="hm3d_v1"):
        comparison_table(summary(), [paper(benchmark="hm3d_v1")])


def test_a_reported_caveat_stays_in_the_table():
    """Notes are the reason a number is not directly comparable; dropping them misleads."""
    row = lines_of(comparison_table(
        summary(), [paper(notes="500-episode subset")]))[2]
    assert cells(row)[5] == "Paper A, Table 1; 500-episode subset"


def test_a_pipe_in_a_method_name_cannot_break_the_table():
    """An unescaped | would shift every later cell of the row one column right."""
    row = lines_of(comparison_table(summary(), [paper(method="A|B")]))[2]
    assert "A\\|B" in row
    assert row.replace("\\|", "").count("|") == 7


def test_our_source_admits_unreachable_ends_and_agent_errors():
    """A DTG over a subset and an SR lowered by crashes must say so in the row."""
    ours = lines_of(comparison_table(
        summary(unreachable=3, agent_errors=2), []))[-1]
    assert "DTG over the 997 that ended with a goal reachable" in ours
    assert "2 agent errors counted as failures" in ours


def test_comparison_table_refuses_a_blank_label_and_foreign_objects():
    """Each would print a table that looks fine and is not."""
    with pytest.raises(HarnessError, match="our_label"):
        comparison_table(summary(), [], our_label=" ")
    with pytest.raises(TypeError, match="ReportedResult"):
        comparison_table(summary(), [{"method": "A"}])
    with pytest.raises(TypeError, match="BenchmarkSummary"):
        comparison_table({"agent": "a"}, [])


# -- reported_results_from_rows --------------------------------------------

def row(**overrides):
    base = {"method": "Method A", "benchmark": "hm3d_v2", "split": "val",
            "success_rate": 0.525, "spl": 0.304,
            "source": "Paper A, Table 1"}
    base.update(overrides)
    return base


def test_reported_results_round_trip_from_rows():
    """Paper numbers typed into a JSON file must become exactly these objects."""
    results = reported_results_from_rows([row(), row(method="B", soft_spl=0.4,
                                                     notes="n=500")])
    assert results == (paper(), ReportedResult(
        method="B", benchmark="hm3d_v2", split="val", success_rate=0.525,
        spl=0.304, source="Paper A, Table 1", soft_spl=0.4, notes="n=500"))


def test_an_unknown_key_in_a_reported_row_is_refused():
    """A typo such as 'sucess_rate' must fail at the file, not drop the number."""
    bad = row()
    bad["sucess_rate"] = bad.pop("success_rate")
    with pytest.raises(HarnessError, match="sucess_rate"):
        reported_results_from_rows([bad])


def test_a_missing_key_in_a_reported_row_is_refused():
    """A number without its source cannot be checked, so the row is refused."""
    bad = row()
    del bad["source"]
    with pytest.raises(HarnessError, match="source"):
        reported_results_from_rows([bad])


def test_a_percent_typed_as_a_fraction_is_refused_naming_the_row():
    """46.2 for 0.462 is the usual slip; the message must point at the row."""
    with pytest.raises(HarnessError, match="reported row 1.*divide by 100"):
        reported_results_from_rows([row(), row(success_rate=46.2)])


@pytest.mark.parametrize("rows", [row(), [row(), "not a row"],
                                  [row(notes=None)]])
def test_malformed_rows_are_refused(rows):
    """One mapping for a list, or a non-mapping row, is a file-format mistake."""
    with pytest.raises(HarnessError):
        reported_results_from_rows(rows)


def test_no_reported_rows_give_a_table_with_only_our_row():
    """The smoke run prints a table with nothing to compare against."""
    assert reported_results_from_rows([]) == ()
    assert len(lines_of(comparison_table(summary(), []))) == 3
