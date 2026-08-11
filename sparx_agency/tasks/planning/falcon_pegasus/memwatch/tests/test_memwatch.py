"""Parsing /proc dumps, and reading a trace as an allocation plus a slope."""
from __future__ import annotations

import pytest

from sparx_agency.tasks.planning.falcon_pegasus.memwatch.sample import (
    Sample,
    parse_proc_dump,
    parse_vmrss_bytes,
    read_csv,
)
from sparx_agency.tasks.planning.falcon_pegasus.memwatch.summary import (
    format_summary,
    summarise,
)

MB = 1024 * 1024


def test_vmrss_is_read_in_kilobytes():
    assert parse_vmrss_bytes("Name:\tfoo\nVmRSS:\t  4096 kB\n") == 4096 * 1024


def test_a_status_without_vmrss_gives_nothing():
    """A process that exits between being listed and being read."""
    assert parse_vmrss_bytes("Name:\tfoo\nState:\tZ (zombie)\n") is None


def test_the_marked_process_is_separated_from_the_container_total():
    watched, total = parse_proc_dump(
        "VmRSS:\t 1024 kB\n"
        "*VmRSS:\t 4096 kB\n"
        "VmRSS:\t 2048 kB\n"
    )
    assert watched == 4096 * 1024
    assert total == (1024 + 4096 + 2048) * 1024


def test_an_absent_watched_process_reads_as_none_not_zero():
    """Zero would look like a real reading of a node holding nothing."""
    watched, total = parse_proc_dump("VmRSS:\t 1024 kB\n")
    assert watched is None
    assert total == 1024 * 1024


def test_several_matches_report_the_largest_not_their_sum():
    """roslaunch carries the node's name on its own command line."""
    watched, _ = parse_proc_dump("*VmRSS:\t 2048 kB\n*VmRSS:\t 400000 kB\n")
    assert watched == 400000 * 1024


def test_an_empty_dump_reads_as_nothing_at_all():
    assert parse_proc_dump("") == (None, None)


def _trace(values_mb, start_s=0.0, step_s=2.0):
    return [
        Sample(start_s + i * step_s, int(v * MB), int(v * MB) + 50 * MB)
        for i, v in enumerate(values_mb)
    ]


def test_startup_is_taken_after_the_settle_point():
    """The first readings catch the node still starting; they are not the plateau."""
    samples = _trace([20, 60, 300, 302, 303, 304], step_s=10.0)
    summary = summarise(samples, settle_s=20.0)
    assert summary.startup_bytes == 300 * MB
    assert summary.final_bytes == 304 * MB


def test_a_flat_trace_after_allocation_shows_no_growth():
    """What a dense array allocated once on the first tick looks like."""
    summary = summarise(_trace([300] * 30, step_s=10.0), settle_s=20.0)
    assert summary.growth_total_bytes == 0
    assert summary.growth_bytes_per_min == pytest.approx(0.0, abs=1.0)


def test_a_climbing_trace_reports_a_slope():
    """One megabyte every ten seconds is six a minute."""
    summary = summarise(_trace([300 + i for i in range(30)], step_s=10.0), settle_s=20.0)
    assert summary.growth_bytes_per_min == pytest.approx(6.0 * MB, rel=0.02)
    assert summary.growth_total_bytes > 0


def test_peak_survives_a_dip():
    summary = summarise(_trace([300, 400, 350], step_s=10.0), settle_s=0.0)
    assert summary.peak_bytes == 400 * MB


def test_the_nodes_exit_is_not_counted_as_shrinkage():
    """A normal run ends with the map being freed; that is not a measurement."""
    samples = _trace([300, 301, 302, 303, 120, 2], step_s=10.0)
    summary = summarise(samples, settle_s=0.0)
    assert summary.teardown_dropped == 2
    assert summary.final_bytes == 303 * MB
    assert summary.growth_total_bytes == 3 * MB
    assert summary.peak_bytes == 303 * MB


def test_a_dip_in_the_middle_still_counts():
    """Only the trailing collapse is teardown; trim from the end alone."""
    summary = summarise(_trace([300, 100, 305, 310], step_s=10.0), settle_s=0.0)
    assert summary.teardown_dropped == 0
    assert summary.final_bytes == 310 * MB


def test_the_report_mentions_what_it_dropped():
    summary = summarise(_trace([300, 301, 302, 2], step_s=10.0), settle_s=0.0)
    assert "dropped as the node exiting" in format_summary(summary)


def test_a_trace_with_no_readings_is_reported_not_crashed():
    summary = summarise([Sample(0.0, None, None), Sample(2.0, None, None)])
    assert summary.samples == 0
    assert "no readings" in format_summary(summary)


def test_the_report_flags_a_startup_the_map_cannot_explain():
    summary = summarise(_trace([2000] * 10, step_s=10.0), settle_s=20.0)
    text = format_summary(summary, expected_grid_bytes=267 * MB)
    assert "NOT the voxel map" in text


def test_the_report_stays_quiet_when_the_map_explains_it():
    summary = summarise(_trace([300] * 10, step_s=10.0), settle_s=20.0)
    text = format_summary(summary, expected_grid_bytes=267 * MB)
    assert "NOT the voxel map" not in text
    assert "89%" in text


def test_csv_round_trip_keeps_the_gaps():
    samples = [Sample(0.0, None, None), Sample(2.0, 300 * MB, 350 * MB)]
    text = "elapsed_s,rss_bytes,container_bytes\n" + "\n".join(
        s.csv_row() for s in samples
    )
    reloaded = read_csv(text)
    assert reloaded[0].rss_bytes is None
    assert reloaded[1].rss_bytes == 300 * MB
    assert reloaded[1].elapsed_s == pytest.approx(2.0)


def test_a_short_window_is_flagged_as_meaningless():
    """15 samples over 29 s fits the planning-cycle swing, not a trend."""
    summary = summarise(_trace([420, 404, 427, 405], step_s=8.0), settle_s=0.0)
    assert "too short to mean anything" in format_summary(summary)


def test_a_long_window_is_reported_plainly():
    summary = summarise(_trace([300 + i for i in range(40)], step_s=10.0), settle_s=20.0)
    assert "too short to mean anything" not in format_summary(summary)
