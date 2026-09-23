"""Reading a run back out of its log — the parsing, not the flying.

Every number `campaign_report` prints is scraped out of `nodes.log` with a
regular expression, which is a class of code that fails silently and reads as
"the run did not do that". This pins the shapes the nodes actually emit.
"""
from __future__ import annotations

from sparx_agency.tasks.planning.sjtu_internvla_n1.scripts.campaign_report import Run

# Verbatim from a real run's nodes.log, ros2 launch prefixes and all.
_PREFIX = "[n1_policy_node-1] [INFO] [1787828595.422942621] [n1_policy_node]: "
_RECORDER = "[n1_run_recorder_node-3] [INFO] [1787828593.857043616] [n1_run_recorder_node]: "


def _run(tmp_path, lines):
    log = tmp_path / "nodes.log"
    log.write_text("\n".join(lines) + "\n")
    return Run(str(log))


def test_it_reads_what_the_nodes_actually_write(tmp_path):
    run = _run(tmp_path, [
        _PREFIX + "committed #1: 33 pts, 1.42 m, from (-0.09, 11.83) after look-down frame [curve]",
        _PREFIX + "committed #2: 2 pts, 0.25 m, from (-0.74, 10.51) after flown [action]",
        _PREFIX + "turn #21: +15.0 deg to heading 13.5 deg (TURN_LEFT)",
        _PREFIX + "N1 FPS  System1=7.0 Hz  System2=0.2 Hz  (action=NO_ACTION)",
        _RECORDER + "N1 COVERAGE  4.2% seen (48 of 1140 m2)",
        _RECORDER + "N1 COVERAGE FINAL  15.7% seen (179 of 1140 m2)",
    ])
    assert len(run.commits) == 2
    assert len(run.curves) == 1 and len(run.actions) == 1
    assert run.turns == [15.0]
    assert run.s2_fps == 0.2
    assert (run.seen_pct, run.seen_m2, run.floor_m2) == (15.7, 179.0, 1140.0)


def test_the_final_coverage_line_wins_over_the_periodic_ones(tmp_path):
    run = _run(tmp_path, [
        _RECORDER + "N1 COVERAGE  4.2% seen (48 of 1140 m2)",
        _RECORDER + "N1 COVERAGE  11.7% seen (133 of 1140 m2)",
        _RECORDER + "N1 COVERAGE FINAL  15.7% seen (179 of 1140 m2)",
    ])
    assert run.seen_pct == 15.7


def test_the_last_periodic_line_is_the_fallback_when_there_is_no_final(tmp_path):
    """A recorder killed outright never writes FINAL, and this must still report.

    The regex used to spell the gap after the label as a single space while the
    recorder writes two, so it matched only the FINAL line -- whose own `\\s+`
    absorbed the pair -- and this fallback was dead. No fixture containing a
    FINAL line can show that, which is why this case has its own test.
    """
    run = _run(tmp_path, [
        _RECORDER + "N1 COVERAGE  4.2% seen (48 of 1140 m2)",
        _RECORDER + "N1 COVERAGE  11.7% seen (133 of 1140 m2)",
    ])
    assert run.seen_pct == 11.7
    assert run.seen_m2 == 133.0


def test_a_run_from_before_coverage_existed_reports_nothing_rather_than_zero(tmp_path):
    run = _run(tmp_path, [
        _PREFIX + "committed #1: 33 pts, 1.42 m, from (0.00, 0.00) after flown [curve]",
    ])
    assert run.seen_pct is None, "no measurement is not the same as 0% seen"


def test_a_capsized_run_says_so(tmp_path):
    run = _run(tmp_path, [
        _PREFIX + "committed #1: 33 pts, 1.42 m, from (0.00, 0.00) after flown [curve]",
        "[trajectory_follower_node-2] [ERROR] [1.0] [trajectory_follower_node]: CAPSIZED",
    ])
    assert run.verdict == "CAPSIZED"
