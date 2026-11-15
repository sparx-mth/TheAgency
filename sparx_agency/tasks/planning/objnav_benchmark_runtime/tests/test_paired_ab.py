"""The paired A/B report: joins on episode id, the flips, the integrity check, the decision rule, the pairs needed."""
import json

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime import paired_ab


def _row(ep, success, spl, steps=100, glances=0, cues=0, path=5.0, answers=None):
    policy = {"glances": {"stats": {"started": glances + cues, "cues": cues}}}
    if answers is not None:
        policy["llm_answers"] = answers
    return {"episode_id": ep, "success": success, "spl": spl, "steps": steps, "path_length_m": path,
            "agent_info": {"policy": policy}}


def _write(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    (root / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_identical_arms_differ_by_nothing_and_pass_the_integrity_check(tmp_path):
    rows = [_row("s/%03d" % i, i % 3 != 0, 0.5 if i % 3 else 0.0) for i in range(30)]
    _write(tmp_path / "a", rows)
    _write(tmp_path / "b", list(reversed(rows)))
    s = paired_ab.compare(paired_ab.load_rows(str(tmp_path / "a")), paired_ab.load_rows(str(tmp_path / "b")), 200)
    assert s["pairs"] == 30 and s["sr_diff"] == 0 and s["sr_ci"] == [0, 0] and s["mcnemar_p"] == 1
    assert s["flip_rate"] == 0 and s["b_never_glanced"] == 30 and not s["b_never_glanced_but_differs"]
    assert s["decision"] == "OFF for now: the SR gain is not significant at this size"


def test_flips_are_counted_both_ways_and_an_unglanced_episode_that_differs_makes_the_run_unreadable(tmp_path):
    a = {"e1": _row("e1", False, 0.0), "e2": _row("e2", True, 0.6), "e3": _row("e3", True, 0.4)}
    b = {"e1": _row("e1", True, 0.3, glances=2), "e2": _row("e2", False, 0.0, glances=1),
         "e3": _row("e3", True, 0.4, steps=101)}
    s = paired_ab.compare(a, b, 200)
    assert s["flips_to_success"] == ["e1"] and s["flips_to_failure"] == ["e2"]
    assert s["flip_rate"] == pytest.approx(2 / 3)
    assert s["b_never_glanced_but_differs"] == ["e3"], "no glance in B, yet a different end"
    assert s["decision"].startswith("not readable")


def test_the_decision_rule_needs_sr_up_and_no_significant_spl_drop():
    base = {"b_never_glanced_but_differs": []}
    assert paired_ab.decision(dict(base, sr_ci=[0.01, 0.05], spl_ci=[-0.02, 0.01])) == "keep geometric glances ON"
    assert paired_ab.decision(dict(base, sr_ci=[0.01, 0.05], spl_ci=[-0.04, -0.01])).startswith("OFF: SPL drops")
    assert paired_ab.decision(dict(base, sr_ci=[-0.01, 0.05], spl_ci=[-0.02, 0.01])).startswith("OFF for now")


def test_pairs_needed_follow_connors_approximation():
    assert paired_ab.mcnemar_pairs_needed(0.10, 0.03) == 870
    assert paired_ab.mcnemar_pairs_needed(0.05, 0.02) == 979
    assert paired_ab.mcnemar_pairs_needed(0.0, 0.02) is None and paired_ab.mcnemar_pairs_needed(0.1, 0.0) is None


def test_arm_b_answer_counts_and_the_cli(tmp_path, capsys):
    _write(tmp_path / "a", [_row("e1", True, 0.5), _row("e2", False, 0.0)])
    _write(tmp_path / "b", [_row("e1", True, 0.5, answers={"replayed": 4, "repeated": 1, "asked": 0}),
                            _row("e2", True, 0.2, glances=1, answers={"replayed": 2, "repeated": 0, "asked": 3})])
    assert paired_ab.main([str(tmp_path / "a"), str(tmp_path / "b"), "--resamples", "100",
                           "--hours-per-pair", "0.02", "--json", str(tmp_path / "s.json")]) == 0
    out = capsys.readouterr().out
    assert "6 replayed from A (60 %)" in out and "1 to success, 0 to failure" in out
    assert json.loads((tmp_path / "s.json").read_text())["pairs_for_decision"]
