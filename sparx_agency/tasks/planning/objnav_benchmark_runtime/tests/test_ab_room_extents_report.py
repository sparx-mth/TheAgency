"""The A/B report of the room-extents mechanism: arms paired by episode, the parting round found, coverage-driven choices flagged."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime import ab_room_extents_report as report


def row(episode_id, success, spl, dtg, steps, orders, readings=None, target="toilet"):
    """An ``episodes.jsonl`` row with the ledgers the report reads: one oracle round per entry of ``orders``."""
    events = [{"step": 10 * (k + 1), "reading": (readings or [{}] * len(orders))[k], "pass_verdict": "first", "house": "rooms found: none yet",
               "p_present": 0.5} for k in range(len(orders))]
    solves = [{"order": list(order), "status": "ok"} for order in orders]
    return {"episode_id": episode_id, "success": success, "spl": spl, "soft_spl": spl, "distance_to_goal_m": dtg,
            "steps": steps, "path_length_m": 10.0, "termination": "stop" if success else "step_limit",
            "target_category": target, "scene_id": episode_id.split("/")[0],
            "agent_info": {"policy": {"room_search_loop": {"estimate_events": events}, "solver_records": solves,
                                      "llm_answers": {"replayed": 3, "repeated": 0, "asked": 2}}}}


def node(pid, label, prob, why, **extent):
    fields = dict(id=pid, kind="room", label=label, tentative=False, area_m2=12.0, frontier_clusters=1, searched_s=0.0,
                  last_inside_ago_s=None, here=False, objects=[], home=False, scanned=None, scanned_ago_s=None, exits=1,
                  doorways=0, glimpsed=[], unseen_m2=0.0)
    fields.update(extent)
    return fields, prob, why


def recording(root, episode_id, step, nodes):
    key = hashlib.sha256(episode_id.encode()).hexdigest()[:12]
    directory = Path(root) / episode_id.split("/")[0] / "recordings" / key
    directory.mkdir(parents=True)
    probs = {str(n[0]["id"]): n[1] for n in nodes}
    reasons = {str(n[0]["id"]): n[2] for n in nodes}
    rows = [{"step": step - 1, "method": {}}, {"step": step, "method": {"reasoning": {"nodes": [n[0] for n in nodes],
                                                                                        "oracle": {"probs": probs, "reasons": reasons}}}}]
    (directory / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def write_arms(tmp_path):
    a_root, b_root = tmp_path / "without", tmp_path / "with"
    for root in (a_root, b_root):
        (root / "scene").mkdir(parents=True)
    # Episode 1: the arms part at round 1 -- arm-with sends the robot to the small unbounded room (3) instead of the
    # big entered room (2), on coverage -- and arm-with finds the toilet.
    a_rows = [row("scene/000001", False, 0.0, 4.2, 500, [[2, 3], [2, 3]]),
              row("scene/000002", True, 0.6, 0.3, 120, [[5, 6]]),                 # unchanged
              row("scene/000003", True, 0.4, 0.2, 300, [[7, 8], [7, 8]])]        # arm-with loses it, parting elsewhere
    b_rows = [row("scene/000001", True, 0.5, 0.4, 210, [[2, 3], [3, 2]],
                  readings=[{}, {"home": "bathroom", "house": "room 3 mostly unseen", "stage": "early"}]),
              row("scene/000002", True, 0.6, 0.3, 120, [[5, 6]]),
              row("scene/000003", False, 0.0, 3.1, 500, [[7, 8], [7, 8]])]
    for root, rows in ((a_root, a_rows), (b_root, b_rows)):
        (root / "scene" / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    recording(a_root, "scene/000001", 20, [node(2, "living_room", 0.4, "big room, little searched"),
                                            node(3, "unknown", 0.2, "small unknown room")])
    recording(b_root, "scene/000001", 20, [node(3, "unknown", 0.55, "small room, 2 walls unseen, at least 3m2 left, bathroom missing",
                                                 area_m2=3.6, footprint_m2=4.0, length_m=2.5, width_m=1.6, length_final=False, width_final=False,
                                                 walls_seen=2, bounded=False, covered_fraction=0.3, covered_m2=1.2, left_m2=2.8,
                                                 unknown_behind_m2=9.0),
                                            node(2, "living_room", 0.05, "living room 90% covered, 2m2 left, toilets not here",
                                                 footprint_m2=24.0, length_m=6.0, width_m=4.0, walls_seen=4, bounded=True,
                                                 covered_fraction=0.9, covered_m2=21.6, left_m2=2.4, unknown_behind_m2=1.0)])
    return str(a_root), str(b_root)


def test_the_report_pairs_the_arms_finds_the_parting_round_and_flags_the_coverage_driven_choice(tmp_path):
    a_root, b_root = write_arms(tmp_path)
    rep = report.build_report(a_root, b_root)
    assert rep["common_episodes"] == 3
    assert rep["without"]["sr"] == pytest.approx(2 / 3) and rep["with"]["sr"] == pytest.approx(2 / 3)
    assert rep["without"]["dtg_m"] == pytest.approx((4.2 + 0.3 + 0.2) / 3) and rep["with"]["dtg_m"] == pytest.approx((0.4 + 0.3 + 3.1) / 3)
    assert rep["changes"]["improved"] == ["scene/000001"] and rep["changes"]["worse"] == ["scene/000003"]
    assert rep["changes"]["same_outcome"] == 1 and rep["changes"]["spl_up"] == 1 and rep["changes"]["spl_down"] == 1
    assert rep["changes"]["dtg_better"] == 1 and rep["changes"]["dtg_worse"] == 1
    assert rep["paired"]["pairs"] == 3 and rep["paired"]["flips_to_success"] == ["scene/000001"]
    assert rep["parting"] == {"at_an_oracle_round": 1, "orders_agreed_throughout": 2}
    improved, worse = rep["changed_episodes"]
    assert improved["parted"]["round"] == 1 and improved["parted"]["without_order"] == [2, 3] and improved["parted"]["with_order"] == [3, 2]
    assert improved["coverage_driven"] and "at least 3m2 left" in improved["coverage_words"]
    lines = [n["line"] for n in improved["parted"]["with_nodes"]]
    assert lines[0].startswith("id=3  ROOM  type=unknown  size=at least 4m2") and "covered=1m2 (at most 30% of the room" in lines[0]
    assert improved["parted"]["with_nodes"][0]["p"] == 0.55
    assert improved["parted"]["without_nodes"][0]["line"].startswith("id=2  ROOM  type=living_room  size=12m2  seen: nothing yet")
    assert worse["parted"]["where"].startswith("never at an oracle round"), "the orders agreed: the loss came from elsewhere"
    assert not worse["coverage_driven"]
    assert rep["coverage_examples"] == [improved]
    text = report.markdown(rep)
    for fragment in ("| without (", "| with (", "improved (failed without, found with): **1**", "worse (found without, failed with): **1**",
                     "scene/000001 (toilet) -- IMPROVED  [coverage-driven]", "parted at oracle round 1",
                     "covered=1m2 (at most 30% of the room, at least 3m2 left)", "-> p=0.55",
                     "scene/000003 (toilet) -- WORSE", "parted: never at an oracle round",
                     "Examples where coverage changed the oracle's decision"):
        assert fragment in text, fragment


def test_the_command_line_writes_both_files(tmp_path, capsys):
    a_root, b_root = write_arms(tmp_path)
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    assert report.main([a_root, b_root, "--markdown", str(md), "--json", str(js), "--levels", ""]) == 0
    assert md.read_text().startswith("# Room size and coverage: A/B on 3 common episodes")
    saved = json.loads(js.read_text())
    assert saved["changes"]["improved"] == ["scene/000001"]
    assert "A/B on 3 common episodes" in capsys.readouterr().out


def test_an_arm_paired_with_itself_changes_nothing(tmp_path):
    a_root, _ = write_arms(tmp_path)
    rep = report.build_report(a_root, a_root)
    assert rep["changes"]["improved"] == [] and rep["changes"]["worse"] == [] and rep["changed_episodes"] == []
    assert rep["parting"]["at_an_oracle_round"] == 0


REAL = os.path.expanduser("~/GIT/TheAgency/runs/rerun2_relook_lookdown_20261105")


@pytest.mark.skipif(not os.path.isdir(REAL), reason="no recorded campaign on this machine")
def test_a_real_campaign_root_pairs_with_itself_without_a_difference():
    rep = report.build_report(REAL, REAL)
    assert rep["common_episodes"] > 5 and rep["changes"]["improved"] == [] and rep["changes"]["worse"] == []
    assert rep["parting"]["at_an_oracle_round"] == 0
    assert "A/B on %d common episodes" % rep["common_episodes"] in report.markdown(rep)
