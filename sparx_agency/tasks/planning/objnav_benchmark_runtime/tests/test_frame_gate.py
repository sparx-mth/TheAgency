import json

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.frame_gate import gate_logits, gate_verdict


def test_gate_logits_reads_the_logit_of_each_answered_view():
    rows = [{"room": "r1", "frame": "f1", "reply": json.dumps({"percent": 90.0, "logit": 2.2})},
            {"room": "r1", "frame": "f2", "reply": ""},
            {"room": "r2", "frame": "f3", "reply": json.dumps({"percent": 10.0, "logit": -2.2})}]
    assert gate_logits(rows) == {"f1": 2.2, "f3": -2.2}


def test_gate_is_kept_only_when_the_gain_over_random_excludes_zero():
    assert gate_verdict((0.03, 0.01, 0.06)) == "keep"
    assert gate_verdict((0.03, -0.01, 0.06)) == "not shown"
