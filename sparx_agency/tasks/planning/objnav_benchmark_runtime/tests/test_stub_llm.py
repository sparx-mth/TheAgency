"""The stand-in for the language model of the paired CPU checks: fixed replies, the same in every arm."""
from __future__ import annotations

import pytest

from sparx_agency.core.mapping.topology.search_node_oracle import (
    OPENING, ROOM, SearchContext, SearchNode, SearchNodeOracle, format_prompt)
from sparx_agency.tasks.planning.objnav_benchmark_runtime import stub_llm
from sparx_agency.tasks.planning.objnav_benchmark_runtime.stub_llm import (
    STUB_CONFIDENCE, STUB_LABEL, STUB_TOTAL_PERCENT, StubLLMClient, stub_reply)

ROOMS = [SearchNode(0, ROOM, "kitchen", area_m2=12.0), SearchNode(3, ROOM, "unknown", area_m2=5.0),
         SearchNode(11, ROOM, "bedroom", area_m2=14.0, objects=("bed",))]


def test_a_labeller_prompt_is_answered_with_the_one_label():
    reply = stub_reply("You classify rooms.", "Room observed objects: sofa x1\nGeometry: 12 m2")
    assert reply == {"label": STUB_LABEL, "confidence": STUB_CONFIDENCE, "reasoning": "stub"}
    assert STUB_LABEL == "living_room" and STUB_CONFIDENCE == 0.8, "the replies of the check of 2026-11-15, kept"


def test_the_nodes_of_an_oracle_prompt_share_ninety_percent_equally():
    """Three nodes 30 % each, nine 10 %, never under 1 %: no opinion about any of them, so RPT*'s order is the
    travel cost's and the exploration state's alone -- and the same in both arms of a check."""
    user = format_prompt("toilet", ROOMS, SearchContext("rooms found: kitchen (entered)", action_time_s=1.0))
    reply = stub_reply("system", user)
    assert [row["id"] for row in reply["nodes"]] == [0, 3, 11] and {row["p"] for row in reply["nodes"]} == {30}
    assert reply["elsewhere"] == 10 and STUB_TOTAL_PERCENT == 90
    assert "pass" not in reply and "home" not in reply, "no reading and no verdict: the stand-in has no opinion"
    for count, share in ((1, 90), (2, 45), (4, 22), (9, 10), (40, 2), (95, 1), (200, 1)):
        nodes = [SearchNode(i, ROOM, "unknown") for i in range(count)]
        rows = stub_reply("", format_prompt("bed", nodes))["nodes"]
        assert len(rows) == count and {row["p"] for row in rows} == {share}, count
    # The reply is one the real parser takes: every node scored, none omitted.
    result = SearchNodeOracle.score(reply, ROOMS)
    assert result.probs == {0: 0.3, 3: 0.3, 11: 0.3} and result.omitted == ()


def test_the_node_ids_are_read_off_the_lines_whatever_else_the_prompt_says():
    """The stand-in keys on ``id=<int>`` at the head of a line and on nothing else, so a change to the wording of
    the prompt (2026-11-16: a new system prompt, new fields on the room line) leaves its answers -- and a check's
    arms -- as they were."""
    door = SearchNode(200001, OPENING, "doorway", via="room 3 (type=unknown)")
    user = format_prompt("couch", ROOMS + [door], SearchContext("rooms found: none yet"))
    assert [row["id"] for row in stub_reply("", user)["nodes"]] == [0, 3, 11, 200001]
    assert stub_reply("", "TARGET: bed\n\nHOUSE: id=7 is not a node line\n\nNODES (0):\n\n")["nodes"] == []


def test_the_client_stands_where_the_verified_client_does_and_names_itself():
    client = StubLLMClient()
    identity = client.health()
    assert identity["model"] == "stub" and "stand-in" in identity["stub"] and identity["digest"] is None
    assert client.probe() is None and client.answer_stats() is None and client.set_answer_scope("x") is None
    reply = client.chat_json("system", format_prompt("bed", ROOMS), reasoning=True)
    assert client.calls == 1 and len(reply["nodes"]) == 3


def test_install_replaces_the_runtimes_client_and_main_refuses_an_unknown_benchmark(monkeypatch, capsys):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.services import llm as service
    monkeypatch.setattr(service, "VerifiedLLMClient", service.VerifiedLLMClient)      # restored after the test
    assert stub_llm.install() is service and service.VerifiedLLMClient is StubLLMClient
    assert stub_llm.main(["habitat"]) == 2 and "usage" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        stub_llm.main(["mp3d", "--no-such-flag"])
