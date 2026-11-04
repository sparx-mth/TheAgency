"""Probe the node oracle against a REAL model: latency, schema, and whether the judgement is the one asked for.

Not a test -- it needs a live Ollama and takes minutes on a CPU. Run from the
repo root with the CPU service up::

    LLM_BASE_URL=http://127.0.0.1:11434 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        .venv/bin/python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.probe_node_oracle

Five single-storey scenarios (the standard benchmarks never offer stairs), each
a judgement the LLM-first prompt of 2026-10-07 is supposed to produce:

1. **toilet, no bathroom yet, early** -- kitchen and living room scanned, a
   bedroom never entered, a 5 m2 unknown room never entered, a hallway with an
   unlooked doorway, action ~150 of 500. Expected: pass "first"; the small
   unknown room and the doorway carry the mass; the bedroom keeps a real share
   (an en-suite opens off it); the scanned kitchen and living room ~0.
2. **sofa, living room already found and scanned, early** -- the one living
   room is scanned without a sofa, two unknown rooms never entered (14 and
   9 m2), a bedroom scanned. Expected: the unknown rooms are NOT valued as a
   second living room (each under 40) and the scanned living room, where the
   sofa may have been missed, is at least a quarter of the best unknown room;
   the bedroom low.
3. **television, kitchen vs bedroom** -- a scanned kitchen, a never-entered
   bedroom, a never-entered unknown room, a scanned bathroom. Expected: the
   bedroom clearly above the kitchen and the bathroom; the bathroom ~0.
4. **toilet, late, everything scanned** -- a bathroom scanned 4 min ago with a
   sink and a bathtub, a bedroom scanned, a kitchen scanned, no opening, no
   frontier, action ~430 of 500. Expected: pass "second"; the bathroom is the
   best node at 15 or more; the kitchen ~0.
5. **chair, early, unexplored places left** -- a scanned living room with a
   sofa, a never-entered unknown 12 m2 room, an unlooked doorway off the
   hallway, action ~60 of 500. Expected: pass "first"; the unknown room and
   the doorway each at least 10; the scanned living room under the unknown room.

Prints the model's distribution and reasons, the wall time, and a PASS/FAIL
against those expectations. The expectations are coarse on purpose: this
checks the prompt is understood, not that the numbers are optimal.
"""
from __future__ import annotations

import os
import sys
import time

from sparx_agency.core.mapping.topology.llm_client import LLMClient, LLMConfig
from sparx_agency.core.mapping.topology.search_node_oracle import (
    OPENING, PASS_FIRST, PASS_SECOND, ROOM, SearchContext, SearchNode, SearchNodeOracle)

DOOR = 200001


def house(*parts):
    return "; ".join(parts)


SCENARIOS = [
    ("toilet",
     SearchContext(house("rooms found: hallway (entered), kitchen (scanned), living_room (scanned), bedroom (never entered)",
                         "1 room unidentified", "1 opening not yet looked into", "unexplored frontier still reachable",
                         "actions used about 150 of 500")),
     [SearchNode(2, ROOM, "kitchen", area_m2=12, frontier_clusters=0, searched_s=30, last_inside_ago_s=120,
                 objects=("oven", "refrigerator", "sink"), scanned="full rotation", scanned_ago_s=120),
      SearchNode(3, ROOM, "living_room", area_m2=24, frontier_clusters=1, searched_s=40, last_inside_ago_s=60,
                 objects=("sofa", "television"), scanned="full rotation", scanned_ago_s=60),
      SearchNode(5, ROOM, "bedroom", area_m2=14, frontier_clusters=1, objects=("bed",)),
      SearchNode(6, ROOM, "unknown", area_m2=5, frontier_clusters=1),
      SearchNode(7, ROOM, "hallway", area_m2=6, frontier_clusters=1, searched_s=10, last_inside_ago_s=0, here=True),
      SearchNode(DOOR, OPENING, "doorway", via="room 7 (type=hallway)")],
     lambda r: (r.pass_verdict == PASS_FIRST and r.probs[6] + r.probs[DOOR] >= 0.5 and r.probs[6] >= r.probs[5]
                and r.probs[5] >= 0.08 and r.probs[2] <= 0.05 and r.probs[3] <= 0.05)),
    ("sofa",
     SearchContext(house("rooms found: bedroom (scanned), living_room (scanned)", "2 rooms unidentified",
                         "0 openings not yet looked into", "unexplored frontier still reachable",
                         "actions used about 175 of 500")),
     [SearchNode(1, ROOM, "living_room", area_m2=22, frontier_clusters=0, searched_s=35, last_inside_ago_s=90,
                 objects=("television", "coffee table", "potted plant"), scanned="full rotation", scanned_ago_s=90),
      SearchNode(2, ROOM, "bedroom", area_m2=13, frontier_clusters=0, searched_s=30, last_inside_ago_s=40,
                 objects=("bed", "wardrobe"), scanned="full rotation", scanned_ago_s=40),
      SearchNode(3, ROOM, "unknown", area_m2=14, frontier_clusters=2),
      SearchNode(4, ROOM, "unknown", area_m2=9, frontier_clusters=1)],
     lambda r: (r.probs[3] < 0.4 and r.probs[4] < 0.4 and r.probs[2] <= 0.05
                and r.probs[1] >= 0.25 * max(r.probs[3], r.probs[4]))),
    ("television",
     SearchContext(house("rooms found: bathroom (scanned), bedroom (never entered), kitchen (scanned)", "1 room unidentified",
                         "0 openings not yet looked into", "unexplored frontier still reachable",
                         "actions used about 100 of 500")),
     [SearchNode(1, ROOM, "kitchen", area_m2=11, frontier_clusters=0, searched_s=30, last_inside_ago_s=80,
                 objects=("oven", "sink", "refrigerator"), scanned="full rotation", scanned_ago_s=80),
      SearchNode(2, ROOM, "bedroom", area_m2=15, frontier_clusters=1, objects=("bed",)),
      SearchNode(3, ROOM, "unknown", area_m2=20, frontier_clusters=2),
      SearchNode(4, ROOM, "bathroom", area_m2=5, frontier_clusters=0, searched_s=15, last_inside_ago_s=30,
                 objects=("toilet", "sink"), scanned="full rotation", scanned_ago_s=30)],
     lambda r: r.probs[2] > r.probs[1] and r.probs[2] > r.probs[4] and r.probs[4] <= 0.03 and r.probs[3] >= 0.2),
    ("toilet",
     SearchContext(house("rooms found: bathroom (scanned), bedroom (scanned), kitchen (scanned)", "every known room identified",
                         "0 openings not yet looked into", "no unexplored frontier left", "actions used about 425 of 500")),
     [SearchNode(1, ROOM, "bathroom", area_m2=5, frontier_clusters=0, searched_s=20, last_inside_ago_s=240,
                 objects=("sink", "bathtub"), scanned="full rotation", scanned_ago_s=240),
      SearchNode(2, ROOM, "bedroom", area_m2=14, frontier_clusters=0, searched_s=30, last_inside_ago_s=150,
                 objects=("bed", "wardrobe"), scanned="full rotation", scanned_ago_s=150),
      SearchNode(3, ROOM, "kitchen", area_m2=12, frontier_clusters=0, searched_s=30, last_inside_ago_s=60,
                 objects=("oven", "sink"), scanned="full rotation", scanned_ago_s=60)],
     lambda r: (r.pass_verdict == PASS_SECOND and r.probs[1] >= 0.15 and r.probs[1] > r.probs[2] > r.probs[3]
                and r.probs[3] <= 0.05)),
    ("chair",
     SearchContext(house("rooms found: hallway (entered), living_room (scanned)", "1 room unidentified",
                         "1 opening not yet looked into", "unexplored frontier still reachable",
                         "actions used about 50 of 500")),
     [SearchNode(1, ROOM, "living_room", area_m2=20, frontier_clusters=0, searched_s=25, last_inside_ago_s=20,
                 objects=("sofa", "television"), scanned="full rotation", scanned_ago_s=20),
      SearchNode(2, ROOM, "unknown", area_m2=12, frontier_clusters=2),
      SearchNode(3, ROOM, "hallway", area_m2=5, frontier_clusters=1, searched_s=5, last_inside_ago_s=0, here=True),
      SearchNode(DOOR, OPENING, "doorway", via="room 3 (type=hallway)")],
     lambda r: (r.pass_verdict == PASS_FIRST and r.probs[2] >= 0.10 and r.probs[DOOR] >= 0.10
                and r.probs[1] < r.probs[2])),
]


def main():
    cfg = LLMConfig.from_env()
    if os.environ.get("LLM_BASE_URL") is None:
        cfg.base_url = "http://127.0.0.1:11434"
    client = LLMClient(cfg)
    print("service %s | default %s | reasoning %s (timeout %.0fs, num_ctx %d)" % (
        cfg.base_url, cfg.model, cfg.reasoning_model, cfg.reasoning_timeout_s, cfg.reasoning_num_ctx))
    if not client.ping():
        print("no LLM service answers; start the CPU Ollama first")
        return 2
    oracle = SearchNodeOracle(client)
    failures = 0
    for target, context, nodes, expectation in SCENARIOS:
        print("\n=== TARGET %s ===" % target)
        print(oracle.prompt(target, nodes, context))
        started = time.monotonic()
        result = oracle.probabilities(target, nodes, context)
        seconds = time.monotonic() - started
        print("--- %.1f s | source=%s | p_present=%.2f | pass=%s | omitted=%s" % (
            seconds, result.source, result.p_present, result.pass_verdict, list(result.omitted)))
        for key, value in result.reading.items():
            print("  %s: %s" % (key, value))
        for node in nodes:
            print("  id=%-7d p=%.2f  %s" % (node.id, result.probs.get(node.id, 0.0), result.reasons.get(node.id, "")))
        ok = result.source == "llm" and expectation(result)
        print("  ->", "PASS" if ok else "FAIL (expectation not met)")
        failures += 0 if ok else 1
    print("\n%d of %d scenarios as expected" % (len(SCENARIOS) - failures, len(SCENARIOS)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
