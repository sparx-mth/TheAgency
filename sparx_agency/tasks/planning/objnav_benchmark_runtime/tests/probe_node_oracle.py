"""Probe the node oracle against a REAL model: latency, schema, and whether the rules are followed.

Not a test -- it needs a live Ollama and takes minutes on a CPU. Run from the
repo root with the CPU service up::

    LLM_BASE_URL=http://127.0.0.1:11434 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        .venv/bin/python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.probe_node_oracle

Three scenarios, each a judgement the prompt is supposed to produce and the
3B model could not:

1. **bed, ground floor mapped** -- a fully observed kitchen, the living room
   the robot stands in, a large unknown room, stairs up. Expected: kitchen ~0,
   stairs and the unknown room carry the mass, living room low.
2. **frying pan, upstairs** -- two bedrooms, a bathroom named from a sink only
   (``bathroom?``), a small unknown room, stairs down to a storey where a
   kitchen was found but barely searched. Expected: bathroom ~0 whatever the
   ``?``, bedrooms ~0, the known kitchen downstairs at least matches the
   unknown room that merely might be one.
3. **toilet, just came up** -- a bedroom, a bathroom-sized unknown room and a
   large unknown room, stairs down the robot arrived by 10 s ago. Expected:
   the unknown rooms carry the mass, the small one at least matches the
   bedroom (a toilet is not in a bedroom), the way back is not first.

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
    ROOM, STAIRS, SearchContext, SearchNode, SearchNodeOracle)

UP, DOWN = 100000, 100001

SCENARIOS = [
    ("bed",
     SearchContext("storey F0: rooms found: kitchen, living_room; 1 unknown; searched 2min; "
                   "2 rooms with frontier left; 2 storeys known to the building (this one at +0.0 m)"),
     [SearchNode(0, ROOM, "kitchen", area_m2=12, frontier_clusters=0, searched_s=40, last_inside_ago_s=30,
                 objects=("fridge", "sink", "oven")),
      SearchNode(1, ROOM, "living_room", area_m2=25, frontier_clusters=1, searched_s=20, last_inside_ago_s=0,
                 here=True, objects=("sofa", "television")),
      SearchNode(2, ROOM, "unknown", area_m2=18, frontier_clusters=3),
      SearchNode(UP, STAIRS, "stairs up", direction=1)],
     lambda p: p[0] <= 0.05 and p[UP] + p[2] >= 0.6 and p[1] <= 0.15),
    ("frying pan",
     SearchContext("storey F1: rooms found: bedroom, bedroom, bathroom; 1 unknown; searched 3min; "
                   "2 rooms with frontier left; 2 storeys known to the building (this one at +2.7 m)",
                   ("storey F0: rooms found: kitchen, hallway; 1 unknown; searched 20s; 3 rooms with frontier left",)),
     [SearchNode(0, ROOM, "bedroom", area_m2=14, frontier_clusters=0, searched_s=50, last_inside_ago_s=120,
                 objects=("bed", "wardrobe")),
      SearchNode(1, ROOM, "bedroom", area_m2=11, frontier_clusters=1, searched_s=30, last_inside_ago_s=60,
                 objects=("bed",)),
      SearchNode(2, ROOM, "bathroom", tentative=True, area_m2=6, frontier_clusters=1, searched_s=0,
                 objects=("sink",)),
      SearchNode(3, ROOM, "unknown", area_m2=9, frontier_clusters=2, here=True),
      SearchNode(DOWN, STAIRS, "stairs down", direction=-1, destination_visited=True,
                 destination="storey F0: rooms found: kitchen, hallway; 1 unknown; searched 20s; 3 rooms with frontier left")],
     lambda p: p[2] <= 0.05 and p[0] <= 0.05 and p[1] <= 0.05 and p[DOWN] >= 0.4 and p[DOWN] >= p[3]),
    ("toilet",
     SearchContext("storey F1: rooms found: bedroom; 2 unknown; searched 10s; 3 rooms with frontier left; "
                   "2 storeys known to the building (this one at +2.7 m)",
                   ("storey F0: rooms found: kitchen, living_room; searched 40s; 2 rooms with frontier left",)),
     [SearchNode(0, ROOM, "bedroom", area_m2=14, frontier_clusters=1, searched_s=10, last_inside_ago_s=0, here=True,
                 objects=("bed",)),
      SearchNode(1, ROOM, "unknown", area_m2=5, frontier_clusters=2),
      SearchNode(2, ROOM, "unknown", area_m2=16, frontier_clusters=3),
      SearchNode(DOWN, STAIRS, "stairs down", direction=-1, destination_visited=True,
                 destination="storey F0: rooms found: kitchen, living_room; searched 40s; 2 rooms with frontier left",
                 arrived_by=True, arrived_ago_s=10)],
     lambda p: p[1] + p[2] >= 0.45 and p[1] >= p[0] and p[DOWN] < max(p[1], p[2])),
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
        print("--- %.1f s | source=%s | p_present=%.2f elsewhere=%.2f | omitted=%s" % (
            seconds, result.source, result.p_present, result.elsewhere, list(result.omitted)))
        for key, value in result.reading.items():
            print("  %s: %s" % (key, value))
        for node in nodes:
            print("  id=%-7d p=%.2f  %s" % (node.id, result.probs.get(node.id, 0.0), result.reasons.get(node.id, "")))
        ok = result.source == "llm" and expectation(result.probs)
        print("  ->", "PASS" if ok else "FAIL (expectation not met)")
        failures += 0 if ok else 1
    print("\n%d of %d scenarios as expected" % (len(SCENARIOS) - failures, len(SCENARIOS)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())




