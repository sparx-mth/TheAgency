#!/usr/bin/env python3
# tasks/mapping/topology/test_llm_nav_planner.py
"""
Test & demo for the LLM navigation planner.

Tests:
  1. Unit tests with a dummy LLM (no Ollama needed)  — always runs
  2. Integration test with Ollama llama3.1:8b          — needs `ollama run llama3.1:8b`

Run:
    python test_llm_nav_planner.py              # dummy only
    python test_llm_nav_planner.py --ollama     # include Ollama tests
    python test_llm_nav_planner.py --task "find the oven"   # custom task with Ollama
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np

# ── Direct file imports (bypass __init__.py chains that pull in ROS2) ────────
import importlib.util

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TOPO_DIR = os.path.abspath(os.path.join(
    _SCRIPT_DIR, "..", "..", "..", "core", "mapping", "topology",
))


def _import_from_file(module_name: str, file_path: str, package: str = None):
    """Import a single .py file without triggering package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    if package:
        mod.__package__ = package
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Register stub parent packages so relative imports resolve
import types
for _pkg in [
    "sparx_agency",
    "sparx_agency.core",
    "sparx_agency.core.mapping",
    "sparx_agency.core.mapping.topology",
]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)


# 1) room_object_graph  (needed first — llm_nav_planner imports from it)
_rog = _import_from_file(
    "sparx_agency.core.mapping.topology.room_object_graph",
    os.path.join(_TOPO_DIR, "room_object_graph.py"),
    package="sparx_agency.core.mapping.topology",
)
NodeType = _rog.NodeType

# 2) llm_nav_planner
_planner_mod = _import_from_file(
    "sparx_agency.core.mapping.topology.llm_nav_planner",
    os.path.join(_TOPO_DIR, "llm_nav_planner.py"),
    package="sparx_agency.core.mapping.topology",
)
NavPlanner          = _planner_mod.NavPlanner
_room_object_dict   = _planner_mod._room_object_dict
_robot_room         = _planner_mod._robot_room
_shortest_room_path = _planner_mod._shortest_room_path
_find_object_room   = _planner_mod._find_object_room
build_filter_prompt = _planner_mod.build_filter_prompt
build_plan_prompt   = _planner_mod.build_plan_prompt
apply_filter_response = _planner_mod.apply_filter_response


# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures — a small house with 4 rooms
# ═══════════════════════════════════════════════════════════════════════════════

HOUSE_LAYOUT = {
    "room_0": {  # living room
        "objects": ["sofa", "tv", "coffee_table", "bookshelf"],
        "vor_nodes": [(5, 5), (5, 10), (5, 15)],
    },
    "room_1": {  # bedroom
        "objects": ["bed", "nightstand", "lamp", "wardrobe"],
        "vor_nodes": [(15, 5), (15, 10)],
    },
    "room_2": {  # kitchen
        "objects": ["fridge", "oven", "sink", "dining_table", "microwave"],
        "vor_nodes": [(25, 5), (25, 10), (25, 15)],
    },
    "room_3": {  # bathroom
        "objects": ["toilet", "shower", "mirror"],
        "vor_nodes": [(15, 20), (15, 25)],
    },
}

ROBOT_START = (5, 5)  # in room_0 (living room)


def build_test_room_object_graph() -> nx.DiGraph:
    """Build the hierarchical root → room → object graph."""
    G = nx.DiGraph()
    G.add_node("root", node_type=NodeType.ROOT, pos_map=(0, 0))

    for rid, (rname, info) in enumerate(HOUSE_LAYOUT.items()):
        centre = info["vor_nodes"][len(info["vor_nodes"]) // 2]
        G.add_node(
            rname,
            node_type=NodeType.ROOM,
            room_id=rid,
            pos_map=centre,
            frontier_points=set(),
            closed_doors=set(),
            open_doors=set(),
        )
        G.add_edge("root", rname)
        for obj in info["objects"]:
            G.add_node(
                obj,
                node_type=NodeType.OBJECT,
                name=obj,
                pos_map=centre,
                bbox=(0, 0, 10, 10),
                semantic_class="furniture",
                room_id=rid,
                closest_vor_node=centre,
            )
            G.add_edge(rname, obj)

    return G


def build_test_separated_graph() -> nx.Graph:
    """Build a separated Voronoi graph with room_id on each node."""
    S = nx.Graph()
    for rid, (rname, info) in enumerate(HOUSE_LAYOUT.items()):
        for node in info["vor_nodes"]:
            S.add_node(node, room_id=rid)
        # intra-room edges
        for i in range(len(info["vor_nodes"]) - 1):
            a, b = info["vor_nodes"][i], info["vor_nodes"][i + 1]
            S.add_edge(a, b, dist=np.linalg.norm(np.array(a) - np.array(b)))

    # inter-room edges (hallway connections)
    inter = [
        ((5, 15), (15, 5)),   # room_0 ↔ room_1
        ((5, 15), (15, 20)),  # room_0 ↔ room_3
        ((15, 10), (25, 10)), # room_1 ↔ room_2
    ]
    for a, b in inter:
        S.add_edge(a, b, dist=np.linalg.norm(np.array(a) - np.array(b)))

    return S


# ═══════════════════════════════════════════════════════════════════════════════
# LLM backends
# ═══════════════════════════════════════════════════════════════════════════════

def dummy_llm(prompt: str) -> str:
    """Deterministic fake LLM for unit testing."""
    if "Remove objects" in prompt or "IRRELEVANT" in prompt:
        # Return a filter response that drops some items
        return json.dumps({
            "room_0": ["sofa", "tv", "coffee_table"],
            "room_1": ["bed", "wardrobe"],
            "room_2": ["fridge", "oven", "sink", "dining_table", "microwave"],
            "room_3": ["toilet", "shower"],
        })
    # Navigation instruction
    return (
        "From the living room, walk straight through the hallway past the "
        "bedroom on your left. Continue to the end and enter the kitchen. "
        "The fridge is on the right wall, next to the oven and across from "
        "the dining table."
    )


def ollama_backend(model: str = "llama3.1:8b") -> Callable[[str], str]:
    """Ollama backend via subprocess."""
    def call(prompt: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(prompt)
            tmp = f.name
        try:
            result = subprocess.run(
                f'cat "{tmp}" | ollama run {model}',
                shell=True, capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Ollama error: {result.stderr.strip()}")
            return result.stdout.strip()
        finally:
            os.unlink(tmp)
    return call


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_helpers():
    """Test pure graph-query helpers (no LLM needed)."""
    G = build_test_room_object_graph()
    S = build_test_separated_graph()

    # _room_object_dict
    rod = _room_object_dict(G)
    assert "room_0" in rod
    assert "tv" in rod["room_0"]
    assert "fridge" in rod["room_2"]
    print("  ✓ _room_object_dict")

    # _robot_room
    room = _robot_room(ROBOT_START, S)
    assert room == "room_0", f"Expected room_0, got {room}"
    print("  ✓ _robot_room")

    # _shortest_room_path
    path = _shortest_room_path(S, "room_0", "room_2")
    assert path[0] == "room_0"
    assert path[-1] == "room_2"
    assert len(path) >= 2
    print(f"  ✓ _shortest_room_path: {' → '.join(path)}")

    # _find_object_room
    match = _find_object_room(G, "fridge")
    assert match is not None
    assert match[0] == "room_2"
    assert match[1] == "fridge"
    print("  ✓ _find_object_room")

    # _find_object_room — not found
    assert _find_object_room(G, "spaceship") is None
    print("  ✓ _find_object_room (miss)")


def test_filter_dummy():
    """Test the filter stage with the dummy LLM."""
    G = build_test_room_object_graph()
    planner = NavPlanner(llm_call=dummy_llm)

    pruned = planner.filter_graph(G, task="find the fridge")

    # bookshelf should have been removed (not in dummy response for room_0)
    rod = _room_object_dict(pruned)
    assert "bookshelf" not in rod["room_0"], "bookshelf should have been filtered"
    assert "tv" in rod["room_0"], "tv should remain"
    assert "fridge" in rod["room_2"], "fridge must remain"
    # lamp and nightstand removed from room_1
    assert "lamp" not in rod["room_1"], "lamp should be filtered"
    assert "mirror" not in rod["room_3"], "mirror should be filtered"
    print("  ✓ filter_graph (dummy)")


def test_plan_dummy():
    """Test the full pipeline with the dummy LLM."""
    G = build_test_room_object_graph()
    S = build_test_separated_graph()
    planner = NavPlanner(llm_call=dummy_llm)

    pruned, instructions = planner.run(
        room_object_graph=G,
        separated_graph=S,
        robot_pos=ROBOT_START,
        task="find the fridge",
    )
    assert len(instructions) > 20, "Instructions too short"
    assert "fridge" in instructions.lower()
    print("  ✓ plan_route (dummy)")
    print(f"    Instructions: {instructions[:120]}...")


def test_prompts_are_reasonable():
    """Verify prompt structure makes sense."""
    G = build_test_room_object_graph()
    rod = _room_object_dict(G)

    fp = build_filter_prompt(rod, "find the TV")
    assert "find the TV" in fp
    assert "room_0" in fp
    assert "JSON" in fp
    print("  ✓ filter prompt structure")

    pp = build_plan_prompt(rod, "room_0", ["room_0", "room_2"], "find the fridge",
                           "room_2", "fridge")
    assert "room_0" in pp
    assert "fridge" in pp
    assert "room_0 → room_2" in pp
    print("  ✓ plan prompt structure")


def test_with_ollama(task: str = "find the fridge"):
    """Integration test with real Ollama llama3.1:8b."""
    print(f"\n{'─' * 60}")
    print(f"OLLAMA INTEGRATION TEST — task: \"{task}\"")
    print(f"{'─' * 60}")

    G = build_test_room_object_graph()
    S = build_test_separated_graph()

    llm = ollama_backend("llama3.1:8b")
    planner = NavPlanner(llm_call=llm)

    # Stage 1: Filter
    print("\n[Stage 1] Filtering irrelevant objects...")
    pruned = planner.filter_graph(G, task)
    rod_before = _room_object_dict(G)
    rod_after = _room_object_dict(pruned)

    for room in sorted(rod_before):
        removed = set(rod_before[room]) - set(rod_after.get(room, []))
        kept = rod_after.get(room, [])
        print(f"  {room}: kept={kept}" + (f"  removed={list(removed)}" if removed else ""))

    # Stage 2: Plan
    print("\n[Stage 2] Generating navigation instructions...")
    instructions = planner.plan_route(
        room_object_graph=pruned,
        separated_graph=S,
        robot_pos=ROBOT_START,
        task=task,
    )
    print(f"\n{'═' * 60}")
    print("NAVIGATION INSTRUCTIONS:")
    print(f"{'═' * 60}")
    print(instructions)
    print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test LLM navigation planner")
    parser.add_argument("--ollama", action="store_true", help="Run integration test with Ollama",  default=True)
    parser.add_argument("--task", type=str, default="find the fridge",
                        help="Task for Ollama test (default: 'find the fridge')")
    args = parser.parse_args()

    print("=" * 60)
    print("LLM NAV PLANNER — TEST SUITE")
    print("=" * 60)

    print("\n[1] Helper functions")
    test_helpers()

    print("\n[2] Filter (dummy LLM)")
    test_filter_dummy()

    print("\n[3] Full pipeline (dummy LLM)")
    test_plan_dummy()

    print("\n[4] Prompt structure")
    test_prompts_are_reasonable()

    print("\n✅ All unit tests passed!\n")

    if args.ollama:
        test_with_ollama(args.task)
    else:
        print("Skipping Ollama tests (pass --ollama to enable)")


if __name__ == "__main__":
    main()