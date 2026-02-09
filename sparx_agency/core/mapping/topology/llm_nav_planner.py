# core/mapping/topology/llm_nav_planner.py
"""
LLM-based navigation planner (inspired by MORE, Werby et al. 2025).

Two-stage pipeline:
  1. **Filter** – an LLM prunes irrelevant objects from the room–object tree.
  2. **Plan**  – an LLM produces human-readable navigation instructions
                 given the pruned tree, the robot's position, and a user query.

The Voronoi graph supplies shortest-path context so the LLM can describe
a concrete route (which rooms to traverse, which turns to take).

Usage:
    planner = NavPlanner(llm_call=my_llm_fn)
    pruned  = planner.filter_graph(room_object_graph, task="find the TV")
    instructions = planner.plan_route(
        room_object_graph=pruned,
        voronoi_graph=vor_graph,
        separated_graph=sep_graph,
        robot_pos=(27, 34),
        task="find the TV",
    )
"""

from __future__ import annotations

import json
import textwrap
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .room_object_graph import NodeType


# ── Helpers ──────────────────────────────────────────────────────────────────

def _room_object_dict(G: nx.DiGraph) -> Dict[str, List[str]]:
    """Extract {room_name: [object_names, …]} from the hierarchical graph."""
    rooms = [n for n in G.successors("root")]
    result = {}
    for room in sorted(rooms):
        objs = [
            n for n in G.successors(room)
            if G.nodes[n].get("node_type") == NodeType.OBJECT
        ]
        result[room] = sorted(objs)
    return result


def _robot_room(
    robot_pos: Tuple[float, float],
    separated_graph: nx.Graph,
) -> Optional[str]:
    """Find which room the robot is in by nearest Voronoi node."""
    if len(separated_graph) == 0:
        return None
    nodes = np.array(list(separated_graph.nodes))
    dists = np.linalg.norm(nodes - np.array(robot_pos), axis=1)
    closest = tuple(nodes[int(np.argmin(dists))])
    # Each node in the separated graph should carry a room_id attribute
    data = separated_graph.nodes[closest]
    room_id = data.get("room_id")
    if room_id is None:
        # Fall back: find which connected component it belongs to
        for i, comp in enumerate(nx.connected_components(separated_graph)):
            if closest in comp:
                return NodeType.roomname(i)
        return None
    if isinstance(room_id, int):
        return NodeType.roomname(room_id)
    return str(room_id)


def _shortest_room_path(
    separated_graph: nx.Graph,
    src_room: str,
    dst_room: str,
) -> List[str]:
    """
    Return the sequence of room labels along the shortest path
    between two rooms in the separated Voronoi graph.

    Works by building a coarse room-adjacency graph from the
    separated_graph's node room_id labels.
    """
    # Build a lightweight room-level graph
    room_graph = nx.Graph()
    for u, v in separated_graph.edges():
        ru = separated_graph.nodes[u].get("room_id")
        rv = separated_graph.nodes[v].get("room_id")
        if ru is None or rv is None:
            continue
        rn_u = NodeType.roomname(ru) if isinstance(ru, int) else str(ru)
        rn_v = NodeType.roomname(rv) if isinstance(rv, int) else str(rv)
        if rn_u != rn_v:
            room_graph.add_edge(rn_u, rn_v)
        else:
            room_graph.add_node(rn_u)

    if src_room == dst_room:
        return [src_room]
    try:
        return nx.shortest_path(room_graph, src_room, dst_room)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [src_room, dst_room]  # fallback


def _find_object_room(
    G: nx.DiGraph,
    object_query: str,
) -> Optional[Tuple[str, str]]:
    """
    Fuzzy-match an object query against graph nodes.
    Returns (room_name, object_node_name) or None.
    """
    query_lower = object_query.lower()
    for room in G.successors("root"):
        for obj in G.successors(room):
            if G.nodes[obj].get("node_type") != NodeType.OBJECT:
                continue
            if query_lower in obj.lower() or obj.lower() in query_lower:
                return room, obj
    return None


# ── Stage 1: LLM object filter ──────────────────────────────────────────────

_FILTER_PROMPT = textwrap.dedent("""\
You are assisting a navigation robot. Its task is: {task}

The robot has discovered the following rooms and objects:
{room_objects}

Remove objects that are clearly IRRELEVANT to this task.
Keep any object that might serve as a landmark or is related to the task.

Respond ONLY with a JSON object mapping each room to its filtered object list:
{{"room_name": ["kept_obj1", "kept_obj2"], ...}}
""")


def build_filter_prompt(room_obj_dict: Dict[str, List[str]], task: str) -> str:
    room_lines = "\n".join(
        f"  - {room}: [{', '.join(objs)}]" for room, objs in room_obj_dict.items()
    )
    return _FILTER_PROMPT.format(task=task, room_objects=room_lines)


def apply_filter_response(
    G: nx.DiGraph,
    kept: Dict[str, List[str]],
) -> nx.DiGraph:
    """Remove object nodes NOT in *kept* from a copy of G."""
    G = G.copy()
    for room in list(G.successors("root")):
        kept_set = {o.lower() for o in kept.get(room, kept.get(room, []))}
        for obj in list(G.successors(room)):
            if G.nodes[obj].get("node_type") != NodeType.OBJECT:
                continue
            if obj.lower() not in kept_set:
                G.remove_node(obj)
    return G


# ── Stage 2: LLM route planner ──────────────────────────────────────────────

_PLAN_PROMPT = textwrap.dedent("""\
You are a navigation assistant for an indoor robot.

TASK: {task}

CURRENT POSITION: The robot is in "{current_room}".

HOUSE MAP (rooms → objects):
{room_objects}

ROUTE (rooms to traverse): {route}

{target_info}

Generate a short, clear, human-readable navigation instruction.
- Use simple terms: "go straight", "turn left/right", "enter the room on your left".
- Mention landmark objects the robot will pass.
- End by describing where the target object is inside its room, relative to other objects.
- Output ONLY the instruction paragraph, nothing else.
""")


def build_plan_prompt(
    room_obj_dict: Dict[str, List[str]],
    current_room: str,
    route: List[str],
    task: str,
    target_room: Optional[str] = None,
    target_obj: Optional[str] = None,
) -> str:
    room_lines = "\n".join(
        f"  - {room}: [{', '.join(objs)}]" for room, objs in room_obj_dict.items()
    )
    route_str = " → ".join(route) if route else current_room

    if target_obj and target_room:
        neighbours = room_obj_dict.get(target_room, [])
        target_info = (
            f"TARGET: '{target_obj}' is in '{target_room}' "
            f"(also contains: {', '.join(n for n in neighbours if n != target_obj)})."
        )
    else:
        target_info = "TARGET: not found in any discovered room. Suggest which room to explore."

    return _PLAN_PROMPT.format(
        task=task,
        current_room=current_room,
        room_objects=room_lines,
        route=route_str,
        target_info=target_info,
    )


# ── Main planner class ──────────────────────────────────────────────────────

class NavPlanner:
    """
    Two-stage LLM navigation planner.

    Args:
        llm_call: A function ``(prompt: str) -> str`` that sends a prompt
                  to any LLM backend and returns the text response.
                  This keeps the planner backend-agnostic (works with
                  OpenAI, Ollama, Anthropic, local models, etc.).
    """

    def __init__(self, llm_call: Callable[[str], str]) -> None:
        self.llm_call = llm_call

    # ── Stage 1 ─────────────────────────────────────────────────────────

    def filter_graph(
        self,
        room_object_graph: nx.DiGraph,
        task: str,
    ) -> nx.DiGraph:
        """
        Ask the LLM to prune irrelevant objects.  Returns a pruned copy.
        """
        rod = _room_object_dict(room_object_graph)
        prompt = build_filter_prompt(rod, task)
        raw = self.llm_call(prompt)

        # Parse JSON from response (tolerant of markdown fences)
        raw = raw.strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        try:
            kept = json.loads(raw)
        except json.JSONDecodeError:
            # If parsing fails, keep everything (safe fallback)
            kept = rod

        return apply_filter_response(room_object_graph, kept)

    # ── Stage 2 ─────────────────────────────────────────────────────────

    def plan_route(
        self,
        room_object_graph: nx.DiGraph,
        separated_graph: nx.Graph,
        robot_pos: Tuple[float, float],
        task: str,
    ) -> str:
        """
        Produce human-readable navigation instructions.

        Args:
            room_object_graph: Hierarchical graph (optionally pre-filtered).
            separated_graph:   Door-separated Voronoi graph with ``room_id``
                               attributes on each node.
            robot_pos:         Robot's current (row, col) in grid coordinates.
            task:              Natural-language user request,
                               e.g. "find the TV" or "go to the kitchen".

        Returns:
            A plain-English navigation instruction string.
        """
        rod = _room_object_dict(room_object_graph)
        current_room = _robot_room(robot_pos, separated_graph)

        # Try to locate the target object/room
        match = _find_object_room(room_object_graph, task)
        if match:
            target_room, target_obj = match
        else:
            target_room, target_obj = None, None

        # Shortest room-level path
        if current_room and target_room:
            route = _shortest_room_path(separated_graph, current_room, target_room)
        else:
            route = [current_room or "unknown"]

        prompt = build_plan_prompt(
            rod, current_room or "unknown", route, task, target_room, target_obj,
        )
        return self.llm_call(prompt)

    # ── Combined convenience method ─────────────────────────────────────

    def run(
        self,
        room_object_graph: nx.DiGraph,
        separated_graph: nx.Graph,
        robot_pos: Tuple[float, float],
        task: str,
    ) -> Tuple[nx.DiGraph, str]:
        """
        Full pipeline: filter → plan.

        Returns:
            (pruned_graph, navigation_instructions)
        """
        pruned = self.filter_graph(room_object_graph, task)
        instructions = self.plan_route(pruned, separated_graph, robot_pos, task)
        return pruned, instructions