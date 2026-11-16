"""The scenarios ``probe_node_oracle.py`` asks a real model: partial maps, each with one ordinal claim.

A scenario is ``(name, target, context, nodes, the claim in words, probs ->
whether it holds)``. The claim is a coarse ranking that a reasoner who knows
homes would not get wrong -- which of two nodes is worth more -- and is
never shown to the model: it measures the model, it does not brief it.

``P_SET`` (twelve) is what a run sends by default: rooms only, size and
coverage on every line, finished rooms not offered, exits as attributes.
``O_SET`` (seven) is the first probe's set, kept so that results stay
comparable: finished rooms offered, two openings as nodes of their own (the
deprecated mode), and five scenarios on lines without size detail.

Data and its builders only: the runner is ``probe_node_oracle.py``.
"""
from __future__ import annotations

from sparx_agency.core.mapping.topology.search_node_oracle import OPENING, ROOM, SearchContext, SearchNode

DOOR = 200001


def context(*parts):
    """The HOUSE line of a scenario, with the benchmark's action time (durations are written in actions)."""
    return SearchContext("; ".join(parts), action_time_s=1.0)


def node(node_id, kind, label, **kwargs):
    """A node of the first probe's set, built field by field."""
    return SearchNode(node_id, kind, label, **kwargs)


def room(node_id, label, area, dims, covered, walls=4, open_m=0.0, tentative=False, frontier=0, exits=0, doorways=0,
         unseen=0.0, glimpsed=(), inside=None, searched=0, objects=(), connected=(), scanned=None, scanned_ago=None):
    """A room as a run renders it: size and coverage on the line.

    ``inside``: actions since the robot was last inside (0: it stands there; None: never entered). ``covered``:
    the share of the whole room, an upper bound when ``walls`` < 4 (a side is not mapped).
    """
    bounded = walls == 4
    return SearchNode(node_id, ROOM, label, tentative=tentative, area_m2=area, footprint_m2=area, length_m=dims[0],
                      width_m=dims[1], walls_seen=walls, bounded=bounded, length_final=bounded, width_final=True,
                      open_perimeter_m=open_m, covered_fraction=covered, covered_m2=area * covered,
                      left_m2=area * (1.0 - covered), frontier_clusters=frontier, exits=exits, doorways=doorways,
                      unseen_m2=unseen, glimpsed=tuple(glimpsed), last_inside_ago_s=inside, here=(inside == 0),
                      searched_s=searched, objects=tuple(objects), connected=tuple(connected), scanned=scanned,
                      scanned_ago_s=scanned_ago, target_seen=False)


def extent(area, length, width, walls=4, covered=None, open_m=0.0):
    """Room-extent fields: every side located when ``walls`` is 4, else a lower bound cut short along the length."""
    bounded = walls == 4
    fields = dict(footprint_m2=area, area_m2=area, length_m=length, width_m=width, walls_seen=walls, bounded=bounded,
                  length_final=bounded, width_final=True, open_perimeter_m=open_m)
    if covered is not None:
        fields.update(covered_fraction=covered, covered_m2=area * covered, left_m2=area * (1.0 - covered))
    return fields


#: ``(name, target, context, nodes, the claim in words, probs -> whether it holds)`` -- what a run sends by default.
P_SET = [
    ("P1 bed: three rooms never entered, one of them a bedroom", "bed",
     context("rooms found: bathroom? (never entered), bedroom? (never entered), hallway (entered), kitchen (never entered)",
             "every known room identified", "0 openings not yet looked into", "unexplored frontier still reachable",
             "actions used about 50 of 500"),
     [room(1, "bedroom", 9, (3.5, 2.5), 0.25, walls=3, open_m=2.5, tentative=True, frontier=1, objects=("wardrobe",),
           connected=("hallway",)),
      room(2, "kitchen", 8, (3.0, 2.5), 0.25, walls=3, open_m=2.5, frontier=1, objects=("oven", "sink"),
           connected=("hallway",)),
      room(3, "bathroom", 3, (2.0, 1.5), 0.35, walls=3, open_m=1.5, tentative=True, frontier=1, objects=("toilet",),
           connected=("hallway",))],
     "the bedroom above the kitchen and the bathroom",
     lambda p: p[1] > p[2] and p[1] > p[3]),
    ("P2 bed: one unidentified room covered, one barely seen", "bed",
     context("rooms found: hallway (entered), kitchen (scanned)", "2 rooms unidentified", "0 openings not yet looked into",
             "unexplored frontier still reachable", "actions used about 125 of 500"),
     [room(1, "unknown", 12, (4.0, 3.0), 0.93, inside=30, searched=45, objects=("chair",)),
      room(2, "unknown", 6, (3.0, 2.0), 0.35, walls=2, open_m=3.0, frontier=2),
      room(3, "hallway", 5, (4.0, 1.5), 0.8, inside=0, searched=15)],
     "the barely seen room above the room covered without a bed",
     lambda p: p[2] > p[1]),
    ("P3 refrigerator: two unidentified rooms, a bedroom covered", "refrigerator",
     context("rooms found: bedroom (entered), dining_room (scanned), hallway (entered)", "2 rooms unidentified",
             "0 openings not yet looked into", "unexplored frontier still reachable", "actions used about 100 of 500"),
     [room(4, "unknown", 8, (3.0, 2.5), 0.2, walls=3, open_m=2.5, frontier=1, connected=("dining_room",)),
      room(5, "unknown", 8, (3.0, 2.5), 0.2, walls=3, open_m=2.5, frontier=1, connected=("bedroom",)),
      room(6, "bedroom", 13, (4.0, 3.5), 0.95, inside=60, searched=30, objects=("bed", "wardrobe"))],
     "the covered bedroom below both unidentified rooms",
     lambda p: p[6] < min(p[4], p[5])),
    ("P4 toilet: a hallway whose doorways have not been looked through", "toilet",
     context("rooms found: hallway (entered), kitchen (scanned), living_room (entered)", "every known room identified",
             "2 openings not yet looked into", "unexplored frontier still reachable", "actions used about 100 of 500"),
     [room(1, "living_room", 24, (6.0, 4.0), 0.8, inside=30, searched=40, objects=("sofa", "television"),
           connected=("hallway",)),
      room(2, "hallway", 6, (4.0, 1.5), 0.8, inside=0, searched=10, frontier=2, exits=2, doorways=2, unseen=20.0,
           connected=("kitchen", "living_room"))],
     "the hallway with two unlooked doorways above the living room",
     lambda p: p[2] > p[1]),
    ("P5 toilet: a sink glimpsed through one room's exit", "toilet",
     context("rooms found: bedroom (entered), hallway (entered), office (entered)", "every known room identified",
             "2 openings not yet looked into", "unexplored frontier still reachable", "actions used about 150 of 500"),
     [room(1, "bedroom", 14, (4.0, 3.5), 0.7, inside=40, searched=20, frontier=1, exits=1, doorways=1, unseen=4.0,
           objects=("bed",), glimpsed=("sink",), connected=("hallway",)),
      room(2, "office", 12, (4.0, 3.0), 0.7, inside=10, searched=20, frontier=1, exits=1, doorways=1, unseen=4.0,
           objects=("chair", "table"), connected=("hallway",))],
     "the bedroom whose exit shows a sink above the office",
     lambda p: p[1] > p[2]),
    ("P6 sofa: an unidentified room barely seen, two covered rooms of other types", "sofa",
     context("rooms found: bathroom (entered), hallway (entered), kitchen (entered)", "1 room unidentified",
             "0 openings not yet looked into", "unexplored frontier still reachable", "actions used about 125 of 500"),
     [room(1, "unknown", 14, (4.0, 3.5), 0.2, walls=2, open_m=4.0, frontier=2, connected=("hallway",)),
      room(2, "bathroom", 5, (2.5, 2.0), 0.9, inside=60, searched=15, objects=("sink", "toilet")),
      room(3, "kitchen", 11, (4.0, 3.0), 0.9, inside=20, searched=25, objects=("oven", "refrigerator"))],
     "the unidentified room above the bathroom and the kitchen",
     lambda p: p[1] > max(p[2], p[3])),
    ("P7 television: a living room covered whole, a bedroom barely seen", "television",
     context("rooms found: bedroom (never entered), hallway (entered), living_room (entered)",
             "every known room identified", "0 openings not yet looked into", "unexplored frontier still reachable",
             "actions used about 125 of 500"),
     [room(1, "living_room", 22, (5.5, 4.0), 1.0, inside=20, searched=60, objects=("coffee table", "sofa")),
      room(2, "bedroom", 9, (3.5, 2.5), 0.25, walls=3, open_m=2.5, frontier=1, objects=("bed",),
           connected=("hallway",))],
     "the barely seen bedroom above the living room covered whole",
     lambda p: p[2] > p[1]),
    ("P8 toilet, late: a bathroom and a bedroom finished from outside", "toilet",
     context("rooms found: bathroom (scanned), bedroom (scanned), kitchen (scanned), living_room (scanned)",
             "every known room identified", "0 openings not yet looked into", "no unexplored frontier left",
             "actions used about 350 of 500"),
     [room(1, "bathroom", 4, (2.5, 1.5), 0.6, objects=("sink",), scanned="seen from another room's scan",
           scanned_ago=200),
      room(2, "bedroom", 12, (4.0, 3.0), 0.8, objects=("bed",), scanned="looked into, no frontier left",
           scanned_ago=120)],
     "the bathroom above the bedroom",
     lambda p: p[1] > p[2]),
    ("P9 sink: a bathroom, a bedroom and a living room", "sink",
     context("rooms found: bathroom? (never entered), bedroom (never entered), hallway (entered), living_room (entered)",
             "every known room identified", "0 openings not yet looked into", "unexplored frontier still reachable",
             "actions used about 75 of 500"),
     [room(1, "bathroom", 4, (2.5, 1.5), 0.3, walls=3, open_m=1.5, tentative=True, frontier=1, objects=("toilet",),
           connected=("hallway",)),
      room(2, "bedroom", 10, (3.5, 3.0), 0.3, walls=3, open_m=3.0, frontier=1, objects=("bed",), connected=("hallway",)),
      room(3, "living_room", 20, (5.0, 4.0), 0.6, inside=25, searched=20, frontier=1, objects=("sofa", "television"),
           connected=("hallway",))],
     "the bathroom above the bedroom and the living room",
     lambda p: p[1] > p[2] and p[1] > p[3]),
    ("P10 bed: an unidentified room barely seen, a hallway and a kitchen covered", "bed",
     context("rooms found: hallway (entered), kitchen (entered)", "1 room unidentified", "0 openings not yet looked into",
             "unexplored frontier still reachable", "actions used about 100 of 500"),
     [room(1, "hallway", 6, (4.0, 1.5), 0.8, inside=0, searched=10, connected=("kitchen",)),
      room(2, "unknown", 10, (3.5, 3.0), 0.2, walls=2, open_m=3.5, frontier=2, connected=("hallway",)),
      room(3, "kitchen", 11, (4.0, 3.0), 0.9, inside=30, searched=25, objects=("oven", "refrigerator", "sink"),
           connected=("hallway",))],
     "the unidentified room above the hallway and the kitchen",
     lambda p: p[2] > max(p[1], p[3])),
    ("P11 chair: a dining room covered whole, an office barely seen", "chair",
     context("rooms found: dining_room (entered), hallway (entered), office? (never entered)",
             "every known room identified", "0 openings not yet looked into", "unexplored frontier still reachable",
             "actions used about 150 of 500"),
     [room(1, "dining_room", 12, (4.0, 3.0), 1.0, inside=20, searched=50, objects=("table",), connected=("hallway",)),
      room(2, "office", 8, (3.0, 2.5), 0.3, walls=3, open_m=2.5, tentative=True, frontier=1, objects=("table",),
           connected=("hallway",))],
     "the barely seen office above the dining room covered whole",
     lambda p: p[2] > p[1]),
    ("P12 bed: a hallway with three unlooked doorways, a kitchen covered", "bed",
     context("rooms found: hallway (entered), kitchen (entered)", "every known room identified",
             "3 openings not yet looked into", "unexplored frontier still reachable", "actions used about 75 of 500"),
     [room(1, "hallway", 8, (5.0, 1.5), 0.9, inside=0, searched=15, frontier=3, exits=3, doorways=3, unseen=30.0,
           connected=("kitchen",)),
      room(2, "kitchen", 11, (4.0, 3.0), 0.9, inside=30, searched=25, objects=("oven", "refrigerator", "sink"),
           connected=("hallway",))],
     "the hallway with three unlooked doorways above the kitchen",
     lambda p: p[1] > p[2]),
]

#: The first probe's seven (2026-11-16): finished rooms offered, two openings as nodes, 1-5 without size detail.
O_SET = [
    ("O1 toilet: no bathroom found yet, early", "toilet",
     context("rooms found: hallway (entered), kitchen (scanned), living_room (scanned), bedroom (never entered)",
             "1 room unidentified", "1 opening not yet looked into", "unexplored frontier still reachable",
             "actions used about 150 of 500"),
     [node(2, ROOM, "kitchen", area_m2=12, searched_s=30, last_inside_ago_s=120, objects=("oven", "refrigerator", "sink"),
           scanned="full rotation", scanned_ago_s=120),
      node(3, ROOM, "living_room", area_m2=24, frontier_clusters=1, searched_s=40, last_inside_ago_s=60,
           objects=("sofa", "television"), scanned="full rotation", scanned_ago_s=60),
      node(5, ROOM, "bedroom", area_m2=14, frontier_clusters=1, objects=("bed",)),
      node(6, ROOM, "unknown", area_m2=5, frontier_clusters=1),
      node(7, ROOM, "hallway", area_m2=6, frontier_clusters=1, searched_s=10, last_inside_ago_s=0, here=True),
      node(DOOR, OPENING, "doorway", via="room 7 (type=hallway)")],
     "an unexplored place (the unknown room or the doorway) above both scanned rooms",
     lambda p: max(p[6], p[DOOR]) > max(p[2], p[3])),
    ("O2 sofa: the living room scanned without one", "sofa",
     context("rooms found: bedroom (scanned), living_room (scanned)", "2 rooms unidentified",
             "0 openings not yet looked into", "unexplored frontier still reachable", "actions used about 175 of 500"),
     [node(1, ROOM, "living_room", area_m2=22, searched_s=35, last_inside_ago_s=90,
           objects=("television", "coffee table", "potted plant"), scanned="full rotation", scanned_ago_s=90),
      node(2, ROOM, "bedroom", area_m2=13, searched_s=30, last_inside_ago_s=40, objects=("bed", "wardrobe"),
           scanned="full rotation", scanned_ago_s=40),
      node(3, ROOM, "unknown", area_m2=14, frontier_clusters=2),
      node(4, ROOM, "unknown", area_m2=9, frontier_clusters=1)],
     "the scanned bedroom below the best unidentified room",
     lambda p: p[2] < max(p[3], p[4])),
    ("O3 television: kitchen and bathroom scanned, a bedroom not entered", "television",
     context("rooms found: bathroom (scanned), bedroom (never entered), kitchen (scanned)", "1 room unidentified",
             "0 openings not yet looked into", "unexplored frontier still reachable", "actions used about 100 of 500"),
     [node(1, ROOM, "kitchen", area_m2=11, searched_s=30, last_inside_ago_s=80, objects=("oven", "sink", "refrigerator"),
           scanned="full rotation", scanned_ago_s=80),
      node(2, ROOM, "bedroom", area_m2=15, frontier_clusters=1, objects=("bed",)),
      node(3, ROOM, "unknown", area_m2=20, frontier_clusters=2),
      node(4, ROOM, "bathroom", area_m2=5, searched_s=15, last_inside_ago_s=30, objects=("toilet", "sink"),
           scanned="full rotation", scanned_ago_s=30)],
     "the bedroom above the scanned kitchen and the scanned bathroom",
     lambda p: p[2] > p[1] and p[2] > p[4]),
    ("O4 toilet: late, every room scanned", "toilet",
     context("rooms found: bathroom (scanned), bedroom (scanned), kitchen (scanned)", "every known room identified",
             "0 openings not yet looked into", "no unexplored frontier left", "actions used about 425 of 500"),
     [node(1, ROOM, "bathroom", area_m2=5, searched_s=20, last_inside_ago_s=240, objects=("sink", "bathtub"),
           scanned="full rotation", scanned_ago_s=240),
      node(2, ROOM, "bedroom", area_m2=14, searched_s=30, last_inside_ago_s=150, objects=("bed", "wardrobe"),
           scanned="full rotation", scanned_ago_s=150),
      node(3, ROOM, "kitchen", area_m2=12, searched_s=30, last_inside_ago_s=60, objects=("oven", "sink"),
           scanned="full rotation", scanned_ago_s=60)],
     "the bathroom above the bedroom and the kitchen",
     lambda p: p[1] > p[2] and p[1] > p[3]),
    ("O5 chair: early, unexplored places left", "chair",
     context("rooms found: hallway (entered), living_room (scanned)", "1 room unidentified",
             "1 opening not yet looked into", "unexplored frontier still reachable", "actions used about 50 of 500"),
     [node(1, ROOM, "living_room", area_m2=20, searched_s=25, last_inside_ago_s=20, objects=("sofa", "television"),
           scanned="full rotation", scanned_ago_s=20),
      node(2, ROOM, "unknown", area_m2=12, frontier_clusters=2),
      node(3, ROOM, "hallway", area_m2=5, frontier_clusters=1, searched_s=5, last_inside_ago_s=0, here=True),
      node(DOOR, OPENING, "doorway", via="room 3 (type=hallway)")],
     "the unidentified room above the scanned living room",
     lambda p: p[2] > p[1]),
    ("O6 bed: one unidentified room covered, one barely seen", "bed",
     context("rooms found: hallway (entered), kitchen (scanned)", "2 rooms unidentified", "0 openings not yet looked into",
             "unexplored frontier still reachable", "actions used about 125 of 500"),
     [node(1, ROOM, "unknown", searched_s=45, last_inside_ago_s=30, objects=("chair",), **extent(12.0, 4.0, 3.0, covered=0.93)),
      node(2, ROOM, "unknown", frontier_clusters=2, **extent(6.0, 3.0, 2.0, walls=2, covered=0.35, open_m=3.0)),
      node(3, ROOM, "hallway", searched_s=15, last_inside_ago_s=0, here=True, **extent(5.0, 4.0, 1.5, covered=0.8))],
     "the barely seen room above the room covered without a bed",
     lambda p: p[2] > p[1]),
    ("O7 refrigerator: a room joined to the dining room, another to a bedroom", "refrigerator",
     context("rooms found: bedroom (scanned), dining_room (scanned), hallway (entered)", "2 rooms unidentified",
             "0 openings not yet looked into", "unexplored frontier still reachable", "actions used about 100 of 500"),
     [node(4, ROOM, "unknown", frontier_clusters=1, connected=("dining_room",), **extent(8.0, 3.0, 2.5, walls=3, covered=0.2, open_m=2.5)),
      node(5, ROOM, "unknown", frontier_clusters=1, connected=("bedroom",), **extent(8.0, 3.0, 2.5, walls=3, covered=0.2, open_m=2.5)),
      node(6, ROOM, "bedroom", searched_s=30, last_inside_ago_s=60, objects=("bed", "wardrobe"), scanned="full rotation",
           scanned_ago_s=60, **extent(13.0, 4.0, 3.5, covered=0.95))],
     "the scanned bedroom below both unidentified rooms",
     lambda p: p[6] < min(p[4], p[5])),
]
SETS = {"P": P_SET, "O": O_SET, "PO": P_SET + O_SET}
