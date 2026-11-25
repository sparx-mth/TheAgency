"""What the planner itself believed about a room when it walked in: the LLM probability it offered for the room, the room's exit count and its label, read from a recorded run."""
from collections import Counter


def planner_view_of_room(steps, arrival, end):
    """``{planner_room, probability, exits, label}`` for the planner's room it stood in most of steps ``arrival``..``end``; ``probability`` is the last one offered before arrival (None if never offered); None if the planner had no room there."""
    inside = steps[arrival:end + 1]
    ids = Counter(s["method"].get("room_id") for s in inside if s["method"].get("room_id") is not None)
    if not ids:
        return None
    room = ids.most_common(1)[0][0]
    return {"planner_room": room, "probability": offered_probability(steps[:arrival], room), **node_attributes(inside, room)}


def offered_probability(steps, room):
    for row in reversed(steps):
        offered = row["method"].get("reasoning", {}).get("oracle", {}).get("probs", {})
        if str(room) in offered:
            return offered[str(room)]
    return None


def node_attributes(steps, room):
    for row in steps:
        for node in row["method"].get("reasoning", {}).get("nodes", []):
            if node.get("id") == room:
                return {"exits": node.get("exits"), "label": node.get("label")}
    return {"exits": None, "label": None}


def exits_percent(exits, few_first=False):
    """Exit counts as a percent within the episode (most exits 100, fewest 0; the other way round with ``few_first``); 50 when all are equal."""
    low, high = min(exits), max(exits)
    if high == low:
        return [50.0] * len(exits)
    scaled = [100.0 * (e - low) / (high - low) for e in exits]
    return [100.0 - s for s in scaled] if few_first else scaled
