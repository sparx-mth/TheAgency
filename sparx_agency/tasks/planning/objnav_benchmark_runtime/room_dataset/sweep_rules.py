"""Rules for choosing the views of a room from a sweep: most floor seen, nearest or farthest from the entrance, or at random; only views that are neither close-ups nor mostly outside the room."""
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import MAX_NEAR_SHARE, MIN_IN_ROOM_SHARE

KEY = {"floor": lambda f: -f["floor_share"], "door": lambda f: f["entrance_m"], "far": lambda f: -f["entrance_m"]}


def usable_frames(frames):
    return {n: f for n, f in frames.items() if f["near"] <= MAX_NEAR_SHARE and f["inside"] >= MIN_IN_ROOM_SHARE and f.get("entrance_m") is not None}


def frame_picks(frames, rule, count, rng, gate=None):
    """The names of up to ``count`` usable views under ``rule`` (``floor``, ``door``, ``far``, ``random``, or ``gate``: highest logit in ``gate``), best first."""
    usable = usable_frames(frames)
    if rule == "gate":
        return sorted((n for n in usable if n in gate), key=lambda n: (-gate[n], n))[:count]
    if rule == "random":
        names = sorted(usable)
        return [names[i] for i in rng.permutation(len(names))[:count]]
    return sorted(usable, key=lambda n: (KEY[rule](usable[n]), n))[:count]
