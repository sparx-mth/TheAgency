"""The fixed single-storey evaluation set: same-storey episodes in the flat MP3D val houses, a seeded sample.

Why a fixed set: the method's single-storey potential is measured by re-running
the SAME episodes before and after a change (``campaign.py --episode-ids-file``),
and the storey model is deliberately left out of the measurement -- stair
traversal is off, and a split-level house breaks the spawn-floor guard whatever
the search does. So an episode qualifies when every one of these holds, read
off the level survey (``hm3d/episode_levels.py`` run over MP3D val, the same
navmesh the benchmark scores on):

* its scene is not split-level: the scenes named in ``SPLIT_LEVEL_SCENES`` are
  out (a third to a half of their starts stand on stairs or on platforms off
  every storey);
* the start stands on a storey (``start_kind == floor``: within 0.15 m of a
  storey's height);
* the nearest goal view point lies on the start's storey and within
  ``max_goal_dz_m`` of the start's height;
* the shortest path to it never rises or drops by more than ``max_path_dz_m``.

From the qualifying pool a ``seed``-ed uniform sample of ``n`` episodes is
drawn -- uniform, not per scene, so the set is a random sample of the
single-storey flat-house population of the split and every house weighs by
how many such episodes it has -- and written as the JSON list the campaign's
``--episode-ids-file`` takes, sorted by scene and index, with a sidecar
``.meta.json`` naming the rules, the pool and the per-scene and per-category
counts.

The shipped set is ``sets/single_storey_flat_v1.json`` (180 episodes from
``sets/mp3d_val_levels.csv``); regenerate it only on purpose, since every
comparison made on it depends on its being the same list::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.single_storey_set \\
        --levels sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/sets/mp3d_val_levels.csv \\
        --out sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/sets/single_storey_flat_v1.json

Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

#: MP3D val scenes whose starts stand on stairs or off-storey platforms in a third or more of the episodes
#: (``episode_levels``: x8F5xyUWy9e 54 %, X7HyMhZNoso 38 %) or whose three storeys hold raised and sunken
#: parts the spawn-floor guard mis-masks after a stair start (EU6Fwq7SyZv, entry 044): out of the set.
SPLIT_LEVEL_SCENES = ("x8F5xyUWy9e", "X7HyMhZNoso", "EU6Fwq7SyZv")
#: The shortest path to the nearest goal view point may rise or drop by at most this much (m).
MAX_PATH_DZ_M = 0.2
#: The nearest goal view point lies within this much (m) of the start's height.
MAX_GOAL_DZ_M = 0.2
DEFAULT_N = 180
DEFAULT_SEED = 17
SETS_DIR = Path(__file__).resolve().parent / "sets"
DEFAULT_LEVELS = SETS_DIR / "mp3d_val_levels.csv"
DEFAULT_SET = SETS_DIR / "single_storey_flat_v1.json"


def qualifies(row: Dict[str, str], *, excluded_scenes: Sequence[str] = SPLIT_LEVEL_SCENES,
              max_path_dz_m: float = MAX_PATH_DZ_M, max_goal_dz_m: float = MAX_GOAL_DZ_M) -> bool:
    """Whether a level-survey row is a same-storey episode in a flat house (the module's rules)."""
    scene = row["episode"].split("/")[0]
    if scene in excluded_scenes or row.get("start_kind") != "floor":
        return False
    if row.get("goal_level", "") == "" or row["goal_level"] != row["start_level"]:
        return False
    try:
        rise, drop, goal_dz = float(row["path_max_rise_m"]), float(row["path_max_drop_m"]), float(row["goal_dz_m"])
    except (KeyError, ValueError):
        return False
    return rise <= max_path_dz_m and drop <= max_path_dz_m and abs(goal_dz) <= max_goal_dz_m


def select(rows: Iterable[Dict[str, str]], n: int = DEFAULT_N, seed: int = DEFAULT_SEED, **rules) -> Dict[str, object]:
    """The seeded uniform sample of the qualifying pool, sorted by scene and index, with its description.

    Returns:
        ``{"episodes": [...], "pool": {...}, "selected": {...}}`` -- the ids, and
        per-scene / per-category counts of the pool and of the sample.
    """
    if int(n) < 1:
        raise ValueError("n must be at least 1, got %r" % (n,))
    pool = [row for row in rows if qualifies(row, **rules)]
    if len(pool) < int(n):
        raise ValueError("the pool holds %d qualifying episodes, fewer than the %d asked for" % (len(pool), n))
    ids = sorted(row["episode"] for row in pool)      # a stable order before the seeded draw
    category = {row["episode"]: row["category"] for row in pool}
    chosen = sorted(random.Random(int(seed)).sample(ids, int(n)))

    def counts(episodes: Sequence[str]) -> Dict[str, object]:
        return {"episodes": len(episodes),
                "by_scene": dict(sorted(Counter(e.split("/")[0] for e in episodes).items())),
                "by_category": dict(sorted(Counter(category[e] for e in episodes).items()))}

    return {"episodes": chosen, "pool": counts(ids), "selected": counts(chosen)}


def read_levels(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--levels", type=Path, default=DEFAULT_LEVELS,
                        help="The episode_levels CSV over the split (default: the shipped MP3D val survey)")
    parser.add_argument("--out", type=Path, default=DEFAULT_SET,
                        help="The JSON list for campaign.py --episode-ids-file; a .meta.json is written beside it")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--exclude-scenes", nargs="*", default=list(SPLIT_LEVEL_SCENES))
    parser.add_argument("--max-path-dz-m", type=float, default=MAX_PATH_DZ_M)
    parser.add_argument("--max-goal-dz-m", type=float, default=MAX_GOAL_DZ_M)
    args = parser.parse_args(argv)
    result = select(read_levels(args.levels), n=args.n, seed=args.seed, excluded_scenes=tuple(args.exclude_scenes),
                    max_path_dz_m=args.max_path_dz_m, max_goal_dz_m=args.max_goal_dz_m)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result["episodes"], indent=0) + "\n")
    meta = {"rules": {"excluded_scenes": list(args.exclude_scenes), "start_kind": "floor",
                      "goal_on_start_level": True, "max_path_dz_m": args.max_path_dz_m,
                      "max_goal_dz_m": args.max_goal_dz_m},
            "sample": {"n": args.n, "seed": args.seed, "draw": "random.Random(seed).sample over the pool sorted by id"},
            "levels": {"file": args.levels.name, "sha256": hashlib.sha256(args.levels.read_bytes()).hexdigest()},
            "pool": result["pool"], "selected": result["selected"]}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("pool %d, selected %d -> %s" % (result["pool"]["episodes"], len(result["episodes"]), args.out))
    print(json.dumps(result["selected"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
