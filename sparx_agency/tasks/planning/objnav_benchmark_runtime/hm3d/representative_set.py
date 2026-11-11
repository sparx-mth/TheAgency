"""A fixed, representative subset of an HM3D split: every scene, every category, seeded, reused run after run.

The 2,000 episodes of HM3D v1 val take a day of GPU per arm; a comparison of
two versions of the method needs a tenth of that, on the SAME episodes every
time. This module draws one such subset by a rule that reads nothing but the
published episode list, so the set is reproducible from the seed and the
split alone, and writes it as the JSON list of ``<scene>/<index>`` ids that
``hm3d/campaign.py --episode-ids-file`` (and ``run_benchmark.sh``) run
exactly, beside a ``.meta.json`` that records the rule and the counts.

The rule, for ``size`` episodes over the split's ``S`` scenes:

1. **Scenes are equal.** Every scene contributes ``size // S`` episodes; a
   remainder goes one each to the scenes with the most published episodes
   (publisher order breaks ties).
2. **Categories are as even as the scenes allow.** The scene quotas are
   filled one episode at a time, round-robin over the scenes in publisher
   order; at each pick the scene takes, among the categories it offers that
   still have episodes left and that it has not yet given ``cap`` episodes
   to, the category with the fewest episodes chosen so far in the whole set
   (ties by the dataset's category index: chair, bed, plant, toilet,
   tv_monitor, sofa). ``cap`` (4 of 10 by default) keeps a scene mixed:
   without it, ``plant`` -- offered by 5 of the 20 v1 scenes -- would take
   over those scenes, and with it the set carries as many plants as five
   mixed scenes can give (20 of 200) rather than its 4 % share of the split.
3. **Within a scene and category the draw is seeded.** The cell's published
   ids are shuffled by a generator seeded from the set's seed, the scene and
   the category, and the first ``n`` are taken, so adding a scene or a
   category never reshuffles another cell.

Cross-storey goals (the levels survey, ``sets/hm3d_v1_val_levels.csv``) are
neither sought nor avoided: the set is representative of the split in that
respect and the metadata says how many it holds, so a breakdown can read
same- and cross-storey apart.

Usage::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.representative_set \\
        --version v1 --size 200 --seed 20261111 --name hm3d_v1_val_rep200

Python 3.8 syntax, standard library plus the dataset loader.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SETS_DIR = Path(__file__).resolve().parent / "sets"
DEFAULT_SEED = 20261111
DEFAULT_CAP = 4


def scene_quotas(scenes: Sequence[str], counts: Mapping[str, int], size: int) -> Dict[str, int]:
    """``size`` split equally over ``scenes`` (publisher order); the remainder to the scenes with the most episodes."""
    if not scenes:
        raise ValueError("no scenes to draw from")
    if size <= 0:
        raise ValueError("size must be positive")
    base, remainder = divmod(int(size), len(scenes))
    quotas = {scene: base for scene in scenes}
    by_size = sorted(scenes, key=lambda s: (-int(counts.get(s, 0)), scenes.index(s)))
    for scene in by_size[:remainder]:
        quotas[scene] += 1
    for scene in scenes:
        if quotas[scene] > int(counts.get(scene, 0)):
            raise ValueError("scene %s has %d episodes, fewer than its quota %d" % (scene, counts.get(scene, 0), quotas[scene]))
    return quotas


def allocate(table: Mapping[str, Mapping[str, int]], scenes: Sequence[str], categories: Sequence[str],
             quotas: Mapping[str, int], cap: int) -> Dict[str, Dict[str, int]]:
    """``{scene: {category: n}}`` -- the round-robin, least-chosen-category allocation of the module docstring."""
    rank = {category: index for index, category in enumerate(categories)}
    chosen = {scene: {category: 0 for category in categories} for scene in scenes}
    totals = {category: 0 for category in categories}
    remaining = {scene: int(quotas[scene]) for scene in scenes}
    while any(remaining[scene] > 0 for scene in scenes):
        progressed = False
        for scene in scenes:
            if remaining[scene] <= 0:
                continue
            offered = [c for c in categories if int(table[scene].get(c, 0)) > chosen[scene][c]]
            open_cells = [c for c in offered if chosen[scene][c] < cap] or offered     # the cap yields before a scene starves
            if not open_cells:
                raise ValueError("scene %s has no episode left for its quota" % scene)
            category = min(open_cells, key=lambda c: (totals[c], rank[c]))
            chosen[scene][category] += 1
            totals[category] += 1
            remaining[scene] -= 1
            progressed = True
        if not progressed:
            break
    return chosen


def cell_seed(seed: int, scene: str, category: str) -> int:
    digest = hashlib.sha256(("%d|%s|%s" % (int(seed), scene, category)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def draw(cells: Mapping[Tuple[str, str], Sequence[str]], allocation: Mapping[str, Mapping[str, int]],
         seed: int, published_order: Optional[Sequence[str]] = None) -> List[str]:
    """The episode ids of the set, in published order: ``n`` seeded picks from each (scene, category) cell.

    ``published_order`` is every id of the split in its published order; by
    default the cells' ids scene by scene, cell by cell. The sort key is the
    id's position in it -- never a hash-ordered set, whose iteration order
    changes from one process to the next (2026-11-11: the same picks came out
    in two orders).
    """
    picked = set()
    for scene, per_category in allocation.items():
        for category, n in per_category.items():
            if n <= 0:
                continue
            ids = list(cells.get((scene, category), ()))
            if len(ids) < n:
                raise ValueError("cell (%s, %s) has %d episodes, %d asked" % (scene, category, len(ids), n))
            rng = random.Random(cell_seed(seed, scene, category))
            rng.shuffle(ids)
            picked.update(ids[:n])
    if published_order is None:
        published_order = [episode_id for ids in cells.values() for episode_id in ids]
    position = {episode_id: index for index, episode_id in enumerate(published_order)}
    return sorted(picked, key=lambda episode_id: (position.get(episode_id, len(position)), episode_id))


def build(episodes: Mapping[str, object], categories: Sequence[str], size: int, seed: int = DEFAULT_SEED,
          cap: int = DEFAULT_CAP, levels: Optional[Mapping[str, Mapping[str, str]]] = None) -> Dict[str, object]:
    """The set and its metadata from ``{episode_id: episode}`` in published order (``HM3DDataset.episodes``)."""
    if not episodes:
        raise ValueError("no episodes")
    scenes = []  # type: List[str]
    counts = {}  # type: Dict[str, int]
    table = {}  # type: Dict[str, Dict[str, int]]
    cells = {}  # type: Dict[Tuple[str, str], List[str]]
    for episode_id, episode in episodes.items():
        scene = str(getattr(episode, "scene_key"))
        category = str(getattr(episode, "category"))
        if scene not in table:
            scenes.append(scene)
            table[scene] = {}
            counts[scene] = 0
        counts[scene] += 1
        table[scene][category] = table[scene].get(category, 0) + 1
        cells.setdefault((scene, category), []).append(str(episode_id))
    quotas = scene_quotas(scenes, counts, size)
    allocation = allocate(table, scenes, categories, quotas, cap)
    ids = draw(cells, allocation, seed, published_order=[str(episode_id) for episode_id in episodes])
    per_category = {category: sum(allocation[s][category] for s in scenes) for category in categories}
    per_scene = {scene: {c: n for c, n in allocation[scene].items() if n} for scene in scenes}
    rows = []
    same_level = 0
    for episode_id in ids:
        episode = episodes[episode_id]
        row = {"episode_id": episode_id, "scene": str(episode.scene_key), "category": str(episode.category)}
        if levels is not None and episode_id in levels:
            flag = str(levels[episode_id].get("goal_same_level", "")).strip()
            row["goal_same_level"] = flag == "1"
            same_level += int(flag == "1")
        rows.append(row)
    return {
        "ids": ids,
        "meta": {
            "rule": ("equal scene quotas (the remainder to the largest scenes); quotas filled round-robin over the "
                     "scenes in publisher order, each pick the scene's least-chosen category so far (ties by dataset "
                     "category index), at most %d episodes of one category per scene; seeded shuffle within each "
                     "(scene, category) cell" % cap),
            "seed": int(seed), "size": len(ids), "cap_per_scene_category": int(cap),
            "scenes": len(scenes), "per_scene": per_scene, "per_category": per_category,
            "goal_same_level": (same_level if levels is not None else None),
            "goal_cross_level": (len(ids) - same_level if levels is not None else None),
            "episodes": rows,
            "benchmark_number": False,
            "note": "a fixed subset for comparisons between versions of the method, never a published benchmark number",
        },
    }


def load_levels(path: Path) -> Dict[str, Dict[str, str]]:
    if not Path(path).is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["episode"]: row for row in csv.DictReader(handle)}


def write(name: str, result: Mapping[str, object], directory: Path = SETS_DIR) -> Tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ids_path = directory / ("%s.json" % name)
    meta_path = directory / ("%s.meta.json" % name)
    ids_path.write_text(json.dumps(list(result["ids"]), indent=0) + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(result["meta"], indent=1) + "\n", encoding="utf-8")
    return ids_path, meta_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", default="v1")
    parser.add_argument("--split", default="val")
    parser.add_argument("--episodes-dir", default=None,
                        help="the split directory (default: $HM3D_EPISODES_DIR, else ~/datasets/objectnav/hm3d/<version>/<split>)")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="at most this many episodes of one category per scene")
    parser.add_argument("--name", default=None, help="file stem under hm3d/sets/ (default hm3d_<version>_<split>_rep<size>)")
    parser.add_argument("--out-dir", type=Path, default=SETS_DIR)
    args = parser.parse_args(argv)
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import CATEGORIES, HM3DDataset
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
    protocol = protocol_for(args.version).with_split(args.split)
    episodes_dir = args.episodes_dir or os.environ.get("HM3D_EPISODES_DIR") or os.path.expanduser(
        "~/datasets/objectnav/hm3d/%s/%s" % (args.version, args.split))
    dataset = HM3DDataset(episodes_dir, None, protocol, full=(args.split == "val"), require_scenes=False)
    levels = load_levels(SETS_DIR / ("hm3d_%s_%s_levels.csv" % (args.version, args.split)))
    result = build(dataset.episodes, CATEGORIES, args.size, seed=args.seed, cap=args.cap, levels=levels or None)
    name = args.name or "hm3d_%s_%s_rep%d" % (args.version, args.split, args.size)
    ids_path, meta_path = write(name, result, args.out_dir)
    meta = result["meta"]
    print("%s: %d episodes over %d scenes -> %s (+ %s)" % (name, meta["size"], meta["scenes"], ids_path, meta_path.name))
    print("per category:", meta["per_category"])
    if meta["goal_same_level"] is not None:
        print("same-storey goals: %d, cross-storey: %d" % (meta["goal_same_level"], meta["goal_cross_level"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
