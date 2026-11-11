"""The smallest HM3D subset in which every scene and every goal category appears at least once.

A plumbing check of a whole split should touch every building and every
target without costing a day. The cover below is one published episode per
scene, so a split's 20 (v1) or 36 (v2) scenes give 20 or 36 episodes, and the
category each scene contributes is chosen so that all six categories are
covered as early as possible and the counts stay as even as the scenes allow.

The rule is deterministic and reads nothing but the published episode list:

1. Scenes are taken in publisher order (the order of the split's shards).
2. A scene contributes the category it offers that has been chosen the
   fewest times so far; ties go to the dataset's own category-index order
   (chair, bed, plant, toilet, tv_monitor, sofa).
3. Within that scene and category the **first episode in published order**
   is taken -- the same start and the same goal every time the plan is made.

So ``plant``, which only five of the twenty v1 scenes offer, is taken from the
first scene that has it, and a category no scene offers is reported rather
than silently dropped. This is a diagnostic subset and is labelled as one
wherever it is run; it is never a benchmark number.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def cover_plan(episodes: Mapping[str, object], categories: Sequence[str],
               scenes: Iterable[str] = None) -> Dict[str, List[str]]:
    """One episode per scene, categories balanced, every category covered where possible.

    Args:
        episodes: ``{episode_id: episode}`` in published order, each episode
            carrying ``scene_key`` and ``category`` (``HM3DDataset.episodes``).
        categories: The goal categories in dataset index order.
        scenes: Restrict the plan to these scene keys, in this order; every
            scene of ``episodes`` in first-seen order by default.

    Returns:
        ``{scene_key: [episode_id]}`` in scene order -- one id per scene.

    Raises:
        ValueError: A requested scene has no episodes, or ``episodes`` is empty.
    """
    if not episodes:
        raise ValueError("No episodes to plan a cover over")
    by_scene = {}  # type: Dict[str, Dict[str, str]]
    order = []  # type: List[str]
    for episode_id, episode in episodes.items():
        scene = str(getattr(episode, "scene_key"))
        category = str(getattr(episode, "category"))
        if scene not in by_scene:
            by_scene[scene] = {}
            order.append(scene)
        by_scene[scene].setdefault(category, str(episode_id))   # the first in published order
    selected = list(scenes) if scenes is not None else order
    missing = [scene for scene in selected if scene not in by_scene]
    if missing:
        raise ValueError("No published episodes for scene(s) %s" % ", ".join(missing))
    rank = {category: index for index, category in enumerate(categories)}
    chosen = {category: 0 for category in categories}
    plan = {}  # type: Dict[str, List[str]]
    for scene in selected:
        offered = [c for c in by_scene[scene] if c in rank]
        if not offered:
            raise ValueError("Scene %s offers no episode of a known category" % scene)
        category = min(offered, key=lambda c: (chosen[c], rank[c]))
        chosen[category] += 1
        plan[scene] = [by_scene[scene][category]]
    return plan


def uncovered(plan: Mapping[str, Sequence[str]], episodes: Mapping[str, object],
              categories: Sequence[str]) -> Tuple[str, ...]:
    """The categories the plan never visits, in dataset order (empty when every one is covered)."""
    seen = {str(getattr(episodes[episode_id], "category"))
            for ids in plan.values() for episode_id in ids}
    return tuple(c for c in categories if c not in seen)


def describe(plan: Mapping[str, Sequence[str]], episodes: Mapping[str, object],
             categories: Sequence[str]) -> Dict[str, object]:
    """A JSON-ready account of the plan: per scene, per category, and what is left out."""
    per_category = {category: 0 for category in categories}
    rows = []
    for scene, ids in plan.items():
        for episode_id in ids:
            category = str(getattr(episodes[episode_id], "category"))
            per_category[category] = per_category.get(category, 0) + 1
            rows.append({"scene": scene, "episode_id": episode_id, "category": category})
    return {"rule": "one episode per scene, in publisher order; the scene's least-chosen category, "
                    "ties by dataset category index; the first published episode of it",
            "episodes": rows, "scenes": len(plan), "per_category": per_category,
            "uncovered_categories": list(uncovered(plan, episodes, categories)),
            "diagnostic_subset": True}
