"""Pick a seeded sample of HM3D episodes whose goal has a view point on the start floor.

Episodes whose goal lies only on another storey cannot be solved by an agent confined to its spawn floor.
"""
import argparse
import gzip
import json
import random
from pathlib import Path

FLOOR_TOLERANCE_M = 0.5


def goal_on_start_floor(dataset, episode):
    key = next(k for k in dataset["goals_by_category"] if k.endswith("_" + episode["object_category"]))
    start_height = episode["start_position"][1]
    return any(abs(viewpoint["agent_state"]["position"][1] - start_height) <= FLOOR_TOLERANCE_M
               for goal in dataset["goals_by_category"][key] for viewpoint in goal["view_points"])


def sample_scene(content_dir, scene, per_scene, seed):
    folder_hash = scene.split("-", 1)[1]
    dataset = json.load(gzip.open(Path(content_dir) / ("%s.json.gz" % folder_hash)))
    solvable = [index for index, episode in enumerate(dataset["episodes"]) if goal_on_start_floor(dataset, episode)]
    chosen = sorted(random.Random("%s/%s" % (seed, scene)).sample(solvable, min(per_scene, len(solvable))))
    return ["%s/%06d" % (scene, index) for index in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-dir", required=True, help="<hm3d v1 val>/content")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--per-scene", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for scene in args.scenes:
        print("\n".join(sample_scene(args.content_dir, scene, args.per_scene, args.seed)))


if __name__ == "__main__":
    main()
