"""The small pure pieces of the Matterport3D room evaluation: which frames, how answers vote, how objects are listed and counted."""
import collections
import math

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_room_types import ROOM_POOL, group_of_answer


def spread_frames(frames, count):
    """Up to ``count`` frames taken round-robin across the standing spots, so the first few already cover the room."""
    by_spot = collections.defaultdict(list)
    for f in frames:
        by_spot[f["spot"]].append(f)
    queues, taken = [by_spot[s] for s in sorted(by_spot)], []
    for rank in range(max(map(len, queues), default=0)):
        taken += [q[rank] for q in queues if len(q) > rank]
    return taken[:count]


def predict_group(answers):
    """The group most answers read as; ties go to the group first in ``ROOM_POOL``; None when no answer reads as a group."""
    counts = collections.Counter(g for g in map(group_of_answer, answers) if g)
    return min(counts, key=lambda g: (-counts[g], ROOM_POOL.index(g))) if counts else None


def objects_text(names):
    counts = collections.Counter(names)
    return ", ".join(n if c == 1 else "%s x%d" % (n, c) for n, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))) or "(no objects)"


def unique_objects(detections, radius_m):
    """The class of every physical object: detections of one class within ``radius_m`` of each other are the same object."""
    kept = []
    for d in detections:
        if not any(k["cls"] == d["cls"] and math.dist(k["point"], d["point"]) <= radius_m for k in kept):
            kept.append(d)
    return [k["cls"] for k in kept]


def accuracy_by_frames(answers_by_room, truth, ks, repeats, rng):
    """Accuracy of the vote over ``k`` random frames of each room, averaged over rooms and ``repeats`` draws."""
    result = {}
    for k in ks:
        correct = []
        for room, answers in answers_by_room.items():
            for _ in range(repeats):
                chosen = [answers[i] for i in rng.permutation(len(answers))[:k]]
                correct.append(predict_group(chosen) == truth[room])
        result[k] = float(np.mean(correct))
    return result


OPEN_PROMPT = 'What room in a house was this photo taken in? Answer with JSON {"room": "<name>"}.'


def list_prompt():
    return 'Which room in a house was this photo taken in? Choose one of: %s. Answer with JSON {"room": "<one of the options>"}.' % ", ".join(ROOM_POOL)


def objects_prompt(objects, with_list):
    options = " Choose one of: %s." % ", ".join(ROOM_POOL) if with_list else ""
    return ('A room of a house contains these objects: %s. What kind of room is it?%s Answer with JSON {"room": "<name>"}.'
            % (objects, options))
