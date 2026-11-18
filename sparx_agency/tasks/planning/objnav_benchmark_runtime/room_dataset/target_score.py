"""Whether a room holds a target category, how well a score ranks rooms by it, and reading a model's percent. numpy only."""
import json
import re

import numpy as np

TARGETS = {
    "chair": {"chair", "armchair", "sofa chair", "dining chair", "desk chair", "office chair", "lounge chair", "swivel chair", "easy chair", "chairs", "bean bag chair",
              "beanbag chair", "bar chair", "rocking chair", "folding chair", "computer chair", "wooden chair", "soft chair", "semi chair", "sofachair"},
    "bed": {"bed", "bedframe", "bedstead"},
    "plant": {"plant", "plants", "potted plant"},
    "toilet": {"toilet", "toilet bowl"},
    "tv_monitor": {"tv", "led tv", "television", "monitor", "computer monitor"},
    "sofa": {"sofa", "couch", "sofa set", "circular sofa", "sofa seat"},
}
BOOTSTRAP_RESAMPLES = 1000


def present(object_names, target):
    """Whether the annotated names of a room include one that is exactly a name of the target's category."""
    return any(name in TARGETS[target] for name in object_names)


def parse_percent(reply):
    """The ``percent`` of a JSON reply as a number in [0, 100]; None if there is none."""
    try:
        value = json.loads(reply)["percent"]
        number = float(re.sub(r"%\s*$", "", value.strip())) if isinstance(value, str) else float(value)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return min(100.0, max(0.0, number)) if np.isfinite(number) else None


def room_score(frame_percents):
    """The mean of the frames' percents that were read; None if none was."""
    read = [p for p in frame_percents if p is not None]
    return float(np.mean(read)) if read else None


def auroc(scores, labels):
    """The chance a room holding the target scores above one that does not, ties counting half; nan without both kinds."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(((positive[:, None] > negative[None, :]).sum() + 0.5 * (positive[:, None] == negative[None, :]).sum()) / (len(positive) * len(negative)))


def auroc_difference(scores_a, scores_b, labels, rng, resamples=BOOTSTRAP_RESAMPLES):
    """``(mean, low, high)`` of AUROC(a) - AUROC(b) with a 95% interval from resampling rooms."""
    a, b, labels = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float), np.asarray(labels, dtype=bool)
    diffs = []
    for _ in range(resamples):
        pick = rng.integers(0, len(labels), len(labels))
        d = auroc(a[pick], labels[pick]) - auroc(b[pick], labels[pick])
        if np.isfinite(d):
            diffs.append(d)
    return auroc(a, labels) - auroc(b, labels), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def scene_top1(houses):
    """Over houses with rooms both holding and lacking the target: the share whose top-scored room holds it, a tie splitting the credit."""
    credits = []
    for rooms in houses.values():
        if not any(p for _, p in rooms) or all(p for _, p in rooms):
            continue
        best = max(s for s, _ in rooms)
        top = [p for s, p in rooms if s == best]
        credits.append(sum(top) / len(top))
    return float(np.mean(credits)) if credits else float("nan")


def room_holds(room, target):
    """Whether a room holds the target: by its goal categories when it lists them (HM3D), else by its annotated names (Matterport3D)."""
    return target in room["target_categories"] if "target_categories" in room else present(room["objects"], target)


MISSING_LOGPROB = -30.0


def yes_no_percent(top_logprobs):
    """``(percent, logit)`` of a yes/no answer from the first token's top log-probabilities, spelling and case variants pooled; None if neither word is among them."""
    pooled = {"yes": [], "no": []}
    for entry in top_logprobs:
        word = str(entry["token"]).strip().lower()
        if word in pooled:
            pooled[word].append(float(entry["logprob"]))
    if not pooled["yes"] and not pooled["no"]:
        return None
    logit = {w: float(np.log(np.sum(np.exp(v)))) if v else MISSING_LOGPROB for w, v in pooled.items()}
    margin = logit["yes"] - logit["no"]
    return float(100.0 / (1.0 + np.exp(-margin))), float(margin)
