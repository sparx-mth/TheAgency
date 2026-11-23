"""Detector scope for the frontier experiment; the dataset mapper owns all goal aliases."""

STAIR_LABELS = frozenset(("stairs", "staircase"))


def target_stair_prompts(target):
    """Stable order, including accepted aliases that are not primary detector prompts."""
    return tuple(dict.fromkeys((*target.detector_prompts, *sorted(target.accept_labels), *sorted(STAIR_LABELS))))
