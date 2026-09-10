"""Fuzzy-lite matching of a natural-language target against a class label.

Detector vocabularies and the words a human uses for the same object rarely
agree character-for-character ("car keys" vs "key", "coffee cup" vs "cup",
"fire_extinguisher box" vs "wall_extinguisher"). This module holds the one
offline rule the tree uses to decide whether a detected class *counts* as the
requested target.

It lives in the widest ring because two different subsystems mean it
identically: the pose-free acquisition gate in
:mod:`sparx_agency.core.planning.visual_servo.confirmation_gate` and the
scene-graph target ladder's offline rung in
:mod:`sparx_agency.core.mapping.topology.target_matcher`. Anything narrower
would force one of them to import the other across the ``planning`` /
``mapping`` boundary.

:func:`normalize_label` is the canonical *form* of a label, for code that
needs an exact comparison rather than this fuzzy one -- the ObjectNav label
tables in :mod:`sparx_agency.core.planning.objnav` compare detector output
against a dataset category's accept set with it, because on a benchmark a
fuzzy accept is a false STOP.

Pure string work: no numpy, no ROS, Python-3.8-safe.
"""
from __future__ import annotations


def normalize_label(label: str) -> str:
    """The canonical form of a class label or category name.

    Lowercased, underscores read as spaces, runs of whitespace collapsed and
    both ends trimmed: ``"TV_Monitor "`` and ``"tv  monitor"`` both become
    ``"tv monitor"``. Hyphens are kept (``"x-ray machine"``), and CamelCase is
    deliberately *not* split -- a table that wants ``"house plant"`` for
    AI2-THOR's ``HousePlant`` says so explicitly rather than trusting a
    heuristic that would also split ``"iPad"``.

    Args:
        label: A detector class, a dataset category, or a phrase.

    Returns:
        The normalised label; empty when the input is blank.
    """
    return " ".join(str(label).replace("_", " ").lower().split())


def label_matches(target: str, label: str) -> bool:
    """Fuzzy-lite class match: exact, substring, or shared whitespace/underscore token.

    Both sides are stripped and lowercased first, and an empty side never
    matches (so ``"" == ""`` cannot read as a hit).

    Args:
        target: The object the mission is looking for.
        label: A class name from the detector's vocabulary.

    Returns:
        True when the label should be accepted as the target.
    """
    t = str(target).strip().lower()
    c = str(label).strip().lower()
    if not t or not c:
        return False
    if t == c or t in c or c in t:
        return True
    t_tokens = set(t.replace("_", " ").split())
    c_tokens = set(c.replace("_", " ").split())
    return bool(t_tokens & c_tokens)
