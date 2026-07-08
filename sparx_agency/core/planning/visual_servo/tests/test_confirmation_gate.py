"""Unit tests for the localization-free target confirmation gate.

Covers the fuzzy label matcher, the best-detection selector, and the
consecutive-frame :class:`TargetConfirmationGate` (streak advance, miss
tolerance bridging, reset semantics) plus config validation.
"""
from __future__ import annotations

from typing import Tuple

import pytest

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.planning.visual_servo.confirmation_gate import (
    ConfirmationGateConfig,
    ConfirmationState,
    TargetConfirmationGate,
    label_matches,
    select_target_detection,
)


def det(label: str, score: float,
        bbox: Tuple[int, int, int, int] = (0, 0, 10, 10)) -> Detection2D:
    """Build a Detection2D with sensible frame dimensions for testing."""
    return Detection2D(label=label, score=score, bbox_xyxy=bbox,
                       frame_w=640, frame_h=480)


# --------------------------------------------------------------------------- #
# label_matches
# --------------------------------------------------------------------------- #
def test_label_matches_exact():
    assert label_matches("hat", "hat") is True


def test_label_matches_case_and_whitespace_insensitive():
    assert label_matches("  Hat ", "HAT") is True


def test_label_matches_substring_both_directions():
    # target is a substring of the label ...
    assert label_matches("cup", "coffee cup") is True
    # ... and label is a substring of the target.
    assert label_matches("coffee cup", "cup") is True


def test_label_matches_shared_token():
    # Neither is a substring of the other, but they share the token "gun".
    assert "hand gun" not in "toy gun"
    assert "toy gun" not in "hand gun"
    assert label_matches("hand gun", "toy gun") is True


def test_label_matches_underscore_tokenized():
    # Underscores are normalized to spaces before tokenizing.
    assert label_matches("fire_extinguisher box", "wall_extinguisher") is True


def test_label_matches_unrelated_false():
    assert label_matches("hat", "dog") is False


def test_label_matches_empty_false():
    assert label_matches("", "hat") is False
    assert label_matches("hat", "") is False


# --------------------------------------------------------------------------- #
# select_target_detection
# --------------------------------------------------------------------------- #
def test_select_picks_highest_scoring_match():
    detections = [
        det("hat", 0.55),
        det("hat", 0.91),   # highest matching score
        det("hat", 0.40),
        det("dog", 0.99),   # higher score but wrong label
    ]
    best = select_target_detection(detections, "hat", min_score=0.30)
    assert best is not None
    assert best.label == "hat"
    assert best.score == pytest.approx(0.91)


def test_select_none_when_no_label_match():
    detections = [det("dog", 0.99), det("cat", 0.80)]
    assert select_target_detection(detections, "hat", min_score=0.30) is None


def test_select_none_when_all_below_min_score():
    detections = [det("hat", 0.20), det("hat", 0.29)]
    assert select_target_detection(detections, "hat", min_score=0.30) is None


def test_select_includes_score_at_threshold():
    # Selector uses score < min_score to reject, so a score == min_score passes.
    detections = [det("hat", 0.30)]
    best = select_target_detection(detections, "hat", min_score=0.30)
    assert best is not None
    assert best.score == pytest.approx(0.30)


def test_select_empty_detections_returns_none():
    assert select_target_detection([], "hat", min_score=0.30) is None


# --------------------------------------------------------------------------- #
# ConfirmationGateConfig validation
# --------------------------------------------------------------------------- #
def test_config_defaults():
    cfg = ConfirmationGateConfig()
    assert cfg.n_confirm == 3
    assert cfg.min_score == pytest.approx(0.30)
    assert cfg.miss_tolerance == 1


def test_config_invalid_n_confirm_raises():
    with pytest.raises(ValueError):
        ConfirmationGateConfig(n_confirm=0)


def test_config_invalid_miss_tolerance_raises():
    with pytest.raises(ValueError):
        ConfirmationGateConfig(miss_tolerance=-1)


# --------------------------------------------------------------------------- #
# TargetConfirmationGate
# --------------------------------------------------------------------------- #
def test_gate_confirms_only_after_n_consecutive():
    cfg = ConfirmationGateConfig(n_confirm=3, min_score=0.30, miss_tolerance=0)
    gate = TargetConfirmationGate("hat", cfg)

    s1 = gate.update([det("hat", 0.9)])
    assert s1.streak == 1 and s1.confirmed is False

    s2 = gate.update([det("hat", 0.9)])
    assert s2.streak == 2 and s2.confirmed is False

    s3 = gate.update([det("hat", 0.9)])
    assert s3.streak == 3 and s3.confirmed is True


def test_gate_best_is_matching_detection():
    gate = TargetConfirmationGate("hat", ConfirmationGateConfig(n_confirm=3))
    high = det("hat", 0.95)
    state = gate.update([det("hat", 0.40), high, det("dog", 0.99)])
    assert isinstance(state, ConfirmationState)
    assert state.best is high
    assert state.best.score == pytest.approx(0.95)


def test_gate_best_none_on_miss_frame():
    gate = TargetConfirmationGate("hat", ConfirmationGateConfig(n_confirm=3))
    state = gate.update([det("dog", 0.99)])
    assert state.best is None
    assert state.streak == 0


def test_gate_miss_beyond_tolerance_resets_streak():
    cfg = ConfirmationGateConfig(n_confirm=3, miss_tolerance=1)
    gate = TargetConfirmationGate("hat", cfg)
    assert gate.update([det("hat", 0.9)]).streak == 1
    assert gate.update([det("hat", 0.9)]).streak == 2
    # First miss is tolerated (bridged): streak held.
    assert gate.update([]).streak == 2
    # Second consecutive miss exceeds tolerance: streak resets.
    assert gate.update([]).streak == 0


def test_gate_strict_zero_tolerance_resets_on_single_miss():
    cfg = ConfirmationGateConfig(n_confirm=3, miss_tolerance=0)
    gate = TargetConfirmationGate("hat", cfg)
    assert gate.update([det("hat", 0.9)]).streak == 1
    assert gate.update([]).streak == 0


def test_gate_miss_tolerance_bridges_single_dropped_frame():
    cfg = ConfirmationGateConfig(n_confirm=3, miss_tolerance=1)
    gate = TargetConfirmationGate("hat", cfg)
    assert gate.update([det("hat", 0.9)]).streak == 1
    assert gate.update([det("hat", 0.9)]).streak == 2
    # A lone dropped frame does not reset the streak ...
    bridged = gate.update([])
    assert bridged.streak == 2 and bridged.confirmed is False
    # ... and the next hit confirms.
    final = gate.update([det("hat", 0.9)])
    assert final.streak == 3 and final.confirmed is True


def test_gate_below_min_score_is_a_miss():
    cfg = ConfirmationGateConfig(n_confirm=2, min_score=0.50, miss_tolerance=0)
    gate = TargetConfirmationGate("hat", cfg)
    # Detection present but under threshold -> counts as a miss.
    s = gate.update([det("hat", 0.40)])
    assert s.streak == 0 and s.best is None


def test_set_target_changes_target_and_resets_streak():
    gate = TargetConfirmationGate("hat", ConfirmationGateConfig(n_confirm=3))
    gate.update([det("hat", 0.9)])
    gate.update([det("hat", 0.9)])  # streak == 2

    gate.set_target("Weapon")
    assert gate.target == "weapon"  # stored lowercased/stripped
    # Streak was reset: a fresh matching hit starts the count at 1.
    s = gate.update([det("weapon", 0.9)])
    assert s.streak == 1


def test_reset_clears_streak():
    gate = TargetConfirmationGate("hat", ConfirmationGateConfig(n_confirm=3))
    gate.update([det("hat", 0.9)])
    gate.update([det("hat", 0.9)])  # streak == 2
    gate.reset()
    s = gate.update([det("hat", 0.9)])
    assert s.streak == 1


def test_gate_default_config_when_none():
    gate = TargetConfirmationGate("hat")
    assert gate.cfg.n_confirm == 3
    assert gate.cfg.miss_tolerance == 1


def test_empty_target_never_confirms():
    gate = TargetConfirmationGate("   ")  # strips to empty
    assert gate.target == ""
    state = gate.update([det("hat", 0.99)])
    assert state.confirmed is False
    assert state.streak == 0
    assert state.best is None
