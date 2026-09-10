"""The canonical form of a label, which every exact label comparison relies on.

:func:`normalize_label` is the one place a dataset's category, a table's
prompt and a detector's class name are brought to a common spelling. If it
drifted -- splitting CamelCase one day, dropping hyphens the next -- every
ObjectNav accept set would silently stop matching what the detector emits.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.common.label_match import normalize_label


def test_underscores_read_as_spaces():
    """HM3D writes ``tv_monitor``; a detector says ``tv monitor``."""
    assert normalize_label("tv_monitor") == "tv monitor"
    assert normalize_label("chest__of_drawers") == "chest of drawers"
    assert normalize_label("tv_ monitor") == "tv monitor"


def test_case_is_folded():
    """RoboTHOR capitalises its categories; detectors usually do not."""
    assert normalize_label("Television") == "television"
    assert normalize_label("TV Monitor") == "tv monitor"


def test_runs_of_whitespace_collapse_and_the_ends_are_trimmed():
    """Stray padding in a table or a detector label must not break a match."""
    assert normalize_label("  potted \t plant\n") == "potted plant"


def test_hyphens_are_kept():
    """``x-ray machine`` and ``x ray machine`` are not assumed to be one label."""
    assert normalize_label("X-Ray_Machine") == "x-ray machine"


def test_camel_case_is_not_split():
    """Splitting would also break ``iPad``; a table spells ``house plant`` itself."""
    assert normalize_label("HousePlant") == "houseplant"
    assert normalize_label("iPad") == "ipad"


@pytest.mark.parametrize("blank", ["", "   ", "_", "__ \t_"])
def test_a_blank_label_normalises_to_the_empty_string(blank):
    """Callers test for blankness with the normalised form, so it must be empty."""
    assert normalize_label(blank) == ""


@pytest.mark.parametrize("label", [
    "tv_monitor", "  Potted  Plant ", "X-Ray_Machine", "HousePlant", "",
])
def test_normalising_twice_changes_nothing(label):
    """TargetLabels checks a label equals its normal form; that needs idempotence."""
    once = normalize_label(label)
    assert normalize_label(once) == once
