"""The support-surface test: exact class membership and the under-the-footprint geometry."""
from __future__ import annotations

import pytest

from sparx_agency.core.mapping.objects.support_surface import (
    SUPPORT_SURFACE_CLASSES, is_support_surface, supports)


def test_support_surface_membership_is_exact_and_case_insensitive():
    assert is_support_surface("Cabinet") and is_support_surface(" dining table ")
    assert not is_support_surface("tv") and not is_support_surface("cabinet door")
    assert is_support_surface("TV STAND", ("tv stand",)) and not is_support_surface("tv stand", ("desk",))
    assert "cabinet" in SUPPORT_SURFACE_CLASSES and "bed" not in SUPPORT_SURFACE_CLASSES


def test_a_surface_supports_a_target_under_its_footprint_at_or_below_its_height():
    tv = (2.0, 0.0, 1.5)
    assert supports(tv, (2.1, 0.05, 0.9), radius_m=0.5)              # the dresser under it
    assert supports(tv, (2.0, 0.0, 1.58), radius_m=0.5)              # depth noise on a low television
    assert not supports(tv, (2.0, 0.0, 1.9), radius_m=0.5)           # a shelf ABOVE the target is not its support
    assert not supports(tv, (2.8, 0.0, 0.9), radius_m=0.5)           # a table across the room
    with pytest.raises(ValueError):
        supports(tv, (2.0, 0.0, 0.9), radius_m=0.0)
