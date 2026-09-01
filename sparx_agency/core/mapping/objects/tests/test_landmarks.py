"""Tests for the object-landmark map: dedupe, confirmation, stable colors."""
from __future__ import annotations

import pytest

from sparx_agency.core.mapping.objects.landmarks import (
    ObjectLandmarkMap,
    class_color,
)


class TestDedupe:
    def test_within_radius_merges_with_running_average(self):
        lmap = ObjectLandmarkMap(dedupe_radius_m=0.70, min_observations=2)
        lmap.observe("chair", (0.0, 0.0))
        lm = lmap.observe("chair", (0.6, 0.0))
        assert len(lmap) == 1
        assert lm.count == 2
        assert lm.xy == pytest.approx((0.3, 0.0))

    def test_dedupe_is_against_the_running_centroid(self):
        """Ported semantics: (0.9, 0) is > 0.7 from the first observation but
        within 0.7 of the running centroid (0.3, 0), so it still merges."""
        lmap = ObjectLandmarkMap()
        lmap.observe("chair", (0.0, 0.0))
        lmap.observe("chair", (0.6, 0.0))
        lm = lmap.observe("chair", (0.9, 0.0))
        assert len(lmap) == 1
        assert lm.count == 3
        assert lm.xy == pytest.approx((0.5, 0.0))

    def test_beyond_radius_opens_a_new_landmark(self):
        lmap = ObjectLandmarkMap(dedupe_radius_m=0.70)
        a = lmap.observe("chair", (0.0, 0.0))
        b = lmap.observe("chair", (1.0, 0.0))
        assert len(lmap) == 2
        assert (a.id, b.id) == (0, 1)
        assert a.xy == pytest.approx((0.0, 0.0))
        assert b.xy == pytest.approx((1.0, 0.0))

    def test_distinct_classes_never_merge(self):
        lmap = ObjectLandmarkMap()
        a = lmap.observe("chair", (0.0, 0.0))
        b = lmap.observe("table", (0.05, 0.0))   # well inside the radius
        assert len(lmap) == 2
        assert a is not b
        assert a.count == 1 and b.count == 1

    def test_merge_returns_the_live_landmark(self):
        lmap = ObjectLandmarkMap()
        first = lmap.observe("bed", (2.0, 3.0))
        again = lmap.observe("bed", (2.1, 3.0))
        assert again is first


class TestConfirmation:
    def test_single_observation_is_not_confirmed(self):
        lmap = ObjectLandmarkMap(min_observations=2)
        lmap.observe("chair", (0.0, 0.0))
        assert lmap.confirmed() == []

    def test_second_observation_confirms(self):
        lmap = ObjectLandmarkMap(min_observations=2)
        lmap.observe("chair", (0.0, 0.0))
        lm = lmap.observe("chair", (0.1, 0.0))
        assert lmap.confirmed() == [lm]

    def test_min_observations_one_confirms_immediately(self):
        lmap = ObjectLandmarkMap(min_observations=1)
        lm = lmap.observe("chair", (0.0, 0.0))
        assert lmap.confirmed() == [lm]

    def test_all_landmarks_lists_unconfirmed_too(self):
        lmap = ObjectLandmarkMap(min_observations=2)
        lmap.observe("chair", (0.0, 0.0))
        lmap.observe("table", (5.0, 5.0))
        lmap.observe("table", (5.1, 5.0))
        assert len(lmap.all_landmarks()) == 2
        assert len(lmap.confirmed()) == 1

    @pytest.mark.parametrize("kwargs", [dict(dedupe_radius_m=0.0),
                                        dict(dedupe_radius_m=-1.0),
                                        dict(min_observations=0)])
    def test_invalid_constructor_args_raise(self, kwargs):
        with pytest.raises(ValueError):
            ObjectLandmarkMap(**kwargs)


class TestClassColor:
    def test_deterministic_across_calls(self):
        assert class_color("chair") == class_color("chair")

    def test_distinct_classes_get_distinct_colors(self):
        # md5-derived hues; verified distinct for these names.
        assert class_color("chair") != class_color("bed")

    def test_channels_are_unit_range(self):
        for name in ("chair", "bed", "person", "door", "extinguisher"):
            r, g, b = class_color(name)
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0

    def test_known_value_pinned(self):
        """PYTHONHASHSEED-independence guard: md5('chair') hue is fixed
        forever, unlike the old node's salted builtin hash()."""
        r, g, b = class_color("chair")
        assert (r, g, b) == pytest.approx(
            (0.19999999999999996, 0.5546639919759275, 1.0))
