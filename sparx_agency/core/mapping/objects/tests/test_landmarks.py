"""Tests for the object-landmark map: dedupe, confirmation, stable colors."""
from __future__ import annotations

import pytest

from sparx_agency.core.mapping.objects.landmarks import (
    ObjectLandmarkMap,
    class_color,
    disc_iou,
)


class TestClassVoting:
    def test_a_lone_misidentification_on_a_well_seen_object_does_not_open_a_new_landmark(self):
        """Five bed frames, then one sofa frame at the same spot: still one bed."""
        lmap = ObjectLandmarkMap(class_votes=True, nearest_match=True)
        for step in range(5):
            lmap.observe("bed", (2.0, 1.0), frame_id=step)
        lm = lmap.observe("sofa", (2.2, 1.1), frame_id=5)
        assert len(lmap) == 1 and lm.class_name == "bed"
        assert lm.votes == {"bed": 5, "sofa": 1} and lm.count == 6
        assert lmap.confirmed() == [lm] and lmap.relabels == []

    def test_a_majority_of_the_other_class_corrects_the_earlier_misidentification(self):
        lmap = ObjectLandmarkMap(class_votes=True)
        lmap.observe("sofa", (0.0, 0.0), frame_id=0)
        for step in range(1, 4):
            lm = lmap.observe("bed", (0.1, 0.0), frame_id=step)
        assert lm.class_name == "bed" and lm.votes == {"sofa": 1, "bed": 3}
        assert lmap.relabels == [(lm.id, "sofa", "bed", 3)]
        assert [l.class_name for l in lmap.confirmed()] == ["bed"], "the sofa is gone from the map"

    def test_a_tie_is_not_confirmed_and_keeps_the_current_class(self):
        lmap = ObjectLandmarkMap(class_votes=True, min_observations=2)
        lmap.observe("sofa", (0.0, 0.0), frame_id=0)
        lmap.observe("sofa", (0.0, 0.0), frame_id=1)
        lm = lmap.observe("bed", (0.0, 0.0), frame_id=2)
        lm = lmap.observe("bed", (0.0, 0.0), frame_id=3)
        assert lm.class_name == "sofa" and lmap.confirmed() == []
        lmap.observe("bed", (0.0, 0.0), frame_id=4)
        assert lm.class_name == "bed" and lmap.confirmed() == [lm]

    def test_one_frame_votes_once_per_landmark(self):
        lmap = ObjectLandmarkMap(class_votes=True)
        lmap.observe("bed", (0.0, 0.0), frame_id=7)
        lm = lmap.observe("sofa", (0.1, 0.0), frame_id=7)
        assert lm.votes == {"bed": 1} and lm.count == 1

    def test_footprint_overlap_associates_a_large_object_seen_from_two_sides(self):
        """Two bed centroids 1.2 m apart (beyond the 0.7 m radius) with 1 m footprints are one bed."""
        lmap = ObjectLandmarkMap(class_votes=True)
        lmap.observe("bed", (0.0, 0.0), frame_id=0, radius_m=1.0)
        lm = lmap.observe("bed", (1.2, 0.0), frame_id=1, radius_m=1.0)
        assert len(lmap) == 1 and lm.count == 2 and lm.xy == pytest.approx((0.6, 0.0))
        assert lm.radius_m == pytest.approx(1.0)

    def test_small_footprints_apart_are_two_instances(self):
        lmap = ObjectLandmarkMap(class_votes=True)
        lmap.observe("cup", (0.0, 0.0), frame_id=0, radius_m=0.1)
        lmap.observe("cup", (0.9, 0.0), frame_id=1, radius_m=0.1)
        assert len(lmap) == 2

    def test_match_reports_the_landmark_an_observation_would_fold_into(self):
        lmap = ObjectLandmarkMap(class_votes=True, nearest_match=True)
        a = lmap.observe("bed", (0.0, 0.0))
        b = lmap.observe("chair", (3.0, 0.0))
        assert lmap.match((0.3, 0.0), class_name="sofa") is a
        assert lmap.match((3.2, 0.1)) is b
        assert lmap.match((1.5, 0.0)) is None
        assert lmap.match((0.3, 0.0), exclude=(a.id,)) is None, "a landmark this frame already fed is skipped"

    def test_without_voting_match_needs_a_class_and_keeps_the_ported_rule(self):
        lmap = ObjectLandmarkMap()
        a = lmap.observe("chair", (0.0, 0.0))
        assert lmap.match((0.1, 0.0), class_name="chair") is a
        assert lmap.match((0.1, 0.0), class_name="table") is None
        with pytest.raises(ValueError):
            lmap.match((0.1, 0.0))

    def test_caller_chosen_landmark_is_honoured(self):
        lmap = ObjectLandmarkMap(class_votes=True)
        a = lmap.observe("bed", (0.0, 0.0))
        lm = lmap.observe("lamp", (5.0, 5.0), landmark=a)
        assert lm is a and a.votes == {"bed": 1, "lamp": 1} and len(lmap) == 1

    @pytest.mark.parametrize("value", [0.0, 1.5, -0.2])
    def test_invalid_footprint_iou_raises(self, value):
        with pytest.raises(ValueError):
            ObjectLandmarkMap(footprint_iou=value)


class TestDiscIoU:
    def test_identical_discs(self):
        assert disc_iou((0, 0), 1.0, (0, 0), 1.0) == pytest.approx(1.0)

    def test_disjoint_discs(self):
        assert disc_iou((0, 0), 1.0, (3, 0), 1.0) == 0.0

    def test_contained_disc(self):
        assert disc_iou((0, 0), 2.0, (0.5, 0), 1.0) == pytest.approx(0.25)

    def test_half_offset_is_between(self):
        value = disc_iou((0, 0), 1.0, (1.0, 0), 1.0)
        assert 0.2 < value < 0.5

    def test_zero_radius_never_overlaps(self):
        assert disc_iou((0, 0), 0.0, (0, 0), 1.0) == 0.0


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
