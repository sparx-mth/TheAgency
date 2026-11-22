from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.check_membership_render import agreement, level_frames

CLUSTERS = [{"facing": 10, "los": 2, "depth": 6, "both": 2}, {"facing": 10, "los": 4, "depth": 8, "both": 4}]


def test_agreement_is_the_share_of_frames_each_test_keeps_among_the_other():
    got = agreement(CLUSTERS)
    assert got == {"depth_among_facing": 14 / 20, "depth_among_los": 1.0, "los_among_depth": 6 / 14}


def test_agreement_without_frames_is_none():
    assert agreement([{"facing": 0, "los": 0, "depth": 0, "both": 0}]) == {"depth_among_facing": None, "depth_among_los": None, "los_among_depth": None}


def test_level_frames_keep_the_steps_with_a_level_camera():
    rows = [{"camera_pitch": 0.0}, {"camera_pitch": -0.5}, {"camera_pitch": 0.04}]
    assert level_frames(rows, last=2) == [0, 2]
