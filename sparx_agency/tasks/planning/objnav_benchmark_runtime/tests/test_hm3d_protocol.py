"""The published HM3D ObjectNav settings, pinned field by field against silent drift.

Every number here is habitat-lab's own ``objectnav_hm3d`` configuration, and
none of them fails loudly when it is wrong: a 0.2 m success radius, a 10-degree
turn, a camera at the body's height rather than the sensor's, or a shipped
navmesh instead of the agent's each produce a complete run with plausible SR and
SPL that is not comparable to any published table. ``report.py`` refuses a run
whose manifest ``protocol`` block differs from this module by so much as one
field, so the whole serialised dataclass -- not a chosen subset -- is the
contract, and the two versions must stay identical everywhere except in what
names the episodes and locates their scenes.

No simulator, no dataset and no scene mesh: the protocol is plain numbers, and
importing this module in a numpy-only environment is itself the check that it
stays that way.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
import math

import pytest

from sparx_agency.core.planning.objnav.camera_intrinsics import (
    intrinsics_from_hfov,
    intrinsics_from_vfov,
)
from sparx_agency.core.planning.objnav.types.actions import (
    ALL_ACTIONS,
    DiscreteAction,
)
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import (
    KinematicTolerance,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import (
    APEXNAV_SUCCESS_DISTANCE_M,
    HM3D_V1,
    HM3D_V2,
    HM3DProtocol,
    PROTOCOLS,
    SCENE_DOWNLOAD_UIDS,
    SCENE_RELEASES,
    SCENE_ROOT_NAMES,
    protocol_for,
)

#: habitat-lab's ``objectnav_hm3d`` benchmark configuration, read off its config
#: rather than inferred, and shared by both published dataset versions.
PUBLISHED = {
    "split": "val",
    "max_steps": 500,
    "width": 640,
    "height": 480,
    "hfov_deg": 79.0,
    "camera_height_m": 0.88,
    "agent_height_m": 0.88,
    "agent_radius_m": 0.18,
    "min_depth_m": 0.5,
    "max_depth_m": 5.0,
    "forward_step_m": 0.25,
    "turn_angle_deg": 30.0,
    "tilt_angle_deg": 30.0,
    "success_distance_m": 0.1,
    "distance_to": "view_points",
    "allow_sliding": False,
    "navmesh": "agent",
    "require_stop_for_success": True,
    "path_length_dimension": "3d",
    "path_length_epsilon_m": 0.0,
    "reference_habitat_lab_version": "0.2.1",
    "reference_habitat_sim_version": "0.2.4",
}

#: What each version calls itself. These three fields are the only ones the two
#: published protocols are allowed to disagree about.
IDENTITIES = {
    "v1": {"dataset_version": "v1", "protocol_id": "hm3d-objectnav-v1-val/1",
           "benchmark": "hm3d_v1"},
    "v2": {"dataset_version": "v2", "protocol_id": "hm3d-objectnav-v2-val/1",
           "benchmark": "hm3d_v2"},
}


@pytest.mark.parametrize("protocol", [HM3D_V1, HM3D_V2],
                         ids=["v1", "v2"])
def test_published_protocols_carry_habitat_labs_objectnav_hm3d_numbers(protocol):
    """The manifest block is compared whole, so an added or edited field is a new experiment."""
    expected = dict(PUBLISHED, **IDENTITIES[protocol.dataset_version])
    assert asdict(protocol) == expected


def test_the_two_dataset_versions_differ_only_in_identity_and_where_their_scenes_live():
    """A v2 run that inherits anything else from v1 is a v1 result filed under the wrong name."""
    v1, v2 = asdict(HM3D_V1), asdict(HM3D_V2)
    assert sorted(k for k in v1 if v1[k] != v2[k]) == [
        "benchmark", "dataset_version", "protocol_id"]
    assert (HM3D_V1.scene_release, HM3D_V1.scene_root_name,
            HM3D_V1.scene_download_uid) == ("hm3d-v0.1", "hm3d", "hm3d_val_v0.1")
    assert (HM3D_V2.scene_release, HM3D_V2.scene_root_name,
            HM3D_V2.scene_download_uid) == ("hm3d-v0.2", "hm3d_v0.2",
                                            "hm3d_val_v0.2")
    assert HM3D_V1.scene_root_name != HM3D_V2.scene_root_name
    assert HM3D_V1.camera() == HM3D_V2.camera()
    assert HM3D_V1.actions() == HM3D_V2.actions()
    assert HM3D_V1.kinematics() == HM3D_V2.kinematics()
    assert HM3D_V1.evaluation_settings() == HM3D_V2.evaluation_settings()


def test_every_known_dataset_version_has_a_release_a_scene_root_and_a_download_uid():
    """A version listed in one map and missing from another only fails at provisioning time."""
    assert (sorted(PROTOCOLS) == sorted(SCENE_RELEASES) == sorted(SCENE_ROOT_NAMES)
            == sorted(SCENE_DOWNLOAD_UIDS) == ["v1", "v2"])
    for version, protocol in PROTOCOLS.items():
        assert protocol.dataset_version == version
        assert protocol.benchmark == "hm3d_" + version
        assert protocol.scene_release and protocol.scene_root_name
        assert protocol.scene_download_uid
    assert len({p.protocol_id for p in PROTOCOLS.values()}) == len(PROTOCOLS)


def test_the_camera_reads_the_published_field_of_view_horizontally():
    """The same 79 degrees taken as vertical shortens both focal lengths by a quarter and nothing downstream can tell."""
    camera = HM3D_V1.camera()
    k = camera.intrinsics
    assert k == intrinsics_from_hfov(640, 480, 79.0)
    assert k.fx == k.fy == pytest.approx(320.0 / math.tan(math.radians(79.0) / 2))
    assert math.degrees(2 * math.atan(240.0 / k.fy)) == pytest.approx(63.453,
                                                                     abs=1e-3)
    assert intrinsics_from_vfov(640, 480, 79.0).fx == pytest.approx(k.fx * 0.75)
    assert (k.cx, k.cy) == ((640 - 1) / 2.0, (480 - 1) / 2.0)
    assert (camera.height_m, camera.min_depth_m, camera.max_depth_m) == (0.88,
                                                                        0.5, 5.0)
    assert replace(HM3D_V1, camera_height_m=1.2).camera().height_m == 1.2


def test_actions_are_all_six_with_a_camera_pitch_habitat_never_clamps():
    """A clamp invented here would refuse LOOKs habitat-sim executes, and half the tilt range with them."""
    spec = HM3D_V1.actions()
    assert spec.actions == ALL_ACTIONS and len(spec.actions) == 6
    assert spec.has_camera_tilt
    assert spec.allows(DiscreteAction.LOOK_UP)
    assert spec.allows(DiscreteAction.LOOK_DOWN)
    assert spec.min_pitch_deg is None and spec.max_pitch_deg is None
    assert spec.min_pitch_rad is None and spec.max_pitch_rad is None
    assert (spec.forward_step_m, spec.turn_angle_deg,
            spec.tilt_angle_deg) == (0.25, 30.0, 30.0)


def test_evaluation_settings_demand_stop_and_exact_three_dimensional_path_accounting():
    """Crediting a fly-by success, or trimming the path with an epsilon, moves SR and SPL against every published row."""
    settings = HM3D_V1.evaluation_settings()
    assert settings.require_stop_for_success is True
    assert settings.path_length_dimension == "3d"
    assert settings.path_length_epsilon_m == 0.0
    assert settings.kinematics == KinematicTolerance()
    assert replace(HM3D_V1, path_length_epsilon_m=0.01
                   ).evaluation_settings().path_length_epsilon_m == 0.01


def test_the_navmesh_is_the_agents_own_and_no_unmeasured_third_option_is_accepted():
    """The navmesh sets what is reachable and therefore the shortest path in every SPL, so it is chosen, never guessed."""
    assert HM3D_V1.navmesh == "agent" and asdict(HM3D_V1)["navmesh"] == "agent"
    assert replace(HM3D_V1, navmesh="published").navmesh == "published"
    with pytest.raises(ValueError) as error:
        replace(HM3D_V1, navmesh="shipped")
    assert "agent" in str(error.value) and "published" in str(error.value)
    for value in (None, "", "Agent", "basis"):
        with pytest.raises(ValueError):
            replace(HM3D_V1, navmesh=value)


def test_rescoring_at_another_radius_renames_the_protocol_and_leaves_the_official_one_intact():
    """ApexNav's 0.2 m row must stay traceable as a deviation; an unrenamed copy would read as the official number."""
    assert APEXNAV_SUCCESS_DISTANCE_M == 0.2 != HM3D_V1.success_distance_m
    loose = HM3D_V1.with_success_distance(APEXNAV_SUCCESS_DISTANCE_M)
    assert loose.success_distance_m == 0.2
    assert loose.protocol_id == "hm3d-objectnav-v1-val/1+success0.2m"
    # The radius is written without rounding, so no two radii share an id.
    # A fixed two-decimal format rendered 1 mm and 2 mm alike, and as "0.00m" --
    # a radius __post_init__ refuses outright.
    millimetres = {HM3D_V1.with_success_distance(m).protocol_id
                   for m in (0.001, 0.002, 0.1, 0.2)}
    assert len(millimetres) == 4
    official, rescored = asdict(HM3D_V1), asdict(loose)
    assert {k for k in official if official[k] != rescored[k]} == {
        "success_distance_m", "protocol_id"}
    assert HM3D_V1.success_distance_m == 0.1
    assert HM3D_V1.protocol_id == "hm3d-objectnav-v1-val/1"
    with pytest.raises(FrozenInstanceError):
        HM3D_V1.success_distance_m = 0.2


def test_a_success_radius_that_is_not_a_small_positive_distance_is_refused():
    """A radius of zero can never succeed and a metres-for-centimetres slip succeeds anywhere in the room."""
    for metres in (0.0, -0.1, 1.01, 2.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            HM3D_V1.with_success_distance(metres)
        with pytest.raises(ValueError):
            replace(HM3D_V1, success_distance_m=metres)
    assert HM3D_V1.with_success_distance(1.0).success_distance_m == 1.0


def test_an_unknown_version_or_a_distance_measured_to_the_object_centre_is_refused():
    """habitat-lab measures to the goal view points; to the object centre every DTG is larger and every SR lower."""
    with pytest.raises(ValueError) as error:
        HM3DProtocol(dataset_version="v3", protocol_id="hm3d-objectnav-v3",
                     benchmark="hm3d_v3")
    assert "v1" in str(error.value) and "v2" in str(error.value)
    for target in ("object_centre", "euclidean", "", None):
        with pytest.raises(ValueError):
            replace(HM3D_V1, distance_to=target)


def test_protocol_for_returns_the_published_object_and_names_what_exists():
    """The version comes off a command line, where a typo must say what to type instead."""
    assert protocol_for("v1") is HM3D_V1
    assert protocol_for("v2") is HM3D_V2
    with pytest.raises(KeyError) as error:
        protocol_for("v3")
    message = str(error.value)
    assert "v3" in message and "v1" in message and "v2" in message
    for version in ("", "V1", "hm3d_v1", "1", None):
        with pytest.raises(KeyError):
            protocol_for(version)


# -- development splits ----------------------------------------------------

@pytest.mark.parametrize("protocol", (HM3D_V1, HM3D_V2))
def test_a_development_split_renames_the_protocol_so_it_cannot_pass_for_validation(protocol):
    """The manifest's id is where a reader learns which split produced a number.

    An id still reading ``-val`` while the run was on ``train`` would be worse
    than no id at all, and the harness files rows under (benchmark, split), so
    the two can never be aggregated together by accident either.
    """
    train = protocol.with_split("train")
    assert train.split == "train"
    assert train.protocol_id == protocol.protocol_id.replace("-val/", "-train/")
    assert "val" not in train.protocol_id
    assert protocol.split == "val" and protocol.with_split("val") is protocol
    assert train.benchmark == protocol.benchmark
    changed = {k for k in asdict(protocol)
               if asdict(protocol)[k] != asdict(train)[k]}
    assert changed == {"split", "protocol_id"}


@pytest.mark.parametrize("split", ("", "   ", None, 3))
def test_a_split_must_be_named(split):
    """A blank split would produce a results directory nothing could identify."""
    with pytest.raises(ValueError):
        HM3D_V2.with_split(split)


def test_a_renamed_split_still_composes_with_a_rescored_radius():
    """Development and re-scoring are independent deviations; both must show."""
    both = HM3D_V2.with_split("train").with_success_distance(0.2)
    assert both.protocol_id == "hm3d-objectnav-v2-train/1+success0.2m"
