"""The HM3D episode reader, on synthetic shards written in the publisher's shape.

Everything this parser gets wrong is invisible at run time: a quaternion read in
the wrong order still flies, episodes renumbered per shard still score, and v1
rows opened against v0.2 geometry still produce a number. So the cases here are
the ones whose failure mode is a plausible-looking result rather than a crash.

No simulator, no meshes, no download: the shards are written into ``tmp_path``
and the one test that touches the real installed split skips when it is absent.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from sparx_agency.core.planning.objnav.labels.datasets.hm3d import CATEGORIES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import (
    HM3DDataset, PUBLISHED_VAL, published_counts, xyzw_to_wxyz)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import (
    HM3D_V1, HM3D_V2)

#: A unit rotation with four distinct components, so XYZW and WXYZ readings of
#: it disagree everywhere. ``w`` is normalised to sqrt(0.86); a rounder 0.927
#: is not a unit quaternion and the loader is right to refuse it.
TILTED_XYZW = (0.1, 0.2, 0.3, 0.92736)

#: One evaluator-only goal row, shaped like the publisher's.
GOAL = {"position": [1.0, 0.0, 2.0], "object_id": 7, "object_category": "chair",
        "view_points": [{"agent_state": {"position": [1.2, 0.0, 2.0],
                                         "rotation": [0.0, 0.0, 0.0, 1.0]},
                         "iou": 0.31}]}

REAL_V2_VAL = Path("~/datasets/objectnav/hm3d/v2/val").expanduser()


def row(folder, category="chair", *, release="hm3d_v0.2", stem=None, episode_id="0",
        rotation=(0.0, 0.0, 0.0, 1.0), position=(1.0, 0.5, 2.0), geodesic=4.25):
    """One episode row as habitat-lab writes it: XYZW rotation, emptied goals."""
    stem = folder.split("-", 1)[1] if stem is None else stem
    return {"episode_id": episode_id,
            "scene_id": "%s/val/%s/%s.basis.glb" % (release, folder, stem),
            "scene_dataset_config": "./data/scene_datasets/%s/x.json" % release,
            "start_position": list(position),
            "start_rotation": list(rotation),
            "info": {"geodesic_distance": geodesic},
            "goals": [],
            "object_category": category}


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)


def write_split(directory, shards, *, split="val", categories=CATEGORIES, goals="derive"):
    """Write ``<split>.json.gz`` and one ``content/<stem>.json.gz`` per scene.

    ``shards`` maps a shard stem to its rows. Like the published index, the
    split file carries the two category tables and no episodes of its own.
    ``goals`` derives the de-duplicated ``goals_by_category`` from the rows by
    default; pass a mapping to break it, or ``None`` for the older shard shape
    that keeps its goals inline on every row.
    """
    directory = Path(directory)
    table = {name: index for index, name in enumerate(categories)}
    _write(directory / ("%s.json.gz" % split),
           {"episodes": [], "category_to_task_category_id": table,
            "category_to_scene_annotation_category_id": table})
    for stem, rows in shards.items():
        payload = {"episodes": rows, "category_to_task_category_id": table}
        if goals == "derive":
            payload["goals_by_category"] = {
                "%s_%s" % (Path(r["scene_id"]).name, r["object_category"]): [GOAL]
                for r in rows}
        elif goals is not None:
            payload["goals_by_category"] = goals
        _write(directory / "content" / ("%s.json.gz" % stem), payload)
    return directory


def load(directory, protocol=HM3D_V2, **options):
    """A labelled subset load: the published counts and the meshes are not needed."""
    settings = {"full": False, "require_scenes": False}
    settings.update(options)
    return HM3DDataset(directory, None, protocol, **settings)


def one_scene(tmp_path, **kwargs):
    return write_split(tmp_path, {"TEEsavR23oF": [row("00800-TEEsavR23oF", **kwargs)]})


def test_the_published_xyzw_start_rotation_is_reordered_into_wxyz(tmp_path):
    """Forwarding the publisher's order would start every episode facing
    somewhere else, and no later stage -- planner, scorer or replay -- could
    tell, because a rotation is a rotation. It is caught here or not at all."""
    assert xyzw_to_wxyz(TILTED_XYZW) == (0.92736, 0.1, 0.2, 0.3)

    episode = next(iter(load(one_scene(tmp_path, rotation=TILTED_XYZW)).episodes.values()))
    assert episode.start_rotation_wxyz == (0.92736, 0.1, 0.2, 0.3)
    assert episode.start_rotation_wxyz[0] == max(episode.start_rotation_wxyz)
    assert episode.start_rotation_wxyz != TILTED_XYZW
    assert episode.start_position == (1.0, 0.5, 2.0)


@pytest.mark.parametrize("rotation", [
    (0.0, 0.0, 0.0, 0.5), (0.1, 0.2, 0.3, 0.927), (2.0, 0.0, 0.0, 0.0),
], ids=["half_length", "rounded_w", "over_length"])
def test_a_start_rotation_that_is_not_a_unit_quaternion_is_refused(tmp_path, rotation):
    """A row that is not a rotation would be normalised away silently by most
    quaternion libraries, turning a corrupt release into a plausible heading."""
    with pytest.raises(ValueError, match="unit quaternion"):
        xyzw_to_wxyz(rotation)
    with pytest.raises(ValueError, match="unit quaternion"):
        load(one_scene(tmp_path, rotation=rotation))


@pytest.mark.parametrize("rotation", [
    (0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0, 0.0), (float("nan"), 0.0, 0.0, 1.0),
], ids=["too_short", "too_long", "not_finite"])
def test_a_start_rotation_that_is_not_four_finite_numbers_is_refused(rotation):
    """The shape check has to run before the unit check, or a three-vector
    unpacks into the wrong variables instead of being reported."""
    with pytest.raises(ValueError, match="must contain 4 finite numbers"):
        xyzw_to_wxyz(rotation)


def test_episode_ids_are_scene_qualified_so_the_publishers_renumbering_cannot_collide(tmp_path):
    """habitat-lab renumbers ``episode_id`` to ``str(i)`` inside each shard, so
    the on-disk ids of two scenes are the same handful of integers. Keyed on
    those, a 1000-episode run would silently score a few dozen episodes."""
    shards = {"TEEsavR23oF": [row("00800-TEEsavR23oF", episode_id="0"),
                              row("00800-TEEsavR23oF", "bed", episode_id="1")],
              "zt1RVoi7PcG": [row("00839-zt1RVoi7PcG", episode_id="0"),
                              row("00839-zt1RVoi7PcG", "sofa", episode_id="1")]}
    dataset = load(write_split(tmp_path, shards))

    assert dataset.episode_ids() == ("00800-TEEsavR23oF/000000", "00800-TEEsavR23oF/000001",
                                     "00839-zt1RVoi7PcG/000000", "00839-zt1RVoi7PcG/000001")
    assert [e.source_episode_id for e in dataset.episodes.values()] == ["0", "1", "0", "1"]
    assert dataset.scene_counts == {"00800-TEEsavR23oF": 2, "00839-zt1RVoi7PcG": 2}
    # Stable: the same release read twice is the same identities, in order, so a
    # resume can match a half-finished run row for row.
    assert load(tmp_path).episode_ids() == dataset.episode_ids()


def test_goals_are_keyed_by_scene_file_and_category(tmp_path):
    """``goals_by_category`` is keyed by ``ObjectGoalNavEpisode.goals_key``,
    which is the scene's *file* name and not its folder; get it wrong and every
    lookup misses rather than resolving to the wrong goal."""
    dataset = load(one_scene(tmp_path, category="tv_monitor"))
    episode = dataset.episodes["00800-TEEsavR23oF/000000"]

    assert episode.goals_key == "TEEsavR23oF.basis.glb_tv_monitor"
    assert episode.scene_key == "00800-TEEsavR23oF"
    assert episode.scene_relative_path == "hm3d_v0.2/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
    assert dataset.goals_for(episode) == [GOAL]


@pytest.mark.parametrize("goals,message", [
    ({"TEEsavR23oF.basis.glb_bed": [GOAL]}, "No goals for"),
    ({"TEEsavR23oF.basis.glb_chair": []}, "is empty"),
], ids=["wrong_category", "empty_list"])
def test_an_episode_whose_goals_are_missing_is_refused(tmp_path, goals, message):
    """An episode with no view points cannot be scored at all: distance-to-goal
    has nothing to measure to, so it would fail at the step budget and read as a
    miss by the method rather than as a broken release."""
    with pytest.raises(ValueError, match=message):
        load(write_split(tmp_path, {"TEEsavR23oF": [row("00800-TEEsavR23oF")]}, goals=goals))


def test_a_shard_that_predates_goal_de_duplication_is_read(tmp_path):
    """Older published shards carry the goals inline on every row instead of in
    ``goals_by_category``; refusing them would reject a legitimate release."""
    rows = [row("00800-TEEsavR23oF"), row("00800-TEEsavR23oF", episode_id="1")]
    for entry in rows:
        entry["goals"] = [GOAL]
    dataset = load(write_split(tmp_path, {"TEEsavR23oF": rows}, goals=None))

    assert len(dataset.episodes) == 2
    assert dataset.goals_for(dataset.episodes["00800-TEEsavR23oF/000001"]) == [GOAL]


def test_episodes_of_another_scene_release_than_the_protocol_owns_are_refused(tmp_path):
    """v1 rows say ``hm3d/val/...`` and v2 rows say ``hm3d_v0.2/val/...``, and
    only the name distinguishes them on disk. Run one against the other's
    geometry and the scenes are simply not the ones the episodes were made on --
    every metric stays well-formed and every one of them is meaningless."""
    directory = write_split(tmp_path, {"TEEsavR23oF": [row("00800-TEEsavR23oF",
                                                           release="hm3d")]})
    with pytest.raises(ValueError, match="scene release"):
        load(directory, HM3D_V2)

    v1 = load(directory, HM3D_V1)
    assert v1.episode_ids() == ("00800-TEEsavR23oF/000000",)
    assert v1.episodes["00800-TEEsavR23oF/000000"].scene_relative_path.startswith("hm3d/val/")


def test_the_habitat_lab_scene_prefix_is_stripped_rather_than_doubled(tmp_path):
    """habitat-lab strips ``data/scene_datasets/`` before joining to the scenes
    directory. Keeping it would point every mesh lookup one release deeper."""
    entry = row("00800-TEEsavR23oF")
    entry["scene_id"] = ("data/scene_datasets/hm3d_v0.2/val/00800-TEEsavR23oF/"
                         "TEEsavR23oF.basis.glb")
    dataset = load(write_split(tmp_path, {"TEEsavR23oF": [entry]}))

    assert dataset.episodes["00800-TEEsavR23oF/000000"].scene_relative_path == (
        "hm3d_v0.2/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb")


def test_a_shard_holding_another_scenes_episodes_is_refused(tmp_path):
    """The shard is named for the stem and the folder is ``00NNN-<stem>``, so a
    mismatch is the one sign that a file was copied or renamed by hand -- and
    the episodes would then be run in a building they were not generated in."""
    misfiled = {"TEEsavR23oF": [row("00839-zt1RVoi7PcG")]}
    with pytest.raises(ValueError, match="holds an episode of scene"):
        load(write_split(tmp_path, misfiled))


def test_an_object_category_outside_the_six_hm3d_goals_is_refused(tmp_path):
    """The category reaches the detector as a prompt; an unmapped spelling would
    be searched for as nothing and score a guaranteed zero."""
    with pytest.raises(ValueError, match="Unknown HM3D object_category"):
        load(write_split(tmp_path, {"TEEsavR23oF": [row("00800-TEEsavR23oF", "couch")]}))


def test_a_release_whose_vocabulary_is_not_the_six_hm3d_goals_is_refused(tmp_path):
    """The index's category table is the release identifying itself; a different
    vocabulary means a different task, not a different spelling."""
    directory = write_split(tmp_path, {"TEEsavR23oF": [row("00800-TEEsavR23oF")]},
                            categories=CATEGORIES + ("sink",))
    with pytest.raises(ValueError, match="six HM3D ObjectNav goals"):
        load(directory)


@pytest.mark.parametrize("geodesic", [-1.0, float("nan")], ids=["negative", "not_finite"])
def test_a_published_geodesic_that_is_not_a_length_is_refused(tmp_path, geodesic):
    """It is the cross-check against our own measurement; a nonsense value would
    make that check pass by never disagreeing."""
    with pytest.raises(ValueError, match="not a length"):
        load(one_scene(tmp_path, geodesic=geodesic))


def _published_shape(directory, protocol, scene_count):
    shards = {}
    for index in range(scene_count):
        stem = "scene%02d" % index
        shards[stem] = [row("00%03d-%s" % (index, stem), CATEGORIES[index % len(CATEGORIES)],
                            release=protocol.scene_root_name)]
    return write_split(directory, shards)


def test_a_full_run_enforces_the_published_counts_and_names_the_version_it_expected(tmp_path):
    """A split short of a few scenes still runs and still prints a success rate.
    Compared against a published table, that number is simply wrong, so the
    counts are a precondition of the run and not a warning."""
    partial = _published_shape(tmp_path / "v2", HM3D_V2, 1)
    with pytest.raises(ValueError, match="HM3D v2 val is documented as 36 scenes"):
        HM3DDataset(partial, None, HM3D_V2, require_scenes=False)
    assert len(load(partial).episodes) == 1

    scenes = PUBLISHED_VAL["v1"]["scenes"]
    thin = _published_shape(tmp_path / "v1", HM3D_V1, scenes)
    with pytest.raises(ValueError, match="HM3D v1 val is documented as 2000 episodes"):
        HM3DDataset(thin, None, HM3D_V1, require_scenes=False)
    assert len(load(thin, HM3D_V1).episodes) == scenes


def test_selecting_scenes_is_refused_on_a_full_run(tmp_path):
    """A subset scored as the published split is the mislabelling this whole
    loader exists to prevent, so the two options may not be combined."""
    directory = one_scene(tmp_path)
    with pytest.raises(ValueError, match="pass full=False"):
        HM3DDataset(directory, None, HM3D_V2, scenes=["TEEsavR23oF"], require_scenes=False)


@pytest.mark.parametrize("selection", [["00800-TEEsavR23oF"], ["TEEsavR23oF"]],
                         ids=["folder_name", "shard_stem"])
def test_a_scene_may_be_named_by_its_folder_or_by_its_shard_stem(tmp_path, selection):
    """One scene has two names on disk -- the folder a person reads off the
    filesystem and the stem the shard is named for -- and a subset run typed
    either way must select the same episodes rather than silently find none."""
    shards = {"TEEsavR23oF": [row("00800-TEEsavR23oF")],
              "zt1RVoi7PcG": [row("00839-zt1RVoi7PcG")]}
    dataset = load(write_split(tmp_path, shards), scenes=selection)

    assert dataset.episode_ids() == ("00800-TEEsavR23oF/000000",)
    assert dataset.selected_scenes == tuple(selection)
    assert dataset.manifest(hash_scenes=False)["selected_scenes"] == list(selection)


def test_selecting_a_scene_the_split_does_not_hold_is_refused_by_name(tmp_path):
    """A typo'd scene name that quietly selected nothing would report a perfect
    score over zero episodes."""
    directory = write_split(tmp_path, {"TEEsavR23oF": [row("00800-TEEsavR23oF")]})
    with pytest.raises(FileNotFoundError, match="No content shard for: zt1RVoi7PcG"):
        load(directory, scenes=["00839-zt1RVoi7PcG"])


def scene_files(root, *, glb=b"mesh", navmesh=b"navmesh"):
    """The two published files of 00800-TEEsavR23oF, either of them omissible."""
    folder = Path(root) / "hm3d_v0.2" / "val" / "00800-TEEsavR23oF"
    folder.mkdir(parents=True)
    for name, payload in (("TEEsavR23oF.basis.glb", glb),
                          ("TEEsavR23oF.basis.navmesh", navmesh)):
        if payload is not None:
            (folder / name).write_bytes(payload)
    return folder


def test_the_published_navmesh_is_the_basis_sibling_of_the_mesh(tmp_path):
    """habitat-sim derives it as ``splitext(scene_id)[0] + ".navmesh"``, which
    keeps the ``.basis`` -- replacing ``.basis.glb`` wholesale would look for a
    file that does not exist and report a complete installation as missing."""
    directory = one_scene(tmp_path / "episodes")
    folder = scene_files(tmp_path / "scenes")
    dataset = HM3DDataset(directory, tmp_path / "scenes", HM3D_V2, full=False)

    glb, navmesh = dataset.scene_assets(dataset.episodes["00800-TEEsavR23oF/000000"])
    assert (glb.name, navmesh.name) == ("TEEsavR23oF.basis.glb", "TEEsavR23oF.basis.navmesh")
    assert glb.parent == folder and navmesh.is_file()


@pytest.mark.parametrize("files,missing", [
    ({"navmesh": None}, "TEEsavR23oF.basis.navmesh"),
    ({"glb": None}, "TEEsavR23oF.basis.glb"),
    ({"glb": b""}, "TEEsavR23oF.basis.glb"),
], ids=["no_navmesh", "no_mesh", "empty_mesh"])
def test_a_missing_scene_asset_is_named_before_the_run_starts(tmp_path, files, missing):
    """Discovering it at episode 900 costs the whole run; an empty file is the
    usual shape of an interrupted multi-gigabyte download."""
    directory = one_scene(tmp_path / "episodes")
    scene_files(tmp_path / "scenes", **files)
    with pytest.raises(FileNotFoundError, match=missing):
        HM3DDataset(directory, tmp_path / "scenes", HM3D_V2, full=False)


def test_a_run_may_not_omit_the_scenes_while_an_episode_only_check_may(tmp_path):
    """Checking the meshes is what distinguishes a real run from a parse of the
    published rows, so it is refused by omission rather than defaulted away."""
    directory = one_scene(tmp_path)
    with pytest.raises(ValueError, match="only an episode-only"):
        HM3DDataset(directory, None, HM3D_V2, full=False)

    dataset = load(directory)
    with pytest.raises(ValueError, match="without scene assets"):
        dataset.scene_assets(dataset.episodes["00800-TEEsavR23oF/000000"])


def test_a_split_directory_without_the_index_or_content_says_which_file_is_missing(tmp_path):
    """The usual mistake is pointing at the dataset root instead of the split."""
    with pytest.raises(FileNotFoundError, match="val.json.gz"):
        load(tmp_path / "nowhere")
    empty = write_split(tmp_path, {})
    with pytest.raises(FileNotFoundError, match="No content"):
        load(empty)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_manifest_hashes_the_episode_files_even_when_the_scenes_are_not_hashed(tmp_path):
    """A swapped release is invisible in every other record of a run, so the
    manifest is the only place a resume can notice it. Hashing the meshes is
    expensive and optional; hashing the episodes never is."""
    directory = one_scene(tmp_path / "episodes")
    scene_files(tmp_path / "scenes")
    dataset = HM3DDataset(directory, tmp_path / "scenes", HM3D_V2, full=False)

    manifest = dataset.manifest(hash_scenes=False)
    assert manifest["release"] == "HM3D ObjectNav v2 val on hm3d-v0.2"
    assert (manifest["dataset_version"], manifest["split"]) == ("v2", "val")
    assert manifest["episode_count"] == 1 and manifest["scene_count"] == 1
    assert manifest["scene_counts"] == {"00800-TEEsavR23oF": 1}
    assert manifest["category_to_task_category_id"] == {
        name: index for index, name in enumerate(CATEGORIES)}
    assert manifest["scene_assets_present"] and not manifest["scene_assets_hashed"]

    hashed = {entry["path"]: entry.get("sha256") for entry in manifest["files"]}
    assert len(hashed) == 4
    for path, digest in hashed.items():
        assert digest == (_digest(path) if path.endswith(".gz") else None)
        assert Path(path).stat().st_size == next(
            e["bytes"] for e in manifest["files"] if e["path"] == path)

    whole = dataset.manifest()
    assert whole["scene_assets_hashed"]
    assert all(entry["sha256"] == _digest(entry["path"]) for entry in whole["files"])


@pytest.mark.skipif(not (REAL_V2_VAL / "val.json.gz").is_file(),
                    reason="the published HM3D-v2 val episodes are not installed here")
def test_the_installed_hm3d_v2_validation_split_parses_as_published():
    """Every other test in this file agrees with a fixture this file wrote, so
    only this one can catch the parser having been written against the wrong
    idea of the publisher's format. The meshes are licensed and usually absent,
    so the rows alone are read."""
    dataset = HM3DDataset(REAL_V2_VAL, None, HM3D_V2, require_scenes=False)

    assert len(dataset.episodes) == PUBLISHED_VAL["v2"]["episodes"] == 1000
    assert len(dataset.scene_counts) == PUBLISHED_VAL["v2"]["scenes"] == 36
    assert sum(dataset.scene_counts.values()) == 1000
    assert len(set(dataset.episode_ids())) == 1000
    for episode in dataset.episodes.values():
        assert dataset.goals_for(episode)
        assert episode.category in CATEGORIES
        assert episode.scene_relative_path.startswith("hm3d_v0.2/val/")
        assert episode.goals_key.endswith(".basis.glb_" + episode.category)
        assert episode.episode_id.startswith(episode.scene_key + "/")
        assert abs(sum(c * c for c in episode.start_rotation_wxyz) - 1.0) < 1e-3
    # The publisher really does reuse its per-shard ids, which is why ours are
    # scene-qualified rather than taken from the row.
    assert len({e.source_episode_id for e in dataset.episodes.values()}) < 1000


# -- bounded development loading -------------------------------------------

def test_a_development_split_has_no_published_count_to_be_complete_against(tmp_path):
    """Only ``val`` is a benchmark. Calling a train run complete is how a subset
    gets quoted as a split, so the loader refuses the word rather than guessing
    a count it has never been told."""
    train = HM3D_V2.with_split("train")
    assert published_counts(train) == {}
    assert published_counts(HM3D_V2) == PUBLISHED_VAL["v2"]
    write_split(tmp_path, {"TEEsavR23oF": [row("00800-TEEsavR23oF", "chair")]},
                split="train")
    with pytest.raises(ValueError, match="published counts"):
        HM3DDataset(tmp_path, None, train, require_scenes=False)


def test_capping_episodes_per_scene_keeps_a_prefix_so_identities_do_not_move(tmp_path):
    """HM3D's train split is roughly seven million episodes, some 48,000 per
    scene, so a development run is only expressible if it can be bounded. It
    must be a prefix: an episode that changed id under a different cap could
    not be compared with itself across two development runs."""
    rows = [row("00800-TEEsavR23oF", category) for category in
            ("chair", "bed", "plant", "toilet")]
    write_split(tmp_path, {"TEEsavR23oF": rows}, split="train")
    train = HM3D_V2.with_split("train")
    whole = HM3DDataset(tmp_path, None, train, full=False, require_scenes=False)
    capped = HM3DDataset(tmp_path, None, train, full=False, require_scenes=False,
                         episodes_per_scene=2)
    assert capped.episode_ids() == whole.episode_ids()[:2]
    assert capped.scene_counts == {"00800-TEEsavR23oF": 2}
    assert capped.manifest()["episodes_per_scene"] == 2
    assert whole.manifest()["episodes_per_scene"] is None


@pytest.mark.parametrize("cap", (0, -1, 1.5, True))
def test_an_unusable_episode_cap_is_refused(tmp_path, cap):
    """A zero or fractional cap would silently produce an empty or partial run."""
    write_split(tmp_path, {"TEEsavR23oF": [row("00800-TEEsavR23oF", "chair")]})
    with pytest.raises(ValueError):
        HM3DDataset(tmp_path, None, HM3D_V2, full=False, require_scenes=False,
                    episodes_per_scene=cap)


def test_a_cap_is_a_subset_and_cannot_be_called_a_complete_split(tmp_path):
    """Otherwise a capped run would be checked against the published count and,
    on a 1-per-scene cap over all 36 scenes, silently fail for the wrong reason."""
    write_split(tmp_path, {"TEEsavR23oF": [row("00800-TEEsavR23oF", "chair")]})
    with pytest.raises(ValueError, match="subset"):
        HM3DDataset(tmp_path, None, HM3D_V2, require_scenes=False,
                    episodes_per_scene=1)
