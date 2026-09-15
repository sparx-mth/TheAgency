"""The HM3D provisioning checker, held to "report it, never fetch it".

Both failures this module exists for are silent ones. A split that is short of
the published table still runs to the end and produces a success rate that
looks comparable to the papers' and is not. And HM3D-v2 episodes spell their
scenes ``hm3d_v0.2/...`` while habitat-sim's downloader only ever creates the
name ``hm3d``, so v2 run against a v0.1 link explores the wrong houses without
a single error. The checks below pin the two reports that make either visible.

The instructions half is pinned to placeholders on purpose: the scene download
needs a Matterport API token, and no test, no run and no log in this repo may
ever fetch, hold or print one.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
import socket
import subprocess
import urllib.request

import pytest

from sparx_agency.core.planning.objnav.labels.datasets.hm3d import CATEGORIES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.assets import (
    EPISODE_ARCHIVES, LICENCE_URL, TOKEN_URL, check_installation, download_instructions,
    episode_dir, main, scenes_dir,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import PUBLISHED_VAL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import HM3D_V1, HM3D_V2


def _write_gz(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)


def _names(index):
    stem = "SCN%03d" % index
    return "%05d-%s" % (index, stem), stem


def _shard(protocol, folder, stem, count):
    rows, goals = [], {}
    for i in range(count):
        category = CATEGORIES[i % len(CATEGORIES)]
        rows.append({
            "episode_id": str(i),
            "scene_id": "%s/val/%s/%s.basis.glb" % (protocol.scene_root_name, folder, stem),
            "object_category": category,
            "start_position": [1.0, 0.1, -2.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "info": {"geodesic_distance": 4.5},
        })
        goals.setdefault("%s.basis.glb_%s" % (stem, category),
                         [{"position": [0.0, 0.0, 0.0],
                           "view_points": [{"agent_state": {"position": [0.5, 0.0, 0.0]}}]}])
    return {"episodes": rows, "goals_by_category": goals}


def _write_episodes(root, protocol, sizes):
    """A split of ``len(sizes)`` content shards holding ``sizes`` episodes each."""
    split = episode_dir(root, protocol)
    _write_gz(split / ("%s.json.gz" % protocol.split),
              {"category_to_task_category_id": {c: i for i, c in enumerate(CATEGORIES)}})
    names = [_names(i) for i in range(len(sizes))]
    for (folder, stem), count in zip(names, sizes):
        _write_gz(split / "content" / ("%s.json.gz" % stem),
                  _shard(protocol, folder, stem, count))
    return names


def _write_scenes(root, protocol, names, *, release_dir=None):
    release = release_dir or (scenes_dir(root) / protocol.scene_root_name)
    for folder, stem in names:
        directory = release / "val" / folder
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ("%s.basis.glb" % stem)).write_bytes(b"glb")
        (directory / ("%s.basis.navmesh" % stem)).write_bytes(b"navmesh")
    return release


def _published_sizes(protocol):
    """Shard sizes that add up to the published split, one scene per shard."""
    expected = PUBLISHED_VAL[protocol.dataset_version]
    sizes = [1] * expected["scenes"]
    sizes[0] += expected["episodes"] - expected["scenes"]
    return sizes


def _install(root, protocol=HM3D_V2):
    names = _write_episodes(root, protocol, _published_sizes(protocol))
    _write_scenes(root, protocol, names)
    return names


def _check(root, protocol=HM3D_V2, **kwargs):
    return check_installation(episode_dir(root, protocol), scenes_dir(root), protocol, **kwargs)


def _navmesh(root, protocol, folder, stem):
    return (scenes_dir(root) / protocol.scene_root_name / "val" / folder
            / ("%s.basis.navmesh" % stem))


def test_a_complete_split_is_ready_with_both_counts_beside_the_published_table(tmp_path):
    """A rate is only comparable to the published tables over the published episodes."""
    _install(tmp_path)
    report = _check(tmp_path)
    assert report["episodes_ready"] and report["scenes_ready"] and report["problems"] == []
    assert (report["episodes_found"], report["episodes_published"]) == (1000, 1000)
    assert (report["scenes_found"], report["scenes_published"]) == (36, 36)
    assert report["scene_release"] == "hm3d-v0.2" and report["scene_root_name"] == "hm3d_v0.2"
    assert report["scenes_complete"] == 36 and report["missing"] == []
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize("protocol,sizes,numbers", [
    (HM3D_V1, [1, 1, 1], ("20", "3")),
    (HM3D_V2, [1] * 35, ("36", "35")),
])
def test_a_short_split_is_not_ready_and_the_problem_names_both_counts(
        tmp_path, protocol, sizes, numbers):
    """Which published table this is measured against is per version, not per dataset."""
    _write_episodes(tmp_path, protocol, sizes)
    report = check_installation(episode_dir(tmp_path, protocol), None, protocol)
    assert not report["episodes_ready"]
    problem = " ".join(report["problems"])
    assert all(number in problem for number in numbers), problem


def test_an_episode_shortfall_is_reported_even_when_every_scene_is_present(tmp_path):
    """Half a shard is the shortfall that a per-scene check would call complete."""
    sizes = _published_sizes(HM3D_V2)
    sizes[0] -= 1
    _write_episodes(tmp_path, HM3D_V2, sizes)
    report = check_installation(episode_dir(tmp_path, HM3D_V2), None, HM3D_V2)
    assert not report["episodes_ready"]
    problem = " ".join(report["problems"])
    assert "1000" in problem and "999" in problem, problem


def test_a_missing_split_is_reported_rather_than_raised(tmp_path):
    """The checker is what a person runs when they do not know what they have."""
    report = check_installation(tmp_path / "nowhere", scenes_dir(tmp_path), HM3D_V2)
    assert not report["episodes_ready"] and not report["scenes_ready"]
    assert "val.json.gz" in report["problems"][0]


def test_scenes_root_none_checks_the_episodes_alone_without_geometry_on_disk(tmp_path):
    """Validating a freshly unzipped episode archive must not require the 5 GB half."""
    _write_episodes(tmp_path, HM3D_V2, _published_sizes(HM3D_V2))
    report = check_installation(episode_dir(tmp_path, HM3D_V2), None, HM3D_V2)
    assert report["episodes_ready"] and report["problems"] == []
    assert report["scenes_root"] is None and not report["scenes_ready"]
    assert "scenes_checked" not in report


def test_one_missing_navmesh_fails_the_scene_half_and_names_the_file(tmp_path):
    """The mesh and its navmesh are separate downloads, and habitat-sim starts happily
    on a scene whose shipped navmesh never arrived."""
    names = _install(tmp_path)
    _navmesh(tmp_path, HM3D_V2, *names[7]).unlink()
    report = _check(tmp_path)
    assert report["episodes_ready"] and not report["scenes_ready"]
    assert report["missing_count"] == 1 and report["scenes_complete"] == 35
    assert report["missing"] == [str(_navmesh(tmp_path, HM3D_V2, *names[7]))]
    assert "1 file(s) missing" in " ".join(report["problems"])


def test_a_wholly_missing_release_lists_twenty_paths_and_still_counts_them_all(tmp_path):
    """A truncated list is for reading; the count is what says how bad it is."""
    names = _install(tmp_path)
    for folder, stem in names:
        _navmesh(tmp_path, HM3D_V2, folder, stem).unlink()
    report = _check(tmp_path)
    assert report["missing_count"] == 36 and len(report["missing"]) == 20
    assert report["missing"] == sorted(report["missing"])
    assert report["scenes_complete"] == 0 and not report["scenes_ready"]


def test_v2_scenes_left_under_the_v1_name_are_reported_and_not_silently_used(tmp_path):
    """This is the failure that otherwise runs v2 episodes on v0.1 geometry with no error."""
    names = _write_episodes(tmp_path, HM3D_V2, _published_sizes(HM3D_V2))
    _write_scenes(tmp_path, HM3D_V2, names, release_dir=scenes_dir(tmp_path) / "hm3d")
    report = _check(tmp_path)
    assert report["episodes_ready"] and not report["scenes_ready"]
    problem = " ".join(report["problems"])
    assert "hm3d_v0.2" in problem and "symlink" in problem, problem
    assert "release_dir_resolves_to" not in report


def test_the_release_name_records_the_directory_it_actually_resolves_to(tmp_path):
    """Nothing on disk distinguishes v0.1 geometry from v0.2 behind the expected name,
    so the one defence is reporting where the name points."""
    names = _write_episodes(tmp_path, HM3D_V2, _published_sizes(HM3D_V2))
    real = _write_scenes(tmp_path, HM3D_V2, names,
                         release_dir=tmp_path / "versioned_data" / "hm3d-0.2" / "hm3d")
    link = scenes_dir(tmp_path) / HM3D_V2.scene_root_name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)
    report = _check(tmp_path)
    assert report["scenes_ready"]
    assert report["release_dir_resolves_to"] == str(real.resolve())


def test_a_sample_checks_that_many_scenes_and_never_declares_the_split_ready(tmp_path):
    """A quick look proves nothing about the 33 scenes it did not open."""
    _install(tmp_path)
    report = _check(tmp_path, sample=3)
    assert report["scenes_checked"] == 3 and report["missing_count"] == 0
    assert not report["scenes_ready"] and report["episodes_ready"]


def test_the_cli_composes_the_documented_layout_from_one_data_root(tmp_path):
    """--data-root is the whole interface; the two halves have to land where the
    instructions put them."""
    assert episode_dir(tmp_path, HM3D_V2) == tmp_path / "objectnav" / "hm3d" / "v2" / "val"
    assert episode_dir(tmp_path, HM3D_V1) == tmp_path / "objectnav" / "hm3d" / "v1" / "val"
    assert scenes_dir(tmp_path) == tmp_path / "scene_datasets"
    assert episode_dir("~/data", HM3D_V1) == Path.home() / "data/objectnav/hm3d/v1/val"
    assert scenes_dir("~/data") == Path.home() / "data" / "scene_datasets"


def test_the_scene_half_is_instructed_with_placeholders_and_never_a_credential(tmp_path):
    """The token belongs to whoever accepted the licence; this process must not see one."""
    root = tmp_path / "datasets"
    text = download_instructions(HM3D_V2, root)
    assert LICENCE_URL in text and TOKEN_URL in text
    commands = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert re.findall(r"--(?:username|password)\s+(\S+)", commands) == [
        "<token-id>", "<token-secret>"]
    assert not root.exists() and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("protocol,other", [(HM3D_V1, HM3D_V2), (HM3D_V2, HM3D_V1)])
def test_the_instructions_name_this_versions_archive_uid_and_link_name(tmp_path, protocol, other):
    """Pasting the other version's download is exactly how the two releases get crossed."""
    text = download_instructions(protocol, tmp_path)
    assert EPISODE_ARCHIVES[protocol.dataset_version] in text
    assert EPISODE_ARCHIVES[other.dataset_version] not in text
    assert "--uids %s" % protocol.scene_download_uid in text
    assert other.scene_download_uid not in text
    assert "scene_datasets/%s" % protocol.scene_root_name in text


def test_printing_instructions_without_check_succeeds_and_writes_nothing(tmp_path):
    """A person with no data at all runs this first; it must not fail and must not fetch."""
    root = tmp_path / "datasets"
    assert main(["--version", "v2", "--data-root", str(root)]) == 0
    assert not root.exists()


def test_check_prints_a_parseable_report_and_exits_non_zero_while_incomplete(tmp_path, capsys):
    """--check is what a setup script branches on, so the status and the stdout both matter."""
    names = _install(tmp_path)
    _navmesh(tmp_path, HM3D_V2, *names[0]).unlink()
    assert main(["--version", "v2", "--data-root", str(tmp_path), "--check"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["missing_count"] == 1 and not report["scenes_ready"]

    _write_scenes(tmp_path, HM3D_V2, names)
    assert main(["--version", "v2", "--data-root", str(tmp_path), "--check"]) == 0
    assert json.loads(capsys.readouterr().out)["scenes_ready"]


def test_checking_and_instructing_never_reach_the_network_or_spawn_a_downloader(
        tmp_path, monkeypatch):
    """This module advises; it never provisions. A fetch here would run unattended
    against a licensed 5 GB dataset."""
    def forbidden(*args, **kwargs):
        raise AssertionError("assets.py must never fetch: %r" % (args,))

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    _install(tmp_path)
    assert main(["--version", "v2", "--data-root", str(tmp_path), "--instructions",
                 "--check"]) == 0
