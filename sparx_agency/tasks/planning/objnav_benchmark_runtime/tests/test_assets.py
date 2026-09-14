"""Safe selected-scene import from synthetic ZIP fixtures, never real scene data."""
from __future__ import annotations

from pathlib import Path
import stat
import struct
import threading
import zipfile

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.assets import import_scene_zip


def make_zip(path, names=None, glb=None):
    if glb is None:
        glb = struct.pack("<4sII", b"glTF", 2, 16) + b"test"
    with zipfile.ZipFile(path, "w") as bundle:
        for name in names or ("release/scenes/Collierville.glb", "release/scenes/Collierville.navmesh"):
            bundle.writestr(name, glb if name.endswith(".glb") else b"test-navmesh")
        bundle.writestr("other/Corozal.glb", b"not selected")
        bundle.writestr("../unrelated.txt", b"not extracted")
    return path


def test_only_selected_scene_is_imported(tmp_path):
    archive = make_zip(tmp_path / "release.zip")
    destination = import_scene_zip(archive, "Collierville", tmp_path / "scenes")
    assert {p.name for p in destination.iterdir()} == {"Collierville.glb", "Collierville.navmesh"}
    assert (destination / "Collierville.navmesh").read_bytes() == b"test-navmesh"
    assert not (tmp_path / "unrelated.txt").exists()


def test_existing_scene_is_not_overwritten(tmp_path):
    archive = make_zip(tmp_path / "release.zip")
    output = tmp_path / "scenes"
    output.mkdir()
    (output / "Collierville.glb").write_bytes(b"keep existing asset")
    with pytest.raises(FileExistsError):
        import_scene_zip(archive, "Collierville", output)
    assert (output / "Collierville.glb").read_bytes() == b"keep existing asset"
    assert not (output / "Collierville.navmesh").exists()


@pytest.mark.parametrize("names", [
    ("Collierville.glb",),
    ("Collierville.glb.json.gz", "Collierville.navmesh"),
    ("one/Collierville.glb", "two/Collierville.glb", "Collierville.navmesh"),
    ("../Collierville.glb", "Collierville.navmesh"),
    ("/Collierville.glb", "Collierville.navmesh"),
])
def test_missing_ambiguous_or_unsafe_members_are_refused(tmp_path, names):
    archive = make_zip(tmp_path / "release.zip", names=names)
    with pytest.raises(ValueError):
        import_scene_zip(archive, "Collierville", tmp_path / "scenes")
    assert not (tmp_path / "scenes" / "Collierville.glb").exists()


def test_corrupt_glb_is_not_installed(tmp_path):
    archive = make_zip(tmp_path / "release.zip", glb=b"this is HTML, not a scene")
    with pytest.raises(ValueError, match="GLB header"):
        import_scene_zip(archive, "Collierville", tmp_path / "scenes")
    assert list((tmp_path / "scenes").iterdir()) == []


def test_symlink_member_is_refused(tmp_path):
    archive = tmp_path / "release.zip"
    member = zipfile.ZipInfo("Collierville.glb")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, "../another-scene.glb")
        bundle.writestr("Collierville.navmesh", b"test")
    with pytest.raises(ValueError, match="Unsafe"):
        import_scene_zip(archive, "Collierville", tmp_path / "scenes")


def test_cancelled_import_leaves_no_partial_assets(tmp_path):
    archive = make_zip(tmp_path / "release.zip")
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        import_scene_zip(archive, "Collierville", tmp_path / "scenes", cancelled)
    assert list((tmp_path / "scenes").iterdir()) == []


