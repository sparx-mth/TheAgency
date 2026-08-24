"""Tests for mesh loading, the COLLADA unit trap, and the cache."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from sparx_agency.tasks.mapping.gazebo_world_occupancy import mesh_cache


def _unit_cube():
    """A 1 m cube as a trimesh, centred on the origin."""
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"


def _write_obj(path):
    """Export a 1 m cube as a Wavefront OBJ."""
    _unit_cube().export(str(path))
    return path


def _write_centimetre_dae(path):
    """Export a 1 m cube written in centimetres, declaring meter="0.01"."""
    mesh = _unit_cube()
    mesh.apply_scale(100.0)
    mesh.export(str(path))

    ET.register_namespace("", COLLADA_NS)
    tree = ET.parse(str(path))
    asset = tree.getroot().find("{%s}asset" % COLLADA_NS)
    unit = asset.find("{%s}unit" % COLLADA_NS)
    if unit is None:
        unit = ET.SubElement(asset, "{%s}unit" % COLLADA_NS)
    unit.set("meter", "0.01")
    unit.set("name", "centimeter")
    tree.write(str(path), xml_declaration=True, encoding="utf-8")
    return path


def test_collada_unit_is_read_from_the_file(tmp_path):
    path = _write_centimetre_dae(tmp_path / "cube.dae")
    assert mesh_cache.collada_unit_metres(path) == pytest.approx(0.01)


def test_non_collada_files_have_no_unit(tmp_path):
    path = _write_obj(tmp_path / "cube.obj")
    assert mesh_cache.collada_unit_metres(path) == 1.0


def test_a_centimetre_dae_loads_in_metres(tmp_path):
    """The silent 100x error: trimesh applies the axis flip but not the unit."""
    mesh_cache.clear_cache()
    path = _write_centimetre_dae(tmp_path / "cube.dae")

    raw = trimesh.load(str(path), force="mesh", process=False)
    assert np.ptp(np.asarray(raw.vertices), axis=0).max() == pytest.approx(100.0)

    vertices, faces = mesh_cache.load_mesh(path)
    assert faces.shape[1] == 3
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [1.0, 1.0, 1.0], atol=1e-6)


def test_sdf_scale_is_applied_on_top_of_the_file_unit(tmp_path):
    mesh_cache.clear_cache()
    path = _write_centimetre_dae(tmp_path / "cube.dae")
    vertices, _faces = mesh_cache.load_mesh(path, (2.0, 3.0, 4.0))
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [2.0, 3.0, 4.0], atol=1e-6)


def test_obj_takes_its_scale_only_from_the_sdf(tmp_path):
    mesh_cache.clear_cache()
    path = _write_obj(tmp_path / "cube.obj")
    vertices, _faces = mesh_cache.load_mesh(path, (0.5, 0.5, 0.5))
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [0.5, 0.5, 0.5], atol=1e-6)


def test_the_same_model_is_loaded_once_per_scale(tmp_path):
    """The chair that appears sixteen times must be parsed once."""
    mesh_cache.clear_cache()
    path = _write_obj(tmp_path / "cube.obj")
    for _ in range(5):
        mesh_cache.load_mesh(path, (1.0, 1.0, 1.0))
    mesh_cache.load_mesh(path, (2.0, 2.0, 2.0))
    info = mesh_cache.cache_info()
    assert info.misses == 2
    assert info.hits == 4


def test_cached_arrays_are_read_only(tmp_path):
    """They are shared between instances; a writable copy would corrupt them all."""
    mesh_cache.clear_cache()
    path = _write_obj(tmp_path / "cube.obj")
    vertices, faces = mesh_cache.load_mesh(path)
    assert not vertices.flags.writeable
    assert not faces.flags.writeable


def test_missing_mesh_raises(tmp_path):
    mesh_cache.clear_cache()
    with pytest.raises(FileNotFoundError):
        mesh_cache.load_mesh(tmp_path / "absent.obj")


def test_bad_scale_raises(tmp_path):
    with pytest.raises(ValueError):
        mesh_cache.load_mesh(tmp_path / "absent.obj", (1.0, 1.0))


def _write_dae(path, asset_body):
    """Write a minimal COLLADA document with the given ``<asset>`` body."""
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<COLLADA xmlns="%s" version="1.4.1">\n'
        "  <asset>%s</asset>\n"
        "</COLLADA>\n" % (COLLADA_NS, asset_body)
    )
    return path


def test_a_dae_declaring_no_unit_is_metres(tmp_path):
    """COLLADA's own default, and the only silence that is safe here."""
    path = _write_dae(tmp_path / "plain.dae", "<up_axis>Z_UP</up_axis>")
    assert mesh_cache.collada_unit_metres(path) == 1.0


def test_unparseable_collada_raises(tmp_path):
    """Guessing 1.0 for a .dae we cannot read is the 100x error itself."""
    path = tmp_path / "broken.dae"
    path.write_text("<COLLADA><asset>")
    with pytest.raises(ValueError):
        mesh_cache.collada_unit_metres(path)


def test_a_unit_that_is_not_a_number_raises(tmp_path):
    path = _write_dae(tmp_path / "text.dae", '<unit meter="one" name="metre"/>')
    with pytest.raises(ValueError):
        mesh_cache.collada_unit_metres(path)


def test_a_non_positive_unit_raises(tmp_path):
    path = _write_dae(tmp_path / "zero.dae", '<unit meter="0" name="none"/>')
    with pytest.raises(ValueError):
        mesh_cache.collada_unit_metres(path)


def test_loading_a_mesh_with_an_unreadable_unit_raises(tmp_path):
    """The failure must reach the caller, not be swallowed by the loader."""
    mesh_cache.clear_cache()
    path = tmp_path / "cube.dae"
    _write_centimetre_dae(path)
    text = path.read_text().replace('meter="0.01"', 'meter="-1"')
    path.write_text(text)
    with pytest.raises(ValueError):
        mesh_cache.load_mesh(path)
