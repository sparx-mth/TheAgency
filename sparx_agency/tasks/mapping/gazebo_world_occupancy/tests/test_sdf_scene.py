"""Tests for flattening an SDF world into world-placed geometry."""
from __future__ import annotations

import math
import textwrap
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sparx_agency.tasks.mapping.gazebo_world_occupancy.sdf_elements import element_pose
from sparx_agency.tasks.mapping.gazebo_world_occupancy.sdf_scene import (
    ModelNotFoundError,
    load_scene,
    pose_to_matrix,
    resolve_model_uri,
)


def _write(path, text):
    """Write dedented XML to ``path``, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n")
    return path


def _model_dir(root, name, body):
    """Create ``root/name/model.sdf`` holding ``body`` inside a <model>."""
    _write(
        root / name / "model.sdf",
        """
        <?xml version="1.0"?>
        <sdf version="1.6">
          <model name="%s">
        %s
          </model>
        </sdf>
        """ % (name, body),
    )
    return root / name


def test_pose_to_matrix_is_extrinsic_xyz():
    """SDF rpy is Rz(yaw) @ Ry(pitch) @ Rx(roll), applied to a translated point."""
    transform = pose_to_matrix([1.0, 2.0, 3.0, 0.0, 0.0, math.pi / 2.0])
    rotated = transform.dot(np.array([1.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(rotated[:3], [1.0, 3.0, 3.0], atol=1e-12)


def test_pose_to_matrix_composes_roll_pitch_yaw_in_order():
    transform = pose_to_matrix([0.0, 0.0, 0.0, math.pi / 2.0, 0.0, math.pi / 2.0])
    # roll +90 sends +y to +z; yaw +90 then leaves +z alone.
    rotated = transform.dot(np.array([0.0, 1.0, 0.0, 1.0]))
    np.testing.assert_allclose(rotated[:3], [0.0, 0.0, 1.0], atol=1e-12)


def test_pose_with_wrong_component_count_raises():
    with pytest.raises(ValueError):
        pose_to_matrix([1.0, 2.0, 3.0])


def test_resolve_model_uri_prefers_the_first_search_path(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    (first / "Chair").mkdir(parents=True)
    (second / "Chair").mkdir(parents=True)
    assert resolve_model_uri("model://Chair", [first, second]) == first / "Chair"
    assert resolve_model_uri("model://Chair", [second, first]) == second / "Chair"


def test_resolve_model_uri_raises_when_absent(tmp_path):
    with pytest.raises(ModelNotFoundError):
        resolve_model_uri("model://Nowhere", [tmp_path])


def test_poses_compose_through_include_link_and_collision(tmp_path):
    """The transform is include x link x collision, each in its parent's frame."""
    models = tmp_path / "models"
    _model_dir(
        models,
        "Widget",
        """
        <link name="body">
          <pose>0 1 0 0 0 0</pose>
          <collision name="collision">
            <pose>0 0 2 0 0 0</pose>
            <geometry><box><size>1 1 1</size></box></geometry>
          </collision>
        </link>
        """,
    )
    world = _write(
        tmp_path / "worlds" / "t.world",
        """
        <sdf version="1.6">
          <world name="w">
            <include>
              <uri>model://Widget</uri>
              <pose>10 0 0 0 0 1.5707963267948966</pose>
            </include>
          </world>
        </sdf>
        """,
    )

    scene = load_scene(world, [models])
    assert len(scene.instances) == 1
    # The link's +y offset is rotated by the include's +90 deg yaw to -x.
    np.testing.assert_allclose(
        scene.instances[0].transform[:3, 3], [9.0, 0.0, 2.0], atol=1e-9
    )


def test_world_level_model_wrapper_pose_is_composed(tmp_path):
    """A <model> wrapping an <include> contributes its own pose too."""
    models = tmp_path / "models"
    _model_dir(
        models,
        "Widget",
        """
        <link name="body">
          <collision name="collision">
            <geometry><box><size>1 1 1</size></box></geometry>
          </collision>
        </link>
        """,
    )
    world = _write(
        tmp_path / "worlds" / "t.world",
        """
        <sdf version="1.6">
          <world name="w">
            <model name="outer">
              <pose>0 0 5 0 0 0</pose>
              <include>
                <uri>model://Widget</uri>
                <pose>3 0 0 0 0 0</pose>
              </include>
            </model>
          </world>
        </sdf>
        """,
    )
    scene = load_scene(world, [models])
    np.testing.assert_allclose(
        scene.instances[0].transform[:3, 3], [3.0, 0.0, 5.0], atol=1e-9
    )


def test_collision_is_preferred_and_visual_is_the_fallback(tmp_path):
    models = tmp_path / "models"
    _model_dir(
        models,
        "Both",
        """
        <link name="body">
          <visual name="visual">
            <geometry><box><size>9 9 9</size></box></geometry>
          </visual>
          <collision name="collision">
            <geometry><box><size>1 1 1</size></box></geometry>
          </collision>
        </link>
        """,
    )
    _model_dir(
        models,
        "VisualOnly",
        """
        <link name="body">
          <visual name="visual">
            <geometry><box><size>2 2 2</size></box></geometry>
          </visual>
        </link>
        """,
    )
    world = _write(
        tmp_path / "worlds" / "t.world",
        """
        <sdf version="1.6">
          <world name="w">
            <include><uri>model://Both</uri></include>
            <include><uri>model://VisualOnly</uri></include>
          </world>
        </sdf>
        """,
    )
    scene = load_scene(world, [models])
    by_source = {i.source: i for i in scene.instances}
    assert len(scene.instances) == 2
    np.testing.assert_allclose(by_source["collision"].size, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(by_source["visual"].size, [2.0, 2.0, 2.0])


def test_models_matching_a_skip_substring_are_excluded(tmp_path):
    """The robot must not be baked into its own map."""
    models = tmp_path / "models"
    for name in ("sjtu_drone", "Table"):
        _model_dir(
            models,
            name,
            """
            <link name="body">
              <collision name="collision">
                <geometry><box><size>1 1 1</size></box></geometry>
              </collision>
            </link>
            """,
        )
    world = _write(
        tmp_path / "worlds" / "t.world",
        """
        <sdf version="1.6">
          <world name="w">
            <include><uri>model://sjtu_drone</uri></include>
            <include><uri>model://Table</uri></include>
          </world>
        </sdf>
        """,
    )
    scene = load_scene(world, [models], skip_substrings=("drone",))
    assert len(scene.instances) == 1
    assert scene.skipped_models == ["sjtu_drone"]


def test_unresolved_models_are_recorded_not_raised(tmp_path):
    """Gazebo built-ins are absent on most machines; that must be visible."""
    world = _write(
        tmp_path / "worlds" / "t.world",
        """
        <sdf version="1.6">
          <world name="w">
            <include><uri>model://sun</uri></include>
          </world>
        </sdf>
        """,
    )
    scene = load_scene(world, [tmp_path / "models"])
    assert scene.instances == []
    assert scene.missing_models == ["model://sun"]


def test_infinite_ground_plane_is_ignored(tmp_path):
    models = tmp_path / "models"
    _model_dir(
        models,
        "Ground",
        """
        <link name="body">
          <collision name="collision">
            <geometry><plane><normal>0 0 1</normal></plane></geometry>
          </collision>
        </link>
        """,
    )
    world = _write(
        tmp_path / "worlds" / "t.world",
        """
        <sdf version="1.6">
          <world name="w">
            <include><uri>model://Ground</uri></include>
          </world>
        </sdf>
        """,
    )
    scene = load_scene(world, [models])
    assert scene.instances == []
    assert scene.ignored_geometry == [("Ground", "plane")]


def test_mesh_scale_and_uri_are_carried_through(tmp_path):
    models = tmp_path / "models"
    model_dir = _model_dir(
        models,
        "Chair",
        """
        <link name="body">
          <collision name="collision">
            <geometry>
              <mesh>
                <uri>model://Chair/meshes/Chair.obj</uri>
                <scale>0.01 0.01 0.01</scale>
              </mesh>
            </geometry>
          </collision>
        </link>
        """,
    )
    (model_dir / "meshes").mkdir(parents=True)
    (model_dir / "meshes" / "Chair.obj").write_text("")
    world = _write(
        tmp_path / "worlds" / "t.world",
        """
        <sdf version="1.6">
          <world name="w">
            <include><uri>model://Chair</uri></include>
          </world>
        </sdf>
        """,
    )
    scene = load_scene(world, [models])
    instance = scene.instances[0]
    assert instance.kind == "mesh"
    assert instance.mesh_path == model_dir / "meshes" / "Chair.obj"
    np.testing.assert_allclose(instance.scale, [0.01, 0.01, 0.01])


def test_missing_world_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scene(tmp_path / "nope.world", [tmp_path])


def _pose_element(attributes=""):
    """A <link> carrying one <pose>, with the given attribute text."""
    return ET.fromstring(
        "<link><pose%s>1 2 3 0 0 0</pose></link>" % (attributes,)
    )


def test_a_plain_pose_is_read():
    transform = element_pose(_pose_element())
    np.testing.assert_allclose(transform[:3, 3], [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "attributes",
    [
        ' degrees="true"',
        ' rotation_format="quat_xyzw"',
        ' relative_to="base_link"',
    ],
)
def test_a_pose_this_reader_would_misread_raises(attributes):
    """Each of these leaves six numbers meaning something else entirely."""
    with pytest.raises(ValueError):
        element_pose(_pose_element(attributes))


def test_an_empty_pose_with_a_frame_attribute_still_raises():
    """relative_to changes the frame even when the numbers are absent."""
    with pytest.raises(ValueError):
        element_pose(ET.fromstring('<link><pose relative_to="a"/></link>'))


def test_a_degrees_pose_in_a_world_is_refused_rather_than_flown(tmp_path):
    """A silent 57x on every angle would place the whole model elsewhere."""
    models = tmp_path / "models"
    _model_dir(
        models,
        "Widget",
        """
        <link name="body">
          <pose degrees="true">0 0 0 0 0 90</pose>
          <collision name="collision">
            <geometry><box><size>1 1 1</size></box></geometry>
          </collision>
        </link>
        """,
    )
    world = _write(
        tmp_path / "worlds" / "t.world",
        """
        <sdf version="1.6">
          <world name="w">
            <include><uri>model://Widget</uri></include>
          </world>
        </sdf>
        """,
    )
    with pytest.raises(ValueError):
        load_scene(world, [models])


def test_an_unresolved_mesh_is_not_an_unresolved_model(tmp_path):
    """A missing mesh file is a missing wall; a missing include may be the sun."""
    models = tmp_path / "models"
    _model_dir(
        models,
        "Widget",
        """
        <link name="body">
          <collision name="collision">
            <geometry><mesh><uri>model://Widget/meshes/gone.dae</uri></mesh></geometry>
          </collision>
        </link>
        """,
    )
    world = _write(
        tmp_path / "worlds" / "t.world",
        """
        <sdf version="1.6">
          <world name="w">
            <include><uri>model://Widget</uri></include>
            <include><uri>model://sun</uri></include>
          </world>
        </sdf>
        """,
    )
    scene = load_scene(world, [models])
    assert scene.missing_meshes == ["model://Widget/meshes/gone.dae"]
    assert scene.missing_models == ["model://sun"]
