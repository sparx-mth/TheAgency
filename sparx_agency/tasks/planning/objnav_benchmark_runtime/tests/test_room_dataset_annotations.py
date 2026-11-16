import json
import struct

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.annotations import (
    instance_centers, read_semantic_table, read_vertices)

SEMANTIC_TXT = 'HM3D Semantic Annotations\n1,6C48EA,"ceiling",0\n2,64FB8F,"wall",0\n7,7E4830,"picture frame",3\n'


def linear_16bit(srgb_bytes):
    encoded = np.asarray(srgb_bytes) / 255.0
    return np.rint(np.power((encoded + 0.055) / 1.055, 2.4) * 65535).astype("<u2")


def glb_with_one_triangle(positions, colors):
    binary = positions.astype("<f4").tobytes() + linear_16bit(colors).tobytes()
    header = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}],
              "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
                              {"buffer": 0, "byteOffset": positions.astype("<f4").nbytes, "byteLength": colors.size * 2}],
              "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                            {"bufferView": 1, "componentType": 5123, "normalized": True, "count": 3, "type": "VEC3"}],
              "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}}]}]}
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    body = struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def test_the_table_maps_a_colour_to_instance_name_and_region(tmp_path):
    path = tmp_path / "s.semantic.txt"
    path.write_text(SEMANTIC_TXT)
    assert read_semantic_table(path)["7E4830"] == {"instance": 7, "name": "picture frame", "region": 3}


def test_the_vertices_come_out_in_habitat_axes_with_their_colours_from_the_glb(tmp_path):
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    colors = np.array([[0x7E, 0x48, 0x30]] * 3)
    path = tmp_path / "s.glb"
    path.write_bytes(glb_with_one_triangle(positions, colors))
    found_positions, found_colors = read_vertices(path)
    assert np.allclose(found_positions, [[0, 0, 0], [1, 0, 0], [0, 0, -1]]) and (found_colors == colors).all()


def test_an_instance_centre_is_the_mean_of_its_vertices_with_its_lowest_point(tmp_path):
    path = tmp_path / "s.semantic.txt"
    path.write_text(SEMANTIC_TXT)
    positions = np.array([[0, 1, 0], [2, 3, 0], [9, 9, 9]], dtype=float)
    colors = np.array([[0x7E, 0x48, 0x30], [0x7E, 0x48, 0x30], [1, 2, 3]])
    centers = instance_centers(positions, colors, read_semantic_table(path))
    assert centers == [{"instance": 7, "name": "picture frame", "region": 3, "center": (1.0, 2.0, 0.0), "lowest_y": 1.0}]
