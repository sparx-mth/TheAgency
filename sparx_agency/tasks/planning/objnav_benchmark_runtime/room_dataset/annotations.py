"""Where each annotated HM3D instance is and which region it is in, read from the scene's .semantic.txt and .semantic.glb with numpy only."""
import csv
import json
import struct

import numpy as np

COMPONENT_DTYPE = {5121: "<u1", 5123: "<u2", 5126: "<f4"}
COMPONENT_MAX = {5121: 255.0, 5123: 65535.0, 5126: 1.0}
COMPONENTS = {"VEC3": 3, "VEC4": 4}


def read_semantic_table(path):
    """``{colour hex: {instance, name, region}}`` from the semantic.txt; the colour is how the mesh marks the instance."""
    with open(path) as handle:
        rows = list(csv.reader(handle.read().splitlines()[1:]))
    return {colour.upper(): {"instance": int(instance), "name": name, "region": int(region)} for instance, colour, name, region in rows}


def _glb_chunks(raw):
    _, _, length = struct.unpack("<4sII", raw[:12])
    chunks, offset = {}, 12
    while offset < length:
        size, kind = struct.unpack("<I4s", raw[offset:offset + 8])
        chunks[kind] = raw[offset + 8:offset + 8 + size]
        offset += 8 + size
    return chunks


def _accessor(document, binary, index):
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype, width = COMPONENT_DTYPE[accessor["componentType"]], COMPONENTS[accessor["type"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    itemsize = np.dtype(dtype).itemsize
    stride = view.get("byteStride", width * itemsize)
    rows = np.ndarray((accessor["count"], width), dtype=dtype, buffer=binary, offset=start, strides=(stride, itemsize))
    return rows, accessor["componentType"]


def _linear_to_srgb(linear):
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)


def _habitat_axes(z_up):
    return np.stack([z_up[:, 0], z_up[:, 2], -z_up[:, 1]], axis=1)


def read_vertices(path):
    """Positions (N, 3) in habitat axes (the file is z-up) and the colours (N, 3) as the 0-255 sRGB ints of the semantic.txt (glTF stores them linear)."""
    chunks = _glb_chunks(open(path, "rb").read())
    document, binary = json.loads(chunks[b"JSON"]), chunks[b"BIN\x00"]
    positions, colours = [], []
    for mesh in document["meshes"]:
        for primitive in mesh["primitives"]:
            rows, _ = _accessor(document, binary, primitive["attributes"]["POSITION"])
            paint, component = _accessor(document, binary, primitive["attributes"]["COLOR_0"])
            positions.append(_habitat_axes(np.array(rows, dtype=float)))
            colours.append(np.rint(_linear_to_srgb(np.array(paint[:, :3], dtype=float) / COMPONENT_MAX[component]) * 255).astype(int))
    return np.concatenate(positions), np.concatenate(colours)


def instance_centers(positions, colours, table):
    """One row per annotated instance seen in the mesh: name, region, mean vertex position and its lowest vertex height."""
    keys = (colours[:, 0] << 16) | (colours[:, 1] << 8) | colours[:, 2]
    unique, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.stack([np.bincount(inverse, weights=positions[:, axis]) for axis in range(3)], axis=1)
    lowest = np.full(len(unique), np.inf)
    np.minimum.at(lowest, inverse, positions[:, 1])
    rows = []
    for index, key in enumerate(unique):
        entry = table.get("%06X" % key)
        if entry:
            rows.append(dict(entry, center=tuple(float(v) for v in sums[index] / counts[index]), lowest_y=float(lowest[index])))
    return sorted(rows, key=lambda r: r["instance"])
