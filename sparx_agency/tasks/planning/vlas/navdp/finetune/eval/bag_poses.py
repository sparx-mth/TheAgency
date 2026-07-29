"""Recover per-frame camera poses from a recording's source rosbag.

``datasets/bag_extract.py`` reads only ``/xtend/rgb``, ``/xtend/depth_m`` and
``/xtend/camera_info``, so the extracted ``flight_dataset`` carries no geometry.
The bags do however publish ``/xtend/april_tag_pose`` (``PoseStamped``, ~9 Hz),
which ``rgbd_full_pipeline.launch.py`` uses as the ``map -> xtend_camera``
transform. That is exactly the ``world_T_cam`` needed to fuse several depth
frames into one map, so this module reads it back out.

The bag is decoded with :mod:`sqlite3` and a small hand-rolled CDR reader rather
than ``rosbag2_py``/``rclpy``: the comparison runs in the ``navdp`` conda env,
which has no ROS. ``PoseStamped`` has a fixed layout, so this stays short.

Convention: the published pose is ``world_T_ros`` -- the camera pose in the world
frame under the ROS body convention (X forward, Y left, Z up), because the
publisher applies ``world_T_cam @ cv_to_ros``. :func:`load_frame_poses` undoes
that so the returned matrices are ``world_T_cam`` in the **OpenCV optical**
convention (X right, Y down, Z forward), which is what
``core.mapping.costmap.depth_to_grid.update_grid_from_depth`` expects.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Tuple

import numpy as np

POSE_TOPIC = "/xtend/april_tag_pose"
DEPTH_TOPIC = "/xtend/depth_m"

# Inverse of the publisher's cv_to_ros: maps ROS body axes back to optical axes.
_CV_TO_ROS = np.array([[0.0, -1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0, 0.0],
                       [1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
_ROS_TO_CV = np.linalg.inv(_CV_TO_ROS)


def _header_stamp_ns(blob: bytes) -> float:
    """Nanosecond header stamp of any stamped message (CDR: sec, nanosec first)."""
    sec = int.from_bytes(blob[4:8], "little", signed=True)
    nsec = int.from_bytes(blob[8:12], "little")
    return sec * 1e9 + nsec


def _pose_fields(blob: bytes) -> np.ndarray:
    """The 7 pose doubles ``[x, y, z, qx, qy, qz, qw]`` of a ``PoseStamped``.

    Walks the CDR body explicitly instead of assuming a fixed offset, so a
    ``frame_id`` other than ``"world"`` still parses. Alignment is measured from
    the start of the body (i.e. after the 4-byte encapsulation header).
    """
    str_len = int.from_bytes(blob[12:16], "little")
    pos = 12 + 4 + str_len              # body-relative offset past frame_id
    pos = (pos + 7) // 8 * 8            # float64 needs 8-byte alignment
    offset = 4 + pos
    fields = np.frombuffer(blob[offset:offset + 56], dtype="<f8")
    if fields.size != 7:
        raise ValueError(f"truncated PoseStamped: got {fields.size} doubles")
    return fields


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion -> 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:
        raise ValueError("degenerate quaternion")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _read_topic(db: Path, topic: str) -> list:
    """All ``(bag_ns, blob)`` rows of one topic, ordered by bag time."""
    con = sqlite3.connect(str(db))
    try:
        row = con.execute("select id from topics where name=?", (topic,)).fetchone()
        if row is None:
            raise KeyError(f"{db.name} has no topic {topic}")
        return [(t, bytes(d)) for t, d in con.execute(
            "select timestamp, data from messages where topic_id=? order by timestamp",
            (row[0],))]
    finally:
        con.close()


def find_bag(bag_root: Path, rec: str) -> Path:
    """The ``.db3`` file of recording ``rec`` under ``bag_root``."""
    candidates = sorted((bag_root / rec).glob("*.db3"))
    if not candidates:
        raise FileNotFoundError(f"no .db3 for recording {rec!r} under {bag_root}")
    return candidates[0]


def load_frame_poses(bag_root: Path, rec: str,
                     max_gap_s: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Per-depth-frame ``world_T_cam`` matrices in the OpenCV optical convention.

    Poses are interpolated onto the depth header stamps: translation linearly,
    rotation by nearest neighbour (adequate here -- the pose stream runs ~10x
    faster than depth, and on the XTEND recordings the two share header stamps
    exactly, so interpolation is usually a no-op).

    Args:
        bag_root: directory holding one subdirectory per recording.
        rec: recording name, e.g. ``"walk_into"``.
        max_gap_s: a depth frame further than this from any pose is rejected.

    Returns:
        ``(frame_idx, world_T_cam)`` where ``frame_idx`` is ``(M,) int`` indices
        into the recording's depth frames (0-based, matching ``bag_extract``'s
        ordering of ``/xtend/depth_m``) and ``world_T_cam`` is ``(M, 4, 4)``.
        Frames without usable pose support are dropped, so ``M <= n_frames``.

    Raises:
        KeyError: the bag has no pose topic (the recording predates AprilTag
            localization) -- there is no way to build the fused judge map.
    """
    db = find_bag(bag_root, rec)
    pose_rows = _read_topic(db, POSE_TOPIC)
    depth_rows = _read_topic(db, DEPTH_TOPIC)

    pose_t = np.array([_header_stamp_ns(b) for _, b in pose_rows])
    pose_v = np.array([_pose_fields(b) for _, b in pose_rows])
    depth_t = np.array([_header_stamp_ns(b) for _, b in depth_rows])

    order = np.argsort(pose_t)
    pose_t, pose_v = pose_t[order], pose_v[order]

    nearest = np.abs(depth_t[:, None] - pose_t[None, :]).argmin(axis=1)
    gap_s = np.abs(depth_t - pose_t[nearest]) / 1e9
    keep = np.flatnonzero(gap_s <= max_gap_s)
    if keep.size == 0:
        raise ValueError(f"{rec}: no depth frame within {max_gap_s}s of a pose")

    mats = np.zeros((keep.size, 4, 4))
    for out_i, frame_i in enumerate(keep):
        xyz = np.array([np.interp(depth_t[frame_i], pose_t, pose_v[:, k]) for k in range(3)])
        qx, qy, qz, qw = pose_v[nearest[frame_i], 3:7]
        world_T_ros = np.eye(4)
        world_T_ros[:3, :3] = _quat_to_matrix(qx, qy, qz, qw)
        world_T_ros[:3, 3] = xyz
        mats[out_i] = world_T_ros @ _ROS_TO_CV
    return keep.astype(int), mats
