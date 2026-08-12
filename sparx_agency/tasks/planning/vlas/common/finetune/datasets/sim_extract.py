"""Export a simulated flight into the on-disk layout ``datasets/recording.py`` reads.

Counterpart to :mod:`bag_extract` for simulator-sourced flights (Isaac Sim /
Pegasus Simulator, see ``sparx_agency/robots/PEGASUS/``) instead of a real rosbag.
This module has **no simulator dependency** — it only writes numpy arrays, images
and JSON, so it can run and be unit-tested outside Isaac Sim. The sim-specific
part (driving the vehicle, pulling RGB/depth frames and ground-truth pose out of
the running simulation) lives in ``robots/PEGASUS/adapters/`` and is handed to
:func:`export_flight` as a plain iterable of :class:`SimFrame`.

Unlike a rosbag extraction, a simulated flight has **exact ground-truth pose**
for every frame, and no localization stack in between to lose any of it. So this
writes the full 6-DoF state, not just the ``(x, y, yaw)`` the legacy schema
carries — see :data:`POSE_COLUMNS`. The first four columns are unchanged, so
every existing reader keeps working untouched.

Output (the ``recording.py`` schema)::

    <out_dir>/
      rgb/000000.jpg         colour frame (optional, only if a frame supplies one)
      depth/000000.png       (H, W) uint16 millimetres  -- or .npy float32 metres
      intrinsics.json        {width,height,fx,fy,cx,cy}
      meta.json              {rate_hz, camera_height_m, pitch_deg, frames, ...}
      poses.npy              (N, 21) float32, see POSE_COLUMNS
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from sparx_agency.core.common.types import Intrinsics

POSE_COLUMNS = (
    "t", "x", "y", "yaw",              # 0-3   the legacy schema; every reader uses these
    "z",                               # 4     world up, metres
    "qx", "qy", "qz", "qw",            # 5-8   body FLU -> world ENU rotation
    "vx", "vy", "vz",                  # 9-11  world-frame linear velocity, m/s
    "wx", "wy", "wz",                  # 12-14 body-frame angular velocity, rad/s
    "ax", "ay", "az",                  # 15-17 world-frame linear acceleration, m/s^2
    "ux", "uy", "uz",                  # 18-20 body-frame linear velocity, m/s
)
"""Column layout of ``poses.npy``.

Columns 0-3 are the original ``[t, x, y, yaw]`` schema and are load-bearing:
:class:`~...datasets.recording.FlightRecording` slices them positionally. The
rest is the extra ground truth a simulator can supply and a real flight (so far)
cannot. A reader that only knows the old schema is unaffected — it never looks
past column 3, and every consumer here slices positionally from the left, so
appending a column is always safe and removing one never is.

Columns 15-20 are recorded because they are free at capture time and expensive
afterwards: Pegasus already has them on the vehicle state, whereas recovering
acceleration from a recording means differentiating a velocity that was sampled
at the render rate, which is noisy and loses the very transients an
acceleration channel is wanted for. Body-frame velocity is kept alongside the
world-frame one because a policy reasons in the body frame and rotating it back
needs the quaternion applied correctly, which is a step worth not repeating.

``t`` is the **simulation clock**, not a frame index divided by a nominal rate,
whenever the capturing code supplies one. A frame rendered a physics step late
is then honestly stamped instead of being silently placed on an even grid, which
matters because the pose in the same row was read at that same instant.
"""

DEPTH_FORMAT_PNG = "png"
DEPTH_FORMAT_NPY = "npy"
DEPTH_PNG_SCALE_M = 0.001
"""Metres per unit in a uint16 depth PNG (i.e. millimetres).

PNG is the default because a data-collection campaign is storage-bound: a
504x392 float32 ``.npy`` costs 790 kB per frame — 1.4 GB for a three-minute
flight — while the same frame as a 16-bit PNG is a few hundred kB and lossless
to the millimetre, far finer than any depth sensor this stack consumes. ``.npy``
stays available for anything that needs exact float metres.
"""


@dataclass(frozen=True)
class SimFrame:
    """One captured instant of a simulated flight.

    Only ``depth`` and ``pose`` are required, so the original two-field
    construction still works. Everything else is ground truth a simulator
    happens to have and a recording is better off keeping.

    Attributes:
        depth: (H, W) float32 metres.
        pose: (x, y, yaw) in the world frame, FLU convention.
        rgb: Optional (H, W, 3) uint8 RGB frame.
        stamp_s: Simulation time this frame was captured at, seconds. When
            omitted the exporter falls back to ``index / rate_hz``.
        z: World-frame altitude, metres.
        quaternion: ``(qx, qy, qz, qw)`` rotating body FLU into world ENU.
        linear_velocity: World-frame ``(vx, vy, vz)``, m/s.
        angular_velocity: Body-frame ``(wx, wy, wz)``, rad/s.
        linear_acceleration: World-frame ``(ax, ay, az)``, m/s^2.
        body_velocity: Body-frame ``(u, v, w)``, m/s.
    """

    depth: np.ndarray
    pose: tuple  # (x, y, yaw)
    rgb: Optional[np.ndarray] = None
    stamp_s: Optional[float] = None
    z: Optional[float] = None
    quaternion: Optional[Sequence[float]] = None
    linear_velocity: Optional[Sequence[float]] = None
    angular_velocity: Optional[Sequence[float]] = None
    linear_acceleration: Optional[Sequence[float]] = None
    body_velocity: Optional[Sequence[float]] = None


MAX_DEPTH_M = 20.0
"""Range beyond which simulated depth is treated as a no-return.

Isaac's depth sensor reports ``inf`` for a ray that never hits anything -- out
a window, through a doorway, past the far plane. In an office recording that
was 1186 of 1516 frames, up to 54% of the pixels in the worst one. Real depth
sources the rest of the stack consumes (the XTEND's metric-depth engine,
rosbag extractions) are always finite, so nothing downstream guards against it
and the non-finite values propagate into every normalisation and label.
"""


def clamp_depth(depth: np.ndarray, max_depth_m: float = MAX_DEPTH_M) -> np.ndarray:
    """Replace non-finite depth with ``max_depth_m`` and clamp the rest to it.

    Args:
        depth: Raw ``(H, W)`` depth in metres, possibly containing ``inf``/``nan``.
        max_depth_m: Saturation range, metres.

    Returns:
        A finite ``(H, W)`` float32 array in ``[0, max_depth_m]``.
    """
    finite = np.nan_to_num(
        depth.astype(np.float32), nan=max_depth_m, posinf=max_depth_m, neginf=0.0,
    )
    return np.clip(finite, 0.0, max_depth_m)


def write_depth(path_stem: Path, depth_m: np.ndarray, depth_format: str) -> None:
    """Write one clamped depth frame in the requested on-disk format.

    Args:
        path_stem: Destination path **without** a suffix.
        depth_m: ``(H, W)`` float32 metres, already clamped.
        depth_format: :data:`DEPTH_FORMAT_PNG` (uint16 millimetres) or
            :data:`DEPTH_FORMAT_NPY` (float32 metres).

    Raises:
        ValueError: On an unknown format.
        RuntimeError: If the image encoder refused to write the file.
    """
    if depth_format == DEPTH_FORMAT_NPY:
        np.save(str(path_stem) + ".npy", depth_m)
        return
    if depth_format != DEPTH_FORMAT_PNG:
        raise ValueError(
            f"unknown depth_format {depth_format!r}; "
            f"expected {DEPTH_FORMAT_PNG!r} or {DEPTH_FORMAT_NPY!r}"
        )
    millimetres = np.clip(
        np.rint(depth_m / DEPTH_PNG_SCALE_M), 0, np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    if not cv2.imwrite(str(path_stem) + ".png", millimetres):
        raise RuntimeError(f"failed to write depth PNG at {path_stem}.png")


def _pose_row(frame: SimFrame, stamp_s: float) -> list:
    """One row of ``poses.npy`` in :data:`POSE_COLUMNS` order."""
    x, y, yaw = frame.pose
    quaternion = frame.quaternion if frame.quaternion is not None else (0.0, 0.0, 0.0, 1.0)
    zeros = (0.0, 0.0, 0.0)
    linear = frame.linear_velocity if frame.linear_velocity is not None else zeros
    angular = frame.angular_velocity if frame.angular_velocity is not None else zeros
    accel = frame.linear_acceleration if frame.linear_acceleration is not None else zeros
    body = frame.body_velocity if frame.body_velocity is not None else zeros
    return ([stamp_s, float(x), float(y), float(yaw),
             float(frame.z) if frame.z is not None else 0.0]
            + [float(v) for v in quaternion]
            + [float(v) for v in linear]
            + [float(v) for v in angular]
            + [float(v) for v in accel]
            + [float(v) for v in body])


def refuse_a_dirty_recording(out_dir: Path) -> None:
    """Fail if ``out_dir`` already holds frames from an earlier flight.

    Frames are written sequentially from ``000000``, and nothing here clears the
    directory first. So a flight recorded into a directory a *longer* flight
    used before it overwrites the first N frames and leaves the old tail in
    place -- and every consumer reads a directory of numbered images as one
    recording. ``ffmpeg -i %06d.jpg`` cannot tell, the dataset loader cannot
    tell, and the poses row count no longer matches the imagery.

    This is not hypothetical. A comparison recording of the cluttered office
    ended with 201 frames of a different scene, spliced on from a run 26
    minutes earlier; it was caught only because the video tooling separately
    checks that the camera and the map panel cover the same span. Anything that
    reached the training set this way would be silent.

    Refusing is right rather than clearing: the frames may be the only copy of
    an expensive flight, and deleting them to make room is a decision for
    whoever knows that, not for the writer.

    Args:
        out_dir: The recording directory about to be written.

    Raises:
        FileExistsError: There are already frames there.
    """
    for name in ("rgb", "depth"):
        existing = sorted((out_dir / name).glob("*")) if (out_dir / name).is_dir() else []
        if existing:
            raise FileExistsError(
                "%s already holds %d %s frame(s) from an earlier flight; frames "
                "are numbered from zero, so writing here would leave the tail of "
                "that flight spliced onto this one. Move or delete %s first."
                % (out_dir, len(existing), name, out_dir))


class FlightWriter:
    """Write a recording one frame at a time, straight to disk.

    A flight is not held in memory. At 504x392 a single float32 depth frame is
    790 kB and its colour frame another 590 kB, so a three-minute flight
    buffered in RAM is well over 2 GB -- per aircraft, and a collection farm
    runs several at once. Each frame is encoded and written the moment it is
    captured; only the pose rows accumulate, and those are 60 bytes each.

    Args:
        out_dir: Destination recording directory (created if missing).
        intrinsics: The camera's pinhole intrinsics. Must describe the
            resolution the depth and RGB arrays are actually at.
        rate_hz: Nominal capture rate. Used only to stamp frames that carry no
            ``stamp_s`` of their own, and recorded in ``meta.json``.
        camera_height_m: Camera height above the floor.
        pitch_deg: Fixed camera pitch, degrees (0 = level, positive = down).
        max_depth_m: Depth saturation range, see :func:`clamp_depth`.
        depth_format: See :func:`write_depth`.
        rgb_ext: Image extension for colour frames, without the dot.
    """

    def __init__(self, out_dir: Path, intrinsics: Intrinsics, rate_hz: float,
                 camera_height_m: float, pitch_deg: float,
                 max_depth_m: float = MAX_DEPTH_M,
                 depth_format: str = DEPTH_FORMAT_PNG, rgb_ext: str = "jpg"):
        """
        Raises:
            FileExistsError: ``out_dir`` already holds frames from an earlier
                flight. See :func:`refuse_a_dirty_recording`.
        """
        self.out_dir = Path(out_dir)
        self.intrinsics = intrinsics
        self.rate_hz = rate_hz
        self.camera_height_m = camera_height_m
        self.pitch_deg = pitch_deg
        self.max_depth_m = max_depth_m
        self.depth_format = depth_format
        self.rgb_ext = rgb_ext

        self._depth_dir = self.out_dir / "depth"
        self._depth_dir.mkdir(parents=True, exist_ok=True)
        refuse_a_dirty_recording(self.out_dir)
        self._poses = []
        self._have_rgb = False

    @property
    def frames(self) -> int:
        """How many frames have been written so far."""
        return len(self._poses)

    def append(self, frame: SimFrame) -> None:
        """Encode and write one frame, and remember its pose row."""
        index = len(self._poses)
        write_depth(self._depth_dir / f"{index:06d}",
                    clamp_depth(frame.depth, self.max_depth_m), self.depth_format)
        if frame.rgb is not None:
            if not self._have_rgb:
                (self.out_dir / "rgb").mkdir(parents=True, exist_ok=True)
                self._have_rgb = True
            cv2.imwrite(str(self.out_dir / "rgb" / f"{index:06d}.{self.rgb_ext}"),
                        frame.rgb[:, :, ::-1])
        stamp = frame.stamp_s if frame.stamp_s is not None else index / self.rate_hz
        self._poses.append(_pose_row(frame, float(stamp)))

    def discard(self) -> None:
        """Delete everything written so far.

        For a flight that captured nothing: an empty recording directory is
        worse than no directory at all, because anything scanning for
        recordings would find it and treat it as one.
        """
        import shutil

        shutil.rmtree(self.out_dir, ignore_errors=True)

    def close(self, extra_meta: Optional[dict] = None) -> dict:
        """Write ``poses.npy``, ``intrinsics.json`` and ``meta.json``.

        A recording with no frames is **discarded**, not written and not raised
        over. A collection campaign has to survive an episode that never got
        airborne -- an exception here would take down the whole worker, and the
        useful record of the failure is the campaign manifest, not an empty
        directory.

        Args:
            extra_meta: Extra provenance merged into ``meta.json`` (scene, seed,
                goal, outcome, ...). Keys this method sets itself always win.

        Returns:
            The stats dict also written to ``meta.json``. For a discarded
            recording: the provenance plus ``frames: 0`` and
            ``discarded: True``, with nothing on disk.
        """
        if not self._poses:
            self.discard()
            stats = dict(extra_meta or {})
            stats.update({"source": "sim:PEGASUS", "frames": 0, "discarded": True,
                          "duration_s": 0.0, "path_length_m": 0.0})
            return stats

        poses = np.array(self._poses, dtype=np.float32)
        np.save(self.out_dir / "poses.npy", poses)

        (self.out_dir / "intrinsics.json").write_text(json.dumps({
            "width": self.intrinsics.width, "height": self.intrinsics.height,
            "fx": self.intrinsics.fx, "fy": self.intrinsics.fy,
            "cx": self.intrinsics.cx, "cy": self.intrinsics.cy,
        }, indent=2))

        stats = dict(extra_meta or {})
        stats.update({
            "source": "sim:PEGASUS", "frames": int(poses.shape[0]),
            "rate_hz": self.rate_hz,
            "camera_height_m": self.camera_height_m, "pitch_deg": self.pitch_deg,
            "has_rgb": self._have_rgb, "rgb_ext": self.rgb_ext,
            "width": self.intrinsics.width, "height": self.intrinsics.height,
            "max_depth_m": self.max_depth_m,
            "depth_format": self.depth_format,
            "depth_scale_m": (DEPTH_PNG_SCALE_M if self.depth_format == DEPTH_FORMAT_PNG
                              else 1.0),
            "pose_columns": list(POSE_COLUMNS),
            "duration_s": float(poses[-1, 0] - poses[0, 0]),
            "path_length_m": path_length_m(poses),
        })
        (self.out_dir / "meta.json").write_text(json.dumps(stats, indent=2))
        return stats


def export_flight(
    frames: Iterable[SimFrame],
    out_dir: Path,
    intrinsics: Intrinsics,
    rate_hz: float,
    camera_height_m: float,
    pitch_deg: float,
    max_depth_m: float = MAX_DEPTH_M,
    depth_format: str = DEPTH_FORMAT_PNG,
    rgb_ext: str = "jpg",
    extra_meta: Optional[dict] = None,
) -> dict:
    """Write a whole flight at once. Thin wrapper over :class:`FlightWriter`.

    Convenient when the frames are already in hand (a test, a re-export). A live
    flight should drive the writer directly instead of buffering itself.

    Args:
        frames: The captured flight, in temporal order.
        out_dir: Destination recording directory (created if missing).
        intrinsics: The camera's pinhole intrinsics.
        rate_hz: Nominal capture rate.
        camera_height_m: Camera height above the floor.
        pitch_deg: Fixed camera pitch, degrees.
        max_depth_m: Depth saturation range.
        depth_format: See :func:`write_depth`.
        rgb_ext: Image extension for colour frames, without the dot.
        extra_meta: Extra provenance merged into ``meta.json``.

    Returns:
        The stats dict also written to ``meta.json``.

    Raises:
        ValueError: If ``frames`` is empty. Unlike :meth:`FlightWriter.close`,
            which is driving a live flight that may legitimately have failed
            before takeoff, a caller who already holds the frames and passes
            none has made a mistake.
    """
    writer = FlightWriter(out_dir, intrinsics, rate_hz, camera_height_m, pitch_deg,
                          max_depth_m=max_depth_m, depth_format=depth_format,
                          rgb_ext=rgb_ext)
    count = 0
    for frame in frames:
        writer.append(frame)
        count += 1
    if count == 0:
        writer.discard()
        raise ValueError("export_flight: no frames were provided")
    return writer.close(extra_meta)


def path_length_m(poses: np.ndarray) -> float:
    """Total 3D distance travelled across a ``poses.npy`` array.

    Args:
        poses: ``(N, >=5)`` array in :data:`POSE_COLUMNS` order.

    Returns:
        Metres. 0.0 for fewer than two rows.
    """
    if poses.shape[0] < 2:
        return 0.0
    xyz = np.stack([poses[:, 1], poses[:, 2], poses[:, 4]], axis=1)
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def resolution_of(intrinsics: Intrinsics, width: int, height: int) -> Intrinsics:
    """Rescale pinhole intrinsics to a different image resolution.

    A calibration is only valid at the resolution it was measured at; rendering
    the same camera at a different one scales the focal lengths and the
    principal point by exactly the axis ratios. Aspect-ratio changes are allowed
    (the scales are independent per axis), which is what makes an arbitrary
    ``--resolution`` safe to ask for.

    Args:
        intrinsics: Calibration at its original resolution.
        width: New image width, pixels.
        height: New image height, pixels.

    Returns:
        The rescaled intrinsics.

    Raises:
        ValueError: If either dimension is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive, got {width}x{height}")
    sx = width / float(intrinsics.width)
    sy = height / float(intrinsics.height)
    return Intrinsics(
        width=int(width), height=int(height),
        fx=intrinsics.fx * sx, fy=intrinsics.fy * sy,
        cx=intrinsics.cx * sx, cy=intrinsics.cy * sy,
    )


def parse_resolution(text: str) -> Tuple[int, int]:
    """Parse a ``"WxH"`` command-line resolution.

    Args:
        text: e.g. ``"504x392"``.

    Returns:
        ``(width, height)``.

    Raises:
        ValueError: If the string is not two positive integers separated by ``x``.
    """
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"resolution must look like 640x480, got {text!r}")
    try:
        width, height = (int(p) for p in parts)
    except ValueError:
        raise ValueError(f"resolution must look like 640x480, got {text!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive, got {text!r}")
    return width, height
