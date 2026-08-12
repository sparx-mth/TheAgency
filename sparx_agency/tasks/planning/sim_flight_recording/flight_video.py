"""Turn recordings into watchable flights: the camera beside the map, in step.

    .venv/bin/python sparx_agency/tasks/planning/sim_flight_recording/flight_video.py \\
        ~/data/sim/showcase --out-dir ~/data/sim/showcase/videos --limit 10 --combined all_flights.mp4

Half the frame is what the drone saw. The other half is the whole surveyed
building with **where it was told to go**, **the A\\* route it was given**, **where
it actually was** and **the path it actually flew** -- the four things you need
together to say whether a flight was good, and which no single view shows.

Both halves advance on the same frame index, because ``poses.npy`` has exactly one
row per recorded camera frame. That is the property that makes this trustworthy:
nothing is being interpolated, aligned or guessed, so the map cannot show the
aircraft somewhere the camera does not. (A comparison video that *did* guess, by
spreading one stream evenly over another's timeline, once made a correctly located
aircraft look badly mislocalised -- see ``vlas/navdp/finetune/world_goal/README.md``.)

Side by side the two panes share a *height*, and each keeps its own width, so
neither is stretched and a tall building stays a tall building. ``--layout stack``
shares a width instead, which suits a wide one.

Needs ``ffmpeg`` on the PATH. It is on the host but not in the isaac-sim
container, so run this on the host against the recordings the container wrote.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from sparx_agency.tasks.planning.sim_flight_recording.flight_map_panel import (
    FLOWN_COLOUR, GOAL_COLOUR, PLANNED_COLOUR, MapPanel, route_points,
)

BACKGROUND = (248, 248, 248)     # BGR, the letterbox behind each half
CAPTION_HEIGHT = 34      # even, like every other dimension -- see _even()
CAPTION_BACKGROUND = (36, 36, 36)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale ``image`` into a ``width`` x ``height`` box, centred, aspect kept.

    Letterboxed rather than stretched: a squashed camera frame is misleading about
    what the lens saw, and a squashed map is misleading about the building.

    Args:
        image: BGR image.
        width: Box width, pixels.
        height: Box height, pixels.

    Returns:
        A ``(height, width, 3)`` BGR image.
    """
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)),
                                 max(1, int(image.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), BACKGROUND, np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def gamma_lut(gamma: float) -> Optional[np.ndarray]:
    """A 256-entry lookup table for a display gamma, or None for 1.0.

    The ``office`` scene is lit like a real office: readable down a corridor and
    genuinely dark facing into a shadow. That is what the policy is trained on and
    it should not be quietly "fixed" -- but a human watching a video needs to see,
    so this is offered as an explicit, labelled display correction rather than
    applied by default.

    Args:
        gamma: Below 1.0 brightens, above 1.0 darkens. Exactly 1.0 means no change.

    Returns:
        A ``(256,)`` uint8 table for ``cv2.LUT``, or None.

    Raises:
        ValueError: If ``gamma`` is not positive.
    """
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if gamma == 1.0:
        return None
    scale = np.arange(256, dtype=np.float64) / 255.0
    return np.clip(np.power(scale, gamma) * 255.0, 0, 255).astype(np.uint8)


def caption_bar(width: int, text: str, legend: bool = True) -> np.ndarray:
    """A dark strip naming the flight, with the route colour key.

    The key is not decoration: two coloured lines on a map are unreadable without
    being told which is the plan and which is the flight.
    """
    bar = np.full((CAPTION_HEIGHT, width, 3), CAPTION_BACKGROUND, np.uint8)
    cv2.putText(bar, text, (10, 23), FONT, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
    if not legend:
        return bar
    entries = (("A* route", PLANNED_COLOUR), ("flown", FLOWN_COLOUR),
               ("goal", GOAL_COLOUR))
    right = width - 10
    for label, colour in reversed(entries):
        (text_width, _), _ = cv2.getTextSize(label, FONT, 0.48, 1)
        right -= text_width
        cv2.putText(bar, label, (right, 23), FONT, 0.48, (235, 235, 235), 1,
                    cv2.LINE_AA)
        right -= 8
        cv2.line(bar, (right - 18, 18), (right, 18), colour, 3, cv2.LINE_AA)
        right -= 34
    return bar


def _frame(camera: Optional[np.ndarray], panel: np.ndarray, panes, layout: str
           ) -> np.ndarray:
    """One composed frame: the camera pane beside (or above) the map pane."""
    (camera_width, camera_height), (panel_width, panel_height) = panes
    view = (np.full((camera_height, camera_width, 3), 30, np.uint8)
            if camera is None else fit(camera, camera_width, camera_height))
    plan = fit(panel, panel_width, panel_height)
    return (np.hstack([view, plan]) if layout == "side"
            else np.vstack([view, plan]))


def pane_sizes(camera_shape, panel_shape, pane_px: int, layout: str):
    """Sizes of the two panes and of the finished video.

    Each pane is given **its own content's aspect ratio** rather than an equal
    share of the picture. Forcing two halves of the same width looks tidier in the
    abstract and is worse here: the ``office`` map is 30.7 m by 74.5 m, so an
    equal half spends 60% of its width on blank margin and shrinks the map to
    match. Sharing one dimension -- the height, side by side -- makes both panes as
    large as that dimension allows and keeps them the same scale as each other.

    Args:
        camera_shape: ``(height, width)`` of a camera frame.
        panel_shape: ``(height, width)`` of a map panel.
        pane_px: The shared dimension: pane height for ``side``, pane width for
            ``stack``.
        layout: ``"side"`` or ``"stack"``.

    Returns:
        ``((camera_pane, panel_pane), (video_width, video_height))`` where each
        pane is ``(width, height)``.
    """
    camera_aspect = camera_shape[1] / camera_shape[0]
    panel_aspect = panel_shape[1] / panel_shape[0]
    shared = _even(pane_px)
    if layout == "side":
        camera_pane = (_even(shared * camera_aspect), shared)
        panel_pane = (_even(shared * panel_aspect), shared)
        size = (camera_pane[0] + panel_pane[0], shared + CAPTION_HEIGHT)
    else:
        camera_pane = (shared, _even(shared / camera_aspect))
        panel_pane = (shared, _even(shared / panel_aspect))
        size = (shared, camera_pane[1] + panel_pane[1] + CAPTION_HEIGHT)
    return (camera_pane, panel_pane), size


def _even(value: float) -> int:
    """``value`` rounded to the nearest even integer, at least 2.

    Every dimension has to be even. ``libx264`` with ``yuv420p`` chroma-subsamples
    by two, and an odd width makes ffmpeg reject the stream on its first frame --
    which arrives as ``BrokenPipeError`` here, from the far end of the pipe, naming
    nothing.
    """
    return max(2, 2 * int(round(value / 2)))


def render(recording_dir: Path, out_path: Path, map_dir: Optional[Path] = None,
           pane_px: int = 640, layout: str = "side",
           stride: int = 1, gamma: float = 1.0) -> Optional[Path]:
    """Render one recording to an MP4.

    Args:
        recording_dir: A recording directory (``poses.npy`` + ``meta.json``).
        out_path: MP4 to write.
        map_dir: Override where surveyed maps are read from.
        pane_px: The dimension the two panes share, pixels: their height in
            ``side`` layout, their width in ``stack``.
        layout: ``"side"`` (camera left, map right) or ``"stack"``.
        stride: Keep every ``stride``-th recorded frame. 2 halves the length of a
            slow flight without changing what it shows.
        gamma: Display gamma for the camera pane only -- see :func:`gamma_lut`.
            Anything but 1.0 is named in the caption, so a brightened video says
            that it is one.

    Returns:
        The written path, or None if the recording could not be drawn.
    """
    from sparx_agency.robots.PEGASUS.adapters.scene_map import load_scene_map
    from sparx_agency.tasks.planning.vlas.common.finetune.datasets.recording import (
        load_recording,
    )

    recording_dir = Path(recording_dir)
    meta = json.loads((recording_dir / "meta.json").read_text())
    recording = load_recording(recording_dir)
    try:
        grid, _map_meta, _layers = load_scene_map(
            meta["scene"], float(meta["altitude_m"]), map_dir)
    except (KeyError, FileNotFoundError, ValueError) as error:
        print(f"[video] {recording_dir.name}: no surveyed map -- {error}")
        return None

    if recording.num_frames == 0:
        # An episode that never armed is written out with its outcome and no
        # frames, which is deliberate -- but there is no flight to draw.
        print(f"[video] {recording_dir.name}: no frames "
              f"({meta.get('outcome', 'unknown outcome')})")
        return None

    poses = recording.poses
    flown = poses[:, 1:3].astype(np.float64)
    planned = route_points(meta, flown[0])
    goal = meta.get("goal_xy")
    panel = MapPanel(grid, pane_px)

    indices = list(range(0, recording.num_frames, max(1, stride)))
    first = recording.rgb(indices[0]) if indices else None
    camera_shape = (recording.intrinsics.height, recording.intrinsics.width) \
        if first is None else first.shape
    panes, size = pane_sizes(camera_shape, (panel.height, panel.width),
                             pane_px, layout)
    fps = max(1.0, recording.rate_hz / max(1, stride))
    lut = gamma_lut(gamma)
    label = _label(recording_dir.name, meta)
    if lut is not None:
        # ASCII only: cv2's Hershey fonts have no glyph for a Greek gamma and
        # draw it as two question marks.
        label += f"   [view gamma {gamma:g}]"

    with _encoder(out_path, size, fps) as sink:
        for index in indices:
            rgb = recording.rgb(index)
            camera = None if rgb is None else rgb[:, :, ::-1].copy()   # RGB -> BGR
            if camera is not None and lut is not None:
                camera = cv2.LUT(camera, lut)
            drawn = panel.draw(flown, index + 1, poses[index, 1:4], planned, goal)
            composed = _frame(camera, drawn, panes, layout)
            elapsed = float(poses[index, 0])
            bar = caption_bar(composed.shape[1], f"{label}   t={elapsed:5.1f}s")
            if not sink.write(np.vstack([bar, composed])):
                break
    if not out_path.is_file():
        return None
    print(f"[video] {out_path.name}: {len(indices)} frames at {fps:.1f} fps")
    return out_path


def _label(name: str, meta: dict) -> str:
    """One line naming the flight, how it ended, and what the drawn route is."""
    error = meta.get("goal_error_m")
    drift = meta.get("estimator_drift_m")
    replans = meta.get("replans") or 0
    parts = [name, meta.get("outcome", "?")]
    if isinstance(error, (int, float)):
        parts.append(f"goal err {error:.2f}m")
    if isinstance(drift, (int, float)):
        parts.append(f"est drift {drift:.2f}m")
    if replans:
        # Only the last route is in the metadata, so on a replanned flight the
        # green line covers the last leg and not the whole flight. Saying so beats
        # letting it read as the aircraft ignoring most of its plan.
        parts.append(f"replanned {int(replans)}x (route shown is the last)")
    return "   ".join(parts)


class _encoder:
    """Pipe raw BGR frames straight into ffmpeg.

    Writing PNGs to a temporary directory first costs more time in libpng than the
    whole render, and a ten-flight run is thousands of files.
    """

    def __init__(self, out_path: Path, size, fps: float):
        self._command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.4f}", "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", str(out_path),
        ]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        self._process = subprocess.Popen(self._command, stdin=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        return self

    def write(self, frame: np.ndarray) -> bool:
        """Push one frame. False once ffmpeg has stopped reading."""
        try:
            self._process.stdin.write(frame.tobytes())
            return True
        except BrokenPipeError:
            # ffmpeg has already exited and will explain why in __exit__. Writing
            # into a closed pipe here would only replace its message with ours.
            return False

    def __exit__(self, *exception):
        # communicate() closes stdin itself; closing it here first makes its own
        # flush raise ValueError on a file it no longer owns.
        try:
            _stdout, stderr = self._process.communicate()
        except (BrokenPipeError, ValueError):
            stderr = b""
            self._process.wait()
        if self._process.returncode != 0 and exception[0] is None:
            raise RuntimeError(
                f"ffmpeg refused the frame stream (exit {self._process.returncode}). "
                f"An odd width or height is the usual cause -- libx264 with yuv420p "
                f"needs both even, see _even(). ffmpeg said:\n"
                f"{stderr.decode('utf-8', 'replace')[-800:]}"
            )
        return False


def combine(clips: Sequence[Path], out_path: Path) -> Optional[Path]:
    """Join clips into one video, in the order given.

    Uses ffmpeg's concat demuxer, which needs the inputs to share resolution and
    frame rate -- they do, because :func:`render` sized them all the same way.

    Args:
        clips: The MP4s to join.
        out_path: The combined MP4.

    Returns:
        The written path, or None if ffmpeg failed or there was nothing to join.
    """
    clips = [Path(clip) for clip in clips if Path(clip).is_file()]
    if not clips:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for clip in clips:
            listing.write(f"file '{clip.resolve()}'\n")
        listing_path = Path(listing.name)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(listing_path), "-c", "copy", str(out_path)],
            capture_output=True, text=True)
    finally:
        listing_path.unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"[video] ffmpeg concat failed:\n{result.stderr[-800:]}")
        return None
    print(f"[video] combined {len(clips)} flights into {out_path}")
    return out_path


def main(argv=None) -> int:
    from sparx_agency.tasks.planning.sim_flight_recording.inspect_recording import (
        find_recordings,
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path,
                        help="a recording directory, or a campaign directory of them")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where to write the MP4s (default: <root>/videos)")
    parser.add_argument("--limit", type=int, default=0,
                        help="render at most this many recordings (0 = all)")
    parser.add_argument("--combined", default=None,
                        help="also join them into this one file, inside --out-dir")
    parser.add_argument("--layout", choices=("side", "stack"), default="side",
                        help="'side': camera left, map right (suits a tall map); "
                             "'stack': camera above map")
    parser.add_argument("--pane", type=int, default=640,
                        help="the dimension the two panes share, pixels: their "
                             "height side by side, their width stacked")
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth recorded frame")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="display gamma for the camera pane; below 1 brightens "
                             "the (deliberately dim) office scene. 1.0 leaves the "
                             "recorded frames exactly as the policy sees them")
    parser.add_argument("--map-dir", type=Path, default=None)
    parser.add_argument("--landed-only", action="store_true",
                        help="skip recordings whose outcome is not 'landed'")
    args = parser.parse_args(argv)

    recordings = find_recordings(args.root)
    if args.landed_only:
        recordings = [path for path in recordings if _outcome(path) == "landed"]
    if not recordings:
        print(f"[video] no recordings under {args.root}", file=sys.stderr)
        return 1
    if args.limit:
        recordings = recordings[:args.limit]

    out_dir = args.out_dir or (Path(args.root) / "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for path in recordings:
        result = render(path, out_dir / f"{path.name}.mp4", args.map_dir,
                        args.pane, args.layout, args.stride, args.gamma)
        if result is not None:
            written.append(result)

    if not written:
        print("[video] nothing rendered", file=sys.stderr)
        return 1
    print(f"[video] {len(written)} flight video(s) in {out_dir}")
    if args.combined:
        combine(written, out_dir / args.combined)
    return 0


def _outcome(recording_dir: Path) -> str:
    """A recording's outcome, or the empty string if unreadable."""
    try:
        return json.loads((recording_dir / "meta.json").read_text()).get("outcome", "")
    except (OSError, ValueError):
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
