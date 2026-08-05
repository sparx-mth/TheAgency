"""Put two arms' flights of the same mission side by side in one video.

    python -m ...world_goal.compare_videos --flights ~/navdp_world_goal/flights \\
        --out ~/navdp_world_goal/flights/comparison

``fly_navdp.py --video`` writes one MP4 per mission per arm from the chase
camera, plus the recorded onboard ``rgb/`` frames it always keeps. Those
chase-cam clips are the right recordings but the wrong shape for looking at:
to see *why* one set of weights crashed where the other did not, the two have
to be on screen together, starting at the same moment, from the same place,
toward the same goal. ``--camera-source onboard`` swaps the chase view for the
drone's own forward-facing camera -- the actual frame NavDP ran inference on.

Each output pairs one mission: baseline left, fine-tuned right, a caption under
each naming the arm and how that flight ended. The two clips rarely last the
same time -- a crash is short, a timeout is long -- so the shorter one holds its
last frame rather than the pair going out of sync or one side going black.

Needs ``ffmpeg`` on the PATH. It is on the host but not in the isaac-sim
container, so run this after copying the flights out.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def outcomes(arm_dir: Path) -> Dict[int, Dict]:
    """``{mission index: result}`` from an arm's ``summary.json``."""
    summary = arm_dir / "summary.json"
    if not summary.is_file():
        return {}
    try:
        stored = json.loads(summary.read_text())
    except (OSError, ValueError):
        return {}
    return {int(r["mission"]): r for r in stored.get("results", [])}


def caption(arm: str, result: Optional[Dict]) -> str:
    """One line naming the arm and how the flight ended."""
    if not result:
        return arm
    clearance = result.get("min_clear_m")
    contact = " HIT" if result.get("collided") else " clear"
    return (f"{arm}  {result.get('outcome', '?')}"
            f"  goal err {result.get('goal_error_m', float('nan')):.1f}m"
            f"  min clear {clearance:.2f}m{contact}"
            if clearance is not None else f"{arm}  {result.get('outcome', '?')}")


def _escape(text: str) -> str:
    """ffmpeg's drawtext parser needs colons and quotes escaped."""
    return text.replace("\\", "").replace(":", "\\:").replace("'", "")


def compose(left: Path, right: Path, out_path: Path,
            left_label: str, right_label: str, height: int = 540) -> bool:
    """Stack two clips horizontally with a caption under each.

    ``tpad`` clones the last frame of whichever clip is shorter, so the pair
    stays aligned to its start instead of one half cutting to black.

    Returns:
        True if ffmpeg produced a file.
    """
    filters = (
        f"[0:v]scale=-2:{height},tpad=stop_mode=clone:stop_duration=600,"
        f"drawtext=text='{_escape(left_label)}':x=10:y=h-30:fontsize=20:"
        f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6[l];"
        f"[1:v]scale=-2:{height},tpad=stop_mode=clone:stop_duration=600,"
        f"drawtext=text='{_escape(right_label)}':x=10:y=h-30:fontsize=20:"
        f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6[r];"
        f"[l][r]hstack=inputs=2,"
        # Both inputs were padded to +600 s; end when the longer real clip does.
        f"trim=duration={_duration(max(_seconds(left), _seconds(right)))}[v]"
    )
    command = ["ffmpeg", "-y", "-i", str(left), "-i", str(right),
               "-filter_complex", filters, "-map", "[v]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", str(out_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[compare] ffmpeg failed for {out_path.name}:\n{result.stderr[-800:]}")
        return False
    return out_path.is_file()


def compose_quad(left_camera: Path, left_track: Path,
                 right_camera: Path, right_track: Path, out_path: Path,
                 left_label: str, right_label: str, cell: int = 480) -> bool:
    """Four panels: each arm's camera above its map, the two arms side by side.

    Layout, which is the whole point of the figure::

        +---------------------+---------------------+
        |  LEFT arm camera    |  RIGHT arm camera   |
        +---------------------+---------------------+
        |  LEFT arm  NavDP    |  RIGHT arm NavDP    |
        |  plan on the map    |  plan on the map    |
        +---------------------+---------------------+

    A thick coloured divider separates the two arms, because the failure mode of
    a four-panel figure is the viewer losing track of which column is which.
    Each column is captioned with its arm and how that flight ended.

    Every input is padded by cloning its last frame, so the four stay aligned to
    a common start even though a crash clip is short and a timeout clip is long.
    That padding aligns the *starts* and nothing else: a camera and a map panel
    covering different spans of the same flight will run at different speeds all
    the way through, and the result looks exactly like a mislocalised aircraft.
    :func:`check_alignment` is there because that already happened once.
    """
    pad = "tpad=stop_mode=clone:stop_duration=600"
    longest = max(_seconds(left_camera), _seconds(right_camera),
                  _seconds(left_track), _seconds(right_track))
    check_alignment(left_camera, left_track)
    check_alignment(right_camera, right_track)

    def column(camera_index: int, track_index: int, label: str, colour: str) -> str:
        """Camera over map, captioned, with the arm's colour as a top border."""
        return (
            f"[{camera_index}:v]scale={cell}:-2,{pad},"
            f"drawtext=text='{_escape(label)}':x=8:y=8:fontsize=17:fontcolor=white:"
            f"box=1:boxcolor={colour}@0.85:boxborderw=6[c{camera_index}];"
            f"[{track_index}:v]scale={cell}:-2,{pad}[t{camera_index}];"
            f"[c{camera_index}][t{camera_index}]vstack=inputs=2[col{camera_index}];"
        )

    filters = (
        column(0, 1, left_label, "0x1b5e20")        # green: fine-tuned
        + column(2, 3, right_label, "0x8a1c1c")     # red: baseline
        + f"[col0]pad=w=iw+8:h=ih:x=0:y=0:color=white[col0p];"
          f"[col0p][col2]hstack=inputs=2,"
          f"trim=duration={_duration(longest)},setsar=1[v]"
    )
    command = ["ffmpeg", "-y",
               "-i", str(left_camera), "-i", str(left_track),
               "-i", str(right_camera), "-i", str(right_track),
               "-filter_complex", filters, "-map", "[v]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", str(out_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[compare] ffmpeg failed for {out_path.name}:\n{result.stderr[-900:]}")
        return False
    return out_path.is_file()


ALIGNMENT_TOLERANCE_S = 1.0
"""How far a camera clip and its map panel may differ in length.

They describe the same flight, so any real difference means one of them does not
cover all of it. A second of slack absorbs rounding in the frame counts and the
last partial capture; the mismatch this exists to catch was ten seconds.
"""


def check_alignment(camera: Path, track: Path) -> bool:
    """Warn if a camera clip and its map panel do not cover the same span.

    Stacking two clips of different lengths and playing both from zero silently
    reinterprets one of them: at 26 seconds of panel against 36 of camera, the map
    ran a third faster than the view, showed the aircraft several metres from where
    the camera showed it, and was read as a localisation failure. Nothing in the
    output says so, so it has to be said here.

    Args:
        camera: The camera clip.
        track: The map-panel clip.

    Returns:
        True if the two agree to :data:`ALIGNMENT_TOLERANCE_S`.
    """
    camera_s, track_s = _seconds(camera), _seconds(track)
    if abs(camera_s - track_s) <= ALIGNMENT_TOLERANCE_S:
        return True
    print(f"[compare] WARNING: {camera.name} is {camera_s:.1f} s but "
          f"{track.name} is {track_s:.1f} s. Stacked and played together the map "
          f"panel will run at a different speed from the view, and the aircraft "
          f"will appear to be somewhere it never was. Delete "
          f"{track.name} and re-render it -- an old track log (schema < 2) has no "
          f"clock and can only be drawn one frame per inference.")
    return False


def _seconds(path: Path) -> float:
    """Duration of a clip, or 0 when ffprobe cannot read it."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _duration(seconds: float) -> str:
    return f"{max(seconds, 0.1):.2f}"


def capture_rate(arm_dir: Path, index: int, fallback: float = 10.0) -> float:
    """The rate the mission's camera frames were captured at, from its ``meta.json``.

    The chase-cam MP4 is encoded one frame per capture, so this is also that
    clip's frame rate -- and rendering the map panel at the same rate is what
    makes the two clips the same length for a given flight, which is what
    :func:`compose_quad` needs to stack them honestly.

    Args:
        arm_dir: One arm's flight directory.
        index: Mission index.
        fallback: Used when the recording has no meta (``FlightRecorder``'s own
            default rate).

    Returns:
        Frames per second.
    """
    meta_path = arm_dir / f"mission_{index:02d}" / "meta.json"
    try:
        return float(json.loads(meta_path.read_text()).get("rate_hz", fallback))
    except (OSError, ValueError, TypeError):
        return fallback


def track_panel(arm_dir: Path, index: int, scene: Optional[str],
                size: int) -> Optional[Path]:
    """The map-panel video for one mission, rendering it if it is not there yet.

    Rendered at the camera's own capture rate so the panel and the camera clip
    cover the same span at the same speed. They used to not: the panel began at
    the flight's *first inference*, some ten seconds after the camera started
    rolling, and both were played from zero -- so the map showed the aircraft
    already under way while the camera still showed it on the ground. That read as
    a localisation fault and was a composition fault.

    Cached next to the flight, because rendering a few hundred matplotlib frames
    is the slowest step here and re-cutting a comparison should not repeat it.
    """
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal import track_video

    existing = arm_dir / f"mission_{index:02d}_track.mp4"
    if existing.is_file():
        return existing
    log = arm_dir / f"mission_{index:02d}_track.json"
    if not log.is_file():
        return None
    return track_video.render(log, existing, scene, size_px=size,
                              fps=capture_rate(arm_dir, index))


def onboard_video(arm_dir: Path, index: int) -> Optional[Path]:
    """The onboard-camera MP4 for one mission, encoding it if it is not there yet.

    ``fly_navdp.py``'s ``mission_NN.mp4`` is the external chase camera -- a
    view for a human, not the sensor input. NavDP is an RGB-D policy fed one
    onboard frame at a time (``core/planning/vlas/navdp/client.py``), and
    those exact frames are already on disk as ``mission_NN/rgb/NNNNNN.<ext>``
    (``FlightRecorder`` writes them regardless of which camera the MP4 uses).
    This just re-encodes that sequence, so it is what NavDP actually saw.

    Cached next to the flight, same reasoning as :func:`track_panel`.
    """
    existing = arm_dir / f"mission_{index:02d}_onboard.mp4"
    if existing.is_file():
        return existing
    mission_dir = arm_dir / f"mission_{index:02d}"
    rgb_dir = mission_dir / "rgb"
    meta_path = mission_dir / "meta.json"
    if not rgb_dir.is_dir() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    rate_hz = meta.get("rate_hz", 10.0)
    ext = meta.get("rgb_ext", "jpg")
    command = ["ffmpeg", "-y", "-framerate", str(rate_hz), "-i",
               str(rgb_dir / f"%06d.{ext}"), "-c:v", "libx264",
               "-pix_fmt", "yuv420p", str(existing)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[compare] ffmpeg failed to encode onboard video for mission "
              f"{index}:\n{result.stderr[-800:]}")
        return None
    return existing if existing.is_file() else None


def pair_missions(flights_dir: Path, out_dir: Path, left_arm: str,
                  right_arm: str, missions: Optional[List[int]] = None,
                  layout: str = "quad", scene: Optional[str] = None,
                  cell: int = 480, camera: str = "chase") -> List[Path]:
    """Compose one comparison video per mission both arms recorded.

    ``layout="quad"`` puts each arm's camera above its map panel; ``"side"``
    shows the cameras alone. Quad needs a ``*_track.json`` per flight, which
    only exists for flights flown with ``--video``.

    ``camera="chase"`` uses ``fly_navdp.py``'s external chase-cam MP4;
    ``"onboard"`` re-encodes the recorded onboard frames instead -- see
    :func:`onboard_video`.
    """
    left_dir, right_dir = flights_dir / left_arm, flights_dir / right_arm
    left_results, right_results = outcomes(left_dir), outcomes(right_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = sorted(set(left_results) & set(right_results))
    written = []
    for index in indices:
        if missions is not None and index not in missions:
            continue
        if camera == "onboard":
            left_video = onboard_video(left_dir, index)
            right_video = onboard_video(right_dir, index)
        else:
            left_video = left_dir / f"mission_{index:02d}.mp4"
            left_video = left_video if left_video.is_file() else None
            right_video = right_dir / f"mission_{index:02d}.mp4"
            right_video = right_video if right_video.is_file() else None
        if left_video is None or right_video is None:
            print(f"[compare] mission {index}: no {camera} video for both arms, "
                  f"skipped")
            continue

        left_label = caption(left_arm, left_results.get(index))
        right_label = caption(right_arm, right_results.get(index))
        out_path = out_dir / f"mission_{index:02d}_{left_arm}_vs_{right_arm}.mp4"

        if layout == "quad":
            left_track = track_panel(left_dir, index, scene, cell)
            right_track = track_panel(right_dir, index, scene, cell)
            if left_track is None or right_track is None:
                print(f"[compare] mission {index}: no track log for both arms, "
                      f"falling back to cameras only")
            elif compose_quad(left_video, left_track, right_video, right_track,
                              out_path, left_label, right_label, cell):
                written.append(out_path)
                print(f"[compare] wrote {out_path}")
                continue

        if compose(left_video, right_video, out_path, left_label, right_label):
            written.append(out_path)
            print(f"[compare] wrote {out_path}")
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--flights", required=True,
                        help="directory holding one subdirectory per arm")
    parser.add_argument("--out", default=None, help="where to write the pairs")
    parser.add_argument("--left", default="trained",
                        help="arm in the left column (default: the fine-tuned one)")
    parser.add_argument("--right", default="baseline",
                        help="arm in the right column")
    parser.add_argument("--layout", default="quad", choices=("quad", "side"),
                        help="'quad': camera over map per arm; 'side': cameras only")
    parser.add_argument("--scene", default=None,
                        help="scene for the map panel; defaults to the track log's")
    parser.add_argument("--cell", type=int, default=480, help="panel width, pixels")
    parser.add_argument("--missions", type=int, nargs="*", default=None,
                        help="only these mission indices (default: all paired)")
    parser.add_argument("--camera-source", default="chase",
                        choices=("chase", "onboard"),
                        help="'chase': the external chase-cam MP4 (default); "
                             "'onboard': re-encode the recorded onboard frames "
                             "-- what NavDP actually saw")
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("[compare] ffmpeg/ffprobe not on PATH")
        return 1

    flights = Path(args.flights).expanduser()
    out_dir = Path(args.out).expanduser() if args.out else flights / "comparison"
    written = pair_missions(flights, out_dir, args.left, args.right, args.missions,
                            layout=args.layout, scene=args.scene, cell=args.cell,
                            camera=args.camera_source)
    if not written:
        print("[compare] nothing written -- were the flights flown with --video?")
        return 1
    print(f"[compare] {len(written)} comparison video(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
