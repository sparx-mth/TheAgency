"""Put two arms' flights of the same mission side by side in one video.

    python -m ...world_goal.compare_videos --flights ~/navdp_world_goal/flights \\
        --out ~/navdp_world_goal/flights/comparison

``fly_navdp.py --video`` writes one MP4 per mission per arm from the chase
camera. Those are the right recordings but the wrong shape for looking at: to
see *why* one set of weights crashed where the other did not, the two have to be
on screen together, starting at the same moment, from the same place, toward the
same goal.

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
    """
    pad = "tpad=stop_mode=clone:stop_duration=600"
    longest = max(_seconds(left_camera), _seconds(right_camera),
                  _seconds(left_track), _seconds(right_track))

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


def track_panel(arm_dir: Path, index: int, scene: Optional[str],
                size: int) -> Optional[Path]:
    """The map-panel video for one mission, rendering it if it is not there yet.

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
    return track_video.render(log, existing, scene, size_px=size)


def pair_missions(flights_dir: Path, out_dir: Path, left_arm: str,
                  right_arm: str, missions: Optional[List[int]] = None,
                  layout: str = "quad", scene: Optional[str] = None,
                  cell: int = 480) -> List[Path]:
    """Compose one comparison video per mission both arms recorded.

    ``layout="quad"`` puts each arm's camera above its map panel; ``"side"``
    shows the cameras alone. Quad needs a ``*_track.json`` per flight, which
    only exists for flights flown with ``--video``.
    """
    left_dir, right_dir = flights_dir / left_arm, flights_dir / right_arm
    left_results, right_results = outcomes(left_dir), outcomes(right_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for left_video in sorted(left_dir.glob("mission_*.mp4")):
        if left_video.stem.endswith("_track"):        # our own rendered panel
            continue
        index = int(left_video.stem.split("_")[1])
        if missions is not None and index not in missions:
            continue
        right_video = right_dir / left_video.name
        if not right_video.is_file():
            print(f"[compare] mission {index}: no {right_arm} video, skipped")
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
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("[compare] ffmpeg/ffprobe not on PATH")
        return 1

    flights = Path(args.flights).expanduser()
    out_dir = Path(args.out).expanduser() if args.out else flights / "comparison"
    written = pair_missions(flights, out_dir, args.left, args.right, args.missions,
                            layout=args.layout, scene=args.scene, cell=args.cell)
    if not written:
        print("[compare] nothing written -- were the flights flown with --video?")
        return 1
    print(f"[compare] {len(written)} comparison video(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
