"""Render a surveyed 2D map to a PNG you can actually read.

    python -m ...sim_flight_recording.render_map_png --scene office
    python -m ...sim_flight_recording.render_map_png --scene office --reachable

``survey_scene.py`` writes ``<scene>_alt<NNNN>cm.npz`` — an occupancy slice at
one flight altitude. It is the map every planner, goal sampler and evaluation in
this stack reasons about, but as an ``.npz`` it is invisible: you cannot tell
whether a wing is enclosed, where the origin sits, or how big the building is
without loading it in Python.

This draws it with metre axes, a scale bar and the world origin marked, so the
picture and the coordinates a mission prints are the same thing. Three cell
states are distinguished, and the distinction matters:

``free``      surveyed and clear — where an aircraft may fly.
``occupied``  surveyed and blocked.
``unknown``   never observed by the sweep. Not the same as free: goal sampling
              draws only from the connected free component, so unknown space
              is space no mission will ever be given.

``--reachable`` shades the largest connected free component separately, which is
the honest picture of where flights can actually go. A wing that looks open but
is cut off by a closed door shows up immediately, and that has already explained
one empty evaluation split in this project.

matplotlib and numpy only — no simulator, no GPU, no container.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

FREE = "#ffffff"
OCCUPIED = "#2b2b2b"
UNKNOWN = "#c9ccd1"
REACHABLE = "#d7ecff"
ORIGIN = "#d81b60"

DEFAULT_MAP_DIR = Path(__file__).resolve().parents[3] / "robots" / "PEGASUS" / "maps"


def map_path(scene: str, altitude_m: float = 1.5,
             map_dir: Optional[Path] = None) -> Path:
    """Where ``survey_scene.py`` writes a scene's slice for one altitude."""
    directory = Path(map_dir) if map_dir else DEFAULT_MAP_DIR
    return directory / f"{scene}_alt{int(round(altitude_m * 100)):04d}cm.npz"


def load(path: Path) -> Tuple[np.ndarray, float, np.ndarray]:
    """``(grid, resolution_m, origin_xy)``. ``-1`` unknown, ``0`` free, ``>0`` blocked."""
    data = np.load(Path(path))
    return data["grid"], float(data["resolution"]), data["origin"]


def largest_free_component(grid: np.ndarray) -> np.ndarray:
    """The biggest connected run of free cells — where flights can actually go.

    Free space outside it is unreachable, and counting it as flyable is what
    makes a coverage figure plateau below 100 % for no visible reason.

    The labelling is ``core.planning.environment.grid_regions`` -- the same one
    ``free_space_sampler`` draws missions from and the coverage trackers divide
    by. It used to be a private ``scipy.ndimage.label`` here, which is how a
    picture of "where flights can go" and the number measuring how much of it
    they went to end up disagreeing about the map.
    """
    from sparx_agency.core.planning.environment.grid_regions import connected_regions

    free = grid == 0
    regions = connected_regions(free, connectivity=4)
    return regions[0] if regions else np.zeros_like(free)


def render(scene: str, out_path: Path, altitude_m: float = 1.5,
           map_dir: Optional[Path] = None, show_reachable: bool = False,
           dpi: int = 160) -> Path:
    """Draw one scene's map and write it as a PNG. Returns the path written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    source = map_path(scene, altitude_m, map_dir)
    if not source.is_file():
        raise FileNotFoundError(
            f"no surveyed map at {source} -- survey it with survey_scene.py")
    grid, resolution, origin = load(source)
    height, width = grid.shape

    # 0 unknown, 1 free, 2 reachable-free, 3 occupied. Painted as one indexed
    # image rather than layered alpha, so the PNG stays crisp at any zoom.
    painted = np.where(grid < 0, 0, np.where(grid > 0, 3, 1)).astype(np.uint8)
    reachable_cells = 0
    if show_reachable:
        mask = largest_free_component(grid)
        painted[mask] = 2
        reachable_cells = int(mask.sum())

    extent = (origin[0], origin[0] + width * resolution,
              origin[1], origin[1] + height * resolution)
    span_x, span_y = extent[1] - extent[0], extent[3] - extent[2]

    figure, axis = plt.subplots(figsize=(max(6.0, span_x / 3.0),
                                         max(6.0, span_y / 3.0)), dpi=dpi)
    axis.imshow(painted, origin="lower", extent=extent, interpolation="nearest",
                cmap=ListedColormap([UNKNOWN, FREE, REACHABLE, OCCUPIED]),
                vmin=0, vmax=3)

    axis.plot([0.0], [0.0], "+", color=ORIGIN, markersize=14, markeredgewidth=2.0)
    axis.annotate("world origin (0, 0)", (0.0, 0.0), textcoords="offset points",
                  xytext=(10, 8), color=ORIGIN, fontsize=8)

    # A scale bar, because a reader should not have to do arithmetic on the
    # ticks to judge whether a gap is a doorway or a loading bay. Drawn in axes
    # fractions on a white backing, never in data coordinates: a filled black
    # rectangle sitting in the map reads as a wall, which is precisely the thing
    # this figure exists to show honestly.
    bar_m = 10.0 if span_x > 30 else 5.0
    bar_fraction = bar_m / span_x
    axis.add_patch(patches.Rectangle((0.04, 0.012), bar_fraction + 0.02, 0.028,
                                     transform=axis.transAxes, facecolor="white",
                                     edgecolor="#999", linewidth=0.6, zorder=5))
    axis.add_patch(patches.Rectangle((0.05, 0.022), bar_fraction, 0.004,
                                     transform=axis.transAxes, facecolor="black",
                                     zorder=6))
    axis.annotate(f"{bar_m:.0f} m", (0.05 + bar_fraction / 2, 0.030),
                  xycoords="axes fraction", ha="center", fontsize=8, zorder=6)

    free_cells = int((grid == 0).sum())
    cell_area = resolution ** 2
    subtitle = (f"{span_x:.1f} x {span_y:.1f} m   ·   {resolution * 100:.0f} cm cells   ·   "
                f"free {free_cells * cell_area:.0f} m²")
    if show_reachable:
        subtitle += f"   ·   reachable {reachable_cells * cell_area:.0f} m²"

    axis.set_title(f"{scene} — occupancy at {altitude_m:.2f} m\n{subtitle}",
                   fontsize=11)
    axis.set_xlabel("x (m, world)")
    axis.set_ylabel("y (m, world)")
    axis.set_aspect("equal")
    axis.grid(True, color="#00000018", linewidth=0.5)

    handles = [patches.Patch(facecolor=FREE, edgecolor="#999", label="free"),
               patches.Patch(facecolor=OCCUPIED, label="occupied"),
               patches.Patch(facecolor=UNKNOWN, label="unknown (never surveyed)")]
    if show_reachable:
        handles.insert(1, patches.Patch(facecolor=REACHABLE,
                                        label="reachable (flights come from here)"))
    axis.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.95)

    figure.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="office")
    parser.add_argument("--altitude", type=float, default=1.5)
    parser.add_argument("--map-dir", default=None)
    parser.add_argument("--out", default=None,
                        help="defaults to <scene>_map.png in the working directory")
    parser.add_argument("--reachable", action="store_true",
                        help="shade the largest connected free component")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)

    out = Path(args.out).expanduser() if args.out else Path(f"{args.scene}_map.png")
    written = render(args.scene, out, args.altitude,
                     Path(args.map_dir).expanduser() if args.map_dir else None,
                     args.reachable, args.dpi)
    print(f"[map] wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
