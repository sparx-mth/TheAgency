"""Build a ground-truth 2D occupancy map from a Gazebo world's own geometry.

Run it, get three files: a nav2 ``map_server`` PGM + YAML for ROS tooling, and
an ``.npz`` carrying an :class:`OccupancyGrid2D` for this repo's planners.

    python -m sparx_agency.tasks.mapping.gazebo_world_occupancy.build_map \\
        --world /path/to/hospital.world --output-dir /path/to/maps

The map is **ground truth**: every cell is either occupied or free, and there
are no unknown cells, because this is computed from the collision meshes the
simulator itself uses rather than accumulated from what a robot happened to
see. Anything not hit by geometry inside the height band is free -- including
the space outside the building. That is honest for a reference map and wrong
for an exploration map; do not confuse the two.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from sparx_agency.core.planning.environment.occupancy_io import (
    occupancy_from_mask,
    save_occupancy_grid,
)
from sparx_agency.tasks.mapping.gazebo_world_occupancy import nav2_map, resource_paths
from sparx_agency.tasks.mapping.gazebo_world_occupancy.model_lookup import (
    is_gazebo_builtin,
)
from sparx_agency.tasks.mapping.gazebo_world_occupancy.scene_raster import (
    grid_spec_for,
    measure_extent,
    rasterise_scene,
)
from sparx_agency.tasks.mapping.gazebo_world_occupancy.sdf_scene import load_scene

DEFAULT_RESOLUTION = 0.05
DEFAULT_Z_MIN = 0.30
DEFAULT_Z_MAX = 2.00
DEFAULT_MARGIN = 1.0
DEFAULT_SKIP = ("drone",)
FRAME_ID = "map"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="build_map",
        description="Ground-truth 2D occupancy map from a Gazebo world's geometry.",
    )
    parser.add_argument(
        "--world", required=True,
        help="Path to a .world file, or a bare name such as 'hospital'.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write the artifacts to.",
    )
    parser.add_argument("--name", default=None, help="Basename (default: the world's).")
    parser.add_argument(
        "--resolution", type=float, default=DEFAULT_RESOLUTION,
        help="Metres per cell (default: %(default)s).",
    )
    parser.add_argument(
        "--z-min", type=float, default=DEFAULT_Z_MIN,
        help="Bottom of the height band, metres (default: %(default)s).",
    )
    parser.add_argument(
        "--z-max", type=float, default=DEFAULT_Z_MAX,
        help="Top of the height band, metres (default: %(default)s).",
    )
    parser.add_argument(
        "--margin", type=float, default=DEFAULT_MARGIN,
        help="Free space added around the geometry, metres (default: %(default)s).",
    )
    parser.add_argument(
        "--search-path", action="append", default=[], metavar="DIR",
        help="Directory model:// URIs resolve against. Repeatable; searched "
             "before the world's sibling models/ and fuel_models/ and before "
             "GAZEBO_MODEL_PATH.",
    )
    parser.add_argument(
        "--skip", action="append", default=None, metavar="SUBSTRING",
        help="Exclude models whose name contains this, case-insensitive. "
             "Repeatable. Defaults to 'drone', to keep the robot out of its "
             "own map.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail if any model:// include does not resolve, instead of "
             "reporting it. Gazebo's own sun and ground_plane are exempt; an "
             "unresolved mesh file is fatal either way.",
    )
    parser.add_argument(
        "--preview", default=None, metavar="PNG",
        help="Also write a PNG preview of the map here.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build one map. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    extra = [Path(p) for p in args.search_path]
    world_path = resource_paths.resolve_world(args.world, extra)
    search_paths = resource_paths.search_paths_for(world_path, extra)
    skip = tuple(args.skip) if args.skip is not None else DEFAULT_SKIP

    print("world:        %s" % (world_path,))
    for path in search_paths:
        print("search path:  %s" % (path,))

    scene = load_scene(world_path, search_paths, skip_substrings=skip)
    _report_scene(scene, strict=args.strict)

    extent = measure_extent(scene.instances, args.z_min, args.z_max, args.margin)
    spec = grid_spec_for(extent, args.resolution)
    occupied, triangles = rasterise_scene(
        scene.instances, spec, args.z_min, args.z_max
    )

    name = args.name or world_path.stem
    output_dir = Path(args.output_dir)
    paths = _write_artifacts(
        output_dir, name, occupied, spec, scene, extent, triangles, args, search_paths
    )
    if args.preview:
        _write_preview(args.preview, occupied)
        paths.append(Path(args.preview))

    _report_map(occupied, spec, extent, triangles, paths)
    return 0


def _report_scene(scene, strict: bool) -> None:
    """Print what the world parse found, and fail on what is not routine."""
    print(
        "geometry:     %d instances (%d skipped by name)"
        % (len(scene.instances), len(scene.skipped_models))
    )
    if scene.ignored_geometry:
        kinds = sorted({kind for _model, kind in scene.ignored_geometry})
        print("ignored:      %s geometry (not mappable)" % (", ".join(kinds),))
    if scene.missing_meshes:
        # Fatal without --strict, because there is no benign version of it:
        # the link is there, it says it is shaped like that file, and the file
        # is gone. The map comes out missing a wall and looking complete.
        raise FileNotFoundError(
            "unresolved mesh files: %s"
            % (", ".join(sorted(set(scene.missing_meshes))),)
        )
    _report_missing_models(scene.missing_models, strict)


def _report_missing_models(missing: Sequence[str], strict: bool) -> None:
    """Report unresolved includes, holding Gazebo's built-ins aside.

    Every world includes ``sun`` and ``ground_plane`` and neither resolves off
    the world's own models directory, so folding them in with the rest made
    ``--strict`` refuse the shipped hospital world and left the warning that
    covers a genuinely missing model reading as routine.

    Args:
        missing: Unresolved ``model://`` URIs, in any order.
        strict: Whether a non-built-in miss is fatal.

    Raises:
        FileNotFoundError: Under ``strict``, if anything but a built-in is
            unresolved.
    """
    unique = sorted(set(missing))
    builtin = [uri for uri in unique if is_gazebo_builtin(uri)]
    unknown = [uri for uri in unique if not is_gazebo_builtin(uri)]
    if builtin:
        print(
            "built-ins:    %s not on the model path (a light and an infinite "
            "plane; neither is mappable)" % (", ".join(builtin),)
        )
    if not unknown:
        return
    message = "unresolved model:// uris: %s" % (", ".join(unknown),)
    if strict:
        raise FileNotFoundError(message)
    print("WARNING:      %s" % (message,))


def _write_artifacts(
    output_dir, name, occupied, spec, scene, extent, triangles, args, search_paths
) -> List[Path]:
    """Write the PGM, the YAML and the .npz. Returns the paths written."""
    pgm_path, yaml_path = nav2_map.write_nav2_map(
        output_dir, name, occupied,
        resolution=spec.resolution, origin_x=spec.origin_x, origin_y=spec.origin_y,
    )
    grid = occupancy_from_mask(
        occupied, spec.resolution, spec.origin_x, spec.origin_y, frame_id=FRAME_ID
    )
    npz_path = save_occupancy_grid(
        Path(output_dir) / (name + ".npz"),
        grid,
        metadata=_metadata(scene, spec, extent, triangles, args, search_paths),
    )
    return [pgm_path, yaml_path, npz_path]


def _metadata(scene, spec, extent, triangles, args, search_paths) -> dict:
    """Provenance carried inside the .npz, so a map can be traced to its world."""
    return {
        "source": "sparx_agency.tasks.mapping.gazebo_world_occupancy.build_map",
        "world_path": str(scene.world_path),
        "world_sha256": _sha256(scene.world_path),
        "search_paths": [str(p) for p in search_paths],
        "skipped_models": sorted(set(scene.skipped_models)),
        "unresolved_models": sorted(set(scene.missing_models)),
        "resolution": float(spec.resolution),
        "origin": [float(spec.origin_x), float(spec.origin_y), 0.0],
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
        "margin": float(args.margin),
        "instance_count": int(extent.instance_count),
        "triangle_count": int(triangles),
        "frame_id": FRAME_ID,
        "unknown_cells": False,
    }


def _write_preview(path, occupied: np.ndarray) -> None:
    """Write a PNG of the map, in image row order, for eyeballing."""
    import cv2

    image = nav2_map.to_pgm_image(occupied)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise IOError("could not write preview to %s" % (target,))


def _report_map(occupied, spec, extent, triangles, paths) -> None:
    """Print the summary a human reads to decide whether the map is sane."""
    rows, cols = np.nonzero(occupied)
    print("triangles:    %d rasterised" % (triangles,))
    print(
        "bounds:       x [%.2f, %.2f]  y [%.2f, %.2f]  (margin included)"
        % (extent.min_x, extent.max_x, extent.min_y, extent.max_y)
    )
    print(
        "grid:         %d x %d cells at %.3f m, origin (%.3f, %.3f)"
        % (spec.width, spec.height, spec.resolution, spec.origin_x, spec.origin_y)
    )
    if rows.size:
        print(
            "occupied bbox: x [%.2f, %.2f]  y [%.2f, %.2f]"
            % (
                spec.origin_x + cols.min() * spec.resolution,
                spec.origin_x + (cols.max() + 1) * spec.resolution,
                spec.origin_y + rows.min() * spec.resolution,
                spec.origin_y + (rows.max() + 1) * spec.resolution,
            )
        )
    print(
        "occupied:     %d cells, %.2f%% of the map"
        % (int(occupied.sum()), 100.0 * float(occupied.mean()))
    )
    for path in paths:
        print("wrote:        %s" % (path,))


def _sha256(path) -> str:
    """Hex digest of a file, so the map records exactly which world it came from."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
