"""Fly around a surveyed voxel map in Open3D. No container, no Isaac Sim.

The map is a self-contained ``.npz``, so once a scene has been surveyed nothing
about looking at it needs a simulator. This opens it in an interactive window
where you can orbit, zoom, slice the ceiling away and — if you point it at a
campaign — see the routes that were actually flown through the building.

::

    .venv/bin/python sparx_agency/tasks/planning/sim_flight_recording/view_voxel_map.py \\
        --scene office --max-z 2.2 --recordings ~/sim_flight_recordings

**Clip the ceiling.** It is the largest surface in the building and it sits
between you and everything you wanted to see; `[` and `]` move the cut while the
window is open, which is the fastest way to understand a floor plan.

Points, not cubes, by default: ``office`` is 1.8 million occupied voxels, and
1.8 million cube meshes do not orbit at an interactive frame rate. ``--cubes``
switches to a true voxel grid, which is worth it on a small map or a thin slice.

Follows the conventions of ``tasks/planning/3D_planning``'s viewers — same
``VisualizerWithKeyCallback`` shape, same coordinate frame, same key idiom.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from sparx_agency.tasks.planning.sim_flight_recording.voxel_export import (
    clip_height, height_colours,
)

# GLFW key codes, as used by tasks/planning/3D_planning/interaction.py. R (reset
# the view) is deliberately absent: Open3D binds it itself, and re-registering it
# would replace that with a worse version.
KEY = {"[": 91, "]": 93, "C": 67, "P": 80, "H": 72}
CLIP_STEP_M = 0.2
FLOWN_COLOUR = (1.0, 0.55, 0.0)     # amber, matching inspect_recording's plan view
START_COLOUR = (0.1, 0.9, 0.1)
GOAL_COLOUR = (0.9, 0.1, 0.1)

HELP = """
  drag              orbit          scroll   zoom          ctrl+drag  pan
  [ / ]             lower / raise the ceiling cut
  C                 cycle: points -> cubes -> points
  P                 print the current cut height
  R                 reset the view
  H                 this help                 Q or Esc   quit
"""


def prefer_x11() -> bool:
    """Make Open3D's GLFW use X11 rather than Wayland, if it can.

    Open3D 0.19's GLFW build cannot get a GL context on a Wayland session --
    ``create_window`` returns False after ``Failed to initialize GLEW``, and
    every later call then fails on a ``None`` render option rather than saying
    what went wrong. Unsetting ``WAYLAND_DISPLAY`` before Open3D is imported
    sends GLFW down the X11 path instead, which works through XWayland.

    Must be called before the first ``import open3d`` anywhere in the process.

    Returns:
        True if the environment was changed.
    """
    if not os.environ.get("WAYLAND_DISPLAY"):
        return False
    if not os.environ.get("DISPLAY"):
        print("WARNING: this is a Wayland session with no DISPLAY, so there is no "
              "X11 fallback and Open3D will fail to open a window.", flush=True)
        return False
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["XDG_SESSION_TYPE"] = "x11"
    return True


def voxel_geometry(points: np.ndarray, voxel_size: float, cubes: bool):
    """Build the geometry for a set of occupied voxel centres.

    Args:
        points: ``(N, 3)`` world-frame voxel centres, metres.
        voxel_size: Edge length, metres. Only meaningful for ``cubes``.
        cubes: Render true voxel cubes rather than points. Far prettier and far
            slower -- a million cube meshes will not orbit smoothly.

    Returns:
        An Open3D geometry.
    """
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(
        height_colours(points).astype(np.float64) / 255.0)
    if not cubes:
        return cloud
    return o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, voxel_size)


def flight_geometries(recordings_dir: Path, altitude_offset: float = 0.0) -> list:
    """Line sets and end markers for every clean flight in a campaign.

    Args:
        recordings_dir: A campaign directory, or one recording.
        altitude_offset: Lift the drawn paths by this much so they do not
            z-fight with the floor voxels, metres.

    Returns:
        A list of Open3D geometries. Empty if there is nothing to draw.
    """
    import open3d as o3d

    from sparx_agency.tasks.planning.sim_flight_recording.inspect_recording import (
        find_recordings,
    )
    from sparx_agency.tasks.planning.vlas.common.finetune.datasets.recording import (
        load_recording,
    )

    geometries = []
    for path in find_recordings(Path(recordings_dir)):
        recording = load_recording(path)
        poses = recording.poses
        if poses.shape[1] < 5 or poses.shape[0] < 2:
            continue
        track = np.stack([poses[:, 1], poses[:, 2], poses[:, 4] + altitude_offset], axis=1)

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(track)
        lines.lines = o3d.utility.Vector2iVector(
            np.stack([np.arange(len(track) - 1), np.arange(1, len(track))], axis=1))
        lines.colors = o3d.utility.Vector3dVector(
            np.tile(FLOWN_COLOUR, (len(track) - 1, 1)))
        geometries.append(lines)

        for point, colour in ((track[0], START_COLOUR), (track[-1], GOAL_COLOUR)):
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.18)
            marker.translate(point)
            marker.paint_uniform_color(colour)
            marker.compute_vertex_normals()
            geometries.append(marker)
    return geometries


class VoxelMapViewer:
    """An interactive window over one voxel map, with a movable ceiling cut.

    Args:
        grid: The surveyed :class:`VoxelGrid3D`.
        max_z: Initial ceiling cut, metres. None starts with the whole map,
            which on an indoor scene means looking at a roof.
        min_z: Floor cut, metres.
        cubes: Start in cube rendering rather than points.
        extras: Geometries to add once and leave alone -- flight paths, markers.
    """

    def __init__(self, grid, max_z=None, min_z=None, cubes: bool = False,
                 extras=None):
        self.grid = grid
        self.points = grid.occupied_points()
        self.max_z = max_z
        self.min_z = min_z
        self.cubes = cubes
        self.extras = list(extras or [])
        self._geometry = None
        self._vis = None

    def _rebuild(self, vis) -> bool:
        """Swap in geometry for the current cut, keeping the camera where it is.

        Open3D has no way to edit a geometry's point set in place here, so the
        cut is applied by replacing it. Removing a geometry resets the view, so
        the camera is saved and restored around the swap -- without that, every
        keypress would throw you back to the default pose.
        """
        control = vis.get_view_control()
        camera = control.convert_to_pinhole_camera_parameters()

        if self._geometry is not None:
            vis.remove_geometry(self._geometry, reset_bounding_box=False)
        visible = clip_height(self.points, self.min_z, self.max_z)
        if len(visible) == 0:
            visible = self.points[:1]
        self._geometry = voxel_geometry(visible, self.grid.resolution, self.cubes)
        vis.add_geometry(self._geometry, reset_bounding_box=False)

        control.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)
        print(f"  cut at {self.max_z if self.max_z is not None else float('inf'):.1f} m"
              f" -- {len(visible)} of {len(self.points)} voxels, "
              f"{'cubes' if self.cubes else 'points'}", flush=True)
        return True

    def _move_cut(self, delta: float):
        def handler(vis):
            ceiling = self.grid.origin_z + self.grid.depth * self.grid.resolution
            current = self.max_z if self.max_z is not None else ceiling
            self.max_z = float(np.clip(current + delta, self.grid.origin_z, ceiling))
            return self._rebuild(vis)
        return handler

    def _toggle_cubes(self, vis) -> bool:
        self.cubes = not self.cubes
        return self._rebuild(vis)

    def _print_cut(self, vis) -> bool:
        print(f"  ceiling cut: {self.max_z} m   floor cut: {self.min_z} m", flush=True)
        return False

    def build(self, vis) -> None:
        """Populate a visualizer with the map, the frame and any extras.

        Split out of :meth:`run` so the same scene can be rendered to a file
        without opening an interactive window.
        """
        import open3d as o3d

        # Extras first, so the bounding box covers the whole scene; the voxels
        # are added last and their later rebuilds must not reset the view.
        for geometry in self.extras:
            vis.add_geometry(geometry)
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
        visible = clip_height(self.points, self.min_z, self.max_z)
        self._geometry = voxel_geometry(
            visible if len(visible) else self.points, self.grid.resolution, self.cubes)
        vis.add_geometry(self._geometry)

        options = vis.get_render_option()
        options.background_color = np.array([0.08, 0.09, 0.11])
        options.point_size = 2.5

    def screenshot(self, path: Path, width: int = 1400, height: int = 900) -> Path:
        """Render the scene straight to a PNG, with no interactive window.

        For looking at a map over SSH, or putting one in a document.

        Args:
            path: Destination PNG.
            width: Image width, pixels.
            height: Image height, pixels.

        Returns:
            The path written.

        Raises:
            RuntimeError: If Open3D could not get a GL context at all.
        """
        import cv2
        import open3d as o3d

        vis = o3d.visualization.Visualizer()
        if not vis.create_window(window_name="offscreen", width=width, height=height,
                                 visible=False):
            raise RuntimeError(
                "Open3D could not create a GL context. On a Wayland session it "
                "needs the X11 fallback -- see prefer_x11()."
            )
        self.build(vis)
        control = vis.get_view_control()
        control.set_front([0.45, -0.75, 0.48])
        control.set_up([0.0, 0.0, 1.0])
        control.set_lookat(self._geometry.get_center())
        control.set_zoom(0.5)
        for _ in range(8):
            vis.poll_events()
            vis.update_renderer()
        image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        vis.destroy_window()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor((image * 255).astype(np.uint8),
                                            cv2.COLOR_RGB2BGR))
        return path

    def run(self, title: str) -> None:
        """Open the window and block until it is closed."""
        import open3d as o3d

        vis = o3d.visualization.VisualizerWithKeyCallback()
        if not vis.create_window(window_name=title, width=1400, height=900):
            raise RuntimeError(
                "Open3D could not create a GL context. On a Wayland session it "
                "needs the X11 fallback -- see prefer_x11()."
            )
        self._vis = vis

        self.build(vis)

        vis.register_key_callback(KEY["["], self._move_cut(-CLIP_STEP_M))
        vis.register_key_callback(KEY["]"], self._move_cut(+CLIP_STEP_M))
        vis.register_key_callback(KEY["C"], self._toggle_cubes)
        vis.register_key_callback(KEY["P"], self._print_cut)
        vis.register_key_callback(KEY["H"], lambda v: (print(HELP, flush=True), False)[1])

        print(HELP, flush=True)
        vis.run()
        vis.destroy_window()


def main() -> int:
    """Open a surveyed voxel map in an interactive Open3D window."""
    from sparx_agency.core.planning.environment import load_voxel_grid
    from sparx_agency.robots.PEGASUS.adapters.scene_map import voxel_map_path

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--scene", default="office", help="a surveyed scene name")
    ap.add_argument("--map", type=Path, default=None,
                    help="a voxel .npz to open instead of the scene's own")
    ap.add_argument("--max-z", type=float, default=None,
                    help="start with the ceiling cut here, metres. Almost always "
                         "worth setting -- '[' and ']' move it once open")
    ap.add_argument("--min-z", type=float, default=None,
                    help="hide everything below this height, metres")
    ap.add_argument("--cubes", action="store_true",
                    help="render true voxel cubes. Prettier, and slow above a few "
                         "hundred thousand voxels")
    ap.add_argument("--recordings", type=Path, default=None,
                    help="a campaign directory whose flown paths to draw over the map")
    ap.add_argument("--screenshot", type=Path, default=None,
                    help="render straight to this PNG and exit, without opening a "
                         "window. For looking at a map over SSH")
    args = ap.parse_args()

    prefer_x11()

    path = args.map or voxel_map_path(args.scene)
    grid, metadata = load_voxel_grid(path)
    print(f"{path.name}: {grid}", flush=True)
    print(f"   {metadata.get('occupied', 0)} occupied voxels at "
          f"{metadata.get('resolution_m')} m", flush=True)

    extras = []
    if args.recordings is not None:
        extras = flight_geometries(args.recordings)
        print(f"   {len(extras) // 3} flown route(s) from {args.recordings}", flush=True)

    viewer = VoxelMapViewer(grid, max_z=args.max_z, min_z=args.min_z,
                            cubes=args.cubes, extras=extras)
    if args.screenshot is not None:
        print(f"wrote {viewer.screenshot(args.screenshot)}", flush=True)
        return 0
    viewer.run(f"{args.scene} voxel map")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
