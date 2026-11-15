"""What the fake's camera sees from a pose: ray-cast depth in the repo's convention, and a grey image.

A **test rig, not a benchmark.** The fake's frames are geometry only, and the
geometry is pinned to the pixel, because a half-pixel slip in a fake's camera
is invisible until a mapper built on it is half a pixel wrong:

* **Depth is geometry, not appearance.** Depth is ray cast through the
  world's voxels by the Isaac stub's ``VoxelDepthCamera`` in the repo's
  convention (optical-frame ``z``, ``+inf`` beyond range). RGB is a constant
  mid-grey: the fake has no appearance, and its only policy that reaches a
  goal is privileged, so nothing reads the pixels.
* **Half a pixel, converted once.** The voxel camera casts pixel ``u``'s ray
  through image coordinate ``u + 0.5``; the doctrine puts pixel ``u``'s centre
  at ``u``. It is handed a principal point shifted by the same half pixel
  (:func:`ray_caster_intrinsics`), so every depth pixel lies exactly where
  ``backproject_depth`` reads it.
* **A frame someone else can write into.** Both images are read-only: a
  policy that edited a shared frame would corrupt the next one.
* **A camera the caster cannot render with.** An infinite depth range (the
  caster samples up to it) or a mount at or above the walls' top (inside the
  ceiling) is refused when the renderer is built.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.objnav.camera_geometry import (
    camera_position_world,
    world_T_camera_optical,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.falcon_pegasus.stub.voxel_camera import (
    VoxelDepthCamera,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.raster import voxel_grid
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld

#: The one value of every RGB pixel: the fake has no appearance.
RGB_GREY = 128


def ray_caster_intrinsics(intrinsics: Intrinsics) -> Intrinsics:
    """The intrinsics to hand ``VoxelDepthCamera`` so its pixels match the doctrine's.

    It casts pixel ``u``'s ray through image coordinate ``u + 0.5``; the
    doctrine (and ``backproject_depth``) puts pixel ``u``'s centre at ``u``.
    Shifting the principal point by the same half pixel makes its ray
    direction ``(u + 0.5 - (cx + 0.5)) / fx``, i.e. ``(u - cx) / fx``: exactly
    the doctrine's, with no change to the shared stub.
    """
    return dataclasses.replace(intrinsics, cx=intrinsics.cx + 0.5,
                               cy=intrinsics.cy + 0.5)


class FrameRenderer:
    """Renders the fake's observations: ray-cast depth and a constant grey RGB, both read-only.

    Every pixel is ray cast (the ray grid is the image), so no depth edge is
    smeared by the stub's coarse-grid expansion.

    Args:
        world: The building.
        camera: The agent's camera; ``max_depth_m`` must be finite (the ray
            caster samples up to it) and the mount below the walls' top.

    Raises:
        ObjNavError: On a camera the ray caster cannot render the world with.
    """

    def __init__(self, world: GridWorld, camera: CameraSpec) -> None:
        _check_camera(world, camera)
        k = camera.intrinsics
        voxels, origin, resolution = voxel_grid(world)
        self._depth_camera = VoxelDepthCamera(
            voxels, origin, resolution, ray_caster_intrinsics(k),
            ray_shape=(k.width, k.height), near_m=camera.min_depth_m,
            far_m=camera.max_depth_m)
        self._camera = camera
        self._rgb = np.full((k.height, k.width, 3), RGB_GREY, dtype=np.uint8)
        self._rgb.setflags(write=False)

    @property
    def camera(self) -> CameraSpec:
        """The camera every frame is rendered with."""
        return self._camera

    def observe(self, pose: AgentPose, target_category: str,
                step: int) -> ObjNavObservation:
        """The observation the agent receives at ``pose``.

        Args:
            pose: The agent's pose; the camera sits ``camera.height_m`` above
                it, pitched by ``pose.camera_pitch``.
            target_category: The running episode's target.
            step: Actions taken so far this episode.

        Returns:
            The grey RGB and the ray-cast depth (optical ``z`` in metres,
            ``inf`` where nothing lies within ``max_depth_m``), both
            read-only, with the pose, camera, target and step.
        """
        depth = self._depth_camera.render(
            camera_position_world(pose, self._camera),
            world_T_camera_optical(pose, self._camera)[:3, :3])
        depth.setflags(write=False)
        return ObjNavObservation(
            rgb=self._rgb, depth_m=depth, pose=pose, camera=self._camera,
            target_category=target_category, step=step)


def _check_camera(world: GridWorld, camera: CameraSpec) -> None:
    """Refuse a camera the ray caster cannot render ``world`` with."""
    if not math.isfinite(camera.max_depth_m):
        raise ObjNavError("the fake's ray caster samples up to max_depth_m, "
                          "so it must be finite, got %r" % (camera.max_depth_m,))
    if camera.height_m >= world.wall_height_m:
        raise ObjNavError(
            "the camera sits %.2f m up, at or above the %.2f m walls: it "
            "would render from inside the ceiling"
            % (camera.height_m, world.wall_height_m))
