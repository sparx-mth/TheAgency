"""Render a depth image by raycasting a ground-truth voxel grid.

The surveyed map in ``robots/PEGASUS/maps`` is the same building Isaac Sim
renders, measured once at 10 cm by Isaac's own occupancy generator. Casting rays
through it produces depth images that are geometrically the same as the ones the
real camera produces -- without a GPU, without Kit, and about two hundred times
faster to start.

That is what makes the stub aircraft useful: everything between the depth image
and the trajectory can be exercised in seconds instead of the five minutes a
Kit boot and a PX4 warm-up cost, and a FALCON configuration problem shows up as
a FALCON configuration problem rather than as "the run did not work".

The depth convention is the one FALCON and Isaac both use: the **perpendicular
distance to the image plane**, i.e. the optical-frame z of the surface, not the
length of the ray. Sampling along ``Z * d`` where ``d = ((u-cx)/fx, (v-cy)/fy,
1)`` makes the sample parameter itself that distance, so no conversion is
needed and none can be forgotten.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

OCCUPIED = 1


class VoxelDepthCamera:
    """Casts rays through an occupancy voxel grid to make depth images.

    Rays are cast on a coarse grid and the result is expanded to the full image
    size, because the cost is linear in ray count and FALCON only reads every
    fourth pixel anyway (``skip_pixel: 4``). The default is close to that
    sampling, so almost nothing is lost and a frame costs tens of milliseconds
    rather than seconds.

    Args:
        voxels: ``(nz, ny, nx)`` int8 grid: 1 occupied, 0 free, -1 unknown.
        origin: World ``(x, y, z)`` of voxel ``(0, 0, 0)``'s corner.
        resolution: Voxel edge length, metres.
        intrinsics: The camera being simulated.
        ray_shape: ``(width, height)`` of the ray grid. None picks a fifth of the
            image in each axis, which is a little coarser than FALCON's own pixel
            skip of 4 and costs 26 ms a frame instead of 110 for a result that
            measured identical on the office map.
        near_m: Closest surface a ray can report.
        far_m: Furthest. Beyond it a ray reports "nothing there" (``inf``), which
            the depth codec turns into carved free space rather than an obstacle.
        step_m: Sample spacing along each ray. Under one voxel, so a wall cannot
            be stepped over; three quarters rather than a half because the extra
            samples cost 2x the frame time and changed nothing measurable.

    Raises:
        ValueError: If the grid is not three-dimensional.
    """

    def __init__(self, voxels: np.ndarray, origin, resolution: float, intrinsics,
                 ray_shape: Optional[Tuple[int, int]] = None, near_m: float = 0.2,
                 far_m: float = 8.0, step_m: Optional[float] = None):
        voxels = np.asarray(voxels)
        if voxels.ndim != 3:
            raise ValueError("expected a (nz, ny, nx) voxel grid, got shape %r"
                             % (voxels.shape,))
        self._occupied = voxels == OCCUPIED
        self._origin = np.asarray(origin, dtype=np.float32)
        self._resolution = float(resolution)
        self.intrinsics = intrinsics
        self._near = float(near_m)
        self._far = float(far_m)

        width, height = ray_shape or (max(intrinsics.width // 5, 8),
                                      max(intrinsics.height // 5, 8))
        self._ray_shape = (int(width), int(height))
        step = step_m if step_m is not None else self._resolution * 0.75
        self._depths = np.arange(self._near, self._far, step, dtype=np.float32)
        self._directions = self._ray_directions()

    def _ray_directions(self) -> np.ndarray:
        """Unnormalised optical-frame ray directions, one per coarse pixel.

        ``z`` is exactly 1, which is what makes the sample parameter the
        perpendicular distance to the image plane.
        """
        width, height = self._ray_shape
        # Sample the pixel centres of the coarse grid mapped onto the full image,
        # so the rendered field of view is the camera's, not a cropped version.
        us = (np.arange(width, dtype=np.float32) + 0.5) * (self.intrinsics.width / width)
        vs = (np.arange(height, dtype=np.float32) + 0.5) * (self.intrinsics.height / height)
        grid_u, grid_v = np.meshgrid(us, vs)
        return np.stack([
            (grid_u - self.intrinsics.cx) / self.intrinsics.fx,
            (grid_v - self.intrinsics.cy) / self.intrinsics.fy,
            np.ones_like(grid_u),
        ], axis=-1).reshape(-1, 3).astype(np.float32)

    def render(self, translation, rotation_world_optical) -> np.ndarray:
        """Depth in metres, as the camera at this pose would see it.

        Args:
            translation: World ``(x, y, z)`` of the optical centre.
            rotation_world_optical: ``(3, 3)`` world-from-optical rotation.

        Returns:
            An ``(image_height, image_width)`` float32 array, metres, with
            ``inf`` where no surface was hit inside ``far_m``.
        """
        origin = np.asarray(translation, dtype=np.float32).reshape(1, 3)
        rotation = np.asarray(rotation_world_optical, dtype=np.float32)
        # (rays, 3) world directions; scaling by the sample depth then gives the
        # world point whose optical-frame z is that depth.
        world_dirs = self._directions.dot(rotation.T)

        # (rays, samples, 3). The one big allocation, and the reason the ray grid
        # is coarse: at 160x120 rays and 160 samples this is 37 MB of float32.
        points = origin[:, None, :] + world_dirs[:, None, :] * self._depths[None, :, None]
        indices = np.floor((points - self._origin) / self._resolution).astype(np.int32)

        shape = self._occupied.shape                     # (nz, ny, nx)
        ix, iy, iz = indices[..., 0], indices[..., 1], indices[..., 2]
        inside = ((ix >= 0) & (ix < shape[2]) & (iy >= 0) & (iy < shape[1])
                  & (iz >= 0) & (iz < shape[0]))
        hit = np.zeros(inside.shape, dtype=bool)
        hit[inside] = self._occupied[iz[inside], iy[inside], ix[inside]]

        # argmax on a boolean gives the first True, or 0 when there is none --
        # which is why `any` is checked separately rather than inferred from it.
        first = np.argmax(hit, axis=1)
        found = hit.any(axis=1)
        depth = np.where(found, self._depths[first], np.float32(np.inf))

        width, height = self._ray_shape
        coarse = depth.reshape(height, width)
        return self._expand(coarse)

    def _expand(self, coarse: np.ndarray) -> np.ndarray:
        """Nearest-neighbour expansion of the ray grid to the full image size.

        Deliberately nearest rather than interpolated: an interpolated depth
        edge invents a surface halfway between a wall and the space beyond it,
        and FALCON would fuse that invented surface as an obstacle.
        """
        height, width = self.intrinsics.height, self.intrinsics.width
        rows = np.minimum((np.arange(height) * coarse.shape[0]) // height,
                          coarse.shape[0] - 1)
        cols = np.minimum((np.arange(width) * coarse.shape[1]) // width,
                          coarse.shape[1] - 1)
        return coarse[rows][:, cols]
