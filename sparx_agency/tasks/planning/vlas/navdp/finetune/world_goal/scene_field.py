"""The surveyed map, resident on the GPU and differentiably sampled.

This is the piece that lets the loss know about geometry the camera cannot see.
The whole office is 745 x 307 cells -- 915 kB as float32 -- so the map that
supervises safety simply lives on the device for the entire run, and each
training sample carries only its ``(x, y, yaw)``. Transforming predicted
waypoints into world coordinates and reading the signed ESDF there is part of the
objective, not part of the dataset.

That matters because the gradient flows back through the lookup: the hinge on
``clearance < margin`` pushes a waypoint away from the wall it is approaching,
including a wall several metres round a corner that no depth frame contains.

Torch only.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


class SceneField:
    """One scene's signed ESDF on the device."""

    def __init__(self, sdf: np.ndarray, resolution: float, origin_x: float,
                 origin_y: float, device: str = "cuda") -> None:
        self.sdf = torch.as_tensor(np.ascontiguousarray(sdf), dtype=torch.float32,
                                   device=device)[None, None]
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.height, self.width = int(sdf.shape[0]), int(sdf.shape[1])

    def sample(self, world_xy: torch.Tensor) -> torch.Tensor:
        """``(B, N, 2)`` world points -> ``(B, N)`` signed clearance, metres.

        Folds the query batch into ``grid_sample``'s height axis so the map is
        never expanded or copied, and stays differentiable with respect to the
        query coordinates.

        The half-cell offset matches ``OccupancyGrid2D.grid_to_world``
        (``x = (gx + 0.5) * res + origin``), which is also what the numpy
        :func:`~.scene.bilinear_sample` used during label generation assumes --
        so a clearance measured offline and the same clearance measured on the
        GPU agree.

        Queries outside the map read the border value. The surveyed border is
        unmapped space, which the ESDF treats as solid, so leaving the map is
        penalised rather than ignored.
        """
        col = (world_xy[..., 0] - self.origin_x) / self.resolution - 0.5
        row = (world_xy[..., 1] - self.origin_y) / self.resolution - 0.5
        gx = 2.0 * col / max(self.width - 1, 1) - 1.0
        gy = 2.0 * row / max(self.height - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)        # (1, B, N, 2)
        sampled = F.grid_sample(self.sdf, grid.to(self.sdf.dtype), mode="bilinear",
                                padding_mode="border", align_corners=True)
        return sampled[0, 0]


class SceneFields:
    """Several scenes at once, indexed by the per-sample scene id."""

    def __init__(self, fields: Sequence[SceneField]) -> None:
        self.fields = list(fields)

    def sample(self, world_xy: torch.Tensor, scene_ids: torch.Tensor) -> torch.Tensor:
        """Sample each row from the scene its sample came from."""
        if len(self.fields) == 1:
            return self.fields[0].sample(world_xy)
        out = world_xy.new_zeros(world_xy.shape[:2])
        for scene_id in torch.unique(scene_ids):
            mask = scene_ids == scene_id
            out[mask] = self.fields[int(scene_id)].sample(world_xy[mask])
        return out
