"""
Semantic type definitions for spatial reasoning.

This module defines semantic features that behaviors and planners can
reason about: portals (doorways/thresholds) and regions (rooms/zones).

Types:
    - Portal2D / Portal3D: Traversable boundaries
    - Region2D / Region3D: Bounded spatial areas

Note:
    These are external features provided by perception or semantic mapping.
    Behaviors do NOT detect these features directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from .geometry import Pose2D, Pose3D

SemanticId = str


@dataclass(frozen=True, slots=True)
class Portal2D:
    """
    A 2D traversable boundary (doorway, gate, threshold).

    Attributes:
        id: Unique identifier.
        center: Portal center pose in world frame.
        normal_yaw: Outward-facing direction (radians), or None if unknown.
        width_m: Opening width in meters, or None if unknown.
        tags: Free-form metadata (e.g., {"type": "doorway", "room": "kitchen"}).
    """

    id: SemanticId
    center: Pose2D
    normal_yaw: Optional[float] = None
    width_m: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Portal3D:
    """
    A 3D traversable boundary.

    Attributes:
        id: Unique identifier.
        center: Portal center pose in world frame.
        normal_yaw: Outward-facing direction (radians), or None if unknown.
        size_m: Opening dimensions (width, height) in meters, or None.
        tags: Free-form metadata.
    """

    id: SemanticId
    center: Pose3D
    normal_yaw: Optional[float] = None
    size_m: Optional[Tuple[float, float]] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Region2D:
    """
    A 2D semantic region (room, zone, area).

    Attributes:
        id: Unique identifier.
        boundary: Region geometry (polygon, mask, etc.), or None.
        portals: IDs of portals connecting to this region.
        tags: Free-form metadata (e.g., {"name": "kitchen", "floor": "1"}).
    """

    id: SemanticId
    boundary: Optional[Any] = None
    portals: Sequence[SemanticId] = field(default_factory=tuple)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Region3D:
    """
    A 3D semantic region (room, volume).

    Attributes:
        id: Unique identifier.
        volume: Region geometry (mesh, voxels, etc.), or None.
        portals: IDs of portals connecting to this region.
        tags: Free-form metadata.
    """

    id: SemanticId
    volume: Optional[Any] = None
    portals: Sequence[SemanticId] = field(default_factory=tuple)
    tags: Dict[str, str] = field(default_factory=dict)