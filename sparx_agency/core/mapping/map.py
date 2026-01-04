
from __future__ import annotations

import gzip
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from sparx_agency.core.common.types import Coord2D, Coord3D, Index2D, Index3D, Number


# ----------------------------
# Base: Map
# ----------------------------

@dataclass
class Map(ABC):
    """
    Abstract base class for spatial maps.

    Provides portable JSON serialization via save/load.
    Layers must be JSON-serializable (numbers, strings,
    lists/dicts). For advanced storage (NumPy, torch, etc.),
    subclasses can override 'serialize_layers' / 'deserialize_layers'.
    """

    frame_id: str = "map"
    origin: Union[Coord2D, Coord3D] = (0.0, 0.0)
    resolution: Union[Number, Tuple[Number, ...]] = 1.0
    bounds: Optional[Tuple[Union[Coord2D, Coord3D], Union[Coord2D, Coord3D]]] = None
    layers: Dict[str, Any] = field(default_factory=dict)

    # ---------- Core abstract API ----------

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the map: 2 for Map2D, 3 for Map3D, etc."""
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Shape (number of cells per axis)."""
        raise NotImplementedError

    @abstractmethod
    def world_to_index(self, coord: Union[Coord2D, Coord3D]) -> Union[Index2D, Index3D]:
        """Convert a world-space coordinate to a discrete map index."""
        raise NotImplementedError

    @abstractmethod
    def index_to_world(self, index: Union[Index2D, Index3D]) -> Union[Coord2D, Coord3D]:
        """Convert a discrete map index to a world-space coordinate."""
        raise NotImplementedError

    @abstractmethod
    def in_bounds(self, coord: Union[Coord2D, Coord3D]) -> bool:
        """Return True if the world coordinate lies within the map bounds."""
        raise NotImplementedError

    @abstractmethod
    def get(self, index: Union[Index2D, Index3D], layer: Optional[str] = None) -> Any:
        """Retrieve value at the given index, optionally from a specific layer."""
        raise NotImplementedError

    @abstractmethod
    def set(self, index: Union[Index2D, Index3D], value: Any, layer: Optional[str] = None) -> None:
        """Set value at the given index, optionally in a specific layer."""
        raise NotImplementedError

    # ---------- Layer utilities ----------

    def add_layer(self, name: str, data: Any) -> None:
        """Register a new arbitrary layer (array, dict, tensor, etc.)."""
        self.layers[name] = data

    def remove_layer(self, name: str) -> None:
        """Remove a layer by name."""
        if name in self.layers:
            del self.layers[name]

    def list_layers(self) -> List[str]:
        """List the names of available layers."""
        return list(self.layers.keys())

    # ---------- Serialization (base) ----------

    def to_dict(self) -> Dict[str, Any]:
        """
        Lightweight serialization. Subclasses should extend with shape/params.
        """
        return {
            "type": self.__class__.__name__,
            "frame_id": self.frame_id,
            "origin": tuple(self.origin),
            "resolution": self.resolution if not isinstance(self.resolution, tuple) else list(self.resolution),
            "bounds": self.bounds,
            "dim": self.dim,
            # serialize layers using hook (for JSON-safe data)
            "layers": self.serialize_layers(self.layers),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Map:
        """
        Base deserialization of metadata only; subclasses should handle shape/params.
        NOTE: when loading from file, use Map.load(file_path) to get the right subclass.
        """
        return cls(
            frame_id=data.get("frame_id", "map"),
            origin=tuple(data.get("origin", (0.0, 0.0))),  # type: ignore
            resolution=data.get("resolution", 1.0),
            bounds=data.get("bounds", None),
            layers=cls.deserialize_layers(data.get("layers", {}))
        )

    # ---------- Hooks for layer (de)serialization ----------

    @staticmethod
    def serialize_layers(layers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default: assume layers are JSON-serializable (lists/dicts/numbers/strings).
        Override in subclasses to handle numpy arrays or custom objects.
        """
        return layers

    @staticmethod
    def deserialize_layers(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default: pass-through for JSON data.
        Override to reconstruct arrays/tensors/etc.
        """
        return data

    # ---------- Save / Load API ----------

    def save(self, file_path: str, compressed: bool = False) -> None:
        """
        Save the map to a JSON file. If 'compressed' is True, save as gzipped JSON (.json.gz).
        """
        payload = self.to_dict_with_shape()
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if compressed or file_path.endswith(".gz"):
            with gzip.open(file_path, "wt", encoding="utf-8") as f:
                f.write(text)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

    @classmethod
    def load(cls, file_path: str) -> Map:
        """
        Load a map from JSON (or gzipped JSON) and instantiate the correct subclass
        by using the 'type' field and a simple subclass registry.
        """
        # Read json text (support gzip by extension)
        if file_path.endswith(".gz"):
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Resolve subclass by 'type'
        type_name = data.get("type", None)
        if type_name is None:
            raise ValueError("Missing 'type' in serialized map.")

        subclass = _resolve_map_subclass(type_name)
        if subclass is None:
            raise ValueError(f"Unknown map type '{type_name}'. Ensure the subclass is imported.")

        # Delegate to subclass-specific from_dict
        return subclass._from_dict_full(data)

    def to_dict_with_shape(self) -> Dict[str, Any]:
        """
        Base payload plus shape and any subclass extras via _extras_for_save.
        """
        base = self.to_dict()
        base.update({
            "shape": self.shape,
        })
        base.update(self._extras_for_save())
        return base

    def _extras_for_save(self) -> Dict[str, Any]:
        """
        Subclasses can override to append their own fields (e.g., nx, ny, nz).
        """
        return {}

    @classmethod
    def _from_dict_full(cls, data: Dict[str, Any]) -> Map:
        """
        Subclasses override to reconstruct full state including shape and layers.
        Default falls back to base 'from_dict'.
        """
        return cls.from_dict(data)


def _resolve_map_subclass(type_name: str) -> Optional[type]:
    """
    Find a Map subclass by name. This relies on subclasses being imported
    into the runtime so they appear in __subclasses__().
    """
    # Search recursively through subclasses
    def all_subclasses(base):
        subs = set()
        for s in base.__subclasses__():
            subs.add(s)
            subs.update(all_subclasses(s))
        return subs

    for s in all_subclasses(Map):
        if s.__name__ == type_name:
            return s
    return None

