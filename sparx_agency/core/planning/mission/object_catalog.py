"""Object catalog: load a JSON list of world-placed objects and pick a target.

A mission names an object to fly to (open-vocabulary) and needs, per object, both
its **label** (fed to the detector as the hunted prompt) and its **world position**
(fed to the planner as the coordinate goal). This module is the single, ROS-free
loader of that catalog, shared by the mission director node and any offline tool.

The on-disk format is a top-level JSON *list*, one entry per object (extra keys are
ignored, so the same file can also carry detector metadata)::

    [
      {"label": "refrigerator", "position_m": {"x": -0.98, "y": -4.12, "z": 0.48}},
      {"label": "chair",        "position_m": {"x": 0.32,  "y": -4.74, "z": 0.48}},
      ...
    ]

Positions are metres in the same world frame the localization / ``/waypoint_nav/goal``
use, so ``(x, y)`` is published directly as the planner's point goal.

Labels are normalised (stripped, lower-cased) on load, mirroring
:mod:`sparx_agency.core.common.detection_message`, so downstream label matching never
depends on how the file cased a label. Labels are **not** unique (a room can hold two
chairs), so lookup-by-label returns a list and the catalog preserves file order.

Deliberately ROS-free and Python 3.8 compatible: the FALCON ROS1/Noetic adapter
imports ``core`` under Python 3.8, so no 3.10+ syntax (no ``match``/``case``, no
``@dataclass(slots=True)``, no bare PEP 604 unions). Malformed input raises
``ValueError`` rather than silently defaulting (CLAUDE.md: prefer raising errors).
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union


@dataclass(frozen=True)
class ObjectGoal:
    """A named object at a fixed world position, in metres.

    Attributes:
        label: The object's name, normalised (stripped + lower-cased). Published to
            the detector as the open-vocabulary prompt.
        x: World X (metres) -- the coordinate goal's X (see the frame note above).
        y: World Y (metres) -- the coordinate goal's Y.
        z: World Z (metres); kept for reference (the 2D planner uses only x, y).
    """

    label: str
    x: float
    y: float
    z: float

    def caption(self) -> str:
        """A short human-readable caption for a menu row or a log line."""
        return "%s  (%.2f, %.2f)" % (self.label, self.x, self.y)


class ObjectCatalog:
    """An ordered, immutable collection of :class:`ObjectGoal` loaded from JSON."""

    def __init__(self, objects: Sequence[ObjectGoal]) -> None:
        self._objects = tuple(objects)

    # ── Construction ──────────────────────────────────────────────────
    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "ObjectCatalog":
        """Load a catalog from a JSON file.

        Args:
            path: Path to the objects JSON file.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If its contents are not a valid catalog (see
                :meth:`from_json`).
        """
        return cls.from_json(Path(path).read_text())

    @classmethod
    def from_json(cls, text: str) -> "ObjectCatalog":
        """Parse a catalog from a JSON string.

        Args:
            text: The raw JSON payload (a top-level list of object entries).

        Raises:
            ValueError: If the payload is not a JSON list, an entry is not an
                object, or an entry is missing ``label`` / ``position_m`` or carries
                a non-numeric position.
        """
        try:
            raw = json.loads(text)
        except ValueError as e:
            raise ValueError("object catalog is not valid JSON: %s" % e)
        if not isinstance(raw, list):
            raise ValueError("object catalog must be a JSON list, got %r"
                             % type(raw).__name__)

        objects = []  # type: List[ObjectGoal]
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError("catalog[%d] must be a JSON object, got %r"
                                 % (i, type(item).__name__))
            try:
                pos = item["position_m"]
                objects.append(ObjectGoal(
                    label=str(item["label"]).strip().lower(),
                    x=float(pos["x"]), y=float(pos["y"]), z=float(pos["z"])))
            except KeyError as e:
                raise ValueError("catalog[%d] missing field %s" % (i, e))
            except (TypeError, ValueError):
                raise ValueError("catalog[%d] has a malformed label/position: %r"
                                 % (i, item))
        return cls(objects)

    # ── Access ────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> Iterator[ObjectGoal]:
        return iter(self._objects)

    def __getitem__(self, index: int) -> ObjectGoal:
        return self._objects[index]

    @property
    def objects(self) -> Sequence[ObjectGoal]:
        """The objects in file order (a tuple; safe to expose, it is immutable)."""
        return self._objects

    def labels(self) -> List[str]:
        """Every label in file order (may repeat, e.g. two ``chair`` entries)."""
        return [o.label for o in self._objects]

    def unique_labels(self) -> List[str]:
        """The distinct labels, in first-seen order."""
        seen = []  # type: List[str]
        for o in self._objects:
            if o.label not in seen:
                seen.append(o.label)
        return seen

    def by_label(self, label: str) -> List[ObjectGoal]:
        """All entries whose (normalised) label matches; ``[]`` if none.

        Returns a list because labels are not unique in the catalog.
        """
        key = str(label).strip().lower()
        return [o for o in self._objects if o.label == key]

    def random(self, rng: Optional[random.Random] = None) -> ObjectGoal:
        """Uniformly pick one object.

        Args:
            rng: An optional :class:`random.Random` for reproducible selection;
                the module-global RNG is used when ``None``.

        Raises:
            IndexError: If the catalog is empty.
        """
        if not self._objects:
            raise IndexError("cannot pick from an empty object catalog")
        chooser = rng if rng is not None else random
        return chooser.choice(self._objects)
