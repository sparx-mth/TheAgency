"""Frames in RAM: the episode's encoded images in a dictionary, dropped when the next episode begins.

The store for a run that reads its frames back while it runs and has no use
for them afterwards: nothing touches the disk, and a thousand-episode
campaign leaves no folder behind. It keeps the same encoded bytes a
:class:`~sparx_agency.core.planning.object_search.frame_log.disk_store.DiskFrameStore`
writes -- not the raw arrays -- so an episode costs what its image files
would, and a frame read from here is identical to the same frame read from
disk.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from sparx_agency.core.planning.object_search.frame_log.interfaces import (
    FrameLogError, FrameRecord, check_frame_name)


class MemoryFrameStore:
    """Encoded frames of one episode at a time, by name.

    Attributes:
        header: The header of the episode in force, as it was handed to
            :meth:`begin` (a copy); None before the first one.
    """

    def __init__(self) -> None:
        self.header = None  # type: Optional[Dict[str, Any]]
        self._images = None  # type: Optional[Dict[str, bytes]]

    def begin(self, episode_id: str, header: Mapping[str, Any]) -> None:
        """Drop the frames of the episode before and start an empty one."""
        self.header = dict(header)
        self._images = {}

    def put(self, record: FrameRecord, image: bytes) -> None:
        """Keep the image's bytes under ``record.name``."""
        self._episode()[check_frame_name(record.name)] = bytes(image)

    def get(self, name: str) -> bytes:
        """The bytes kept under ``name`` in the episode in force."""
        images = self._episode()
        if check_frame_name(name) not in images:
            raise FrameLogError("no frame named %r in the episode held in memory" % (name,))
        return images[name]

    def describe(self) -> Dict[str, Any]:
        """``{"kind": "memory"}``: there is nowhere to point at."""
        return {"kind": "memory"}

    def _episode(self) -> Dict[str, bytes]:
        if self._images is None:
            raise FrameLogError("no episode has begun in the in-memory frame store")
        return self._images
