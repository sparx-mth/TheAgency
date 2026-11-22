"""Frames on disk: one folder per episode, the images beside the index that maps each to its capture pose.

::

    <root>/
      <episode slug>-<12 hex>/      one episode (:func:`episode_directory_name`)
        episode.json                which episode this is, how its frames are encoded, the camera, what a pose means
        frames.jsonl                one line per frame, in capture order: its name, its step, its pose
        step_000000.jpg             the frame the episode started with
        step_000001.jpg             the frame after the first action
        ...

**The index is the record.** A frame's line is appended to ``frames.jsonl``
only after its image is whole on disk (written beside itself and renamed),
so every line names a complete image and a reader that goes through the
index (:func:`read_frame_log`) never meets half a file. A process killed
between the two leaves at most one image the index does not name.

**An episode logged again replaces itself.** A run resumed after a crash
re-runs the episode it died in from its first step, so :meth:`DiskFrameStore.begin`
removes what an earlier attempt left in the episode's folder -- the header,
the index, every file named like a frame and any write left unfinished;
nothing else, and never a directory. Left in place, the higher-numbered
frames of a longer earlier attempt would sit beside the new ones as if they
were the same walk. Two different runs pointed at one root therefore
overwrite each other's episodes: give each run its own root.

**The folder name can be read and cannot collide.** It is the episode id with
everything but letters, digits, ``.``, ``_`` and ``-`` replaced, then the
first twelve hex digits of the id's SHA-256 -- the same digits the benchmark
runtime's recorder names an episode's ``recordings/<key>/`` folder with, so
an episode's frames and its recording are found by one key.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from sparx_agency.core.planning.object_search.frame_log.interfaces import (
    FRAME_GLOB, SCHEMA, FrameLogError, FrameRecord, check_frame_name)

#: The episode's header: its identity, the image format, the camera, the pose convention.
HEADER_NAME = "episode.json"
#: The lookup from each frame's name to its capture pose, one JSON line per frame.
INDEX_NAME = "frames.jsonl"
#: What a file is called while it is being written; renamed to its own name when whole.
PART_SUFFIX = ".part"
#: Characters of the episode id kept in the folder's readable part.
SLUG_LIMIT = 80


def episode_directory_name(episode_id: str) -> str:
    """The folder an episode's frames are kept in: a readable slug of the id, then twelve hex digits of its hash.

    Args:
        episode_id: The episode's identifier (``"00877-4ok3usBNeis/000003"``).

    Returns:
        One directory level (``"00877-4ok3usBNeis_000003-<12 hex>"``),
        different for different ids whatever the slug loses.

    Raises:
        FrameLogError: If ``episode_id`` is not a non-blank string.
    """
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise FrameLogError("an episode id is a non-blank string, got %r" % (episode_id,))
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", episode_id).strip("._-")[:SLUG_LIMIT] or "episode"
    return "%s-%s" % (slug, hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:12])


def _write_whole(path: pathlib.Path, data: bytes) -> None:
    """``data`` at ``path``, never a part of it: written beside the file and renamed over it."""
    temporary = path.with_name(path.name + PART_SUFFIX)
    temporary.write_bytes(data)
    os.replace(str(temporary), str(path))


class DiskFrameStore:
    """Encoded frames as image files, one folder per episode under ``root``.

    Args:
        root: The directory the episode folders are made in; created with
            the first episode.

    Attributes:
        root: That directory.
        directory: The folder of the episode in force; None before the first
            :meth:`begin`.
    """

    def __init__(self, root) -> None:
        self.root = pathlib.Path(root).expanduser()
        self.directory = None  # type: Optional[pathlib.Path]

    def begin(self, episode_id: str, header: Mapping[str, Any]) -> None:
        """Start the episode's folder empty of frames and write its header.

        What an earlier attempt at the same episode left there is removed
        first (see the module docstring); other files in the folder stay.
        """
        self.directory = None
        directory = self.root / episode_directory_name(episode_id)
        directory.mkdir(parents=True, exist_ok=True)
        for pattern in (HEADER_NAME, INDEX_NAME, FRAME_GLOB, "*" + PART_SUFFIX):
            for stale in directory.glob(pattern):
                if stale.is_file():
                    stale.unlink()
        _write_whole(directory / HEADER_NAME,
                     (json.dumps(dict(header), indent=2, allow_nan=False) + "\n").encode("utf-8"))
        (directory / INDEX_NAME).write_bytes(b"")
        self.directory = directory

    def put(self, record: FrameRecord, image: bytes) -> None:
        """Write the image whole, then append the record's line to the index.

        Raises:
            FrameLogError: Before :meth:`begin`, or for a record named like
                one of the store's own files (the header, the index, an
                unfinished write): its image would overwrite them.
        """
        directory = self._episode()
        name = check_frame_name(record.name)
        if name in (HEADER_NAME, INDEX_NAME) or name.endswith(PART_SUFFIX):
            raise FrameLogError("%r is one of the store's own file names, not a frame's" % (name,))
        _write_whole(directory / name, bytes(image))
        with (directory / INDEX_NAME).open("a", encoding="utf-8") as index:
            index.write(json.dumps(record.to_row(), allow_nan=False) + "\n")

    def get(self, name: str) -> bytes:
        """The bytes of the image file ``name`` in the episode's folder."""
        path = self._episode() / check_frame_name(name)
        if not path.is_file():
            raise FrameLogError("no frame named %r in %s" % (name, path.parent))
        return path.read_bytes()

    def describe(self) -> Dict[str, Any]:
        """``{"kind": "disk", "root": <the root directory>}``."""
        return {"kind": "disk", "root": str(self.root)}

    def _episode(self) -> pathlib.Path:
        if self.directory is None:
            raise FrameLogError("no episode has begun in the frame store under %s" % self.root)
        return self.directory


def read_frame_log(directory) -> Tuple[Dict[str, Any], Dict[str, FrameRecord]]:
    """What an episode's folder holds: its header, and the lookup from each frame's name to its record.

    The reader for anything that looks at a logged episode afterwards. An
    image is then ``FrameCodec.decode((directory / name).read_bytes())``.

    Args:
        directory: One episode's folder, as a :class:`DiskFrameStore` wrote it.

    Returns:
        ``(header, records)``: the header's dictionary, and the records by
        frame name in capture order.

    Raises:
        FrameLogError: If the header or the index is missing or of another
            schema version, a line is not a frame record (named with its
            line number, never skipped: a line cut short by a kill is a frame
            whose pose is unknown), or two lines name one frame.
    """
    directory = pathlib.Path(directory)
    try:
        header = json.loads((directory / HEADER_NAME).read_text(encoding="utf-8"))
        lines = (directory / INDEX_NAME).read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError) as exc:
        raise FrameLogError("%s is not an episode's frame folder: %s: %s" % (directory, type(exc).__name__, exc)) from exc
    if not isinstance(header, dict) or header.get("schema") != SCHEMA:
        raise FrameLogError("%s was written under schema %r; this reader reads %r"
                            % (directory / HEADER_NAME, header.get("schema") if isinstance(header, dict) else None, SCHEMA))
    records = {}  # type: Dict[str, FrameRecord]
    for number, line in enumerate(lines, start=1):
        where = "%s line %d of %d" % (directory / INDEX_NAME, number, len(lines))
        try:
            record = FrameRecord.from_row(json.loads(line))
        except (ValueError, FrameLogError) as exc:
            raise FrameLogError("%s is not a frame record (a write cut short?): %s" % (where, exc)) from exc
        if record.name in records:
            raise FrameLogError("%s names %r, which an earlier line already did" % (where, record.name))
        records[record.name] = record
    return header, records
