"""Every ``<arg>`` a roslaunch file declares, with the comments explaining it.

The FALCON launch files are the stack's real documentation: a few hundred
``<arg>`` declarations, each wrapped in the prose that says what it does and why
its default is what it is. An XML parser reaches the arguments but throws that
prose away, so this scans the text instead and keeps both, associating each
comment with the ``<arg>`` it sits on by position.

Only ``<arg>`` elements carrying a ``default`` are returned. An ``<arg>`` with a
``value`` inside an ``<include>`` is not a knob -- it is this file *passing* one
down to a child, and offering it for editing would suggest a choice that does
not exist.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from pathlib import Path
from xml.sax.saxutils import unescape

from ..spec import ROSLAUNCH, ParamSpec
from .pysource import clean_heading, is_heading

#: Comments and <arg> tags, in one pass so their order is preserved.
_TOKEN_RE = re.compile(r"<!--(?P<comment>.*?)-->|<arg\b(?P<attrs>[^>]*?)/?>", re.S)
_ATTR_RE = re.compile(r"""(?P<key>\w+)\s*=\s*(?P<quote>["'])(?P<value>.*?)(?P=quote)""", re.S)


def _line_index(text: str) -> list[int]:
    """Offsets at which each line starts, for O(log n) offset -> line lookups."""
    starts, offset = [0], text.find("\n")
    while offset != -1:
        starts.append(offset + 1)
        offset = text.find("\n", offset + 1)
    return starts


def _tidy(comment: str) -> str:
    """Flatten a multi-line XML comment into one readable line."""
    lines = [line.strip() for line in comment.strip().splitlines()]
    return " ".join(line for line in lines if line)


def _split_banner(comment: str) -> tuple[str, str]:
    """Separate a banner comment's title from the prose beneath it.

    The FALCON launch files open a section with a ruled title and then explain
    it for a paragraph inside the same comment::

        <!-- ── The STAGING VANTAGE POINT ──────────────────────
             ONE place that decides where the drone stands ... -->

    Taking the whole comment as the section name would put that paragraph in
    every group header, so the first line becomes the title and the rest is
    handed on as documentation for the argument that follows.
    """
    lines = [line.strip() for line in comment.strip().splitlines()]
    lines = [line for line in lines if line]
    title = clean_heading(lines[0]) if lines else ""
    return title, " ".join(lines[1:]).strip()


def discover(path: str | Path) -> list[ParamSpec]:
    """Read the declared ``<arg>`` set out of a roslaunch file.

    Args:
        path: The ``.launch`` XML file.

    Returns:
        One :class:`~..spec.ParamSpec` per declared arg, in file order. A
        comment ending on the same line as the arg documents it; otherwise the
        comment block directly above does. Banner comments become sections.

    Raises:
        OSError: If the file cannot be read.
    """
    text = Path(path).read_text(encoding="utf-8")
    label = Path(path).name
    starts = _line_index(text)
    line_of = lambda offset: bisect_right(starts, offset) - 1  # noqa: E731

    params: list[ParamSpec] = []
    section, pending, detail, pending_line = "", "", "", -1

    for token in _TOKEN_RE.finditer(text):
        if (comment := token.group("comment")) is not None:
            body = _tidy(comment)
            if is_heading(body):
                title, prose = _split_banner(comment)
                section, pending, detail = title or section, "", prose
            # A comment that closes on the line an <arg> ended on documents that
            # arg, not the next one: attach it and clear the pending block.
            elif params and line_of(token.start()) == pending_line:
                if not params[-1].doc:
                    params[-1].doc = body
            else:
                pending = body
            continue

        attrs = {m.group("key"): unescape(m.group("value"))
                 for m in _ATTR_RE.finditer(token.group("attrs"))}
        if "name" not in attrs or "default" not in attrs:
            continue
        params.append(ParamSpec(
            name=attrs["name"],
            default=attrs["default"],
            doc=pending,
            detail=detail,
            section=section,
            syntax=ROSLAUNCH,
            source=label,
        ))
        pending, pending_line = "", line_of(token.end())

    return params
