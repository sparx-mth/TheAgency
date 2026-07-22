"""Read a Python file as a parameter declaration site.

Both parameter styles this repo uses -- a ROS2 node's ``declare_parameter`` and a
plain script's ``argparse`` -- are function calls with literal arguments, so they
are recovered from the AST rather than by regex: a call spanning several lines,
or a default that is a negative number or a tuple, is then no harder than the
one-liner case.

The AST is only half of it. What a parameter MEANS is written in the ``#``
comments around the call, which the AST discards, so this module keeps the raw
lines alongside the tree and reads the comments back out by line number.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

#: A comment that is really a heading for the block that follows it: a rule of
#: dashes/equals/box-drawing, an ALL-CAPS line, or "<something> params".
_RULE_RE = re.compile(r"[-=_~─═]{4,}")
_GROUP_RE = re.compile(r"^\w[\w /]*\b(params|parameters|args|arguments)\b:?$", re.I)


def is_heading(text: str) -> bool:
    """True when a comment reads as a section heading rather than an explanation.

    A heading groups the parameters after it; an explanation describes the one
    parameter it sits on. Getting this wrong only mislabels a group box in the
    editor, so the test stays deliberately simple.

    Args:
        text: The comment body, ``#`` and surrounding whitespace already stripped.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if _RULE_RE.search(stripped):
        return True
    if _GROUP_RE.match(stripped):
        return True
    words = stripped.rstrip(":").split()
    return (len(words) <= 6 and stripped.rstrip(":").isupper()
            and any(c.isalpha() for c in stripped))


def clean_heading(text: str) -> str:
    """Strip the decoration off a heading comment, leaving its words."""
    return _RULE_RE.sub("", text).strip(" -=_:─═").strip()


@dataclass
class PythonSource:
    """A parsed Python file plus its comment lines, keyed by line number."""

    path: Path
    tree: ast.Module
    lines: list[str]
    #: Section heading in force at each line; built lazily by :meth:`heading_at`.
    _headings: list[str] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "PythonSource":
        """Parse ``path``.

        Raises:
            OSError: If the file cannot be read.
            SyntaxError: If it is not valid Python.
        """
        text = Path(path).read_text(encoding="utf-8")
        return cls(path=Path(path), tree=ast.parse(text), lines=text.splitlines())

    def _comment_on(self, lineno: int) -> str | None:
        """The comment body on 1-based ``lineno``, if that line is only a comment."""
        if not 1 <= lineno <= len(self.lines):
            return None
        stripped = self.lines[lineno - 1].strip()
        return stripped[1:].strip() if stripped.startswith("#") else None

    def doc_above(self, lineno: int) -> str:
        """The contiguous comment block directly above ``lineno``, as one line.

        A heading terminates the block: it belongs to the section, not to this
        parameter.
        """
        block: list[str] = []
        probe = lineno - 1
        while (comment := self._comment_on(probe)) is not None:
            if is_heading(comment):
                break
            block.insert(0, comment)
            probe -= 1
        return " ".join(block).strip()

    def trailing_doc(self, lineno: int) -> str:
        """The ``# ...`` comment at the end of ``lineno`` itself, if any.

        Only trusted when the ``#`` is outside every quote on the line, so a
        default value that happens to contain a hash is not read as a comment.
        """
        if not 1 <= lineno <= len(self.lines):
            return ""
        line = self.lines[lineno - 1]
        quote = None
        for index, char in enumerate(line):
            if quote:
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "#":
                return line[index + 1:].strip()
        return ""

    def heading_at(self, lineno: int) -> str:
        """The section heading in force at ``lineno``, or ``""``.

        A heading governs everything after it until the next one, so this is a
        forward sweep cached on first use -- not a search back up from the line,
        which would stop at the first line of code and leave every parameter but
        the first of a group unsectioned.
        """
        if self._headings is None:
            self._headings = []
            current = ""
            for line in self.lines:
                stripped = line.strip()
                if stripped.startswith("#") and is_heading(stripped[1:].strip()):
                    current = clean_heading(stripped[1:].strip()) or current
                self._headings.append(current)
        return self._headings[lineno - 1] if 1 <= lineno <= len(self._headings) else ""

    def calls(self, attr: str) -> list[ast.Call]:
        """Every call to a method named ``attr``, in source order.

        Args:
            attr: The method name, e.g. ``"declare_parameter"``.
        """
        found = [node for node in ast.walk(self.tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == attr]
        return sorted(found, key=lambda n: n.lineno)


def literal(node: ast.AST | None) -> object:
    """``ast.literal_eval`` that answers ``None`` instead of raising.

    A non-literal default (a name, a call, an f-string) is not something the
    editor can round-trip, so it is reported as absent rather than guessed at.
    """
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def as_command_value(value: object) -> str:
    """Render a Python literal the way a command line must receive it.

    ROS2 and roslaunch both parse their values as YAML, where a boolean is
    lowercase; ``str(True)`` would give ``True``, which neither accepts.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def keyword(call: ast.Call, name: str) -> ast.AST | None:
    """The AST node of keyword argument ``name``, if the call passes it."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None
