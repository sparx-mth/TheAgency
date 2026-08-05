"""The only place in this package that starts a process.

Everything here is built on poppler-utils — ``pdfinfo``, ``pdftotext``,
``pdftoppm``, ``pdfimages`` — rather than on a Python PDF library, for one
practical reason: poppler is already installed on every machine in this project
and on every Ubuntu image we build from, while ``PyMuPDF``/``pdfplumber`` are
not, and adding a wheel to read a paper is a poor trade. The cost is that the
parsers upstream of this module have to work from what the command line tools
emit, which is why :mod:`layout` exists.

A missing binary or a poppler error raises. Falling back to a partial
extraction would produce a workspace that looks complete and quietly is not,
and the reader has no way to tell.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

REQUIRED_BINARIES = ("pdfinfo", "pdftotext", "pdftoppm", "pdfimages")
"""The poppler tools this package needs. All ship in the ``poppler-utils`` package."""

INSTALL_HINT = "install them with:  sudo apt install poppler-utils"


class PopplerNotInstalled(RuntimeError):
    """Raised when a required poppler binary is not on ``PATH``."""


class PopplerFailed(RuntimeError):
    """Raised when a poppler binary runs but exits non-zero."""


def missing_binaries() -> List[str]:
    """Return the names of required poppler binaries that are not on ``PATH``."""
    return [name for name in REQUIRED_BINARIES if shutil.which(name) is None]


def require_poppler() -> None:
    """Raise :class:`PopplerNotInstalled` unless every required binary is present.

    Call this once, early, rather than letting the first extraction step fail
    halfway through a workspace.
    """
    missing = missing_binaries()
    if missing:
        raise PopplerNotInstalled(
            "missing poppler tools: {} — {}".format(", ".join(missing), INSTALL_HINT)
        )


def run(binary: str, args: Sequence[str], timeout_s: float = 300.0) -> str:
    """Run a poppler binary and return its standard output as text.

    Args:
        binary: Binary name, e.g. ``"pdftotext"``.
        args: Arguments after the binary name.
        timeout_s: Hard limit; a malformed PDF can send poppler into a very long
            loop, and a hung extraction is worse than a failed one.

    Returns:
        Standard output, decoded as UTF-8 with undecodable bytes replaced.

    Raises:
        PopplerNotInstalled: If ``binary`` is not on ``PATH``.
        PopplerFailed: If it exits non-zero or exceeds ``timeout_s``.
    """
    if shutil.which(binary) is None:
        raise PopplerNotInstalled("{} not found — {}".format(binary, INSTALL_HINT))

    command = [binary] + [str(a) for a in args]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PopplerFailed(
            "{} timed out after {:.0f}s: {}".format(binary, timeout_s, " ".join(command))
        ) from exc

    if completed.returncode != 0:
        raise PopplerFailed(
            "{} exited {}: {}\n{}".format(
                binary,
                completed.returncode,
                " ".join(command),
                completed.stderr.decode("utf-8", "replace").strip(),
            )
        )
    return completed.stdout.decode("utf-8", "replace")


def page_range_args(first: Optional[int], last: Optional[int]) -> List[str]:
    """Build ``-f``/``-l`` arguments for a 1-based, inclusive page range.

    Args:
        first: First page, or None for "from the beginning".
        last: Last page, or None for "to the end".

    Returns:
        The argument list, empty when both bounds are None.

    Raises:
        ValueError: If a bound is below 1 or the range is inverted.
    """
    args: List[str] = []
    if first is not None:
        if first < 1:
            raise ValueError("first page is 1-based, got {}".format(first))
        args += ["-f", str(first)]
    if last is not None:
        if last < 1:
            raise ValueError("last page is 1-based, got {}".format(last))
        args += ["-l", str(last)]
    if first is not None and last is not None and last < first:
        raise ValueError("inverted page range: {} to {}".format(first, last))
    return args


def check_pdf(path: Path) -> None:
    """Raise unless ``path`` exists and is a readable PDF.

    The failure this guards against is specific and common: a paywalled or
    redirected download saves an HTML error page under the name ``paper.pdf``,
    and every tool downstream then fails with something that does not mention
    the real problem.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file does not begin with a PDF header.
    """
    if not path.is_file():
        raise FileNotFoundError("no such file: {}".format(path))
    with path.open("rb") as handle:
        header = handle.read(5)
    if header[:4] != b"%PDF":
        raise ValueError(
            "{} is not a PDF (starts with {!r}) — a download that returned an "
            "HTML error page looks exactly like this".format(path, header)
        )
