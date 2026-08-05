"""Getting the PDF, from whatever the user happened to paste.

An arXiv identifier, an abstract page, a direct PDF link and a path on disk are
four different things that all mean "this paper", and sorting out which one is
in hand is not work worth doing by hand each time.

The failure this module exists to prevent is silent. A paywalled or redirected
download returns an HTML error page with HTTP 200, it gets saved under the name
``paper.pdf``, and every tool afterwards fails with a message about the PDF
being malformed rather than about the download having failed. So the first bytes
are checked before the file is kept, and a bad download is deleted and reported
as what it is.

Only the standard library is used, so this runs in any interpreter on any of the
machines in this project without a virtualenv being right first.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib import error, parse, request
from xml.etree import ElementTree

ARXIV_PDF_URL = "https://arxiv.org/pdf/{}"
ARXIV_ABS_URL = "https://arxiv.org/abs/{}"
ARXIV_API_URL = "https://export.arxiv.org/api/query?id_list={}"

USER_AGENT = "sparx-agency-pdf-parser/1.0 (research paper reader)"
"""arXiv rejects requests with no user agent, and asks that tools identify themselves."""

TIMEOUT_S = 60.0

_ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
_ARXIV_IN_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
_ATOM = "{http://www.w3.org/2005/Atom}"


class DownloadFailed(RuntimeError):
    """Raised when a paper could not be fetched, or what came back was not a PDF."""


@dataclass(frozen=True)
class PaperSource:
    """Where a paper is coming from.

    Attributes:
        kind: ``arxiv``, ``url`` or ``local``.
        value: The arXiv id, the URL, or the path, according to ``kind``.
        arxiv_id: The identifier when one could be determined, else None.
    """

    kind: str
    value: str
    arxiv_id: Optional[str] = None

    @property
    def pdf_url(self) -> Optional[str]:
        """The URL to download, or None for a local file."""
        if self.kind == "arxiv":
            return ARXIV_PDF_URL.format(self.arxiv_id)
        if self.kind == "url":
            return self.value
        return None

    @property
    def abstract_url(self) -> Optional[str]:
        """The landing page, which usually carries the code link, or None."""
        if self.arxiv_id:
            return ARXIV_ABS_URL.format(self.arxiv_id)
        return None


def resolve_source(raw: str) -> PaperSource:
    """Work out what a user-supplied paper reference actually is.

    Args:
        raw: An arXiv id, an arXiv URL, any PDF URL, or a filesystem path.

    Returns:
        The classified :class:`PaperSource`.

    Raises:
        ValueError: If the string is empty, or names a local file that is absent.
    """
    text = raw.strip()
    if not text:
        raise ValueError("no paper given")

    bare = _ARXIV_ID.match(text)
    if bare:
        return PaperSource("arxiv", text, bare.group(1))

    in_url = _ARXIV_IN_URL.search(text)
    if in_url:
        return PaperSource("arxiv", text, in_url.group(1))

    if text.lower().startswith(("http://", "https://")):
        return PaperSource("url", text)

    path = Path(text).expanduser()
    if path.exists():
        return PaperSource("local", str(path.resolve()))
    raise ValueError(
        "{!r} is not an arXiv id, a URL, or an existing file".format(raw)
    )


def _get(url: str) -> bytes:
    """Fetch a URL, following redirects, and return its body.

    Raises:
        DownloadFailed: On any network or HTTP error.
    """
    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with request.urlopen(req, timeout=TIMEOUT_S) as response:
            return response.read()
    except (error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
        raise DownloadFailed("could not fetch {}: {}".format(url, exc)) from exc


def download_pdf(url: str, dest: Path) -> Path:
    """Download a PDF and refuse to keep anything that is not one.

    Args:
        url: The URL to fetch.
        dest: Where to write it; the parent is created if absent.

    Returns:
        ``dest``.

    Raises:
        DownloadFailed: If the fetch failed, or the body is not a PDF.
    """
    body = _get(url)
    if body[:4] != b"%PDF":
        preview = body[:120].decode("utf-8", "replace").replace("\n", " ")
        raise DownloadFailed(
            "{} returned {} bytes that are not a PDF — starts {!r}. A paywall or "
            "a redirect to a landing page looks exactly like this.".format(
                url, len(body), preview
            )
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return dest


def fetch(source: PaperSource, dest: Path) -> Path:
    """Put the paper at ``dest``, downloading or copying as appropriate.

    Args:
        source: What :func:`resolve_source` returned.
        dest: Where the PDF should end up.

    Returns:
        ``dest``.

    Raises:
        DownloadFailed: If a remote fetch failed or returned a non-PDF.
        ValueError: If a local source is not a PDF.
    """
    if source.kind == "local":
        origin = Path(source.value)
        with origin.open("rb") as handle:
            if handle.read(4) != b"%PDF":
                raise ValueError("{} is not a PDF".format(origin))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if origin.resolve() != dest.resolve():
            shutil.copy2(origin, dest)
        return dest

    url = source.pdf_url
    if url is None:
        raise DownloadFailed("no URL to fetch for {}".format(source))
    return download_pdf(url, dest)


def arxiv_metadata(arxiv_id: str) -> Dict[str, object]:
    """Fetch title, authors, abstract and categories from the arXiv API.

    Worth having because the PDF's own embedded title is usually blank, and
    because the abstract in a citable form saves transcribing it off page one.

    Args:
        arxiv_id: The identifier, without a version suffix.

    Returns:
        A dict with ``title``, ``authors``, ``abstract``, ``published``,
        ``categories`` and ``id``. Values are empty when arXiv did not supply
        them.

    Raises:
        DownloadFailed: If the API could not be reached or returned no entry.
    """
    body = _get(ARXIV_API_URL.format(parse.quote(arxiv_id)))
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        raise DownloadFailed("arXiv API returned unparseable XML: {}".format(exc)) from exc

    entry = root.find(_ATOM + "entry")
    if entry is None:
        raise DownloadFailed("arXiv has no entry for {}".format(arxiv_id))

    def _text(tag: str) -> str:
        node = entry.find(_ATOM + tag)
        return " ".join((node.text or "").split()) if node is not None else ""

    authors: List[str] = []
    for author in entry.findall(_ATOM + "author"):
        name = author.find(_ATOM + "name")
        if name is not None and name.text:
            authors.append(name.text.strip())

    categories = [
        node.attrib["term"]
        for node in entry.findall(_ATOM + "category")
        if "term" in node.attrib
    ]

    return {
        "id": arxiv_id,
        "title": _text("title"),
        "abstract": _text("summary"),
        "published": _text("published"),
        "authors": authors,
        "categories": categories,
    }
