"""The repository the paper is asking you to look at.

Nearly every paper that ships code prints the URL in its own text — in the
abstract, in a footnote on page one, or at the end of the introduction. Reading
it out of the extracted text is more reliable than searching the web for it,
because it is the authors' own link rather than a fork, a mirror or a reading
group's notes.

Two habits of typesetting have to be worked around. URLs get broken across lines
at the right margin, sometimes with a hyphen that is not part of the address, so
the scan is run a second time over the text with its line breaks removed. And
URLs at the end of a sentence pick up the full stop, so trailing punctuation is
trimmed. The second pass can invent a link that the page does not contain, which
is why these are reported as candidates: opening one costs a moment, and missing
the repository costs the whole point of reading the paper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

REPO_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "huggingface.co")
"""Hosts whose links are code or model releases rather than references."""

_REPO = re.compile(
    r"(?:https?://)?(?:www\.)?(?:" + "|".join(host.replace(".", r"\.") for host in REPO_HOSTS) + r")"
    r"/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*",
    re.IGNORECASE,
)

_ARXIV = re.compile(
    r"(?:arxiv[:\s]+|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE
)

_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)

_PROJECT_PAGE = re.compile(
    r"https?://[A-Za-z0-9_.\-]+\.(?:github\.io|io|ai|org|com|net|edu)/[A-Za-z0-9_./\-]*",
    re.IGNORECASE,
)

_TRAILING_JUNK = re.compile(r"[.,;:)\]}\s]+$")

_SENTENCE_BOUNDARY = re.compile(r"\.(?=[A-Z])")
"""A full stop followed by a capital — the end of the sentence the URL ended.

The line-repair pass closes up the break between "...github.com/org/repo." and
the heading that followed it, and without this the heading arrives welded to the
repository name.
"""


@dataclass
class PaperLinks:
    """Every address worth following that the paper printed.

    Attributes:
        repositories: Code and model releases, normalised to ``https://`` URLs.
        arxiv_ids: arXiv identifiers, without the version suffix.
        dois: DOIs.
        project_pages: Other project or demo pages, repository hosts excluded.
    """

    repositories: List[str] = field(default_factory=list)
    arxiv_ids: List[str] = field(default_factory=list)
    dois: List[str] = field(default_factory=list)
    project_pages: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when the paper printed no links at all."""
        return not (self.repositories or self.arxiv_ids or self.dois or self.project_pages)

    def to_dict(self) -> Dict[str, List[str]]:
        """A JSON-serialisable view, for ``links.json``."""
        return {
            "repositories": self.repositories,
            "arxiv_ids": self.arxiv_ids,
            "dois": self.dois,
            "project_pages": self.project_pages,
        }


def _clean(raw: str) -> str:
    """Strip sentence punctuation glued to the address and normalise the scheme."""
    trimmed = _SENTENCE_BOUNDARY.split(raw.strip())[0]
    trimmed = _TRAILING_JUNK.sub("", trimmed)
    if trimmed.lower().startswith("http"):
        return trimmed
    return "https://" + trimmed


def _dedupe(values: Sequence[str]) -> List[str]:
    """Remove case-insensitive duplicates, keeping first-seen order."""
    seen = set()
    unique = []
    for value in values:
        key = value.lower().rstrip("/")
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _scan_variants(text: str) -> List[str]:
    """Return the text as given, and again with its line breaks closed up.

    The second variant repairs URLs the typesetter split at the right margin.
    A hyphen immediately before the break is dropped, since it is far more often
    hyphenation than part of the address.
    """
    unwrapped = re.sub(r"-\s*\n\s*", "", text)
    unwrapped = re.sub(r"\s*\n\s*", "", unwrapped)
    return [text, unwrapped]


def find_links(text: str) -> PaperLinks:
    """Scan extracted paper text for code, model and reference links.

    Args:
        text: The paper's text, typically ``ExtractedText.full``.

    Returns:
        The populated :class:`PaperLinks`.
    """
    repositories: List[str] = []
    arxiv_ids: List[str] = []
    dois: List[str] = []
    project_pages: List[str] = []

    for variant in _scan_variants(text):
        repositories += [_clean(match.group(0)) for match in _REPO.finditer(variant)]
        arxiv_ids += [match.group(1) for match in _ARXIV.finditer(variant)]
        dois += [_TRAILING_JUNK.sub("", match.group(0)) for match in _DOI.finditer(variant)]
        project_pages += [
            _clean(match.group(0))
            for match in _PROJECT_PAGE.finditer(variant)
            if not any(host in match.group(0).lower() for host in REPO_HOSTS)
        ]

    return PaperLinks(
        repositories=_dedupe(repositories),
        arxiv_ids=_dedupe(arxiv_ids),
        dois=_dedupe(dois),
        project_pages=_dedupe(project_pages),
    )
