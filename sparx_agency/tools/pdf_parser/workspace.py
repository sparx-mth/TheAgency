"""Where the extraction of one paper lives on disk.

One directory per paper, with a fixed shape, so that a reader arriving at a
workspace they did not create knows where to look without being told, and so
that a second run over the same paper overwrites cleanly instead of
interleaving with the first.

    <root>/
      paper.pdf                the source, copied in rather than read in place
      MANIFEST.md              what came out, and what did not — read this first
      captions.json            every figure/table/algorithm, with page and box
      links.json               code, model and DOI links found in the text
      text/full.txt            whole paper, column layout preserved
      text/pages/pNNN.txt      one file per page; the filename is the page number
      pages/p-NN.png           a render of every page, for reading figures
      figures/figure-1.png     caption-driven crops: diagrams, charts, plots
      figures/table-2.png      the table as it is laid out, for checking the parse
      figures/embedded/        raster images lifted out of the PDF as stored
      tables/table-2.md|.csv   parsed table cells
      pseudocode/algorithm-1.txt   algorithm blocks, verbatim with indentation

Nothing is written outside the root, and the root is never a repository
directory by default — a cloned paper repo and a 40 MB PDF inside a source tree
end up in somebody's commit.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

DEFAULT_ROOT = Path.home() / "papers"
"""Where workspaces go unless the caller says otherwise."""

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(title: str, max_words: int = 6) -> str:
    """Turn a paper title into a short directory name.

    Args:
        title: The paper title, or any human string.
        max_words: How many words to keep. Six is enough to stay recognisable
            and short enough to type.

    Returns:
        A lowercase hyphenated slug, or ``"paper"`` if nothing survived.
    """
    words = [word for word in _SLUG_STRIP.sub("-", title.lower()).split("-") if word]
    return "-".join(words[:max_words]) or "paper"


@dataclass(frozen=True)
class PaperWorkspace:
    """The directory layout for one paper.

    Attributes:
        root: The workspace directory. Every other path is under it.
    """

    root: Path

    @classmethod
    def for_slug(cls, slug: str, root: Path = DEFAULT_ROOT) -> "PaperWorkspace":
        """Build the workspace for ``<root>/<slug>``."""
        return cls(Path(root).expanduser() / slug)

    @property
    def pdf(self) -> Path:
        """The copied source PDF."""
        return self.root / "paper.pdf"

    @property
    def manifest(self) -> Path:
        """The human-readable summary of what was extracted."""
        return self.root / "MANIFEST.md"

    @property
    def captions_json(self) -> Path:
        """The machine-readable caption index."""
        return self.root / "captions.json"

    @property
    def links_json(self) -> Path:
        """The machine-readable link index."""
        return self.root / "links.json"

    @property
    def text_dir(self) -> Path:
        """Directory holding ``full.txt``."""
        return self.root / "text"

    @property
    def full_text(self) -> Path:
        """The whole paper as one layout-preserved text file."""
        return self.text_dir / "full.txt"

    @property
    def page_text_dir(self) -> Path:
        """Directory holding one text file per page."""
        return self.text_dir / "pages"

    @property
    def page_image_dir(self) -> Path:
        """Directory holding one PNG render per page."""
        return self.root / "pages"

    @property
    def figure_dir(self) -> Path:
        """Directory holding caption-driven crops."""
        return self.root / "figures"

    @property
    def embedded_dir(self) -> Path:
        """Directory holding rasters lifted out of the PDF unchanged."""
        return self.figure_dir / "embedded"

    @property
    def table_dir(self) -> Path:
        """Directory holding parsed table cells as markdown and CSV."""
        return self.root / "tables"

    @property
    def pseudocode_dir(self) -> Path:
        """Directory holding algorithm blocks as verbatim text."""
        return self.root / "pseudocode"

    def directories(self) -> List[Path]:
        """Every directory the extractor writes into, in creation order."""
        return [
            self.root,
            self.text_dir,
            self.page_text_dir,
            self.page_image_dir,
            self.figure_dir,
            self.embedded_dir,
            self.table_dir,
            self.pseudocode_dir,
        ]

    def create(self) -> "PaperWorkspace":
        """Create every directory, leaving existing content alone."""
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def clear_outputs(self) -> None:
        """Delete previously extracted output, keeping ``paper.pdf``.

        A re-run after a fix must not leave the previous run's figures behind
        looking like part of the new one.
        """
        for directory in self.directories():
            if directory == self.root or not directory.exists():
                continue
            shutil.rmtree(directory)
        for stray in (self.manifest, self.captions_json, self.links_json):
            stray.unlink(missing_ok=True)
        self.create()

    def relative(self, path: Path) -> str:
        """Render ``path`` relative to the workspace root, for the manifest."""
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)
