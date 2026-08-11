# pdf_parser — reading a research paper mechanically

Turns one PDF into a directory you can work from: the text with its columns
intact, an image of every page, an enlarged image of every labelled figure and
chart, every table as cells, and every algorithm with its indentation rebuilt.

Written for the moment someone drops a paper into the team and asks what it
means for us. Reading it is the interesting part. Getting at it is not, and
doing that by hand is where the mistakes come from — a number transposed out of
a results table, a figure skipped because the caption seemed to describe it, an
algorithm flattened into a list of statements with its loop structure gone.

```sh
# from the repository root
python -m sparx_agency.tools.pdf_parser extract 1706.03762
python -m sparx_agency.tools.pdf_parser extract ~/Downloads/navdp.pdf --slug navdp
```

Everything lands in `~/papers/<slug>/`. Start with `MANIFEST.md`.

## What you get

```
~/papers/<slug>/
  paper.pdf
  MANIFEST.md                what was extracted, and what was not
  captions.json              every exhibit, with page and rectangle
  links.json                 code, model and DOI links printed in the paper
  text/full.txt              whole paper, columns preserved
  text/pages/pNNN.txt        one file per page — quote from these
  pages/p-NN.png             150-DPI render of every page
  figures/figure-1.png       300-DPI crop of each figure and chart, with caption
  figures/table-2.png        the table as printed, for checking the parse
  figures/embedded/          raster images stored inside the PDF
  tables/table-2.md|.csv     parsed cells
  pseudocode/algorithm-1.txt algorithm blocks, indentation intact
```

| you want | open |
|---|---|
| exact wording to quote | `text/pages/pNNN.txt` — the filename is the page number |
| to see a page as printed | `pages/p-NN.png` |
| one figure or chart, enlarged | `figures/figure-N.png` |
| a table's numbers | `tables/table-N.md`, checked against `figures/table-N.png` |
| an algorithm | `pseudocode/algorithm-N.txt` |
| the paper's own code link | `links.json` |

## Requirements

**poppler-utils, and nothing else.** No PDF library, no model, no network beyond
fetching the paper. `sudo apt install poppler-utils` if `pdfinfo` is missing.

Only the standard library is imported, so this runs under any interpreter in the
repository — the venv, a conda env, or the Noetic container's Python 3.8.

## Commands

| command | does |
|---|---|
| `extract <paper>` | fetch if needed, then extract everything into a workspace |
| `captions <pdf>` | list the figures, tables and algorithms it can find |
| `links <pdf>` | list the code and reference links printed in the paper |
| `crop <pdf> --page N --box X0 Y0 X1 Y1 --out F` | render one rectangle, for a figure that needs a closer look |
| `meta <arxiv-id>` | title, authors and abstract from the arXiv API |

`extract` accepts an arXiv id (`2505.08712`), an arXiv or PDF URL, or a local
path. Useful flags: `--slug` names the workspace, `--workspace` overrides the
location entirely, `--keep` leaves a previous run's output in place.

## From Python

```python
from pathlib import Path
from sparx_agency.tools.pdf_parser import extract_paper, PaperWorkspace

result = extract_paper(Path("~/Downloads/navdp.pdf").expanduser(),
                       PaperWorkspace.for_slug("navdp"))

for exhibit in result.of_kind("figure"):
    print(exhibit.caption.label, exhibit.caption.page, exhibit.image)
print(result.links.repositories)
```

## How it works, and where it can be wrong

A PDF records glyphs at coordinates. It does **not** record that a drawing is a
figure, that a set of numbers is a table, or where either one begins and ends.
All of that is inferred here, so it is worth knowing how.

**Positions come from `pdftotext -bbox-layout`**, which reports a rectangle for
every word, line and block. That is the foundation; without coordinates none of
the rest is possible. Poppler's XML is repaired before it is parsed: a symbol
font whose glyphs have no Unicode mapping makes it write the raw glyph code into
a `<word>`, and codes below `0x20` are legal in a PDF but illegal in XML, so one
equation would otherwise cost the document every figure, table and algorithm
while the text and page renders still looked fine. Those characters are replaced
by `U+FFFD`, keeping the word and its rectangle
(`layout.py:sanitise_control_characters`).

**Captions are the index.** A paper lays its figures out however it likes but
always labels them. Every crop, table and listing starts from a caption. Two
rules keep prose out: a caption starts a block, and it separates its number from
its text (`Figure 2: ...`, `Fig. 1. ...`). "Figure 2 shows the architecture" is
therefore not a caption — and an *unlabelled* figure is invisible to this tool.
That trade is deliberate: a missed figure is still on the page render, while a
cropped paragraph is noise you have to read to dismiss.

**Extent is grown from the caption outward.** Figures grow upward, tables and
algorithms downward, matching the templates. Growth stops at running prose,
another caption, or too large a gap. "Running prose" is decided by word spacing,
words per line and how far lines reach across the column — *not* by word count,
because an architecture diagram's labels are plenty of words and are not a
paragraph.

**Columns come from the whitespace** — the vertical stripes no word crosses. A
fifth of the rows are allowed to bridge a stripe, which is what stops a heading
centred over two sub-columns from merging them.

**Rows come from vertical position**, not from poppler's lines. Inside a table
poppler usually makes each *cell* its own line, so a row arrives as several
lines sharing a baseline; they are clustered back together first.

**Mathematics does not survive text extraction, anywhere.** A PDF's text layer
has no superscripts, no subscripts and no two-dimensional structure. A fraction
comes out as three lines, `d_k` comes out as `dk`, and `3.3 · 10^18` comes out
as `3.3 · 1018`. That is poppler faithfully reporting what the file contains, not
a bug here, and no text-based tool can do better. Read equations, exponents and
units-with-powers off `pages/` or a crop — including numbers inside a parsed
table.

Other known limits, which show up as an obviously wrong picture rather than as
silently wrong data:

- An unlabelled figure is not found. The manifest's words-per-page table is the
  way to catch one: a page that is nearly all picture is flagged.
- Two columns packed tightly enough that no stripe survives merge into one.
- A cell that wraps onto two lines arrives as two rows.
- The header is taken to be the first row; multi-row headers arrive as rows.
- A scanned paper has no text layer at all — the manifest says so, and the page
  renders are the whole output.

**Every inference is published next to a picture of what it was drawn from.**
That is the pairing that makes the output safe to quote: the markdown is for
quoting, and the image beside it is for checking. Where the two disagree, the
image is right.

## Tests

```sh
python -m pytest sparx_agency/tools/pdf_parser -q
```

101 tests, about two seconds. The end-to-end ones in `test_pipeline.py` run the
real poppler toolchain over real PDFs — `tests/minimal_pdf.py` writes them from
scratch at test time, so the suite carries no binary fixtures and no third
party's copyright. They skip themselves if poppler is not installed.

## Related

The `/study-paper` skill drives this tool as the extraction step of a longer
workflow: fetch, extract, read, read the paper's code, then work out how it maps
onto this repository.
