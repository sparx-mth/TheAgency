"""Command line for the parser.

The whole package exists to be used from a terminal or from an agent's shell, so
the interface is deliberately small: one command that does everything, and three
that answer a single question without building a workspace.

    python -m sparx_agency.tools.pdf_parser extract 1706.03762
    python -m sparx_agency.tools.pdf_parser extract ~/Downloads/paper.pdf --slug navdp
    python -m sparx_agency.tools.pdf_parser captions paper.pdf
    python -m sparx_agency.tools.pdf_parser links paper.pdf
    python -m sparx_agency.tools.pdf_parser crop paper.pdf --page 4 --box 60 90 550 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from sparx_agency.tools.pdf_parser import (
    captions as captions_mod,
    extract as extract_mod,
    fetch as fetch_mod,
    layout as layout_mod,
    links as links_mod,
    page_render,
    poppler,
    text_extract,
)
from sparx_agency.tools.pdf_parser.geometry import BBox
from sparx_agency.tools.pdf_parser.workspace import DEFAULT_ROOT, PaperWorkspace, slugify


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m sparx_agency.tools.pdf_parser",
        description="Extract a research paper's text, figures, tables and algorithms.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("extract", help="fetch if needed, then extract into a workspace")
    run.add_argument("paper", help="arXiv id or URL, a PDF URL, or a local path")
    run.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                     help="parent directory for workspaces (default: %(default)s)")
    run.add_argument("--slug", help="workspace directory name (default: from the title)")
    run.add_argument("--workspace", type=Path,
                     help="exact workspace directory, overriding --root/--slug")
    run.add_argument("--no-pages", action="store_true", help="skip whole-page renders")
    run.add_argument("--no-embedded", action="store_true", help="skip embedded raster images")
    run.add_argument("--keep", action="store_true", help="keep output from a previous run")

    for name, help_text in (
        ("captions", "list the figures, tables and algorithms found"),
        ("links", "list the code and reference links printed in the paper"),
    ):
        query = sub.add_parser(name, help=help_text)
        query.add_argument("pdf", type=Path)
        query.add_argument("--json", action="store_true", help="emit JSON instead of text")

    crop = sub.add_parser("crop", help="render one rectangle of one page to a PNG")
    crop.add_argument("pdf", type=Path)
    crop.add_argument("--page", type=int, required=True, help="1-based page number")
    crop.add_argument("--box", type=float, nargs=4, required=True,
                      metavar=("X0", "Y0", "X1", "Y1"),
                      help="rectangle in PDF points, origin top-left")
    crop.add_argument("--out", type=Path, required=True)
    crop.add_argument("--dpi", type=float, default=page_render.CROP_DPI)

    meta = sub.add_parser("meta", help="fetch title, authors and abstract from arXiv")
    meta.add_argument("arxiv_id")

    return parser


def _resolve_workspace(args: argparse.Namespace, source: fetch_mod.PaperSource) -> PaperWorkspace:
    """Decide where the workspace goes, naming it from the paper if need be."""
    if args.workspace:
        return PaperWorkspace(Path(args.workspace).expanduser())

    slug = args.slug
    if not slug and source.arxiv_id:
        try:
            slug = slugify(str(fetch_mod.arxiv_metadata(source.arxiv_id)["title"]))
        except fetch_mod.DownloadFailed:
            slug = "arxiv-{}".format(source.arxiv_id)
    if not slug:
        slug = slugify(Path(source.value).stem if source.kind == "local" else "paper")
    return PaperWorkspace.for_slug(slug, args.root)


def _report(result: extract_mod.ExtractionResult) -> None:
    """Print a short summary of what an extraction produced."""
    workspace = result.workspace
    print("workspace: {}".format(workspace.root))
    print("pages:     {} ({} rendered)".format(result.document.page_count, len(result.page_images)))
    print("exhibits:  {} figures, {} tables, {} algorithms".format(
        len(result.of_kind(captions_mod.FIGURE)),
        len(result.of_kind(captions_mod.TABLE)),
        len(result.of_kind(captions_mod.ALGORITHM)),
    ))
    if result.links and result.links.repositories:
        print("code:      {}".format(", ".join(result.links.repositories[:3])))
    for warning in result.warnings:
        print("warning:   {}".format(warning), file=sys.stderr)
    print("read next: {}".format(workspace.manifest))


def _cmd_extract(args: argparse.Namespace) -> int:
    """Run the ``extract`` subcommand."""
    source = fetch_mod.resolve_source(args.paper)
    workspace = _resolve_workspace(args, source).create()
    fetch_mod.fetch(source, workspace.pdf)
    result = extract_mod.extract_paper(
        pdf_path=workspace.pdf,
        workspace=workspace,
        render_pages=not args.no_pages,
        extract_embedded=not args.no_embedded,
        clear=not args.keep,
    )
    _report(result)
    return 0


def _cmd_captions(args: argparse.Namespace) -> int:
    """Run the ``captions`` subcommand."""
    found = captions_mod.find_all_captions(layout_mod.load_layout(args.pdf))
    if args.json:
        print(json.dumps([caption.to_dict() for caption in found], indent=2))
        return 0
    for caption in found:
        print("p{:<4} {:<14} {}".format(caption.page, caption.label, caption.text[:90]))
    return 0


def _cmd_links(args: argparse.Namespace) -> int:
    """Run the ``links`` subcommand."""
    found = links_mod.find_links(text_extract.extract_text(args.pdf).full)
    if args.json:
        print(json.dumps(found.to_dict(), indent=2))
        return 0
    if found.is_empty:
        print("no links printed in this paper")
        return 0
    for label, values in found.to_dict().items():
        for value in values:
            print("{:<14} {}".format(label, value))
    return 0


def _cmd_crop(args: argparse.Namespace) -> int:
    """Run the ``crop`` subcommand."""
    x0, y0, x1, y1 = args.box
    page_render.render_region(
        pdf_path=args.pdf,
        page_number=args.page,
        region=BBox(x0, y0, x1, y1),
        out_path=args.out,
        dpi=args.dpi,
    )
    print(args.out)
    return 0


def _cmd_meta(args: argparse.Namespace) -> int:
    """Run the ``meta`` subcommand."""
    print(json.dumps(fetch_mod.arxiv_metadata(args.arxiv_id), indent=2))
    return 0


_COMMANDS = {
    "extract": _cmd_extract,
    "captions": _cmd_captions,
    "links": _cmd_links,
    "crop": _cmd_crop,
    "meta": _cmd_meta,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 on an error that was reported to stderr.
    """
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return _COMMANDS[args.command](args)
    except (
        poppler.PopplerNotInstalled,
        poppler.PopplerFailed,
        fetch_mod.DownloadFailed,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
