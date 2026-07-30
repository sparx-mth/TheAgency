"""Workspace naming and layout."""
from __future__ import annotations

from pathlib import Path

from sparx_agency.tools.pdf_parser.workspace import PaperWorkspace, slugify


def test_slugify_shortens_and_hyphenates():
    assert slugify("Attention Is All You Need") == "attention-is-all-you-need"


def test_slugify_strips_punctuation_and_caps_length():
    slug = slugify("NavDP: Learning Sim-to-Real Navigation Diffusion Policy!")
    assert slug == "navdp-learning-sim-to-real-navigation"
    assert " " not in slug


def test_slugify_never_returns_empty():
    assert slugify("!!!") == "paper"


def test_every_path_sits_under_the_root(tmp_path: Path):
    workspace = PaperWorkspace.for_slug("demo", tmp_path)
    for path in (
        workspace.pdf,
        workspace.manifest,
        workspace.full_text,
        workspace.page_text_dir,
        workspace.page_image_dir,
        workspace.figure_dir,
        workspace.embedded_dir,
        workspace.table_dir,
        workspace.pseudocode_dir,
        workspace.captions_json,
        workspace.links_json,
    ):
        assert str(path).startswith(str(tmp_path / "demo"))


def test_create_makes_every_directory(tmp_path: Path):
    workspace = PaperWorkspace(tmp_path / "ws").create()
    assert all(directory.is_dir() for directory in workspace.directories())


def test_clear_outputs_keeps_the_pdf(tmp_path: Path):
    workspace = PaperWorkspace(tmp_path / "ws").create()
    workspace.pdf.write_bytes(b"%PDF-1.4 pretend")
    (workspace.figure_dir / "figure-1.png").write_bytes(b"stale")
    workspace.manifest.write_text("stale", encoding="utf-8")

    workspace.clear_outputs()

    assert workspace.pdf.read_bytes() == b"%PDF-1.4 pretend"
    assert not (workspace.figure_dir / "figure-1.png").exists()
    assert not workspace.manifest.exists()
    assert workspace.figure_dir.is_dir()


def test_relative_paths_are_reported_from_the_root(tmp_path: Path):
    workspace = PaperWorkspace(tmp_path / "ws").create()
    assert workspace.relative(workspace.figure_dir / "figure-1.png") == "figures/figure-1.png"


def test_relative_falls_back_for_an_outside_path(tmp_path: Path):
    workspace = PaperWorkspace(tmp_path / "ws").create()
    outside = tmp_path / "elsewhere.png"
    assert workspace.relative(outside) == str(outside)
