"""Working out what a pasted paper reference is. No network is used here."""
from __future__ import annotations

from pathlib import Path

import pytest

from sparx_agency.tools.pdf_parser import fetch


def test_a_bare_arxiv_id():
    source = fetch.resolve_source("1706.03762")
    assert source.kind == "arxiv"
    assert source.arxiv_id == "1706.03762"
    assert source.pdf_url == "https://arxiv.org/pdf/1706.03762"


def test_a_versioned_arxiv_id():
    assert fetch.resolve_source("2504.08838v2").arxiv_id == "2504.08838"


def test_an_abstract_url_yields_the_pdf_url():
    source = fetch.resolve_source("https://arxiv.org/abs/1706.03762")
    assert source.pdf_url == "https://arxiv.org/pdf/1706.03762"
    assert source.abstract_url == "https://arxiv.org/abs/1706.03762"


def test_a_plain_pdf_url_is_used_as_given():
    source = fetch.resolve_source("https://example.org/papers/navdp.pdf")
    assert source.kind == "url"
    assert source.pdf_url == "https://example.org/papers/navdp.pdf"
    assert source.abstract_url is None


def test_a_local_file(tmp_path: Path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4")
    source = fetch.resolve_source(str(path))
    assert source.kind == "local"
    assert source.pdf_url is None


def test_an_unknown_string_is_rejected():
    with pytest.raises(ValueError):
        fetch.resolve_source("some paper I half remember")


def test_an_empty_string_is_rejected():
    with pytest.raises(ValueError):
        fetch.resolve_source("   ")


def test_a_local_file_that_is_not_a_pdf_is_refused(tmp_path: Path):
    origin = tmp_path / "notes.pdf"
    origin.write_text("<html>paywall</html>", encoding="utf-8")
    with pytest.raises(ValueError):
        fetch.fetch(fetch.resolve_source(str(origin)), tmp_path / "out.pdf")


def test_a_local_pdf_is_copied_into_the_workspace(tmp_path: Path):
    origin = tmp_path / "origin.pdf"
    origin.write_bytes(b"%PDF-1.4 body")
    destination = tmp_path / "ws" / "paper.pdf"
    fetch.fetch(fetch.resolve_source(str(origin)), destination)
    assert destination.read_bytes() == b"%PDF-1.4 body"


def test_copying_a_file_onto_itself_is_harmless(tmp_path: Path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4 body")
    fetch.fetch(fetch.resolve_source(str(path)), path)
    assert path.read_bytes() == b"%PDF-1.4 body"
