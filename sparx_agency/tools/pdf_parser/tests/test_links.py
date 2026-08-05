"""Reading the repository link out of the paper's own text."""
from __future__ import annotations

from sparx_agency.tools.pdf_parser.links import find_links


def test_a_github_link_is_found_and_normalised():
    found = find_links("Code is available at github.com/OpenGVLab/NavDP for reproduction.")
    assert found.repositories == ["https://github.com/OpenGVLab/NavDP"]


def test_a_trailing_full_stop_is_not_part_of_the_address():
    found = find_links("See https://github.com/tensorflow/tensor2tensor.")
    assert found.repositories == ["https://github.com/tensorflow/tensor2tensor"]


def test_a_url_split_across_lines_is_repaired():
    """Typesetters break long URLs at the right margin."""
    found = find_links("available at https://github.com/OpenGVLab/\nNavDP and elsewhere")
    assert "https://github.com/OpenGVLab/NavDP" in found.repositories


def test_the_next_heading_is_not_welded_onto_the_repository_name():
    """The line-repair pass glues the following heading on unless it is trimmed."""
    found = find_links("code at https://github.com/tensorflow/tensor2tensor.\nAcknowledgements")
    assert found.repositories == ["https://github.com/tensorflow/tensor2tensor"]


def test_huggingface_counts_as_a_release():
    found = find_links("Weights: https://huggingface.co/facebook/dinov2-base")
    assert found.repositories == ["https://huggingface.co/facebook/dinov2-base"]


def test_arxiv_ids_are_found_in_both_forms():
    found = find_links("arXiv:1706.03762v5 and https://arxiv.org/abs/2504.08838")
    assert found.arxiv_ids == ["1706.03762", "2504.08838"]


def test_dois_are_found():
    assert find_links("doi 10.1109/ICRA.2019.8794198 here").dois == [
        "10.1109/ICRA.2019.8794198"
    ]


def test_repository_hosts_are_not_repeated_as_project_pages():
    found = find_links("https://github.com/a/b and https://someproject.github.io/site")
    assert found.repositories == ["https://github.com/a/b"]
    assert found.project_pages == ["https://someproject.github.io/site"]


def test_duplicates_are_collapsed():
    found = find_links("github.com/a/b appears twice: github.com/a/b")
    assert found.repositories == ["https://github.com/a/b"]


def test_a_paper_with_no_links():
    found = find_links("This paper releases no code and cites nothing by URL.")
    assert found.is_empty
    assert found.to_dict()["repositories"] == []
