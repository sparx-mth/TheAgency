"""Finding columns from the empty stripes between them."""
from __future__ import annotations

from typing import List

from sparx_agency.tools.pdf_parser import columns
from sparx_agency.tools.pdf_parser.geometry import BBox


def word(x: float, width: float = 30.0) -> BBox:
    """A word box at ``x``, on an arbitrary but consistent line."""
    return BBox(x, 100.0, x + width, 110.0)


def test_one_column_when_there_is_no_gutter():
    boxes = [word(x) for x in (60.0, 95.0, 130.0, 165.0)]
    assert len(columns.detect_column_spans(boxes, min_gap_pt=8.0)) == 1


def test_a_wide_gap_splits_columns():
    boxes = [word(60.0), word(95.0), word(300.0), word(335.0)]
    spans = columns.detect_column_spans(boxes, min_gap_pt=8.0)
    assert len(spans) == 2
    assert spans[0][1] < spans[1][0]


def test_narrow_spans_are_dropped():
    """A stray mark far from the text is not a column."""
    boxes = [word(60.0, 200.0), word(500.0, 4.0)]
    spans = columns.detect_column_spans(boxes, min_gap_pt=8.0, min_column_pt=20.0)
    assert len(spans) == 1


def test_empty_input_gives_no_columns():
    assert columns.detect_column_spans([]) == []


def _table_rows() -> List[List[BBox]]:
    """Three data rows in two columns, under one heading that spans both."""
    return [
        [word(60.0, 200.0), word(300.0, 200.0)],  # a spanning heading
        [word(60.0, 40.0), word(300.0, 40.0)],
        [word(60.0, 40.0), word(300.0, 40.0)],
        [word(60.0, 40.0), word(300.0, 40.0)],
    ]


def test_a_spanning_heading_does_not_merge_two_columns():
    """The case that breaks a strict 'empty on every row' rule."""
    rows = _table_rows()
    rows[0] = [word(60.0, 440.0)]  # one heading centred across both columns
    spans = columns.detect_column_spans_from_rows(rows, min_gap_pt=8.0)
    assert len(spans) == 2


def test_zero_tolerance_does_merge_them():
    """Confirms the tolerance is what recovers the columns, not something else."""
    rows = _table_rows()
    rows[0] = [word(60.0, 440.0)]
    spans = columns.detect_column_spans_from_rows(
        rows, min_gap_pt=8.0, max_bridge_fraction=0.0
    )
    assert len(spans) == 1


def test_column_index_assigns_by_overlap():
    spans = [(50.0, 250.0), (300.0, 500.0)]
    assert columns.column_index(spans, word(60.0)) == 0
    assert columns.column_index(spans, word(310.0)) == 1


def test_column_index_falls_back_to_nearest_centre():
    """A cell straddling a gutter lands in one column rather than being lost."""
    spans = [(50.0, 100.0), (400.0, 500.0)]
    assert columns.column_index(spans, BBox(120.0, 0.0, 140.0, 10.0)) == 0


def test_column_index_without_columns():
    assert columns.column_index([], word(10.0)) == -1


def test_spans_covering_takes_every_touched_column():
    """A full-width caption must yield a full-width region."""
    spans = [(50.0, 250.0), (300.0, 500.0)]
    assert columns.spans_covering(spans, BBox(60.0, 0.0, 480.0, 10.0)) == (50.0, 500.0)
    assert columns.spans_covering(spans, BBox(60.0, 0.0, 200.0, 10.0)) == (50.0, 250.0)


def test_spans_covering_falls_back_to_the_box():
    box = BBox(600.0, 0.0, 700.0, 10.0)
    assert columns.spans_covering([(50.0, 250.0)], box) == (600.0, 700.0)
