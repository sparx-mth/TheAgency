"""The agreement filter: confirmed target-kind boxes stay, unconfirmed ones go, other labels pass."""
from collections import namedtuple

from sparx_agency.tasks.planning.objnav_benchmark_runtime.trace_tools.agreement import agreeing

D = namedtuple("D", "cls conf xyxy")
BOX = (100.0, 100.0, 200.0, 200.0)


def test_a_confirmed_box_stays_and_an_unconfirmed_one_goes():
    first = [D("bed", 0.9, BOX), D("chair", 0.9, (300.0, 300.0, 400.0, 400.0))]
    assert [d.cls for d in agreeing(first, [D("bed", 0.5, BOX)])] == ["bed"]


def test_synonyms_of_one_kind_confirm_each_other():
    assert agreeing([D("sofa", 0.9, BOX)], [D("couch", 0.6, BOX)])


def test_a_weak_partner_does_not_confirm():
    assert not agreeing([D("bed", 0.9, BOX)], [D("bed", 0.1, BOX)])


def test_labels_that_are_not_target_kinds_pass_through():
    assert [d.cls for d in agreeing([D("door", 0.4, BOX), D("cabinet", 0.9, BOX)], [])] == ["door", "cabinet"]
