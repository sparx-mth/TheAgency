import csv

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.label_store import ROOM_TYPES, LabelStore


def test_a_label_is_kept_and_read_back(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    store.set("00009_r7", "kitchen")
    assert store.get("00009_r7") == "kitchen" and store.get("00009_r8") is None


def test_labels_survive_a_restart(tmp_path):
    LabelStore(tmp_path / "labels.csv").set("00009_r7", "kitchen")
    assert LabelStore(tmp_path / "labels.csv").all() == {"00009_r7": "kitchen"}


def test_a_new_label_replaces_the_old_one(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    store.set("00009_r7", "kitchen")
    store.set("00009_r7", "living room")
    assert LabelStore(tmp_path / "labels.csv").all() == {"00009_r7": "living room"}


def test_the_file_is_a_csv_with_one_row_per_room(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    store.set("a", "bedroom")
    store.set("b", "unsure")
    rows = list(csv.reader(open(tmp_path / "labels.csv")))
    assert rows == [["room", "room_type"], ["a", "bedroom"], ["b", "unsure"]]


def test_only_listed_room_types_are_accepted(tmp_path):
    with pytest.raises(ValueError):
        LabelStore(tmp_path / "labels.csv").set("a", "spaceship")


def test_the_list_has_the_ways_out_and_no_duplicates():
    assert "unsure" in ROOM_TYPES and "not a room" in ROOM_TYPES and len(set(ROOM_TYPES)) == len(ROOM_TYPES)
