import json

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.label_app import Rooms


def scene_folder(tmp_path, views):
    folder = tmp_path / "00001-abc"
    for sub in ("frames", "depth"):
        (folder / sub).mkdir(parents=True)
    room = {"room": "00001_r1", "scene": "00001-abc", "region": 1, "area_m2": 9.0, "floor_y": 0.0, "objects": []}
    (folder / "rooms.json").write_text(json.dumps([room]))
    np.savez_compressed(folder / "cells.npz", cells=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]), regions=np.array([1, 1]))
    for view in views:
        cv2.imwrite(str(folder / "frames" / (view["frame"] + ".jpg")), np.zeros((8, 8, 3), np.uint8))
        cv2.imwrite(str(folder / "depth" / (view["frame"] + ".png")), np.full((8, 8), 3000, np.uint16))
    (folder / "views.jsonl").write_text("".join(json.dumps(v) + "\n" for v in views))


def preselected(n):
    return [{"room": "00001_r1", "scene": "00001-abc", "region": 1, "frame": "v%d" % i, "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "seen_cells": 5} for i in range(n)]


def test_preselected_views_are_shown_in_their_given_order_without_the_annotation_filter(tmp_path):
    scene_folder(tmp_path, preselected(3))
    rooms = Rooms(tmp_path, None, selected=True)
    assert rooms.rooms[0]["frames"] == ["/frame/00001-abc/v%d.jpg" % i for i in range(3)] and not rooms.rooms[0]["mixed"]


def test_at_most_six_preselected_views_are_shown(tmp_path):
    scene_folder(tmp_path, preselected(8))
    assert len(Rooms(tmp_path, None, selected=True).rooms[0]["frames"]) == 6
