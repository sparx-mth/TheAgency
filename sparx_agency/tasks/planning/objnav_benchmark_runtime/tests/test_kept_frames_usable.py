import json

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import kept_frames

GOOD = {"near": 0.1, "inside": 0.9, "floor_share": 0.5, "entrance_m": 2.0}
CLOSE_UP = {"near": 0.9, "inside": 0.9, "floor_share": 0.5, "entrance_m": 2.0}


def test_only_usable_drops_close_ups_and_keeps_every_frame_otherwise(tmp_path):
    folder = tmp_path / "scene"
    folder.mkdir()
    entry = {"frames": {"a": GOOD, "b": CLOSE_UP}, "methods": {"sweep_all": ["a", "b"]}}
    (folder / "selection.json").write_text(json.dumps({"room1": entry}))
    assert kept_frames(tmp_path, 1, ("sweep_all",))["room1"][1] == ["a", "b"]
    assert kept_frames(tmp_path, 1, ("sweep_all",), only_usable=True)["room1"][1] == ["a"]
