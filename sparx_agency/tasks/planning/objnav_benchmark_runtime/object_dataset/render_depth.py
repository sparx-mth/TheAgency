"""Re-render the depth of every view of the object dataset from its stored pose (``depth/NNNNN.png``, millimetres, uint16), checking the rgb matches."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.sample_frames import yaw_quaternion_wxyz
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import is_held_out

DEPTH_MM_MAX = 65000
MAX_MEAN_RGB_DIFF = 6.0


def habitat_position(view):
    """The stored pose is the public ENU one: x = -habitat z, y = -habitat x, z = habitat y."""
    return np.array([-view["y"], view["z"], -view["x"]])


def is_complete(folder):
    return (folder / "depth_check.json").exists() and (folder / "depth").exists() and len(list((folder / "depth").iterdir())) == len((folder / "views.jsonl").read_text().splitlines())


def render_scene(scene, folder, scenes, protocol, seed):
    stem = scene.split("-", 1)[1]
    glb, navmesh = (scenes / "hm3d" / "train" / scene / (stem + suffix) for suffix in (".basis.glb", ".basis.navmesh"))
    simulator = HabitatRGBDSimulator(protocol.camera(), protocol.actions(), protocol.agent_radius_m, protocol.allow_sliding, 0,
                                     height_m=protocol.agent_height_m, navmesh=protocol.navmesh, depth_holes=protocol.depth_holes())
    (folder / "depth").mkdir(exist_ok=True)
    differences = {}
    for line in (folder / "views.jsonl").read_text().splitlines():
        view = json.loads(line)
        rgb, depth, _ = simulator.reset(glb, navmesh, habitat_position(view), yaw_quaternion_wxyz(view["yaw"]), seed)
        stored = cv2.cvtColor(cv2.imread(str(folder / "frames" / view["frame"])), cv2.COLOR_BGR2RGB)
        differences[view["frame"]] = float(np.abs(stored.astype(float) - rgb.astype(float)).mean())
        millimetres = np.clip(np.nan_to_num(depth.astype(np.float64)) * 1000, 0, DEPTH_MM_MAX).astype(np.uint16)
        cv2.imwrite(str(folder / "depth" / (view["frame"].replace(".jpg", ".png"))), millimetres)
    simulator.close()
    (folder / "depth_check.json").write_text(json.dumps(differences))
    return differences


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"))
    p.add_argument("--held-out-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    for folder in sorted(d for d in args.views.iterdir() if (d / "views.jsonl").exists()):
        if is_held_out(folder.name, args.held_out_fraction) or is_complete(folder):
            continue
        differences = render_scene(folder.name, folder, Path(args.scenes_dir), protocol, args.seed)
        mismatched = sum(d > MAX_MEAN_RGB_DIFF for d in differences.values())
        print("%s: %d of %d views differ from the stored frame by more than %.0f" % (folder.name, mismatched, len(differences), MAX_MEAN_RGB_DIFF), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
