"""Real-renderer smoke on a temporary synthetic room, NEVER a benchmark result.

Requires habitat-sim and a free rendering GPU. No downloaded scene, semantic
annotation, model or ROS is used. Navmesh generation here is only for this
synthetic fixture; the benchmark itself always loads the published navmesh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import check_motion
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator


def run(gpu_device=0):
    """Render RGB-D and verify turns/forward/STOP against shared motion checks."""
    memory = subprocess.run(
        ["nvidia-smi", "--id=%d" % gpu_device, "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    if int(memory.stdout.strip()) > 512:
        raise RuntimeError("Renderer smoke requires a free GPU; no ownership override")
    import habitat_sim

    camera, actions = PROTOCOL.camera(), PROTOCOL.actions()
    bridge = HabitatRGBDSimulator(camera, actions, PROTOCOL.agent_radius_m,
                                  PROTOCOL.allow_sliding, gpu_device)
    with tempfile.TemporaryDirectory(prefix="objnav-renderer-") as tmp:
        scene, mesh = Path(tmp) / "room.obj", Path(tmp) / "room.navmesh"
        vertices = [(-4, 0, -4), (4, 0, -4), (4, 0, 4), (-4, 0, 4),
                    (-4, 3, -4), (4, 3, -4), (4, 3, 4), (-4, 3, 4)]
        faces = [(1, 3, 2), (1, 4, 3), (1, 2, 6), (1, 6, 5),
                 (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8),
                 (2, 3, 7), (2, 7, 6)]
        scene.write_text("".join("v %g %g %g\n" % v for v in vertices)
                         + "".join("f %d %d %d\n" % f for f in faces))
        sim = habitat_sim.Simulator(bridge._configuration(scene))
        try:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_height = camera.height_m
            settings.agent_radius = PROTOCOL.agent_radius_m
            if not sim.recompute_navmesh(sim.pathfinder, settings):
                raise RuntimeError("Cannot build synthetic smoke navmesh")
            sim.pathfinder.save_nav_mesh(str(mesh))
            position = list(sim.pathfinder.snap_point([0.0, 0.0, 0.0]))
        finally:
            sim.close()
        try:
            frame = bridge.reset(scene, mesh, position, (1, 0, 0, 0), seed=0)
            initial = ObjNavObservation(*frame, camera, "chair", 0)
            centre_depth = float(initial.depth_m[camera.intrinsics.height // 2,
                                                camera.intrinsics.width // 2])
            if not 3.9 < centre_depth < 4.1:
                raise AssertionError("Synthetic wall should be 4 m ahead, got %s" % centre_depth)
            before = initial.pose
            sequence = (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT,
                        DiscreteAction.MOVE_FORWARD, DiscreteAction.STOP)
            for step, action in enumerate(sequence, 1):
                frame = bridge.step(action)
                observation = ObjNavObservation(*frame, camera, "chair", step)
                check_motion(action, before, observation.pose, actions, PROTOCOL.kinematics())
                before = observation.pose
            if not np.isclose(before.x - initial.pose.x, actions.forward_step_m, atol=1e-4):
                raise AssertionError("Forward action did not advance in public ENU +x")
            return {"synthetic_renderer_smoke": "passed", "benchmark_result": False,
                    "habitat_sim": habitat_sim.__version__, "wall_depth_m": centre_depth,
                    "actions_checked": [a.name for a in sequence],
                    "rgb_shape": list(initial.rgb.shape)}
        finally:
            bridge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-device", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.gpu_device), indent=2))


if __name__ == "__main__":
    main()
