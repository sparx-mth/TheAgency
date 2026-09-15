"""Visualization-only smooth replay of one recorded MP3D episode.

Interpolated frames never enter policy observations, actions or scoring: this
writes a separate ``smooth_rgb.mp4`` beside the original recording and touches
nothing else. Run it only after the evaluation has released the GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.replay import render_replay
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import MP3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import PROTOCOL


class SceneRenderer:
    """Render an arbitrary pose in one scene, with the protocol's camera."""

    def __init__(self, mesh, navmesh, gpu_device=0):
        self.simulator = HabitatRGBDSimulator(
            PROTOCOL.camera(), PROTOCOL.actions(), PROTOCOL.agent_radius_m,
            PROTOCOL.allow_sliding, gpu_device, height_m=PROTOCOL.agent_height_m)
        self.mesh, self.navmesh = mesh, navmesh
        self._ready = False

    def __call__(self, pose):
        import habitat_sim
        import quaternion

        position = np.array([-pose.y, pose.z, -pose.x], dtype=np.float32)
        body = quaternion.from_rotation_vector([0.0, pose.yaw, 0.0])
        if not self._ready:
            self.simulator.reset(self.mesh, self.navmesh, position,
                                 quaternion.as_float_array(body), 0)
            self._ready = True
        agent = self.simulator._sim.get_agent(0)
        # get_state() hands back a fresh copy each call, so the tilt has to be
        # written into the SAME object that is then set, and sensor states have
        # to carry their own pose once inference is turned off.
        state = agent.get_state()
        state.position = position
        state.rotation = body
        # REP-103 pitch is positive downward; a rotation of +a about the
        # camera's local +x tilts its -z forward axis UP by a, so negate.
        tilt = quaternion.from_rotation_vector([-pose.camera_pitch, 0.0, 0.0])
        mount = np.array([0.0, self.simulator.camera.height_m, 0.0], dtype=np.float32)
        for sensor_state in state.sensor_states.values():
            sensor_state.position = position + mount
            sensor_state.rotation = body * tilt
        agent.set_state(state, reset_sensors=False, infer_sensor_states=False)
        raw = self.simulator._sim.get_sensor_observations()
        return np.asarray(raw["rgb"])[..., :3].copy()

    def close(self):
        self.simulator.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="one recordings/<episode> directory")
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--subframes", type=int, default=4)
    args = parser.parse_args(argv)
    metrics = json.loads((args.recording / "metrics.json").read_text())
    scene = metrics["scene_id"]
    dataset = MP3DDataset(args.episodes_dir, args.scenes_dir, full=False, scene=scene)
    episode = next(e for e in dataset.episodes.values() if e.scene == scene)
    renderer = SceneRenderer(*dataset.scene_paths(episode), gpu_device=args.gpu_device)
    try:
        output = render_replay(args.recording, renderer, subframes=args.subframes)
    finally:
        renderer.close()
    print("Wrote", output)


if __name__ == "__main__":
    raise SystemExit(main())
