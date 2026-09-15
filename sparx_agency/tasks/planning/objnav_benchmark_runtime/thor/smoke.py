"""Measure, on a live AI2-THOR build, the things that were only read about.

Four facts about the installed Unity build decide whether a RoboTHOR sweep
produces a real number, and none of them can be settled by reading a source
tree — the build is a binary, and the one the challenge pins is four years
older than the Python client that drives it:

1. **Is the pose conversion right?** Drive one action of each kind and put the
   realised motion through the harness's own ``check_motion``. A mirrored or
   transposed frame leaves every path length intact, so this is the only thing
   that can see it.
2. **Is depth in metres?** Move a measured distance along the optical axis and
   check that the centre reading falls by exactly that much. This is
   scene-independent and it catches the real hazard: the pinned 2021 build
   packs depth into three 8-bit channels where a modern build sends float32,
   and the client decodes whichever it is given with a multiplier derived from
   the clip planes. A factor-of-20 scale error here looks like a perfectly
   ordinary depth image.

   (Planar versus radial is **not** measured here, because no scene-content
   test for it is reliable -- a second pixel is usually looking at a different
   surface. It is settled from the build's own shader instead: AI2-THOR
   linearises Unity's ``_CameraDepthTexture``, which stores eye-space ``z``,
   so the depth is optical-frame ``z``.)
3. **What does an empty pixel read?** AI2-THOR returns a finite far-plane
   value, not ``inf`` or ``NaN``. The camera's ``max_depth_m`` has to sit below
   it or the sky maps as a wall.
4. **Where is the camera, and where does a LOOK stop?** The mount height and
   the horizon clamp both changed between the pinned build and ai2thor 5.0.0.

It prints what it measured beside what the protocol declares and exits non-zero
on a disagreement, so it can gate a sweep. It is **not** a benchmark score: it
touches one scene and no episode.

Run it with the RoboTHOR dataset absent; it needs only the build.

Python 3.8 syntax.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark.kinematics import check_motion
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.protocol import (
    PROTOCOL, SCENES,
)


def _reachable_start(controller, samples=12):
    """A published reachable pose with a long clear run straight ahead.

    Not an arbitrary one. The depth-scale calibration needs the agent to
    actually be able to drive toward something, and a randomly chosen
    reachable cell in a furnished apartment is very often wedged against a
    bed or a wall -- which reports a perfectly healthy-looking frame where
    nothing is more than half a metre away and no forward step succeeds.
    Positions come from the simulator's own ``GetReachablePositions``;
    nothing here invents one.
    """
    positions = controller.step({"action": "GetReachablePositions"}
                                ).metadata["actionReturn"]
    if not positions:
        raise RuntimeError("The scene reports no reachable positions")
    stride = max(1, len(positions) // samples)
    best = None
    for position in positions[::stride]:
        for rotation in (0.0, 90.0, 180.0, 270.0):
            event = controller.step({
                "action": "TeleportFull", **position,
                "rotation": {"x": 0.0, "y": rotation, "z": 0.0},
                "horizon": 0.0, "standing": True})
            if not event.metadata.get("lastActionSuccess"):
                continue
            depth = _raw_depth_of(event)
            ahead = float(depth[depth.shape[0] // 2, depth.shape[1] // 2])
            if best is None or ahead > best[0]:
                best = (ahead, position, rotation)
    if best is None:
        raise RuntimeError("No reachable position could be occupied")
    return best[1], best[2]


def measure(scene=SCENES[0], build=PROTOCOL.thor_build_id, platform=None):
    """Drive one scene and report what the build actually does."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.thor.simulator import (
        AI2ThorRGBDSimulator, metric_depth)

    camera = PROTOCOL.camera()
    bridge = AI2ThorRGBDSimulator(
        camera, PROTOCOL.actions(), PROTOCOL.agent_radius_m,
        height_m=PROTOCOL.body_height_m,
        origin_height_m=PROTOCOL.origin_height_m,
        initialize=PROTOCOL.initialize(), commit_id=build,
        width=PROTOCOL.width, height=PROTOCOL.height, platform=platform,
        # Checked below rather than enforced, so the measurement is reported
        # even when it disagrees with the protocol.
        far_plane_sentinel_m=None,
        camera_offset_tolerance_m=float("inf"))
    findings = {"scene": scene, "build": build,
                "platform": getattr(platform, "__name__", str(platform))}
    try:
        bridge._controller = bridge._controller_factory(bridge)
        bridge._controller.reset(scene)
        bridge._scene = scene
        start, bearing = _reachable_start(bridge._controller)
        _, _, pose = bridge.reset(scene, start, bearing, 0.0)
        findings["start"] = dict(start, rotation=bearing)

        metadata = bridge.metadata
        camera_y = metadata.get("cameraPosition", {}).get("y")
        floor_z = pose.z
        findings["camera_height_m"] = (
            None if camera_y is None else round(float(camera_y) - floor_z, 4))
        findings["declared_camera_height_m"] = camera.height_m

        findings.update(_depth_scale(bridge))
        findings.update(_depth_range(bridge, camera, metric_depth))
        # Both of these move the agent, so they run after the calibration that
        # needed the clear run ahead.
        findings["motion"] = _motion_check(bridge, bridge.observe()[2])
        findings["horizon_clamp"] = _horizon_clamp(bridge)
    finally:
        bridge.close()
    findings["problems"] = _problems(findings, camera)
    return findings


def _motion_check(bridge, pose):
    """One action of each kind, judged by the harness's own motion check."""
    spec = PROTOCOL.actions()
    tolerance = PROTOCOL.kinematics()
    results = {}
    for action in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT,
                   DiscreteAction.MOVE_FORWARD, DiscreteAction.LOOK_DOWN,
                   DiscreteAction.LOOK_UP, DiscreteAction.STOP):
        after = bridge.step(action)[2]
        try:
            check_motion(action, pose, after, spec, tolerance)
            results[action.name] = "ok"
        except Exception as exc:  # EnvContractError, reported not raised
            results[action.name] = "REFUSED: %s" % exc
        pose = after
    return results


def _horizon_clamp(bridge):
    """Where a LOOK actually stops, measured by walking into the limit."""
    limits = {}
    for action, key in ((DiscreteAction.LOOK_DOWN, "max_pitch_deg"),
                        (DiscreteAction.LOOK_UP, "min_pitch_deg")):
        reached = math.degrees(bridge.observe()[2].camera_pitch)
        for _ in range(6):
            pitch = math.degrees(bridge.step(action)[2].camera_pitch)
            if not bridge.last_action_succeeded:
                break
            reached = pitch
        limits[key] = round(reached, 3)
    return limits


def _depth_scale(bridge):
    """Move a measured distance and watch the centre reading follow it.

    Planar and radial depth coincide exactly on the optical axis, so this
    cannot tell them apart -- but it is a direct, scene-independent
    calibration of the *units*, which is the thing that actually breaks when a
    client and a build disagree about the depth encoding.
    """
    from sparx_agency.core.planning.objnav.types.actions import DiscreteAction

    samples = []
    _, _, before_pose = bridge.observe()
    raw = _raw_depth(bridge)
    height, width = raw.shape
    before = float(raw[height // 2, width // 2])
    for _ in range(6):
        _, _, after_pose = bridge.step(DiscreteAction.MOVE_FORWARD)
        if not bridge.last_action_succeeded:
            break
        travelled = math.hypot(after_pose.x - before_pose.x,
                               after_pose.y - before_pose.y)
        after = float(_raw_depth(bridge)[height // 2, width // 2])
        closed = before - after
        if travelled > 0.05 and closed > 0.05:
            samples.append(travelled / closed)
        before, before_pose = after, after_pose
    if not samples:
        return {"depth_scale_samples": 0, "depth_metres_per_unit": None}
    return {"depth_scale_samples": len(samples),
            "depth_metres_per_unit": round(sum(samples) / len(samples), 4)}


def _raw_depth(bridge):
    """The build's depth frame, before any of our clipping."""
    return _raw_depth_of(bridge._event)


def _raw_depth_of(event):
    """The same, from a bare event."""
    raw = np.asarray(event.depth_frame, dtype=np.float64)
    return raw[..., 0] if raw.ndim == 3 else raw


def _depth_range(bridge, camera, metric_depth):
    """What the build returns, and what survives our clipping."""
    raw = _raw_depth(bridge)
    converted = metric_depth(raw, camera)
    return {
        "depth_min_seen_m": round(float(raw.min()), 4),
        "depth_max_seen_m": round(float(raw.max()), 4),
        "declared_far_sentinel_m": round(
            PROTOCOL.camera_far_plane_m - PROTOCOL.camera_near_plane_m, 4),
        "converted_finite_fraction": round(
            float(np.isfinite(converted).mean()), 4),
    }


def _problems(findings, camera):
    """Everything measured that disagrees with the declared protocol."""
    problems = []
    for name, result in findings.get("motion", {}).items():
        if result != "ok":
            problems.append("%s: %s" % (name, result))
    measured = findings.get("camera_height_m")
    if measured is not None and abs(measured - camera.height_m) > 0.01:
        problems.append(
            "camera sits %.4f m above the floor, protocol declares %.4f m"
            % (measured, camera.height_m))
    clamp = findings.get("horizon_clamp", {})
    for key, declared in (("max_pitch_deg", PROTOCOL.max_pitch_deg),
                          ("min_pitch_deg", PROTOCOL.min_pitch_deg)):
        if key in clamp and abs(clamp[key] - declared) > 0.5:
            problems.append(
                "%s measured %.1f, protocol declares %.1f -- this build is not "
                "the pinned one" % (key, clamp[key], declared))
    scale = findings.get("depth_metres_per_unit")
    if scale is None:
        problems.append(
            "the depth scale could not be calibrated: no forward step both "
            "moved and closed on a surface. Re-run facing down a clear run")
    elif abs(scale - 1.0) > 0.05:
        problems.append(
            "depth is NOT in metres: one metre of travel closes %.4f units, so "
            "this client decodes this build's depth encoding differently "
            "(the 2021 build packs three 8-bit channels, a modern one sends "
            "float32)" % (1.0 / scale))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=SCENES[0])
    parser.add_argument("--thor-build", default=PROTOCOL.thor_build_id)
    parser.add_argument("--platform", choices=("reference-linux64",
                                               "cloud-rendering"),
                        default="reference-linux64")
    args = parser.parse_args(argv)
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.run import (
        thor_platform)
    findings = measure(args.scene, args.thor_build, thor_platform(args.platform))
    print(json.dumps(findings, indent=2, sort_keys=True))
    print("\nThis is a build check on one scene, not a benchmark score.")
    return 1 if findings["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
