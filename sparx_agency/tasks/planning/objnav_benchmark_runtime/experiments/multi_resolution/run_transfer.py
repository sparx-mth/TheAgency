"""Explore at a coarse resolution, carry the map sparsely into a finer run, explore again."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from sparx_agency.core.planning.exploration.prior_map import REPRESENTATIVES, sparsify_to_fine
from sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.run_gt_falcon import (
    resolve_scene_assets)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.transfer_eval import (
    navigable_lattice_enu)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.transfer_figures import (
    write_transfer_overview)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.transfer_phase import (
    MAP_FILE, PhaseConfig, load_prior, run_phase)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import HM3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings

HERE = Path(__file__).parent


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--episodes-dir", type=Path, required=True)
    command.add_argument("--scene-release-root", type=Path, required=True)
    command.add_argument("--scene-directory", default="hm3d")
    command.add_argument("--scene", action="append", required=True)
    command.add_argument("--version", choices=("v1", "v2"), default="v2")
    command.add_argument("--split", default="train")
    command.add_argument("--coarse-profile", type=Path, default=HERE / "prescan_0p5.json")
    command.add_argument("--fine-profile", type=Path, default=HERE / "fine_0p25.json")
    command.add_argument("--mode", choices=("transfer", "coarse", "cold"), default="transfer",
                         help="coarse: prescan only; cold: fine run without a prior")
    command.add_argument("--representative", choices=REPRESENTATIVES, default="center")
    command.add_argument("--coarse-max-actions", type=int, default=600)
    command.add_argument("--fine-max-actions", type=int, default=1200)
    command.add_argument("--bootstrap-turns", type=int, default=12)
    command.add_argument("--lattice-step-m", type=float, default=0.25)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--gpu-device", type=int, default=0)
    command.add_argument("--output-dir", type=Path, required=True)
    return command


def load_settings(path):
    return RPTSettings(**json.loads(Path(path).read_text(encoding="utf-8")))


def make_simulator(protocol, gpu_device):
    return HabitatRGBDSimulator(protocol.camera(), protocol.actions(), protocol.agent_radius_m,
                                protocol.allow_sliding, gpu_device, height_m=protocol.agent_height_m,
                                navmesh=protocol.navmesh)


def pick_episodes(dataset, scenes):
    chosen = []
    for scene in scenes:
        found = next((row.episode_id for row in dataset.episodes.values() if row.scene_key == scene), None)
        if found is None:
            raise ValueError("No episode found for scene %r" % scene)
        chosen.append(found)
    return chosen


def run_episode(args, dataset, protocol, episode_id, coarse_settings, fine_settings):
    row = dataset.episodes[episode_id]
    scene, navmesh = resolve_scene_assets(row, args.scene_release_root, args.scene_directory)
    out = args.output_dir / row.scene_key
    simulator = make_simulator(protocol, args.gpu_device)
    try:
        simulator.reset(scene, navmesh, row.start_position, row.start_rotation_wxyz, args.seed)
        gt = navigable_lattice_enu(simulator.pathfinder, row.start_position[1], row.start_position, args.lattice_step_m)
        phase = lambda label, settings, limit, prior: run_phase(  # noqa: E731
            simulator, protocol, row, scene, navmesh, PhaseConfig(label, settings, limit, args.bootstrap_turns),
            prior, out / label, args.seed, gt)
        summary = {"episode_id": episode_id, "scene": row.scene_key, "mode": args.mode,
                   "representative": args.representative, "gt_lattice_points": len(gt)}
        prior = None
        if args.mode != "cold":
            summary["coarse"] = phase("coarse", coarse_settings, args.coarse_max_actions, None)
        if args.mode == "transfer":
            coarse_map = load_prior(out / "coarse" / MAP_FILE)
            prior = sparsify_to_fine(coarse_map, fine_settings.map_resolution_m, args.representative, args.seed)
        if args.mode != "coarse":
            summary["fine"] = phase("fine", fine_settings, args.fine_max_actions, prior)
    finally:
        simulator.close()
    if args.mode == "transfer":
        write_transfer_overview(out / "transfer_overview.png", [("Coarse prescan", coarse_map), ("Sparse prior", prior),
                                                                ("Fine result", load_prior(out / "fine" / MAP_FILE))])
    (out / "transfer.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def headline(summary):
    keys = ("actions", "termination", "wall_s")
    row = {"scene": summary["scene"], "mode": summary["mode"]}
    for phase in ("coarse", "fine"):
        if phase in summary:
            data = summary[phase]
            row[phase] = {k: data[k] for k in keys}
            row[phase]["observed"] = data["coverage"]["observed_coverage"]
    return row


def main(argv=None):
    args = parser().parse_args(argv)
    protocol = protocol_for(args.version).with_split(args.split)
    dataset = HM3DDataset(args.episodes_dir, None, protocol, full=False, scenes=args.scene,
                          require_scenes=False, episodes_per_scene=1)
    coarse_settings, fine_settings = load_settings(args.coarse_profile), load_settings(args.fine_profile)
    if coarse_settings.map_size_m != fine_settings.map_size_m:
        raise ValueError("coarse and fine profiles must share map_size_m so their frames nest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for episode_id in pick_episodes(dataset, args.scene):
        print(json.dumps(headline(run_episode(args, dataset, protocol, episode_id, coarse_settings, fine_settings)), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
