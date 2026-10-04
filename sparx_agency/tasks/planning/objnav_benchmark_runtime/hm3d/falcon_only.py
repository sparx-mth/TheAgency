"""Standalone FALCON exploration, no detector/LLM traffic.

Dev tool for the multi-resolution FALCON pre-scan work (see repo-root
HANDOFF.md): reuses ``hm3d.run``'s own dataset/env/policy/config assembly via
``prepare()``, then swaps the policy's detector/llm_client for inline fakes
*before* ``reset()`` runs -- so no real YOLO/Ollama HTTP calls happen during
the episode, and no model service needs to be reachable at all except for the
one-time health check ``prepare()`` performs on the real (discarded) client.

This is pure exploration, not a benchmark run: SR/SPL are meaningless here
(the fake detector never reports a target), only the agent_info exploration
trace (coverage, kind/action histogram, when FALCON itself declares the
bounded search exhausted) matters.

Example, one coarse pass:
    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.falcon_only \\
        --scenes 00877-4ok3usBNeis --cell-size-m 8 --voxel-m 0.5 --scope-radius-m 35 \\
        --output ~/objnav_benchmark/hm3d_v2/falcon_only_00877 --record
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile

from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import run_evaluation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run import parser, prepare


class FakeLLM:
    """Neutral room classifier + node oracle; no network call."""

    def chat_json(self, system, user, **kwargs):
        if "Room observed objects" in user:
            return {"label": "unknown", "confidence": 0.0, "reasoning": "exploration-only run"}
        ids = [int(nid) for nid in re.findall(r"^id=(\d+)", user, re.MULTILINE)]
        share = max(1, int(round(90 / max(1, len(ids)))))
        return {"nodes": [{"id": nid, "why": "no opinion", "p": share} for nid in ids],
                "elsewhere": max(0, 100 - share * len(ids))}


class FakeDetector:
    def detect(self, rgb):
        return []


def falcon_only_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell-size-m", type=float, default=8.0,
                   help="FalconParams.cell_size_m -- the connectivity decomposition's coarse cell size")
    p.add_argument("--voxel-m", type=float, default=0.5,
                   help="RPTSettings.map_resolution_m -- the observed-map grid resolution")
    p.add_argument("--scope-radius-m", type=float, default=35.0,
                   help="FalconParams.scope_radius_m -- the anchored local envelope's radius")
    return p


def main(argv=None):
    own, rest = falcon_only_parser().parse_known_args(argv)
    overrides = {"map_resolution_m": own.voxel_m, "target_closing_enabled": False,
                 "falcon": {"cell_size_m": own.cell_size_m, "scope_radius_m": own.scope_radius_m}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(overrides, handle)
        policy_config_path = handle.name
    args = parser().parse_args(["--explorer", "falcon", "--policy-config", policy_config_path, *rest])
    protocol = protocol_for(args.version).with_split(args.split)
    env, policy, label_mapper, config, issues = prepare(args)
    if issues:
        print(json.dumps({"ready": False, "issues": issues}, indent=2))
        return 1

    policy.detector = FakeDetector()
    policy.llm_client = FakeLLM()
    config["method"]["detector"] = {"backend": "fake", "note": "exploration-only run, no real detector"}
    config["method"]["llm_identity"] = {"backend": "fake", "note": "exploration-only run, no real LLM"}
    selected = tuple(config["selected_episode_ids"])

    def progress(index, total_count, row):
        print("%d/%d %s %s SR=%d SPL=%.3f DTG=%s SoftSPL=%.3f (SR/SPL meaningless -- fake detector)"
              % (index, total_count, row.episode_id, row.target_category,
                 row.success, row.spl, row.distance_to_goal_m, row.soft_spl), flush=True)

    summary = run_evaluation(
        env, policy, label_mapper, output=args.output, config=config,
        settings=protocol.evaluation_settings(), episode_ids=selected,
        record=args.record, video_fps=args.video_fps, progress=progress)
    print("DONE", summary.overall.n_episodes, "episode(s) ->", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
