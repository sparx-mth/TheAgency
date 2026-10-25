"""Exploration-only probe of one config on real HM3D episodes: a fake detector and LLM, so only the simulator runs.

The target is never found, so every episode spends its whole budget exploring; ``summarize_probe.py`` reads the
observed area and the prescan record. SR and SPL mean nothing here.
"""
from __future__ import annotations

import sys

from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import run_evaluation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.falcon_only import FakeDetector, FakeLLM
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run import parser, prepare


def main(argv=None):
    args = parser().parse_args(["--explorer", "frontier", "--lean", *(sys.argv[1:] if argv is None else argv)])
    protocol = protocol_for(args.version).with_split(args.split)
    env, policy, label_mapper, config, issues = prepare(args)
    if issues:
        print("not ready:", *issues, sep="\n  ")
        return 1
    policy.detector, policy.llm_client = FakeDetector(), FakeLLM()
    config["method"]["detector"] = {"backend": "fake", "note": "exploration-only probe"}
    config["method"]["llm_identity"] = {"backend": "fake", "note": "exploration-only probe"}
    run_evaluation(env, policy, label_mapper, output=args.output, config=config, settings=protocol.evaluation_settings(),
                   episode_ids=tuple(config["selected_episode_ids"]), record=args.record, video_fps=args.video_fps,
                   save_frames=args.save_frames,
                   progress=lambda index, total, row: print("%d/%d %s steps=%d" % (index, total, row.episode_id, row.steps),
                                                            flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
