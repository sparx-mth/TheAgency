"""Freeze an exact HM3D configuration, then run the published split under it.

    python -m ...hm3d.frozen_eval freeze --lock PATH --note "..." -- [run args]
    python -m ...hm3d.frozen_eval run   --lock PATH            -- [run args]

The lock captures the actual Python sources, the built policy and its converter,
the dataset manifest, the runtime versions, the protocol and the exact ordered
episode selection. ``run`` refuses any drift in any of them, and refuses a
selection that is not the publisher's complete validation split.

**A lock is a reproducibility commitment, not evidence that the split was
unseen.** It cannot know what was tuned against what beforehand; that is what
``--note`` is for, and the note is mandatory precisely because a freeze
performed without one would look like a held-out claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import (
    evaluation_configuration)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import (
    freeze_configuration)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import PUBLISHED_VAL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run import (
    main as run_main, parser as run_parser, prepare)


def _published_selection(args, config, issues):
    """The run's selection, checked to be the whole published split."""
    if issues:
        raise ValueError("; ".join(issues))
    protocol = protocol_for(args.version)
    expected = PUBLISHED_VAL[protocol.dataset_version]
    ids = list(config["selected_episode_ids"])
    if not config["full_split"] or len(ids) != expected["episodes"]:
        raise ValueError(
            "A frozen evaluation is the complete published split: %d episodes "
            "of HM3D-%s %s, with no --scenes, --limit, sharding or exclusion. "
            "This selection has %d." % (expected["episodes"],
                                        protocol.dataset_version, protocol.split,
                                        len(ids)))
    return protocol, ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--note", default="",
                        help="Required for freeze: what this configuration was "
                             "already tuned or developed against")
    known, remainder = parser.parse_known_args(argv)
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    args = run_parser().parse_args(remainder)
    env, policy, _, config, issues = prepare(args)
    try:
        protocol, ids = _published_selection(args, config, issues)
        if known.action == "freeze":
            if not known.note.strip():
                parser.error("--note is required: record what this configuration "
                             "has already been developed or tuned against")
            prepared = evaluation_configuration(
                config, ids, protocol.evaluation_settings(), policy=policy)
            freeze_configuration(prepared, known.lock, development_note=known.note)
            print("Frozen before execution:", known.lock)
            return 0
    finally:
        if env is not None:
            env.close()
    if args.output is None:
        raise ValueError("A frozen execution needs an explicit --output directory")
    result = run_main(remainder, frozen_lock=known.lock, expected_episode_ids=ids)
    if result == 0:
        (args.output / "evaluation_role.json").write_text(json.dumps({
            "role": "frozen_published_validation",
            "lock": str(known.lock.resolve()),
            "benchmark": protocol.benchmark, "split": protocol.split,
            "episodes": len(ids), "held_out_claim": False}, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
