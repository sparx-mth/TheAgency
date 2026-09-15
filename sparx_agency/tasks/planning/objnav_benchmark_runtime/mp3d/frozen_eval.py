"""Freeze or execute the full published MP3D validation selection explicitly.

Usage::

    python -m ...mp3d.frozen_eval freeze --lock PATH --note "..." -- [run args]
    python -m ...mp3d.frozen_eval run   --lock PATH -- [run args]

A lock is a commitment to one configuration -- source, method and model
identity, runtime, seed, data hashes, motion tolerances and the exact ordered
episode list. It is **not** proof that validation never influenced development;
``--note`` records that exposure in the lock, and the shared lock schema keeps
``held_out_claim`` false.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import evaluation_configuration
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import freeze_configuration
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import (
    PROTOCOL, PUBLISHED_VAL_EPISODES)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.run import (
    main as run_main, parser as run_parser, prepare)


def published_selection(env, config):
    """The complete published split, in the loader's stable order.

    Raises:
        ValueError: The preparation is a subset, a shard, or a release whose
            episode count is not the published one; none of those can be
            frozen as, or run as, the full validation split.
    """
    if not config.get("full_split"):
        raise ValueError("Freezing requires the complete split: no --scene, --limit or sharding")
    ids = list(env.episode_ids())
    if len(ids) != PUBLISHED_VAL_EPISODES:
        raise ValueError("The published %s split has %d episodes; this release offers %d"
                         % (PROTOCOL.split, PUBLISHED_VAL_EPISODES, len(ids)))
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--note", help="prior development/tuning exposure, required to freeze")
    known, remainder = parser.parse_known_args(argv)
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    args = run_parser().parse_args(remainder)
    env, policy, config, issues = prepare(args)
    try:
        if issues:
            raise ValueError("; ".join(issues))
        ids = published_selection(env, config)
        if known.action == "freeze":
            if not known.note or not known.note.strip():
                parser.error("--note must record prior development/validation exposure")
            prepared = evaluation_configuration(config, ids, PROTOCOL.evaluation(), policy=policy)
            freeze_configuration(prepared, known.lock, development_note=known.note)
            print("Frozen before execution:", known.lock)
            return 0
    finally:
        # prepare() returns env=None when the dataset could not be loaded at
        # all; closing None there would mask the real preflight message.
        if env is not None:
            env.close()
    if args.output is None:
        raise ValueError("Frozen execution requires an explicit --output directory")
    result = run_main(remainder, frozen_lock=known.lock, expected_episode_ids=ids)
    if result == 0:
        (args.output / "evaluation_role.json").write_text(json.dumps({
            "role": "frozen_published_validation",
            "lock": str(known.lock.resolve()),
            "held_out_claim": False}, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
