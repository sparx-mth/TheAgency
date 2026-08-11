"""Command-line entry point for the model registry.

Shell contract: the resolved path goes to stdout, everything else (logs,
errors) goes to stderr -- so callers can do::

    ENGINE="$(python -m sparx_agency.tasks.common.model_registry.cli path \\
        --model da3_metric_large --role depth_only --precision fp16 \\
        --resolution 546x364)" || exit 1
"""
from __future__ import annotations

import argparse
import sys

from sparx_agency.tasks.common.hardware.detect import detect as detect_hardware
from sparx_agency.tasks.common.model_registry import manifest as manifest_mod
from sparx_agency.tasks.common.model_registry.errors import ModelRegistryError
from sparx_agency.tasks.common.model_registry.paths import REPO_ROOT, cache_root, search_path_dirs
from sparx_agency.tasks.common.model_registry.resolver import resolve


def _add_key_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="model_id, e.g. da3_metric_large")
    parser.add_argument("--role", default=None)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--resolution", default=None, help="'HxW', e.g. 546x364")
    parser.add_argument("--target-tag", default=None, help="override the detected device tag")


def _cmd_path(args) -> int:
    try:
        artifact = resolve(
            args.model, role=args.role, precision=args.precision,
            resolution=args.resolution, target_tag=args.target_tag,
            allow_download=args.download, allow_build=args.build)
    except ModelRegistryError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"[model-registry] origin={artifact.origin} target_tag={artifact.target_tag}",
          file=sys.stderr)
    print(str(artifact.path))
    return 0


def _cmd_list(args) -> int:
    profile = detect_hardware()
    print(f"hardware: arch={profile.arch} sm={profile.sm} target_tag={profile.target_tag}",
          file=sys.stderr)
    root = cache_root()
    print(f"cache root: {root}")
    roots = [root] + search_path_dirs() \
        + manifest_mod.search_roots(manifest_mod.load_manifest(), REPO_ROOT)
    for base in roots:
        engines_dir = base / "engines"
        if not engines_dir.is_dir():
            continue
        for engine in sorted(engines_dir.rglob("*.engine")):
            size_mb = engine.stat().st_size / (1 << 20)
            print(f"  {engine}  ({size_mb:.0f} MB)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="model_registry")
    sub = parser.add_subparsers(dest="command", required=True)

    p_path = sub.add_parser("path", help="resolve one artifact, print its path to stdout")
    _add_key_args(p_path)
    # Plain store_true/store_false pair, not argparse.BooleanOptionalAction (3.9+) --
    # this module runs under Python 3.8 inside the FALCON container.
    p_path.add_argument("--download", dest="download", action="store_true", default=None,
                        help="force-allow downloading (default: env-driven)")
    p_path.add_argument("--no-download", dest="download", action="store_false",
                        help="force-disallow downloading")
    p_path.add_argument("--build", action="store_true", help="allow building from ONNX if needed")
    p_path.set_defaults(func=_cmd_path)

    p_list = sub.add_parser("list", help="show the cache root, search roots, and local artifacts")
    p_list.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
