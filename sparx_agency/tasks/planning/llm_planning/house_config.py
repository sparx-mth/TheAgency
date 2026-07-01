#!/usr/bin/env python3
"""
house_config.py - Centralized configuration loader.

Loads defaults from config.json (next to this file), then applies any
CLI --overrides.  Every script imports this module and calls get_config()
once at startup.

Usage in any script:
    from house_config import get_config
    cfg = get_config()          # parses CLI args automatically
    print(cfg.web_port)         # flat attribute access
    print(cfg.unified_rooms_json)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _CONFIG_DIR / "config.json"


def _flatten(d, prefix=""):
    """Flatten nested dict: {"paths": {"data_dir": "data"}} -> {"data_dir": "data"}"""
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix))  # no prefix nesting — keys are unique
        else:
            flat[k] = v
    return flat


def _load_json_config(path=None):
    """Load config.json and return flattened dict."""
    p = Path(path) if path else _DEFAULT_CONFIG
    if not p.exists():
        print(f"WARNING: config file {p} not found, using built-in defaults")
        return {}
    with open(p) as f:
        return _flatten(json.load(f))


def _build_parser(defaults):
    """Build argparse parser from flattened defaults dict."""
    parser = argparse.ArgumentParser(
        description="House Mapping System — override any config.json value via CLI",
        add_help=True,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to alternate config.json file")

    for key, val in sorted(defaults.items()):
        flag = f"--{key}"
        if val is None and key in ("room_bbox",):
            parser.add_argument(flag, nargs="+", type=int, default=None,
                                help=f"(default: None)")
        elif val is None:
            parser.add_argument(flag, default=None, help=f"(default: None)")
        elif isinstance(val, bool):
            parser.add_argument(flag, type=_str_to_bool, default=val,
                                help=f"(default: {val})")
        elif isinstance(val, list):
            elem_type = type(val[0]) if val else int
            parser.add_argument(flag, nargs="+", type=elem_type, default=val,
                                help=f"(default: {val})")
        elif isinstance(val, int):
            parser.add_argument(flag, type=int, default=val,
                                help=f"(default: {val})")
        elif isinstance(val, float):
            parser.add_argument(flag, type=float, default=val,
                                help=f"(default: {val})")
        else:
            parser.add_argument(flag, type=str, default=val,
                                help=f"(default: {val})")
    return parser


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


_cached_config = None


def get_config(argv=None):
    """
    Load config.json, merge CLI overrides, return SimpleNamespace.

    Parameters
    ----------
    argv : list or None
        Pass explicit args (e.g. [] to skip CLI parsing when imported as lib).
        None = use sys.argv.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    # 1. Load defaults from config.json
    defaults = _load_json_config()

    # 2. Build parser & parse CLI
    parser = _build_parser(defaults)
    # Use parse_known_args to be tolerant of extra flags from subprocesses
    args, _unknown = parser.parse_known_args(argv)

    # 3. If --config was passed, reload from that file
    if args.config:
        defaults = _load_json_config(args.config)
        parser = _build_parser(defaults)
        args, _unknown = parser.parse_known_args(argv)

    # 4. Merge: start from defaults, override with non-None CLI values
    merged = dict(defaults)
    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is not None:
            merged[key] = val

    # Convert list args back to tuples where expected (grid_cells, room_bbox)
    for tup_key in ("grid_cells", "room_bbox"):
        v = merged.get(tup_key)
        if isinstance(v, list) and v:
            merged[tup_key] = tuple(v)

    _cached_config = SimpleNamespace(**merged)
    return _cached_config


def reset_config():
    """Reset cached config (useful for tests)."""
    global _cached_config
    _cached_config = None