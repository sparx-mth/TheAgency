"""Profile a fresh detector through the REAL HTTP client/server, then release it.

One process/model/device at a time. Use only when no simulator owns the GPU.
This is also the matched-input evaluation entrypoint for an isolated CUDA run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

from sparx_agency.tasks.common.model_registry.download.verify import sha256_of
from sparx_agency.tasks.mapping.scene_graph.serve.backends import build_detector
from sparx_agency.tasks.mapping.scene_graph.serve.compare_detectors import collect, summarize
from sparx_agency.tasks.mapping.scene_graph.serve.detection_server import (
    _make_server, _ServerContext, parse_args)


def profile(args, server_args):
    if args.output.exists():
        raise FileExistsError("Use a fresh result path")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("split") not in ("train", "development", "synthetic"):
        raise ValueError("Only development/training/synthetic captures are allowed")
    if args.limit:
        manifest = dict(manifest, frames=manifest["frames"][:args.limit])
    config = parse_args(server_args + ["--classes", ",".join(manifest["vocabulary"])])
    if config.selftest:
        raise ValueError("A profile must load a real model, not --selftest")
    server = thread = None
    start = time.perf_counter()
    result = {"manifest_sha256": sha256_of(args.manifest), "requested": vars(config)}
    try:
        detector, metadata = build_detector(config, manifest["vocabulary"])
        result["cold_start_s"] = time.perf_counter() - start
        server = _make_server(_ServerContext(detector, Path(config.model).name,
                                            config.device, manifest["vocabulary"], metadata),
                              "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        result.update(collect(manifest, args.manifest.parent,
                              "http://127.0.0.1:%d" % server.server_port))
        if not args.limit:
            result["summary"] = summarize(manifest, result)
        if config.device.startswith("cuda"):
            import torch
            result["cuda"] = {"device": torch.cuda.get_device_name(config.device),
                              "peak_allocated_mib": torch.cuda.max_memory_allocated(config.device) / 2**20,
                              "peak_reserved_mib": torch.cuda.max_memory_reserved(config.device) / 2**20}
        result["ok"] = True
    except Exception as exc:
        result.update(ok=False, error=repr(exc))
        raise
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Optional timing-only subset; no calibration")
    args, server_args = parser.parse_known_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    profile(args, server_args)


if __name__ == "__main__":
    main()
