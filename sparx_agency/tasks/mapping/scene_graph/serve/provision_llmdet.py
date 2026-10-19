"""Explicit, pinned public checkpoint provisioning; never called by the server."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from sparx_agency.tasks.common.model_registry.download.verify import sha256_of

MODEL_ID = "iSEE-Laboratory/llmdet_large"
MODEL_REVISION = "bec37f296f05b22f6c6b39bc05a6c611239f4e31"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character commit, not main")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; existing snapshots are never overwritten")
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(args.repo_id, revision=args.revision, files_metadata=True, token=False)
    if info.sha != args.revision:
        raise RuntimeError("Resolved checkpoint revision differs from requested revision")
    snapshot_download(args.repo_id, revision=args.revision, local_dir=args.output,
                      token=False, allow_patterns=["*.json", "*.txt", "model.safetensors", "README.md", "LICENSE"])
    files = {p.name: sha256_of(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    weights = next(item for item in info.siblings if item.rfilename == "model.safetensors")
    if weights.lfs is None or weights.lfs.sha256 != files["model.safetensors"]:
        raise RuntimeError("Downloaded weights do not match the publisher's LFS SHA-256")
    source = {"repo_id": args.repo_id, "revision": args.revision,
              "url": "https://huggingface.co/%s/tree/%s" % (args.repo_id, args.revision),
              "license": "apache-2.0", "files": files}
    (args.output / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    print("Verified local checkpoint:", args.output)


if __name__ == "__main__":
    main()
