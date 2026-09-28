"""Download the explicitly requested public models; inference never calls this."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparx_agency.tasks.common.model_registry.download.verify import locked, sha256_of
from sparx_agency.tasks.common.model_registry.download.http import HttpArtifactStore


MODELS = {
    "grounding-dino-base": {
        "repo_id": "IDEA-Research/grounding-dino-base",
        "revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        "license": "apache-2.0",
    },
    "blip2-flan-t5-xl": {
        "repo_id": "Salesforce/blip2-flan-t5-xl",
        "revision": "0eb0d3b46c14c1f8c7680bca2693baafdb90bb28",
        "license": "mit",
    },
}


def provision_yolo(output):
    """Keep legacy X-v2 and its text encoder explicit, local and checksum-pinned."""
    artifacts = (
        ("yolov8x-worldv2.pt", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-worldv2.pt",
         "41e771bfbbb8894dd857f3fef7cac3b3578dffd49fd3547101efa6a606a02a0e"),
        ("clip/ViT-B-32.pt", "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
         "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"),
    )
    for name, url, expected in artifacts:
        path = Path(output).expanduser() / name
        with locked(path):
            if not path.exists():
                HttpArtifactStore().download(url, path)
            if not path.is_file() or sha256_of(path) != expected:
                raise RuntimeError("Checkpoint checksum mismatch; file retained for inspection: %s" % path)
            print("Verified local checkpoint:", path, flush=True)


def provision_model(destination, spec):
    """Resume only this pinned download; never overwrite an unrelated snapshot.

    A pending manifest allows interrupted large downloads to resume. The final
    source record is published only after every LFS file passes SHA-256 and
    every small file passes the publisher's Git blob hash.
    """
    from huggingface_hub import HfApi, snapshot_download

    destination = Path(destination).expanduser()
    with locked(destination):
        destination.mkdir(parents=True, exist_ok=True)
        source, pending = destination / "source.json", destination / "download-pending.json"
        record = source if source.exists() else pending
        if record.exists():
            previous = json.loads(record.read_text())
            if any(previous.get(key) != spec[key] for key in ("repo_id", "revision")):
                raise ValueError("Refusing to replace a different model: %s" % destination)
        elif any(destination.iterdir()):
            raise ValueError("Non-empty model directory without provisioning identity: %s" % destination)
        api = HfApi()
        info = api.model_info(spec["repo_id"], revision=spec["revision"],
                              files_metadata=True, token=False)
        if info.sha != spec["revision"]:
            raise RuntimeError("Publisher revision changed")
        entries = [item for item in (info.siblings or ()) if _inference_file(item.rfilename)]
        if not any(item.rfilename.endswith(".safetensors") for item in entries):
            raise RuntimeError("No safetensors checkpoint in selected revision")
        if not source.exists():
            pending.write_text(json.dumps(spec, indent=2) + "\n")
            snapshot_download(spec["repo_id"], revision=spec["revision"],
                              local_dir=str(destination), token=False, max_workers=2,
                              allow_patterns=[item.rfilename for item in entries])
        files = _verify_download(destination, entries)
        result = dict(spec, files=files,
                      url="https://huggingface.co/%s/tree/%s" % (spec["repo_id"], spec["revision"]))
        temporary = destination / "source.json.part"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(source)
        if pending.exists():
            pending.unlink()
        return destination


def _inference_file(name):
    """Only root inference assets; exclude pickle indexes and alternate formats."""
    path = Path(name)
    return (path.name == name and not name.startswith("pytorch_model")
            and (path.suffix in (".json", ".txt", ".model", ".safetensors")
                 or name in ("README.md", "LICENSE", "LICENSE.md")))


def _verify_download(destination, entries):
    import hashlib

    files = {}
    for item in entries:
        path = destination / item.rfilename
        if not path.is_file() or path.stat().st_size != item.size:
            raise RuntimeError("Missing/truncated download: %s" % path)
        digest = sha256_of(path)
        if item.lfs is not None:
            if digest != item.lfs.sha256:
                raise RuntimeError("Publisher SHA-256 mismatch: %s" % path)
        else:
            blob = b"blob " + str(item.size).encode() + b"\0" + path.read_bytes()
            if hashlib.sha1(blob).hexdigest() != item.blob_id:
                raise RuntimeError("Publisher Git blob mismatch: %s" % path)
        files[item.rfilename] = digest
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.home() / "models" / "objnav")
    parser.add_argument("--model", choices=tuple(MODELS) + ("all",), default="all")
    parser.add_argument("--include-yolo", action="store_true", help="Also provision X-v2 and local CLIP")
    args = parser.parse_args(argv)
    for name, spec in MODELS.items():
        if args.model in (name, "all"):
            print("Provisioning %s ..." % spec["repo_id"], flush=True)
            print("Verified local checkpoint:", provision_model(args.output / name, spec), flush=True)
    if args.include_yolo:
        provision_yolo(args.output)


if __name__ == "__main__":
    main()

