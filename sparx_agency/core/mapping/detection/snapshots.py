"""Local inference-bundle validation; no hub client, torch or implicit downloads."""
from __future__ import annotations

import json
from pathlib import Path


def snapshot_files(directory):
    """Return every local inference asset, rejecting incomplete/unsafe shards."""
    path = Path(directory).expanduser()
    if not path.is_dir() or (path / "download-pending.json").exists():
        raise FileNotFoundError("Missing/incomplete local model snapshot: %s" % path)
    weights = path / "model.safetensors"
    index = path / "model.safetensors.index.json"
    if not weights.is_file():
        if not index.is_file():
            raise FileNotFoundError("Local safetensors checkpoint required: %s" % path)
        mapping = json.loads(index.read_text()).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Invalid safetensors shard index: %s" % index)
        for name in set(mapping.values()):
            if (not isinstance(name, str) or Path(name).name != name
                    or not name.endswith(".safetensors")):
                raise ValueError("Unsafe safetensors shard path: %r" % name)
            if not (path / name).is_file():
                raise FileNotFoundError("Missing safetensors shard: %s" % (path / name))
    return sorted(p for p in path.iterdir() if p.is_file()
                  and p.suffix in (".json", ".txt", ".model", ".safetensors")
                  and not p.name.startswith("pytorch_model"))
