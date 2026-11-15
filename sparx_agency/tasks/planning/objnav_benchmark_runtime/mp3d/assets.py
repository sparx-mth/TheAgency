"""Check an MP3D installation, fetch the public episodes, and say how to get the scenes.

The two halves of the data are asymmetric, and the tooling here respects that:

Episodes
    ``objectnav_mp3d_v1.zip`` from ``dl.fbaipublicfiles.com``: public,
    unauthenticated, 173 MB (the validation split is a few MB of it). This
    module **can download it** (``--fetch-episodes``), because no credential
    is involved; the launcher does so when the split is missing.

Scenes
    ``mp3d_habitat.zip`` (15 GB; 90 buildings, each ``<scene>.glb`` beside
    ``<scene>.navmesh``, ``<scene>.house`` and ``<scene>_semantic.ply``) is
    licensed by Matterport: it is obtained by signing the Matterport3D terms
    of use and asking the dataset's maintainers for the Habitat task download
    script, which is the account holder's to do and nobody else's. This module
    **never** downloads a scene and never handles a credential; it reports
    what is present and what is missing.

Semantic config
    ``mp3d.scene_dataset_config.json`` (809 bytes, public): the publisher's
    scene dataset config, which registers each ``_semantic.ply`` to its
    ``.glb``. Only ``--perception ground-truth`` needs it; this module **can
    download it** (``--fetch-semantic-config``) and the launcher does so for a
    ground-truth run.

Where things go (``--data-root``, default ``~/datasets``)::

    <root>/objectnav/mp3d/v1/<split>/<split>.json.gz      episodes
    <root>/objectnav/mp3d/v1/<split>/content/<scene>.json.gz
    <root>/scene_datasets/mp3d/<scene>/<scene>.glb         scenes
    <root>/scene_datasets/mp3d/<scene>/<scene>.navmesh
    <root>/scene_datasets/mp3d/mp3d.scene_dataset_config.json   ground-truth perception only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import (
    SEMANTIC_SCENE_DATASET_CONFIG, SEMANTIC_SCENE_DATASET_CONFIG_URL, MP3DDataset)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import (
    PROTOCOLS, PUBLISHED_VAL, protocol_for)

#: The public episode archive. No credentials. (The ``m3d`` path component is
#: the publisher's, verbatim; ``.../objectnav/mp3d/...`` is not served.)
EPISODE_ARCHIVES = {
    "v1": "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip",
}
#: The splits the archive holds; ``val`` is the benchmark.
EPISODE_SPLITS = ("train", "val", "val_mini")

#: Where to ask for the scenes. Accept the terms of use, then request the
#: Habitat-ready download from the maintainers' form.
LICENCE_URL = "https://niessner.github.io/Matterport/"
SCENE_ARCHIVE = "mp3d_habitat.zip"
SCENE_DOWNLOAD_GB = 15.0


def episode_dir(data_root, protocol):
    """Where this version's episodes belong under ``data_root``."""
    return (Path(data_root).expanduser() / "objectnav" / "mp3d"
            / protocol.dataset_version / protocol.split)


def scenes_dir(data_root):
    """What habitat-lab calls ``data/scene_datasets``, under ``data_root``."""
    return Path(data_root).expanduser() / "scene_datasets"


def check_installation(episodes_dir, scenes_root, protocol, *, sample=None):
    """Report exactly what is present and what is missing, without downloading.

    Args:
        episodes_dir: The split directory that should hold ``<split>.json.gz``
            and ``content/``.
        scenes_root: The directory the episodes' ``scene_id`` values resolve
            against (the parent of ``mp3d/``), or ``None`` to check the
            episodes only.
        protocol: The :class:`MP3DProtocol` being provisioned for.
        sample: Check only this many scenes, for a quick look.

    Returns:
        A JSON-safe report: whether the episodes and scenes are complete, the
        counts found against the counts published, and the missing files.
    """
    expected = PUBLISHED_VAL[protocol.dataset_version]
    report = {"dataset_version": protocol.dataset_version,
              "split": protocol.split,
              "scene_release": protocol.scene_release,
              "scene_root_name": protocol.scene_root_name,
              "episodes_dir": str(Path(episodes_dir).expanduser()),
              "scenes_root": None if scenes_root is None else str(Path(scenes_root).expanduser()),
              "episodes_ready": False, "scenes_ready": False, "problems": []}
    try:
        # full=False on purpose: the published counts are what this function
        # exists to REPORT, so the loader must not raise on them first.
        dataset = MP3DDataset(episodes_dir, None, protocol, full=False,
                              require_scenes=False)
    except (OSError, ValueError) as exc:
        report["problems"].append("Episodes: %s" % exc)
        return report
    report["episodes_found"] = len(dataset.episodes)
    report["episodes_published"] = expected["episodes"]
    report["scenes_found"] = len(dataset.scene_counts)
    report["scenes_published"] = expected["scenes"]
    report["categories_found"] = len(dataset.category_to_task_category_id)
    report["episodes_ready"] = (len(dataset.episodes) == expected["episodes"]
                                and len(dataset.scene_counts) == expected["scenes"])
    if not report["episodes_ready"]:
        report["problems"].append(
            "Episodes: found %d episodes in %d scenes, published split is %d in %d"
            % (len(dataset.episodes), len(dataset.scene_counts),
               expected["episodes"], expected["scenes"]))
    if scenes_root is None:
        return report

    root = Path(scenes_root).expanduser()
    release_dir = root / protocol.scene_root_name
    if not release_dir.exists():
        report["problems"].append(
            "Scenes: %s does not exist. Unpack %s so that it holds mp3d/<scene>/<scene>.glb "
            "-- see --instructions." % (release_dir, SCENE_ARCHIVE))
        return report
    report["release_dir_resolves_to"] = str(release_dir.resolve())

    wanted, seen = [], set()
    for episode in dataset.episodes.values():
        if episode.scene_key not in seen:
            seen.add(episode.scene_key)
            wanted.append(episode)
    if sample is not None:
        wanted = wanted[:sample]
    missing, total_bytes, present = [], 0, 0
    for episode in wanted:
        glb = root / episode.scene_relative_path
        navmesh = glb.with_name(glb.name[:-len(".glb")] + ".navmesh")
        ok = True
        for path in (glb, navmesh):
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
                ok = False
            else:
                total_bytes += path.stat().st_size
        present += int(ok)
    report["scenes_checked"] = len(wanted)
    report["scenes_complete"] = present
    report["scene_bytes"] = total_bytes
    report["missing_count"] = len(missing)
    report["missing"] = sorted(missing)[:20]
    report["scenes_ready"] = not missing and len(wanted) == expected["scenes"]
    if missing:
        report["problems"].append(
            "Scenes: %d file(s) missing under %s" % (len(missing), release_dir))
    return report


def _archive_layout(names):
    """``(prefix, splits)``: the wrapper directory (``""`` for none) and the split directories the archive holds.

    A split counts only when its index file ``<split>/<split>.json.gz`` is
    there, so a stray directory of the right name cannot pass as one.
    """
    for prefix in ("",) + tuple(sorted({Path(n).parts[0] for n in names if len(Path(n).parts) > 1})):
        found = set()
        for split in EPISODE_SPLITS:
            index = "%s/%s.json.gz" % (split, split)
            if (prefix + "/" + index if prefix else index) in names:
                found.add(split)
        if found:
            return prefix, found
    return "", set()


def fetch_episodes(data_root, protocol, *, url=None, keep_archive=False, log=print):
    """Download the public episode archive and unpack it under ``data_root``.

    Idempotent: a split that is already in place is left alone. The archive
    holds ``{train,val,val_mini}/`` at its top level (verified 2026-11-04:
    ``val/val.json.gz`` and ``val/content/<scene>.json.gz``); a re-packaged
    one with a single wrapper directory is accepted too. Every split is
    unpacked, so a development run finds ``train`` beside ``val``.

    Returns:
        The split directory of ``protocol`` (``.../objectnav/mp3d/v1/<split>``).

    Raises:
        RuntimeError: The archive was not what the publisher serves (no split
            directories inside), or the download was cut short.
    """
    target_root = Path(data_root).expanduser() / "objectnav" / "mp3d" / protocol.dataset_version
    split_dir = target_root / protocol.split
    if (split_dir / ("%s.json.gz" % protocol.split)).is_file():
        log("episodes already present: %s" % split_dir)
        return split_dir
    url = url or EPISODE_ARCHIVES[protocol.dataset_version]
    target_root.mkdir(parents=True, exist_ok=True)
    archive = target_root / Path(url).name
    if not archive.is_file():
        log("downloading %s -> %s" % (url, archive))
        partial = archive.with_suffix(archive.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            expected = response.headers.get("Content-Length")
            shutil.copyfileobj(response, out, length=4 * 1024 * 1024)
        if expected is not None and partial.stat().st_size != int(expected):
            partial.unlink()
            raise RuntimeError("Download of %s was cut short (%s of %s bytes)"
                               % (url, partial.stat().st_size if partial.exists() else 0, expected))
        partial.rename(archive)
    log("unpacking %s" % archive)
    with zipfile.ZipFile(archive) as zipped:
        prefix, splits = _archive_layout(zipped.namelist())
        if not splits:
            raise RuntimeError("%s does not hold <split>/<split>.json.gz for any of %s; it is not the "
                               "published archive" % (archive, ", ".join(EPISODE_SPLITS)))
        with tempfile.TemporaryDirectory(dir=str(target_root)) as scratch:
            zipped.extractall(scratch)
            unpacked = Path(scratch) / prefix if prefix else Path(scratch)
            for split in sorted(splits):
                source, destination = unpacked / split, target_root / split
                if destination.exists():
                    log("keeping existing %s" % destination)
                    continue
                shutil.move(str(source), str(destination))
                log("installed %s" % destination)
    if not keep_archive:
        archive.unlink()
    if not (split_dir / ("%s.json.gz" % protocol.split)).is_file():
        raise RuntimeError("The archive held no %s split" % protocol.split)
    return split_dir


def fetch_semantic_config(scenes_root, protocol, *, url=SEMANTIC_SCENE_DATASET_CONFIG_URL, log=print):
    """Install the publisher's ``mp3d.scene_dataset_config.json`` beside the scenes; idempotent.

    Returns:
        The installed path, ``<scenes_root>/mp3d/mp3d.scene_dataset_config.json``.

    Raises:
        FileNotFoundError: The scenes are not installed under ``scenes_root``.
        RuntimeError: The download is not a scene dataset config with semantic frames.
    """
    release = Path(scenes_root).expanduser() / protocol.scene_root_name
    target = release / SEMANTIC_SCENE_DATASET_CONFIG
    if target.is_file() and target.stat().st_size > 0:
        log("semantic config already present: %s" % target)
        return target
    if not release.is_dir():
        raise FileNotFoundError("No MP3D scenes at %s; install %s first" % (release, SCENE_ARCHIVE))
    log("downloading %s -> %s" % (url, target))
    with urllib.request.urlopen(url, timeout=60) as response:
        body = response.read()
    attributes = json.loads(body.decode("utf-8")).get("stages", {}).get("default_attributes", {})
    if "semantic_up" not in attributes or "semantic_asset" not in attributes:
        raise RuntimeError("%s is not the MP3D scene dataset config (no semantic frame in it)" % url)
    partial = target.with_suffix(".part")
    partial.write_bytes(body)
    partial.rename(target)
    return target


def download_instructions(protocol, data_root):
    """What to run (episodes) and what to request (scenes), ready to paste."""
    root = Path(data_root).expanduser()
    version = protocol.dataset_version
    return "\n".join([
        "# ---- 1. Episodes (public, no credentials, 173 MB; the launcher does this itself) ----",
        "python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.assets \\",
        "  --version %s --data-root %s --fetch-episodes" % (version, root),
        "#   -> %s/objectnav/mp3d/%s/{train,val,val_mini}/" % (root, version),
        "",
        "# ---- 2. Scenes (~%.0f GB, licensed) ----" % SCENE_DOWNLOAD_GB,
        "# Sign the Matterport3D terms of use (%s) and request the Habitat task" % LICENCE_URL,
        "# download from the maintainers; it provides %s (90 scenes: <scene>.glb, .navmesh, .house," % SCENE_ARCHIVE,
        "# _semantic.ply). Nothing in this repo downloads it or sees the credentials.",
        "mkdir -p %s/scene_datasets" % root,
        "unzip -q %s -d %s/scene_datasets      # -> %s/scene_datasets/mp3d/<scene>/" % (SCENE_ARCHIVE, root, root),
        "",
        "# ---- 3. Ground-truth perception only: the publisher's semantic config (public; the launcher does this) ----",
        "python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.assets \\",
        "  --version %s --data-root %s --fetch-semantic-config" % (version, root),
        "#   -> %s/scene_datasets/mp3d/%s" % (root, SEMANTIC_SCENE_DATASET_CONFIG),
        "",
        "# ---- 4. Verify before running anything ----",
        "python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.assets \\",
        "  --version %s --data-root %s --check" % (version, root),
        "",
    ])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", choices=sorted(PROTOCOLS), default="v1")
    p.add_argument("--split", default="val")
    p.add_argument("--data-root", type=Path, default=Path("~/datasets").expanduser(),
                   help="The directory holding objectnav/ and scene_datasets/")
    p.add_argument("--episodes-dir", type=Path,
                   help="Override the episode split directory")
    p.add_argument("--scenes-dir", type=Path,
                   help="Override what data/scene_datasets points at")
    p.add_argument("--sample", type=int, help="Check only this many scenes")
    p.add_argument("--check", action="store_true")
    p.add_argument("--fetch-episodes", action="store_true",
                   help="Download and unpack the public episode archive under --data-root (no credentials)")
    p.add_argument("--fetch-semantic-config", action="store_true",
                   help="Install the publisher's mp3d.scene_dataset_config.json beside the scenes (public; "
                        "--perception ground-truth needs it)")
    p.add_argument("--instructions", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    protocol = protocol_for(args.version).with_split(args.split)
    if args.fetch_episodes:
        fetch_episodes(args.data_root, protocol, log=lambda m: print(m, file=sys.stderr))
    if args.fetch_semantic_config:
        fetch_semantic_config(args.scenes_dir or scenes_dir(args.data_root), protocol,
                              log=lambda m: print(m, file=sys.stderr))
    fetched = args.fetch_episodes or args.fetch_semantic_config
    if args.instructions or not (args.check or fetched):
        print(download_instructions(protocol, args.data_root))
        if not args.check:
            return 0
        print()
    if not args.check:
        return 0
    episodes = args.episodes_dir or episode_dir(args.data_root, protocol)
    scenes = args.scenes_dir or scenes_dir(args.data_root)
    report = check_installation(episodes, scenes, protocol, sample=args.sample)
    print(json.dumps(report, indent=2))
    return 0 if report["episodes_ready"] and report["scenes_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
