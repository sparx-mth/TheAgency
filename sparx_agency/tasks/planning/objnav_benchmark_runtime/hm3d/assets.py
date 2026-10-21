"""Check an HM3D installation and print the exact commands to complete it.

This module **never downloads anything and never handles a credential.** The
episode datasets are public; the scene meshes are not, and obtaining them means
accepting Matterport's academic licence and generating an API token, which is
the account holder's to do and nobody else's. So the tooling here answers two
questions and stops: *what is missing*, and *what command would the owner of
that licence run to get it*.

What the two halves are, and why they are asymmetric:

Episodes
    ``objectnav_hm3d_v1.zip`` / ``objectnav_hm3d_v2.zip`` from
    ``dl.fbaipublicfiles.com``. Public, unauthenticated, ~139 MB and ~260 MB,
    of which the validation splits are 1.8 MB and 4.6 MB.

Scenes
    ``hm3d_val_v0.1`` / ``hm3d_val_v0.2`` via habitat-sim's downloader, which
    takes the Matterport API **token id as --username and the token secret as
    --password**. Roughly 3.8 GB and 5.3 GB per version for the val split.

The one trap that silently ruins a run: v1 episodes spell their scenes
``hm3d/val/...`` and v2 episodes spell them ``hm3d_v0.2/val/...``, but the
downloader creates only the name ``hm3d`` and repoints it at whichever release
was fetched last. Symlinking ``hm3d_v0.2`` at a single v0.1 download (or the
reverse) makes one of the two versions run against the wrong geometry, and
nothing fails -- the scenes simply are not the ones the episodes were generated
on. :func:`check_installation` looks for exactly that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import (
    HM3DDataset, PUBLISHED_VAL)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import (
    PROTOCOLS, protocol_for)

#: Where the public episode archives live. No credentials.
EPISODE_ARCHIVES = {
    "v1": "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip",
    "v2": "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip",
}

#: Accept the academic licence here, then mint a token under Account -> devtools.
LICENCE_URL = "https://matterport.com/habitat-matterport-3d-research-dataset"
TOKEN_URL = "https://my.matterport.com/settings/account/devtools"

#: Approximate val-split download sizes, from the publisher's own tables.
SCENE_DOWNLOAD_GB = {"v1": 3.8, "v2": 5.3}


def episode_dir(data_root, protocol):
    """Where this version's episodes belong under ``data_root``."""
    return (Path(data_root).expanduser() / "objectnav" / "hm3d"
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
            against, or ``None`` to check the episodes only.
        protocol: The :class:`HM3DProtocol` being provisioned for.
        sample: Check only this many scenes, for a quick look. ``None`` checks
            every scene the split references.

    Returns:
        A JSON-safe report: whether the episodes and scenes are complete, the
        counts found against the counts published, and the missing files
        (capped at twenty, with the full count beside them).
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
        # exists to REPORT. Letting the loader enforce them here would raise
        # before the four count keys are filled in, so a script reading the
        # JSON could not tell how short the installation is.
        dataset = HM3DDataset(episodes_dir, None, protocol, full=False,
                              require_scenes=False)
    except (OSError, ValueError) as exc:
        report["problems"].append("Episodes: %s" % exc)
        return report
    report["episodes_found"] = len(dataset.episodes)
    report["episodes_published"] = expected["episodes"]
    report["scenes_found"] = len(dataset.scene_counts)
    report["scenes_published"] = expected["scenes"]
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
            "Scenes: %s does not exist. HM3D-%s episodes name their scenes %r, "
            "and habitat-sim's downloader only ever creates 'hm3d' -- see the "
            "symlink step in --instructions."
            % (release_dir, protocol.dataset_version, protocol.scene_root_name))
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


def download_instructions(protocol, data_root):
    """The exact commands whoever holds the HM3D licence would run.

    Returns:
        A single block of shell, ready to paste. It is printed, never executed:
        the scene half needs a Matterport API token, which this process must
        not see, hold or write anywhere.
    """
    root = Path(data_root).expanduser()
    version = protocol.dataset_version
    release = "0.1" if version == "v1" else "0.2"
    return "\n".join([
        "# ---- 1. Episodes (public, no credentials, %s) ----" % (
            "139 MB" if version == "v1" else "260 MB"),
        "mkdir -p %s/objectnav/hm3d" % root,
        "wget -O /tmp/objectnav_hm3d_%s.zip \\" % version,
        "  %s" % EPISODE_ARCHIVES[version],
        "unzip -q /tmp/objectnav_hm3d_%s.zip -d %s/objectnav/hm3d" % (version, root),
        "# each archive holds a single objectnav_hm3d_<version>/ directory:",
        "mv %s/objectnav/hm3d/objectnav_hm3d_%s %s/objectnav/hm3d/%s"
        % (root, version, root, version),
        "",
        "# ---- 2. Scenes (~%.1f GB, licensed) ----" % SCENE_DOWNLOAD_GB[version],
        "# Accept the academic licence: %s" % LICENCE_URL,
        "# Mint an API token:          %s" % TOKEN_URL,
        "# The token ID is --username and the token SECRET is --password.",
        "# Run this yourself; nothing in this repo should ever see the secret.",
        "python -m habitat_sim.utils.datasets_download \\",
        "  --username <token-id> --password <token-secret> \\",
        "  --uids %s --data-path %s/" % (protocol.scene_download_uid, root),
        "",
        "# ---- 3. Name the release the way HM3D-%s episodes spell it ----" % version,
        "# The downloader writes versioned_data/hm3d-%s/hm3d and links it as" % release,
        "# scene_datasets/hm3d, whichever version was fetched last. HM3D-%s" % version,
        "# episodes say %r, so make that name point at this release:"
        % protocol.scene_root_name,
        'ln -sfn "%s/versioned_data/hm3d-%s/hm3d" %s/scene_datasets/%s'
        % (root, release, root, protocol.scene_root_name),
        "",
        "# ---- 4. Verify before running anything ----",
        "python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.assets \\",
        "  --version %s --data-root %s --check" % (version, root),
    ])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", choices=sorted(PROTOCOLS), default="v2")
    p.add_argument("--data-root", type=Path, default=Path("~/datasets").expanduser(),
                   help="The directory holding objectnav/ and scene_datasets/")
    p.add_argument("--episodes-dir", type=Path,
                   help="Override the episode split directory")
    p.add_argument("--scenes-dir", type=Path,
                   help="Override what data/scene_datasets points at")
    p.add_argument("--sample", type=int, help="Check only this many scenes")
    p.add_argument("--check", action="store_true")
    p.add_argument("--instructions", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    protocol = protocol_for(args.version)
    if args.instructions or not args.check:
        print(download_instructions(protocol, args.data_root))
        if not args.check:
            return 0
        print()
    episodes = args.episodes_dir or episode_dir(args.data_root, protocol)
    scenes = args.scenes_dir or scenes_dir(args.data_root)
    report = check_installation(episodes, scenes, protocol, sample=args.sample)
    print(json.dumps(report, indent=2))
    return 0 if report["episodes_ready"] and report["scenes_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
