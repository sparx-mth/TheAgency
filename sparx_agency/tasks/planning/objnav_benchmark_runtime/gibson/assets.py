"""Import one licensed Gibson Habitat scene from a user-provided local ZIP.

No downloads, licence acceptance, navmesh regeneration or mesh conversion are
performed here. Only the selected publisher files are copied, never extractall.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
import threading
import zipfile

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import SCENES


DATA_ACCESS = "https://forms.gle/36TW9uVpjrE1Mkf9A"
MAX_ASSET_BYTES = 2 * 1024 ** 3


def import_scene_zip(archive, scene, destination, cancelled=None, *, allowed_scenes=SCENES):
    """Validate both members, stage both, then install without overwriting files.

    Args:
        archive: The publisher's Habitat-sim ZIP, obtained after licence acceptance.
        scene: A member of the explicitly supplied scene allowlist.
        destination: Directory that will contain <scene>.glb and <scene>.navmesh.
        cancelled: Optional threading.Event; checked between copy chunks.

    Returns:
        Resolved destination directory. Existing files are never overwritten.
    """
    if scene not in allowed_scenes or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", scene):
        raise ValueError("Choose a known Gibson validation scene")
    destination = Path(destination).expanduser().resolve()
    wanted = (scene + ".glb", scene + ".navmesh")
    with zipfile.ZipFile(Path(archive).expanduser()) as bundle:
        selected = {}
        for name in wanted:
            candidates = [member for member in bundle.infolist()
                          if PurePosixPath(member.filename).name == name and not member.is_dir()]
            if len(candidates) != 1:
                raise ValueError("Expected exactly one %s in the ZIP; found %d. "
                                 "Download Gibson for Habitat-sim, not ObjectNav metadata "
                                 "or the raw Gibson OBJ archive." % (name, len(candidates)))
            member = candidates[0]
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                    or stat.S_ISLNK(member.external_attr >> 16)
                    or not 0 < member.file_size <= MAX_ASSET_BYTES):
                raise ValueError("Unsafe, empty or oversized scene member: %s" % name)
            if (destination / name).exists():
                raise FileExistsError("Not overwriting existing asset: %s. Use another directory." % (destination / name))
            selected[name] = member
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".gibson-import-", dir=str(destination)) as staging:
            staging = Path(staging)
            for name, member in selected.items():
                with bundle.open(member) as source, (staging / name).open("wb") as target:
                    while True:
                        if cancelled is not None and cancelled.is_set():
                            raise RuntimeError("Scene import cancelled")
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        target.write(block)
                if (staging / name).stat().st_size != member.file_size:
                    raise ValueError("Incomplete archive member: %s" % name)
            with (staging / wanted[0]).open("rb") as stream:
                header = stream.read(12)
            if len(header) != 12:
                raise ValueError("Scene is not a binary glTF (GLB) file")
            magic, version, size = struct.unpack("<4sII", header)
            if magic != b"glTF" or version not in (1, 2) or size != selected[wanted[0]].file_size:
                raise ValueError("Scene has an invalid GLB header; not installing it")
            installed = []
            try:
                for name in wanted:
                    # Same-filesystem link is atomic and fails if another writer
                    # created the destination after our initial check.
                    os.link(str(staging / name), str(destination / name))
                    installed.append(destination / name)
            except BaseException:
                for path in installed:
                    path.unlink()
                raise
    return destination


def import_scene_dialog(app):
    """UI entrypoint; extraction runs off the Tk thread, completion via its queue."""
    from tkinter import filedialog

    if app.running:
        app.events.put(("message", "Stop the current operation before importing a scene."))
        return
    archive = filedialog.askopenfilename(
        parent=app.window, title="Select licensed Gibson Habitat-sim scene ZIP",
        initialdir=str(Path.home() / "Downloads"), filetypes=[("ZIP archives", "*.zip")])
    if not archive:
        return
    scene = app.variables["scene"].get()
    destination = app.variables["scenes_dir"].get()
    app.cancelled.clear()
    app.running = True
    app.run_button.configure(state="disabled")
    app.events.put(("message", "Importing only %s.glb and %s.navmesh…" % (scene, scene)))

    def worker():
        try:
            path = import_scene_zip(archive, scene, destination, app.cancelled)
            app.events.put(("imported", str(path)))
            app.events.put(("message", "Scene imported. Confirm the simulator-version option, then Run one-scene demo."))
        except Exception as exc:
            app.events.put(("message", "Scene import failed: %s" % exc))
        finally:
            app.events.put(("finished", ""))

    threading.Thread(target=worker, daemon=False).start()

