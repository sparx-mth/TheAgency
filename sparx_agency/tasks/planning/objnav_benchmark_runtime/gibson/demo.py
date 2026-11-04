"""One-button, one-scene Gibson demo UI. No synthetic substitute or full-dataset mode."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import webbrowser

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.assets import DATA_ACCESS, import_scene_dialog
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.demo_services import (
    REPO_ROOT, environment_python, start_detector, start_ollama, stop_process)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import SCENES


SETTINGS_PATH = Path.home() / ".config" / "sparx" / "gibson-demo.json"


def defaults():
    base = Path.home() / "datasets"
    candidates = [base / "objectnav/gibson/v1.1/val",
                  base / "objectnav/gibson/objectnav/gibson/v1.1/val"]
    episodes = next((p for p in candidates if (p / "val_info.pbz2").exists()), candidates[0])
    config = {"scene": SCENES[0], "episodes": 1, "episodes_dir": os.environ.get("GIBSON_EPISODES_DIR", str(episodes)),
              "scenes_dir": os.environ.get("GIBSON_SCENES_DIR", str(base / "gibson/scenes")),
              "habitat_python": os.environ.get("GIBSON_PYTHON", ""),
              "detector_python": os.environ.get("GIBSON_DETECTOR_PYTHON", ""),
              "checkpoint": str(REPO_ROOT / "yolov8x-worldv2.pt"),
              "output_root": str(Path.home() / "objnav_benchmark/gibson/demos"),
              "start_services": True, "allow_version_mismatch": False}
    if SETTINGS_PATH.exists():
        config.update(json.loads(SETTINGS_PATH.read_text()))
    return config


def validate_demo(config):
    """Refuse missing real-scene assets before starting rendering or model inference."""
    scene = config["scene"]
    if scene not in SCENES or not 1 <= int(config["episodes"]) <= 10:
        raise ValueError("Choose one Gibson validation scene and 1–10 demo episodes")
    missing = [Path(config["scenes_dir"]).expanduser() / (scene + suffix)
               for suffix in (".glb", ".navmesh")]
    episodes = Path(config["episodes_dir"]).expanduser()
    missing += [episodes / "val_info.pbz2", episodes / "content" / (scene + "_episodes.json.gz")]
    missing = [str(p) for p in missing if not p.is_file()]
    if missing:
        raise FileNotFoundError("Real Gibson assets are missing:\n" + "\n".join(missing)
                                + "\nThe downloaded *.glb.json.gz file is episode metadata, not a 3-D mesh. "
                                  "Open Dataset licence/download, complete the publisher's form, and obtain "
                                  "Gibson for Habitat-sim (gibson_habitat_trainval.zip). Then use Import scene ZIP "
                                  "or Browse to an already-extracted scene directory.\n" + DATA_ACCESS)
    if not config.get("allow_version_mismatch"):
        raise ValueError("Acknowledge the recorded Habitat 0.2.4 vs reference 0.1.5 difference, "
                         "or run the reference CLI with your 0.1.5 environment")


def evaluation_command(config, output):
    """Always one named scene, an explicit small limit, and recording enabled."""
    command = [environment_python("habitat", config.get("habitat_python", "")), "-u", "-m",
               "sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run",
               "--scene", config["scene"], "--limit", str(int(config["episodes"])),
               "--episodes-dir", config["episodes_dir"], "--scenes-dir", config["scenes_dir"],
               "--detector-url", "http://127.0.0.1:18092", "--record", "--output", str(output)]
    if config.get("allow_version_mismatch"):
        command.append("--allow-sim-version-mismatch")
    return command


class DemoWindow:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.window = tk.Tk()
        self.window.title("Gibson — one-scene algorithm demo")
        self.window.geometry("940x740")
        self.config = defaults()
        self.events = queue.Queue()
        self.cancelled = threading.Event()
        self.process = self.detector_process = self.server = None
        self.report_url = None
        self.running = False
        body = ttk.Frame(self.window, padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Run your object-search algorithm in ONE Gibson scene", font=("Sans", 15, "bold")).pack(anchor="w")
        ttk.Label(body, text="GT RGB-D/pose · predicted objects · LLM room reasoning · RPT* · video + SR/SPL/DTG/SoftSPL").pack(anchor="w", pady=8)
        form = ttk.Frame(body)
        form.pack(fill="x")
        self.variables = {}
        for row, (key, label) in enumerate((
                ("scene", "Gibson scene"), ("episodes", "Episodes (1–10; default 1)"),
                ("scenes_dir", "Scene .glb/.navmesh directory"), ("episodes_dir", "ObjectNav v1.1/val directory"),
                ("habitat_python", "Habitat Python (blank = auto)"), ("detector_python", "Detector Python (blank = auto)"),
                ("checkpoint", "Local YOLO-World checkpoint"), ("output_root", "Results directory"))):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=4)
            variable = tk.StringVar(value=str(self.config[key]))
            self.variables[key] = variable
            if key == "scene":
                widget = ttk.Combobox(form, textvariable=variable, values=SCENES, state="readonly")
            else:
                widget = ttk.Entry(form, textvariable=variable)
            widget.grid(row=row, column=1, sticky="ew", padx=10)
            if key.endswith("_dir") or key == "output_root":
                ttk.Button(form, text="Browse", command=lambda k=key: self.browse(k)).grid(row=row, column=2)
        form.columnconfigure(1, weight=1)
        self.start_services = tk.BooleanVar(value=self.config.get("start_services", True))
        self.allow_version = tk.BooleanVar(value=self.config.get("allow_version_mismatch", False))
        ttk.Checkbutton(body, text="Start/reuse the authorized CPU-only Ollama and dedicated YOLO-World services", variable=self.start_services).pack(anchor="w", pady=5)
        ttk.Checkbutton(body, text="Acknowledge Habitat 0.2.4 port (reference 0.1.5); not an exact SOTA reproduction", variable=self.allow_version).pack(anchor="w")
        controls = ttk.Frame(body)
        controls.pack(fill="x", pady=12)
        self.run_button = ttk.Button(controls, text="▶ Run one-scene demo", command=self.start)
        self.run_button.pack(side="left", padx=4)
        ttk.Button(controls, text="■ Stop", command=self.stop).pack(side="left", padx=4)
        ttk.Button(controls, text="Open results / live view", command=self.open_results).pack(side="left", padx=4)
        ttk.Button(controls, text="Import scene ZIP…", command=lambda: import_scene_dialog(self)).pack(side="right", padx=4)
        ttk.Button(controls, text="Dataset licence/download", command=lambda: webbrowser.open(DATA_ACCESS)).pack(side="right")
        self.status = tk.StringVar(value="Ready to check the selected real scene. No full-dataset run is available here.")
        ttk.Label(body, textvariable=self.status, wraplength=880).pack(fill="x", pady=5)
        self.log = tk.Text(body, height=15, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.after(150, self.poll)
        try:
            validate_demo(self.config)
        except Exception as exc:
            self.events.put(("message", str(exc)))

    def browse(self, key):
        from tkinter import filedialog
        path = filedialog.askdirectory(parent=self.window)
        if path:
            self.variables[key].set(path)

    def start(self):
        if self.running:
            return
        config = {key: variable.get() for key, variable in self.variables.items()}
        config.update(start_services=self.start_services.get(), allow_version_mismatch=self.allow_version.get())
        try:
            validate_demo(config)
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(config, indent=2))
        except Exception as exc:
            self.events.put(("message", str(exc)))
            return
        self.config = config
        self.cancelled.clear()
        self.running = True
        self.run_button.configure(state="disabled")
        self.status.set("Starting the real one-scene run…")
        threading.Thread(target=self.worker, args=(config,), daemon=False).start()

    def worker(self, config):
        try:
            output = Path(config["output_root"]).expanduser() / (config["scene"] + "-" + time.strftime("%Y%m%d-%H%M%S"))
            output.mkdir(parents=True, exist_ok=False)
            if config["start_services"]:
                self.events.put(("message", "Checking CPU-only model services…"))
                start_ollama()
                from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
                self.detector_process = start_detector(environment_python("navdp", config["detector_python"]),
                                                       config["checkpoint"], gibson_label_mapper().vocabulary(), output,
                                                       cancelled=self.cancelled)
            if self.cancelled.is_set():
                return
            if self.server is not None:
                self.server.shutdown()
                self.server.server_close()
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(output)))
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            self.report_url = "http://127.0.0.1:%d/" % self.server.server_port
            self.events.put(("message", "Results: %s" % output))
            environment = dict(os.environ)
            if not environment.get("IMAGEIO_FFMPEG_EXE"):
                try:
                    encoder = Path(environment_python("navdp", config["detector_python"])).parent / "ffmpeg"
                    if encoder.is_file():
                        environment["IMAGEIO_FFMPEG_EXE"] = str(encoder)
                except ValueError:
                    pass  # evaluator preflight checks its own encoder candidates
            with (output / "run.log").open("w") as logfile:
                self.process = subprocess.Popen(evaluation_command(config, output), cwd=str(REPO_ROOT),
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                                bufsize=1, start_new_session=True, env=environment)
                for line in self.process.stdout:
                    logfile.write(line)
                    logfile.flush()
                    self.events.put(("message", line.rstrip()))
                    if (output / "live.html").exists() and not getattr(self, "_opened", False):
                        self._opened = True
                        self.events.put(("open", self.report_url + "live.html"))
                status = self.process.wait()
            if status == 0 and (output / "index.html").exists():
                self.events.put(("open", self.report_url + "index.html"))
                self.events.put(("message", "Completed. Metrics, video, path and reasoning are available in the report."))
            else:
                self.events.put(("message", "Run did not complete (exit %d). See run.log; no success is fabricated." % status))
        except Exception as exc:
            self.events.put(("message", "Demo blocked: %s" % exc))
        finally:
            stop_process(self.detector_process)
            self.detector_process = None
            self.events.put(("finished", ""))

    def poll(self):
        while not self.events.empty():
            kind, value = self.events.get()
            if kind == "open":
                webbrowser.open(value)
            elif kind == "imported":
                self.variables["scenes_dir"].set(value)
                self.config.update({key: variable.get() for key, variable in self.variables.items()})
                self.config.update(start_services=self.start_services.get(), allow_version_mismatch=self.allow_version.get())
                SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
                SETTINGS_PATH.write_text(json.dumps(self.config, indent=2))
            elif kind == "finished":
                self.running = False
                self._opened = False
                self.run_button.configure(state="normal")
            else:
                self.status.set(value[-500:])
                self.log.configure(state="normal")
                self.log.insert("end", value + "\n")
                self.log.see("end")
                self.log.configure(state="disabled")
        self.window.after(150, self.poll)

    def stop(self):
        self.cancelled.set()
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)  # allow video/metrics cleanup
            process = self.process
            threading.Thread(target=self._finish_stop, args=(process,), daemon=False).start()
        self.status.set("Stop requested; completed episode rows remain on disk.")

    @staticmethod
    def _finish_stop(process):
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            stop_process(process)

    def open_results(self):
        if self.report_url:
            webbrowser.open(self.report_url + "live.html")

    def close(self):
        self.stop()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        self.window.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check saved/default one-scene data without opening the UI")
    args = parser.parse_args()
    if args.check:
        try:
            validate_demo(defaults())
        except Exception as exc:
            print(str(exc))
            return 1
        return 0
    DemoWindow().window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




