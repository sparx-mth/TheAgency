"""Single-card launch profiles for the published ObjectNav splits. No simulator or model imports."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import socket

from sparx_agency.tasks.planning.objnav_benchmark_runtime.resource_monitor import gpu_sample, ram_sample
from sparx_agency.tasks.planning.objnav_benchmark_runtime.scene_pool import max_workers


@dataclass(frozen=True)
class SplitProfile:
    episodes: int
    workers: int
    renderer_mib: int
    worker_ram_gib: int
    detector_port: int


PROFILES = {"hm3d": SplitProfile(2000, 7, 768, 4, 18092),
            "mp3d": SplitProfile(2195, 4, 1400, 6, 18093)}
RUNTIME = Path(__file__).resolve().parent
HOST_HEADROOM_MIB = 16 * 1024  # Model staging, detector and OS in addition to worker RSS.


def profile_environment(profile, slots, batch, port):
    """Settings consumed by benchmark_services.sh, gpu_plan.py and the scene children."""
    env = os.environ
    if (slots > 2 or batch > 256) and not env.get("LLM_VRAM_MIB"):
        raise ValueError("set a measured LLM_VRAM_MIB before increasing past 2 slots or a 256-token batch")
    settings = dict(
        SPLIT="val", SEED=env.get("SEED", "17"), VALIDATE_STARTS="1", EXCLUDE_UNREACHABLE="0", SEGMENTATION="0",
        LLM_BACKEND="ollama", OLLAMA_MODE="native", LLM_BASE_URL=f"http://127.0.0.1:{port}",
        LLM_MODEL="qwen2.5vl:7b", OLLAMA_NUM_PARALLEL=str(slots), OLLAMA_MAX_LOADED_MODELS="1",
        OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0", OLLAMA_CONTEXT_LENGTH="4096",
        LLM_NUM_CTX="4096", LLM_NUM_BATCH=str(batch), LLM_MAX_TOKENS="32", LLM_SERVERS="1",
        LLM_KEEP_ALIVE="30m", LLM_EXCLUSIVE="0", LLM_ANSWERS_REPLAY="",
        DETECTOR_HOST="127.0.0.1", DETECTOR_PORT=env.get("DETECTOR_PORT", str(profile.detector_port)),
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", DETECTOR_THREADS="2",
        GPU_DEVICE=env.get("GPU_DEVICE", "0"), GPU_PLAN="auto", HABITAT_RENDER="gpu",
        GPU_PLAN_LLM="auto", GPU_PLAN_YOLO="auto", GPU_PLAN_HABITAT="auto", HABITAT_CPU_RENDER="0",
        GPU_RESERVE_MIB=env.get("GPU_RESERVE_MIB", "2048"),
        LLM_VRAM_MIB=env.get("LLM_VRAM_MIB", "14336"), YOLO_VRAM_MIB=env.get("YOLO_VRAM_MIB", "2048"),
        HABITAT_VRAM_MIB=env.get("HABITAT_VRAM_MIB", str(profile.renderer_mib)),
        HAB_PY=env.get("HAB_PY", str(Path.home() / "miniconda3/envs/objnav-habitat/bin/python")),
    )
    if any(int(settings[key]) < 1 for key in ("LLM_VRAM_MIB", "YOLO_VRAM_MIB", "HABITAT_VRAM_MIB", "GPU_RESERVE_MIB")):
        raise ValueError("VRAM budgets and reserve must be positive MiB")
    if not 1 <= int(settings["DETECTOR_PORT"]) <= 65535 or int(settings["DETECTOR_PORT"]) == port:
        raise ValueError("DETECTOR_PORT must be a valid port distinct from the Ollama port")
    return settings


def workstation_jobs(profile, ceiling, settings):
    """Budget the cold shared services before sizing the worker pool, not after loading the first scene."""
    device = int(settings["GPU_DEVICE"])
    gpu = gpu_sample(device)
    if gpu is None or gpu[2] < 23 * 1024:
        raise ValueError(f"GPU {device}: a 24 GiB-class card visible to nvidia-smi is required; no services started")
    _, used, total = gpu
    free = int(total - used) - int(settings["LLM_VRAM_MIB"]) - int(settings["YOLO_VRAM_MIB"])
    reserve, renderer = int(settings["GPU_RESERVE_MIB"]), int(settings["HABITAT_VRAM_MIB"])
    used_ram, total_ram = ram_sample()
    by_ram = int((total_ram - used_ram - HOST_HEADROOM_MIB) // (1024 * profile.worker_ram_gib))
    if free < reserve + renderer or by_ram < 1:
        raise ValueError("not enough available VRAM/RAM for the services, one renderer and headroom; "
                         "free the card and host memory before starting")
    return max_workers(min(ceiling, by_ram), cpu_count=len(os.sched_getaffinity(0)),
                       free_vram_mib=free, renderer_mib=renderer, reserve_mib=reserve)


def campaign_jobs(output, dataset, jobs, settings):
    """Persist the admitted concurrency. Don't mix runs with different server settings or replay an unsafe plan."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "workstation_4090.json"
    if path.is_file():
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or type(record.get("jobs")) is not int:
            raise ValueError(f"{path} is not a valid workstation profile")
        if record.get("dataset") != dataset or record.get("settings") != settings:
            raise ValueError(f"{path} has different settings; use the original profile or a fresh --output")
        original = record["jobs"]
        if not 1 <= original <= jobs:
            raise ValueError(f"resume needs {original} workers but only {jobs} are budgeted; restore headroom/--jobs")
        return original
    if any(output.iterdir()):
        raise ValueError(f"{output} is not empty and has no workstation profile; choose a fresh --output")
    with path.open("x") as stream:
        json.dump({"dataset": dataset, "jobs": jobs, "settings": settings}, stream, indent=2)
        stream.write("\n")
    return jobs


def require_idle_ports(ports):
    """A pre-existing Ollama would silently retain its own slot count, KV type and device."""
    for port in ports:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise ValueError(f"cannot use dedicated port {port}: {exc}; choose a free port") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("dataset", choices=PROFILES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the profile only (default); no hardware probes")
    mode.add_argument("--run", action="store_true", help="on the PC only: check resources and start the real benchmark")
    parser.add_argument("--jobs", type=int, help="worker ceiling (default: 7 HM3D scenes or 4 MP3D shards)")
    parser.add_argument("--slots", type=int, default=os.environ.get("OLLAMA_NUM_PARALLEL", "2"), help="Ollama request slots")
    parser.add_argument("--batch", type=int, default=os.environ.get("LLM_NUM_BATCH", "256"), help="Ollama prefill tokens")
    parser.add_argument("--llm-port", type=int, default=11450, help="dedicated loopback port (default: 11450)")
    parser.add_argument("--output", type=Path, help="campaign root to create or resume (default: runs/<dataset>_v1_val_4090)")
    args = parser.parse_args(argv)
    dataset, profile = args.dataset, PROFILES[args.dataset]
    ceiling = profile.workers if args.jobs is None else args.jobs
    if min(ceiling, args.slots, args.batch) < 1 or not 1 <= args.llm_port <= 65535:
        parser.error("jobs, slots and batch must be positive; the port must be in 1..65535")
    if dataset == "hm3d" and ceiling > 20:
        parser.error("HM3D v1 val has only 20 scene workers")
    output = (args.output or Path(os.environ.get("RUNS_ROOT", "runs")) / f"{dataset}_v1_val_4090").expanduser().absolute()
    try:
        settings = profile_environment(profile, args.slots, args.batch, args.llm_port)
        if dataset == "mp3d":
            settings["VRAM_CAP"] = "0.88"
        jobs = ceiling
        if args.run:
            jobs = workstation_jobs(profile, ceiling, settings)
            require_idle_ports((args.llm_port, int(settings["DETECTOR_PORT"])))
            jobs = campaign_jobs(output, dataset, jobs, settings)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    plan = output / "gpu_plan.json"
    settings["GPU_PLAN_REPLAY"] = str(plan) if args.run and plan.is_file() else ""
    settings["HABITAT_RENDERERS"] = str(jobs)
    command = [str(RUNTIME / dataset / "run_benchmark.sh"), "--version", "v1", "--agent", "rpt",
               "--perception", "service", "--jobs", str(jobs), "--output", str(output)]
    if dataset == "mp3d":
        command += ["--shard-episodes", "20", "--llm-servers", "1"]
    print(f"{dataset.upper()} v1 val: {profile.episodes} episodes | workers {jobs} (ceiling {ceiling}) | "
          f"one VLM server, {args.slots} slots, {args.batch}-token prefill")
    print(shlex.join(["env", *(f"{k}={v}" for k, v in settings.items()), *command]), flush=True)
    if args.run:
        os.execve(command[0], command, {**os.environ, **settings})
    print("Dry run only: no hardware probes, services or episodes. Use --run on the workstation.")


if __name__ == "__main__":
    main()
