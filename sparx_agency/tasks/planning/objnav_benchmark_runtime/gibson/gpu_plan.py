"""GPU placement for the Gibson runtime's three processes: who gets the card, in which order, who falls back to the CPU.

The runtime is three processes on one workstation -- the room/node LLM
(Ollama), the YOLO-World detector service and the Habitat renderer inside
``gibson.run`` -- and one GPU. Measured on the CPU, the LLM is the bottleneck
by an order of magnitude (seconds per room-classifier call, minutes per
node-oracle judgement on a 14B model), the detector next (a 640x480 YOLO-World
X frame is hundreds of milliseconds on four cores), and the renderer last (a
640x480 RGB-D frame is cheap either way). So the card is handed out in that
order, **LLM, then YOLO, then Habitat**, and whatever does not fit in the
VRAM that is left runs on the CPU -- the least bottlenecking component
first, by construction.

One constraint comes before the priorities: a component that *cannot* run on
the CPU in this installation is reserved first, or nothing runs at all. The
conda ``habitat-sim`` headless build renders through EGL on a GPU device;
software rendering (``HABITAT_CPU_RENDER=1``, Mesa's EGL on llvmpipe) is an
experiment the operator opts into, not a default this planner may assume.

Every footprint is an explicit, dated estimate (``*_VRAM_MIB`` environment
overrides exist for each) and is written into the plan with the measurement
it was decided on; the plan goes into the run directory and the frozen run
configuration, never into the policy's observations.

```
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.gpu_plan            # JSON
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.gpu_plan --format env  # shell exports
```
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import subprocess
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: Explicit VRAM estimates (MiB), measured on the development workstation 2026-10-08.
#: Ollama: resident Q4 weights x 1.35 (KV cache at 8k context plus compute buffers); the manifests give the weights.
LLM_OVERHEAD_FACTOR = 1.35
YOLO_WORLD_X_MIB = 2048          # YOLO-World X-v2 + CLIP ViT-B/32 text encoder, 640x480 inference, fp32
HABITAT_RENDER_MIB = 1536        # habitat-sim 0.2.4 headless, one agent, 640x480 RGB-D (observed 1.3 GB)
DEFAULT_RESERVE_MIB = 512        # the display / driver headroom the ``gibson.run`` GPU gate also keeps
PARTIAL_FRACTION = 0.5           # an LLM whose estimate fits at least this far is offloaded partially, not pushed to the CPU


@dataclass(frozen=True)
class Component:
    """One process competing for the card."""
    name: str
    priority: int                 # 1 is served first
    need_mib: int
    cpu_capable: bool
    partial: bool = False         # may split layers between GPU and CPU (Ollama)
    note: str = ""


@dataclass
class Placement:
    component: str
    device: str                   # "gpu", "gpu-partial" or "cpu"
    need_mib: int
    granted_mib: int
    reason: str


@dataclass
class GpuPlan:
    gpu_index: int
    gpu_name: str
    total_mib: int
    used_mib: int
    reserve_mib: int
    free_mib: int
    placements: List[Placement] = field(default_factory=list)
    shared: bool = False          # more than one of ours on the card: ``gibson.run`` needs --allow-shared-gpu
    warnings: List[str] = field(default_factory=list)

    def placement(self, name: str) -> Placement:
        for item in self.placements:
            if item.component == name:
                return item
        raise KeyError(name)

    def environment(self) -> Dict[str, str]:
        """The variables the launch scripts read to start each process on its device."""
        llm, yolo, habitat = (self.placement(n) for n in ("llm", "yolo", "habitat"))
        gpu = str(self.gpu_index)
        env = {"OBJNAV_LLM_DEVICE": llm.device, "OBJNAV_YOLO_DEVICE": yolo.device, "OBJNAV_HABITAT_DEVICE": habitat.device,
               # `ollama serve`: the card or none; per request, every layer or none (Ollama splits a partial load itself).
               "OLLAMA_CUDA_VISIBLE_DEVICES": gpu if llm.device != "cpu" else "-1",
               "LLM_NUM_GPU": "0" if llm.device == "cpu" else ("-1" if llm.device == "gpu" else ""),
               "DETECTOR_DEVICE": "cuda:0" if yolo.device == "gpu" else "cpu",
               "DETECTOR_CUDA_VISIBLE_DEVICES": gpu if yolo.device == "gpu" else "",
               "HABITAT_GPU_DEVICE": gpu,
               "HABITAT_SOFTWARE_RENDER": "1" if habitat.device == "cpu" else "0",
               "ALLOW_SHARED_GPU": "1" if self.shared else "0"}
        return env


def nvidia_smi_memory(gpu_index: int = 0, run: Callable[[Sequence[str]], str] = None) -> Tuple[str, int, int]:
    """``(name, total MiB, used MiB)`` of one GPU from nvidia-smi; raises when the tool or the card is missing."""
    command = ["nvidia-smi", "--id=%d" % gpu_index, "--query-gpu=name,memory.total,memory.used",
               "--format=csv,noheader,nounits"]
    text = (run or _run)(command)
    parts = [part.strip() for part in text.strip().splitlines()[0].split(",")]
    if len(parts) != 3:
        raise RuntimeError("Unexpected nvidia-smi output: %r" % text)
    return parts[0], int(float(parts[1])), int(float(parts[2]))


def _run(command: Sequence[str]) -> str:
    return subprocess.run(list(command), capture_output=True, text=True, check=True).stdout


def ollama_model_mib(models_dir: Path, name: str) -> Optional[int]:
    """Resident size of one pulled Ollama model from its manifest (sum of layer sizes), or None when not pulled."""
    repo, _, tag = name.partition(":")
    tag = tag or "latest"
    base = Path(models_dir).expanduser() / "manifests"
    if not base.is_dir():
        return None
    candidates = [base / "registry.ollama.ai" / "library" / repo / tag]
    candidates += [path for path in base.glob("*/*/%s/%s" % (repo, tag))]
    for path in candidates:
        if path.is_file():
            try:
                manifest = json.loads(path.read_text())
                return int(round(sum(int(layer["size"]) for layer in manifest["layers"]) / 2 ** 20))
            except (ValueError, KeyError, TypeError):
                return None
    return None


def llm_need_mib(models: Sequence[str], models_dir: Path, override: Optional[int] = None) -> Tuple[int, str]:
    """The LLM's VRAM estimate: both resident models' weights x the overhead factor, or the explicit override."""
    if override is not None:
        return int(override), "LLM_VRAM_MIB override"
    sizes = {name: ollama_model_mib(models_dir, name) for name in dict.fromkeys(models)}
    missing = [name for name, size in sizes.items() if size is None]
    if missing:
        raise RuntimeError("No Ollama manifest for %s under %s; pull the model or set LLM_VRAM_MIB"
                           % (", ".join(missing), models_dir))
    weights = sum(sizes.values())
    return int(round(weights * LLM_OVERHEAD_FACTOR)), "%s weights %d MiB x %.2f" % (
        " + ".join("%s %d" % item for item in sizes.items()), weights, LLM_OVERHEAD_FACTOR)


def components(llm_mib: int, llm_note: str, yolo_mib: int = YOLO_WORLD_X_MIB, habitat_mib: int = HABITAT_RENDER_MIB,
               habitat_cpu_render: bool = False) -> Tuple[Component, ...]:
    return (Component("llm", 1, llm_mib, cpu_capable=True, partial=True, note=llm_note),
            Component("yolo", 2, yolo_mib, cpu_capable=True, note="YOLO-World X-v2 + CLIP text encoder, 640x480"),
            Component("habitat", 3, habitat_mib, cpu_capable=habitat_cpu_render,
                      note="habitat-sim headless EGL renderer; CPU only with HABITAT_CPU_RENDER=1 (software EGL, experimental)"))


def plan(parts: Sequence[Component], gpu_index: int, gpu_name: str, total_mib: int, used_mib: int,
         reserve_mib: int = DEFAULT_RESERVE_MIB) -> GpuPlan:
    """Hand out the free VRAM: GPU-only components first (they have no alternative), then by priority.

    A component that fits takes its estimate off the budget; one that does
    not goes to the CPU -- except an LLM whose estimate fits at least
    ``PARTIAL_FRACTION`` of the way, which is loaded partially (Ollama
    splits layers between the card and the CPU itself) and takes the rest of
    the budget.
    """
    free = max(0, int(total_mib) - int(used_mib) - int(reserve_mib))
    result = GpuPlan(gpu_index, gpu_name, int(total_mib), int(used_mib), int(reserve_mib), free)
    budget = free
    order = sorted(parts, key=lambda c: (c.cpu_capable, c.priority))
    on_gpu = 0
    for part in order:
        if part.need_mib <= budget:
            result.placements.append(Placement(part.name, "gpu", part.need_mib, part.need_mib,
                                               "fits: %d MiB of %d left (%s)" % (part.need_mib, budget, part.note)))
            budget -= part.need_mib
            on_gpu += 1
        elif part.partial and budget >= PARTIAL_FRACTION * part.need_mib:
            result.placements.append(Placement(part.name, "gpu-partial", part.need_mib, budget,
                                               "partial: %d MiB of the %d MiB estimate fit; Ollama keeps the rest on the CPU"
                                               % (budget, part.need_mib)))
            budget = 0
            on_gpu += 1
        elif part.cpu_capable:
            result.placements.append(Placement(part.name, "cpu", part.need_mib, 0,
                                               "offloaded: needs %d MiB, %d MiB left" % (part.need_mib, budget)))
        else:
            result.placements.append(Placement(part.name, "gpu", part.need_mib, max(0, budget),
                                               "GPU-only component placed with %d MiB left; expect allocation failure"
                                               % budget))
            result.warnings.append("%s needs %d MiB but only %d MiB is free: stop whatever holds the card, or raise the "
                                   "reserve on another machine" % (part.name, part.need_mib, budget))
            budget = 0
            on_gpu += 1
    result.placements.sort(key=lambda item: [c.name for c in parts].index(item.component))
    result.shared = on_gpu > 1 or int(used_mib) > reserve_mib
    for item in result.placements:
        if item.component == "habitat" and item.device == "cpu":
            result.warnings.append("habitat on the CPU is software EGL rendering: slow, and unsupported by the conda headless build")
    return result


def plan_from_environment(gpu_index: int = 0, run: Callable[[Sequence[str]], str] = None,
                          env: Optional[Dict[str, str]] = None) -> GpuPlan:
    """The plan for this machine, from nvidia-smi and the ``LLM_*`` / ``*_VRAM_MIB`` environment."""
    env = os.environ if env is None else env
    name, total, used = nvidia_smi_memory(gpu_index, run)
    models = [env.get("LLM_MODEL", "qwen2.5:3b-instruct"), env.get("LLM_REASONING_MODEL", "qwen2.5:14b-instruct")]
    models_dir = Path(env.get("OLLAMA_MODELS_DIR", env.get("OLLAMA_MODELS", "~/models/objnav/ollama")))
    override = env.get("LLM_VRAM_MIB")
    llm_mib, note = llm_need_mib(models, models_dir, int(override) if override else None)
    parts = components(llm_mib, note, int(env.get("YOLO_VRAM_MIB", YOLO_WORLD_X_MIB)),
                       int(env.get("HABITAT_VRAM_MIB", HABITAT_RENDER_MIB)),
                       env.get("HABITAT_CPU_RENDER", "0") == "1")
    return plan(parts, gpu_index, name, total, used, int(env.get("GPU_RESERVE_MIB", DEFAULT_RESERVE_MIB)))


def render_env(result: GpuPlan) -> str:
    return "\n".join("export %s=%s" % (key, _quote(value)) for key, value in result.environment().items()) + "\n"


def _quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu-device", type=int, default=int(os.environ.get("GPU_DEVICE", "0")))
    parser.add_argument("--format", choices=("json", "env"), default="json")
    parser.add_argument("--write", type=Path, help="Also write the plan as JSON to this path")
    args = parser.parse_args(argv)
    result = plan_from_environment(args.gpu_device)
    record = dict(asdict(result), environment=result.environment())
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(record, indent=2) + "\n")
    print(render_env(result) if args.format == "env" else json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
