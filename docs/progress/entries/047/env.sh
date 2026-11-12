# The one environment of the MP3D 14B ground-truth run (2026-11-11): sourced by the pilot, the full run and every resume.
# The launcher reads these; nothing else is set on the command line but the run's own flags.
export HAB_PY="$HOME/miniconda3/envs/objnav-habitat/bin/python"
export MP3D_SCENES_DIR="$HOME/datasets/mp3d/scene_datasets"          # mp3d_habitat.zip unpacked here (90 scenes)
export RUNS_ROOT="$HOME/GIT/TheAgency/runs"
# The LLM: ONE model in both roles (room labels and node oracle), resident native Ollama on 11435 started by
# start_ollama.sh with Ollama's own defaults otherwise (f16 KV cache, flash attention auto): 4 request slots, 8k
# context = 15.1 GB. The pilot of 2026-11-11 showed the LLM capped at ~0.5 requests/s by the JSON-grammar sampling
# of the client's format=json (CPU, serialised in Ollama's runner) whatever the slot count (4 or 8) or the worker
# count (7 or 12), so 7-8 renderers saturate it and a smaller KV cache (q8_0, tried in the pilot) buys nothing.
# The launcher budgets a resident server as placed.
export OLLAMA_MODE=native LLM_BASE_URL=http://127.0.0.1:11435
export LLM_MODEL=qwen2.5:14b-instruct LLM_REASONING_MODEL=qwen2.5:14b-instruct
export OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_MODELS_DIR="$HOME/models/objnav/ollama"
# One renderer's VRAM for the sharded pool's start gate (refined by the pilot's measurement) and the cap it fills to.
export HABITAT_VRAM_MIB="${HABITAT_VRAM_MIB:-1000}"     # measured 2026-11-11: 270-1,180 MiB per MP3D GT renderer by scene, ~820 mean
export VRAM_CAP="${VRAM_CAP:-0.90}"
# Perception: ground truth with the on-object filter at the module default (0.25 m, ON); nothing overridden here.
unset GT_ON_OBJECT_M GT_MIN_PIXELS
export PERCEPTION=ground-truth
