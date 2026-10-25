# The services every benchmark launcher brings up, once: the GPU plan, the renderer, Ollama, the YOLO-World
# detector started with the benchmark's own vocabulary, the optional Segment Anything service, FFmpeg for
# recordings, and the cleanup trap that takes them down again. Sourced -- never executed -- by
# gibson/run_benchmark.sh and hm3d/run_benchmark.sh, so the two launchers cannot drift apart in how the
# method's services are provisioned; what differs per benchmark (episodes, scenes, the run module, the
# campaign shape) stays in the launcher.
#
# Two entry points:
#
#   objnav_services_defaults      before the command line is parsed: the environment defaults for the
#                                 interpreters, the weights, the ports, Ollama and the LLM, the GPU plan.
#                                 Every value is ${VAR:-default}; a launcher may set its own before or after.
#   objnav_services_up            after the command line is parsed and SERVICE_DIR/LOG/AGENT/RECORD/RUN_MODULE
#                                 are known: GPU plan -> renderer -> Ollama -> detector -> segmenter -> FFmpeg,
#                                 and the EXIT trap. Afterwards the launcher reads DETECTOR_URL, SEGMENTER_FLAGS,
#                                 RENDER_WRAP, RENDER_FLAGS, GPU_DEVICE and ALLOW_SHARED_GPU (both possibly
#                                 changed by the plan) and OBJNAV_{LLM,YOLO,HABITAT}_DEVICE.
#
# The launcher provides: HAB_PY DETECT_PY RUN_MODULE SERVICE_DIR LOG AGENT RECORD, and say()/die().
# RUN_MODULE is the benchmark's run CLI (its --print-vocabulary names the detector's classes, in the order
# the policy's label table pins).

objnav_services_defaults() {
  HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/habitat/bin/python}"
  DETECT_PY="${DETECT_PY:-$HOME/.venvs/objnav-detector/bin/python}"
  DATA="${DATA:-$HOME/datasets}"
  MODELS="${MODELS:-$HOME/models/objnav}"
  YOLO_WEIGHTS="${YOLO_WEIGHTS:-$MODELS/yolov8x-worldv2.pt}"
  DETECTOR_HOST="${DETECTOR_HOST:-127.0.0.1}"
  DETECTOR_PORT="${DETECTOR_PORT:-18095}"
  DETECTOR_THREADS="${DETECTOR_THREADS:-4}"
  # The segmentation experiment (2026-10-12): SEGMENTATION=1 starts the box-prompted Segment Anything service
  # beside the detector and runs the policy with its `segmentation` setting on; 0 (the default) is the box pipeline.
  SEGMENTATION="${SEGMENTATION:-0}"
  SAM_WEIGHTS="${SAM_WEIGHTS:-$MODELS/mobile_sam.pt}"
  SEGMENTER_PORT="${SEGMENTER_PORT:-18096}"
  SEGMENTER_DEVICE="${SEGMENTER_DEVICE:-cpu}"        # cpu: ~0.3 s per frame with target boxes (MobileSAM, 4 threads)
  SEGMENTER_THREADS="${SEGMENTER_THREADS:-4}"
  # Ollama: "docker" starts/stops the named container, "native" runs `ollama serve`, "external" only checks it.
  OLLAMA_MODE="${OLLAMA_MODE:-docker}"
  OLLAMA_CONTAINER="${OLLAMA_CONTAINER:-ollama-scene-graph}"
  OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$MODELS/ollama}"
  # Requests one native Ollama serves at once. 1 for a single run; a campaign with --jobs N needs more, or its
  # children queue on the LLM. Each slot holds its own KV cache (~1.5 GiB for the 14B at 8192 tokens).
  OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
  export LLM_BACKEND="${LLM_BACKEND:-ollama}"
  export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11434}"
  export LLM_MODEL="${LLM_MODEL:-qwen2.5:3b-instruct}"
  export LLM_REASONING_MODEL="${LLM_REASONING_MODEL:-qwen2.5:14b-instruct}"
  export LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-120}" LLM_SEED="${LLM_SEED:-17}" LLM_TEMPERATURE="${LLM_TEMPERATURE:-0}"
  export LLM_KEEP_ALIVE="${LLM_KEEP_ALIVE:-30m}" LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-768}"
  export LLM_REASONING_TIMEOUT_S="${LLM_REASONING_TIMEOUT_S:-900}" LLM_REASONING_MAX_TOKENS="${LLM_REASONING_MAX_TOKENS:-1536}"
  export LLM_REASONING_NUM_CTX="${LLM_REASONING_NUM_CTX:-8192}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-quiet}" MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
  SEED="${SEED:-17}"
  GPU_DEVICE="${GPU_DEVICE:-0}"
  ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
  # auto: gpu_plan.py decides each process's device from the free VRAM; off: CPU services, Habitat on the GPU.
  GPU_PLAN="${GPU_PLAN:-auto}"
  # gpu (default): habitat renders on the CUDA GPU. software: Mesa's llvmpipe, with /dev/dri hidden from the run in an
  # unprivileged user namespace so that Mesa's software device is the only EGL device (the Gibson 16384^2 texture does
  # not fit an iGPU, and habitat picks EGL device 0) -- for a box whose NVIDIA EGL is unusable, e.g. after a driver
  # upgrade until the reboot that loads the matching kernel module (2026-10-12). ~30 ms a frame; recorded in run.json.
  HABITAT_RENDER="${HABITAT_RENDER:-gpu}"
  CLIP_WEIGHTS="${CLIP_WEIGHTS:-$MODELS/clip/ViT-B-32.pt}"
  RUNS_ROOT="${RUNS_ROOT:-$REPO/runs}"
}

objnav_services_up() {
  [[ -n "${RUN_MODULE:-}" ]] || die "benchmark_services.sh: the launcher must set RUN_MODULE before objnav_services_up"
  [[ -n "${SERVICE_DIR:-}" && -n "${LOG:-}" ]] || die "benchmark_services.sh: the launcher must set SERVICE_DIR and LOG before objnav_services_up"

  # -- the GPU: who gets the card (LLM, then YOLO, then Habitat; the rest on the CPU) -------------------------
  if [[ "$AGENT" == falcon-yolo ]]; then
    # The baseline runs no LLM: the card goes to YOLO and Habitat, and the plan needs no Ollama manifest.
    export GPU_PLAN_LLM="${GPU_PLAN_LLM:-cpu}" LLM_VRAM_MIB="${LLM_VRAM_MIB:-0}"
  fi
  if [[ "$GPU_PLAN" == "auto" ]]; then
    export OLLAMA_MODELS_DIR
    PLAN_ENV="$("$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gpu_plan \
                 --gpu-device "$GPU_DEVICE" --format env --write "$SERVICE_DIR/gpu_plan.json")" \
      || die "gpu_plan.py could not place the processes (nvidia-smi? Ollama manifests under $OLLAMA_MODELS_DIR? set LLM_VRAM_MIB)"
    OPERATOR_SHARED_GPU="$ALLOW_SHARED_GPU"
    eval "$PLAN_ENV"                      # sets ALLOW_SHARED_GPU=1 when more than one process (or a desktop) holds the card
    [[ "$OPERATOR_SHARED_GPU" == 1 ]] && ALLOW_SHARED_GPU=1
    export OBJNAV_GPU_PLAN_JSON="$SERVICE_DIR/gpu_plan.json"
    say "GPU plan: LLM=$OBJNAV_LLM_DEVICE  YOLO=$OBJNAV_YOLO_DEVICE  Habitat=$OBJNAV_HABITAT_DEVICE  (gpu_plan.json; GPU_PLAN=off for CPU services)" | tee -a "$LOG"
    "$HAB_PY" - "$SERVICE_DIR/gpu_plan.json" <<'PY' | tee -a "$LOG"
import json, sys
plan = json.load(open(sys.argv[1]))
print("  %s: %d MiB total, %d used, %d free after a %d MiB reserve" % (plan["gpu_name"], plan["total_mib"], plan["used_mib"], plan["free_mib"], plan["reserve_mib"]))
for item in plan["placements"]:
    print("  %-8s %-11s %s" % (item["component"], item["device"], item["reason"]))
for warning in plan["warnings"]:
    print("  WARNING: " + warning)
PY
    GPU_DEVICE="$HABITAT_GPU_DEVICE"
    [[ "$HABITAT_SOFTWARE_RENDER" == 1 ]] && HABITAT_RENDER=software
  else
    OBJNAV_LLM_DEVICE=cpu OBJNAV_YOLO_DEVICE=cpu OBJNAV_HABITAT_DEVICE=gpu
    OLLAMA_CUDA_VISIBLE_DEVICES=-1 DETECTOR_DEVICE=cpu DETECTOR_CUDA_VISIBLE_DEVICES=""
    export LLM_NUM_GPU=""
    say "GPU plan off: CPU services, Habitat on GPU $GPU_DEVICE" | tee -a "$LOG"
  fi

  # -- the renderer -------------------------------------------------------------------------------
  RENDER_WRAP=() RENDER_FLAGS=()
  case "$HABITAT_RENDER" in
    gpu) ;;
    software)
      unshare -rm true 2>/dev/null || die "HABITAT_RENDER=software needs unprivileged user namespaces (unshare -rm) to hide /dev/dri from Mesa"
      RENDER_WRAP=(unshare -rm sh -c 'mount -t tmpfs none /dev/dri && exec "$@"' _)
      RENDER_FLAGS=(--software-render)
      say "renderer: Mesa llvmpipe (software), /dev/dri hidden from the run; the GPU gate is skipped" | tee -a "$LOG" ;;
    *) die "HABITAT_RENDER must be gpu or software" ;;
  esac

  # -- services ----------------------------------------------------------------------------------
  STARTED_OLLAMA=0 DETECTOR_PID="" SEGMENTER_PID="" OLLAMA_PID=""
  trap objnav_services_cleanup EXIT
  trap 'exit 130' INT TERM

  if [[ "$AGENT" == falcon-yolo ]]; then
    say "LLM: none (--agent falcon-yolo: goal-free FALCON + YOLO + the same closing; Ollama is neither started nor checked)" | tee -a "$LOG"
  else
  say "LLM: $OLLAMA_MODE at $LLM_BASE_URL ($LLM_MODEL for room labels, $LLM_REASONING_MODEL for the node oracle)" | tee -a "$LOG"
  if ! curl -fsS --max-time 5 "$LLM_BASE_URL/api/tags" > /dev/null 2>&1; then
    case "$OLLAMA_MODE" in
      docker) docker start "$OLLAMA_CONTAINER" > /dev/null || die "docker start $OLLAMA_CONTAINER failed"; STARTED_OLLAMA=1 ;;
      native) command -v ollama > /dev/null || die "ollama not on PATH (OLLAMA_MODE=native)"
              mkdir -p "$OLLAMA_MODELS_DIR"
              if [[ "$OLLAMA_CUDA_VISIBLE_DEVICES" == -1 ]]; then say "starting ollama on the CPU" | tee -a "$LOG"
              else say "starting ollama on GPU $OLLAMA_CUDA_VISIBLE_DEVICES (OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL)" | tee -a "$LOG"; fi
              CUDA_VISIBLE_DEVICES="$OLLAMA_CUDA_VISIBLE_DEVICES" OLLAMA_HOST="${LLM_BASE_URL#http://}" OLLAMA_MODELS="$OLLAMA_MODELS_DIR" \
                OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL" OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=30m ollama serve > "$SERVICE_DIR/ollama.log" 2>&1 &
              OLLAMA_PID=$!; STARTED_OLLAMA=1 ;;
      external) die "no LLM service answers at $LLM_BASE_URL (OLLAMA_MODE=external: start it yourself)" ;;
      *) die "OLLAMA_MODE must be docker, native or external" ;;
    esac
    objnav_wait_for_url "$LLM_BASE_URL/api/tags" 120 "Ollama"
  elif [[ "$OBJNAV_LLM_DEVICE" != cpu ]]; then
    say "NOTE: an Ollama already answers at $LLM_BASE_URL; the plan's LLM device ($OBJNAV_LLM_DEVICE) only holds if that server sees the card" | tee -a "$LOG"
  fi
  if [[ "$OLLAMA_MODE" == docker && "$STARTED_OLLAMA" == 1 && "$OBJNAV_LLM_DEVICE" != cpu ]]; then
    say "NOTE: the $OLLAMA_CONTAINER container's GPU access was fixed when it was created; the plan's LLM device is advisory in docker mode" | tee -a "$LOG"
  fi
  for model in "$LLM_MODEL" "$LLM_REASONING_MODEL"; do
    if ! objnav_ollama_has "$model"; then
      say "pulling $model (first time only; several GB)" | tee -a "$LOG"
      case "$OLLAMA_MODE" in
        docker) docker exec "$OLLAMA_CONTAINER" ollama pull "$model" ;;
        native) OLLAMA_HOST="${LLM_BASE_URL#http://}" ollama pull "$model" ;;
        *) die "$model is not provisioned on $LLM_BASE_URL" ;;
      esac
    fi
  done
  fi

  DETECTOR_URL="http://$DETECTOR_HOST:$DETECTOR_PORT"
  VOCAB="$(OMP_NUM_THREADS=2 "$HAB_PY" -m "$RUN_MODULE" --print-vocabulary 2>/dev/null | tail -1)"
  [[ -n "$VOCAB" ]] || die "$RUN_MODULE --print-vocabulary printed nothing (does $HAB_PY import the repo?)"
  printf '%s\n' "$VOCAB" > "$SERVICE_DIR/vocabulary.txt"
  if curl -fsS --max-time 5 "$DETECTOR_URL/health" > /dev/null 2>&1; then
    # A detector is pinned to its class list in that order (HttpDetector refuses another), so one started for
    # another benchmark's table -- Gibson's and HM3D's share the words but not the order -- cannot be reused.
    SERVED="$(curl -fsS --max-time 10 "$DETECTOR_URL/health" | "$HAB_PY" -c 'import json,sys; print(",".join(json.load(sys.stdin)["classes"]))')"
    [[ "$SERVED" == "$VOCAB" ]] || die "the detector at $DETECTOR_URL serves another vocabulary ($SERVED); set DETECTOR_PORT to start one with this benchmark's table, or stop it"
    say "detector already up at $DETECTOR_URL with this benchmark's vocabulary (left running)" | tee -a "$LOG"
  else
    DETECTOR_ARGS=(--backend yolo_world --model "$YOLO_WEIGHTS" --device "$DETECTOR_DEVICE" --host "$DETECTOR_HOST" --port "$DETECTOR_PORT"
                   --conf 0.05 --torch-threads "$DETECTOR_THREADS" --classes "$VOCAB")
    [[ -f "$CLIP_WEIGHTS" ]] && DETECTOR_ARGS+=(--clip-model "$CLIP_WEIGHTS")     # the local text encoder: no download on the card
    # The service keeps the same >512 MiB occupancy gate as the run; the plan has already accounted for who holds the card.
    [[ "$DETECTOR_DEVICE" == cuda* && "$ALLOW_SHARED_GPU" == 1 ]] && DETECTOR_ARGS+=(--allow-shared-gpu)
    say "starting the YOLO-World detector on $DETECTOR_DEVICE at $DETECTOR_URL" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$DETECTOR_CUDA_VISIBLE_DEVICES" "$DETECT_PY" -u -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
      "${DETECTOR_ARGS[@]}" > "$SERVICE_DIR/detector.log" 2>&1 &
    DETECTOR_PID=$!
    objnav_wait_for_url "$DETECTOR_URL/health" 300 "the detector"
  fi

  # -- the segmentation service (SEGMENTATION=1): Segment Anything prompted with the target's boxes ---------------
  SEGMENTER_FLAGS=()
  if [[ "$SEGMENTATION" == 1 ]]; then
    SEGMENTER_URL="http://$DETECTOR_HOST:$SEGMENTER_PORT"
    if curl -fsS --max-time 5 "$SEGMENTER_URL/health" > /dev/null 2>&1; then
      say "segmenter already up at $SEGMENTER_URL (left running)" | tee -a "$LOG"
    else
      SEGMENTER_ARGS=(--model "$SAM_WEIGHTS" --device "$SEGMENTER_DEVICE" --host "$DETECTOR_HOST" --port "$SEGMENTER_PORT"
                      --torch-threads "$SEGMENTER_THREADS")
      [[ "$SEGMENTER_DEVICE" == cuda* && "$ALLOW_SHARED_GPU" == 1 ]] && SEGMENTER_ARGS+=(--allow-shared-gpu)
      SEGMENTER_CUDA=""; [[ "$SEGMENTER_DEVICE" == cuda* ]] && SEGMENTER_CUDA="${SEGMENTER_DEVICE#cuda:}"
      say "starting the Segment Anything service ($(basename "$SAM_WEIGHTS")) on $SEGMENTER_DEVICE at $SEGMENTER_URL" | tee -a "$LOG"
      CUDA_VISIBLE_DEVICES="$SEGMENTER_CUDA" "$DETECT_PY" -u -m sparx_agency.tasks.mapping.scene_graph.serve.segmentation_server \
        "${SEGMENTER_ARGS[@]}" > "$SERVICE_DIR/segmenter.log" 2>&1 &
      SEGMENTER_PID=$!
      objnav_wait_for_url "$SEGMENTER_URL/health" 300 "the segmenter"
    fi
    SEGMENTER_FLAGS=(--segmentation --segmenter-url "$SEGMENTER_URL" --segmenter-model "$(basename "$SAM_WEIGHTS")")
    say "segmentation ON: the target's depth is read on its SAM mask (policy setting segmentation=true)" | tee -a "$LOG"
  fi

  # -- FFmpeg for --record: PATH, then any imageio-ffmpeg binary an environment on this machine ships ------
  if [[ "$RECORD" == 1 && -z "${IMAGEIO_FFMPEG_EXE:-}" ]] && ! command -v ffmpeg > /dev/null 2>&1; then
    FOUND="$("$DETECT_PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
    if [[ -z "$FOUND" || ! -x "$FOUND" ]]; then
      FOUND="$(ls -1 "$HOME"/miniconda3/envs/*/lib/python3.*/site-packages/imageio_ffmpeg/binaries/ffmpeg-* 2>/dev/null | head -1 || true)"
    fi
    if [[ -n "$FOUND" && -x "$FOUND" ]]; then
      export IMAGEIO_FFMPEG_EXE="$FOUND"
      say "ffmpeg for the recordings: $FOUND" | tee -a "$LOG"
    else
      die "--record needs FFmpeg: apt install ffmpeg, or set IMAGEIO_FFMPEG_EXE to an encoder with libx264"
    fi
  fi
}

objnav_services_cleanup() {
  code=$?
  trap - EXIT
  if [[ -n "$DETECTOR_PID" ]]; then kill -TERM "$DETECTOR_PID" 2>/dev/null || true; wait "$DETECTOR_PID" 2>/dev/null || true; fi
  if [[ -n "$SEGMENTER_PID" ]]; then kill -TERM "$SEGMENTER_PID" 2>/dev/null || true; wait "$SEGMENTER_PID" 2>/dev/null || true; fi
  if [[ "$STARTED_OLLAMA" == 1 ]]; then
    case "$OLLAMA_MODE" in
      docker) docker stop --timeout 15 "$OLLAMA_CONTAINER" > /dev/null 2>&1 || true ;;
      native) if [[ -n "$OLLAMA_PID" ]]; then kill -TERM "$OLLAMA_PID" 2>/dev/null || true; wait "$OLLAMA_PID" 2>/dev/null || true; fi ;;
    esac
  fi
  date -u +%FT%TZ > "$SERVICE_DIR/finished_utc.txt"
  printf '%s\n' "$code" > "$SERVICE_DIR/exit_code.txt"
  say "exit $code (log: $LOG)"
  exit "$code"
}

objnav_wait_for_url() {   # url, seconds, label
  local url="$1" seconds="$2" label="$3" waited=0
  until curl -fsS --max-time 5 "$url" > /dev/null 2>&1; do
    sleep 2; waited=$((waited + 2))
    [[ $waited -lt $seconds ]] || die "$label did not answer at $url within ${seconds}s"
  done
}

objnav_ollama_has() { curl -fsS --max-time 10 "$LLM_BASE_URL/api/tags" | "$HAB_PY" -c "import json,sys; sys.exit(0 if '$1' in [m['name'] for m in json.load(sys.stdin)['models']] else 1)"; }
