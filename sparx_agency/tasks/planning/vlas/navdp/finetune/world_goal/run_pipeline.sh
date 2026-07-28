#!/usr/bin/env bash
# The whole offline pipeline, in order, with one command.
#
#   bash .../world_goal/run_pipeline.sh --preview   # ~15 min, proves the chain
#   bash .../world_goal/run_pipeline.sh             # the real run
#
# Both modes run the *same* six stages against the same code; --preview only
# strides the frames harder, draws fewer goals per frame and caps the training
# steps. So a green preview means the full run will work, and the full run is
# then literally the same command without the flag.
#
#   dataset   recordings + surveyed map -> labelled samples, split three ways
#   features  frozen-ViT patch tokens (optional; ~30x faster training)
#   train     the fine-tune, with the loss curves
#   evaluate  paired baseline-vs-trained on the held-out TEST wing
#   export    merge the fine-tune into a full NavDP checkpoint
#   report    everything above -> one self-contained HTML page
#
# Stages are skipped when their output already exists, so re-running after a
# failure resumes rather than repeating half an hour of A*. Force a stage with
# --redo <stage>, or run one stage on its own with --only <stage>.
#
# Everything runs in the `navdp` conda env. Not `.venv`: its pip-installed ompl
# bindings corrupt the heap at interpreter shutdown, so a run there prints
# correct results and then aborts with exit 134.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)"
WG=sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal
CONFIG_DIR="${REPO}/sparx_agency/tasks/planning/vlas/navdp/finetune/world_goal/configs"

OUT="${NAVDP_WG_OUT:-$HOME/data/navdp/world_goal}"
CKPT="${NAVDP_CKPT:-$HOME/Downloads/navdp-cross-modal.ckpt}"
SCENE="${NAVDP_WG_SCENE:-office}"
RECORDINGS="${NAVDP_WG_RECORDINGS:-$HOME/data/sim/office_v1 $HOME/data/sim/office_v2 $HOME/data/sim/office_v3 $HOME/data/sim/office_v4 $HOME/data/sim/falcon_pegasus}"
WORKERS="${NAVDP_WG_WORKERS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"

# Full-run defaults. --preview overrides all five.
RUN="${NAVDP_WG_RUN:-run1}"
DATASET="${NAVDP_WG_DATASET:-$OUT/dataset}"
STRIDE="${NAVDP_WG_STRIDE:-2}"
GOALS="${NAVDP_WG_GOALS:-12}"
TRAIN_EXTRA=()
EVAL_EXTRA=()

ONLY=""
REDO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preview)
      RUN="preview"; DATASET="$OUT/dataset_preview"; STRIDE=10; GOALS=8
      TRAIN_EXTRA=(--max-steps 1200 --val-every 100)
      EVAL_EXTRA=(--max-batches 25); shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --redo) REDO="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --run) RUN="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --no-features) SKIP_FEATURES=1; shift ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
FEATURES="${DATASET}_features"

cd "$REPO"
export NAVDP_REPO="${NAVDP_REPO:-$HOME/PycharmProjects/NavDP/baselines/navdp}"
py() { conda run --no-capture-output -n navdp python "$@"; }
want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
have() { [[ -e "$1" && "$REDO" != "$2" ]]; }
say() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

[[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT (set NAVDP_CKPT)" >&2; exit 1; }
say "run=$RUN  dataset=$DATASET  stride=$STRIDE  goals/frame=$GOALS  workers=$WORKERS"

if want dataset; then
  if have "$DATASET/index.json" dataset; then
    say "dataset: already built ($DATASET) -- --redo dataset to rebuild"
  else
    say "dataset: labelling against the surveyed $SCENE map"
    py -m $WG.build_dataset --recordings $RECORDINGS --scene "$SCENE" \
        --splits "$CONFIG_DIR/splits_${SCENE}.yaml" --out "$DATASET" \
        --frame-stride "$STRIDE" --goals-per-frame "$GOALS" --workers "$WORKERS"
  fi
fi

if want features && [[ -z "${SKIP_FEATURES:-}" ]]; then
  if have "$FEATURES/meta.json" features; then
    say "features: cache present ($FEATURES)"
  else
    say "features: pre-computing frozen DINOv2 tokens"
    py -m $WG.cache_features --dataset "$DATASET" --out "$FEATURES" --ckpt "$CKPT"
  fi
fi

FEATURE_ARG=()
[[ -f "$FEATURES/meta.json" && -z "${SKIP_FEATURES:-}" ]] && FEATURE_ARG=(--features "$FEATURES")

if want train; then
  if have "$OUT/$RUN/best.pth" train; then
    say "train: $OUT/$RUN/best.pth exists -- --redo train to retrain"
  else
    say "train: fine-tuning (validation is a different wing of the building)"
    py -m $WG.train --dataset "$DATASET" "${FEATURE_ARG[@]}" \
        --out "$OUT/$RUN" --ckpt "$CKPT" "${TRAIN_EXTRA[@]}"
  fi
fi

if want evaluate; then
  say "evaluate: paired baseline vs trained on the held-out TEST wing"
  py -m $WG.evaluate --dataset "$DATASET" "${FEATURE_ARG[@]}" --run "$OUT/$RUN" \
      --ckpt "$CKPT" "${EVAL_EXTRA[@]}"
fi

if want export; then
  say "export: merging the fine-tune into a full NavDP checkpoint"
  py -m $WG.export_checkpoint --run "$OUT/$RUN" --base "$CKPT" \
      --out "$OUT/$RUN/navdp-world-goal.ckpt"
fi

if want report; then
  say "report"
  py -m $WG.report --run "$OUT/$RUN" --dataset "$DATASET" \
      --flights "$OUT/$RUN/flights"
  echo "open $OUT/$RUN/report.html"
fi
