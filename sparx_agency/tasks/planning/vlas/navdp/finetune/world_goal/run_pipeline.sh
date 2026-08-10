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
# One building (the default) or several:
#
#   NAVDP_WG_SCENE=office                       # one, splits_office.yaml
#   NAVDP_WG_SCENES="office warehouse_shelves"  # several
#   NAVDP_WG_SPLITS=.../splits_multiscene.yaml  # required with SCENES, since
#                                               # the filename can no longer be
#                                               # derived from one scene name
#
# Check a multi-scene split plan puts samples in all three splits with a cheap
# `build_dataset --frame-stride 10` before spending a training run on it: a
# split band is a claim about *reachable* area, and an empty TEST split is only
# discovered by evaluate, after training has already cost hours.
#
# Everything runs in the `navdp` conda env. Not `.venv`: its pip-installed ompl
# bindings corrupt the heap at interpreter shutdown, so a run there prints
# correct results and then aborts with exit 134.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)"
WG=sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal
CONFIG_DIR="${REPO}/sparx_agency/tasks/planning/vlas/navdp/finetune/world_goal/configs"

OUT="${NAVDP_WG_OUT:-$HOME/data/navdp/world_goal}"
CKPT="${NAVDP_CKPT:-$HOME/GIT/NavDP/baselines/navdp/checkpoints/navdp-cross-modal.ckpt}"
SCENE="${NAVDP_WG_SCENE:-office}"
# Several buildings, space separated. Each recording is labelled against the one
# it was actually flown in (meta.json says which), so this is not a claim that
# every recording belongs to every scene. An unseen *building* is a far stronger
# generalisation test than an unseen wing of a building the policy trained on,
# which is what splits_multiscene.yaml exists for -- but the split YAML could
# not be reached from here before, because the filename was derived from the
# single scene name.
SCENES="${NAVDP_WG_SCENES:-}"
SPLITS="${NAVDP_WG_SPLITS:-}"
RECORDINGS="${NAVDP_WG_RECORDINGS:-$HOME/data/sim/office_v1 $HOME/data/sim/office_v2 $HOME/data/sim/office_v3 $HOME/data/sim/office_v4 $HOME/data/sim/falcon_pegasus}"
WORKERS="${NAVDP_WG_WORKERS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"

# Full-run defaults. --preview overrides all five.
#
# DATASET is deliberately *not* resolved here. It defaults to a path under $OUT,
# and --out can still change $OUT further down, so computing it now would leave
# `--out elsewhere` writing its run into the new directory and its dataset into
# the old one. It is resolved after the argument loop instead.
RUN="${NAVDP_WG_RUN:-run1}"
DATASET="${NAVDP_WG_DATASET:-}"
STRIDE="${NAVDP_WG_STRIDE:-2}"
GOALS="${NAVDP_WG_GOALS:-12}"
PREVIEW=0
# Extra arguments for the two stages that have knobs worth reaching from here --
# above all `--config`, since the training config is per machine (see
# navdp_world_goal_4090.yaml, whose batch size and worker count are measured on
# a 24 GB card and are not guesses). --preview adds its caps on top.
# shellcheck disable=SC2206  # deliberate word split: these are argument lists
TRAIN_EXTRA=(${NAVDP_WG_TRAIN_EXTRA:-})
# shellcheck disable=SC2206
EVAL_EXTRA=(${NAVDP_WG_EVAL_EXTRA:-})

ONLY=""
REDO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preview)
      RUN="preview"; PREVIEW=1; STRIDE=10; GOALS=8
      TRAIN_EXTRA+=(--max-steps 1200 --val-every 100)
      EVAL_EXTRA+=(--max-batches 25); shift ;;
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

# Now that --out and --preview have both been seen, $OUT is final.
if [[ -z "$DATASET" ]]; then
  DATASET="$OUT/dataset"
  (( PREVIEW )) && DATASET="$OUT/dataset_preview"
fi
FEATURES="${DATASET}_features"

cd "$REPO"
export NAVDP_REPO="${NAVDP_REPO:-$HOME/GIT/NavDP/baselines/navdp}"
NAVDP_REPO="${NAVDP_REPO/#\~/$HOME}"
py() { conda run --no-capture-output -n navdp python "$@"; }
want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
have() { [[ -e "$1" && "$REDO" != "$2" ]]; }
# A run is "trained" only once RunLogger.finish has written its summary. Gating
# on best.pth instead would treat a run killed at step 40k of 260k as complete,
# and then evaluate, export and report a model that saw 15% of its schedule --
# at a learning rate still near peak, with nothing anywhere saying so.
trained() { [[ "$REDO" != train && -f "$1" ]] && grep -q '"summary"' "$1"; }
say() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# Both preconditions are checked before the first stage, because the dataset
# stage takes hours and does not need either -- so a wrong path here otherwise
# surfaces as a failure late at night, after the expensive part.
[[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT (set NAVDP_CKPT)" >&2; exit 1; }
[[ -f "$NAVDP_REPO/policy_network.py" ]] || {
  echo "NavDP repo not found: $NAVDP_REPO" >&2
  echo "  set NAVDP_REPO to the directory containing policy_network.py" >&2
  exit 1
}
if [[ -n "$SCENES" ]]; then
  # shellcheck disable=SC2206  # deliberate word split: several scene names
  SCENE_ARG=(--scenes $SCENES)
else
  SCENE_ARG=(--scene "$SCENE")
fi
[[ -n "$SPLITS" ]] || SPLITS="$CONFIG_DIR/splits_${SCENE}.yaml"
[[ -f "$SPLITS" ]] || { echo "split plan not found: $SPLITS (set NAVDP_WG_SPLITS)" >&2; exit 1; }

say "run=$RUN  scenes=${SCENES:-$SCENE}  splits=$(basename "$SPLITS")"
say "dataset=$DATASET  stride=$STRIDE  goals/frame=$GOALS  workers=$WORKERS"

if want dataset; then
  if have "$DATASET/index.json" dataset; then
    say "dataset: already built ($DATASET) -- --redo dataset to rebuild"
  else
    say "dataset: labelling against the surveyed ${SCENES:-$SCENE} map(s)"
    py -m $WG.build_dataset --recordings $RECORDINGS "${SCENE_ARG[@]}" \
        --splits "$SPLITS" --out "$DATASET" \
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
  if trained "$OUT/$RUN/run.json"; then
    say "train: $OUT/$RUN ran to completion -- --redo train to retrain"
  else
    [[ -f "$OUT/$RUN/best.pth" ]] && say "train: $OUT/$RUN was interrupted -- training again from the start"
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
