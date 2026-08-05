# Set up and train the NavDP fine-tune

Portable, torch-only path (no GPU-specific TensorRT engines). Everything is driven by a
few path variables. **Part A** is a 5-minute local smoke test to confirm everything runs
and to watch the training log; **Part B** is the full setup on a fresh machine.

You need a **CUDA GPU** (~6 GB free is plenty; the head-only fine-tune is small).

---

## Part A — Quick local test (verify it all runs)

Runs on this machine as-is. It makes a small label set on **one** recording, then trains a
few hundred steps with **frequent validation** so you see ~20 log rows and a saved
`best.pth`. Expect ~5 minutes.

```bash
export PYTHONPATH=$HOME/GIT/TheAgency
export NAVDP_REPO=$HOME/PycharmProjects/NavDP/baselines/navdp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/miniconda3/envs/navdp/bin/python
PKG=sparx_agency.tasks.planning.vlas.navdp.finetune.pixel_goal

# 1) labels on ONE recording, small + fast (torch backend = the portable path)
$PY -m $PKG.pixel_labels --backend torch --recs rec2 \
  --n-per-frame 20 --frame-stride 3 --exclude-bottom-frac 0.2 \
  --corrector esdf --clearance 0.3 --max-shift 0.2 --out-name labels_test.npz

# 2) short training with FREQUENT validation -> many log rows + best.pth
$PY -m $PKG.train_pixel --labels $HOME/flight_dataset/rec2/labels_test.npz \
  --epochs 3 --batch-size 4 --lr 5e-5 --l2sp 1e-2 \
  --val-frac 0.15 --val-every 25 --max-steps 500 \
  --out-dir $HOME/flight_dataset/rec2/run_test
```

You'll see a table like:

```
train=782  val=138  steps/epoch=195  target_steps=500  batch=4  lr=5.0e-05
  step | epoch |    lr    | train_tot | val_act val_crit val_esdf | val_tot | best_val | status
-----------------------------------------------------------------------------------------------
    25 |  0.13 | 5.00e-05 |    12.803 |   0.071  10.240   0.014   |  10.325 |   10.325 | *** NEW BEST -> best.pth
    50 |  0.26 | 5.00e-05 |     9.114 |   0.058   7.900   0.012   |   7.970 |    7.970 | *** NEW BEST -> best.pth
    75 |  0.38 | 5.00e-05 |     7.640 |   0.055   8.210   0.011   |   8.276 |    7.970 | no improve x1
   ...
```

`best.pth` (lowest `val_tot`) is re-saved on every improvement; `ema_latest.pth` is the
most recent. When it finishes, look at the two models on that recording:

```bash
$PY -m $PKG.interactive_compare --dataset $HOME/flight_dataset --rec rec2 \
  --finetuned $HOME/flight_dataset/rec2/run_test/best.pth
```

If that all worked, the full pipeline is correct — scale it up with Part B / Part C.

---

## Part B — Full setup on another machine

### Step 0 — get the code + a package manager

```bash
# a) miniconda (skip if you already have conda)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# b) this repo (the SPARX code)
git clone <your-remote>/TheAgency.git

# c) the external NavDP repo — it ALSO contains depth_anything, which the model needs
git clone https://github.com/InternRobotics/NavDP.git
```

Also copy over two files that are not in git: the checkpoint **`navdp-cross-modal.ckpt`**
and your **`flight_dataset/`** recordings (extracted `rgb/` + `depth/` per recording).

### Step 1 — environment + libraries

```bash
conda create -y -n navdp python=3.10
conda activate navdp

# PyTorch matching THIS machine's CUDA — pick the right line from pytorch.org, e.g. CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# everything else (no transformers / timm / tensorrt / pycuda on the torch path)
pip install numpy scipy opencv-python matplotlib pyyaml diffusers pillow

# optional: only silences the "[costmap] Numba unavailable" notice — it does NOT change
# any results (numba isn't on the label-gen / correction path, which is numpy + scipy)
pip install numba
```

### Step 2 — define the paths (edit these lines)

```bash
export REPO=$HOME/TheAgency                                  # this repo
export NAVDP_REPO=$HOME/NavDP/baselines/navdp                # note the .../baselines/navdp subdir
export DATA=$HOME/flight_dataset                             # recordings
export CKPT=$HOME/navdp-cross-modal.ckpt                     # the checkpoint
export PYTHONPATH=$REPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PKG=sparx_agency.tasks.planning.vlas.navdp.finetune.pixel_goal        # module path used in Part C
```

> All the Part C commands below reuse these variables, so run this whole block once
> per shell session (or add it to your `~/.bashrc`).

Sanity-check the model loads:

```bash
python -c "from sparx_agency.tasks.planning.vlas.navdp.finetune.pixel_goal.navdp_torch import TorchNavDP; \
TorchNavDP('$CKPT','$NAVDP_REPO'); print('NavDP loads OK')"
```

---

## Part C — the full run

Labels → train (with validation) → evaluate on the held-out recording. `walk_into` is held
out (never trained on) so the eval measures generalization.

```bash
# (uses $PKG, $CKPT, $NAVDP_REPO, $DATA from Step 2 — set that block first)

# (1) LABELS — 7 recordings, 150 goals/frame from the top 4/5, very-low ESDF correction
python -m $PKG.pixel_labels --backend torch --ckpt "$CKPT" --navdp-repo "$NAVDP_REPO" \
  --dataset "$DATA" --out-name labels_train.npz \
  --recs rec2 rec3 rec4 rec5 rec6 rosbag2_2026_06_02-16_38_54 rosbag2_2026_06_09-17_38_17 \
  --n-per-frame 150 --exclude-bottom-frac 0.2 \
  --corrector esdf --clearance 0.3 --max-shift 0.2 --smooth 0.5

# (2) TRAIN — all 7 concatenated, 10% held out for validation, logs every 500 steps + epoch
python -m $PKG.train_pixel \
  --labels $DATA/rec2/labels_train.npz $DATA/rec3/labels_train.npz $DATA/rec4/labels_train.npz \
           $DATA/rec5/labels_train.npz $DATA/rec6/labels_train.npz \
           $DATA/rosbag2_2026_06_02-16_38_54/labels_train.npz \
           $DATA/rosbag2_2026_06_09-17_38_17/labels_train.npz \
  --ckpt "$CKPT" --navdp-repo "$NAVDP_REPO" \
  --epochs 3 --batch-size 4 --lr 5e-5 --l2sp 1e-2 --ema-decay 0.999 \
  --val-frac 0.1 --val-every 500 --out-dir $DATA/run_new

# (3) EVALUATE the best checkpoint on the held-out recording
python -m $PKG.evaluate --dataset "$DATA" --rec walk_into \
  --ckpt "$CKPT" --navdp-repo "$NAVDP_REPO" \
  --finetuned $DATA/run_new/best.pth --out $DATA/run_new/eval.png
```

Use **`best.pth`** (lowest validation loss). `interactive_compare` (Part A) shows the two
models live.

**Knobs:** `--batch-size` (raise with more GPU memory), `--val-every` (log cadence),
`--val-frac`, `--epochs`, `--max-steps` (cap total steps), `--lr`, `--l2sp` (higher = stays
closer to pretrained). Correction strength is set at label time (`--clearance`/`--max-shift`).
Camera geometry self-corrects per frame (the occupancy fits the ground plane), so you do
**not** set pitch/height.

> On this machine you may use `--backend trt` for faster label generation (the prebuilt
> engines are present); `torch` is the portable default that works everywhere.
