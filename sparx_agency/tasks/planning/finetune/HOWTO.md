# VLA fine-tune — HOW TO RUN (data · visualization · training)

One page. Copy-paste the commands. Details live in `verify/README.md` (the tool) and
`README.md` (the method/architecture).

---

## 1. Where is all the data?

| what | path | notes |
|---|---|---|
| **Extracted training frames** | `~/flight_dataset/<rec>/` | **881 synced RGB+depth pairs, 614 MB, 8 recordings.** This is what you train/visualize on. |
| ↳ per recording | `rgb/NNNNNN.png` · `depth/NNNNNN.npy` · `intrinsics.json` · `meta.json` · `pairs.csv` | same index `NNNNNN` = one matched color+depth pair (float32 meters depth) |
| **Source ROS bags** | `~/Downloads/OneDrive_1_6-1-2026/` | 3.3 GB, the raw Humble rosbag2 recordings (rec2…rec6, rosbag2_*, walk_into) |
| **NavDP weights / engines** | `~/Downloads/navdp-cross-modal.ckpt`, `sparx_agency/tasks/planning/navdp/engines/nvidiageforcertx_sm120/` | the built fp16 TensorRT engines the tool runs |
| **Env** | `~/miniconda3/envs/navdp/` | conda env with TensorRT + numpy + matplotlib (run everything here) |

Recording → pairs:  rec2 139 · rec3 136 · rec4 136 · rec5 162 · rec6 91 ·
rosbag2_2026_06_02-16_38_54 45 · rosbag2_2026_06_09-17_38_17 91 · walk_into 81.

---

## 2. ▶ Run the visualization (do this BEFORE training)

Verify the training signal (NavDP trajectory vs. PF/ESDF-corrected target) by clicking a
pixel goal on a real frame. **This is the bottom line — one command:**

```bash
PYTHONPATH=/home/nadavc/GIT/TheAgency \
  ~/miniconda3/envs/navdp/bin/python -m \
  sparx_agency.tasks.planning.finetune.verify.interactive_verify \
  --dataset ~/flight_dataset --rec walk_into --frame 40
```

Then **click the colour or depth image** to set a goal. You'll see four panels: the
colour + depth images with your clicked pixel, the **instantaneous potential field / ESDF**
with NavDP (orange) and the **corrected** trajectory (green), and a **side-by-side
comparison**. Sliders retune the push live (corrector, clearance, max-shift, camera
pitch/height). `◀ frame / rec ▶` to move around; `sample 25` overlays many goals.

**Headless variant** (renders a PNG grid, no window — good over SSH or to share):

```bash
PYTHONPATH=/home/nadavc/GIT/TheAgency \
  ~/miniconda3/envs/navdp/bin/python -m \
  sparx_agency.tasks.planning.finetune.verify.batch_preview \
  --rec walk_into --frame 40 --n 6 --out /tmp/preview.png
```

Full knob reference: `verify/README.md`.

---

## 3. (Re)extract the data from the bags — only if needed

`~/flight_dataset` already exists. To regenerate it from the raw bags:

```bash
PYTHONPATH=/home/nadavc/GIT/TheAgency .venv/bin/python -m \
  sparx_agency.tasks.planning.finetune.datasets.bag_extract \
  --bags-root ~/Downloads/OneDrive_1_6-1-2026 --out-root ~/flight_dataset
```

(Reads the `.db3` bags directly, matches each depth frame to its RGB by header
timestamp. Runs in the plain `.venv`; ~4 s for all 8.)

---

## 4. Run the training (pose-free, pixel-goal — your method)

The loop is: **generate labels → short train → evaluate → adjust → repeat**, all in the
`navdp` conda env, all pose-free (`train/` package). Prefix every command with
`PYTHONPATH=/home/nadavc/GIT/TheAgency NAVDP_REPO=~/PycharmProjects/NavDP/baselines/navdp`
and run with `~/miniconda3/envs/navdp/bin/python`.

```bash
# 1) labels: sample pixel goals, run NavDP, correct+smooth -> per-frame labels (~2-3 min)
python -m sparx_agency.tasks.planning.finetune.train.pixel_labels \
  --rec walk_into --n-per-frame 12            # -> ~/flight_dataset/walk_into/labels.npz

# 2) SHORT train on one video (frozen backbone, head only)
python -m sparx_agency.tasks.planning.finetune.train.train_pixel \
  --labels ~/flight_dataset/walk_into/labels.npz \
  --epochs 4 --batch-size 2 --ema-decay 0.99 --out-dir ~/flight_dataset/walk_into/run1

# 3) EVALUATE trained vs untrained (metrics table + side-by-side PNG)
python -m sparx_agency.tasks.planning.finetune.train.evaluate \
  --rec walk_into --finetuned ~/flight_dataset/walk_into/run1/ema_latest.pth --out eval.png
```

**GPU note:** this is an 8 GB GPU; use `--batch-size 2` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Free other GPU processes first if you
hit CUDA OOM.

**Before the long run on all videos:** measure the camera **pitch** and **height** on the
XTEND (placeholders 0°/1.0 m — set them via `--pitch/--height` in label-gen + eval and
confirm the ESDF walls match the scene in the verifier), then generate labels for every
recording and train longer with the full `navdp_finetune.yaml` (EMA 0.999, more epochs).

> The pose-based trainer (`navdp/train.py`, needs `poses.npy`) still exists for the
> flown-future labeling method; the `train/` package above is the pixel-goal method.
