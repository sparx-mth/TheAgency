# Segmentation

Box-prompted instance segmentation: RGB frame + detection boxes → one boolean
`H×W` mask per box, the pixels of the object each box was drawn around.
ROS-free and platform-independent; depth lifting and navigation remain
separate concerns.

## Why it exists

A detection box overhangs its object and admits whatever stands **in front**
of it. The depth read for a detection
(`core/mapping/objects/geometry.py::robust_bbox_depth`, a low percentile of
the shrunken box) therefore reads an occluder's range when one overlaps the
box -- a chair in front of a sofa makes the sofa "reached" a metre early. The
mask restricts the read to the object's own pixels
(`robust_mask_depth`), which is the one thing a box cannot do.

## Interface

`SegmentationModel` lives in `core/mapping/interfaces/segmentation_model.py`:

- `segment(rgb, boxes_xyxy)` takes an RGB uint8 H×W×3 array and the pixel
  boxes, and returns `len(boxes)` boolean H×W masks **in box order**. A box
  the model cannot resolve is an all-`False` mask, never a missing entry, so
  the caller lines masks up with its detections by position.

Backends import their model dependencies lazily. Importing the package and
constructing a segmenter need no Torch.

## Backends

- **`SamPromptedSegmenter` / `SamPromptedConfig`** — the Segment Anything
  family through `ultralytics.SAM`, prompted with the boxes. One class serves
  every variant; the checkpoint decides: `mobile_sam.pt` (default, 39 MB,
  ViT-Tiny encoder -- the one a CPU beside a simulator runs per frame),
  `sam2.1_t.pt` (76 MB, the stronger small model), `sam_b.pt` and up. No
  checkpoint is downloaded: provision it explicitly (the ultralytics asset
  release, `attempt_download_asset`) and point `model_path` at it.

Measured on the development laptop (4 CPU threads, 640 × 480, three boxes):
MobileSAM 0.29 s/frame, SAM 2.1-tiny 0.60 s/frame. The image encoder
dominates, so segmenting every box of a frame in one call costs about the
same as one box. On the synthetic sofa-with-a-chair-in-front frame both
variants' sofa masks left the chair out (61.6 k of the box's 80 k pixels).

Not chosen: FastSAM (YOLOv8-seg "segment everything" with a box *filter*:
the mask is picked by IoU with the box, so an occluder inside the box can be
picked with it), and `yolov8-seg` (closed vocabulary; the open-vocabulary
detector's boxes are the prompt here).

## Where it is used

The HTTP service `tasks/mapping/scene_graph/serve/segmentation_server.py`
(torch venv) serves it beside the detector service; the ObjectNav runtime's
`HttpSegmenter` client and the `segmentation` setting of
`tasks/planning/objnav_benchmark_runtime/methods/rpt_settings.py` turn the
masked depth read on for the target's boxes (off by default).

Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
sparx_agency/core/mapping/segmentation/tests -q` (no torch; the model is
faked).
