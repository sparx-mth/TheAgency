# Detector and perception review: ApexNav, SG-Nav, OSG Navigator

Reviewed 2026-09-15. **Research only: none of these papers' additional algorithms is implemented.** The user deferred them and separately selected YOLO-World X-v2 as the operational default, with LLMDet retained as an option.

## Sources and versions

1. **ApexNav: An Adaptive Exploration Strategy for Zero-Shot Object Navigation with Target-centric Semantic Fusion**, Mingjie Zhang et al., RA-L 2025 / ICRA 2026. [Paper 2504.14478v3](https://arxiv.org/abs/2504.14478v3), dated 2025-09-05. [Official code](https://github.com/Robotics-STAR-Lab/ApexNav), reviewed commit `049971b613fcd2554b034bd4a853d79cc1e61613`.
2. **SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation**, Hang Yin et al., NeurIPS 2024. [Paper 2410.08189v1](https://arxiv.org/abs/2410.08189v1). [Official code](https://github.com/bagh2178/SG-Nav), reviewed commit `d56863c96dea311aaa67fb0d39a1a8ccc3f0487f`.
3. **Open Scene Graphs for Open-World Object-Goal Navigation**, Joel Loo, Zhanxin Wu and David Hsu. [Paper 2508.04678v1](https://arxiv.org/abs/2508.04678v1), 2025 preprint; the PDF retains an unassigned IJRR DOI. No official implementation/checkpoint configuration was verified from its links and repository searches.

Current released repository defaults are not necessarily the exact historical experiment settings. Checkpoint-specific findings below come from code; method claims and tables come from the papers.

## Actual perception components

| Paper | Detectors and segmentation | Evidence |
|---|---|---|
| ApexNav | **YOLOv7-E6E** for COCO categories, **GroundingDINO Swin-T OGC** for other categories, **MobileSAM** box-prompted masks; BLIP-2 image/text scores. | Paper p.4 §IV-C2 and p.6; code `vlm/detector/yolov7.py:163`, `vlm/detector/grounding_dino.py:18-19`, README weight list. |
| SG-Nav | Released code uses **GLIP-L / Swin-L** for object/goal detection and **GroundingDINO Swin-T OGC + SAM ViT-H** for instance/scene-graph perception; some goal categories use the mask branch. | `SG_Nav.py:83-96,224-255`; `scenegraph.py:166-175,365-395`; paper p.4 describes online 3D instance segmentation. |
| OSG Navigator | **GroundingDINO** localization, **BLIP-2** crop/place descriptions and LLM goal matching. Exact detector checkpoint/backbone is not specified. | Paper p.2 and p.8 §5.2. |

YOLOv7 is closed-set COCO detection, not YOLO-World. SAM/MobileSAM supply masks, not category names by themselves. Zero-shot navigation does not imply unseen detector-training categories.

## FPS and precision/recall

| Paper | Numeric speed evidence | Object-box precision / recall / small-object recall |
|---|---|---|
| ApexNav | Detection **plus segmentation**: 190–320 ms on RTX 4060 Laptop and 65–125 ms on RTX 4090. Reciprocal equivalents: **3.1–5.3 FPS** and **8.0–15.4 FPS**. Desired perception interval 250 ms (4 Hz). Table VI, p.9. | **Not reported.** |
| SG-Nav | No numeric detector/full-loop FPS found. Figure 5 measures scene-graph **edge-generation** time, not detector throughput. | **Not reported.** |
| OSG Navigator | No numeric detector/full-loop FPS found; “near real-time” is not an FPS measurement. | **Not reported.** |

These are not comparable to detector-only HTTP timing on another device/vocabulary. ApexNav's FPS equivalents are derived from reported latency ranges, not measured full-loop throughput.

**OSG Table 7, p.15, is not detector precision/recall.** It measures scene-graph nodes/relations against human-annotated graphs: floor nodes 100/100%, room nodes 84.6/88.0%, door nodes 84.5/80.0%. Table 6 measures data-association accuracy. Table 10, p.32, lists a 200-pixel² minimum object-size filter, not small-object recall.

## Added algorithms and evidence

**ApexNav:** keeps competing target/confuser labels on spatial object clusters; extracts mask/depth point clouds, removes noise with DBSCAN, and weights multi-frame fusion by downsampled point counts. Expected-visible non-detections can reduce confidence. It also adds adaptive semantic/geometric exploration and safety-aware waypoint execution (pp.3–6). An offline LLM suggests confusers and thresholds, not statistically calibrated precision. Code caveat: missed detections use zero evidence for auxiliary labels but BLIP-2's image/text score for the target (`object_map2d.cpp:109-122`). HM3Dv2 navigation SR is 55.0% without fusion at the same low detector gate versus 76.2% with fusion (Table IV, p.8).

**SG-Nav:** builds object/group/room graphs from fused instances, generates/prunes relations and reasons hierarchically about frontiers. It approaches candidate goals and accumulates confidence weighted by distance-dependent subgraph plausibility, rejecting candidates that fail bounded verification (p.6). Paper settings include Nmax=10 and credibility threshold 0.8; LLaVA-1.6/Mistral-7B verifies short graph edges, with LLaMA-7B or GPT-4-0613 for reasoning (p.7). Current code also has category-specific observation counts and newer language-model setup. Re-perception raises HM3D navigation SR from 49.6% to 53.9% (Table 2, p.8).

**OSG Navigator:** generates/canonicalizes/verifies graph schemas, describes crops with BLIP-2, matches objects/places and goals semantically, and reasons coarse-to-fine through topological memory (pp.7–10). Simulation uses geometric local control; real robots use ViNT image-goal control. Its particle-filter multi-hypothesis mapper (§5.5) is a proposed extension, not evidence that all reported runs used it. Structural schema verification does not establish visual object presence.

## Conclusion

None establishes that its raw detector beats our YOLO-World X or LLMDet-L on matched images. Older models are not automatically worse; higher navigation SR does not establish better detector precision/recall. The strongest relevant idea is **reversible competing-label evidence**, not another model-size increase. All such algorithm work remains deferred.

Paper PDFs, extracted text/figures and upstream clones remain outside git under `$HOME/papers/`. Key numbers were cross-checked against arXiv HTML because the editor could not render the extracted PNGs; no visual-only claims are made. See [deferred integration notes](INTEGRATION.md) and the [local detector measurements](../../../sparx_agency/tasks/mapping/scene_graph/serve/DETECTOR_EVALUATION_20260915.md).

