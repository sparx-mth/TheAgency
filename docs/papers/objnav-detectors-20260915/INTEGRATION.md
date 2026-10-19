# Deferred perception ideas for ObjectNav

**Status: deferred by the user, 2026-09-15. No ApexNav, SG-Nav or OSG Navigator algorithm has been implemented.** YOLO-World X-v2 is the selected default; LLMDet remains an explicit alternative. This document preserves research findings for possible later work, not a current implementation plan.

## Most relevant idea

ApexNav's reversible, competing-label evidence addresses a gap that repeated detection alone cannot close. A persistent false label can satisfy our current frame-count and viewpoint-separation gate. A future experiment could retain target/confuser support on one spatial hypothesis and penalize non-detection when geometry shows that the hypothesized object should be visible.

## Current safeguards versus the papers

| Existing infrastructure | Additional idea, not implemented |
|---|---|
| Alias-aware box suppression | Competing target/confuser labels on one object hypothesis. |
| Depth projection and nearest per-class 2D landmark association | Segmentation-backed 3D instance clouds and spatial fusion as in ApexNav/SG-Nav. |
| One observation per frame and viewpoint-separated target confirmation | Reversible weighted confidence and expected-visible negative observations; SG-Nav additionally uses scene-graph plausibility. |
| Revisable room labels, LLM probabilities, RPT room ordering, weighted A* and discrete actions | OSG schemas and semantic place recognition; its real-world ViNT controller is a different navigation component, not a detector upgrade. |

The existing hooks are `core/mapping/objects/landmarks.py` and the shared runtime's `methods/object_evidence.py`, `perception.py`, and `room_labels.py`. FALCON is not running in this ObjectNav pipeline. None of the proposed ideas requires replacing exploration, camera control, RPT or action execution merely to test a perception hypothesis.

## Possible future bounded experiment

1. Log competing-label evidence without initially allowing it to alter STOP.
2. Penalize expected-visible non-detections only when depth/occlusion evidence supports visibility; an occluded or out-of-frame object must not be penalized.
3. Compare the current gates against reversible evidence on annotated training/development sequences with persistent false positives, alternating labels and missed small objects.
4. Calibrate on designated development scenes, freeze settings, then measure precision/recall, false STOPs and lost true confirmations on separate measurement scenes.

This diagnostic experiment could start without new model weights. Exact replication of ApexNav's mask/point-cloud fusion would require additional perception payloads beyond our current label/score/box contract.

## Constraints

- Keep shared core logic ROS-free and lightweight, model imports in their separate runtime, and camera/world units explicit.
- Segmentation, GLIP-L/SAM-H and crop language-model queries add cost; their paper timings do not establish 8 GB co-resident renderer or Jetson performance.
- Do not copy LLM-suggested thresholds as empirical calibration. ApexNav's released target-label non-detection update also uses BLIP-2 context rather than always zero confidence.
- Scene-graph plausibility is not independent proof of object presence and can reinforce a persistent false label.
- OSG's published room/door node precision/recall evaluates graph reconstruction, not bounding boxes; its probabilistic mapper is a proposed extension.
- ApexNav code is GPLv3; SG-Nav code is MIT with separate upstream model/library terms. OSG implementation/checkpoint details were not verified. Direct code reuse would require license review.

Acceptance would require matched, frozen development evidence of better localization and fewer incorrect STOPs without losing true confirmations or exceeding the deployment budget. Until such work is explicitly requested, retain the implemented detector choices and existing navigation/evidence algorithms unchanged.

