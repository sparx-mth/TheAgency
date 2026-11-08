# ObjectNav open-YOLO failure analysis (2026-11-08)

Status: interim. Includes the room-label replay (qwen2.5:3b) and the VLM object-listing test (qwen3-vl:8b); both are small, unverified experiments (no ground-truth room labels yet).
Raw traces, review pages and scripts are kept locally (HM3D frames are not committed: Matterport license).

## Question
Why does the open YOLO-World detector fail in the LLM-guided method (HM3D val, 4 scenes), and how much comes from room classification (MORE)
versus detection versus closing on an object. The algorithm is unchanged.

## A/B results (paired, same method, deterministic seeds, not strictly reproducible against the baseline)
| arm | scenes | pairs | SR base | SR arm | diff (95% CI) | p |
|---|---|---|---|---|---|---|
| X: RF-DETR for the six COCO targets + YOLO-World for the rest | 4 | 396 | 0.447 | 0.457 | +1.0 pt (-2.3..+4.5) | 0.665 |
| Y: X + VLM confirm at STOP | 2 | 198 | 0.343 | 0.384 | +4.0 pt (-1.0..+9.1) | 0.185 |
No detectable improvement. A 2-point gain would need about 2,400 pairs. The baseline runs were lost in a reboot, so per-category base vs X cannot be shown.

## Failures by target (X arm, 396 episodes)
tv 70% wrong stops, sofa 49%, plant 100% (n=17); bed/chair/toilet fail mostly by timeout.

## Open-YOLO trace: 75 sofa/tv episodes, scenes 00853 and 00890
59 wrong stops, 11 timeouts, 5 successes; sofa 35 of 37 episodes end in a wrong stop. The 59 wrong stops land on 16 spots (top two hold 24).
Visual review of the frame at each spot (not checked against ground truth):
- looks like a real sofa/tv: 8 spots, 29 stops (49%) - possible annotation / goal-definition gap, needs the ground-truth check;
- ambiguous: 2 spots, 15 stops (a dark glossy slab, a small black box on a shelf);
- false positive: 6 spots, 15 stops (painting as tv, railing, dining table/chairs, upholstered chair as sofa, bag in a mirrored closet, table).

## Room classification
- The room-label prompt gets the object list + room geometry. The list is the map's confirmed 3D landmarks (same-class landmarks merge within a radius,
  confirmed after a minimum number of observations), accumulated over the episode and credited to a room by floor cell, else the nearest room floor within footprint + 0.6 m. It is not the frame's detections.
- Duplicated landmarks (20 landmark-logged episodes, 270 room states): 54% have a same-class pair closer than 0.9 m. Merging within 0.9 m: beds 242 -> 118, cabinets 752 -> 389, ovens 47 -> 28 (an upper bound for cabinets; a real row also merges).
- Cabinet dominates every room (bedrooms: 72 cabinets vs 53 beds) and says nothing about the room type. Implausible lists (refrigerator in a bedroom, 17 cabinets + 2 ovens in an office) appear.
- Missing evidence: the detector vocabulary has 34 classes (no rug, ceiling fan, lamp, curtain, fireplace), so a living room with desk and chairs but no detected sofa/tv is labelled office.
- Label mix over 112 room states: hallway 27, bedroom 26, unknown 12, office 9, bathroom 9, storage closet 7, laundry 7, living room 5, foyer 5, dining room 3, kitchen 2.
- Not yet measured: how many labels are wrong and from which cause (false-positive landmark / wrong room / wrong label from a correct list). Needs a human review of the room cards.

## Room-label replay (qwen2.5:3b, temperature 0, 230 distinct prompts from the 20 landmark-logged episodes)
Label changed versus the traced label:
| change to the object list | rooms changed |
|---|---|
| none (same prompt again; noise floor) | 11 of 230 (5%) |
| drop cabinet | 46 (20%) |
| drop oven and refrigerator | 35 (15%) |
| drop cabinet, oven and refrigerator | 62 (27%) |
| merge same-class landmarks within 0.9 m | 21 (9%) |
| merge duplicates and drop the three | 72 (31%) |
The object list moves the label substantially: rooms whose only evidence is cabinets flip (storage_closet -> hallway 11 times, kitchen -> dining_room/laundry_room). Merging duplicates alone changes little (9%, 4 points above noise).
The replay shows sensitivity, not correctness: which version is right needs human judgement of the rooms. Replay does not model the classifier's cache or later re-queries.

## VLM object-listing test (qwen3-vl:8b on the frames saved while the agent stood inside each room)
Prompt: list the large furniture and fixtures clearly visible that help tell the room type; names are added to the list the label model receives.
- 78 of 112 rooms had saved frames inside the room; the VLM returned objects for 27 of them (the rest returned nothing within 1,500 output tokens, as a thinking model; not investigated further).
- For the 27 rooms with objects added the label changed in 15 (56%); for the 85 rooms with none added it changed in 5 (6%, noise).
- Changes that look right on reading the list: a living room labelled office (list had sofa x2, tv, desk, chair; VLM added ceiling fan, coffee table, fireplace, rug), empty "hallway" rooms to laundry_room (washer, dryer), foyer to bathroom (bathtub, shower, toilet), hallway to kitchen (sink, kitchen cabinets). Questionable: bathroom to bedroom, hallway to office with bed+desk+tv (frames chosen by agent position can include a neighbouring room).
- Not verified against ground truth. Frames are 320x240 and chosen by agent position inside the room outline.

## Next
Review the room cards and tally errors by cause; ground-truth check of the real-looking sofa spots (Nadav); review the changed rooms to decide whether the VLM-added objects improve labels, then decide whether to feed VLM object names to MORE or add discriminative classes (rug, fireplace, nightstand, ceiling fan, ...) to the detector vocabulary; per-category thresholds and vetoes once ground-truth boxes exist.
