"""Does the vision vote plus the object list beat either alone on the Matterport3D regions: agreement, who is right when they differ, and the fused answer."""
import argparse
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import agreement_summary, predict_group
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_mp3d_rooms import load_answers, load_truth

VISION = "vlm_qwen3-vl_4b-instruct_list"
LISTS = ("annotated", "yolo_world_inside", "llmdet_official_inside")
TEXT = "text_qwen2.5_14b-instruct_%s_%s"
FUSED = "fused_qwen2.5_14b-instruct_%s_list"


def predictions(path):
    return {room: predict_group(answers) for room, answers in load_answers(path).items()}


def accuracy(predicted, truth):
    rooms = [r for r in predicted if r in truth]
    return sum(predicted[r] == truth[r] for r in rooms) / max(1, len(rooms))


def report(rooms):
    truth, vision = load_truth(rooms), predictions(rooms / (VISION + ".jsonl"))
    out = ["# Vision vote plus object list on %d Matterport3D regions\n" % len(truth),
           "Vision vote = %s over about 6 photos per region. Text = qwen2.5 14B on the object list. Fused = the same model given the object list and the photo votes.\n" % VISION,
           "| object list | vision alone | text alone (open / list) | either right | fused | agree | right when agree | only vision right | only text right | neither |", "|---|---|---|---|---|---|---|---|---|---|"]
    for source in LISTS:
        texts = {p: rooms / ((TEXT % (source, p)) + ".jsonl") for p in ("open", "list")}
        fused_path = rooms / ((FUSED % source) + ".jsonl")
        if not all(f.exists() for f in texts.values()):
            continue
        text = predictions(texts["list"])
        s = agreement_summary(vision, text, truth)
        fused = "%.1f%%" % (100 * accuracy(predictions(fused_path), truth)) if fused_path.exists() else "-"
        out.append("| %s | %.1f%% | %.1f%% / %.1f%% | %.1f%% | %s | %d%% | %.0f%% | %d | %d | %d |"
                   % (source, 100 * accuracy(vision, truth), 100 * accuracy(predictions(texts["open"]), truth), 100 * accuracy(text, truth),
                      100 * (s["agree_right"] + s["only_first"] + s["only_second"]) / s["rooms"], fused, 100 * s["agree"] / s["rooms"],
                      100 * s["agree_right"] / max(1, s["agree"]), s["only_first"], s["only_second"], s["neither"]))
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    args = p.parse_args(argv)
    text = report(args.rooms)
    (args.rooms / "report_combination.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
