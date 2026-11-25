"""Run A then B on the frozen IDs for this branch. No benchmark is started without --run."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
from urllib.request import urlopen

MP3D = Path("sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d")
ROOT = Path(__file__).resolve().parents[1]


def read_rows(root, ids):
    records = {}
    for path in sorted(root.glob("*/episodes.jsonl")):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                eid = row["episode_id"]
                if eid in records or row.get("agent_error"):
                    raise ValueError("duplicate ID or agent error: %s" % eid)
                records[eid] = row
    if set(records) != set(ids):
        raise ValueError("incomplete or wrong cohort: %s; missing %s; extra %s" %
                         (root, sorted(set(ids) - set(records)), sorted(set(records) - set(ids))))
    return records


def model_digest(env):
    with urlopen(env["LLM_BASE_URL"].rstrip("/") + "/api/tags", timeout=8) as response:
        matches = [m["digest"] for m in json.load(response)["models"] if m["name"] == env["LLM_MODEL"]]
    if len(matches) != 1:
        raise ValueError("local Ollama endpoint does not advertise " + env["LLM_MODEL"])
    return matches[0]


def summary(a, b, ids):
    keep = [k for k in ids if a[k]["success"] and b[k]["success"]]
    return {"n": len(ids), "SR_A": mean(a[k]["success"] for k in ids), "SR_B": mean(b[k]["success"] for k in ids),
            "SPL_A": mean(a[k]["spl"] for k in ids), "SPL_B": mean(b[k]["spl"] for k in ids),
            "paired_dSR_pp": 100 * mean(int(b[k]["success"]) - int(a[k]["success"]) for k in ids),
            "paired_dSPL_pp": 100 * mean(b[k]["spl"] - a[k]["spl"] for k in ids),
            "gained": [k for k in ids if not a[k]["success"] and b[k]["success"]],
            "lost": [k for k in ids if a[k]["success"] and not b[k]["success"]],
            "both_succeeded": len(keep),
            "median_paired_path_delta_m_on_both_success": median(
                b[k]["path_length_m"] - a[k]["path_length_m"] for k in keep) if keep else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    exp = ROOT / MP3D / "sets/ranked_ab"
    manifest = json.loads((exp / "cohorts.json").read_text())
    ids = json.loads((exp / "episodes.json").read_text())
    gs = manifest["groups"]
    if sorted(gs["primary"] + gs["supplement"] + gs["guard"]) != sorted(ids) or len(set(ids)) != 160:
        raise ValueError("wrong episode manifest")
    out = (args.output or ROOT / "runs" / ("ab_rank%d" % manifest["rank"])).resolve()
    env = dict(os.environ)
    env.update(OBJNAV_REPO=str(ROOT), SEED="17", LLM_SEED="17", LLM_TEMPERATURE="0",
               LLM_BACKEND="ollama", OLLAMA_MODE="external", SPLIT="val", VALIDATE_STARTS="1",
               LLM_BASE_URL=env.get("LLM_BASE_URL", "http://127.0.0.1:11450"),
               LLM_MODEL=env.get("LLM_MODEL", "qwen2.5vl:7b"), PYTHONPATH=str(ROOT))
    for key in ("LLM_ANSWERS_REPLAY", "LLM_ANSWERS_RECORD"):
        env.pop(key, None)
    print("rank %d | %s | %d primary + %d supplemental + %d guard | %s" %
          (manifest["rank"], env["LLM_MODEL"], len(gs["primary"]), len(gs["supplement"]), len(gs["guard"]), out))
    if not args.run:
        return
    digest = model_digest(env)
    # An externally served cold model may live in a Docker volume invisible to the repo's model planner.
    # Budget before launching any renderer; never tell the planner an unloaded model costs zero.
    if not env.get("LLM_VRAM_MIB"):
        with urlopen(env["LLM_BASE_URL"].rstrip("/") + "/api/tags", timeout=8) as response:
            weight = next(m["size"] for m in json.load(response)["models"] if m["digest"] == digest)
        env["LLM_VRAM_MIB"] = str((weight + 1048575) // 1048576 + 2048)
    env.update(GPU_PLAN_LLM="gpu", GPU_PLAN_HABITAT="gpu", GPU_PLAN_YOLO="cpu")
    out.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (out / "identity.json").write_text(json.dumps({"commit": head, "model": env["LLM_MODEL"], "digest": digest,
        "seed": 17, "rank": manifest["rank"]}, indent=2) + "\n")
    arms = []
    for arm in ("A", "B"):
        if model_digest(env) != digest:
            raise RuntimeError("Ollama model changed between arms")
        subprocess.run(["bash", str(ROOT / MP3D / "run_benchmark.sh"), "--agent", "rpt", "--perception",
                        "ground-truth", "--jobs", "1", "--full-records", "--episode-ids-file",
                        str(exp / "episodes.json"), "--policy-config", str(exp / (arm + ".json")),
                        "--output", str(out / arm)], cwd=ROOT, env=env, check=True)
        arms.append(read_rows(out / arm, ids))
    if model_digest(env) != digest:
        raise RuntimeError("Ollama model changed during treatment")
    report = {g: summary(*arms, keys) for g, keys in gs.items() if keys}
    report["target_all"] = summary(*arms, gs["primary"] + gs["supplement"])
    text = json.dumps(report, indent=2) + "\n"
    (out / "paired.json").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
