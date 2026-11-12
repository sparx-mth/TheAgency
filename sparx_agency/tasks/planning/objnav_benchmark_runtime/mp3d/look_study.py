"""The look studies' plan and dry run: what an arm would run, on what, with what -- and nothing started.

``look_study.sh <study> <arm> --dry-run`` calls :func:`main` with the arm's
resolved environment. It prints the plan -- the study, the arm's policy
configuration, the episode subset by scene, the LLM model and server, the
answer stores recorded and replayed, the renderer, the workers, a time
estimate -- and checks that the machine and the data fit it:

* the habitat-sim interpreter imports ``habitat_sim``;
* the episode split and every scene the subset needs are installed (mesh,
  navmesh and, under ground-truth perception, the ``.house`` / ``_semantic.ply``
  annotations and the publisher's semantic scene-dataset config);
* an Ollama answers at the LLM address with the model, or the ``ollama``
  binary is there for the launcher to start one and pull it;
* the GPU holds the LLM and the renderers the arm asks for, or the arm is
  told to render on the CPU or run fewer workers;
* a variant arm finds the ``without`` arm's answer store to replay;
* the disk has room for the recordings.

Every problem is one line that says what is missing and what to do; the exit
code is 1 when the arm cannot run as planned. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

#: The arms of each study, in the order they are worth running: the baseline first, then the variants by priority.
#: (``without`` is one arm shared by both studies: the method as it stands, the ten-turn spin and the frontier walk.)
STUDIES: Dict[str, List[str]] = {
    "warmup": ["without", "p1_b25", "p3_b25", "p1_b15", "p3_b40", "p0_b15"],
    "explore": ["without", "explore_viewpoint", "explore_viewpoint_transparent"],
}
#: Each arm's policy configuration file under ``mp3d/sets/`` (None: the method's defaults).
ARM_CONFIGS: Dict[str, Optional[str]] = {
    "without": None,
    "p0_b15": "policy_warmup_p0_b15_gt.json", "p1_b15": "policy_warmup_p1_b15_gt.json",
    "p1_b25": "policy_warmup_p1_b25_gt.json", "p3_b25": "policy_warmup_p3_b25_gt.json",
    "p3_b40": "policy_warmup_p3_b40_gt.json",
    "explore_viewpoint": "policy_explore_viewpoint_gt.json",
    "explore_viewpoint_transparent": "policy_explore_viewpoint_transparent_gt.json",
}
ARM_WORDS: Dict[str, str] = {
    "without": "the method as it stands: ten turns in place, then the frontier walk",
    "p0_b15": "vantage warm-up, no walk: a greedy sweep in place (ends when nothing is worth a turn), budget 15",
    "p1_b15": "vantage warm-up, one vantage point at most, budget 15 actions",
    "p1_b25": "vantage warm-up, one vantage point at most, budget 25 actions",
    "p3_b25": "vantage warm-up, up to three vantage points, budget 25 actions",
    "p3_b40": "vantage warm-up, up to three vantage points, budget 40 actions",
    "explore_viewpoint": "explore visit by viewpoints (greedy look planner in the room; survival prior 2 m)",
    "explore_viewpoint_transparent": "explore visit by viewpoints with MapEx's transparent unknown (no survival prior)",
}
GROUND_TRUTH_FILES = ("{scene}.house", "{scene}_semantic.ply")
SEMANTIC_CONFIG = "mp3d.scene_dataset_config.json"
#: VRAM the plan budgets: a 3B model at an 8k window, one habitat-sim renderer.
LLM_MIB_3B, LLM_MIB_14B, RENDERER_MIB = 3200, 15200, 1100


def say(lines, text):
    lines.append(text)


def replay_dirs(store_root: Path, arm: str):
    """The answer stores a variant arm replays: `without` plus every other arm's, as many as agree with each other.

    A store that holds two answers to one question replays neither
    (``llm_answer_cache.LLMAnswerStore`` raises), and two arms that ran at the
    same time -- neither replaying the other -- can disagree on a key the live
    server answered twice. The stores are added one at a time, `without`
    first, then the others in name order, and a store that conflicts with
    those already taken is left out and named. Returns ``(taken, dropped)``.
    """
    from sparx_agency.core.mapping.topology.llm_answer_cache import LLMAnswerStore
    candidates = [d for d in sorted(store_root.glob("*")) if d.is_dir() and d.name != arm and any(d.glob("answers-*.jsonl"))]
    candidates.sort(key=lambda d: (d.name != "without", d.name))
    taken, dropped = [], []
    for directory in candidates:
        try:
            LLMAnswerStore(replay_dirs=taken + [directory])
        except ValueError as exc:
            dropped.append((directory, str(exc)))
            continue
        taken.append(directory)
    return taken, dropped


def check_interpreter(hab_py):
    if not hab_py or not Path(hab_py).is_file():
        return None, "HAB_PY=%r is not a file: point HAB_PY at the habitat-sim 0.2.4 interpreter (~/miniconda3/envs/habitat/bin/python)" % hab_py
    try:
        out = subprocess.run([hab_py, "-c", "import habitat_sim, numpy, scipy; print(habitat_sim.__version__)"],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, "HAB_PY could not be run: %s" % exc
    if out.returncode != 0:
        return None, "HAB_PY cannot import habitat_sim: %s" % out.stderr.strip().splitlines()[-1:]
    return out.stdout.strip(), None


def check_data(episodes_dir, scenes_dir, scenes, ground_truth):
    problems, notes = [], []
    split = Path(episodes_dir).expanduser()
    content = split / "content"
    if not (split / "val.json.gz").is_file() and not content.is_dir():
        notes.append("episodes: %s holds no val.json.gz / content/ -- the launcher fetches the public archive (173 MB) itself" % split)
    else:
        missing = [s for s in scenes if not (content / ("%s.json.gz" % s)).is_file()]
        if missing and not (split / "val.json.gz").is_file():
            problems.append("episodes: %s lacks content/<scene>.json.gz for %s" % (split, ", ".join(missing)))
    release = Path(scenes_dir).expanduser() / "mp3d"
    if not release.is_dir():
        problems.append("scenes: %s does not exist. Unpack mp3d_habitat.zip (licensed; Matterport's terms) so that it holds "
                        "mp3d/<scene>/<scene>.glb, or set MP3D_SCENES_DIR to where it is" % release)
        return problems, notes
    for scene in scenes:
        needed = ["%s.glb" % scene, "%s.navmesh" % scene] + ([f.format(scene=scene) for f in GROUND_TRUTH_FILES] if ground_truth else [])
        missing = [name for name in needed if not (release / scene / name).is_file()]
        if missing:
            problems.append("scene %s: missing %s under %s%s" % (
                scene, ", ".join(missing), release / scene,
                " (the .house and _semantic.ply ship in mp3d_habitat.zip; ground-truth perception needs them)"
                if any(m.endswith((".house", ".ply")) for m in missing) else ""))
    if ground_truth and not (release / SEMANTIC_CONFIG).is_file():
        notes.append("scenes: %s is absent -- the launcher fetches it (809 bytes, public) before a ground-truth run" % (release / SEMANTIC_CONFIG))
    return problems, notes


def check_llm(base_url, models):
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=5) as fh:
            tags = json.load(fh)
        present = {m.get("name") for m in tags.get("models", [])}
        missing = [m for m in models if m not in present and m + ":latest" not in present]
        if missing:
            return ("an Ollama answers at %s but lacks %s: `ollama pull %s` (the launcher pulls it when OLLAMA_MODE=native)"
                    % (base_url, ", ".join(missing), missing[0])), True
        return "an Ollama answers at %s with %s" % (base_url, ", ".join(sorted(set(models)))), True
    except Exception:
        if shutil.which("ollama") is None:
            return "no Ollama answers at %s and no `ollama` binary is on PATH: install Ollama or start a server there" % base_url, False
        return ("no Ollama answers at %s: the launcher starts one (OLLAMA_MODE=native) and pulls %s if needed"
                % (base_url, ", ".join(sorted(set(models))))), True


def gpu_state():
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        name, total, used = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "total_mib": int(total), "used_mib": int(used)}
    except Exception:
        return None


def estimate_hours(episodes, mean_steps, seconds_per_action, jobs, replay):
    """Wall hours for the arm: actions at the per-action time over the workers; a replaying arm asks the model less."""
    actions = episodes * mean_steps
    per_action = seconds_per_action * (0.85 if replay else 1.0)
    return actions * per_action / max(1, jobs) / 3600.0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["replay-dirs"]:
        # ``look_study.sh`` asks for the colon-separated replay list of an arm (the dropped stores go to stderr).
        sub_parser = argparse.ArgumentParser()
        sub_parser.add_argument("replay_dirs")
        sub_parser.add_argument("--store-root", required=True)
        sub_parser.add_argument("--arm", required=True)
        sub_args = sub_parser.parse_args(argv)
        taken, dropped = replay_dirs(Path(sub_args.store_root), sub_args.arm)
        for directory, why in dropped:
            print("WARNING: not replaying %s: %s" % (directory, why), file=sys.stderr)
        print(os.pathsep.join(str(d) for d in taken))
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True, choices=sorted(STUDIES))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--set", required=True, help="the episode subset (a JSON list of <scene>/<index>)")
    parser.add_argument("--sets-dir", required=True, help="mp3d/sets, where the arms' policy configs live")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--hab-py", default=os.environ.get("HAB_PY", ""))
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-reasoning-model", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--perception", default="ground-truth")
    parser.add_argument("--renderer", default="gpu")
    parser.add_argument("--seconds-per-action", type=float, default=float(os.environ.get("SECONDS_PER_ACTION", "1.2")))
    parser.add_argument("--all-arms", action="store_true", help="estimate every arm of the study, not only --arm")
    args = parser.parse_args(argv)
    lines, problems = [], []
    arms = STUDIES[args.study]
    if args.arm not in arms and args.arm != "report":
        print("arm %r is not one of the %s study's: %s" % (args.arm, args.study, ", ".join(arms)))
        return 1
    ids = json.loads(Path(args.set).read_text())
    by_scene: Dict[str, int] = {}
    for episode_id in ids:
        by_scene[episode_id.split("/")[0]] = by_scene.get(episode_id.split("/")[0], 0) + 1
    meta_path = Path(args.set).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    mean_steps = float(meta.get("baseline_mean_steps", 220.0))
    config = ARM_CONFIGS.get(args.arm)
    say(lines, "== %s study, arm %s: %s" % (args.study, args.arm, ARM_WORDS.get(args.arm, "")))
    say(lines, "   policy config: %s" % ("the method's defaults" if config is None else Path(args.sets_dir) / config))
    if config is not None:
        path = Path(args.sets_dir) / config
        if not path.is_file():
            problems.append("the arm's policy config %s is missing from the checkout" % path)
        else:
            say(lines, "   %s" % path.read_text().strip())
    say(lines, "   episodes: %d in %d scenes (%s)" % (len(ids), len(by_scene), ", ".join("%s x%d" % kv for kv in sorted(by_scene.items()))))
    say(lines, "   perception: %s%s" % (args.perception, "" if args.perception == "ground-truth" else
                                        "  (NOTE: the arms' configs set detection_range_m 5.0 for ground truth; a detector wants 3.5)"))
    say(lines, "   LLM: %s (labels) + %s (node oracle) at %s, temperature 0, seed 17"
        % (args.llm_model, args.llm_reasoning_model, args.llm_base_url))
    store_root = Path(args.runs_root) / "look_study" / "llm_answers"
    without_store = store_root / "without"
    if args.arm == "without":
        say(lines, "   LLM answers: RECORDED into %s (the store every other arm replays)" % without_store)
    elif args.arm != "report":
        taken, dropped = replay_dirs(store_root, args.arm) if store_root.is_dir() else ([], [])
        say(lines, "   LLM answers: REPLAYED from the stores of %s; new ones recorded into %s"
            % (", ".join(d.name for d in taken) if taken else "(none yet)", store_root / args.arm))
        for directory, why in dropped:
            say(lines, "   note: the store of %s is NOT replayed -- it disagrees with the stores taken before it (%s); "
                       "two arms ran at once without replaying each other" % (directory.name, why.split(" (")[0]))
        if not any(without_store.glob("answers-*.jsonl")):
            problems.append("the `without` arm's answer store %s is empty: run `without` first, so the variants replay its answers"
                            % without_store)
    say(lines, "   output: %s" % (Path(args.runs_root) / "look_study" / args.arm))
    say(lines, "   renderer: %s, %d worker(s) (one scene process each)" % (args.renderer, args.jobs))
    # -- the machine --------------------------------------------------------------------------------------------
    version, problem = check_interpreter(args.hab_py)
    if problem:
        problems.append(problem)
    else:
        say(lines, "   habitat-sim %s at %s" % (version, args.hab_py))
    data_problems, notes = check_data(args.episodes_dir, args.scenes_dir, sorted(by_scene), args.perception == "ground-truth")
    problems += data_problems
    for note in notes:
        say(lines, "   note: " + note)
    if not data_problems:
        say(lines, "   data: the %d scenes of the subset are installed under %s" % (len(by_scene), Path(args.scenes_dir).expanduser() / "mp3d"))
    llm_note, llm_ok = check_llm(args.llm_base_url, [args.llm_model, args.llm_reasoning_model])
    say(lines, "   " + llm_note)
    if not llm_ok:
        problems.append(llm_note)
    gpu = gpu_state()
    if gpu is None:
        say(lines, "   GPU: none seen by nvidia-smi -- the LLM runs on the CPU (slow) and habitat needs HABITAT_RENDER=software")
        if args.renderer == "gpu":
            problems.append("no NVIDIA GPU: set HABITAT_RENDER=software (Mesa llvmpipe) for the renderer")
    else:
        llm_mib = LLM_MIB_14B if "14b" in (args.llm_model + args.llm_reasoning_model).lower() else LLM_MIB_3B
        renderers = RENDERER_MIB * args.jobs if args.renderer == "gpu" else 0
        free = gpu["total_mib"] - gpu["used_mib"]
        resident = llm_ok and "answers at" in llm_note and "lacks" not in llm_note
        need = renderers + (0 if resident else llm_mib)
        say(lines, "   GPU: %s, %d of %d MiB used; this arm needs about %d MiB (%s%d renderer(s) x %d)" % (
            gpu["name"], gpu["used_mib"], gpu["total_mib"], need,
            "" if resident else "LLM %d + " % llm_mib, args.jobs if args.renderer == "gpu" else 0, RENDERER_MIB))
        if need > free:
            problems.append("the GPU has %d MiB free and the arm wants about %d: run fewer workers (--jobs), render on the CPU "
                            "(HABITAT_RENDER=software), or stop what holds the card" % (free, need))
    usage = shutil.disk_usage(str(Path(args.runs_root).expanduser().parent if not Path(args.runs_root).exists() else args.runs_root))
    free_gb = usage.free / 1e9
    say(lines, "   disk: %.1f GB free under %s (an arm's recordings take about 0.5-1 GB)" % (free_gb, args.runs_root))
    if free_gb < 2.0:
        problems.append("under 2 GB of disk free for the recordings")
    # -- the time -----------------------------------------------------------------------------------------------------
    say(lines, "   time (at %.2f s per action, %d worker(s), %.0f actions per episode on the baseline):" % (
        args.seconds_per_action, args.jobs, mean_steps))
    for arm in (arms if args.all_arms else [args.arm] if args.arm != "report" else []):
        hours = estimate_hours(len(ids), mean_steps, args.seconds_per_action, args.jobs, arm != "without")
        say(lines, "     %-32s ~%.1f h%s" % (arm, hours, "  (asks the model for every answer)" if arm == "without" else "  (replays the baseline's answers)"))
    say(lines, "   set SECONDS_PER_ACTION after the first few episodes of `without` (progress.json shows the rate) for a better estimate")
    print("\n".join(lines))
    if problems:
        print("\nThis arm cannot run as planned:")
        for problem in problems:
            print("  - " + problem)
        return 1
    print("\nDry run: nothing was started. Drop --dry-run to run the arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
