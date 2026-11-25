"""Paired comparison of two view-selection methods, possibly from different output folders, on the regions both scored."""
import argparse
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import first_k_accuracy, paired_difference
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_mp3d_rooms import load_truth
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_mp3d_selection import KS, VLM_FILE, answers_by_method


def parse(spec):
    folder, method = spec.rsplit(":", 1)
    return Path(folder), method


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--a", required=True, help="folder:method")
    p.add_argument("--b", required=True, help="folder:method")
    p.add_argument("--file-a", default=VLM_FILE, help="answer file of A")
    p.add_argument("--file-b", default=VLM_FILE, help="answer file of B")
    args = p.parse_args(argv)
    truth, rng = load_truth(args.rooms), np.random.default_rng(0)
    (folder_a, method_a), (folder_b, method_b) = parse(args.a), parse(args.b)
    a = answers_by_method(folder_a, args.file_a, {method_a})[method_a]
    b = answers_by_method(folder_b, args.file_b, {method_b})[method_b]
    shared = {r for r in a if r in b}
    print("%d regions scored by both" % len(shared))
    for k in KS:
        right_a = first_k_accuracy({r: a[r] for r in shared}, truth, k)
        right_b = first_k_accuracy({r: b[r] for r in shared}, truth, k)
        mean, low, high = paired_difference(right_a, right_b, rng)
        print("K=%d  %s %.1f%%  %s %.1f%%  gain %+.1f [%+.1f, %+.1f]" % (k, method_a, 100 * np.mean(list(right_a.values())), method_b,
                                                                       100 * np.mean(list(right_b.values())), 100 * mean, 100 * low, 100 * high))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
