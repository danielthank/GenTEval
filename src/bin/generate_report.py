import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict

from logger import setup_logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
setup_logging()

from utils import run_dirs  # noqa: E402


def ac_at_k(answer, ranks, k):
    return answer in ranks[:k] if k <= len(ranks) else False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate report")
    argparser.add_argument(
        "--root_dir",
        type=str,
        help="Directory containing the output",
    )
    argparser.add_argument(
        "--compressors",
        type=str,
        nargs="+",
    )
    args = argparser.parse_args()

    root_dir = pathlib.Path(args.root_dir)

    report = defaultdict(lambda: defaultdict(list))
    services = set()

    for app_name, service, fault, run in run_dirs():
        for compressor in args.compressors:
            if compressor not in ["original", "gent"] and not compressor.startswith(
                "head_sampling"
            ):
                continue

            results_path = root_dir.joinpath(
                app_name,
                f"{service}_{fault}",
                str(run),
                compressor,
                "evaluated",
                "results.json",
            )
            if not results_path.exists():
                print(f"Results file {results_path} does not exist, skipping.")
                continue

            results = json.loads(results_path.read_text())
            ranks = results.get("ranks", [])
            for rank in ranks:
                services.add(rank)
            for k in range(1, 6):
                report[f"{compressor}"][f"ac{k}"].append(ac_at_k(service, ranks, k))
    for fault in report:
        for k in range(1, 6):
            report[fault][f"ac{k}"] = sum(report[fault][f"ac{k}"]) / len(
                report[fault][f"ac{k}"]
            )
        report[fault]["avg5"] = sum(report[fault][f"ac{k}"] for k in range(1, 6)) / 5

    for k in range(1, 6):
        report["random"][f"ac{k}"] = k / 7
    report["random"]["avg5"] = sum(report["random"][f"ac{k}"] for k in range(1, 6)) / 5
    print(json.dumps(report, indent=4))
