import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict

from logger import setup_logging
from scipy.stats import wasserstein_distance

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
setup_logging()

from utils import run_dirs  # noqa: E402


def generate_report_trace_rca(compressors, root_dir):
    def ac_at_k(answer, ranks, k):
        return answer in ranks[:k] if k <= len(ranks) else False

    report = defaultdict(lambda: defaultdict(list))
    services = set()

    for app_name, service, fault, run in run_dirs():
        for compressor in compressors:
            if compressor not in ["original", "gent"] and not compressor.startswith(
                "head_sampling"
            ):
                continue

            report_group = f"{app_name}_{compressor}"

            results_path = root_dir.joinpath(
                app_name,
                f"{service}_{fault}",
                str(run),
                compressor,
                "evaluated",
                "trace_rca_results.json",
            )
            if not results_path.exists():
                print(f"Results file {results_path} does not exist, skipping.")
                continue

            results = json.loads(results_path.read_text())
            ranks = results.get("ranks", [])
            for rank in ranks:
                services.add(rank)
            for k in range(1, 6):
                report[report_group][f"ac{k}"].append(ac_at_k(service, ranks, k))

            compressed_dir = root_dir.joinpath(
                app_name,
                f"{service}_{fault}",
                str(run),
                compressor,
                "compressed",
                "data",
            )
            total_size = 0
            for file in compressed_dir.glob("**/*"):
                if file.is_file():
                    total_size += file.stat().st_size
            report[report_group]["size"].append(total_size)

    for fault in report:
        for k in range(1, 6):
            report[fault][f"ac{k}"] = sum(report[fault][f"ac{k}"]) / len(
                report[fault][f"ac{k}"]
            )
        report[fault]["avg5"] = sum(report[fault][f"ac{k}"] for k in range(1, 6)) / 5
        report[fault]["size"] = sum(report[fault]["size"]) / len(report[fault]["size"])
    return report


def generate_report_duration(compressors, root_dir):
    report = defaultdict(lambda: defaultdict(list))

    for app_name, service, fault, run in run_dirs():
        for compressor in compressors:
            if compressor not in ["original", "gent"] and not compressor.startswith(
                "head_sampling"
            ):
                print(
                    f"Compressor {compressor} is not supported, skipping for {app_name}_{service}_{fault}_{run}."
                )
                continue

            if compressor == "original" or compressor == "head_sampling_1":
                print(
                    f"Compressor {compressor} is not supported for duration evaluation, skipping for {app_name}_{service}_{fault}_{run}."
                )
                continue

            original_results_path = root_dir.joinpath(
                app_name,
                f"{service}_{fault}",
                str(run),
                "head_sampling_1",
                "evaluated",
                "duration_results.json",
            )
            if not original_results_path.exists():
                print(
                    f"Original results file {original_results_path} does not exist, skipping."
                )
                continue

            results_path = root_dir.joinpath(
                app_name,
                f"{service}_{fault}",
                str(run),
                compressor,
                "evaluated",
                "duration_results.json",
            )
            if not results_path.exists():
                print(f"Results file {results_path} does not exist, skipping.")
                continue

            original_distribution = json.loads(original_results_path.read_text())[
                "duration_distribution"
            ]
            results_distribution = json.loads(results_path.read_text())[
                "duration_distribution"
            ]

            # calculate the Wasserstein distance
            for group in original_distribution:
                if group not in results_distribution:
                    continue
                wdis = wasserstein_distance(
                    original_distribution[group], results_distribution[group]
                )
                report_group = f"{app_name}_{compressor}"
                report[report_group]["wdis"].append(wdis)
        for group in report:
            report[group]["wdis_avg"] = sum(report[group]["wdis"]) / len(
                report[group]["wdis"]
            )
            del report[group]["wdis"]

    return report


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
    argparser.add_argument(
        "--evaluators",
        type=str,
        nargs="+",
    )
    args = argparser.parse_args()

    root_dir = pathlib.Path(args.root_dir)

    if "trace_rca" in args.evaluators:
        report = generate_report_trace_rca(args.compressors, root_dir)
        print("trace_rca")
        print(json.dumps(report, indent=4))
    if "duration" in args.evaluators:
        report = generate_report_duration(args.compressors, root_dir)
        print("duration")
        print(json.dumps(report, indent=4))
