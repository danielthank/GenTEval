import argparse
import json
import os
import pathlib
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils import run_dirs  # noqa: E402

from reports import (  # noqa: E402
    DurationReport,
    OperationReport,
    RCAReport,
    SizeReport,
    SpanCountReport,
)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate report")
    argparser.add_argument(
        "--app",
        type=str,
        default=None,
        help="Application to run",
    )
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

    def run_dirs_func():
        return run_dirs(args.app)

    if "duration" in args.evaluators:
        report_generator = DurationReport(args.compressors, root_dir)
        report = report_generator.generate(run_dirs_func)
        print("duration")
        print(json.dumps(report, indent=4))

    if "operation" in args.evaluators:
        report_generator = OperationReport(args.compressors, root_dir)
        report = report_generator.generate(run_dirs_func)
        print("operation")
        print(json.dumps(report, indent=4))

    if "trace_rca" in args.evaluators:
        report_generator = RCAReport(
            args.compressors, root_dir, "trace_rca_results.json"
        )
        report = report_generator.generate(run_dirs_func)
        print("trace_rca")
        print(json.dumps(report, indent=4))

    if "micro_rank" in args.evaluators:
        report_generator = RCAReport(
            args.compressors, root_dir, "micro_rank_results.json"
        )
        report = report_generator.generate(run_dirs_func)
        print("micro_rank")
        print(json.dumps(report, indent=4))

    if "size" in args.evaluators:
        report_generator = SizeReport(args.compressors, root_dir)
        report = report_generator.generate(run_dirs_func)
        print("size")
        print(json.dumps(report, indent=4))

    if "span_count" in args.evaluators:
        report_generator = SpanCountReport(args.compressors, root_dir)
        report = report_generator.generate(run_dirs_func)
        print("span_count")
        print(json.dumps(report, indent=4))
