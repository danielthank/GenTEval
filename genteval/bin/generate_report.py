import argparse
import json
import pathlib

from genteval.reports import (
    DurationReport,
    GraphReport,
    OperationReport,
    RateOverTimeReport,
    RCAReport,
    SizeReport,
    SpanCountReport,
    TimeReport,
)

from .utils import run_dirs


def main():
    argparser = argparse.ArgumentParser(
        description="Generate evaluation reports for GenTEval compression methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    argparser.add_argument(
        "--apps",
        type=str,
        nargs="*",
        help="Application to run (e.g., RE2-OB, RE2-TT)",
    )
    argparser.add_argument(
        "--services",
        type=str,
        nargs="*",
        help="Specific services to include (e.g., checkoutservice, ts-auth-service)",
    )
    argparser.add_argument(
        "--faults",
        type=str,
        nargs="*",
        help="Specific fault types to include (e.g., cpu, delay, disk, loss, mem, socket)",
    )
    argparser.add_argument(
        "--runs",
        type=int,
        nargs="*",
        help="Specific run numbers to include (default: [1, 2, 3])",
    )
    argparser.add_argument(
        "--root_dir",
        type=str,
        default="./output",
        help="Directory containing the output (default: ./output)",
    )
    argparser.add_argument(
        "--compressors",
        type=str,
        nargs="+",
    )
    argparser.add_argument(
        "--evaluators",
        type=str,
        nargs="*",
        default=[
            "duration",
            "rate_over_time",
            "operation",
            "trace_rca",
            "micro_rank",
            "size",
            "span_count",
            "time",
            "graph",
        ],
        help="Evaluators to run (default: all evaluators)",
    )
    argparser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Enable plotting for duration evaluator (default: True)",
    )
    argparser.add_argument(
        "--no_plot",
        dest="plot",
        action="store_false",
        help="Disable plotting for duration evaluator",
    )
    argparser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON report",
    )
    args = argparser.parse_args()

    root_dir = pathlib.Path(args.root_dir)

    def run_dirs_func():
        return run_dirs(
            applications=args.apps,
            services=args.services,
            faults=args.faults,
            runs=args.runs,
        )

    # Collect all reports
    all_reports = {}

    if "duration" in args.evaluators:
        report_generator = DurationReport(args.compressors, root_dir, plot=args.plot)
        report = report_generator.generate(run_dirs_func)
        all_reports["duration"] = report

    if "rate_over_time" in args.evaluators:
        report_generator = RateOverTimeReport(
            args.compressors, root_dir, plot=args.plot
        )
        report = report_generator.generate(run_dirs_func)
        all_reports["rate_over_time"] = report

    if "operation" in args.evaluators:
        report_generator = OperationReport(args.compressors, root_dir)
        report = report_generator.generate(run_dirs_func)
        all_reports["operation"] = report

    if "trace_rca" in args.evaluators:
        report_generator = RCAReport(
            args.compressors, root_dir, "trace_rca_results.json"
        )
        report = report_generator.generate(run_dirs_func)
        all_reports["trace_rca"] = report

    if "micro_rank" in args.evaluators:
        report_generator = RCAReport(
            args.compressors, root_dir, "micro_rank_results.json"
        )
        report = report_generator.generate(run_dirs_func)
        all_reports["micro_rank"] = report

    if "size" in args.evaluators:
        report_generator = SizeReport(args.compressors, root_dir)
        report = report_generator.generate(run_dirs_func)
        all_reports["size"] = report

    if "span_count" in args.evaluators:
        report_generator = SpanCountReport(args.compressors, root_dir)
        report = report_generator.generate(run_dirs_func)
        all_reports["span_count"] = report

    if "time" in args.evaluators:
        report_generator = TimeReport(args.compressors, root_dir)
        report = report_generator.generate(run_dirs_func)
        all_reports["time"] = report

    if "graph" in args.evaluators:
        report_generator = GraphReport(args.compressors, root_dir)
        report = report_generator.generate(run_dirs_func)
        all_reports["graph"] = report

    # Save JSON report if output file is specified
    if args.output and all_reports:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metadata": {
                "generator": "GenTEval Report Generator",
                "report_types": list(all_reports.keys()),
            },
            "reports": all_reports,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"Report saved to: {output_path}")
