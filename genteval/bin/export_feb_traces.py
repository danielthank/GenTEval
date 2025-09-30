#!/usr/bin/env python3
"""
Export February 2025 Traces to CSV

This script processes trace files and exports all spans from traces
starting in February 2025 to a CSV format matching example.csv structure.
Optionally applies duration adjustment to ensure parent spans encompass child spans.
"""

import argparse
import csv
import json
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from tqdm import tqdm

from genteval.data_collection.utils.duration_adjuster import DurationAdjuster, SpanData


def load_trace_file(file_path: str) -> dict[str, Any]:
    """Load and parse a single trace JSON file."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def convert_microseconds_to_datetime(microseconds: int) -> datetime:
    """Convert microsecond timestamp to datetime."""
    return datetime.fromtimestamp(microseconds / 1_000_000)


def extract_spans_for_csv(trace_data: dict[str, Any]) -> list[SpanData]:
    """Extract all spans from a trace and return as SpanData objects."""
    spans = []

    if "data" not in trace_data:
        return spans

    for trace_entry in trace_data["data"]:
        if "spans" not in trace_entry:
            continue

        trace_id = trace_entry.get("traceID", "")

        for span in trace_entry["spans"]:
            if "startTime" not in span:
                continue

            start_time_micros = span["startTime"]
            start_datetime = convert_microseconds_to_datetime(start_time_micros)

            # Check if this span starts in February 2025
            if start_datetime.year != 2025 or start_datetime.month != 2:
                continue

            # Extract parent span ID from CHILD_OF references
            parent_span_id = ""
            if "references" in span:
                for ref in span["references"]:
                    if ref.get("refType") == "CHILD_OF":
                        parent_span_id = ref.get("spanID", "")
                        break  # Use first CHILD_OF reference

            flags_value = span.get("flags", 0)

            method_name = ""
            has_error = False
            if "tags" in span:
                for tag in span["tags"]:
                    if (
                        tag.get("key") == "http.method"
                        or tag.get("key") == "grpc.method"
                    ):
                        method_name = tag.get("value", "")
                    elif tag.get("key") == "error" and tag.get("value") is True:
                        has_error = True

            row = {
                "time": start_datetime.strftime("%H:%M"),
                "traceID": trace_id,
                "spanID": span.get("spanID", ""),
                "serviceName": span.get("process", {}).get("serviceName", ""),
                "methodName": method_name,
                "operationName": span.get("operationName", ""),
                "startTimeMillis": str(
                    start_time_micros // 1000
                ),  # Convert to milliseconds
                "startTime": str(start_time_micros),
                "duration": str(span.get("duration", 0)),
                "flags": str(flags_value),
                "error": "true" if has_error else "false",
                "parentSpanID": parent_span_id,
            }

            span_data = SpanData(row)
            spans.append(span_data)

    return spans


def process_single_file(file_path: Path) -> list[SpanData]:
    """Worker function to process a single JSON trace file.

    This function is designed to be used with multiprocessing.Pool.
    It loads a trace file and extracts all February 2025 spans.

    Args:
        file_path: Path to the JSON trace file

    Returns:
        List of SpanData objects extracted from the trace
    """
    trace_data = load_trace_file(str(file_path))
    return extract_spans_for_csv(trace_data)


def process_traces_to_csv(
    trace_dir: str,
    output_file: str,
    apply_duration_adjustment: bool = False,
    verbose: bool = False,
    num_workers: int | None = None,
):
    """Process all trace files and export February 2025 spans to CSV.

    Args:
        trace_dir: Directory containing trace JSON files
        output_file: Output CSV file path
        apply_duration_adjustment: Whether to apply duration adjustment
        verbose: Whether to show detailed processing information
        num_workers: Number of worker processes (default: cpu_count - 1)
    """
    trace_path = Path(trace_dir)

    if verbose:
        print(f"Processing traces in: {trace_path}")

    # Get all JSON files
    json_files = list(trace_path.glob("*.json"))
    total_files = len(json_files)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    if verbose:
        print(f"Found {total_files} trace files")
        print(f"Using {num_workers} worker processes")

    all_spans = []
    processed_traces = 0

    with Pool(processes=num_workers) as pool:
        chunksize = max(1, total_files // (num_workers * 4))

        for spans in tqdm(
            pool.imap_unordered(process_single_file, json_files, chunksize=chunksize),
            total=total_files,
            desc="Processing trace files",
            unit="file",
        ):
            if spans:
                all_spans.extend(spans)
                processed_traces += 1

    if verbose:
        print(
            f"Extracted {len(all_spans)} spans from {processed_traces} traces with February 2025 data"
        )

    # Write to CSV
    if all_spans:
        fieldnames = [
            "time",
            "traceID",
            "spanID",
            "serviceName",
            "methodName",
            "operationName",
            "startTimeMillis",
            "startTime",
            "duration",
            "flags",
            "error",
            "parentSpanID",
        ]

        if apply_duration_adjustment:
            if verbose:
                print("Applying duration adjustments...")

            span_dict = {span.span_id: span for span in all_spans}

            for span in tqdm(
                all_spans, desc="Building relationships", disable=not verbose
            ):
                if span.parent_span_id and span.parent_span_id in span_dict:
                    parent = span_dict[span.parent_span_id]
                    parent.children.append(span.span_id)

            adjuster = DurationAdjuster(verbose=verbose)
            adjuster.spans = span_dict
            adjusted_count = adjuster.adjust_durations()

            if verbose:
                if adjusted_count > 0:
                    print(f"Adjusted {adjusted_count} spans")
                else:
                    print("No duration adjustments were needed.")

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([span.to_csv_row() for span in all_spans])

        if verbose:
            print(f"CSV export completed: {output_file}")
            print(f"Total rows: {len(all_spans)}")
    elif verbose:
        print("No spans found for February 2025")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description="Export February 2025 traces to CSV with duration adjustment"
    )

    parser.add_argument(
        "--trace-dir",
        "-i",
        default="data/uber-trace1",
        help="Directory containing trace JSON files (default: data/uber-trace1)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="data/uber-trace1-transformed/202502/1/traces.csv",
        help="Output CSV file path (default: data/uber-trace1-transformed/202502/1/traces.csv)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed processing information",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of worker processes for parallel processing (default: cpu_count - 1)",
    )

    args = parser.parse_args()

    # Ensure trace directory exists
    if not os.path.exists(args.trace_dir):
        print(f"Error: Directory {args.trace_dir} does not exist")
        return 1

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("Starting trace export...")

    process_traces_to_csv(
        args.trace_dir,
        args.output,
        apply_duration_adjustment=True,
        verbose=args.verbose,
        num_workers=args.workers,
    )

    return 0


if __name__ == "__main__":
    exit(main())
