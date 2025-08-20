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
from pathlib import Path
from typing import Any

from tqdm import tqdm

from genteval.data_collection.utils.duration_adjuster import DurationAdjuster


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


def extract_spans_for_csv(trace_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all spans from a trace and format for CSV export."""
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

            # Use flags as statusCode, fallback to empty string
            status_code = span.get("flags", "")

            # Extract span data matching example.csv format
            span_data = {
                "time": start_datetime.strftime("%H:%M"),
                "traceID": trace_id,
                "spanID": span.get("spanID", ""),
                "serviceName": span.get("process", {}).get("serviceName", ""),
                "methodName": "",  # Will extract from tags if available
                "operationName": span.get("operationName", ""),
                "startTimeMillis": start_time_micros // 1000,  # Convert to milliseconds
                "startTime": start_time_micros,
                "duration": span.get("duration", 0),
                "statusCode": status_code,
                "parentSpanID": parent_span_id,
            }

            # Extract methodName from tags if available
            if "tags" in span:
                for tag in span["tags"]:
                    if (
                        tag.get("key") == "http.method"
                        or tag.get("key") == "grpc.method"
                    ):
                        span_data["methodName"] = tag.get("value", "")
                        break

            spans.append(span_data)

    return spans


def process_traces_to_csv(
    trace_dir: str,
    output_file: str,
    apply_duration_adjustment: bool = False,
    verbose: bool = False,
):
    """Process all trace files and export February 2025 spans to CSV."""
    trace_path = Path(trace_dir)

    if verbose:
        print(f"Processing traces in: {trace_path}")

    # Get all JSON files
    json_files = list(trace_path.glob("*.json"))
    total_files = len(json_files)
    if verbose:
        print(f"Found {total_files} trace files")

    all_spans = []
    processed_traces = 0

    for file_path in tqdm(json_files, desc="Processing trace files", unit="file"):
        trace_data = load_trace_file(file_path)
        spans = extract_spans_for_csv(trace_data)

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
            "statusCode",
            "parentSpanID",
        ]

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for span in tqdm(all_spans, desc="Writing CSV rows", unit="row"):
                writer.writerow(span)

        if verbose:
            print(f"CSV export completed: {output_file}")
            print(f"Total rows: {len(all_spans)}")

        # Apply duration adjustment if requested
        if apply_duration_adjustment:
            if verbose:
                print("Applying duration adjustments...")
            adjuster = DurationAdjuster(verbose=verbose)
            adjuster.load_csv(output_file)
            adjuster.build_relationships()
            adjusted_count = adjuster.adjust_durations()

            if adjusted_count > 0:
                # Overwrite the original file with adjusted data
                adjuster.write_csv(output_file)
                if verbose:
                    print(f"Duration-adjusted CSV saved: {output_file}")
                    adjuster.print_adjustment_summary()
            elif verbose:
                print("No duration adjustments were needed.")
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
    )

    return 0


if __name__ == "__main__":
    exit(main())
