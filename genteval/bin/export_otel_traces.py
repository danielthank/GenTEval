#!/usr/bin/env python3
"""
Export OpenTelemetry Traces to CSV

This script processes OpenTelemetry trace files in JSONL format and exports
all spans to a CSV format for analysis.
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


def extract_attribute_value(attr_value: dict[str, Any]) -> Any:
    """Extract the actual value from an OpenTelemetry attribute value object."""
    if "stringValue" in attr_value:
        return attr_value["stringValue"]
    elif "intValue" in attr_value:
        return attr_value["intValue"]
    elif "doubleValue" in attr_value:
        return attr_value["doubleValue"]
    elif "boolValue" in attr_value:
        return attr_value["boolValue"]
    return str(attr_value)


def get_span_attribute(span: dict[str, Any], key: str) -> Any:
    """Get a specific attribute value from a span."""
    for attr in span.get("attributes", []):
        if attr["key"] == key:
            return extract_attribute_value(attr["value"])
    return None


def get_resource_attribute(resource: dict[str, Any], key: str) -> Any:
    """Get a specific attribute value from a resource."""
    for attr in resource.get("attributes", []):
        if attr["key"] == key:
            return extract_attribute_value(attr["value"])
    return None


def extract_spans_for_csv(trace_data: dict[str, Any]) -> list[SpanData]:
    """Extract all spans from an OpenTelemetry trace and return as SpanData objects."""
    spans = []

    for resource_spans in trace_data.get("resourceSpans", []):
        # Extract service name from resource
        resource = resource_spans.get("resource", {})
        service_name = get_resource_attribute(resource, "service.name") or ""

        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                # Extract basic span fields
                trace_id = span.get("traceId", "")
                span_id = span.get("spanId", "")
                parent_span_id = span.get("parentSpanId", "")
                operation_name = span.get("name", "")

                # Extract timestamps
                start_time_nano = int(span.get("startTimeUnixNano", 0))
                end_time_nano = int(span.get("endTimeUnixNano", 0))

                # Convert timestamps
                start_time_micros = start_time_nano // 1000
                start_time_millis = start_time_nano // 1_000_000
                duration_micros = (end_time_nano - start_time_nano) // 1000

                # Convert to HH:MM format
                dt = datetime.fromtimestamp(start_time_nano / 1_000_000_000)
                time_str = dt.strftime("%H:%M")

                # Extract method name from various possible attributes
                method_name = (
                    get_span_attribute(span, "http.method")
                    or get_span_attribute(span, "grpc.method")
                    or get_span_attribute(span, "rpc.method")
                    or ""
                )

                # Extract http.status_code
                http_status_code = get_span_attribute(span, "http.status_code") or ""

                # Extract status code (0=UNSET, 1=OK, 2=ERROR)
                status = span.get("status", {})
                status_code = status.get("code", 0)

                row = {
                    "time": time_str,
                    "traceID": trace_id,
                    "spanID": span_id,
                    "serviceName": service_name,
                    "methodName": method_name,
                    "operationName": operation_name,
                    "startTimeMillis": str(start_time_millis),
                    "startTime": str(start_time_micros),
                    "duration": str(duration_micros),
                    "parentSpanID": parent_span_id,
                    "http.status_code": str(http_status_code) if http_status_code else "",
                    "status.code": str(status_code),
                }

                span_data = SpanData(row)
                spans.append(span_data)

    return spans


def process_single_file(file_path: Path) -> list[SpanData]:
    """Worker function to process a single JSONL trace file.

    Args:
        file_path: Path to the JSONL trace file

    Returns:
        List of SpanData objects extracted from the trace
    """
    all_spans = []

    try:
        with open(file_path) as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    trace_data = json.loads(line)
                    spans = extract_spans_for_csv(trace_data)
                    all_spans.extend(spans)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {file_path}: {e}")
                    continue

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return all_spans


def process_traces_to_csv(
    trace_dir: str,
    output_file: str,
    apply_duration_adjustment: bool = False,
    verbose: bool = False,
    num_workers: int | None = None,
):
    """Process all trace files and export spans to CSV.

    Args:
        trace_dir: Directory containing trace JSONL files
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

    if verbose:
        print(f"Extracted {len(all_spans)} spans from {total_files} trace files")

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
            "parentSpanID",
            "http.status_code",
            "status.code",
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
        print("No spans found")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description="Export OpenTelemetry traces to CSV"
    )

    parser.add_argument(
        "--trace-dir",
        "-i",
        default="data/otel-demo",
        help="Directory containing trace JSONL files (default: data/otel-demo)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="data/otel-demo-transformed/202510/1/traces.csv",
        help="Output CSV file path (default: data/otel-demo-transformed/202510/1/traces.csv)",
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
