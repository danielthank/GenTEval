#!/usr/bin/env python3
"""
Analyze OpenTelemetry Trace Attributes

This script analyzes attributes in OpenTelemetry trace data to provide:
1. List of all unique attributes
2. Coverage: how many spans (and ratio) have each attribute
3. Cardinality: number of unique values for each attribute
"""

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from tqdm import tqdm


class AttributeAnalyzer:
    """Analyzes attributes from OpenTelemetry spans."""

    def __init__(self):
        # Separate tracking for span and resource attributes
        self.span_attr_count = defaultdict(int)
        self.span_attr_values = defaultdict(set)
        self.resource_attr_count = defaultdict(int)
        self.resource_attr_values = defaultdict(set)
        self.total_spans = 0
        self.total_resources = 0

    def extract_attribute_value(self, attr_value: dict[str, Any]) -> Any:
        """Extract the actual value from an OpenTelemetry attribute value object."""
        if "stringValue" in attr_value:
            return attr_value["stringValue"]
        elif "intValue" in attr_value:
            return attr_value["intValue"]
        elif "doubleValue" in attr_value:
            return attr_value["doubleValue"]
        elif "boolValue" in attr_value:
            return attr_value["boolValue"]
        elif "arrayValue" in attr_value:
            # For arrays, convert to a hashable representation
            return str(attr_value["arrayValue"])
        elif "kvlistValue" in attr_value:
            # For key-value lists, convert to a hashable representation
            return str(attr_value["kvlistValue"])
        return str(attr_value)

    def process_span(self, span: dict[str, Any]):
        """Process a single span and extract its attributes."""
        self.total_spans += 1

        # Process span attributes
        for attr in span.get("attributes", []):
            key = attr["key"]
            value = self.extract_attribute_value(attr["value"])

            self.span_attr_count[key] += 1
            # Convert value to string for set storage (to handle unhashable types)
            self.span_attr_values[key].add(str(value))

    def process_resource(self, resource: dict[str, Any]):
        """Process resource attributes."""
        self.total_resources += 1

        # Process resource attributes
        for attr in resource.get("attributes", []):
            key = attr["key"]
            value = self.extract_attribute_value(attr["value"])

            self.resource_attr_count[key] += 1
            self.resource_attr_values[key].add(str(value))

    def merge(self, other: "AttributeAnalyzer"):
        """Merge another analyzer's results into this one."""
        self.total_spans += other.total_spans
        self.total_resources += other.total_resources

        for key, count in other.span_attr_count.items():
            self.span_attr_count[key] += count

        for key, values in other.span_attr_values.items():
            self.span_attr_values[key].update(values)

        for key, count in other.resource_attr_count.items():
            self.resource_attr_count[key] += count

        for key, values in other.resource_attr_values.items():
            self.resource_attr_values[key].update(values)

    def print_report(self):
        """Print a formatted analysis report."""
        print("=" * 80)
        print("OpenTelemetry Attribute Analysis Report")
        print("=" * 80)
        print(f"\nTotal spans analyzed: {self.total_spans:,}")
        print(f"Total resource instances: {self.total_resources:,}")

        # Span attributes section
        print("\n" + "=" * 80)
        print("SPAN ATTRIBUTES")
        print("=" * 80)
        print(f"\nTotal unique span attributes: {len(self.span_attr_count)}")
        print(
            "\n{:<50} {:>12} {:>10} {:>12}".format(
                "Attribute", "Count", "Ratio", "Cardinality"
            )
        )
        print("-" * 80)

        # Sort by count (descending)
        sorted_span_attrs = sorted(
            self.span_attr_count.items(), key=lambda x: x[1], reverse=True
        )

        for key, count in sorted_span_attrs:
            ratio = count / self.total_spans if self.total_spans > 0 else 0
            cardinality = len(self.span_attr_values[key])
            print(
                f"{key:<50} {count:>12,} {ratio:>9.1%} {cardinality:>12,}"
            )

        # Resource attributes section
        print("\n" + "=" * 80)
        print("RESOURCE ATTRIBUTES")
        print("=" * 80)
        print(f"\nTotal unique resource attributes: {len(self.resource_attr_count)}")
        print(
            "\n{:<50} {:>12} {:>10} {:>12}".format(
                "Attribute", "Count", "Ratio", "Cardinality"
            )
        )
        print("-" * 80)

        # Sort by count (descending)
        sorted_resource_attrs = sorted(
            self.resource_attr_count.items(), key=lambda x: x[1], reverse=True
        )

        for key, count in sorted_resource_attrs:
            ratio = count / self.total_resources if self.total_resources > 0 else 0
            cardinality = len(self.resource_attr_values[key])
            print(
                f"{key:<50} {count:>12,} {ratio:>9.1%} {cardinality:>12,}"
            )

        print("\n" + "=" * 80)


def process_single_file(file_path: Path) -> AttributeAnalyzer:
    """Worker function to process a single JSONL trace file.

    Args:
        file_path: Path to the JSONL trace file

    Returns:
        AttributeAnalyzer with accumulated statistics
    """
    analyzer = AttributeAnalyzer()

    try:
        with open(file_path) as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {file_path}: {e}")
                    continue

                # Process each resourceSpans entry
                for resource_spans in data.get("resourceSpans", []):
                    # Process resource attributes
                    if "resource" in resource_spans:
                        analyzer.process_resource(resource_spans["resource"])

                    # Process all spans
                    for scope_spans in resource_spans.get("scopeSpans", []):
                        for span in scope_spans.get("spans", []):
                            analyzer.process_span(span)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return analyzer


def analyze_otel_traces(
    trace_dir: str,
    verbose: bool = False,
    num_workers: int | None = None,
):
    """Analyze OpenTelemetry trace attributes.

    Args:
        trace_dir: Directory containing trace JSONL files
        verbose: Whether to show detailed processing information
        num_workers: Number of worker processes (default: cpu_count - 1)
    """
    trace_path = Path(trace_dir)

    if not trace_path.exists():
        print(f"Error: Directory {trace_path} does not exist")
        return

    if verbose:
        print(f"Analyzing traces in: {trace_path}")

    # Get all JSON files
    json_files = list(trace_path.glob("*.json"))
    total_files = len(json_files)

    if total_files == 0:
        print(f"No JSON files found in {trace_path}")
        return

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    if verbose:
        print(f"Found {total_files} trace files")
        print(f"Using {num_workers} worker processes")

    # Process files in parallel
    main_analyzer = AttributeAnalyzer()

    with Pool(processes=num_workers) as pool:
        chunksize = max(1, total_files // (num_workers * 4))

        for analyzer in tqdm(
            pool.imap_unordered(process_single_file, json_files, chunksize=chunksize),
            total=total_files,
            desc="Processing trace files",
            unit="file",
            disable=not verbose,
        ):
            main_analyzer.merge(analyzer)

    # Print the analysis report
    main_analyzer.print_report()


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze OpenTelemetry trace attributes"
    )

    parser.add_argument(
        "--trace-dir",
        "-i",
        default="data/otel-demo",
        help="Directory containing trace JSONL files (default: data/otel-demo)",
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

    analyze_otel_traces(
        args.trace_dir,
        verbose=args.verbose,
        num_workers=args.workers,
    )

    return 0


if __name__ == "__main__":
    exit(main())
