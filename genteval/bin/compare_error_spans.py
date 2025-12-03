#!/usr/bin/env python3
"""
CLI tool to compare ERROR spans (status.code=2) across different GenT runs.

Compares:
1. http.status_code distribution for ERROR spans
2. Time bucket distribution of errors
"""

import argparse
import pathlib
from collections import defaultdict

import pandas as pd


def load_error_spans(spans_path: pathlib.Path) -> pd.DataFrame:
    """Load spans.csv and filter for ERROR spans (status.code=2)."""
    df = pd.read_csv(spans_path)
    # Filter for ERROR spans (status.code == 2)
    return df[df["status.code"] == 2].copy()


def compute_time_bucket(start_time: int, bucket_duration_us: int = 60_000_000) -> int:
    """Compute time bucket from startTime (microseconds)."""
    return start_time // bucket_duration_us


def normalize_status_code(val):
    """Convert status code to clean string (empty string or integer string)."""
    if pd.isna(val) or val == "":
        return ""
    # Convert float to int string (200.0 -> "200")
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val)


def analyze_error_spans(error_df: pd.DataFrame) -> dict:
    """Analyze ERROR spans and return statistics."""
    # Normalize http.status_code to clean strings
    error_df["http.status_code_normalized"] = error_df["http.status_code"].apply(
        normalize_status_code
    )

    # http.status_code distribution
    status_counts = error_df["http.status_code_normalized"].value_counts().to_dict()

    # Time bucket distribution
    error_df["time_bucket"] = error_df["startTime"].apply(compute_time_bucket)
    time_bucket_counts = error_df["time_bucket"].value_counts().sort_index().to_dict()

    # Service distribution
    service_counts = error_df["serviceName"].value_counts().to_dict()

    # Combined: time_bucket -> http.status_code -> count
    time_status_counts = defaultdict(lambda: defaultdict(int))
    for _, row in error_df.iterrows():
        bucket = row["time_bucket"]
        status = row["http.status_code_normalized"]
        time_status_counts[bucket][status] += 1

    return {
        "total": len(error_df),
        "status_counts": status_counts,
        "time_bucket_counts": time_bucket_counts,
        "service_counts": service_counts,
        "time_status_counts": dict(time_status_counts),
    }


def print_comparison_table(results: dict[str, dict], label: str = "http.status_code"):
    """Print a comparison table across all runs."""
    # Collect all unique values
    all_values = set()
    for stats in results.values():
        if label == "http.status_code":
            all_values.update(stats["status_counts"].keys())
        elif label == "time_bucket":
            all_values.update(stats["time_bucket_counts"].keys())
        elif label == "service":
            all_values.update(stats["service_counts"].keys())

    all_values = sorted(all_values, key=lambda x: (x == "", str(x)))

    # Print header
    run_names = list(results.keys())
    header = f"| {label:<20} |" + "|".join(f" {name:>12} " for name in run_names) + "|"
    separator = "|" + "-" * 22 + "|" + "|".join("-" * 14 for _ in run_names) + "|"

    print(f"\n## {label} Distribution\n")
    print(header)
    print(separator)

    # Print rows
    for value in all_values:
        display_value = "(empty)" if value == "" else str(value)
        row = f"| {display_value:<20} |"
        for name in run_names:
            stats = results[name]
            if label == "http.status_code":
                count = stats["status_counts"].get(value, 0)
            elif label == "time_bucket":
                count = stats["time_bucket_counts"].get(value, 0)
            elif label == "service":
                count = stats["service_counts"].get(value, 0)
            row += f" {count:>12} |"
        print(row)

    # Print total
    total_row = f"| {'TOTAL':<20} |"
    for name in run_names:
        total_row += f" {results[name]['total']:>12} |"
    print(separator)
    print(total_row)


def print_time_bucket_detail(results: dict[str, dict]):
    """Print detailed time bucket comparison with http.status_code breakdown."""
    # Collect all time buckets
    all_buckets = set()
    for stats in results.values():
        all_buckets.update(stats["time_bucket_counts"].keys())
    all_buckets = sorted(all_buckets)

    print("\n## Time Bucket Detail (bucket -> status_code counts)\n")

    for bucket in all_buckets:
        print(f"\n### Time Bucket: {bucket}")
        for name, stats in results.items():
            time_status = stats["time_status_counts"].get(bucket, {})
            if time_status:
                # Sort by string representation to handle mixed types
                sorted_items = sorted(time_status.items(), key=lambda x: str(x[0]))
                status_str = ", ".join(
                    f"{k if k else '(empty)'}:{v}" for k, v in sorted_items
                )
                print(f"  {name}: {status_str}")
            else:
                print(f"  {name}: (no errors)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ERROR spans across GenT runs"
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=pathlib.Path,
        help="Directories containing dataset/spans.csv files",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed time bucket breakdown",
    )

    args = parser.parse_args()

    results = {}

    for dir_path in args.directories:
        spans_path = dir_path / "dataset" / "spans.csv"
        if not spans_path.exists():
            print(f"Warning: {spans_path} not found, skipping")
            continue

        # Use directory name as label
        label = dir_path.name
        error_df = load_error_spans(spans_path)
        results[label] = analyze_error_spans(error_df)
        print(f"Loaded {len(error_df)} ERROR spans from {label}")

    if not results:
        print("No valid data found")
        return

    # Print comparison tables
    print_comparison_table(results, "http.status_code")
    print_comparison_table(results, "time_bucket")
    print_comparison_table(results, "service")

    if args.detail:
        print_time_bucket_detail(results)


if __name__ == "__main__":
    main()
