"""Generate synthetic traces by replicating a template trace across time.

This tool reads a single trace from a CSV file and replicates it at a specified
rate over a given duration. Each replica gets new trace and span IDs while
preserving the original structure, parent-child relationships, and durations.

Example usage:
    generate_synthetic_traces \\
        --input data/otel-demo-transformed/simple/1/single.csv \\
        --output data/synthetic/otel-demo-60rpm-5min \\
        --rate 60 \\
        --duration 5
"""

import argparse
import sys
from pathlib import Path
from random import choices

import pandas as pd

from genteval.bin.logger import setup_logging


setup_logging()


def generate_trace_id():
    """Generate a random 16-character hex trace ID."""
    return "".join(choices("0123456789abcdef", k=16))


def generate_span_id():
    """Generate a random 8-character hex span ID."""
    return "".join(choices("0123456789abcdef", k=8))


def replicate_trace(template_df, new_trace_id, time_offset):
    """Create a replica of the template trace with new IDs and time offset.

    Args:
        template_df: DataFrame containing the template trace spans
        new_trace_id: New trace ID for the replica
        time_offset: Microseconds to add to all timestamps

    Returns:
        DataFrame with the replicated trace
    """
    # Create a copy of the template
    replica_df = template_df.copy()

    # Create mapping of old span IDs to new span IDs
    old_span_ids = replica_df["spanID"].unique()
    span_id_mapping = {old_id: generate_span_id() for old_id in old_span_ids}

    # Update trace ID
    replica_df["traceID"] = new_trace_id

    # Update span IDs using the mapping
    replica_df["spanID"] = replica_df["spanID"].map(span_id_mapping)

    # Update parent span IDs using the mapping (preserve empty/null parents)
    replica_df["parentSpanID"] = replica_df["parentSpanID"].apply(
        lambda x: span_id_mapping.get(x, x) if pd.notna(x) and x != "" else x
    )

    # Adjust timestamps by adding the time offset
    replica_df["startTime"] = replica_df["startTime"] + time_offset

    return replica_df


def generate_synthetic_dataset(template_df, rate, duration_minutes):
    """Generate a synthetic dataset by replicating the template trace.

    Args:
        template_df: DataFrame containing the template trace spans
        rate: Number of traces per minute
        duration_minutes: Duration in minutes

    Returns:
        DataFrame containing all replicated traces
    """
    # Calculate total number of traces to generate
    total_traces = rate * duration_minutes

    # Calculate time between traces in microseconds
    time_between_traces = (60_000_000) // rate  # 60 seconds in microseconds / rate

    print(
        f"Generating {total_traces} traces at {rate} traces/minute over {duration_minutes} minutes"
    )
    print(f"Time between traces: {time_between_traces / 1000:.2f} ms")

    all_traces = []

    for i in range(total_traces):
        # Generate new trace ID
        new_trace_id = generate_trace_id()

        # Calculate time offset for this replica
        time_offset = i * time_between_traces

        # Create replica
        replica = replicate_trace(template_df, new_trace_id, time_offset)
        all_traces.append(replica)

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{total_traces} traces...")

    # Concatenate all replicas
    result_df = pd.concat(all_traces, ignore_index=True)

    print(f"Total spans generated: {len(result_df)}")
    print(f"Total traces generated: {total_traces}")

    return result_df


def main():
    """Main entry point for the synthetic trace generation tool."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic traces by replicating a template trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    generate_synthetic_traces \\
        --input data/otel-demo-transformed/simple/1/single.csv \\
        --output data/synthetic/otel-demo-60rpm-5min \\
        --rate 60 \\
        --duration 5
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to input CSV file containing the template trace",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory path for generated traces",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=60,
        help="Number of traces to generate per minute (default: 60)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=5,
        help="Duration in minutes (default: 5)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist")
        return 1

    if args.rate <= 0:
        print(f"Error: Rate must be positive, got {args.rate}")
        return 1

    if args.duration <= 0:
        print(f"Error: Duration must be positive, got {args.duration}")
        return 1

    # Load template trace
    print(f"Loading template trace from {args.input}")
    template_df = pd.read_csv(args.input)

    # Convert http.status_code to integer type (preserving empty values)
    if "http.status_code" in template_df.columns:
        # Convert to Int64 (nullable integer type) to preserve empty values
        template_df["http.status_code"] = pd.to_numeric(
            template_df["http.status_code"], errors="coerce"
        ).astype("Int64")

    # Get unique trace IDs in the template
    template_trace_ids = template_df["traceID"].unique()
    print(
        f"Template contains {len(template_trace_ids)} trace(s) with {len(template_df)} spans"
    )

    if len(template_trace_ids) > 1:
        print("Warning: Template contains multiple traces, using all spans")

    # Generate synthetic dataset
    synthetic_df = generate_synthetic_dataset(template_df, args.rate, args.duration)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_file = args.output / "traces.csv"
    print(f"Saving synthetic traces to {output_file}")

    # Ensure http.status_code is saved as integer (not float)
    # Use na_rep='' to write NaN/empty values as empty strings
    synthetic_df.to_csv(output_file, index=False, na_rep="")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
