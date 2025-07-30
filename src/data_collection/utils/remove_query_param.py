#!/usr/bin/env python3
"""
CLI tool to remove query parameters from operationName column in CSV files.
"""

import argparse
import csv
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def remove_query_params(operation_name):
    """
    Remove query parameters from a URL or path.

    Args:
        operation_name (str): The operation name which may contain query parameters

    Returns:
        str: The operation name without query parameters
    """
    if not operation_name or "?" not in operation_name:
        return operation_name

    # Parse the URL/path to remove query parameters
    parsed = urlparse(operation_name)
    # Create a new URL without query parameters
    clean_url = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            "",  # Remove query
            parsed.fragment,
        )
    )

    # If it was just a path (no scheme), return just the path
    if not parsed.scheme and not parsed.netloc:
        return parsed.path

    return clean_url


def process_csv(input_file, output_file):
    """
    Process CSV file to remove query parameters from operationName column.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with (
            open(input_path, newline="", encoding="utf-8") as infile,
            open(output_path, "w", newline="", encoding="utf-8") as outfile,
        ):
            reader = csv.DictReader(infile)

            if "operationName" not in reader.fieldnames:
                print(
                    "Error: CSV file does not contain 'operationName' column.",
                    file=sys.stderr,
                )
                sys.exit(1)

            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()

            rows_processed = 0
            rows_modified = 0

            for row in reader:
                original_operation = row["operationName"]
                cleaned_operation = remove_query_params(original_operation)

                if original_operation != cleaned_operation:
                    rows_modified += 1

                row["operationName"] = cleaned_operation
                writer.writerow(row)
                rows_processed += 1

            print(f"Successfully processed {rows_processed} rows.")
            print(f"Modified {rows_modified} rows with query parameters.")
            print(f"Output written to: {output_file}")

    except Exception as e:
        print(f"Error processing CSV file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Remove query parameters from operationName column in CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.csv -o output.csv
  %(prog)s output/test.csv -o output/test-clean.csv
        """,
    )

    parser.add_argument("input_file", help="Input CSV file path")

    parser.add_argument("-o", "--output", required=True, help="Output CSV file path")

    args = parser.parse_args()

    process_csv(args.input_file, args.output)


if __name__ == "__main__":
    main()
