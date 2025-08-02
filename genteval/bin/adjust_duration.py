#!/usr/bin/env python3
"""
Duration Adjustment CLI Tool

Adjusts the end time and duration of spans to match the latest end time of their children.
This ensures that parent spans properly encompass all child span execution times.
"""

import argparse
import sys

from genteval.data_collection.utils.duration_adjuster import DurationAdjuster


def main():
    """Main entry point for the adjust-duration CLI"""
    parser = argparse.ArgumentParser(
        description="Adjust span durations to encompass all child spans"
    )

    parser.add_argument("input_file", help="Input CSV file with span data")

    parser.add_argument(
        "--output", "-o", required=True, help="Output CSV file for adjusted data"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed adjustment information",
    )

    args = parser.parse_args()

    try:
        # Initialize the adjuster
        adjuster = DurationAdjuster(verbose=True)  # Always verbose for CLI usage

        # Load and process the data
        adjuster.load_csv(args.input_file)
        adjuster.build_relationships()
        adjusted_count = adjuster.adjust_durations()

        # Save results
        adjuster.write_csv(args.output)

        # Show summary
        if args.verbose:
            adjuster.print_adjustment_summary()

        if adjusted_count > 0:
            print(f"\nSuccessfully adjusted {adjusted_count} spans.")
        else:
            print(
                "\nNo adjustments were needed - all spans already properly encompass their children."
            )

    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
