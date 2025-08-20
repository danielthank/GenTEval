#!/usr/bin/env python3
"""
Duration Adjustment CLI Tool

Adjusts the end time and duration of spans to match the latest end time of their children.
This ensures that parent spans properly encompass all child span execution times.
"""

import argparse
import logging
import sys
import time

from genteval.bin.logger import setup_logging
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

    # Setup logging using the colored logger
    setup_logging(
        log_level=logging.INFO,
        simplified=True,
    )

    logger = logging.getLogger(__name__)

    logger.info("Starting duration adjustment process")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output}")
    start_time = time.time()

    try:
        # Initialize the adjuster
        logger.info("Initializing DurationAdjuster")
        adjuster = DurationAdjuster(verbose=True)  # Always verbose for CLI usage

        # Load and process the data
        logger.info("Loading CSV data...")
        adjuster.load_csv(args.input_file)

        logger.info("Building parent-child relationships...")
        adjuster.build_relationships()

        logger.info("Adjusting span durations...")
        adjusted_count = adjuster.adjust_durations()

        # Save results
        logger.info("Writing adjusted data to output file...")
        adjuster.write_csv(args.output)

        # Show summary
        if args.verbose:
            adjuster.print_adjustment_summary()

        elapsed_time = time.time() - start_time
        logger.info(f"Duration adjustment completed in {elapsed_time:.2f} seconds")

        if adjusted_count > 0:
            print(f"\nSuccessfully adjusted {adjusted_count} spans.")
            logger.info(
                f"Adjusted {adjusted_count} spans out of {len(adjuster.spans)} total spans"
            )
        else:
            print(
                "\nNo adjustments were needed - all spans already properly encompass their children."
            )
            logger.info("No adjustments were needed")

    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
