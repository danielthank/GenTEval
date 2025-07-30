#!/usr/bin/env python3
"""
HTTP Request Tree Recorder for GenTEval

A tool that records HTTP requests with parent-child relationships using Playwright.
Shows how requests initiate other requests, creating a tree structure.

This is integrated from the request-tree project into GenTEval for trace data collection.
"""

import argparse
import asyncio
import csv
import sys
import traceback

from ..data_collection.request_recorder_api import RequestRecorderAPI


def parse_schedule_file(file_path):
    """
    Parse a schedule file containing rate and speed configuration.

    Expected format:
    rate: [60] * 60
    speed: ["4g", "4g", "wifi", "wifi"] * 15

    Returns a tuple (rate_list, speed_list)
    """
    rate_list = None
    speed_list = None

    try:
        with open(file_path) as f:
            content = f.read()

        # Parse the file content line by line
        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith("rate:"):
                # Extract the rate expression
                rate_expr = line.split("rate:", 1)[1].strip()
                # Safely evaluate the expression
                rate_list = eval(rate_expr)
            elif line.startswith("speed:"):
                # Extract the speed expression
                speed_expr = line.split("speed:", 1)[1].strip()
                # Safely evaluate the expression
                speed_list = eval(speed_expr)

        if rate_list is None:
            raise ValueError("Schedule file must contain a 'rate:' line")
        if speed_list is None:
            raise ValueError("Schedule file must contain a 'speed:' line")

        # Ensure both lists have the same length
        if len(rate_list) != len(speed_list):
            raise ValueError(
                f"Rate list length ({len(rate_list)}) must match speed list length ({len(speed_list)})"
            )

        return rate_list, speed_list

    except FileNotFoundError:
        raise ValueError(f"Schedule file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error parsing schedule file: {e}")


async def main_async():
    parser = argparse.ArgumentParser(
        description="Record HTTP requests with parent-child relationships for GenTEval"
    )

    parser.add_argument(
        "--wait", "-w", type=int, default=5, help="Wait time in seconds (default: 5)"
    )
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument(
        "--detailed-initiators",
        action="store_true",
        help="Show detailed initiator information (requires single URL)",
    )

    parser.add_argument(
        "--url",
        "-u",
        required=True,
        help="Single URL to record and print tree for",
    )

    parser.add_argument(
        "--multiple-visits",
        action="store_true",
        help="Enable multiple visits mode to record the same URL multiple times (use with --rate/--speed or --schedule-file)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval duration in seconds for each slot in multiple visits mode (default: 60)",
    )

    parser.add_argument(
        "--rate",
        "-r",
        type=float,
        nargs="+",
        help="Rate of visits per minute for each slot in multiple visits mode (e.g., 10 20 30 40). Optional if using --schedule-file",
    )

    parser.add_argument(
        "--speed",
        type=str,
        nargs="+",
        help="Network speed for each slot in multiple visits mode (e.g., 3g 3g 4g 4g). Optional if using --schedule-file",
    )

    parser.add_argument(
        "--schedule-file",
        type=str,
        help="Path to a schedule file containing rate and speed configuration",
    )

    args = parser.parse_args()

    try:
        # Initialize the API
        api = RequestRecorderAPI()

        if args.multiple_visits:
            # Validate arguments for multiple visits mode
            if not args.interval:
                print(
                    "Error: --interval is required for multiple visits mode",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Handle schedule file or command line arguments
            if args.schedule_file:
                # Read from schedule file
                try:
                    rate_list, speed_list = parse_schedule_file(args.schedule_file)
                    print(f"Loaded schedule from file: {args.schedule_file}")
                except ValueError as e:
                    print(f"Error: {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                # Use command line arguments
                if not args.rate:
                    print(
                        "Error: --rate is required for multiple visits mode (or use --schedule-file)",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                if not args.speed:
                    print(
                        "Error: --speed is required for multiple visits mode (or use --schedule-file)",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                if len(args.rate) != len(args.speed):
                    print(
                        "Error: Number of rate values must match number of speed values",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                rate_list = args.rate
                speed_list = args.speed

            # Handle multiple visits mode
            print(f"Recording multiple visits for: {args.url}")
            print(f"Interval: {args.interval} seconds per slot")
            print(f"Rates (visits/min): {rate_list}")
            print(f"Speeds: {speed_list}")

            # Create schedule for each slot
            schedule = []
            for i, (rate, speed) in enumerate(zip(rate_list, speed_list, strict=False)):
                slot = {
                    "start_time": i * args.interval,
                    "duration": args.interval,
                    "rate_per_minute": rate,
                    "speed": speed,
                }
                schedule.append(slot)
                print(
                    f"Slot {i + 1}: {args.interval}s, {rate} visits/min, {speed} speed"
                )

            results = await api.get_spans_for_scheduled_visits(
                args.url, schedule=schedule, wait_time=args.wait, headless=False
            )

            # Print summary of results
            total_visits = len(results)
            total_requests = sum(len(spans) for spans in results.values())

            print("\nMultiple Visits Summary:")
            print(f"Total visits completed: {total_visits}")
            print(f"Total requests recorded: {total_requests}")
            if total_visits > 0:
                print(
                    f"Average requests per visit: {total_requests / total_visits:.1f}"
                )

            # Optionally save results to CSV if output is specified
            if args.output:
                # Get all recorders for this URL
                all_recorders = api.get_all_recorders()
                url_recorders = {
                    k: v
                    for k, v in all_recorders.items()
                    if k.startswith(args.url + ":")
                }

                if url_recorders:
                    with open(args.output, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        header_written = False

                        # Write data from all recorders using the generator
                        for recorder_key, recorder in url_recorders.items():
                            for i, row in enumerate(recorder.iter_otel_csv_rows()):
                                if i == 0:  # Header row
                                    if not header_written:
                                        writer.writerow(row)
                                        header_written = True
                                else:  # Data row
                                    writer.writerow(row)

                    print(f"Results saved to: {args.output}")
                else:
                    print("No recorders found to save.")

        else:
            # Handle single URL tree printing (original behavior)
            print(f"Recording and printing tree for: {args.url}")
            success = await api.record_and_print_tree(
                args.url, args.wait, headless=False
            )
            if not success:
                sys.exit(1)

            # Show detailed initiator information if requested
            if args.detailed_initiators:
                api.print_detailed_initiator_info_for_url(args.url)

            # Optionally save to CSV if output is specified
            if args.output:
                recorder = api.get_recorder_for_url(args.url)
                if recorder:
                    recorder.save_to_csv(args.output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
