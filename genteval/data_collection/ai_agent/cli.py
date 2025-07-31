"""CLI interface for trajectory conversion."""

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .trajectory_converter import TrajectoryConverter


def validate_input_file(input_file: str) -> None:
    """Validate that input file exists and is valid JSON."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        with input_path.open() as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise TypeError("Input file must contain a JSON array")

        if not data:
            raise ValueError("Input file contains empty array")

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in input file: {e}") from e


def ensure_output_directory(output_file: str) -> None:
    """Ensure output directory exists."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def display_stats(stats: dict, console: Console) -> None:
    """Display conversion statistics."""
    if not stats:
        return

    console.print("\n[bold green]Conversion Summary:[/bold green]")

    # Create summary table
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Total spans generated:", str(stats["total_spans"]))
    table.add_row("Root spans:", str(stats["root_spans"]))
    table.add_row(
        "Parent-child relationships:", str(stats["parent_child_relationships"])
    )

    console.print(table)

    # Service breakdown
    if stats.get("service_counts"):
        console.print("\n[bold]Service Breakdown:[/bold]")
        service_table = Table()
        service_table.add_column("Service", style="cyan")
        service_table.add_column("Span Count", style="bold", justify="right")

        for service, count in sorted(stats["service_counts"].items()):
            service_table.add_row(service, str(count))

        console.print(service_table)


def display_span_summaries(summaries: list[dict], console: Console) -> None:
    """Display human-readable span summaries."""
    if not summaries:
        return

    console.print("\n[bold blue]Span Summaries:[/bold blue]")

    # Create spans table
    table = Table()
    table.add_column("Time", style="dim")
    table.add_column("ID", style="bold blue", justify="center")
    table.add_column("Source", style="bright_green", justify="center")
    table.add_column("Service", style="cyan")
    table.add_column("Operation", style="yellow")
    table.add_column("Duration", justify="right", style="green")
    table.add_column("Cause", style="magenta", justify="center")
    table.add_column("Message", style="white")

    for summary in summaries:
        # Format duration
        duration_str = f"{summary['duration_ms']:.1f}ms"

        # Add indent for child spans
        service_name = summary["service"]
        if summary["is_child"]:
            service_name = f"  └─ {service_name}"

        # Format message (truncate if needed)
        message = summary["message"] or f"[{summary['action']}]"

        # Format cause
        cause_str = str(summary["cause"]) if summary["cause"] is not None else ""

        table.add_row(
            summary["time"],
            str(summary["id"]),
            summary["source"],
            service_name,
            summary["operation"],
            duration_str,
            cause_str,
            message,
        )

    console.print(table)


def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Convert AI agent trajectory JSON to distributed tracing CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/traj.json data/traces.csv
  %(prog)s input.json output.csv --verbose
        """,
    )

    parser.add_argument("input_file", help="Input trajectory JSON file")

    parser.add_argument("output_file", help="Output traces CSV file")

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed conversion statistics",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output except errors"
    )

    args = parser.parse_args()

    console = (
        Console(file=Path(os.devnull).open("w"))  # noqa: SIM115
        if args.quiet
        else Console()
    )

    try:
        # Validate input
        console.print(f"[dim]Validating input file: {args.input_file}[/dim]")
        validate_input_file(args.input_file)

        # Ensure output directory exists
        ensure_output_directory(args.output_file)

        # Convert with progress indicator
        converter = TrajectoryConverter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=args.quiet,
        ) as progress:
            task = progress.add_task("Converting trajectory to traces...", total=None)

            converter.convert(args.input_file, args.output_file)

            progress.update(task, description="[green]Conversion completed!")

        # Display results
        if not args.quiet:
            console.print(
                "\n[bold green]✓[/bold green] Successfully converted trajectory data"
            )
            console.print(f"[dim]Output written to: {args.output_file}[/dim]")

            # Always display span summaries
            summaries = converter.get_span_summaries()
            display_span_summaries(summaries, console)

            if args.verbose:
                stats = converter.get_conversion_stats()
                display_stats(stats, console)

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}", file=sys.stderr)
        sys.exit(1)

    except ValueError as e:
        console.print(f"[bold red]Validation Error:[/bold red] {e}", file=sys.stderr)
        sys.exit(1)

    except OSError as e:
        console.print(f"[bold red]File Error:[/bold red] {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            console.print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
