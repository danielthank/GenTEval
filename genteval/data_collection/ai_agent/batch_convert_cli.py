"""Batch CLI interface for trajectory conversion."""

import argparse
import gzip
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from genteval.data_collection.utils.duration_adjuster import DurationAdjuster

from .trajectory_converter import TrajectoryConverter


def find_trajectory_files(root_dir: str) -> list[tuple[Path, Path]]:
    """Find all trajectory files and determine their output paths.

    Returns:
        List of (input_path, output_path) tuples
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    # Find all trajectory files matching the pattern
    pattern = "evaluation/1.0.0/**/trajectories/*.json.gz"
    trajectory_files = list(root_path.glob(pattern))

    if not trajectory_files:
        raise ValueError(
            f"No trajectory files found in {root_dir} matching pattern {pattern}"
        )

    # Generate output paths
    file_pairs = []
    for traj_file in trajectory_files:
        # Extract the relative path components we need
        parts = traj_file.parts

        # Find the indices for key parts
        try:
            eval_idx = parts.index("evaluation")
            version_idx = eval_idx + 1  # "1.0.0"
            model_idx = (
                version_idx + 1
            )  # e.g., "20241217_OpenHands-0.14.2-gemini-1.5-pro"
        except ValueError:
            continue

        if model_idx >= len(parts):
            continue

        model_name = parts[model_idx]
        filename = Path(traj_file.stem).stem  # Remove both .gz and .json extensions

        # Create output path: output_dir/model_name/1/traces.csv
        file_pairs.append((traj_file, model_name, filename))

    return file_pairs


def validate_trajectory_file(input_file: Path) -> bool:
    """Validate that trajectory file exists and contains valid JSON."""
    if not input_file.exists():
        return False

    try:
        # First check if it's a valid gzip file
        with gzip.open(input_file, "rb") as f:
            # Try to read the first few bytes to validate gzip format
            f.read(10)

        # If that worked, try to parse as JSON
        with gzip.open(input_file, "rt", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return False

        return bool(data)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, gzip.BadGzipFile):
        return False


def ensure_output_directory(output_file: Path) -> None:
    """Ensure output directory exists."""
    output_file.parent.mkdir(parents=True, exist_ok=True)


def process_single_file(
    input_file: Path,
    filename: str,
    output_dir: Path,
    verbose: bool,
) -> tuple[bool, list[str]]:
    """Process a single trajectory file.

    Returns:
        (success: bool, csv_lines: list[str])
    """
    try:
        # Create temporary output file
        temp_output = output_dir / f"temp_{filename}.csv"
        temp_adjusted = output_dir / f"temp_adjusted_{filename}.csv"

        # Load gzipped trajectory data
        with gzip.open(input_file, "rt", encoding="utf-8") as f:
            trajectory_data = json.load(f)

        # Convert trajectory data
        converter = TrajectoryConverter()
        converter.convert_data(trajectory_data)
        converter.write_traces(str(temp_output))

        # Apply duration adjustment
        adjuster = DurationAdjuster(verbose=verbose)
        adjuster.load_csv(str(temp_output))
        adjuster.build_relationships()
        adjuster.adjust_durations()
        adjuster.write_csv(str(temp_adjusted))

        # Read the adjusted data
        with temp_adjusted.open("rt", encoding="utf-8") as f:
            content = f.read()
            csv_lines = content.splitlines(keepends=True)

        # Clean up temp files
        temp_output.unlink(missing_ok=True)
        temp_adjusted.unlink(missing_ok=True)

    except (json.JSONDecodeError, OSError, gzip.BadGzipFile):
        # Clean up temp files on error
        if "temp_output" in locals():
            temp_output.unlink(missing_ok=True)
        if "temp_adjusted" in locals():
            temp_adjusted.unlink(missing_ok=True)
        return False, []
    else:
        return True, csv_lines


def main() -> None:
    """Main batch CLI function."""
    parser = argparse.ArgumentParser(
        description="Batch convert AI agent trajectory JSON files to distributed tracing CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --root_dir ./data/the-agent-company --output_dir ./output
  %(prog)s --root_dir /path/to/data --output_dir /path/to/output --verbose
        """,
    )

    parser.add_argument(
        "--root_dir",
        default="./data/the-agent-company",
        help="Root directory containing trajectory files (default: ./data/the-agent-company)",
    )

    parser.add_argument(
        "--output_dir",
        default="./data/the-agent-company-transformed",
        help="Output directory for converted CSV files (default: ./data/the-agent-company-transformed)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed conversion statistics for each file",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what files would be processed without actually converting them",
    )

    args = parser.parse_args()

    # Simple console setup
    console = Console()

    try:
        # Find all trajectory files
        console.print(f"[dim]Scanning for trajectory files in: {args.root_dir}[/dim]")
        file_pairs = find_trajectory_files(args.root_dir)

        console.print(f"[green]Found {len(file_pairs)} trajectory files[/green]")

        if args.dry_run:
            console.print("\n[bold]Files that would be processed:[/bold]")
            for input_file, model_name, _filename in file_pairs:
                output_file = Path(args.output_dir) / model_name / "1" / "traces.csv"
                console.print(f"  {input_file} -> {output_file}")
            return

        # Group files by model to create separate CSV files
        model_groups = {}
        for input_file, model_name, filename in file_pairs:
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append((input_file, filename))

        total_files = len(file_pairs)
        converted_files = 0
        failed_files = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            disable=False,
        ) as progress:
            overall_task = progress.add_task(
                "Converting trajectory files...", total=total_files
            )

            for model_name, files in model_groups.items():
                # Create output directory for this model
                output_file = Path(args.output_dir) / model_name / "1" / "traces.csv"
                ensure_output_directory(output_file)

                model_task = progress.add_task(
                    f"Processing {model_name}...", total=len(files)
                )

                # Process files sequentially
                all_traces_data = []
                for input_file, filename in files:
                    # Validate file first
                    if not validate_trajectory_file(input_file):
                        console.print(
                            f"[yellow]Warning: Skipping invalid/corrupted file {input_file}[/yellow]"
                        )
                        failed_files += 1
                        progress.update(model_task, advance=1)
                        progress.update(overall_task, advance=1)
                        continue

                    try:
                        success, csv_lines = process_single_file(
                            input_file,
                            filename,
                            output_file.parent,
                            args.verbose,
                        )
                        if success:
                            if csv_lines:
                                # Skip header for all but first file
                                start_line = 1 if all_traces_data else 0
                                if start_line == 0:  # First file, include header
                                    all_traces_data.extend(csv_lines)
                                else:  # Subsequent files, skip header
                                    all_traces_data.extend(csv_lines[1:])
                            converted_files += 1
                        else:
                            failed_files += 1
                    except (OSError, json.JSONDecodeError, gzip.BadGzipFile) as e:
                        console.print(f"[red]Processing error: {e}[/red]")
                        failed_files += 1

                    progress.update(model_task, advance=1)
                    progress.update(overall_task, advance=1)

                # Write combined traces for this model
                if all_traces_data:
                    with output_file.open("w", encoding="utf-8") as f:
                        f.writelines(all_traces_data)

                    console.print(
                        f"[green]✓[/green] Model {model_name}: {len(files)} files -> {output_file}"
                    )

                progress.remove_task(model_task)

        # Final summary
        console.print("\n[bold green]Batch conversion completed![/bold green]")
        console.print(f"[dim]Converted: {converted_files} files[/dim]")
        if failed_files > 0:
            console.print(f"[dim]Failed: {failed_files} files[/dim]")
        console.print(f"[dim]Output directory: {args.output_dir}[/dim]")

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
