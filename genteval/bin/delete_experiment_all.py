#!/usr/bin/env python3
"""
Delete experiment output directories for all runs.

This script lists and optionally deletes experiment output directories
(specified by output_dir_name) from all app/service/fault/run combinations.
"""

import argparse
import pathlib
import shutil

from .all_utils import add_common_arguments, display_configuration
from .utils import get_dir_with_root, run_dirs


def find_experiment_dirs(
    root_dir: pathlib.Path,
    output_dir_name: str,
    applications: list[str] | None = None,
    services: list[str] | None = None,
    faults: list[str] | None = None,
    runs: list[int] | None = None,
) -> list[pathlib.Path]:
    """
    Find all experiment directories matching the criteria.

    Args:
        root_dir: Root directory containing experiment outputs
        output_dir_name: Name of the experiment output directory to find
        applications: List of applications to filter by
        services: List of services to filter by
        faults: List of faults to filter by
        runs: List of runs to filter by

    Returns:
        List of paths to experiment directories
    """
    experiment_dirs = []

    for app_name, service, fault, run in run_dirs(applications, services, faults, runs):
        experiment_dir = (
            get_dir_with_root(root_dir, app_name, service, fault, run) / output_dir_name
        )

        if experiment_dir.exists() and experiment_dir.is_dir():
            experiment_dirs.append(experiment_dir)

    return experiment_dirs


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_directory_size(path: pathlib.Path) -> int:
    """Get the total size of a directory in bytes."""
    total_size = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
    except (OSError, PermissionError):
        # Skip directories we can't access
        pass
    return total_size


def list_experiment_dirs(experiment_dirs: list[pathlib.Path], show_sizes: bool = False):
    """List experiment directories with optional size information."""
    if not experiment_dirs:
        print("No experiment directories found.")
        return

    print(f"\nFound {len(experiment_dirs)} experiment directories:")
    print("-" * 80)

    total_size = 0
    for i, exp_dir in enumerate(experiment_dirs, 1):
        relative_path = exp_dir.relative_to(exp_dir.parents[3])  # Relative to root_dir

        if show_sizes:
            size = get_directory_size(exp_dir)
            total_size += size
            size_str = format_size(size)
            print(f"{i:3d}. {relative_path} ({size_str})")
        else:
            print(f"{i:3d}. {relative_path}")

    if show_sizes:
        print("-" * 80)
        print(f"Total size: {format_size(total_size)}")


def delete_experiment_dirs(experiment_dirs: list[pathlib.Path]):
    """Delete experiment directories."""
    if not experiment_dirs:
        print("No directories to delete.")
        return 0, 0

    successful = 0
    failed = 0

    for exp_dir in experiment_dirs:
        relative_path = exp_dir.relative_to(exp_dir.parents[3])  # Relative to root_dir

        try:
            shutil.rmtree(exp_dir)
            print(f"Deleted: {relative_path}")
            successful += 1
        except Exception as e:
            print(f"Failed to delete {relative_path}: {e}")
            failed += 1

    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Delete experiment output directories for all runs"
    )

    # Required argument
    parser.add_argument(
        "output_dir_name",
        help="Name of the experiment output directory to delete (e.g., 'gent', 'markov_gent', 'head_sampling_50')",
    )

    # Root directory
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./output",
        help="Directory containing the experiment outputs (default: ./output)",
    )

    # Action arguments
    parser.add_argument(
        "--list_only",
        action="store_true",
        help="Only list directories, don't delete them",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt before deleting",
    )

    # Add common filtering arguments
    add_common_arguments(parser)

    args = parser.parse_args()

    # Convert root_dir to Path
    root_dir = pathlib.Path(args.root_dir).resolve()

    if not root_dir.exists():
        print(f"Error: Root directory {root_dir} does not exist.")
        return 1

    # Display configuration
    print(f"Target experiment directory: '{args.output_dir_name}'")
    print(f"Root directory: {root_dir}")
    display_configuration(args.apps, args.services, args.faults, args.runs)

    # Find experiment directories
    experiment_dirs = find_experiment_dirs(
        root_dir, args.output_dir_name, args.apps, args.services, args.faults, args.runs
    )

    # List directories
    list_experiment_dirs(experiment_dirs, show_sizes=True)

    # If only listing, exit here
    if args.list_only:
        return 0

    if not experiment_dirs:
        return 0

    # Confirm deletion unless forced
    if not args.force:
        response = input(
            f"\nAre you sure you want to delete {len(experiment_dirs)} directories? [y/N]: "
        )
        if response.lower() not in ("y", "yes"):
            print("Deletion cancelled.")
            return 0

    # Delete directories
    print("\nDeleting experiment directories...")
    successful, failed = delete_experiment_dirs(experiment_dirs)

    print("\nResults:")
    print(f"  Successfully deleted: {successful}")
    if failed > 0:
        print(f"  Failed to delete: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
