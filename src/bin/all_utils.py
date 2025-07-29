"""
Common utilities for _all.py scripts.
Provides reusable functions for argument parsing, configuration display,
and async processing patterns.
"""

import argparse
import asyncio
import pathlib
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from .utils import run_dirs


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common filtering arguments to an argument parser."""
    parser.add_argument(
        "--apps",
        nargs="*",
        help="List of applications (e.g., 'RE2-OB RE2-TT'). Default: all",
    )
    parser.add_argument(
        "--services",
        nargs="*",
        help="List of services to process (e.g., 'checkoutservice emailservice')",
    )
    parser.add_argument(
        "--faults",
        nargs="*",
        help="List of faults to process (e.g., 'cpu mem disk')",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="*",
        help="List of run numbers to process (e.g., '1 2 3')",
    )


def display_configuration(
    applications, services, faults, runs, extra_config: Optional[Dict[str, Any]] = None
):
    """Display the current configuration."""
    print("Configuration:")
    print(f"  Applications: {applications or 'ALL'}")
    print(f"  Services: {services or 'ALL'}")
    print(f"  Faults: {faults or 'ALL'}")
    print(f"  Runs: {runs or 'ALL'}")

    if extra_config:
        for key, value in extra_config.items():
            print(f"  {key}: {value}")
    print()


async def process_all_combinations(
    applications: Optional[List[str]],
    services: Optional[List[str]],
    faults: Optional[List[str]],
    runs: Optional[List[int]],
    task_factory: Callable,
    max_workers: int = 4,
    description: str = "Processing",
) -> tuple[int, int]:
    """
    Process all combinations using the provided task factory.

    Args:
        applications: List of applications to process
        services: List of services to process
        faults: List of faults to process
        runs: List of runs to process
        task_factory: Function that creates a task for each combination
        max_workers: Maximum number of concurrent workers
        description: Description for the progress bar

    Returns:
        Tuple of (successful_count, failed_count)
    """
    semaphore = asyncio.Semaphore(max_workers)

    # Create all tasks
    tasks = []
    for app_name, service, fault, run in run_dirs(applications, services, faults, runs):
        task = task_factory(app_name, service, fault, run, semaphore)
        tasks.append(task)

    if not tasks:
        print("No combinations found to process.")
        return 0, 0

    print(f"Processing {len(tasks)} combinations...")

    # Process tasks with progress bar
    successful = 0
    failed = 0

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=description):
        try:
            success = await task
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"Error during processing: {e}")

    print(f"Processing complete. Successful: {successful}, Failed: {failed}")
    return successful, failed


class ScriptProcessor:
    """Base class for script processors that run external scripts."""

    def __init__(self, module_name: str, root_dir: pathlib.Path):
        self.module_name = module_name
        self.root_dir = root_dir

    async def run_script(self, args: List[str], semaphore: asyncio.Semaphore) -> bool:
        """Run the script with given arguments."""
        async with semaphore:
            try:
                # Get the project root directory (two levels up from this file)
                project_root = pathlib.Path(__file__).parent.parent.parent

                process = await asyncio.create_subprocess_exec(
                    "uv",
                    "run",
                    self.module_name,
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(project_root),
                )

                async def stream_output(stream):
                    while True:
                        chunk = await stream.read(1024)  # Read in 1KB chunks
                        if not chunk:
                            break
                        print(
                            chunk.decode("utf-8", errors="ignore"), end="", flush=True
                        )

                stdout_task = asyncio.create_task(stream_output(process.stdout))
                stderr_task = asyncio.create_task(stream_output(process.stderr))

                await asyncio.gather(process.wait(), stdout_task, stderr_task)

                if process.returncode != 0:
                    print(f"Process returned {process.returncode}")
                    return False
                return True

            except Exception as e:
                print(f"Error running script: {e}")
                return False

    def get_dataset_dir(
        self,
        app_name: str,
        service: str,
        fault: str,
        run: int,
        subdir: str = "original",
    ) -> pathlib.Path:
        """Get the dataset directory path."""
        return (
            self.root_dir
            / app_name
            / f"{service}_{fault}"
            / str(run)
            / subdir
            / "dataset"
        )

    def get_output_dir(
        self, app_name: str, service: str, fault: str, run: int, output_name: str
    ) -> pathlib.Path:
        """Get the output directory path."""
        return self.root_dir / app_name / f"{service}_{fault}" / str(run) / output_name


def create_standard_parser(
    description: str, additional_args: Optional[Callable] = None
) -> argparse.ArgumentParser:
    """Create a standard argument parser with common arguments."""
    parser = argparse.ArgumentParser(description=description)

    # Common arguments
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./output",
        help="Directory containing the dataset (default: ./output)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="Maximum number of parallel processes (default: 12)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reprocessing even if output directories already exist",
    )

    # Add filtering arguments
    add_common_arguments(parser)

    # Add any additional arguments
    if additional_args:
        additional_args(parser)

    return parser


async def run_standard_processing(
    description: str,
    task_factory: Callable,
    additional_args_parser: Optional[Callable] = None,
    extra_config_display: Optional[Callable] = None,
    progress_description: str = "Processing",
):
    """
    Standard processing pipeline for _all.py scripts.

    Args:
        description: Description for the argument parser
        task_factory: Function that creates tasks for processing
        additional_args_parser: Optional function to add custom arguments
        extra_config_display: Optional function to get extra config for display
        progress_description: Description for the progress bar
    """
    # Create parser and parse arguments
    parser = create_standard_parser(description, additional_args_parser)
    args = parser.parse_args()

    # Extract filtering arguments
    applications, services, faults, runs = (
        args.apps,
        args.services,
        args.faults,
        args.runs,
    )

    # Display configuration
    extra_config = extra_config_display(args) if extra_config_display else None
    display_configuration(applications, services, faults, runs, extra_config)

    # Process all combinations
    successful, failed = await process_all_combinations(
        applications,
        services,
        faults,
        runs,
        lambda app, service, fault, run, sem: task_factory(
            app, service, fault, run, sem, args
        ),
        args.max_workers,
        progress_description,
    )

    return successful, failed
