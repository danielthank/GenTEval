"""
Template for scripts that need custom processing logic.
This shows how to use all_utils for scripts that don't fit the standard pattern.
"""

import asyncio
import pathlib

from tqdm import tqdm

from .all_utils import (
    create_standard_parser,
    display_configuration,
)
from .utils import get_dir_with_root


async def evaluate_task(
    app_name: str,
    service: str,
    fault: str,
    run: int,
    compressor: str,
    evaluator: str,
    labels_path: pathlib.Path,
    root_dir: pathlib.Path,
    force: bool,
    semaphore: asyncio.Semaphore,
):
    """Evaluate a single dataset with a specific compressor and evaluator."""
    dataset_dir = (
        get_dir_with_root(root_dir, app_name, service, fault, run)
        / compressor
        / "dataset"
    )
    evaluated_dir = (
        get_dir_with_root(root_dir, app_name, service, fault, run)
        / compressor
        / "evaluated"
    )
    result_file = evaluated_dir / f"{evaluator}_results.json"

    if not force and result_file.exists():
        print(f"Skipping {dataset_dir} - {evaluator} as it is already processed.")
        return True

    async with semaphore:
        try:
            print(f"Processing {dataset_dir} with {evaluator} evaluator...")
            # Get the project root directory (two levels up from this file)
            project_root = pathlib.Path(__file__).parent.parent.parent

            process = await asyncio.create_subprocess_exec(
                "uv",
                "run",
                "evaluate",
                "--dataset_dir",
                str(dataset_dir),
                "--labels_path",
                str(labels_path),
                "--evaluator",
                evaluator,
                "-o",
                str(evaluated_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(project_root),
            )

            # Stream output
            async def stream_output(stream):
                async for line in stream:
                    print(line.decode().strip(), flush=True)

            stdout_task = asyncio.create_task(stream_output(process.stdout))
            stderr_task = asyncio.create_task(stream_output(process.stderr))

            await asyncio.gather(process.wait(), stdout_task, stderr_task)

            if process.returncode != 0:
                print(
                    f"Error processing {dataset_dir}: Process returned {process.returncode}"
                )
                return False
            return True

        except Exception as e:
            print(f"Error processing {dataset_dir}: {e}")
            return False


async def evaluate_all():
    # Custom argument parsing for evaluate_all
    parser = create_standard_parser("Evaluate all datasets")
    parser.add_argument(
        "--compressors",
        type=str,
        nargs="+",
        required=True,
        help="List of compressors to evaluate",
    )
    parser.add_argument(
        "--evaluators",
        type=str,
        nargs="*",
        default=[
            "duration_over_time",
            "error_over_time",
            "operation",
            "trace_rca",
            "micro_rank",
            "size",
            "span_count",
            "time",
            "rate_over_time",
            "graph",
        ],
        help="Evaluators to run (default: all evaluators)",
    )
    args = parser.parse_args()

    # Parse filtering arguments
    applications, services, faults, runs = (
        args.apps,
        args.services,
        args.faults,
        args.runs,
    )

    # Display configuration with custom fields
    extra_config = {"Compressors": args.compressors, "Evaluators": args.evaluators}
    display_configuration(applications, services, faults, runs, extra_config)

    # Custom processing logic
    root_dir = pathlib.Path(args.root_dir)
    semaphore = asyncio.Semaphore(args.max_workers)
    tasks = []

    # Generate all combinations including compressors and evaluators
    from .utils import run_dirs

    for app_name, service, fault, run in run_dirs(applications, services, faults, runs):
        for compressor in args.compressors:
            # Filter valid compressors
            if (
                compressor not in ["original"]
                and not compressor.startswith("head_sampling")
                and not compressor.startswith("gent")
                and not compressor.startswith("markov_gent")
                and not compressor.startswith("simple_gent")
            ):
                continue

            labels_path = (
                get_dir_with_root(root_dir, app_name, service, fault, run)
                / "original"
                / "dataset"
                / "labels.pkl"
            )

            for evaluator in args.evaluators:
                task = evaluate_task(
                    app_name,
                    service,
                    fault,
                    run,
                    compressor,
                    evaluator,
                    labels_path,
                    root_dir,
                    args.force,
                    semaphore,
                )
                tasks.append(task)

    # Process all tasks
    if not tasks:
        print("No combinations found to process.")
        return

    print(f"Processing {len(tasks)} combinations...")

    successful = 0
    failed = 0

    for task in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Evaluation Processing"
    ):
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


def main():
    asyncio.run(evaluate_all())
