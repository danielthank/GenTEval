import argparse
import asyncio
import pathlib

from tqdm import tqdm
from utils import run_dirs


async def evaluate(
    dataset_dir: pathlib.Path,
    labels_path: pathlib.Path,
    evaluator: str,
    output_dir: pathlib.Path,
    force: bool,
    semaphore: asyncio.Semaphore,
):
    result_file = output_dir / f"{evaluator}_results.json"
    if not force and result_file.exists():
        print(f"Skipping {dataset_dir} - {evaluator} as it is already processed.")
        return True

    async with semaphore:
        try:
            print(f"Processing {dataset_dir} with {evaluator} evaluator...")
            current_dir = pathlib.Path(__file__).parent
            script = current_dir / "evaluate.py"
            process = await asyncio.create_subprocess_exec(
                "python3",
                str(script),
                "--dataset_dir",
                str(dataset_dir),
                "--labels_path",
                str(labels_path),
                "--evaluator",
                evaluator,
                "-o",
                str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

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


async def main():
    argparser = argparse.ArgumentParser(description="Evaluate all datasets")
    argparser.add_argument(
        "--app",
        type=str,
        default=None,
        help="Application to run",
    )
    argparser.add_argument(
        "--root_dir", type=str, help="Directory containing the normalized dataset"
    )
    # array of compressors to use
    argparser.add_argument(
        "--compressors",
        type=str,
        nargs="+",
        help="List of compressors to evaluate",
    )
    argparser.add_argument(
        "--evaluators",
        type=str,
        nargs="+",
    )
    argparser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="Maximum number of parallel processes (default: 12)",
    )
    argparser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-evaluation of all datasets, even if output directories already exist",
    )
    args = argparser.parse_args()

    root_dir = pathlib.Path(args.root_dir)

    semaphore = asyncio.Semaphore(args.max_workers)

    tasks = []
    for app_name, service, fault, run in run_dirs(args.app):
        for compressor in args.compressors:
            if (
                compressor not in ["original"]
                and not compressor.startswith("head_sampling")
                and not compressor.startswith("gent")
            ):
                continue
            dataset_dir = root_dir.joinpath(
                app_name, f"{service}_{fault}", str(run), compressor, "dataset"
            )
            labels_path = root_dir.joinpath(
                app_name,
                f"{service}_{fault}",
                str(run),
                "original",
                "dataset",
                "labels.pkl",
            )
            evaluated_dir = root_dir.joinpath(
                app_name, f"{service}_{fault}", str(run), compressor, "evaluated"
            )

            for evaluator in args.evaluators:
                tasks.append(
                    evaluate(
                        dataset_dir,
                        labels_path,
                        evaluator,
                        evaluated_dir,
                        args.force,
                        semaphore,
                    )
                )

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


if __name__ == "__main__":
    asyncio.run(main())
