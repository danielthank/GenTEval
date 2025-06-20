import argparse
import asyncio
import pathlib

from tqdm import tqdm

from .utils import run_dirs


async def normalize(
    run_dir: pathlib.Path, output_path: pathlib.Path, semaphore: asyncio.Semaphore
):
    async with semaphore:
        try:
            current_dir = pathlib.Path(__file__).parent
            normalize_script = current_dir / "normalize.py"
            process = await asyncio.create_subprocess_exec(
                "python3",
                str(normalize_script),
                "--run_dir",
                str(run_dir),
                "-o",
                str(output_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if stdout:
                print(stdout.decode())

            if process.returncode != 0:
                print(f"Error processing {run_dir}: {stderr.decode()}")
                return False
            return True
        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            return False


async def main():
    argparser = argparse.ArgumentParser(description="Normalize all traces")
    argparser.add_argument(
        "--app",
        type=str,
        default=None,
        help="Application to run",
    )
    argparser.add_argument(
        "--root_dir", type=str, help="Directory containing the trace data"
    )
    argparser.add_argument(
        "--output_dir", type=str, help="Directory to save the normalized traces"
    )
    argparser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="Maximum number of parallel processes (default: 12)",
    )
    args = argparser.parse_args()

    root_dir = pathlib.Path(args.root_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.max_workers)

    tasks = []
    for app_name, service, fault, run in run_dirs(args.app):
        run_dir = root_dir.joinpath(app_name, f"{service}_{fault}", str(run))
        output_path = output_dir.joinpath(
            app_name, f"{service}_{fault}", str(run), "original", "dataset"
        )
        tasks.append(normalize(run_dir, output_path, semaphore))

    successful = 0
    failed = 0

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Normalizing"):
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
