import argparse
import asyncio
import pathlib

from tqdm import tqdm
from utils import run_dirs


async def head_sampling(
    dataset_dir: pathlib.Path,
    sampling_rate: int,
    output_dir: pathlib.Path,
    semaphore: asyncio.Semaphore,
):
    # if both compressed and dataset directories already exist, skip processing
    if (output_dir / "compressed").exists() and (output_dir / "dataset").exists():
        print(f"Skipping {dataset_dir} as it is already processed.")
        return True

    async with semaphore:
        try:
            print(f"Processing {dataset_dir}...")
            current_dir = pathlib.Path(__file__).parent
            script = current_dir / "head_sampling.py"
            process = await asyncio.create_subprocess_exec(
                "python3",
                str(script),
                "--dataset_dir",
                str(dataset_dir),
                "--sampling_rate",
                str(sampling_rate),
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
    argparser = argparse.ArgumentParser(description="Head sampling all dataset")
    argparser.add_argument(
        "--app",
        type=str,
        default=None,
        help="Application to run",
    )
    argparser.add_argument(
        "--root_dir", type=str, help="Directory containing the normalized dataset"
    )
    argparser.add_argument(
        "--sampling_rate",
        type=int,
        help="Sampling rate for head sampling",
    )
    argparser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="Maximum number of parallel processes (default: 12)",
    )
    args = argparser.parse_args()

    root_dir = pathlib.Path(args.root_dir)

    semaphore = asyncio.Semaphore(args.max_workers)

    tasks = []
    for app_name, service, fault, run in run_dirs(args.app):
        dataset_dir = root_dir.joinpath(
            app_name, f"{service}_{fault}", str(run), "original", "dataset"
        )
        output_dir = root_dir.joinpath(
            app_name,
            f"{service}_{fault}",
            str(run),
            f"head_sampling_{args.sampling_rate}",
        )
        tasks.append(
            head_sampling(dataset_dir, args.sampling_rate, output_dir, semaphore)
        )

    successful = 0
    failed = 0

    for task in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Head Sampling Processing"
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
