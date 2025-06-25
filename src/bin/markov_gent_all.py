import argparse
import asyncio
import pathlib

from tqdm import tqdm
from utils import run_dirs


async def markov_gent(
    dataset_dir: pathlib.Path,
    output_dir: pathlib.Path,
    force: bool,
    semaphore: asyncio.Semaphore,
    start_time_latent_dim: int = 16,
    start_time_epochs: int = 10,
    markov_order: int = 1,
    max_depth: int = 10,
    metadata_epochs: int = 10,
    metadata_hidden_dim: int = 128,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    if (
        not force
        and (output_dir / "compressed").exists()
        and (output_dir / "dataset").exists()
    ):
        print(f"Skipping {dataset_dir} as it is already processed.")
        return True

    async with semaphore:
        try:
            print(f"Processing {dataset_dir}...")
            current_dir = pathlib.Path(__file__).parent
            script = current_dir / "markov_gent.py"
            process = await asyncio.create_subprocess_exec(
                "python3",
                str(script),
                "--dataset_dir",
                str(dataset_dir),
                "-o",
                str(output_dir),
                "--start_time_latent_dim",
                str(start_time_latent_dim),
                "--start_time_epochs",
                str(start_time_epochs),
                "--markov_order",
                str(markov_order),
                "--max_depth",
                str(max_depth),
                "--metadata_epochs",
                str(metadata_epochs),
                "--metadata_hidden_dim",
                str(metadata_hidden_dim),
                "--batch_size",
                str(batch_size),
                "--learning_rate",
                str(learning_rate),
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
    argparser = argparse.ArgumentParser(description="Run MarkovGenT on all traces")
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
        "--output_dir_name",
        type=str,
        default="markov_gent",
        help="Output directory name (default: markov_gent)",
    )
    argparser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of parallel processes (default: 4)",
    )
    argparser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reprocessing even if output directories already exist",
    )
    argparser.add_argument(
        "--start_time_latent_dim",
        type=int,
        default=16,
        help="Latent dimension for start time VAE (default: 16)",
    )
    argparser.add_argument(
        "--start_time_epochs",
        type=int,
        default=10,
        help="Training epochs for start time VAE (default: 10)",
    )
    argparser.add_argument(
        "--markov_order",
        type=int,
        default=1,
        help="Order of the Markov chain (default: 1)",
    )
    argparser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Maximum depth for Markov states (default: 10)",
    )
    argparser.add_argument(
        "--metadata_epochs",
        type=int,
        default=10,
        help="Training epochs for metadata neural network (default: 10)",
    )
    argparser.add_argument(
        "--metadata_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for metadata neural network (default: 128)",
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for neural networks (default: 0.001)",
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
            app_name, f"{service}_{fault}", str(run), args.output_dir_name
        )
        tasks.append(
            markov_gent(
                dataset_dir,
                output_dir,
                args.force,
                semaphore,
                args.start_time_latent_dim,
                args.start_time_epochs,
                args.markov_order,
                args.max_depth,
                args.metadata_epochs,
                args.metadata_hidden_dim,
                args.batch_size,
                args.learning_rate,
            )
        )

    successful = 0
    failed = 0

    for task in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="MarkovGenT Processing"
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
