import argparse
import asyncio
import pathlib

from genteval.bin.all_utils import ScriptProcessor, run_standard_processing


class SimpleGentProcessor(ScriptProcessor):
    """Processor for SimpleGenT operations."""

    def __init__(self, root_dir: pathlib.Path):
        super().__init__("simple_gent", root_dir)

    async def process_combination(
        self,
        app_name: str,
        service: str,
        fault: str,
        run: int,
        semaphore: asyncio.Semaphore,
        args,
    ) -> bool:
        """Process a single app/service/fault/run combination."""
        dataset_dir = (
            self.get_dir(app_name, service, fault, run) / "original" / "dataset"
        )

        # Process all iterations
        all_success = True
        for iteration in range(1, args.iterations + 1):
            output_dir = (
                self.get_dir(app_name, service, fault, run)
                / f"{args.output_dir_prefix}_{iteration}"
            )

            # Skip if already processed (unless forced)
            if (
                not args.force
                and (output_dir / "compressed").exists()
                and (output_dir / "dataset").exists()
            ):
                print(f"Skipping {output_dir} as it is already processed.")
                continue

            # Prepare script arguments
            script_args = [
                "--dataset_dir",
                str(dataset_dir),
                "-o",
                str(output_dir),
                "--num_processes",
                "12",
            ]

            print(f"Processing {output_dir}...")
            success = await self.run_script(script_args, semaphore)
            if not success:
                all_success = False

        return all_success


def add_simple_gent_arguments(parser: argparse.ArgumentParser):
    """Add SimpleGenT-specific arguments."""
    parser.add_argument(
        "--output_dir_prefix",
        type=str,
        default="gent_experiment",
        help="Output directory prefix (default: gent_experiment)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to run (default: 3)",
    )


def get_simple_gent_config(args):
    """Get SimpleGenT-specific configuration for display."""
    config = {
        "Output Directory Prefix": args.output_dir_prefix,
        "Iterations": args.iterations,
    }
    return config


async def simple_gent_task_factory(
    app_name: str,
    service: str,
    fault: str | None,
    run: int,
    semaphore: asyncio.Semaphore,
    args,
):
    """Factory function to create SimpleGenT processing tasks."""
    processor = SimpleGentProcessor(pathlib.Path(args.root_dir))
    return await processor.process_combination(
        app_name, service, fault, run, semaphore, args
    )


def main():
    asyncio.run(
        run_standard_processing(
            description="Run SimpleGenT on all traces",
            task_factory=simple_gent_task_factory,
            additional_args_parser=add_simple_gent_arguments,
            extra_config_display=get_simple_gent_config,
            progress_description="SimpleGenT Processing",
        )
    )


if __name__ == "__main__":
    main()