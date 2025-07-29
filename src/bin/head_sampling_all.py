import argparse
import asyncio
import pathlib

from .all_utils import ScriptProcessor, run_standard_processing


class HeadSamplingProcessor(ScriptProcessor):
    """Processor for head sampling operations."""

    def __init__(self, root_dir: pathlib.Path, sampling_rate: int):
        super().__init__("head_sampling", root_dir)
        self.sampling_rate = sampling_rate

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
        dataset_dir = self.get_dataset_dir(app_name, service, fault, run)
        output_dir = self.get_output_dir(
            app_name, service, fault, run, f"head_sampling_{args.sampling_rate}"
        )

        # Skip if already processed
        if (output_dir / "compressed").exists() and (output_dir / "dataset").exists():
            print(f"Skipping {dataset_dir} as it is already processed.")
            return True

        # Prepare script arguments
        script_args = [
            "--dataset_dir",
            str(dataset_dir),
            "--sampling_rate",
            str(args.sampling_rate),
            "-o",
            str(output_dir),
        ]

        print(f"Processing {dataset_dir}...")
        return await self.run_script(script_args, semaphore)


def add_head_sampling_arguments(parser: argparse.ArgumentParser):
    """Add head sampling-specific arguments."""
    parser.add_argument(
        "--sampling_rate",
        type=int,
        required=True,
        help="Sampling rate for head sampling",
    )


def get_head_sampling_config(args):
    """Get head sampling-specific configuration for display."""
    return {"Sampling Rate": args.sampling_rate}


async def head_sampling_task_factory(
    app_name: str,
    service: str,
    fault: str,
    run: int,
    semaphore: asyncio.Semaphore,
    args,
):
    """Factory function to create head sampling processing tasks."""
    processor = HeadSamplingProcessor(pathlib.Path(args.root_dir), args.sampling_rate)
    return await processor.process_combination(
        app_name, service, fault, run, semaphore, args
    )


def main():
    asyncio.run(run_standard_processing(
        description="Run head sampling on all datasets",
        task_factory=head_sampling_task_factory,
        additional_args_parser=add_head_sampling_arguments,
        extra_config_display=get_head_sampling_config,
        progress_description="Head Sampling Processing",
    ))
