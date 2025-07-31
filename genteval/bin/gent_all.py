import argparse
import asyncio
import pathlib

from .all_utils import ScriptProcessor, run_standard_processing


class GentProcessor(ScriptProcessor):
    """Processor for GenT operations."""

    def __init__(self, root_dir: pathlib.Path, output_dir_name: str = "gent"):
        super().__init__("gent", root_dir)
        self.output_dir_name = output_dir_name

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
            app_name, service, fault, run, args.output_dir_name
        )

        # Skip if already processed (unless forced)
        if (
            not args.force
            and (output_dir / "compressed").exists()
            and (output_dir / "dataset").exists()
        ):
            print(f"Skipping {dataset_dir} as it is already processed.")
            return True

        # Prepare script arguments
        script_args = ["--dataset_dir", str(dataset_dir), "-o", str(output_dir)]

        print(f"Processing {dataset_dir}...")
        return await self.run_script(script_args, semaphore)


def add_gent_arguments(parser: argparse.ArgumentParser):
    """Add GenT-specific arguments."""
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="gent",
        help="Output directory name (default: gent)",
    )


def get_gent_config(args):
    """Get GenT-specific configuration for display."""
    return {"Output Directory": args.output_dir_name}


async def gent_task_factory(
    app_name: str,
    service: str,
    fault: str,
    run: int,
    semaphore: asyncio.Semaphore,
    args,
):
    """Factory function to create GenT processing tasks."""
    processor = GentProcessor(pathlib.Path(args.root_dir), args.output_dir_name)
    return await processor.process_combination(
        app_name, service, fault, run, semaphore, args
    )


def main():
    asyncio.run(
        run_standard_processing(
            description="Run GenT compression on all traces",
            task_factory=gent_task_factory,
            additional_args_parser=add_gent_arguments,
            extra_config_display=get_gent_config,
            progress_description="GenT Processing",
        )
    )
