import argparse
import asyncio
import pathlib

from .all_utils import ScriptProcessor, run_standard_processing


class NormalizeProcessor(ScriptProcessor):
    """Processor for normalize operations."""

    def __init__(self, root_dir: pathlib.Path, output_dir: pathlib.Path):
        super().__init__("normalize", root_dir)
        self.output_root = output_dir

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
        run_dir = self.root_dir / app_name / f"{service}_{fault}" / str(run)
        output_path = (
            self.output_root
            / app_name
            / f"{service}_{fault}"
            / str(run)
            / "original"
            / "dataset"
        )

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare script arguments
        script_args = ["--run_dir", str(run_dir), "-o", str(output_path)]

        print(f"Processing {run_dir}...")
        return await self.run_script(script_args, semaphore)


def add_normalize_arguments(parser: argparse.ArgumentParser):
    """Add normalize-specific arguments."""
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the normalized traces",
    )


def get_normalize_config(args):
    """Get normalize-specific configuration for display."""
    return {"Output Directory": args.output_dir}


async def normalize_task_factory(
    app_name: str,
    service: str,
    fault: str,
    run: int,
    semaphore: asyncio.Semaphore,
    args,
):
    """Factory function to create normalize processing tasks."""
    processor = NormalizeProcessor(
        pathlib.Path(args.root_dir), pathlib.Path(args.output_dir)
    )
    return await processor.process_combination(
        app_name, service, fault, run, semaphore, args
    )


def main():
    asyncio.run(
        run_standard_processing(
            description="Normalize all traces",
            task_factory=normalize_task_factory,
            additional_args_parser=add_normalize_arguments,
            extra_config_display=get_normalize_config,
            progress_description="Normalizing",
        )
    )
