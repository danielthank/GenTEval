import argparse
import asyncio
import pathlib

from all_utils import ScriptProcessor, run_standard_processing


class MarkovGentProcessor(ScriptProcessor):
    """Processor for MarkovGenT operations."""

    def __init__(self, root_dir: pathlib.Path, output_dir_name: str = "markov_gent"):
        super().__init__("markov_gent.py", root_dir)
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
        script_args = [
            "--dataset_dir",
            str(dataset_dir),
            "-o",
            str(output_dir),
            "--num_processes",
            "12",
        ]

        # Add wandb configuration if enabled
        if hasattr(args, 'use_wandb') and args.use_wandb:
            script_args.extend([
                "--use_wandb",
                "--wandb_project", args.wandb_project,
                "--wandb_group", f"{app_name}_{service}_{fault}",
                "--wandb_name", f"{app_name}_{service}_{fault}_run{run}",
            ])
            if args.wandb_tags:
                script_args.extend(["--wandb_tags"] + args.wandb_tags)

        print(f"Processing {dataset_dir}...")
        return await self.run_script(script_args, semaphore)


def add_markov_gent_arguments(parser: argparse.ArgumentParser):
    """Add MarkovGenT-specific arguments."""
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="markov_gent",
        help="Output directory name (default: markov_gent)",
    )
    
    # Wandb arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (default: False)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="markov-gent-eval",
        help="Wandb project name (default: markov-gent-eval)",
    )
    parser.add_argument(
        "--wandb_tags",
        nargs="*",
        help="Wandb tags for the run (e.g., 'experiment1 baseline')",
    )
    parser.add_argument(
        "--wandb_notes",
        type=str,
        help="Notes for the wandb run",
    )


def get_markov_gent_config(args):
    """Get MarkovGenT-specific configuration for display."""
    config = {
        "Output Directory": args.output_dir_name,
    }
    
    # Add wandb configuration if enabled
    if hasattr(args, 'use_wandb') and args.use_wandb:
        config.update({
            "Wandb Enabled": True,
            "Wandb Project": args.wandb_project,
            "Wandb Tags": args.wandb_tags or "None",
            "Wandb Notes": args.wandb_notes or "None",
        })
    
    return config


async def markov_gent_task_factory(
    app_name: str,
    service: str,
    fault: str,
    run: int,
    semaphore: asyncio.Semaphore,
    args,
):
    """Factory function to create MarkovGenT processing tasks."""
    processor = MarkovGentProcessor(pathlib.Path(args.root_dir), args.output_dir_name)
    return await processor.process_combination(
        app_name, service, fault, run, semaphore, args
    )


async def main():
    await run_standard_processing(
        description="Run MarkovGenT on all traces",
        task_factory=markov_gent_task_factory,
        additional_args_parser=add_markov_gent_arguments,
        extra_config_display=get_markov_gent_config,
        progress_description="MarkovGenT Processing",
    )


if __name__ == "__main__":
    asyncio.run(main())
