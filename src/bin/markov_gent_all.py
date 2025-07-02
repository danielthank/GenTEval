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

        # Prepare script arguments with all the MarkovGenT parameters
        script_args = [
            "--dataset_dir",
            str(dataset_dir),
            "-o",
            str(output_dir),
            "--start_time_latent_dim",
            str(args.start_time_latent_dim),
            "--start_time_epochs",
            str(args.start_time_epochs),
            "--markov_order",
            str(args.markov_order),
            "--max_depth",
            str(args.max_depth),
            "--metadata_epochs",
            str(args.metadata_epochs),
            "--metadata_hidden_dim",
            str(args.metadata_hidden_dim),
            "--batch_size",
            str(args.batch_size),
            "--learning_rate",
            str(args.learning_rate),
            "--num_processes",
            "1",
        ]

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
    parser.add_argument(
        "--start_time_latent_dim",
        type=int,
        default=16,
        help="Latent dimension for start time VAE (default: 16)",
    )
    parser.add_argument(
        "--start_time_epochs",
        type=int,
        default=10,
        help="Training epochs for start time VAE (default: 10)",
    )
    parser.add_argument(
        "--markov_order",
        type=int,
        default=1,
        help="Order of the Markov chain (default: 1)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Maximum depth for Markov states (default: 10)",
    )
    parser.add_argument(
        "--metadata_epochs",
        type=int,
        default=10,
        help="Training epochs for metadata neural network (default: 10)",
    )
    parser.add_argument(
        "--metadata_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for metadata neural network (default: 128)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for neural networks (default: 0.001)",
    )


def get_markov_gent_config(args):
    """Get MarkovGenT-specific configuration for display."""
    return {
        "Output Directory": args.output_dir_name,
        "Start Time Latent Dim": args.start_time_latent_dim,
        "Start Time Epochs": args.start_time_epochs,
        "Markov Order": args.markov_order,
        "Max Depth": args.max_depth,
        "Metadata Epochs": args.metadata_epochs,
        "Metadata Hidden Dim": args.metadata_hidden_dim,
        "Batch Size": args.batch_size,
        "Learning Rate": args.learning_rate,
    }


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
