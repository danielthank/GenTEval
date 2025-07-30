import argparse
import pathlib

import wandb

from ..compressors import CompressedDataset
from ..compressors.markov_gen_t import (
    MarkovGenTCompressor,
    MarkovGenTConfig,
)
from ..dataset import RCAEvalDataset
from .logger import setup_logging


setup_logging()


def main():
    argparser = argparse.ArgumentParser(
        description="Compress and decompress traces using MarkovGenT"
    )
    argparser.add_argument(
        "--dataset_dir", type=str, help="Directory containing the preprocessed traces"
    )
    argparser.add_argument(
        "--gent_dir",
        "-o",
        type=str,
        help="Output file to save the compressed traces and recovered traces using MarkovGenT",
    )
    argparser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes for parallel processing (default: 8)",
    )

    # Wandb arguments
    argparser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (default: False)",
    )
    argparser.add_argument(
        "--wandb_project",
        type=str,
        default="GenT",
        help="Wandb project name (default: markov-gent-eval)",
    )
    argparser.add_argument(
        "--wandb_group",
        type=str,
        help="Wandb group name for organizing runs",
    )
    argparser.add_argument(
        "--wandb_name",
        type=str,
        help="Wandb run name",
    )
    argparser.add_argument(
        "--wandb_tags",
        nargs="*",
        help="Wandb tags for the run",
    )

    args = argparser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    gent_dir = pathlib.Path(args.gent_dir)
    dataset = RCAEvalDataset().load(dataset_dir)

    # Initialize wandb if enabled
    wandb_run = None
    if args.use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            tags=args.wandb_tags or ["markov_gent"],
            config={
                "num_processes": args.num_processes,
                "dataset_dir": str(dataset_dir),
                "output_dir": str(gent_dir),
            },
        )
        print(f"Wandb run initialized: {wandb_run.name if wandb_run else 'Failed'}")

    # Create MarkovGenT configuration with default values
    config = MarkovGenTConfig()

    print("Using MarkovGenT configuration with default values:")
    print(
        f"- Start Time VAE: latent_dim={config.start_time_latent_dim}, epochs={config.start_time_epochs}"
    )
    print(f"- Markov Chain: order={config.markov_order}, max_depth={config.max_depth}")
    print(
        f"- Metadata NN: hidden_dim={config.metadata_hidden_dim}, epochs={config.metadata_epochs}"
    )
    print(f"- Root Duration Model: {config.root_model.value}")
    print(f"- Training: batch_size={config.batch_size}, lr={config.learning_rate}")

    # Compress the dataset
    print("Initializing MarkovGenT compressor...")
    compressor = MarkovGenTCompressor(config, num_processes=args.num_processes)

    print("Compressing dataset...")
    compressed_dataset = compressor.compress(dataset)
    compressed_dataset.save(gent_dir / "compressed")
    print(f"Compressed dataset saved to {gent_dir / 'compressed'}")

    # Decompress the dataset
    print("Loading compressed dataset...")
    compressed_dataset = CompressedDataset.load(gent_dir / "compressed")

    print("Decompressing dataset (generating new traces)...")
    recovered_dataset = compressor.decompress(compressed_dataset)
    recovered_dataset = RCAEvalDataset.from_dataset(recovered_dataset)
    recovered_dataset.save(gent_dir / "dataset")
    print(f"Generated dataset saved to {gent_dir / 'dataset'}")

    print("MarkovGenT compression and generation completed successfully!")

    # Log final results to wandb
    if wandb_run:
        wandb.finish()
        print("Wandb run finished.")
