import argparse
import os
import pathlib
import sys

from logger import setup_logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
setup_logging()

from compressors import CompressedDataset  # noqa: E402
from compressors.markov_gen_t import (  # noqa: E402
    MarkovGenTCompressor,
    MarkovGenTConfig,
)
from dataset import RCAEvalDataset  # noqa: E402

if __name__ == "__main__":
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
        "--start_time_latent_dim",
        type=int,
        default=16,
        help="Latent dimension for start time VAE (default: 16)",
    )
    argparser.add_argument(
        "--start_time_epochs",
        type=int,
        default=10,
        help="Training epochs for start time VAE (default: 100)",
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
        help="Training epochs for metadata neural network (default: 150)",
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

    dataset_dir = pathlib.Path(args.dataset_dir)
    gent_dir = pathlib.Path(args.gent_dir)
    dataset = RCAEvalDataset().load(dataset_dir)

    # Create MarkovGenT configuration
    config = MarkovGenTConfig(
        start_time_latent_dim=args.start_time_latent_dim,
        start_time_epochs=args.start_time_epochs,
        markov_order=args.markov_order,
        max_depth=args.max_depth,
        metadata_epochs=args.metadata_epochs,
        metadata_hidden_dim=args.metadata_hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print("Using MarkovGenT configuration:")
    print(
        f"- Start Time VAE: latent_dim={config.start_time_latent_dim}, epochs={config.start_time_epochs}"
    )
    print(f"- Markov Chain: order={config.markov_order}, max_depth={config.max_depth}")
    print(
        f"- Metadata NN: hidden_dim={config.metadata_hidden_dim}, epochs={config.metadata_epochs}"
    )
    print(f"- Training: batch_size={config.batch_size}, lr={config.learning_rate}")

    # Compress the dataset
    print("Initializing MarkovGenT compressor...")
    compressor = MarkovGenTCompressor(config)

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
