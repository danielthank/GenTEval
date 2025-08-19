import argparse
import pathlib

from genteval.bin.logger import setup_logging
from genteval.compressors import CompressedDataset
from genteval.compressors.simple_gent import SimpleGenTCompressor, SimpleGenTConfig
from genteval.dataset import RCAEvalDataset


setup_logging()


def main():
    argparser = argparse.ArgumentParser(
        description="Compress and decompress traces using SimpleGenT"
    )
    argparser.add_argument(
        "--dataset_dir", type=str, help="Directory containing the preprocessed traces"
    )
    argparser.add_argument(
        "--gent_dir",
        "-o",
        type=str,
        help="Output file to save the compressed traces and recovered traces using SimpleGenT",
    )
    argparser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes for parallel processing (default: 8)",
    )

    args = argparser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    gent_dir = pathlib.Path(args.gent_dir)
    dataset = RCAEvalDataset().load(dataset_dir)

    # Create SimpleGenT configuration with default values
    config = SimpleGenTConfig()

    print("Using SimpleGenT configuration with default values:")
    print(f"- Time bucket duration: {config.time_bucket_duration_us} microseconds")
    print(f"- Min samples for GMM: {config.min_samples_for_gmm}")
    print(f"- Max GMM components: {config.max_gmm_components}")
    print(f"- Max depth: {config.max_depth}")
    print(f"- Max children: {config.max_children}")
    print(f"- Random seed: {config.random_seed}")

    # Compress the dataset
    print("Initializing SimpleGenT compressor...")
    compressor = SimpleGenTCompressor(config)

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

    print("SimpleGenT compression and generation completed successfully!")


if __name__ == "__main__":
    main()
