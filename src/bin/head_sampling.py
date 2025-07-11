import argparse
import pathlib

from ..compressors import CompressedDataset, HeadSamplingCompressor
from ..dataset import RCAEvalDataset
from .logger import setup_logging

setup_logging()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Compress and decompress traces")
    argparser.add_argument(
        "--dataset_dir", type=str, help="Directory containing the preprocessed traces"
    )
    argparser.add_argument(
        "--sampling_rate",
        type=int,
        default=150,
        help="Sampling rate for head sampling (default: 150)",
    )
    argparser.add_argument(
        "--head_sampling_dir",
        "-o",
        type=str,
        help="Output file to save the compressed traces and recovered traces using head sampling",
    )
    args = argparser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    head_sampling_dir = pathlib.Path(args.head_sampling_dir)
    dataset = RCAEvalDataset().load(dataset_dir)

    # Compress the dataset
    compressor = HeadSamplingCompressor(args.sampling_rate)
    compressed_dataset = compressor.compress(dataset)
    compressed_dataset.save(head_sampling_dir / "compressed")

    # Decompress the dataset
    compressed_dataset = CompressedDataset.load(head_sampling_dir / "compressed")
    recovered_dataset = compressor.decompress(compressed_dataset)
    recovered_dataset = RCAEvalDataset.from_dataset(recovered_dataset)
    recovered_dataset.save(head_sampling_dir / "dataset")
