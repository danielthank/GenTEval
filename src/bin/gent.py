import argparse
import pathlib

from ..compressors import CompressedDataset, GenTCompressor, GenTConfig
from ..dataset import RCAEvalDataset
from .logger import setup_logging


setup_logging()


def main():
    argparser = argparse.ArgumentParser(description="Compress and decompress traces")
    argparser.add_argument(
        "--dataset_dir", type=str, help="Directory containing the preprocessed traces"
    )
    argparser.add_argument(
        "--gent_dir",
        "-o",
        type=str,
        help="Output file to save the compressed traces and recovered traces using GenT",
    )
    args = argparser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    gent_dir = pathlib.Path(args.gent_dir)
    dataset = RCAEvalDataset().load(dataset_dir)

    # Compress the dataset
    config = GenTConfig()
    compressor = GenTCompressor(config)
    compressed_dataset = compressor.compress(dataset)
    compressed_dataset.save(gent_dir / "compressed")

    # Decompress the dataset
    compressed_dataset = CompressedDataset.load(gent_dir / "compressed")
    recovered_dataset = compressor.decompress(compressed_dataset)
    recovered_dataset = RCAEvalDataset.from_dataset(recovered_dataset)
    recovered_dataset.save(gent_dir / "dataset")
