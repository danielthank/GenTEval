import argparse
import os
import pathlib
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from compressors import (  # noqa: E402
    GenTCompressor,
    GenTConfig,
)
from dataset import RCAEvalDataset  # noqa: E402

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Compress and decompress traces")
    argparser.add_argument(
        "--dataset_dir", type=str, help="Directory containing the preprocessed traces"
    )
    argparser.add_argument(
        "--process_dir",
        "-o",
        type=str,
        help="Output file to save the compressed traces and recovered traces",
    )
    args = argparser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.process_dir = pathlib.Path(args.process_dir)
    dataset = RCAEvalDataset().load(args.dataset_dir)
    config = GenTConfig()
    compressor = GenTCompressor(config)
    compressed_dataset = compressor.compress(dataset)
    compressed_dataset.save(args.process_dir / "compressed")
    recovered_dataset = compressor.decompress(compressed_dataset)
    recovered_dataset.save(args.process_dir / "recovered")
