import argparse
import os
import pathlib
import sys

from logger import setup_logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
setup_logging()

from dataset import RCAEvalDataset  # noqa: E402

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Evaluate the compression algorithm"
    )
    argparser.add_argument(
        "--process_dir",
        type=str,
        help="Directory containing the compressed traces and recovered traces",
    )
    argparser.add_argument(
        "--evaluation_dir",
        "-o",
        type=str,
        help="Output file to save the evaluation results",
    )
    args = argparser.parse_args()

    args.process_dir = pathlib.Path(args.process_dir)
    args.evaluation_dir = pathlib.Path(args.evaluation_dir)

    recovered_dataset = RCAEvalDataset().load(args.process_dir / "recovered")
