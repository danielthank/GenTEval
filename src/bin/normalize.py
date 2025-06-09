import argparse
import os
import pathlib
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.rca_eval_dataset import RCAEvalDataset  # noqa: E402

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Normalize trace data")
    argparser.add_argument(
        "--run_dir",
        type=str,
        help="Directory containing the trace data for a run",
    )
    argparser.add_argument(
        "--dataset_dir",
        "-o",
        type=str,
    )
    args = argparser.parse_args()

    dataset = RCAEvalDataset(pathlib.Path(args.run_dir))
    dataset.save(pathlib.Path(args.dataset_dir))
