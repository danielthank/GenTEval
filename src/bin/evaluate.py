import argparse
import json
import os
import pathlib
import pickle
import sys

from logger import setup_logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
setup_logging()

from dataset import RCAEvalDataset  # noqa: E402
from evaluations import TraceRCAEvaluation  # noqa: E402

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Evaluate the compression algorithm"
    )
    argparser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory containing the dataset traces",
    )
    argparser.add_argument(
        "--labels_path",
        type=str,
        help="Path to the labels file containing the ground truth labels",
    )
    argparser.add_argument(
        "--evaluated_dir",
        "-o",
        type=str,
        help="Output file to save the evaluation results",
    )
    args = argparser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.labels_path = pathlib.Path(args.labels_path)
    args.evaluated_dir = pathlib.Path(args.evaluated_dir)

    original_dataset = RCAEvalDataset().load(args.dataset_dir)
    labels = pickle.load(open(args.labels_path, "rb"))

    results = TraceRCAEvaluation().evaluate(original_dataset, labels)
    args.evaluated_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.evaluated_dir / "results.json", "w"), indent=4)
