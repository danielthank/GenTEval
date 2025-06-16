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
from evaluators import (  # noqa: E402
    DurationEvaluator,
    OperationEvaluator,
    TraceRCAEvaluator,
)

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
        "--evaluators",
        type=str,
        nargs="+",
    )
    argparser.add_argument(
        "--evaluated_dir",
        "-o",
        type=str,
        help="Output file to save the evaluation results",
    )
    args = argparser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    labels_path = pathlib.Path(args.labels_path)
    evaluated_dir = pathlib.Path(args.evaluated_dir)

    dataset = RCAEvalDataset().load(dataset_dir)
    labels = pickle.load(open(labels_path, "rb"))

    evaluated_dir.mkdir(parents=True, exist_ok=True)
    if "trace_rca" in args.evaluators:
        results = TraceRCAEvaluator().evaluate(dataset, labels)
        json.dump(
            results,
            open(evaluated_dir / "trace_rca_results.json", "w"),
        )
    if "duration" in args.evaluators:
        results = DurationEvaluator().evaluate(dataset, labels)
        json.dump(
            results,
            open(evaluated_dir / "duration_results.json", "w"),
        )
    if "operation" in args.evaluators:
        results = OperationEvaluator().evaluate(dataset, labels)
        json.dump(
            results,
            open(evaluated_dir / "operation_results.json", "w"),
        )
