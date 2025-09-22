import argparse
import json
import pathlib
import pickle

from genteval.dataset import RCAEvalDataset
from genteval.evaluators import (
    DurationEvaluator,
    MicroRankEvaluator,
    OperationEvaluator,
    SpanCountEvaluator,
    TimeEvaluator,
    TraceRCAEvaluator,
)

from .logger import setup_logging


setup_logging()


def main():
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
        "--evaluator",
        type=str,
        help="Single evaluator to run (duration, operation, trace_rca, micro_rank, span_count)",
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
    if args.evaluator == "duration":
        results = DurationEvaluator().evaluate(dataset, labels)
        json.dump(
            results,
            open(evaluated_dir / "duration_results.json", "w"),
        )
    elif args.evaluator == "operation":
        results = OperationEvaluator().evaluate(dataset, labels)
        json.dump(
            results,
            open(evaluated_dir / "operation_results.json", "w"),
        )
    elif args.evaluator == "trace_rca":
        results = TraceRCAEvaluator().evaluate(dataset, labels)
        json.dump(
            results,
            open(evaluated_dir / "trace_rca_results.json", "w"),
        )
    elif args.evaluator == "micro_rank":
        results = MicroRankEvaluator().evaluate(dataset, labels)
        json.dump(
            results,
            open(evaluated_dir / "micro_rank_results.json", "w"),
        )
    elif args.evaluator == "span_count":
        results = SpanCountEvaluator().evaluate(dataset, labels)
        json.dump(
            results,
            open(evaluated_dir / "span_count_results.json", "w"),
        )
    elif args.evaluator == "time":
        results = TimeEvaluator().evaluate(dataset, labels)
        json.dump(
            results,
            open(evaluated_dir / "time_results.json", "w"),
        )
    elif args.evaluator == "size":
        # Size does not require a specific evaluator. It just needs reporting.
        pass
    else:
        raise ValueError(f"Unknown evaluator: {args.evaluator}")
