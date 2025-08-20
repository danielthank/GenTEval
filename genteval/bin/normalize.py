import argparse
import pathlib
import pickle

from genteval.dataset.rca_eval_dataset import RCAEvalDataset


def main():
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

    run_dir = pathlib.Path(args.run_dir)
    dataset_dir = pathlib.Path(args.dataset_dir)
    dataset = RCAEvalDataset(run_dir)
    dataset.save(dataset_dir)
    labels = {}
    if (run_dir / "inject_time.txt").exists():
        with (run_dir / "inject_time.txt").open() as f:
            inject_time = f.read().strip()
            labels["inject_time"] = inject_time
    with (dataset_dir / "labels.pkl").open("wb") as f:
        pickle.dump(labels, f)
