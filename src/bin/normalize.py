import argparse
import os
import pathlib
import pickle
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

    run_dir = pathlib.Path(args.run_dir)
    dataset_dir = pathlib.Path(args.dataset_dir)
    dataset = RCAEvalDataset(run_dir)
    dataset.save(dataset_dir)
    labels = {}
    with open(run_dir.joinpath("inject_time.txt")) as f:
        labels["inject_time"] = int(f.read().strip())
    pickle.dump(labels, open(dataset_dir.joinpath("labels.pkl"), "wb"))
