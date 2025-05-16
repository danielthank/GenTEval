import pathlib
import pickle
from typing import Optional

import pandas as pd

from dataset import Dataset


class RCAEvalDataset(Dataset):
    def __init__(self, run_dir: Optional[pathlib.Path] = None):
        super().__init__()
        self.run_dir = run_dir

    def load_spans(self):
        if self.spans is None and self.run_dir is not None:
            self.spans = pd.read_csv(self.run_dir.joinpath("traces.csv"))
        return self

    def load_labels(self):
        if self.labels is None and self.run_dir is not None:
            self.labels = {}
            with open(self.run_dir.joinpath("inject_time.txt")) as f:
                self.labels["inject_time"] = int(f.read().strip())
        return self

    def spans_to_traces(self):
        if self.spans is None:
            self.load_spans()

        all_cnt = len(self.spans)
        valid_cnt = 0

        filtered_df = self.spans.dropna(subset=["startTime", "duration"])
        self.traces = {}
        for trace_id, trace_group in filtered_df.groupby("traceID"):
            all_spans_set = set(trace_group["spanID"])
            valid_parent_mask = trace_group["parentSpanID"].isna() | trace_group[
                "parentSpanID"
            ].isin(all_spans_set)

            valid_trace_group = trace_group[valid_parent_mask]
            if valid_trace_group.empty:
                continue

            valid_cnt += len(valid_trace_group)
            spans_dict = (
                valid_trace_group.set_index("spanID")
                .apply(
                    lambda row: {
                        "nodeName": f"{row['serviceName']}@{row['methodName']}@{row['operationName']}",
                        "startTime": row["startTime"],
                        "duration": row["duration"],
                        "parentSpanId": None
                        if pd.isna(row["parentSpanID"])
                        else row["parentSpanID"],
                        "statusCode": None
                        if pd.isna(row["statusCode"])
                        else row["statusCode"],
                    },
                    axis=1,
                )
                .to_dict()
            )

            self.traces[trace_id] = spans_dict

        print(f"{self.run_dir}: {valid_cnt}/{all_cnt}")
        return self

    def save(self, dir: pathlib.Path):
        dir.mkdir(parents=True, exist_ok=True)
        if self.spans is None:
            self.load_spans()
            self.spans.to_pickle(dir.joinpath("spans.pkl"))
        if self.labels is None:
            self.load_labels()
            pickle.dump(self.labels, open(dir.joinpath("labels.pkl"), "wb"))
        if self.traces is None:
            self.spans_to_traces()
            pickle.dump(self.traces, open(dir.joinpath("traces.pkl"), "wb"))

    def load(self, dir: pathlib.Path):
        self.spans = pd.read_pickle(dir.joinpath("spans.pkl"))
        self.labels = pickle.load(open(dir.joinpath("labels.pkl"), "rb"))
        self.traces = pickle.load(open(dir.joinpath("traces.pkl"), "rb"))
        return self

    def traces_to_spans(self):
        """Convert traces dictionary back to spans dataframe."""
        pass
