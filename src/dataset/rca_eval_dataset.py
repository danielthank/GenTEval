import pathlib
import pickle
from typing import Optional

import pandas as pd

from .dataset import Dataset


class RCAEvalDataset(Dataset):
    def __init__(self, run_dir: Optional[pathlib.Path] = None):
        super().__init__()
        self.run_dir = run_dir
        self._spans = None

    @staticmethod
    def from_dataset(dataset: Dataset) -> "RCAEvalDataset":
        rca_eval_dataset = RCAEvalDataset()
        rca_eval_dataset._traces = dataset.traces
        return rca_eval_dataset

    @property
    def spans(self):
        if self._spans is None and self._traces is not None:
            self._traces_to_spans()
        if self._spans is None and self.run_dir is not None:
            self._load_spans_from_run_dir()
        return self._spans

    @property
    def traces(self):
        if self._traces is None and self.run_dir is not None:
            self._spans_to_traces()
        return self._traces

    @traces.setter
    def traces(self, traces):
        self._traces = traces

    def save(self, dir: pathlib.Path):
        dir.mkdir(parents=True, exist_ok=True)
        spans = self.spans
        if spans is not None:
            spans.to_pickle(dir.joinpath("spans.pkl"))
            spans.to_csv(dir.joinpath("spans.csv"), index=False)

        traces = self.traces
        if traces is not None:
            pickle.dump(traces, open(dir.joinpath("traces.pkl"), "wb"))

    def load(self, dataset_dir: pathlib.Path):
        self._spans = pd.read_pickle(dataset_dir.joinpath("spans.pkl"))
        self.traces = pickle.load(open(dataset_dir.joinpath("traces.pkl"), "rb"))
        return self

    def _load_spans_from_run_dir(self):
        self._spans = pd.read_csv(self.run_dir.joinpath("traces.csv"))
        return self

    def _spans_to_traces(self):
        all_cnt = len(self._spans)
        valid_cnt = 0

        filtered_df = self._spans.dropna(subset=["startTime", "duration"])
        self._traces = {}
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
                        "nodeName": f"{row['serviceName']}!@#{row['methodName']}!@#{row['operationName']}",
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

            self._traces[trace_id] = spans_dict

        print(f"{self.run_dir}: {valid_cnt}/{all_cnt}")
        return self

    def _traces_to_spans(self):
        def parse_node_name(node_name):
            parts = node_name.split("!@#")
            return [None if part == "nan" else part for part in parts]

        rows = []

        for trace_id, trace in self.traces.items():
            for span_id, span in trace.items():
                service_name, method_name, operation_name = parse_node_name(
                    span["nodeName"]
                )
                rows.append(
                    {
                        "traceID": trace_id,
                        "spanID": span_id,
                        "serviceName": service_name,
                        "methodName": method_name,
                        "operationName": operation_name,
                        "startTime": span["startTime"],
                        "duration": span["duration"],
                        "parentSpanID": span["parentSpanId"],
                        "statusCode": span["statusCode"],
                    }
                )
        dtypes = {
            "traceID": "string",
            "spanID": "string",
            "serviceName": "string",
            "methodName": "string",
            "operationName": "string",
            "startTime": "int64",
            "duration": "int64",
            "parentSpanID": "string",
            "statusCode": "string",
        }
        self._spans = pd.DataFrame(rows).astype(dtypes)
