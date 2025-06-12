import numpy as np
import pandas as pd

from dataset import Dataset, RCAEvalDataset
from evaluations import Evaluation

pd.options.mode.copy_on_write = True


# Adapted from https://github.com/phamquiluan/RCAEval/blob/30e40bb356a3c0a20fd4373e0d5015d005f460cc/RCAEval/e2e/tracerca.py
def get_operation_slo(span_df):
    """Calculate the mean of duration and variance of each operation
    :arg
        span_df: span data with operations and duration columns
    :return
        operation dict of the mean of and variance
        {
            # operation: {mean, variance}
            "Currencyservice_Convert": [600, 3]}
        }
    """
    operation_slo = {}
    for op in span_df["operation"].dropna().unique():
        # get mean and std of Duration column of the corresponding operation
        mean = round(span_df[span_df["operation"] == op]["duration"].mean(), 2)
        std = round(span_df[span_df["operation"] == op]["duration"].std(), 2)

        # operation_slo[op] = [mean, std]
        operation_slo[op] = {"mean": mean, "std": std}

    # print(json.dumps(operation_slo, sort_keys=True, indent=2))
    return operation_slo


# Adapted from https://github.com/phamquiluan/RCAEval/blob/30e40bb356a3c0a20fd4373e0d5015d005f460cc/RCAEval/e2e/tracerca.py
def tracerca(data, inject_time=None):
    # span_df = pd.read_csv("./data/mm-ob/checkoutservice_delay/1/traces.csv")
    span_df = data
    # span_df["methodName"] = span_df["methodName"].fillna(span_df["operationName"])
    span_df["operation"] = span_df["serviceName"]

    inject_time = int(inject_time) * 1_000_000  # convert from seconds to microseconds

    normal_df = span_df[span_df["startTime"] + span_df["duration"] < inject_time]
    anomal_df = span_df[span_df["startTime"] + span_df["duration"] >= inject_time]

    # 1. TRACE ANOMALY DETECTION
    normal_slo = get_operation_slo(normal_df)

    anomal_df["mean"] = anomal_df["operation"].apply(
        lambda op: normal_slo.get(op, {"mean": 1000000000})["mean"]
    )
    anomal_df["std"] = anomal_df["operation"].apply(
        lambda op: normal_slo.get(op, {"std": 1000000000})["std"]
    )
    anomal_df["abnormal"] = (
        anomal_df["duration"] >= anomal_df["mean"] + 3 * anomal_df["std"]
    )

    # 2. SUSPICIOUS MICROSERVICE SET MINING

    # calculate support and confidence
    operations = list(anomal_df.operation.unique())
    support_dict = {op: 0 for op in operations}
    confidence_dict = {op: 0 for op in operations}

    # support = |abnormal_traces of operation A| / |total abnormal traces|
    # abnormal traces of operation A is when "abnormal" col is True
    for op in operations:
        if anomal_df["abnormal"].sum() == 0:
            continue
        support_dict[op] = (
            anomal_df[anomal_df["operation"] == op]["abnormal"].sum()
            / anomal_df["abnormal"].sum()
        )

    # confidence = |abnormal traces of operation A| / |total traces of operation A|
    for op in operations:
        if anomal_df[anomal_df["operation"] == op]["operation"].count() == 0:
            continue
        confidence_dict[op] = (
            anomal_df[anomal_df["operation"] == op]["abnormal"].sum()
            / anomal_df[anomal_df["operation"] == op]["operation"].count()
        )
    ji_dict = {op: 0 for op in operations}
    # ji = 2 *s * c / (s + c)
    for op in operations:
        if support_dict[op] + confidence_dict[op] == 0:
            continue
        ji_dict[op] = (
            2
            * support_dict[op]
            * confidence_dict[op]
            / (support_dict[op] + confidence_dict[op])
        )

    # 3. MICROSERVICE RANKING
    # rank by ji
    sorted_ji = sorted(ji_dict.items(), key=lambda x: x[1], reverse=True)

    ranks = []
    for op, ji in sorted_ji:
        # ignore ji nan
        if np.isnan(ji):
            continue
        ranks.append(op)

    # print(top_list, score_list)
    return {
        "ranks": ranks,
    }


class TraceRCAEvaluation(Evaluation):
    def evaluate(self, dataset: Dataset, labels):
        dataset = RCAEvalDataset.from_dataset(dataset)
        return tracerca(dataset.spans, labels["inject_time"])
