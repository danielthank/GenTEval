"""MicroRank evaluator for root cause analysis."""

import math
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import Dataset, RCAEvalDataset
from evaluators import Evaluator

warnings.filterwarnings("ignore")

pd.options.mode.copy_on_write = True


# refactor from https://github.com/phamquiluan/RCAEval/blob/bc49dbd85bd14032101fb9a69a5a37e9d6d55178/RCAEval/e2e/microrank.py
def pageRank(p_ss, p_sr, p_rs, v, operation_length, trace_length, d=0.85, alpha=0.01):
    iteration = 25
    service_ranking_vector = np.ones((operation_length, 1)) / float(
        operation_length + trace_length
    )
    request_ranking_vector = np.ones((trace_length, 1)) / float(
        operation_length + trace_length
    )

    for i in range(iteration):
        updated_service_ranking_vector = d * (
            np.dot(p_sr, request_ranking_vector)
            + alpha * np.dot(p_ss, service_ranking_vector)
        )
        updated_request_ranking_vector = (
            d * np.dot(p_rs, service_ranking_vector) + (1.0 - d) * v
        )
        service_ranking_vector = updated_service_ranking_vector / np.amax(
            updated_service_ranking_vector
        )
        request_ranking_vector = updated_request_ranking_vector / np.amax(
            updated_request_ranking_vector
        )

    normalized_service_ranking_vector = service_ranking_vector / np.amax(
        service_ranking_vector
    )
    return normalized_service_ranking_vector


def trace_pagerank(
    operation_operation, operation_trace, trace_operation, pr_trace, anomaly
):
    """Calculate pagerank weight of anormaly_list or normal_list
    :arg
    :return
        operation weight:
        weight[operation][0]: operation
        weight[operation][1]: weight
    """
    operation_length = len(operation_operation)
    trace_length = len(operation_trace)

    p_ss = np.zeros((operation_length, operation_length), dtype=np.float32)
    p_sr = np.zeros((operation_length, trace_length), dtype=np.float32)
    p_rs = np.zeros((trace_length, operation_length), dtype=np.float32)

    # matrix = np.zeros((n, n), dtype=np.float32)
    pr = np.zeros((trace_length, 1), dtype=np.float32)

    node_list = []
    for key in operation_operation.keys():
        node_list.append(key)

    trace_list = []
    for key in operation_trace.keys():
        trace_list.append(key)

    # matrix node*node
    for operation in operation_operation:
        child_num = len(operation_operation[operation])

        for child in operation_operation[operation]:
            p_ss[node_list.index(child)][node_list.index(operation)] = 1.0 / child_num

    # matrix node*request
    for trace_id in operation_trace:
        child_num = len(operation_trace[trace_id])
        for child in operation_trace[trace_id]:
            p_sr[node_list.index(child)][trace_list.index(trace_id)] = 1.0 / child_num

    # matrix request*node
    for operation in trace_operation:
        child_num = len(trace_operation[operation])

        for child in trace_operation[operation]:
            p_rs[trace_list.index(child)][node_list.index(operation)] = 1.0 / child_num

    kind_list = np.zeros(len(trace_list))
    p_srt = p_sr.T
    for i in range(len(trace_list)):
        index_list = [i]
        if kind_list[i] != 0:
            continue
        n = 0
        for j in range(i, len(trace_list)):
            if (p_srt[i] == p_srt[j]).all():
                index_list.append(j)
                n += 1
        for index in index_list:
            kind_list[index] = n

    num_sum_trace = 0
    kind_sum_trace = 0
    if not anomaly:
        for trace_id in pr_trace:
            num_sum_trace += 1.0 / kind_list[trace_list.index(trace_id)]
        for trace_id in pr_trace:
            pr[trace_list.index(trace_id)] = (
                1.0 / kind_list[trace_list.index(trace_id)] / num_sum_trace
            )
    else:
        for trace_id in pr_trace:
            kind_sum_trace += 1.0 / kind_list[trace_list.index(trace_id)]
            num_sum_trace += 1.0 / len(pr_trace[trace_id])
        for trace_id in pr_trace:
            pr[trace_list.index(trace_id)] = (
                1.0
                / (
                    kind_list[trace_list.index(trace_id)] / kind_sum_trace * 0.5
                    + 1.0 / len(pr_trace[trace_id])
                )
                / num_sum_trace
                * 0.5
            )

    if anomaly:
        print("\nAnomaly_PageRank:")
    else:
        print("\nNormal_PageRank:")
    result = pageRank(p_ss, p_sr, p_rs, pr, operation_length, trace_length)

    weight = {}
    sum = 0
    for operation in operation_operation:
        sum += result[node_list.index(operation)][0]

    trace_num_list = {}
    for operation in operation_operation:
        trace_num_list[operation] = 0
        i = node_list.index(operation)
        for j in range(len(trace_list)):
            if p_sr[i][j] != 0:
                trace_num_list[operation] += 1

    for operation in operation_operation:
        weight[operation] = (
            result[node_list.index(operation)][0] * sum / len(operation_operation)
        )

    return weight, trace_num_list


def calculate_spectrum_without_delay_list(
    anomaly_result,
    normal_result,
    anomaly_list_len,
    normal_list_len,
    top_max,
    normal_num_list,
    anomaly_num_list,
    spectrum_method,
):
    spectrum = {}

    for node in anomaly_result:
        spectrum[node] = {}
        spectrum[node]["ef"] = anomaly_result[node] * anomaly_num_list[node]
        spectrum[node]["nf"] = anomaly_result[node] * (
            anomaly_list_len - anomaly_num_list[node]
        )
        if node in normal_result:
            spectrum[node]["ep"] = normal_result[node] * normal_num_list[node]
            spectrum[node]["np"] = normal_result[node] * (
                normal_list_len - normal_num_list[node]
            )
        else:
            spectrum[node]["ep"] = 0.0000001
            spectrum[node]["np"] = 0.0000001

    for node in normal_result:
        if node not in spectrum:
            spectrum[node] = {}
            spectrum[node]["ep"] = (1 + normal_result[node]) * normal_num_list[node]
            spectrum[node]["np"] = normal_list_len - normal_num_list[node]
            if node not in anomaly_result:
                spectrum[node]["ef"] = 0.0000001
                spectrum[node]["nf"] = 0.0000001

    result = {}

    for node in spectrum:
        # Dstar2
        if spectrum_method == "dstar2":
            result[node] = (
                spectrum[node]["ef"]
                * spectrum[node]["ef"]
                / (spectrum[node]["ep"] + spectrum[node]["nf"])
            )
        # Ochiai
        elif spectrum_method == "ochiai":
            result[node] = spectrum[node]["ef"] / math.sqrt(
                (spectrum[node]["ep"] + spectrum[node]["ef"])
                * (spectrum[node]["ef"] + spectrum[node]["nf"])
            )

        elif spectrum_method == "jaccard":
            result[node] = spectrum[node]["ef"] / (
                spectrum[node]["ef"] + spectrum[node]["ep"] + spectrum[node]["nf"]
            )

        elif spectrum_method == "sorensendice":
            result[node] = (
                2
                * spectrum[node]["ef"]
                / (
                    2 * spectrum[node]["ef"]
                    + spectrum[node]["ep"]
                    + spectrum[node]["nf"]
                )
            )

        elif spectrum_method == "m1":
            result[node] = (spectrum[node]["ef"] + spectrum[node]["np"]) / (
                spectrum[node]["ep"] + spectrum[node]["nf"]
            )

        elif spectrum_method == "m2":
            result[node] = spectrum[node]["ef"] / (
                2 * spectrum[node]["ep"]
                + 2 * spectrum[node]["nf"]
                + spectrum[node]["ef"]
                + spectrum[node]["np"]
            )
        elif spectrum_method == "goodman":
            result[node] = (
                2 * spectrum[node]["ef"] - spectrum[node]["nf"] - spectrum[node]["ep"]
            ) / (2 * spectrum[node]["ef"] + spectrum[node]["nf"] + spectrum[node]["ep"])
        # Tarantula
        elif spectrum_method == "tarantula":
            result[node] = (
                spectrum[node]["ef"]
                / (spectrum[node]["ef"] + spectrum[node]["nf"])
                / (
                    spectrum[node]["ef"] / (spectrum[node]["ef"] + spectrum[node]["nf"])
                    + spectrum[node]["ep"]
                    / (spectrum[node]["ep"] + spectrum[node]["np"])
                )
            )
        # RussellRao
        elif spectrum_method == "russellrao":
            result[node] = spectrum[node]["ef"] / (
                spectrum[node]["ef"]
                + spectrum[node]["nf"]
                + spectrum[node]["ep"]
                + spectrum[node]["np"]
            )

        # Hamann
        elif spectrum_method == "hamann":
            result[node] = (
                spectrum[node]["ef"]
                + spectrum[node]["np"]
                - spectrum[node]["ep"]
                - spectrum[node]["nf"]
            ) / (
                spectrum[node]["ef"]
                + spectrum[node]["nf"]
                + spectrum[node]["ep"]
                + spectrum[node]["np"]
            )

        # Dice
        elif spectrum_method == "dice":
            result[node] = (
                2
                * spectrum[node]["ef"]
                / (spectrum[node]["ef"] + spectrum[node]["nf"] + spectrum[node]["ep"])
            )

        # SimpleMatching
        elif spectrum_method == "simplematcing":
            result[node] = (spectrum[node]["ef"] + spectrum[node]["np"]) / (
                spectrum[node]["ef"]
                + spectrum[node]["np"]
                + spectrum[node]["nf"]
                + spectrum[node]["ep"]
            )

        # RogersTanimoto
        elif spectrum_method == "rogers":
            result[node] = (spectrum[node]["ef"] + spectrum[node]["np"]) / (
                spectrum[node]["ef"]
                + spectrum[node]["np"]
                + 2 * spectrum[node]["nf"]
                + 2 * spectrum[node]["ep"]
            )

    # Top-n节点列表
    top_list = []
    score_list = []
    for index, score in enumerate(
        sorted(result.items(), key=lambda x: x[1], reverse=True)
    ):
        if index < top_max + 6:
            top_list.append(score[0])
            score_list.append(score[1])
    return top_list, score_list


def get_pagerank_graph(df):
    """
    Query the pagerank graph

    :return
        operation_operation 存储子节点 Call graph
        operation_operation[operation_name] = [operation_name1 , operation_name1 ]

        operation_trace 存储trace经过了哪些operation, 右上角 coverage graph
        operation_trace[traceid] = [operation_name1 , operation_name2]

        trace_operation 存储 operation被哪些trace 访问过, 左下角 coverage graph
        trace_operation[operation_name] = [traceid1, traceid2]

        pr_trace: 存储trace id 经过了哪些operation，不去重
        pr_trace[traceid] = [operation_name1 , operation_name2]
    """
    operation_operation = {}
    operation_trace = {}
    trace_operation = {}
    pr_trace = {}

    # loop df
    op_dict = {}  # spanid -> ops
    par_dict = {}  # spanid -> parent_spanid
    child_dict = {}  # spanid -> child_spanids
    for idx, row in tqdm(df.iterrows()):
        traceid = row["traceID"]
        spanid = row["spanID"]
        op = row["operation"]
        parent_span_id = row["parentSpanID"]

        op_dict[spanid] = op
        par_dict[spanid] = parent_span_id
        if parent_span_id not in child_dict:
            child_dict[parent_span_id] = []
        child_dict[parent_span_id].append(spanid)

        # operation_trace[traceid] = []
        if traceid not in operation_trace:
            operation_trace[traceid] = []
        operation_trace[traceid].append(op)

        if op not in trace_operation:
            trace_operation[op] = []
        trace_operation[op].append(traceid)

    for idx, row in tqdm(df.iterrows()):
        spanid = row["spanID"]
        op = row["operation"]
        parent_span_id = row["parentSpanID"]

        if op not in operation_operation:
            operation_operation[op] = []

        # append child operation
        if spanid in child_dict and len(child_dict[spanid]) > 0:
            operation_operation[op].extend(
                [op_dict[child_spanid] for child_spanid in child_dict[spanid]]
            )

    pr_trace = deepcopy(operation_trace)

    for k, v in operation_operation.items():
        operation_operation[k] = list(set(v))
    for k, v in operation_trace.items():
        operation_trace[k] = list(set(v))
    for k, v in trace_operation.items():
        trace_operation[k] = list(set(v))

    return operation_operation, operation_trace, trace_operation, pr_trace


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


def microrank(data, inject_time=None, dataset=None, **kwargs):
    span_df = data
    # span_df["methodName"] = span_df["methodName"].fillna(span_df["operationName"])
    # span_df["operation"] = span_df["serviceName"] + "_" + span_df["methodName"]
    span_df["operation"] = span_df["serviceName"]

    inject_time = int(inject_time) * 1_000_000  # convert from seconds to microseconds

    normal_df = span_df[span_df["startTime"] + span_df["duration"] < inject_time]
    normal_slo = get_operation_slo(normal_df)
    normal_traceid = normal_df["traceID"].unique()

    anomal_df = span_df[span_df["startTime"] + span_df["duration"] >= inject_time]

    normal_df["mean"] = normal_df["operation"].apply(
        lambda op: normal_slo.get(op, {"mean": 1000000000})["mean"]
    )
    normal_df["std"] = normal_df["operation"].apply(
        lambda op: normal_slo.get(op, {"std": 1000000000})["std"]
    )

    anomal_df["mean"] = anomal_df["operation"].apply(
        lambda op: normal_slo.get(op, {"mean": 1000000000})["mean"]
    )
    anomal_df["std"] = anomal_df["operation"].apply(
        lambda op: normal_slo.get(op, {"std": 1000000000})["std"]
    )

    normal_traces_df = anomal_df[
        anomal_df["duration"] / 1_000 < anomal_df["mean"] + 3 * anomal_df["std"]
    ]
    anomal_traces_df = anomal_df[
        anomal_df["duration"] / 1_000 >= anomal_df["mean"] + 3 * anomal_df["std"]
    ]

    normal_traceid = normal_traces_df["traceID"].unique()
    anomal_traceid = anomal_traces_df["traceID"].unique()

    operation_operation, operation_trace, trace_operation, pr_trace = (
        get_pagerank_graph(normal_df)
    )

    normal_trace_result, normal_num_list = trace_pagerank(
        operation_operation, operation_trace, trace_operation, pr_trace, False
    )

    a_operation_operation, a_operation_trace, a_trace_operation, a_pr_trace = (
        get_pagerank_graph(anomal_df)
    )

    anomaly_trace_result, anomaly_num_list = trace_pagerank(
        a_operation_operation, a_operation_trace, a_trace_operation, a_pr_trace, True
    )

    top_list, score_list = calculate_spectrum_without_delay_list(
        anomaly_result=anomaly_trace_result,
        normal_result=normal_trace_result,
        anomaly_list_len=len(anomal_traceid),
        normal_list_len=len(normal_traceid),
        top_max=5,
        anomaly_num_list=anomaly_num_list,
        normal_num_list=normal_num_list,
        spectrum_method="dstar2",
    )

    return {
        "ranks": top_list,
    }


class MicroRankEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        dataset = RCAEvalDataset.from_dataset(dataset)
        return microrank(dataset.spans, labels["inject_time"])
