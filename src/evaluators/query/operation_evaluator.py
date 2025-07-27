from collections import defaultdict

from ...dataset import Dataset
from ..evaluator import Evaluator


class OperationEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        operation_set = defaultdict(set)
        operation_pair_set = defaultdict(set)
        for trace in dataset.traces.values():
            for span in trace.values():
                service = span["nodeName"].split("!@#")[0]
                operation_set["all"].add(service)
                if span["parentSpanId"] is not None:
                    parent_span = trace.get(span["parentSpanId"])
                    if parent_span:
                        parent_service = parent_span["nodeName"].split("!@#")[0]
                        operation_pair_set["all"].add(str((parent_service, service)))
        for operation in operation_set:
            operation_set[operation] = list(operation_set[operation])
        for operation_pair in operation_pair_set:
            operation_pair_set[operation_pair] = list(
                operation_pair_set[operation_pair]
            )
        return {
            "operation": operation_set,
            "operation_pair": operation_pair_set,
        }
